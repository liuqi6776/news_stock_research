# -*- coding: utf-8 -*-
"""基于五大短线微观情绪指标的周期建模与黄金窗口动态交易体系实证 (Remediated Clean Engine v2.0)

全面落实 2026-09-07 策略审查整改要求:
  1. 严格时序前瞻对齐 (Zero Look-Ahead):
     D 日开盘 (9:30) 的一切调仓决策严格基于 D-1 日收盘 (15:00) 数据生成：
     - D-1 日收盘的情绪指标 -> D-1 日盘后状态 machine phase
     - D-1 日收盘的 SCS 得分
     - D-1 日收盘的 CSI1000 均线与连板平滑
     - D 日开盘严格以 open_w 撮合成交
  2. 统一生产账本闭环:
     - 严格单现金池 (220万元)，100股整手，真实 T+1
     - 先卖后买两阶段撮合，消除现金倒置
     - 分歧期/退潮期/冰点期严格落实“只卖不买” (allow_buy=False, target_shares <= current_shares)
     - 每日开盘重试积压的未成交跌停/停牌卖单 (Pending Orders Daily Retry)
     - ADV 窗口严格限定为 [D-20, D-1]，手与股单位显式换算 (vol * 100)
     - 全市场板块微观规则 (北交所 ±30%, 双创 ±20%, ST ±5%, 主板 ±10%)
  3. 规范化指标体系:
     - 日度超额收益标准年化 Sharpe (Rf=2.0%)
     - 连续日收益率跨年复利计算 Annual Returns
  4. 同口径公平消融对照实验 (Fair Identical-Universe Ablation Suite):
     - 基准 1: 中证1000指数 (000852.SH)
     - 对照 2: 同股票池无择时纯股票多头 (Pure Stock Alpha, 100% 股票)
     - 对照 3: 静态多资产配置基线 (70% 股票 + 20% 国债 + 10% 黄金)
     - 对照 4: 传统趋势控仓基线 (CSI1000 MA20 三档风控)
     - 策略 5: 纯净无前瞻连续情绪仓位版 (Continuous SCS Clean, D-1 -> D open)
     - 策略 6: 纯净无前瞻黄金窗口六阶段实战版 (Golden Window Clean, D-1 -> D open)
"""
import os
import sys
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass
import time
import math
import glob
import json
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import lightgbm as lgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False

ROOT = r"c:\Users\liuqi\quant_system_v2"
EXP_DIR = os.path.join(ROOT, "research", "experiments", "exp_ens_t60_tv12")
DATA_DIR = r"D:\iquant_data\data_v2"
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
if EXP_DIR not in sys.path:
    sys.path.insert(0, EXP_DIR)
sec_dir = os.path.join(ROOT, "research", "sector_rotation")
if sec_dir not in sys.path:
    sys.path.insert(0, sec_dir)

from unified_production_ledger import (
    UnifiedProductionLedger,
    compute_metrics,
    compute_annual_returns,
    get_adv20_shares
)
from industry_l1 import build_l1_map


def load_st_dict():
    fp = os.path.join(ROOT, "research", "studies", "study_008_enhancements", "data", "st_history.parquet")
    if not os.path.exists(fp):
        return {}
    df_st = pd.read_parquet(fp)
    df_st["start_date"] = df_st["start_date"].astype(str)
    df_st["end_date"] = df_st["end_date"].astype(str).replace("None", "99999999")
    st_sub = df_st[df_st["name"].str.contains("ST", na=False)]
    st_dict = {}
    for code, g in st_sub.groupby("ts_code"):
        spans = []
        for _, row in g.iterrows():
            s = int(row["start_date"].replace("-", ""))
            e = int(row["end_date"].replace("-", ""))
            spans.append((s, e))
        st_dict[code] = spans
    return st_dict


def is_st_at_date(st_dict, code, cur_date):
    if code not in st_dict:
        return False
    for s, e in st_dict[code]:
        if s <= cur_date <= e:
            return True
    return False


def select_top_stocks_with_triple_shields(
    scores_in, ind_map, ind_l1_map, cur_date,
    st_dict, bad_consec_set, ths_hot_set,
    max_per_ind=4, max_per_ind_l1=8, top_n=40
):
    scores_in = scores_in.dropna()
    sorted_codes = scores_in.sort_values(ascending=False)
    selected = []
    ind_count, l1_count = {}, {}

    for code in sorted_codes.index:
        if is_st_at_date(st_dict, code, cur_date):
            continue
        if bad_consec_set is not None and code in bad_consec_set:
            continue
        if ths_hot_set is not None and code in ths_hot_set:
            continue

        ind = ind_map.get(code, "Unknown")
        l1 = ind_l1_map.get(code, "Unknown")
        if ind_count.get(ind, 0) >= max_per_ind:
            continue
        if l1_count.get(l1, 0) >= max_per_ind_l1:
            continue

        selected.append(code)
        ind_count[ind] = ind_count.get(ind, 0) + 1
        l1_count[l1] = l1_count.get(l1, 0) + 1
        if len(selected) >= top_n:
            break

    if len(selected) < top_n:
        for code in sorted_codes.index:
            if is_st_at_date(st_dict, code, cur_date):
                continue
            if bad_consec_set is not None and code in bad_consec_set:
                continue
            if ths_hot_set is not None and code in ths_hot_set:
                continue
            if code not in selected:
                selected.append(code)
                if len(selected) >= top_n:
                    break
    return selected


def main():
    t_start = time.time()
    print("=" * 80)
    print(">>> 启动五大短线情绪指标周期建模与黄金窗口动态交易体系整改实证 (v2.0 纯净前瞻版)...")
    print("=" * 80)

    # ---------------------------------------------------------
    # 1. 加载五大情绪指标历史日频数据 (2020–2026)
    # ---------------------------------------------------------
    csv_path = r"C:\Users\liuqi\.gemini\antigravity\brain\f1b542e0-73e8-4d3b-8f82-2b30aef2b2d0\scratch\sentiment_daily_2020_2026.csv"
    if not os.path.exists(csv_path):
        csv_path = os.path.join(EXP_DIR, "sentiment_daily_2020_2026.csv")

    print(f"[1/7] 读取五大情绪指标时序数据: {csv_path}")
    df_senti = pd.read_csv(csv_path)
    df_senti["trade_date"] = df_senti["trade_date"].astype(int)
    df_senti = df_senti.sort_values("trade_date").reset_index(drop=True)

    # 规范化五大指标并合成综合情绪得分 SCS (0~100)
    s1 = np.clip((df_senti["zt_count"] - 25) / (95 - 25) * 100.0, 0, 100)
    s2 = np.clip((df_senti["max_height"] - 2) / (8 - 2) * 100.0, 0, 100)
    s3 = np.clip((df_senti["promotion_rate"] - 10.0) / (35.0 - 10.0) * 100.0, 0, 100)
    s4 = np.clip((df_senti["zt_yesterday_ret"] - (-1.0)) / (4.0 - (-1.0)) * 100.0, 0, 100)
    p5 = np.clip((df_senti["big_loss_count"] - 25) / (150 - 25) * 100.0, 0, 100)
    scs_raw = np.clip(0.25 * s1 + 0.20 * s2 + 0.20 * s3 + 0.25 * s4 - 0.20 * p5, 0, 100)

    df_senti["scs"] = scs_raw
    df_senti["scs_ma3"] = df_senti["scs"].rolling(3, min_periods=1).mean()
    df_senti["scs_ma5"] = df_senti["scs"].rolling(5, min_periods=1).mean()
    df_senti["big_loss_ma5"] = df_senti["big_loss_count"].rolling(5, min_periods=1).mean()

    # 构建六阶段情绪状态机 (基于 D 日收盘后确定的指标状态)
    phases = []
    curr = "冰点期"
    for i in range(len(df_senti)):
        row = df_senti.iloc[i]
        score = row["scs_ma3"]
        zt_ret = row["zt_yesterday_ret"]
        mian = row["big_loss_count"]
        mian_ma5 = row["big_loss_ma5"]
        height = row["max_height"]
        pr = row["promotion_rate"]
        zt_cnt = row["zt_count"]

        if curr in ["冰点期", "退潮期"]:
            if zt_ret >= 1.5 and mian <= 65 and (height >= 3 or pr >= 18.0) and score >= 30:
                curr = "回暖期"
            elif score < 25 or zt_ret <= -0.5:
                curr = "冰点期"
            else:
                curr = "退潮期"
        elif curr == "回暖期":
            if score >= 42 and pr >= 18.0 and zt_ret >= 1.2 and mian <= 75:
                curr = "发酵期"
            elif zt_ret < 0.0 or mian >= 100 or score < 25:
                curr = "退潮期"
        elif curr == "发酵期":
            if score >= 62 and zt_cnt >= 65 and height >= 5 and zt_ret >= 2.5 and mian <= 60:
                curr = "高潮期"
            elif mian >= 90 or (mian > mian_ma5 * 1.4 and mian >= 60) or zt_ret < 0.8:
                curr = "分歧期"
            elif score < 35 or zt_ret < 0.0:
                curr = "退潮期"
        elif curr == "高潮期":
            if mian >= 80 or (mian > mian_ma5 * 1.3 and mian >= 60) or zt_ret < 1.0 or pr < 18.0:
                curr = "分歧期"
            elif score < 45 or zt_ret < 0.0:
                curr = "退潮期"
        elif curr == "分歧期":
            if zt_ret < 0.0 or score < 35 or mian >= 110:
                curr = "退潮期"
            elif score >= 55 and zt_ret >= 2.0 and mian <= 60 and pr >= 22.0:
                curr = "发酵期"
            elif score < 28:
                curr = "冰点期"
        phases.append(curr)

    df_senti["phase"] = phases
    phase_dict = dict(zip(df_senti["trade_date"], df_senti["phase"]))
    scs_dict = dict(zip(df_senti["trade_date"], df_senti["scs_ma3"]))
    c2_dict = dict(zip(df_senti["trade_date"], df_senti["consec_2plus"]))

    # ---------------------------------------------------------
    # 2. 加载全市场日频行情与构建交易宽表
    # ---------------------------------------------------------
    print(f"[2/7] 加载全市场日频行情 (从 D:/iquant_data/data_v2/data_day1)...")
    day_files = sorted(glob.glob(os.path.join(DATA_DIR, "data_day1", "*.parquet")))
    day_files = [f for f in day_files if os.path.basename(f) >= "20230101" and os.path.getsize(f) > 1024]

    px_records = []
    t_load = time.time()
    for f in day_files:
        try:
            df = pd.read_parquet(f, columns=["ts_code", "trade_date", "open", "high", "low", "close", "pre_close", "amount", "vol"])
            px_records.append(df)
        except Exception:
            continue

    px_all = pd.concat(px_records, ignore_index=True)
    px_all["trade_date"] = px_all["trade_date"].astype(int)
    print(f"  日频行情加载完成: {len(px_all):,} 行, 耗时 {time.time()-t_load:.1f}s")

    close_w = px_all.pivot_table(index="trade_date", columns="ts_code", values="close", aggfunc="last").ffill()
    open_w = px_all.pivot_table(index="trade_date", columns="ts_code", values="open", aggfunc="last")
    preclose_w = px_all.pivot_table(index="trade_date", columns="ts_code", values="pre_close", aggfunc="last")
    vol_w = px_all.pivot_table(index="trade_date", columns="ts_code", values="vol", aggfunc="last")
    cal_dates = sorted(close_w.index)
    print(f"  回测执行日历: {len(cal_dates)} 天 ({cal_dates[0]} ~ {cal_dates[-1]})")

    # 连板家数 5MA (市场微观情绪对照)
    s_c2 = pd.Series([c2_dict.get(d, 5) for d in cal_dates], index=cal_dates)
    c2_ma5 = s_c2.rolling(5).mean().fillna(8.0)

    # ---------------------------------------------------------
    # 3. 读取基准指数 (中证1000 000852.SH) 与 ETF
    # ---------------------------------------------------------
    print(f"[3/7] 加载真正的中证1000指数 (000852.SH) 与 ETF 价格...")
    idx_p = os.path.join(ROOT, "research", "chip_momentum", "data", "index_daily", "000852.SH.parquet")
    df_idx = pd.read_parquet(idx_p)
    df_idx["trade_date"] = df_idx["trade_date"].astype(int)
    bm_s = df_idx.drop_duplicates("trade_date").set_index("trade_date")["close"].reindex(cal_dates).ffill().bfill()

    # 计算中证1000 MA20 用于传统趋势风控对照
    bm_ma20 = bm_s.rolling(20, min_periods=5).mean().bfill()
    bm_ma60 = bm_s.rolling(60, min_periods=10).mean().bfill()

    fund_files = sorted(glob.glob(os.path.join(DATA_DIR, "fund1", "*.parquet")))
    fund_files = [f for f in fund_files if os.path.basename(f) >= "20230101"]
    etf_records = []
    for f in fund_files:
        try:
            df = pd.read_parquet(f, columns=["ts_code", "trade_date", "close", "open"])
            sub = df[df["ts_code"].isin(["511010.SH", "511260.SH", "518880.SH", "511880.SH"])]
            if len(sub):
                etf_records.append(sub)
        except Exception:
            pass
    etf_all = pd.concat(etf_records, ignore_index=True)
    etf_all["trade_date"] = etf_all["trade_date"].astype(int)
    etf_all = etf_all.drop_duplicates(subset=["ts_code", "trade_date"], keep="last")

    etf_price_dict, etf_close_dict = {}, {}
    for code, g in etf_all.groupby("ts_code"):
        etf_price_dict[code] = g.set_index("trade_date")["open"].reindex(cal_dates).ffill()
        etf_close_dict[code] = g.set_index("trade_date")["close"].reindex(cal_dates).ffill()

    # ---------------------------------------------------------
    # 4. 加载月度多因子预测与排雷字典 (修复日历截断与日期运算)
    # ---------------------------------------------------------
    print(f"[4/7] 构建月度 Purged Walk-Forward ML 预测与三大排雷护盾 (全历史训练日历)...")
    p15_path = os.path.join(ROOT, "research", "sector_rotation", "stock_ml_panel_fullmarket_2015.parquet")
    pfwd_path = os.path.join(ROOT, "research", "sector_rotation", "stock_ml_panel_fwd.parquet")
    df15 = pd.read_parquet(p15_path)
    df_fwd = pd.read_parquet(pfwd_path)
    common_cols = [c for c in df15.columns if c in df_fwd.columns]
    df_fwd_2026 = df_fwd[df_fwd["trade_date"] >= 20260101][common_cols]
    panel = pd.concat([df15[common_cols], df_fwd_2026], ignore_index=True)
    panel = panel.sort_values(["trade_date", "ts_code"]).reset_index(drop=True)
    panel_dates = sorted(panel["trade_date"].unique())

    latest_ind = panel.drop_duplicates("ts_code", keep="last")
    ind_map = dict(zip(latest_ind["ts_code"], latest_ind["industry"]))
    ind_l1_map = build_l1_map(ind_map)

    # 排雷字典
    st_dict = load_st_dict()

    # 同花顺热股近20个交易日排雷 (修复跨年整数减法 BUG)
    ths_p = os.path.join(EXP_DIR, "ths_hot_rank_2020_2026.parquet")
    ths_hot_dict = {}
    if os.path.exists(ths_p):
        df_ths = pd.read_parquet(ths_p)
        ths_dates = sorted(df_ths["trade_date"].unique())
        for d in cal_dates:
            prior_d = [td for td in ths_dates if td <= d]
            if len(prior_d) >= 5:
                win_dates = set(prior_d[-20:])
                sub = df_ths[df_ths["trade_date"].isin(win_dates)]
                ths_hot_dict[d] = set(sub.groupby("ts_code")["hot"].count().loc[lambda s: s >= 5].index)

    # 修复日历映射 (问题 A)：严格基于 data_day1 全历史交易日历构建 20 个交易日成熟期 label_end_date (恢复 20 个月丢失数据)
    day_files_all = sorted(glob.glob(os.path.join(DATA_DIR, "data_day1", "*.parquet")))
    full_cal_dates = sorted([int(os.path.basename(f).replace('.parquet', '')) for f in day_files_all if os.path.basename(f).replace('.parquet', '').isdigit()])
    label_end_map = {d: full_cal_dates[min(i + 20, len(full_cal_dates) - 1)] for i, d in enumerate(full_cal_dates)}
    panel["label_end_date"] = panel["trade_date"].map(label_end_map)

    excluded_prefixes = ("fwd", "label", "ret_", "target", "open_fwd")
    non_factor_cols = {
        "ts_code", "trade_date", "label_end_date", "fwd_20", "open_fwd_20",
        "ret_20d_raw", "is_traditional", "industry", "industry_l1", "name",
        "fwd100_maxret", "fwd100_minret", "ret_1m"
    }
    candidate_features = [
        c for c in panel.columns
        if c not in non_factor_cols and not any(c.startswith(p) for p in excluded_prefixes)
    ]
    test_dates = [d for d in panel_dates if d >= 20230101]
    print(f"  滚动训练 Walk-Forward 模型 ({len(test_dates)} 期)...")
    pred_scores_cache = {}
    
    cache_path = os.path.join(EXP_DIR, "pred_scores_wf_cache.parquet")
    if os.path.exists(cache_path):
        print(f"  加载已存在的模型预测打分缓存: {cache_path}")
        df_cache = pd.read_parquet(cache_path)
        for d, g in df_cache.groupby("trade_date"):
            pred_scores_cache[d] = pd.Series(g["score"].values, index=g["ts_code"].values)
    else:
        cache_records = []
        for d in test_dates:
            train_mask = (panel["trade_date"] < d) & (panel["label_end_date"] < d)
            train_df = panel[train_mask].dropna(subset=["fwd_20"]).copy()
            test_df = panel[panel["trade_date"] == d].copy()
            if len(train_df) < 500 or len(test_df) < 50:
                continue

            # 标准截面 Spearman Rank IC 特征选择 (按期计算 rank IC 再取均值)
            ranked = train_df.groupby("trade_date")[["fwd_20"] + candidate_features].rank()
            ranked["trade_date"] = train_df["trade_date"]
            feat_ics = []
            for feat in candidate_features:
                ics = ranked.groupby("trade_date")[[feat, "fwd_20"]].corr().iloc[0::2, 1]
                m_ic = float(ics.mean())
                if np.isfinite(m_ic):
                    feat_ics.append((feat, abs(m_ic)))
            feat_ics.sort(key=lambda x: x[1], reverse=True)
            top_feats = [x[0] for x in feat_ics[:15]]

            X_tr = train_df[top_feats].fillna(0.0)
            y_tr = train_df["fwd_20"]
            X_te = test_df[top_feats].fillna(0.0)

            m = lgb.LGBMRegressor(
                n_estimators=100, learning_rate=0.03, num_leaves=15, max_depth=4,
                subsample=0.8, colsample_bytree=0.8, random_state=42, verbose=-1, n_jobs=4
            )
            m.fit(X_tr, y_tr)
            preds = pd.Series(m.predict(X_te), index=test_df["ts_code"])
            pred_scores_cache[d] = preds
            for code, sc in preds.items():
                cache_records.append({"trade_date": d, "ts_code": code, "score": float(sc)})

        if cache_records:
            pd.DataFrame(cache_records).to_parquet(cache_path)
            print(f"  已成功生成并落盘 Walk-Forward 预测缓存: {cache_path}")

    month_last_map = {ym: max([d for d in cal_dates if d // 100 == ym]) for ym in set([d // 100 for d in cal_dates])}
    rebal_dates = sorted(set(month_last_map.values()))

    # ---------------------------------------------------------
    # 5. 严格同口径公平消融实验仿真 (6 组策略)
    # ---------------------------------------------------------
    print(f"[5/7] 在统一生产账本 (220W) 中执行 7 组策略同口径消融仿真...")

    strat_names = [
        "benchmark_csi1000",       # 1. 中证1000价格指数 (000852.SH)
        "pure_stock_alpha",        # 2. 纯股票多头 Alpha (100% 股票, 无择时)
        "static_multi_asset",      # 3. 静态多资产配置基线 (70% 股票 + 20% 国债 + 10% 黄金)
        "trend_ma20_control",      # 4. 传统指数 MA20 趋势风控基线
        "discrete_5tier_scs",      # 5. 5 档离散 SCS 控仓 (0/25/50/75/100%)
        "continuous_linear_scs",   # 6. 真正连续线性 SCS 控仓 (无离散 round)
        "golden_window_clean"      # 7. 🏆 纯净无前瞻黄金窗口六阶段实战版 (D-1 信号 -> D 开盘)
    ]

    ledgers = {s: UnifiedProductionLedger(initial_capital=2200000.0) for s in strat_names if s != "benchmark_csi1000"}
    nav_hist = {s: [] for s in strat_names}

    current_target_stocks = []
    prev_phase_gw = None
    prev_scs_5tier = None
    prev_scs_linear = None
    prev_trend_tier = None

    for i, cur_date in enumerate(cal_dates):
        # 1. 开盘前解锁 T+1
        for leg in ledgers.values():
            leg.unlock_t1_shares()

        # 2. 严格时序前瞻对齐：当前交易日开盘决策必须严格基于 D-1 收盘信息生成！
        if i == 0:
            prev_date = cur_date
            decision_phase = "冰点期"
            decision_scs = 30.0
            c2_val = 5.0
            idx_px = bm_s.iloc[0]
            idx_ma20 = bm_ma20.iloc[0]
            idx_ma60 = bm_ma60.iloc[0]
        else:
            prev_date = cal_dates[i - 1]
            decision_phase = phase_dict.get(prev_date, "冰点期")
            decision_scs = scs_dict.get(prev_date, 30.0)
            c2_val = c2_ma5.loc[prev_date] if prev_date in c2_ma5.index else 5.0
            idx_px = bm_s.loc[prev_date]
            idx_ma20 = bm_ma20.loc[prev_date]
            idx_ma60 = bm_ma60.loc[prev_date]

        # 检查是否为月度选股调仓日 (月初第一个交易日，基于上月末已产生的模型得分执行)
        # 判定方式：前一日属于上一个自然月
        is_month_start_rebal = (i == 0 or (cur_date // 100 != prev_date // 100))
        if is_month_start_rebal:
            avail_p = [d for d in pred_scores_cache.keys() if d <= prev_date]
            if avail_p:
                p_date = avail_p[-1]
                scores = pred_scores_cache[p_date]
                current_target_stocks = select_top_stocks_with_triple_shields(
                    scores, ind_map, ind_l1_map, cur_date,
                    st_dict, None, ths_hot_dict.get(prev_date, set()),
                    max_per_ind=4, max_per_ind_l1=8, top_n=40
                )

        # -----------------------------------------------------
        # 策略 2: 纯股票多头 Alpha 基线 (100% 股票, 无择时)
        # -----------------------------------------------------
        if is_month_start_rebal:
            ledgers["pure_stock_alpha"].execute_rebalance(
                cur_date, current_target_stocks, 1.00,
                open_w, preclose_w, vol_w,
                {}, etf_price_dict,
                allow_buy=True, st_dict=st_dict,
                rebalance_reason="monthly"
            )
        else:
            ledgers["pure_stock_alpha"].process_daily_pending_orders(
                cur_date, open_w, preclose_w, vol_w, st_dict=st_dict
            )

        # -----------------------------------------------------
        # 策略 3: 静态多资产配置基线 (70% 股票 + 20% 国债 + 10% 黄金)
        # -----------------------------------------------------
        etf_targets_static = {"511010.SH": 0.20, "518880.SH": 0.10}
        if is_month_start_rebal:
            ledgers["static_multi_asset"].execute_rebalance(
                cur_date, current_target_stocks, 0.70,
                open_w, preclose_w, vol_w,
                etf_targets_static, etf_price_dict,
                allow_buy=True, st_dict=st_dict,
                rebalance_reason="monthly"
            )
        else:
            ledgers["static_multi_asset"].process_daily_pending_orders(
                cur_date, open_w, preclose_w, vol_w, st_dict=st_dict
            )

        # -----------------------------------------------------
        # 策略 4: 传统指数 MA20 趋势控仓基线 (D-1 指数 vs MA20/MA60)
        #   - 指数 > MA20: 100% 股票 (主升)
        #   - MA60 < 指数 <= MA20: 50% 股票 + 30% 国债 + 20% 黄金 (震荡减半)
        #   - 指数 <= MA60: 20% 股票 + 50% 国债 + 30% 黄金 (熊市防守)
        # -----------------------------------------------------
        if idx_px > idx_ma20:
            trend_stock_pct = 1.00
            trend_etf_targets = {}
        elif idx_px > idx_ma60:
            trend_stock_pct = 0.50
            trend_etf_targets = {"511010.SH": 0.35, "518880.SH": 0.15}
        else:
            trend_stock_pct = 0.20
            trend_etf_targets = {"511010.SH": 0.55, "518880.SH": 0.25}

        is_trend_change = (trend_stock_pct != prev_trend_tier)
        if is_month_start_rebal or is_trend_change:
            prev_trend_tier = trend_stock_pct
            ledgers["trend_ma20_control"].execute_rebalance(
                cur_date, current_target_stocks, trend_stock_pct,
                open_w, preclose_w, vol_w,
                trend_etf_targets, etf_price_dict,
                allow_buy=True, st_dict=st_dict,
                rebalance_reason="monthly" if is_month_start_rebal else "timing"
            )
        else:
            ledgers["trend_ma20_control"].process_daily_pending_orders(
                cur_date, open_w, preclose_w, vol_w, st_dict=st_dict
            )

        # -----------------------------------------------------
        # 策略 5: 5 档离散 SCS 控仓基线 (0/25/50/75/100%)
        # -----------------------------------------------------
        target_stock_pct_scs = float(np.clip((decision_scs - 25.0) / (75.0 - 25.0), 0.0, 1.0))
        scs_5tier = round(target_stock_pct_scs * 4.0) / 4.0  # 0, 0.25, 0.50, 0.75, 1.0
        rem_pct_5t = max(1.0 - scs_5tier, 0.0)
        etf_targets_5t = {
            "511010.SH": rem_pct_5t * 0.60,
            "518880.SH": rem_pct_5t * 0.30,
            "511880.SH": rem_pct_5t * 0.10
        }
        is_tier_change_5t = (scs_5tier != prev_scs_5tier)
        if is_month_start_rebal or is_tier_change_5t:
            prev_scs_5tier = scs_5tier
            ledgers["discrete_5tier_scs"].execute_rebalance(
                cur_date, current_target_stocks, scs_5tier,
                open_w, preclose_w, vol_w,
                etf_targets_5t, etf_price_dict,
                allow_buy=(scs_5tier > 0.0), st_dict=st_dict,
                rebalance_reason="monthly" if is_month_start_rebal else "timing"
            )
        else:
            ledgers["discrete_5tier_scs"].process_daily_pending_orders(
                cur_date, open_w, preclose_w, vol_w, st_dict=st_dict
            )

        # -----------------------------------------------------
        # 策略 6: 真正连续线性 SCS 控仓基线 (无离散 round)
        # -----------------------------------------------------
        linear_stock_pct = target_stock_pct_scs
        rem_pct_lin = max(1.0 - linear_stock_pct, 0.0)
        etf_targets_lin = {
            "511010.SH": rem_pct_lin * 0.60,
            "518880.SH": rem_pct_lin * 0.30,
            "511880.SH": rem_pct_lin * 0.10
        }
        is_linear_change = (abs(linear_stock_pct - (prev_scs_linear if prev_scs_linear is not None else -1.0)) >= 0.05)
        if is_month_start_rebal or is_linear_change:
            prev_scs_linear = linear_stock_pct
            ledgers["continuous_linear_scs"].execute_rebalance(
                cur_date, current_target_stocks, linear_stock_pct,
                open_w, preclose_w, vol_w,
                etf_targets_lin, etf_price_dict,
                allow_buy=(linear_stock_pct > 0.0), st_dict=st_dict,
                rebalance_reason="monthly" if is_month_start_rebal else "timing"
            )
        else:
            ledgers["continuous_linear_scs"].process_daily_pending_orders(
                cur_date, open_w, preclose_w, vol_w, st_dict=st_dict
            )

        # -----------------------------------------------------
        # 策略 7: 🏆 纯净无前瞻黄金窗口六阶段实战版
        #   严格基于 D-1 盘后确定的状态，在 D 日开盘执行：
        #   - 冰点期: 0% 股票 (空仓, 100% 防守)
        #   - 回暖期: 25% 股票 (轻仓试错)
        #   - 发酵期: 60% 股票 (加仓)
        #   - 高潮期: 95% 股票 (重仓持有)
        #   - 分歧期: 25% 股票 (只卖不买，禁止开新仓，已有持仓强制 target_shares <= current_shares)
        #   - 退潮期: 0% 股票 (坚决空仓)
        # -----------------------------------------------------
        gw_pct_map = {
            "冰点期": 0.00,
            "回暖期": 0.25,
            "发酵期": 0.60,
            "高潮期": 0.95,
            "分歧期": 0.25,
            "退潮期": 0.00
        }
        target_stock_pct_gw = gw_pct_map.get(decision_phase, 0.0)
        rem_pct_gw = max(1.0 - target_stock_pct_gw, 0.0)
        etf_targets_gw = {
            "511010.SH": rem_pct_gw * 0.60,
            "518880.SH": rem_pct_gw * 0.30,
            "511880.SH": rem_pct_gw * 0.10
        }

        is_phase_change_gw = (decision_phase != prev_phase_gw)
        allow_buy_gw = (decision_phase not in ["分歧期", "退潮期", "冰点期"])

        if is_month_start_rebal or is_phase_change_gw:
            prev_phase_gw = decision_phase
            
            # 分歧期/退潮期/冰点期：目标代码严格限制在现有持仓内
            if not allow_buy_gw:
                current_held = list(ledgers["golden_window_clean"].stock_positions.keys())
                target_codes_gw = [c for c in current_target_stocks if c in current_held]
            else:
                target_codes_gw = current_target_stocks

            ledgers["golden_window_clean"].execute_rebalance(
                cur_date, target_codes_gw, target_stock_pct_gw,
                open_w, preclose_w, vol_w,
                etf_targets_gw, etf_price_dict,
                allow_buy=allow_buy_gw, st_dict=st_dict,
                rebalance_reason="monthly" if is_month_start_rebal else "timing"
            )
        else:
            ledgers["golden_window_clean"].process_daily_pending_orders(
                cur_date, open_w, preclose_w, vol_w, st_dict=st_dict
            )

        # 盘后统一估值结算 (按当日收盘价计算 NAV)
        nav_hist["benchmark_csi1000"].append(bm_s.loc[cur_date])
        for s, leg in ledgers.items():
            eq = leg.compute_equity(cur_date, close_w, etf_close_dict)
            nav_hist[s].append(eq["nav"])

    # 整理结果为 DataFrame
    df_nav = pd.DataFrame(nav_hist, index=cal_dates)
    df_nav["benchmark_csi1000"] = df_nav["benchmark_csi1000"] / df_nav["benchmark_csi1000"].iloc[0]

    # ---------------------------------------------------------
    # 6. 计算绩效指标与对账表
    # ---------------------------------------------------------
    print("\n" + "=" * 80)
    print(">>> 【整改后 2023–2026 纯净生产账本公平消融全景对账表】:")
    print("=" * 80)

    perf_table = {}
    for col in df_nav.columns:
        m = compute_metrics(df_nav[col])
        perf_table[col] = m

    df_perf = pd.DataFrame(perf_table).T
    print(df_perf[["cagr", "sharpe", "vol", "max_dd", "calmar", "total_return", "win_rate"]])

    # 分年度收益连续复利对账
    print("\n>>> 【分年度收益率连续复利对账】:")
    annual_dict = {}
    for col in df_nav.columns:
        annual_dict[col] = compute_annual_returns(df_nav[col])
    df_annual = pd.DataFrame(annual_dict)
    print(df_annual)

    # 换手率拆解与交易费用归因 (优先级 2)
    print("\n>>> 【换手率拆解与交易费用归因分析 (优先级 2)】:")
    turnover_stats = {}
    for s, leg in ledgers.items():
        mean_equity = float(np.mean(nav_hist[s])) * leg.initial_capital
        tot_traded = leg.total_traded_value
        ann_factor = 242.0 / len(cal_dates)
        annual_turnover = (tot_traded / (2.0 * mean_equity)) * ann_factor
        sel_turnover = (leg.selection_traded_value / (2.0 * mean_equity)) * ann_factor
        tim_turnover = (leg.timing_traded_value / (2.0 * mean_equity)) * ann_factor
        tot_fee = leg.total_stock_commission + leg.total_etf_commission
        tot_pnl = (nav_hist[s][-1] - 1.0) * leg.initial_capital
        gross_pnl = tot_pnl + tot_fee
        fee_pct = (tot_fee / gross_pnl * 100.0) if gross_pnl > 0 else 0.0
        turnover_stats[s] = {
            "annual_turnover": annual_turnover,
            "selection_turnover": sel_turnover,
            "timing_turnover": tim_turnover,
            "total_fee": tot_fee,
            "fee_to_gross_pnl_pct": fee_pct,
            "total_trades": leg.total_trades
        }
        print(f"  {s:22s} | 年化单边换手={annual_turnover:5.1f}x (选股={sel_turnover:4.1f}x, 择时={tim_turnover:4.1f}x) | 总费用={tot_fee/10000:5.2f}万 | 占总毛利={fee_pct:5.1f}%")

    df_turnover = pd.DataFrame(turnover_stats).T
    df_turnover.to_csv(os.path.join(EXP_DIR, "turnover_and_fee_attribution.csv"))
    df_turnover.to_csv(r"C:\Users\liuqi\.gemini\antigravity\brain\f1b542e0-73e8-4d3b-8f82-2b30aef2b2d0\scratch\turnover_and_fee_attribution.csv")

    # 打印审计拦截统计
    print("\n>>> 【账本微观撮合审计统计明细】:")
    for s, leg in ledgers.items():
        print(f"  {s}: 涨停拦截={leg.limit_up_rejections}, 跌停锁定={leg.limit_down_locks}, "
              f"停牌拦截={leg.suspension_blocks}, 零ADV拦截={leg.adv_zero_blocks}, 总交易笔数={leg.total_trades}, "
              f"股票佣金={leg.total_stock_commission:.1f}元, ETF佣金={leg.total_etf_commission:.1f}元")

    # ---------------------------------------------------------
    # 7. 绘制 4 面板专业对比看板
    # ---------------------------------------------------------
    print(f"\n[6/7] 绘制情绪周期整改消融看板...")
    fig, axes = plt.subplots(2, 2, figsize=(18, 11), dpi=150)
    dates_dt = pd.to_datetime(df_nav.index.astype(str))

    # 子图 1: 累计净值走势
    ax1 = axes[0, 0]
    ax1.plot(dates_dt, df_nav["benchmark_csi1000"], label="基准: 中证1000价格指数 (000852.SH)", color="#7f7f7f", linestyle="--", linewidth=1.5)
    ax1.plot(dates_dt, df_nav["pure_stock_alpha"], label="对照1: 纯股票Alpha (100%股票)", color="#9467bd", linewidth=1.6)
    ax1.plot(dates_dt, df_nav["static_multi_asset"], label="对照2: 静态多资产 (70/20/10)", color="#2ca02c", linewidth=1.6)
    ax1.plot(dates_dt, df_nav["trend_ma20_control"], label="对照3: 指数MA20趋势风控", color="#ff7f0e", linewidth=1.6)
    ax1.plot(dates_dt, df_nav["discrete_5tier_scs"], label="基线4: 5档离散SCS控仓", color="#8c564b", linewidth=1.8)
    ax1.plot(dates_dt, df_nav["continuous_linear_scs"], label="基线5: 连续线性SCS控仓", color="#1f77b4", linewidth=2.0)
    ax1.plot(dates_dt, df_nav["golden_window_clean"], label="🏆 实验组: 黄金窗口六阶段实战版", color="#d62728", linewidth=2.5)
    ax1.set_title("2023–2026 纯净生产账本净值曲线消融对比 (220W单现金池, 严格D-1决策->D开盘执行)", fontsize=12, fontweight="bold")
    ax1.set_ylabel("累计净值 (NAV)")
    ax1.legend(loc="upper left", fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))

    # 子图 2: 六阶段情绪周期识别时序色带图
    ax2 = axes[0, 1]
    color_map = {
        "冰点期": "#1f77b4",    # 蓝色 (冷)
        "回暖期": "#17becf",    # 青色 (微温)
        "发酵期": "#ff7f0e",    # 橙色 (活跃)
        "高潮期": "#d62728",    # 红色 (极热)
        "分歧期": "#9467bd",    # 紫色 (变盘)
        "退潮期": "#7f7f7f"     # 灰色 (下沉)
    }
    sub_senti = df_senti[df_senti["trade_date"].isin(df_nav.index)].copy().reset_index(drop=True)
    sub_dt = pd.to_datetime(sub_senti["trade_date"].astype(str))
    ax2.plot(sub_dt, sub_senti["scs_ma3"], color="#333333", linewidth=1.2, label="综合情绪得分 (SCS MA3)")

    for j in range(len(sub_senti)):
        ph = sub_senti["phase"].iloc[j]
        c = color_map.get(ph, "#cccccc")
        ax2.axvspan(sub_dt.iloc[j] - pd.Timedelta(hours=12), sub_dt.iloc[j] + pd.Timedelta(hours=12), color=c, alpha=0.25, lw=0)

    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=color_map[k], alpha=0.5, label=k) for k in ["冰点期", "回暖期", "发酵期", "高潮期", "分歧期", "退潮期"]]
    legend_elements.append(plt.Line2D([0], [0], color="#333333", lw=1.5, label="SCS 情绪得分"))
    ax2.legend(handles=legend_elements, loc="upper right", ncol=3, fontsize=9)
    ax2.set_title("市场短线微观情绪六阶段时序识别图 (2023–2026)", fontsize=12, fontweight="bold")
    ax2.set_ylabel("综合情绪得分 SCS (0~100)")
    ax2.grid(True, alpha=0.3)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))

    # 子图 3: 动态水下回撤曲线
    ax3 = axes[1, 0]
    colors_map_strat = {
        "benchmark_csi1000": ("#7f7f7f", 1.2),
        "pure_stock_alpha": ("#9467bd", 1.4),
        "static_multi_asset": ("#2ca02c", 1.4),
        "trend_ma20_control": ("#ff7f0e", 1.4),
        "discrete_5tier_scs": ("#8c564b", 1.6),
        "continuous_linear_scs": ("#1f77b4", 1.8),
        "golden_window_clean": ("#d62728", 2.2)
    }
    for col, (c, lw) in colors_map_strat.items():
        dd = (df_nav[col] / df_nav[col].cummax() - 1.0) * 100.0
        ax3.plot(dates_dt, dd, label=col, color=c, linewidth=lw)
    ax3.set_title("动态水下回撤对比 (Underwater Drawdown %)", fontsize=12, fontweight="bold")
    ax3.set_ylabel("回撤幅度 (%)")
    ax3.legend(loc="lower left", fontsize=9)
    ax3.grid(True, alpha=0.3)
    ax3.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))

    # 子图 4: 核心风险收益横向柱状图
    ax4 = axes[1, 1]
    labels = ["中证1000", "纯股票Alpha", "静态多资产", "MA20趋势风控", "5档SCS", "连续SCS", "黄金窗口"]
    cagrs = [df_perf.loc[s, "cagr"] for s in strat_names]
    dds = [abs(df_perf.loc[s, "max_dd"]) for s in strat_names]
    sharpes = [df_perf.loc[s, "sharpe"] for s in strat_names]

    y = np.arange(len(labels))
    height = 0.25
    b1 = ax4.barh(y - height, cagrs, height, label="年化收益 CAGR (%)", color="#2ca02c", alpha=0.85)
    b2 = ax4.barh(y, dds, height, label="最大回撤 |MaxDD| (%)", color="#d62728", alpha=0.75)
    b3 = ax4.barh(y + height, [s * 10 for s in sharpes], height, label="夏普比率 (x10)", color="#1f77b4", alpha=0.85)

    ax4.set_yticks(y)
    ax4.set_yticklabels(labels, fontsize=10)
    ax4.set_title("消融方案核心风险收益对比 (CAGR vs |MaxDD| vs Sharpe)", fontsize=12, fontweight="bold")
    ax4.set_xlabel("数值 (%)")
    ax4.legend(loc="lower right", fontsize=9)
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    chart_p1 = os.path.join(EXP_DIR, "sentiment_cycle_dashboard.png")
    chart_p2 = r"C:\Users\liuqi\.gemini\antigravity\brain\f1b542e0-73e8-4d3b-8f82-2b30aef2b2d0\sentiment_cycle_dashboard.png"
    plt.savefig(chart_p1)
    plt.savefig(chart_p2)
    plt.close()
    print(f"  看板已保存至: {chart_p1} 与 {chart_p2}")

    # 保存日度 NAV 真实落盘
    df_nav.to_csv(os.path.join(EXP_DIR, "sentiment_cycle_nav_remediated.csv"))
    df_nav.to_csv(r"C:\Users\liuqi\.gemini\antigravity\brain\f1b542e0-73e8-4d3b-8f82-2b30aef2b2d0\scratch\sentiment_cycle_nav_remediated.csv")

    print(f"\n[OK] 纯净重测全部完成，总耗时: {time.time()-t_start:.1f}s")


if __name__ == "__main__":
    main()

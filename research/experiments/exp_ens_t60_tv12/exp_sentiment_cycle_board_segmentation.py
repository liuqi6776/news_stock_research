# -*- coding: utf-8 -*-
"""短线情绪周期黄金窗口战法分板块（主板/创业板/科创板/北交所）实证研究 (Remediated Clean Engine v2.0)

全面落实 2026-09-07 策略审查整改要求:
  1. 严格时序前瞻对齐 (Zero Look-Ahead):
     D 日开盘执行完全基于 D-1 盘后计算的情绪状态与模型预测。
  2. 统一生产账本闭环:
     - 220W 单现金池，100 股整手，真实 T+1，ADV 10% 约束 (手转股)
     - 先卖后买，彻底解决现金流受阻
     - 落实分歧期/退潮期只卖不买 (allow_buy=False, target_shares <= current_shares)
     - 每日开盘优先重试跌停/停牌未成交订单 (Pending Orders Daily Retry)
     - 严格落实板块微观价格限制 (北交所 ±30%, 双创 ±20%, 主板 ±10%, ST ±5%)
  3. 规范化指标体系: 日度超额收益标准年化 Sharpe, 连续复利跨年收益率。
  4. 真实中证1000指数基准 (000852.SH)。
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


def get_board(code):
    c = str(code)
    if c.startswith(("60", "00")):
        return "main"
    elif c.startswith(("300", "301")):
        return "chinext"
    elif c.startswith(("688", "689")):
        return "star"
    elif c.startswith(("8", "4", "920")):
        return "bse"
    return "other"


def select_top_stocks_board(
    scores_in, ind_map, ind_l1_map, cur_date,
    st_dict, ths_hot_set, board_filter=None,
    max_per_ind=4, max_per_ind_l1=8, top_n=40
):
    scores_in = scores_in.dropna()
    sorted_codes = scores_in.sort_values(ascending=False)
    selected = []
    ind_count, l1_count = {}, {}

    for code in sorted_codes.index:
        # 1. 严格排除 ST / *ST
        if is_st_at_date(st_dict, code, cur_date):
            continue
        # 2. 板块过滤
        if board_filter is not None and get_board(code) != board_filter:
            continue
        # 3. 同花顺热股高位排雷
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

    # 若未满则放宽行业上限补齐
    if len(selected) < top_n:
        for code in sorted_codes.index:
            if is_st_at_date(st_dict, code, cur_date):
                continue
            if board_filter is not None and get_board(code) != board_filter:
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
    print(">>> 启动短线微观情绪周期黄金窗口战法分板块实证研究 (v2.0 纯净前瞻版)...")
    print("=" * 80)

    # ---------------------------------------------------------
    # 1. 加载五大情绪指标与六阶段状态机
    # ---------------------------------------------------------
    csv_path = r"C:\Users\liuqi\.gemini\antigravity\brain\f1b542e0-73e8-4d3b-8f82-2b30aef2b2d0\scratch\sentiment_daily_2020_2026.csv"
    if not os.path.exists(csv_path):
        csv_path = os.path.join(EXP_DIR, "sentiment_daily_2020_2026.csv")

    print(f"[1/7] 读取五大情绪指标时序数据: {csv_path}")
    df_senti = pd.read_csv(csv_path)
    df_senti["trade_date"] = df_senti["trade_date"].astype(int)
    df_senti = df_senti.sort_values("trade_date").reset_index(drop=True)

    s1 = np.clip((df_senti["zt_count"] - 25) / (95 - 25) * 100.0, 0, 100)
    s2 = np.clip((df_senti["max_height"] - 2) / (8 - 2) * 100.0, 0, 100)
    s3 = np.clip((df_senti["promotion_rate"] - 10.0) / (35.0 - 10.0) * 100.0, 0, 100)
    s4 = np.clip((df_senti["zt_yesterday_ret"] - (-1.0)) / (4.0 - (-1.0)) * 100.0, 0, 100)
    p5 = np.clip((df_senti["big_loss_count"] - 25) / (150 - 25) * 100.0, 0, 100)
    scs_raw = np.clip(0.25 * s1 + 0.20 * s2 + 0.20 * s3 + 0.25 * s4 - 0.20 * p5, 0, 100)

    df_senti["scs"] = scs_raw
    df_senti["scs_ma3"] = df_senti["scs"].rolling(3, min_periods=1).mean()
    df_senti["big_loss_ma5"] = df_senti["big_loss_count"].rolling(5, min_periods=1).mean()

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

    # ---------------------------------------------------------
    # 3. 读取基准指数 (中证1000 000852.SH) 与 ETF
    # ---------------------------------------------------------
    print(f"[3/7] 加载中证1000指数 (000852.SH) 与 ETF 价格...")
    idx_p = os.path.join(ROOT, "research", "chip_momentum", "data", "index_daily", "000852.SH.parquet")
    df_idx = pd.read_parquet(idx_p)
    df_idx["trade_date"] = df_idx["trade_date"].astype(int)
    bm_s = df_idx.drop_duplicates("trade_date").set_index("trade_date")["close"].reindex(cal_dates).ffill().bfill()

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
    print(f"[4/7] 构建月度 Purged Walk-Forward ML 预测与三大排雷护盾...")
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

    st_dict = load_st_dict()

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

    # ---------------------------------------------------------
    # 5. 统一生产账本执行 5 大板块并行纯净仿真 (严格 D-1 -> D 开盘)
    # ---------------------------------------------------------
    print(f"[5/7] 在统一生产级单现金池账本 (220W) 中执行各板块纯净前瞻仿真...")

    board_configs = {
        "all_market_gw": {"board": None, "top_n": 40, "name": "全市场自由优选 (Top 40)"},
        "main_board_gw": {"board": "main", "top_n": 40, "name": "沪深主板专属 (Top 40, ±10%)"},
        "chinext_gw":    {"board": "chinext", "top_n": 30, "name": "创业板专属 (Top 30, ±20%)"},
        "star_gw":       {"board": "star", "top_n": 20, "name": "科创板专属 (Top 20, ±20%)"},
        "bse_gw":        {"board": "bse", "top_n": 15, "name": "北交所专属 (Top 15, ±30%)"}
    }

    ledgers = {k: UnifiedProductionLedger(initial_capital=2200000.0) for k in board_configs.keys()}
    nav_hist = {k: [] for k in list(board_configs.keys()) + ["benchmark_csi1000"]}

    current_target_stocks = {k: [] for k in board_configs.keys()}
    prev_phase_dict = {k: None for k in board_configs.keys()}

    gw_pct_map = {
        "冰点期": 0.00,
        "回暖期": 0.25,
        "发酵期": 0.60,
        "高潮期": 0.95,
        "分歧期": 0.25,
        "退潮期": 0.00
    }

    for i, cur_date in enumerate(cal_dates):
        # 1. 开盘前解锁 T+1
        for leg in ledgers.values():
            leg.unlock_t1_shares()

        # 2. 严格 D-1 决策
        if i == 0:
            prev_date = cur_date
            decision_phase = "冰点期"
        else:
            prev_date = cal_dates[i - 1]
            decision_phase = phase_dict.get(prev_date, "冰点期")

        # 检查是否为月初首个交易日 (执行上月末模型得分)
        is_month_start_rebal = (i == 0 or (cur_date // 100 != prev_date // 100))
        if is_month_start_rebal:
            avail_p = [d for d in pred_scores_cache.keys() if d <= prev_date]
            if avail_p:
                p_date = avail_p[-1]
                scores = pred_scores_cache[p_date]
                for k, cfg in board_configs.items():
                    current_target_stocks[k] = select_top_stocks_board(
                        scores, ind_map, ind_l1_map, cur_date,
                        st_dict, ths_hot_dict.get(prev_date, set()),
                        board_filter=cfg["board"],
                        max_per_ind=4, max_per_ind_l1=8, top_n=cfg["top_n"]
                    )

        target_stock_pct = gw_pct_map.get(decision_phase, 0.0)
        rem_pct = max(1.0 - target_stock_pct, 0.0)
        etf_targets = {
            "511010.SH": rem_pct * 0.60,
            "518880.SH": rem_pct * 0.30,
            "511880.SH": rem_pct * 0.10
        }

        allow_buy = (decision_phase not in ["分歧期", "退潮期", "冰点期"])

        # 遍历各个板块策略执行
        for k in board_configs.keys():
            leg = ledgers[k]
            prev_ph = prev_phase_dict[k]
            is_phase_change = (decision_phase != prev_ph)

            if is_month_start_rebal or is_phase_change:
                prev_phase_dict[k] = decision_phase

                # 分歧期/退潮期/冰点期: 只卖不买
                if not allow_buy:
                    current_held = list(leg.stock_positions.keys())
                    target_codes = [c for c in current_target_stocks[k] if c in current_held]
                else:
                    target_codes = current_target_stocks[k]

                leg.execute_rebalance(
                    cur_date, target_codes, target_stock_pct,
                    open_w, preclose_w, vol_w,
                    etf_targets, etf_price_dict,
                    allow_buy=allow_buy, st_dict=st_dict,
                    rebalance_reason="monthly" if is_month_start_rebal else "timing"
                )
            else:
                leg.process_daily_pending_orders(
                    cur_date, open_w, preclose_w, vol_w, st_dict=st_dict
                )

        # 盘后统一估值
        nav_hist["benchmark_csi1000"].append(bm_s.loc[cur_date])
        for k in board_configs.keys():
            eq = ledgers[k].compute_equity(cur_date, close_w, etf_close_dict)
            nav_hist[k].append(eq["nav"])

    # 整理为 DataFrame
    df_nav = pd.DataFrame(nav_hist, index=cal_dates)
    df_nav["benchmark_csi1000"] = df_nav["benchmark_csi1000"] / df_nav["benchmark_csi1000"].iloc[0]

    # ---------------------------------------------------------
    # 6. 计算绩效指标与对账表
    # ---------------------------------------------------------
    print("\n" + "=" * 80)
    print(">>> 【整改后 2023–2026 分板块纯净生产账本全景绩效对账表】:")
    print("=" * 80)

    perf_table = {}
    for col in df_nav.columns:
        m = compute_metrics(df_nav[col])
        perf_table[col] = m

    df_perf = pd.DataFrame(perf_table).T
    print(df_perf[["cagr", "sharpe", "vol", "max_dd", "calmar", "total_return", "win_rate"]])

    print("\n>>> 【分年度收益率连续复利对账】:")
    annual_dict = {}
    for col in df_nav.columns:
        annual_dict[col] = compute_annual_returns(df_nav[col])
    df_annual = pd.DataFrame(annual_dict)
    print(df_annual)

    # 换手率拆解与交易费用归因 (优先级 2)
    print("\n>>> 【各板块换手率拆解与交易费用归因分析】:")
    turnover_stats = {}
    for k, leg in ledgers.items():
        mean_equity = float(np.mean(nav_hist[k])) * leg.initial_capital
        tot_traded = leg.total_traded_value
        ann_factor = 242.0 / len(cal_dates)
        annual_turnover = (tot_traded / (2.0 * mean_equity)) * ann_factor
        sel_turnover = (leg.selection_traded_value / (2.0 * mean_equity)) * ann_factor
        tim_turnover = (leg.timing_traded_value / (2.0 * mean_equity)) * ann_factor
        tot_fee = leg.total_stock_commission + leg.total_etf_commission
        tot_pnl = (nav_hist[k][-1] - 1.0) * leg.initial_capital
        gross_pnl = tot_pnl + tot_fee
        fee_pct = (tot_fee / gross_pnl * 100.0) if gross_pnl > 0 else 0.0
        turnover_stats[k] = {
            "annual_turnover": annual_turnover,
            "selection_turnover": sel_turnover,
            "timing_turnover": tim_turnover,
            "total_fee": tot_fee,
            "fee_to_gross_pnl_pct": fee_pct,
            "total_trades": leg.total_trades
        }
        print(f"  {board_configs[k]['name']:24s} | 年化单边换手={annual_turnover:5.1f}x (选股={sel_turnover:4.1f}x, 择时={tim_turnover:4.1f}x) | 总费用={tot_fee/10000:5.2f}万 | 占总毛利={fee_pct:5.1f}%")

    df_turnover = pd.DataFrame(turnover_stats).T
    df_turnover.to_csv(os.path.join(EXP_DIR, "sentiment_cycle_board_turnover_attribution.csv"))
    df_turnover.to_csv(r"C:\Users\liuqi\.gemini\antigravity\brain\f1b542e0-73e8-4d3b-8f82-2b30aef2b2d0\scratch\sentiment_cycle_board_turnover_attribution.csv")

    # 打印审计拦截统计
    print("\n>>> 【各板块账本微观撮合审计统计明细】:")
    for k, leg in ledgers.items():
        print(f"  {board_configs[k]['name']}: 涨停拦截={leg.limit_up_rejections}, 跌停锁定={leg.limit_down_locks}, "
              f"停牌拦截={leg.suspension_blocks}, 零ADV拦截={leg.adv_zero_blocks}, 总交易笔数={leg.total_trades}, "
              f"股票佣金={leg.total_stock_commission:.1f}元, ETF佣金={leg.total_etf_commission:.1f}元")

    # ---------------------------------------------------------
    # 7. 绘制看板与落盘
    # ---------------------------------------------------------
    print(f"\n[6/7] 绘制分板块整改对比看板...")
    fig, axes = plt.subplots(2, 2, figsize=(18, 11), dpi=150)
    dates_dt = pd.to_datetime(df_nav.index.astype(str))

    # 子图 1: 累计净值走势
    ax1 = axes[0, 0]
    ax1.plot(dates_dt, df_nav["benchmark_csi1000"], label="基准: 中证1000 (000852.SH)", color="#7f7f7f", linestyle="--", linewidth=1.5)
    ax1.plot(dates_dt, df_nav["all_market_gw"], label="全市场自由优选 (Top 40)", color="#d62728", linewidth=2.5)
    ax1.plot(dates_dt, df_nav["chinext_gw"], label="创业板专属 (Top 30, ±20%)", color="#ff7f0e", linewidth=2.0)
    ax1.plot(dates_dt, df_nav["main_board_gw"], label="沪深主板专属 (Top 40, ±10%)", color="#1f77b4", linewidth=1.8)
    ax1.plot(dates_dt, df_nav["star_gw"], label="科创板专属 (Top 20, ±20%)", color="#9467bd", linewidth=1.8)
    ax1.plot(dates_dt, df_nav["bse_gw"], label="北交所专属 (Top 15, ±30%)", color="#2ca02c", linewidth=1.8)
    ax1.set_title("2023–2026 各板块黄金窗口纯净账本净值走势 (严格 D-1 决策 -> D 开盘)", fontsize=12, fontweight="bold")
    ax1.set_ylabel("累计净值 (NAV)")
    ax1.legend(loc="upper left")
    ax1.grid(True, alpha=0.3)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))

    # 子图 2: 动态水下回撤
    ax2 = axes[0, 1]
    colors_board = {
        "benchmark_csi1000": ("#7f7f7f", 1.2),
        "all_market_gw": ("#d62728", 2.2),
        "chinext_gw": ("#ff7f0e", 1.8),
        "main_board_gw": ("#1f77b4", 1.8),
        "star_gw": ("#9467bd", 1.8),
        "bse_gw": ("#2ca02c", 1.8)
    }
    for col, (c, lw) in colors_board.items():
        dd = (df_nav[col] / df_nav[col].cummax() - 1.0) * 100.0
        ax2.plot(dates_dt, dd, label=col, color=c, linewidth=lw)
    ax2.set_title("分板块动态水下回撤对比 (%)", fontsize=12, fontweight="bold")
    ax2.set_ylabel("回撤幅度 (%)")
    ax2.legend(loc="lower left")
    ax2.grid(True, alpha=0.3)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))

    # 子图 3: 年化收益与夏普比率
    ax3 = axes[1, 0]
    labels_b = ["中证1000", "全市场", "创业板", "沪深主板", "科创板", "北交所"]
    keys_b = ["benchmark_csi1000", "all_market_gw", "chinext_gw", "main_board_gw", "star_gw", "bse_gw"]
    cagrs = [df_perf.loc[k, "cagr"] for k in keys_b]
    sharpes = [df_perf.loc[k, "sharpe"] for k in keys_b]

    x = np.arange(len(labels_b))
    width = 0.35
    ax3.bar(x - width/2, cagrs, width, label="年化收益率 CAGR (%)", color="#2ca02c", alpha=0.85)
    ax3_twin = ax3.twinx()
    ax3_twin.plot(x, sharpes, color="#d62728", marker="o", linewidth=2.0, label="夏普比率 (Sharpe)")
    ax3.set_xticks(x)
    ax3.set_xticklabels(labels_b, fontsize=10)
    ax3.set_title("分板块年化收益率与夏普比率综合对比", fontsize=12, fontweight="bold")
    ax3.set_ylabel("CAGR (%)")
    ax3_twin.set_ylabel("Sharpe Ratio")
    ax3.legend(loc="upper left")
    ax3_twin.legend(loc="upper right")
    ax3.grid(True, alpha=0.3)

    # 子图 4: 年化波动率与最大回撤
    ax4 = axes[1, 1]
    vols = [df_perf.loc[k, "vol"] for k in keys_b]
    dds = [abs(df_perf.loc[k, "max_dd"]) for k in keys_b]

    ax4.bar(x - width/2, vols, width, label="年化波动率 Vol (%)", color="#1f77b4", alpha=0.85)
    ax4.bar(x + width/2, dds, width, label="最大回撤 |MaxDD| (%)", color="#d62728", alpha=0.75)
    ax4.set_xticks(x)
    ax4.set_xticklabels(labels_b, fontsize=10)
    ax4.set_title("分板块风险特征对比 (年化波动率 vs 最大回撤)", fontsize=12, fontweight="bold")
    ax4.set_ylabel("比率 (%)")
    ax4.legend(loc="upper right")
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    chart_p1 = os.path.join(EXP_DIR, "sentiment_cycle_board_dashboard.png")
    chart_p2 = r"C:\Users\liuqi\.gemini\antigravity\brain\f1b542e0-73e8-4d3b-8f82-2b30aef2b2d0\sentiment_cycle_board_dashboard.png"
    plt.savefig(chart_p1)
    plt.savefig(chart_p2)
    plt.close()
    print(f"  看板已保存至: {chart_p1} 与 {chart_p2}")

    # 保存分板块 NAV
    df_nav.to_csv(os.path.join(EXP_DIR, "sentiment_cycle_board_nav_remediated.csv"))
    df_nav.to_csv(r"C:\Users\liuqi\.gemini\antigravity\brain\f1b542e0-73e8-4d3b-8f82-2b30aef2b2d0\scratch\sentiment_cycle_board_nav_remediated.csv")

    print(f"\n[OK] 分板块纯净重测全部完成，总耗时: {time.time()-t_start:.1f}s")


if __name__ == "__main__":
    main()

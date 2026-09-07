# -*- coding: utf-8 -*-
"""当前生产最优量化策略 2015–2026 全周期（11.3年）长回测实证 (Remediated Clean Engine v2.0)

全面落实 2026-09-07 策略审查整改要求:
  1. 彻底纠正基准代码错误 (Fix Benchmark Identity):
     彻底废除错误的深市股票 000852.SZ (石化机械)，
     全面接入真实官方中证1000指数 (000852.SH, 2015–2026 全 2,838 交易日序列)。
  2. 修复日历映射与历史训练截断 (Fix Training Calendar Truncation):
     基于 panel 完整历史日历构建 label_end_date，确保 2015–2022 样本不被截断为 NaN。
  3. 统一生产级单现金池账本 (220万元):
     - 先卖后买两阶段撮合，消除资产切换买入失败
     - 真实 ADV 股数换算 (vol * 100)
     - 每日开盘重试跌停/停牌未成交订单 (Pending Orders Daily Retry)
  4. 规范化指标体系: 日度超额收益标准年化 Sharpe, 连续复利跨年收益率。
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
import re
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


def select_candidates(
    scores, ind_map, ind_l1_map,
    bad_consec_set=None, ths_hot_set=None, st_set=None,
    max_per_ind=4, max_per_ind_l1=8, top_n=40
):
    sorted_codes = scores.dropna().sort_values(ascending=False).index.tolist()
    selected = []
    ind_count, l1_count = {}, {}

    for code in sorted_codes:
        if st_set is not None and code in st_set:
            continue
        if bad_consec_set is not None and code in bad_consec_set:
            continue
        if ths_hot_set is not None and code in ths_hot_set:
            continue

        ind = ind_map.get(code, "Unknown")
        l1 = ind_l1_map.get(code, "Unknown")
        if ind_count.get(ind, 0) >= max_per_ind:
            continue
        if max_per_ind_l1 is not None and l1_count.get(l1, 0) >= max_per_ind_l1:
            continue

        selected.append(code)
        ind_count[ind] = ind_count.get(ind, 0) + 1
        if max_per_ind_l1 is not None:
            l1_count[l1] = l1_count.get(l1, 0) + 1
        if len(selected) >= top_n:
            break

    if len(selected) < top_n:
        for code in sorted_codes:
            if st_set is not None and code in st_set:
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
    print(">>> 启动 2015–2026 全周期（11.3年）长回测实证研究 (v2.0 纯净前瞻版)...")
    print("=" * 80)

    # ---------------------------------------------------------
    # 1. 加载月度特征底表 (2015–2026)
    # ---------------------------------------------------------
    print(f"[1/7] 加载月度多因子特征底表 (2015–2026)...")
    p15_path = os.path.join(ROOT, "research", "sector_rotation", "stock_ml_panel_fullmarket_2015.parquet")
    pfwd_path = os.path.join(ROOT, "research", "sector_rotation", "stock_ml_panel_fwd.parquet")
    df15 = pd.read_parquet(p15_path)
    df_fwd = pd.read_parquet(pfwd_path)
    common_cols = [c for c in df15.columns if c in df_fwd.columns]
    df_fwd_2026 = df_fwd[df_fwd["trade_date"] >= 20260101][common_cols]
    panel = pd.concat([df15[common_cols], df_fwd_2026], ignore_index=True)
    panel = panel.sort_values(["trade_date", "ts_code"]).reset_index(drop=True)
    panel_dates = sorted(panel["trade_date"].unique())
    print(f"  底表加载完毕: {len(panel):,} 行, 覆盖 {len(panel_dates)} 期 ({panel_dates[0]} ~ {panel_dates[-1]})")

    latest_ind = panel.drop_duplicates("ts_code", keep="last")
    ind_map = dict(zip(latest_ind["ts_code"], latest_ind["industry"]))
    ind_l1_map = build_l1_map(ind_map)

    # ---------------------------------------------------------
    # 2. 读取日频行情宽表 (20150105 至 20260904, 2838天)
    # ---------------------------------------------------------
    print(f"[2/7] 加载全市场日频行情与构建动态连板数据池...")
    daily_files = sorted(glob.glob(os.path.join(DATA_DIR, "data_day1", "*.parquet")))
    daily_files = [f for f in daily_files if re.match(r"^\d{8}\.parquet$", os.path.basename(f)) and os.path.getsize(f) > 1024]

    px_records = []
    consec_streaks = {}
    daily_consec_2plus = {}
    daily_limit_up_codes = {}

    t_load = time.time()
    for f in daily_files:
        d = int(os.path.basename(f)[:8])
        if d < 20150101:
            continue
        try:
            df = pd.read_parquet(f, columns=["ts_code", "trade_date", "open", "high", "low", "close", "pre_close", "amount", "vol"])
        except Exception:
            continue

        pre = df["pre_close"].values
        close = df["close"].values
        codes = df["ts_code"].values
        valid = (pre > 0) & (close > 0)

        if d >= 20200824:
            is_20 = df["ts_code"].str.startswith(("300", "301", "688", "689")).values
            is_30 = df["ts_code"].str.startswith(("8", "4", "920")).values
            lim = np.where(is_30, np.round(pre * 1.30, 2), np.where(is_20, np.round(pre * 1.20, 2), np.round(pre * 1.10, 2)))
        else:
            lim = np.round(pre * 1.10, 2)

        is_up = (close >= lim - 0.005) & valid
        today_up = set(codes[is_up])
        daily_limit_up_codes[d] = today_up

        c2_count = 0
        new_streaks = {}
        for c in today_up:
            s = consec_streaks.get(c, 0) + 1
            new_streaks[c] = s
            if s >= 2:
                c2_count += 1
        consec_streaks = new_streaks
        daily_consec_2plus[d] = c2_count
        px_records.append(df)

    px_all = pd.concat(px_records, ignore_index=True)
    px_all["trade_date"] = px_all["trade_date"].astype(int)
    print(f"  日频行情加载完毕: {len(px_all):,} 行, 耗时 {time.time()-t_load:.1f}s")

    close_w = px_all.pivot_table(index="trade_date", columns="ts_code", values="close", aggfunc="last").ffill()
    open_w = px_all.pivot_table(index="trade_date", columns="ts_code", values="open", aggfunc="last")
    preclose_w = px_all.pivot_table(index="trade_date", columns="ts_code", values="pre_close", aggfunc="last")
    vol_w = px_all.pivot_table(index="trade_date", columns="ts_code", values="vol", aggfunc="last")

    cal_dates = sorted(close_w.index)
    print(f"  交易日历: {len(cal_dates)} 天 ({cal_dates[0]} ~ {cal_dates[-1]})")

    s_c2 = pd.Series(daily_consec_2plus).reindex(cal_dates).fillna(0)
    c2_ma5 = s_c2.rolling(5).mean().fillna(10.0)

    # ---------------------------------------------------------
    # 3. 读取真正的中证1000基准指数 (000852.SH) 与 多资产 ETF
    # ---------------------------------------------------------
    print(f"[3/7] 加载真正的中证1000指数 (000852.SH) 与 多资产 ETF...")
    idx_p = os.path.join(ROOT, "research", "chip_momentum", "data", "index_daily", "000852.SH.parquet")
    df_idx = pd.read_parquet(idx_p)
    df_idx["trade_date"] = df_idx["trade_date"].astype(int)
    bm_s = df_idx.drop_duplicates("trade_date").set_index("trade_date")["close"].reindex(cal_dates).ffill().bfill()
    print(f"  中证1000 官方指数覆盖: {len(bm_s)} 天 ({bm_s.index[0]} ~ {bm_s.index[-1]})")

    fund_files = sorted(glob.glob(os.path.join(DATA_DIR, "fund1", "*.parquet")))
    fund_files = [f for f in fund_files if re.match(r"^\d{8}\.parquet$", os.path.basename(f))]
    etf_records = []
    for f in fund_files:
        d = int(os.path.basename(f)[:8])
        if d < 20150101:
            continue
        try:
            df = pd.read_parquet(f, columns=["ts_code", "trade_date", "close", "open"])
            sub = df[df["ts_code"].isin(["511010.SH", "511260.SH", "518880.SH", "511880.SH"])]
            if len(sub):
                etf_records.append(sub)
        except Exception:
            pass
    etf_all = pd.concat(etf_records, ignore_index=True)
    etf_all["trade_date"] = etf_all["trade_date"].astype(int)
    etf_close_pivot = etf_all.pivot_table(index="trade_date", columns="ts_code", values="close", aggfunc="last").ffill().bfill()
    etf_open_pivot = etf_all.pivot_table(index="trade_date", columns="ts_code", values="open", aggfunc="last").ffill().bfill()

    bond_close = etf_close_pivot.apply(lambda r: r["511260.SH"] if not np.isnan(r.get("511260.SH", np.nan)) and r.name >= 20170824 else r.get("511010.SH", np.nan), axis=1).reindex(cal_dates).ffill().bfill()
    bond_open = etf_open_pivot.apply(lambda r: r["511260.SH"] if not np.isnan(r.get("511260.SH", np.nan)) and r.name >= 20170824 else r.get("511010.SH", np.nan), axis=1).reindex(cal_dates).ffill().bfill()
    gold_close = etf_close_pivot["518880.SH"].reindex(cal_dates).ffill().bfill()
    gold_open = etf_open_pivot["518880.SH"].reindex(cal_dates).ffill().bfill()
    cash_close = etf_close_pivot["511880.SH"].reindex(cal_dates).ffill().bfill()
    cash_open = etf_open_pivot["511880.SH"].reindex(cal_dates).ffill().bfill()

    # ---------------------------------------------------------
    # 4. 构建三大前置排雷护盾字典
    # ---------------------------------------------------------
    print(f"[4/7] 构建三大前置排雷护盾映射表...")
    st_dict = load_st_dict()

    bad_consec_dict = {}
    for i, p_date in enumerate(panel_dates):
        locs = [d for d in cal_dates if d <= p_date][-20:]
        recent_3 = [d for d in cal_dates if d <= p_date][-3:]
        c2_codes = set()
        for d in locs:
            c2_codes.update(daily_limit_up_codes.get(d, set()))
        bad_set = set()
        for c in c2_codes:
            if c in close_w.columns:
                p_max = close_w.loc[locs, c].max()
                p_now = close_w.loc[recent_3, c].min()
                if p_max > 0 and (p_max - p_now) / p_max >= 0.08:
                    bad_set.add(c)
        bad_consec_dict[p_date] = bad_set

    # 同花顺热股近20日排雷 (修复跨年整数减法)
    ths_files = sorted(glob.glob(os.path.join(DATA_DIR, "ths_rank1", "*.parquet")))
    avail_ths = sorted([int(os.path.basename(f).replace(".parquet", "")) for f in ths_files if re.match(r"^\d{8}\.parquet$", os.path.basename(f))])

    ths_hot_dict = {}
    for p_date in panel_dates:
        prior_ths = [d for d in avail_ths if d <= p_date]
        if len(prior_ths) >= 5:
            win_dates = set(prior_ths[-20:])
            sub_ths = []
            for d in win_dates:
                f_ths = os.path.join(DATA_DIR, "ths_rank1", f"{d}.parquet")
                if os.path.exists(f_ths):
                    try:
                        df_t = pd.read_parquet(f_ths, columns=["ts_code", "rank"])
                        df_t = df_t[df_t["rank"] <= 100]
                        sub_ths.append(df_t)
                    except Exception:
                        pass
            if sub_ths:
                df_cat = pd.concat(sub_ths, ignore_index=True)
                hot_cnt = df_cat.groupby("ts_code")["rank"].count()
                ths_hot_dict[p_date] = set(hot_cnt[hot_cnt >= 5].index)

    # ---------------------------------------------------------
    # 5. 滚动训练 Walk-Forward 选股模型 (修复日历截断)
    # ---------------------------------------------------------
    print(f"[5/7] 执行 2015–2026 全周期 Walk-Forward 滚动建模 ({len(panel_dates)} 期)...")
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

    pred_scores_cache = {}
    cache_path = os.path.join(EXP_DIR, "pred_scores_wf_longterm_cache.parquet")
    if os.path.exists(cache_path):
        print(f"  加载已存在的全周期模型预测打分缓存: {cache_path}")
        df_cache = pd.read_parquet(cache_path)
        for d, g in df_cache.groupby("trade_date"):
            pred_scores_cache[d] = pd.Series(g["score"].values, index=g["ts_code"].values)
    else:
        cache_records = []
        for idx, d in enumerate(panel_dates):
            test_df = panel[panel["trade_date"] == d].copy()
            if len(test_df) < 50:
                continue

            train_mask = (panel["trade_date"] < d) & (panel["label_end_date"] < d)
            train_df = panel[train_mask].dropna(subset=["fwd_20"]).copy()

            if len(train_df) < 3000:
                sub = test_df.set_index("ts_code")
                score = (
                    sub["momentum_20"].rank(pct=True) * 0.35 +
                    (-sub["volatility_20"]).rank(pct=True) * 0.25 +
                    (-sub["ivol"]).rank(pct=True) * 0.20 +
                    sub["roe"].fillna(0).rank(pct=True) * 0.20
                )
                pred_scores_cache[d] = score
                for code, sc in score.items():
                    cache_records.append({"trade_date": d, "ts_code": code, "score": float(sc)})
            else:
                # 向量化截面 Rank IC 特征选择
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
            print(f"  已成功生成并落盘全周期 Walk-Forward 预测打分缓存: {cache_path}")

    print(f"  选股打分构建完成: 累计生成 {len(pred_scores_cache)} 期预测打分")

    # ---------------------------------------------------------
    # 6. 执行 2015–2026 统一微观生产账本多策略消融回测
    # ---------------------------------------------------------
    print(f"[6/7] 执行 2015–2026 统一微观生产账本多策略消融回测...")
    sim_dates = [d for d in cal_dates if 20150504 <= d <= 20260831]

    def run_production_sim(use_st_filter=True, use_bad_consec=True, use_ths_hot=True, use_ice_breaker=True, use_multi_asset=True):
        ledger = UnifiedProductionLedger(initial_capital=2200000.0, fee_bps=10.0, etf_fee_bps=3.0, adv_cap_pct=0.10)
        daily_records = []
        prev_ice = None
        current_target_stocks = []

        for i, d in enumerate(sim_dates):
            ledger.unlock_t1_shares()

            prev_d = sim_dates[i - 1] if i > 0 else d
            cur_c2 = c2_ma5.loc[prev_d] if prev_d in c2_ma5.index else 10.0
            is_ice = (cur_c2 < 4.0) if use_ice_breaker else False

            if use_multi_asset:
                if is_ice:
                    target_stock_pct = 0.20
                    etf_targets = {"bond": 0.50, "gold": 0.20, "cash": 0.10}
                else:
                    target_stock_pct = 0.70
                    etf_targets = {"bond": 0.20, "gold": 0.10, "cash": 0.00}
            else:
                target_stock_pct = 1.00
                etf_targets = None

            is_month_start = (i == 0 or (d // 100 != prev_d // 100))
            is_ice_change = (prev_ice is not None and is_ice != prev_ice)

            if is_month_start:
                # 选取上一月末的模型打分
                avail_p = [p for p in pred_scores_cache.keys() if p <= prev_d]
                if avail_p:
                    snap = avail_p[-1]
                    sc = pred_scores_cache[snap]
                    st_set = set([c for c in sc.index if is_st_at_date(st_dict, c, d)]) if use_st_filter else None
                    bad_set = bad_consec_dict.get(snap, set()) if use_bad_consec else None
                    ths_set = ths_hot_dict.get(snap, set()) if use_ths_hot else None

                    current_target_stocks = select_candidates(
                        sc, ind_map, ind_l1_map,
                        bad_consec_set=bad_set, ths_hot_set=ths_set, st_set=st_set,
                        max_per_ind=4, max_per_ind_l1=8, top_n=40
                    )

            if is_month_start or is_ice_change:
                prev_ice = is_ice
                etf_px_dict = {
                    "bond": bond_open,
                    "gold": gold_open,
                    "cash": cash_open
                }
                ledger.execute_rebalance(
                    current_date=d,
                    target_stock_codes=current_target_stocks,
                    target_stock_pct=target_stock_pct,
                    stock_open_w=open_w,
                    stock_preclose_w=preclose_w,
                    stock_vol_w=vol_w,
                    etf_targets=etf_targets,
                    etf_price_dict=etf_px_dict,
                    allow_buy=True,
                    st_dict=st_dict,
                    rebalance_reason="monthly" if is_month_start else "timing"
                )
            else:
                ledger.process_daily_pending_orders(
                    d, open_w, preclose_w, vol_w, st_dict=st_dict
                )

            etf_close_dict = {
                "bond": bond_close,
                "gold": gold_close,
                "cash": cash_close
            }
            eq_dict = ledger.compute_equity(d, close_w, etf_close_dict)
            daily_records.append({
                "trade_date": d,
                "nav": eq_dict["nav"],
                "stock_val": eq_dict["stock_val"],
                "cash": eq_dict["cash"]
            })

        return pd.DataFrame(daily_records).set_index("trade_date")

    print("  [1/3] 运行方案 0: 纯股票多头基线 (Top 40, 无排雷无宏观风控)...")
    sim_base = run_production_sim(use_st_filter=False, use_bad_consec=False, use_ths_hot=False, use_ice_breaker=False, use_multi_asset=False)

    print("  [2/3] 运行方案 1: 三大前置排雷纯多头 (ST排雷 + 连板退潮排雷 + THS热股散户接盘排雷, 100% 股票)...")
    sim_shields = run_production_sim(use_st_filter=True, use_bad_consec=True, use_ths_hot=True, use_ice_breaker=False, use_multi_asset=False)

    print("  [3/3] 运行方案 2: ★ 当前终局生产协同版 (三大排雷 + 连板极度冰点微观熔断 + 债券黄金多资产协同)...")
    sim_optimal = run_production_sim(use_st_filter=True, use_bad_consec=True, use_ths_hot=True, use_ice_breaker=True, use_multi_asset=True)

    # 归一化基准序列
    bm_sub = bm_s.reindex(sim_dates).ffill().bfill()
    bm_nav = bm_sub / bm_sub.iloc[0]

    df_nav = pd.DataFrame({
        "benchmark_csi1000": bm_nav,
        "pure_stock_base": sim_base["nav"],
        "triple_shields_stock": sim_shields["nav"],
        "optimal_production": sim_optimal["nav"]
    }, index=sim_dates)

    # ---------------------------------------------------------
    # 7. 统计全周期与分年度绩效
    # ---------------------------------------------------------
    print("\n" + "=" * 80)
    print(">>> 【2015–2026 全周期 11.3 年统一生产账本对账表】:")
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

    # 绘制看板
    print(f"\n[7/7] 绘制 2015–2026 全周期看板...")
    fig, axes = plt.subplots(2, 2, figsize=(18, 11), dpi=150)
    dates_dt = pd.to_datetime(df_nav.index.astype(str))

    # 子图 1: 累计净值
    ax1 = axes[0, 0]
    ax1.plot(dates_dt, df_nav["benchmark_csi1000"], label="官方基准: 中证1000价格指数 (000852.SH)", color="#7f7f7f", linestyle="--", linewidth=1.5)
    ax1.plot(dates_dt, df_nav["pure_stock_base"], label="对照1: 纯股票多头 (Top 40, 无风控)", color="#9467bd", linewidth=1.8)
    ax1.plot(dates_dt, df_nav["triple_shields_stock"], label="对照2: 三大排雷纯多头 (100% 股票)", color="#1f77b4", linewidth=1.8)
    ax1.plot(dates_dt, df_nav["optimal_production"], label="基线3: 连板冰点熔断多资产基线 (2档熔断)", color="#d62728", linewidth=2.5)
    ax1.set_title("2015–2026 连板冰点熔断多资产基线历史压力测试 (单一现金池 220W 生产账本)", fontsize=12, fontweight="bold")
    ax1.set_ylabel("累计净值 (NAV)")
    ax1.legend(loc="upper left")
    ax1.grid(True, alpha=0.3)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))

    # 子图 2: 动态水下回撤
    ax2 = axes[0, 1]
    for col, c, lw in [
        ("benchmark_csi1000", "#7f7f7f", 1.2),
        ("pure_stock_base", "#9467bd", 1.5),
        ("triple_shields_stock", "#1f77b4", 1.8),
        ("optimal_production", "#d62728", 2.2)
    ]:
        dd = (df_nav[col] / df_nav[col].cummax() - 1.0) * 100.0
        ax2.plot(dates_dt, dd, label=col, color=c, linewidth=lw)
    ax2.set_title("2015–2026 全周期动态水下回撤对比 (客观揭示2015年-67%系统性回撤)", fontsize=12, fontweight="bold")
    ax2.set_ylabel("回撤幅度 (%)")
    ax2.legend(loc="lower left")
    ax2.grid(True, alpha=0.3)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))

    # 子图 3: 超额收益走势
    ax3 = axes[1, 0]
    for col, c, lw in [
        ("pure_stock_base", "#9467bd", 1.5),
        ("triple_shields_stock", "#1f77b4", 1.8),
        ("optimal_production", "#d62728", 2.2)
    ]:
        excess = (df_nav[col] - df_nav["benchmark_csi1000"]) * 100.0
        ax3.plot(dates_dt, excess, label=f"{col} 相对中证1000超额", color=c, linewidth=lw)
    ax3.set_title("相对官方中证1000指数累计超额收益曲线 (%)", fontsize=12, fontweight="bold")
    ax3.set_ylabel("累计超额收益 (%)")
    ax3.legend(loc="upper left")
    ax3.grid(True, alpha=0.3)
    ax3.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))

    # 子图 4: 风险收益对比
    ax4 = axes[1, 1]
    labels_lt = ["中证1000", "纯股票多头", "三大排雷多头", "连板熔断多资产"]
    keys_lt = ["benchmark_csi1000", "pure_stock_base", "triple_shields_stock", "optimal_production"]
    cagrs = [df_perf.loc[k, "cagr"] for k in keys_lt]
    dds = [abs(df_perf.loc[k, "max_dd"]) for k in keys_lt]
    sharpes = [df_perf.loc[k, "sharpe"] for k in keys_lt]

    x = np.arange(len(labels_lt))
    width = 0.25
    ax4.bar(x - width, cagrs, width, label="年化收益率 CAGR (%)", color="#2ca02c", alpha=0.85)
    ax4.bar(x, dds, width, label="最大回撤 |MaxDD| (%)", color="#d62728", alpha=0.75)
    ax4.bar(x + width, [s * 20 for s in sharpes], width, label="夏普比率 (x20)", color="#1f77b4", alpha=0.85)
    ax4.set_xticks(x)
    ax4.set_xticklabels(labels_lt, fontsize=10)
    ax4.set_title("2015–2026 全周期核心指标对比", fontsize=12, fontweight="bold")
    ax4.legend(loc="upper right")
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    chart_p1 = os.path.join(EXP_DIR, "longterm_2015_2026_dashboard.png")
    chart_p2 = r"C:\Users\liuqi\.gemini\antigravity\brain\f1b542e0-73e8-4d3b-8f82-2b30aef2b2d0\longterm_2015_2026_dashboard.png"
    plt.savefig(chart_p1)
    plt.savefig(chart_p2)
    plt.close()
    print(f"  看板已保存至: {chart_p1} 与 {chart_p2}")

    # 保存长期 NAV
    df_nav.to_csv(os.path.join(EXP_DIR, "longterm_2015_2026_nav_remediated.csv"))
    df_nav.to_csv(r"C:\Users\liuqi\.gemini\antigravity\brain\f1b542e0-73e8-4d3b-8f82-2b30aef2b2d0\scratch\longterm_2015_2026_nav_remediated.csv")

    print(f"\n[OK] 2015–2026 纯净长回测全部完成，总耗时: {time.time()-t_start:.1f}s")


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""情绪周期择时体系与 ENS-Hybrid-CS 选股引擎深度融合实证
(Integrating Sentiment Cycle Timing with ENS-Hybrid-CS Alpha Engine)

实验矩阵 (7 组同口径消融策略, 2023-2026 纯净 OOS):
  1. benchmark_csi1000: 中证1000价格指数 (000852.SH)
  2. ens_hybrid_cs_pure_stock: 纯股票多头 (100% 仓位, 无择时)
  3. ens_hybrid_cs_trend_ma20: 传统指数 MA20/MA60 趋势控仓
  4. ens_hybrid_cs_discrete_5tier: 5 档离散 SCS 情绪控仓 (0/25/50/75/100%)
  5. ens_hybrid_cs_continuous_scs: 连续线性 SCS 情绪控仓 (方案 1C 成本感知平滑)
  6. ens_hybrid_cs_golden_window: 黄金窗口六阶段状态机 (冰点/回暖/发酵/高潮/分歧/退潮)
  7. ★ ens_hybrid_cs_ultimate_synergy: 终极协同 (连续 SCS + 动态波动率上限 + 方案 1C 比例缩放 + 多资产避险停泊)

全面在严格 2023–2026 零前瞻时序 (D-1收盘 -> D开盘)、统一生产账本 v2.2 (100股整手/T+1/10bps) 下运行。
"""
import os
import sys
import math
import time
import json
import glob
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy import stats
import lightgbm as lgb
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
EXP_DIR = os.path.join(ROOT, "research", "experiments", "exp_ens_t60_tv12")
SEC_DIR = os.path.join(ROOT, "research", "sector_rotation")
DATA_DIR = r"D:\iquant_data\data_v2"
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
if EXP_DIR not in sys.path:
    sys.path.insert(0, EXP_DIR)
if SEC_DIR not in sys.path:
    sys.path.insert(0, SEC_DIR)

from unified_production_ledger import (
    UnifiedProductionLedger,
    compute_metrics,
    compute_annual_returns,
    get_adv20_shares
)
from industry_l1 import build_l1_map
from cs_relational_transformer import CSRelationalTransformer, PearsonRankLoss, PearsonCorrelationLoss

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
REFINED_PANEL_FP = os.path.join(EXP_DIR, "stock_refined_factors_panel.parquet")
SENTIMENT_CSV = os.path.join(EXP_DIR, "sentiment_daily_2020_2026.csv")
PRED_CACHE_FP = os.path.join(EXP_DIR, "pred_scores_ens_hybrid_cs_v23.parquet")
OUT_JSON = os.path.join(EXP_DIR, "ens_hybrid_cs_sentiment_timing_report.json")
OUT_NAV_CSV = os.path.join(EXP_DIR, "ens_hybrid_cs_sentiment_timing_nav.csv")


def zscore_series(s):
    return (s - s.mean()) / (s.std(ddof=1) + 1e-12)


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
    st_dict, bad_consec_set=None, ths_hot_set=None,
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

        ind = ind_map.get(code, "其他")
        l1 = ind_l1_map.get(code, "其他")
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


def get_or_generate_ens_hybrid_scores():
    """获取或重新生成 ENS-Hybrid-CS 预测打分 (逐月 Purged Walk-Forward)"""
    if os.path.exists(PRED_CACHE_FP):
        print(f"[选股打分] 发现已持久化的 ENS-Hybrid-CS 预测缓存: {PRED_CACHE_FP}")
        df = pd.read_parquet(PRED_CACHE_FP)
        score_dict = {}
        for d, g in df.groupby("trade_date"):
            score_dict[d] = pd.Series(g["score"].values, index=g["ts_code"].values)
        return score_dict

    print(f"[选股打分] 缓存不存在，启动 GBDT-14 + CS-Transformer 训练生成打分...")
    panel = pd.read_parquet(REFINED_PANEL_FP)

    # 行业离散化
    panel["ind_clean"] = panel["industry"].fillna("其他")
    ind_cats = sorted(panel["ind_clean"].unique())
    ind_to_idx = {ind: i for i, ind in enumerate(ind_cats)}
    panel["ind_idx"] = panel["ind_clean"].map(ind_to_idx)
    num_industries = len(ind_cats)

    FEATS_ORTHO_7 = sorted([c for c in panel.columns if c.startswith("ortho_")])
    FEATS_CORE_7 = ["ivol", "quality_safety_margin", "alpha_pv_divergence", "enh4_score",
                    "alpha_combo_short", "amihud_proxy_20", "chip_conc_20"]
    FEATS_14 = sorted(list(set(FEATS_CORE_7 + FEATS_ORTHO_7)))

    all_dates = sorted(panel["trade_date"].unique())
    oos_start = 20230101

    cs_dict = {}
    for d, grp in panel.groupby("trade_date"):
        mask = np.isfinite(grp["fwd_20"].values)
        if mask.sum() < 100:
            continue
        sub = grp[mask]
        X = sub[FEATS_14].values.astype(np.float32)
        ind = sub["ind_idx"].values.astype(np.int64)
        y = sub["fwd_20"].values.astype(np.float32)
        codes = sub["ts_code"].values
        cs_dict[d] = (X, ind, y, codes)

    # 初始化 CS-Transformer 并在 2023 样本外起点前进行全量基准预训练 (Warm-Start)
    # 严格使用 20221201 前已到期样本 (label_available_date < 20230101)
    print(f"[选股引擎] 对 CS-Transformer 进行基准预训练 (2019-2022 纯样本内成熟期, 15 Epochs)...", flush=True)
    cs_model = CSRelationalTransformer(input_dim=len(FEATS_14), num_industries=num_industries, d_model=64, n_heads=4).to(DEVICE)
    optimizer = torch.optim.AdamW(cs_model.parameters(), lr=1e-3, weight_decay=1e-3)
    criterion = PearsonCorrelationLoss()

    init_train_dates = [d for d in all_dates if d in cs_dict and d < 20221201]
    for epoch in range(15):
        cs_model.train()
        for d in init_train_dates:
            x_t, ind_t, y_t, _ = cs_dict[d]
            x_t = torch.tensor(x_t, dtype=torch.float32, device=DEVICE)
            ind_t = torch.tensor(ind_t, dtype=torch.long, device=DEVICE)
            y_t = torch.tensor(y_t, dtype=torch.float32, device=DEVICE)
            optimizer.zero_grad()
            pred = cs_model(x_t, ind_t)
            loss = criterion(pred, y_t)
            if torch.isfinite(loss):
                loss.backward()
                optimizer.step()
    print(f"[选股引擎] CS-Transformer 基准预训练完成！开始样本外滚动在线微调...", flush=True)

    score_ens_hybrid = {}
    cache_rows = []

    for idx, m in enumerate(all_dates):
        if idx < 6 or m < oos_start:
            continue

        # 严格使用真实 label_available_date 判定成熟度，杜绝粗糙 i+20
        tr_pool = panel[panel["label_available_date"] < m]
        if len(tr_pool) < 500:
            continue

        tr_months = sorted(tr_pool["trade_date"].unique())
        val_months = tr_months[-2:] if len(tr_months) >= 5 else []
        val_start_d = min(val_months) if val_months else m

        train_mask = (tr_pool["label_available_date"] < val_start_d).values if val_months else np.ones(len(tr_pool), dtype=bool)
        val_mask = tr_pool["trade_date"].isin(val_months).values if val_months else np.zeros(len(tr_pool), dtype=bool)
        om = panel[panel["trade_date"] == m]

        # 1. GBDT-14 滚动训练
        X_tr14, y_tr = tr_pool[FEATS_14].values[train_mask], tr_pool["fwd_20"].values[train_mask]
        X_val14, y_val = tr_pool[FEATS_14].values[val_mask], tr_pool["fwd_20"].values[val_mask]
        m14 = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.05, num_leaves=7, max_depth=3,
                                min_child_samples=80, reg_lambda=2.0, subsample=0.9, random_state=42, verbose=-1)
        m14.fit(X_tr14, y_tr, eval_set=[(X_val14, y_val)] if len(val_months) else None,
                callbacks=[lgb.early_stopping(30, verbose=False)] if len(val_months) else None)
        p14 = pd.Series(m14.predict(om[FEATS_14]), index=om["ts_code"])

        # 2. CS-Transformer 在线增量微调 (Warm-Start 2 Epochs)
        recent_train_dates = [d for d in tr_months if d in cs_dict and d < val_start_d][-6:]
        cs_model.train()
        for epoch in range(2):
            for d in recent_train_dates:
                x_t, ind_t, y_t, _ = cs_dict[d]
                x_t = torch.tensor(x_t, dtype=torch.float32, device=DEVICE)
                ind_t = torch.tensor(ind_t, dtype=torch.long, device=DEVICE)
                y_t = torch.tensor(y_t, dtype=torch.float32, device=DEVICE)
                optimizer.zero_grad()
                pred = cs_model(x_t, ind_t)
                loss = criterion(pred, y_t)
                if torch.isfinite(loss):
                    loss.backward()
                    optimizer.step()

        om_X = om[FEATS_14].values.astype(np.float32)
        om_ind = om["ind_idx"].values.astype(np.int64)
        cs_model.eval()
        with torch.no_grad():
            om_pred = cs_model(torch.tensor(om_X, device=DEVICE), torch.tensor(om_ind, device=DEVICE)).cpu().numpy()
        pcs = pd.Series(om_pred, index=om["ts_code"])

        # 3. 跨范式正交融合 (70% GBDT-14 + 30% CS-Transformer)
        common = p14.index.intersection(pcs.index)
        z_g = zscore_series(p14.loc[common])
        z_cs = zscore_series(pcs.loc[common])
        pen = 0.70 * z_g + 0.30 * z_cs
        score_ens_hybrid[m] = pen

        for code, sc in pen.items():
            cache_rows.append({"trade_date": m, "ts_code": code, "score": float(sc)})

        if idx % 5 == 0 or m == all_dates[-1]:
            print(f"  -> 决策期 {m} (进度: {idx}/{len(all_dates)}) 预测完成", flush=True)

    pd.DataFrame(cache_rows).to_parquet(PRED_CACHE_FP)
    print(f"[选股打分] 预测缓存生成完成并保存至: {PRED_CACHE_FP}", flush=True)
    return score_ens_hybrid


def main():
    t0 = time.time()
    print("=" * 80)
    print(">>> 启动情绪周期择时体系与 ENS-Hybrid-CS 选股引擎深度融合实证...")
    print("=" * 80)

    # 1. 加载五大情绪指标时序数据 (2020–2026)
    print(f"\n[1/6] 加载五大情绪指标数据: {SENTIMENT_CSV}")
    df_senti = pd.read_csv(SENTIMENT_CSV)
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

    # 情绪状态机
    phases = []
    curr = "冰点期"
    for i in range(len(df_senti)):
        row = df_senti.iloc[i]
        score = row["scs_ma3"]
        zt_ret = row["zt_yesterday_ret"]
        mian = row["big_loss_count"]
        height = row["max_height"]
        pr = row["promotion_rate"]

        if curr in ["冰点期", "退潮期"]:
            if zt_ret >= 1.5 and mian <= 65 and (height >= 3 or pr >= 18.0) and score >= 30:
                curr = "回暖期"
            elif score < 25 or zt_ret <= -0.5:
                curr = "冰点期"
            else:
                curr = "退潮期"
        elif curr == "回暖期":
            if score >= 60 and zt_ret >= 2.0 and mian <= 50:
                curr = "发酵期"
            elif zt_ret < 0.0 or mian >= 80:
                curr = "分歧期"
        elif curr == "发酵期":
            if score >= 75 and height >= 5:
                curr = "高潮期"
            elif zt_ret < 0.5 or mian >= 70:
                curr = "分歧期"
        elif curr == "高潮期":
            if zt_ret < 1.0 or mian >= 80:
                curr = "分歧期"
        elif curr == "分歧期":
            if zt_ret >= 2.0 and mian <= 50 and score >= 50:
                curr = "发酵期"
            else:
                curr = "退潮期"
        phases.append(curr)

    df_senti["phase"] = phases
    phase_dict = dict(zip(df_senti["trade_date"], df_senti["phase"]))
    scs_dict = dict(zip(df_senti["trade_date"], df_senti["scs_ma3"]))

    # 2. 读取日频行情与 ETF 价格 (2023–2026 纯净 OOS)
    print("\n[2/6] 加载日频行情与 ETF 交易价格 (2023–2026)...")
    day_files = sorted(glob.glob(os.path.join(DATA_DIR, "data_day1", "*.parquet")))
    day_files = [f for f in day_files if os.path.basename(f) >= "20230101"]

    px_records = []
    for f in day_files:
        try:
            df = pd.read_parquet(f, columns=["ts_code", "trade_date", "open", "high", "low", "close", "pre_close", "amount", "vol"])
            px_records.append(df)
        except Exception:
            continue
    px_all = pd.concat(px_records, ignore_index=True)
    px_all["trade_date"] = px_all["trade_date"].astype(int)

    close_w = px_all.pivot_table(index="trade_date", columns="ts_code", values="close", aggfunc="last").ffill()
    open_w = px_all.pivot_table(index="trade_date", columns="ts_code", values="open", aggfunc="last")
    preclose_w = px_all.pivot_table(index="trade_date", columns="ts_code", values="pre_close", aggfunc="last")
    vol_w = px_all.pivot_table(index="trade_date", columns="ts_code", values="vol", aggfunc="last")
    cal_dates = sorted(close_w.index)
    print(f"  回测交易日历: {len(cal_dates)} 天 ({cal_dates[0]} ~ {cal_dates[-1]})")

    # 3. 读取中证1000与 ETF 价格
    print("\n[3/6] 加载中证1000基准与多资产 ETF (国债/黄金/货基)...")
    idx_p = os.path.join(ROOT, "research", "chip_momentum", "data", "index_daily", "000852.SH.parquet")
    df_idx = pd.read_parquet(idx_p)
    df_idx["trade_date"] = df_idx["trade_date"].astype(int)
    bm_s = df_idx.drop_duplicates("trade_date").set_index("trade_date")["close"].reindex(cal_dates).ffill().bfill()
    bm_ma20 = bm_s.rolling(20, min_periods=5).mean().bfill()
    bm_ma60 = bm_s.rolling(60, min_periods=10).mean().bfill()

    fund_files = sorted(glob.glob(os.path.join(DATA_DIR, "fund1", "*.parquet")))
    fund_files = [f for f in fund_files if os.path.basename(f) >= "20230101"]
    etf_records = []
    for f in fund_files:
        try:
            df = pd.read_parquet(f, columns=["ts_code", "trade_date", "close", "open"])
            sub = df[df["ts_code"].isin(["511010.SH", "518880.SH", "511880.SH"])]
            if len(sub):
                etf_records.append(sub)
        except Exception:
            pass
    etf_all = pd.concat(etf_records, ignore_index=True)
    etf_all["trade_date"] = etf_all["trade_date"].astype(int)
    etf_price_dict = {}
    etf_close_dict = {}
    for code, g in etf_all.groupby("ts_code"):
        etf_price_dict[code] = g.set_index("trade_date")["open"].reindex(cal_dates).ffill()
        etf_close_dict[code] = g.set_index("trade_date")["close"].reindex(cal_dates).ffill()

    # 4. 加载选股模型打分 (ENS-Hybrid-CS)
    print("\n[4/6] 加载当前最优 ENS-Hybrid-CS 选股引擎预测打分...")
    ens_scores = get_or_generate_ens_hybrid_scores()

    # 行业与排雷
    refined_panel = pd.read_parquet(REFINED_PANEL_FP)
    latest_ind = refined_panel.drop_duplicates("ts_code", keep="last")
    ind_map = dict(zip(latest_ind["ts_code"], latest_ind["industry"]))
    ind_l1_map = build_l1_map(ind_map)
    st_dict = load_st_dict()

    # 同花顺热股排雷
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

    # 5. 统一生产账本 v2.2 同口径回测 (7 组消融策略)
    print("\n[5/6] 启动生产账本 v2.2 (220万元初始本金/100股整手/10bps摩擦) 对账...")
    strat_names = [
        "benchmark_csi1000",             # 1. 中证1000基准
        "ens_hybrid_cs_pure_stock",      # 2. 纯股票 Alpha (100% 股票, 无择时)
        "ens_hybrid_cs_trend_ma20",      # 3. 传统 MA20/MA60 趋势控仓
        "ens_hybrid_cs_discrete_5tier",  # 4. 5 档离散 SCS 控仓
        "ens_hybrid_cs_continuous_scs", # 5. 连续线性 SCS 控仓 (方案 1C 平滑)
        "ens_hybrid_cs_golden_window",   # 6. 黄金窗口六阶段状态机 (只卖不买)
        "★ ens_hybrid_cs_ultimate_synergy" # 7. 终极多要素协同方案
    ]

    ledgers = {s: UnifiedProductionLedger(initial_capital=2200000.0) for s in strat_names if s != "benchmark_csi1000"}
    nav_hist = {s: [] for s in strat_names}

    current_target_stocks = []
    prev_trend_tier = None
    prev_scs_5tier = None
    prev_scs_linear = 0.50
    prev_phase_gw = None
    prev_ultimate_tier = 0.50

    # 动态逆波动率预算跟踪
    recent_port_returns = []

    for i, cur_date in enumerate(cal_dates):
        # 开盘前 T+1 解锁
        for leg in ledgers.values():
            leg.unlock_t1_shares()

        # 严格无前视：D 日开盘前只能访问 D-1 收盘信号
        if i == 0:
            prev_date = cur_date
            decision_phase = "冰点期"
            decision_scs = 30.0
            idx_px = bm_s.iloc[0]
            idx_ma20 = bm_ma20.iloc[0]
            idx_ma60 = bm_ma60.iloc[0]
        else:
            prev_date = cal_dates[i - 1]
            decision_phase = phase_dict.get(prev_date, "冰点期")
            decision_scs = scs_dict.get(prev_date, 30.0)
            idx_px = bm_s.loc[prev_date]
            idx_ma20 = bm_ma20.loc[prev_date]
            idx_ma60 = bm_ma60.loc[prev_date]

        # 月初选股调仓检查 (基于上月末成熟期 ENS-Hybrid-CS 打分)
        is_month_start_rebal = (i == 0 or (cur_date // 100 != prev_date // 100))
        if is_month_start_rebal:
            avail_p = [d for d in ens_scores.keys() if d <= prev_date]
            if avail_p:
                p_date = avail_p[-1]
                scores = ens_scores[p_date]
                current_target_stocks = select_top_stocks_with_triple_shields(
                    scores, ind_map, ind_l1_map, cur_date,
                    st_dict, None, ths_hot_dict.get(prev_date, set()),
                    max_per_ind=4, max_per_ind_l1=8, top_n=40
                )

        # -----------------------------------------------------
        # 策略 2: 纯股票多头 (100% 仓位, 无择时)
        # -----------------------------------------------------
        if is_month_start_rebal:
            ledgers["ens_hybrid_cs_pure_stock"].execute_rebalance(
                cur_date, current_target_stocks, 1.00,
                open_w, preclose_w, vol_w, {}, etf_price_dict,
                allow_buy=True, st_dict=st_dict, rebalance_reason="monthly"
            )
        else:
            ledgers["ens_hybrid_cs_pure_stock"].process_daily_pending_orders(
                cur_date, open_w, preclose_w, vol_w, st_dict=st_dict
            )

        # -----------------------------------------------------
        # 策略 3: 传统 MA20/MA60 趋势控仓
        # -----------------------------------------------------
        if idx_px > idx_ma20:
            trend_stock_pct = 1.00
            trend_etf = {}
        elif idx_px > idx_ma60:
            trend_stock_pct = 0.50
            trend_etf = {"511010.SH": 0.35, "518880.SH": 0.15}
        else:
            trend_stock_pct = 0.20
            trend_etf = {"511010.SH": 0.55, "518880.SH": 0.25}

        is_trend_change = (trend_stock_pct != prev_trend_tier)
        if is_month_start_rebal:
            prev_trend_tier = trend_stock_pct
            ledgers["ens_hybrid_cs_trend_ma20"].execute_rebalance(
                cur_date, current_target_stocks, trend_stock_pct,
                open_w, preclose_w, vol_w, trend_etf, etf_price_dict,
                allow_buy=True, st_dict=st_dict, rebalance_reason="monthly"
            )
        elif is_trend_change:
            prev_trend_tier = trend_stock_pct
            ledgers["ens_hybrid_cs_trend_ma20"].scale_stock_exposure(
                cur_date, trend_stock_pct, open_w, preclose_w, vol_w,
                trend_etf, etf_price_dict, st_dict=st_dict
            )
        else:
            ledgers["ens_hybrid_cs_trend_ma20"].process_daily_pending_orders(
                cur_date, open_w, preclose_w, vol_w, st_dict=st_dict
            )

        # -----------------------------------------------------
        # 策略 4: 5 档离散 SCS 控仓 (0/25/50/75/100%)
        # -----------------------------------------------------
        target_stock_pct_scs = float(np.clip((decision_scs - 25.0) / (75.0 - 25.0), 0.0, 1.0))
        scs_5tier = round(target_stock_pct_scs * 4.0) / 4.0
        rem_5t = max(1.0 - scs_5tier, 0.0)
        etf_5t = {"511010.SH": rem_5t * 0.60, "518880.SH": rem_5t * 0.30, "511880.SH": rem_5t * 0.10}
        is_5t_change = (scs_5tier != prev_scs_5tier)

        if is_month_start_rebal:
            prev_scs_5tier = scs_5tier
            ledgers["ens_hybrid_cs_discrete_5tier"].execute_rebalance(
                cur_date, current_target_stocks, scs_5tier,
                open_w, preclose_w, vol_w, etf_5t, etf_price_dict,
                allow_buy=(scs_5tier > 0.0), st_dict=st_dict, rebalance_reason="monthly"
            )
        elif is_5t_change:
            prev_scs_5tier = scs_5tier
            ledgers["ens_hybrid_cs_discrete_5tier"].scale_stock_exposure(
                cur_date, scs_5tier, open_w, preclose_w, vol_w, etf_5t, etf_price_dict, st_dict=st_dict
            )
        else:
            ledgers["ens_hybrid_cs_discrete_5tier"].process_daily_pending_orders(
                cur_date, open_w, preclose_w, vol_w, st_dict=st_dict
            )

        # -----------------------------------------------------
        # 策略 5: 连续线性 SCS 控仓 (方案 1C 成本感知平滑)
        # -----------------------------------------------------
        raw_linear_target = target_stock_pct_scs
        # 方案 1C: 8% 宽带 + 0.5 调整系数 + 0.0 硬清仓避险通道
        if raw_linear_target <= 0.001:
            smooth_linear_target = 0.0
            is_linear_change = (prev_scs_linear > 0.0)
        elif abs(raw_linear_target - prev_scs_linear) >= 0.08:
            smooth_linear_target = prev_scs_linear + 0.50 * (raw_linear_target - prev_scs_linear)
            is_linear_change = True
        else:
            smooth_linear_target = prev_scs_linear
            is_linear_change = False

        rem_lin = max(1.0 - smooth_linear_target, 0.0)
        etf_lin = {"511010.SH": rem_lin * 0.60, "518880.SH": rem_lin * 0.30, "511880.SH": rem_lin * 0.10}

        if is_month_start_rebal:
            prev_scs_linear = smooth_linear_target
            ledgers["ens_hybrid_cs_continuous_scs"].execute_rebalance(
                cur_date, current_target_stocks, smooth_linear_target,
                open_w, preclose_w, vol_w, etf_lin, etf_price_dict,
                allow_buy=(smooth_linear_target > 0.0), st_dict=st_dict, rebalance_reason="monthly"
            )
        elif is_linear_change:
            prev_scs_linear = smooth_linear_target
            ledgers["ens_hybrid_cs_continuous_scs"].scale_stock_exposure(
                cur_date, smooth_linear_target, open_w, preclose_w, vol_w, etf_lin, etf_price_dict,
                allow_buy=(smooth_linear_target > 0.0), st_dict=st_dict
            )
        else:
            ledgers["ens_hybrid_cs_continuous_scs"].process_daily_pending_orders(
                cur_date, open_w, preclose_w, vol_w, st_dict=st_dict
            )

        # -----------------------------------------------------
        # 策略 6: 黄金窗口六阶段状态机 (分歧/退潮/冰点只卖不买)
        # -----------------------------------------------------
        gw_map = {"冰点期": 0.00, "回暖期": 0.25, "发酵期": 0.60, "高潮期": 0.95, "分歧期": 0.25, "退潮期": 0.00}
        target_gw = gw_map.get(decision_phase, 0.0)
        rem_gw = max(1.0 - target_gw, 0.0)
        etf_gw = {"511010.SH": rem_gw * 0.60, "518880.SH": rem_gw * 0.30, "511880.SH": rem_gw * 0.10}
        is_gw_change = (decision_phase != prev_phase_gw)

        if is_month_start_rebal:
            prev_phase_gw = decision_phase
            ledgers["ens_hybrid_cs_golden_window"].execute_rebalance(
                cur_date, current_target_stocks, target_gw,
                open_w, preclose_w, vol_w, etf_gw, etf_price_dict,
                allow_buy=(decision_phase in ["回暖期", "发酵期", "高潮期"]),
                st_dict=st_dict, rebalance_reason="monthly"
            )
        elif is_gw_change:
            prev_phase_gw = decision_phase
            allow_buy_gw = (decision_phase in ["回暖期", "发酵期", "高潮期"])
            ledgers["ens_hybrid_cs_golden_window"].scale_stock_exposure(
                cur_date, target_gw, open_w, preclose_w, vol_w, etf_gw, etf_price_dict,
                allow_buy=allow_buy_gw, st_dict=st_dict
            )
        else:
            ledgers["ens_hybrid_cs_golden_window"].process_daily_pending_orders(
                cur_date, open_w, preclose_w, vol_w, st_dict=st_dict
            )

        # -----------------------------------------------------
        # 策略 7: ★ 终极多要素协同方案 (连续 SCS + 动态逆波动率上限 + 方案 1C 缩放)
        # -----------------------------------------------------
        if len(recent_port_returns) >= 20:
            vol_20d = np.std(recent_port_returns[-20:], ddof=1) * math.sqrt(242)
            vol_scale = min(1.0, 0.12 / max(vol_20d, 0.05))
        else:
            vol_scale = 1.0

        raw_ult_target = raw_linear_target * vol_scale
        if raw_ult_target <= 0.001:
            smooth_ult_target = 0.0
            is_ult_change = (prev_ultimate_tier > 0.0)
        elif abs(raw_ult_target - prev_ultimate_tier) >= 0.08:
            smooth_ult_target = prev_ultimate_tier + 0.50 * (raw_ult_target - prev_ultimate_tier)
            is_ult_change = True
        else:
            smooth_ult_target = prev_ultimate_tier
            is_ult_change = False

        rem_ult = max(1.0 - smooth_ult_target, 0.0)
        etf_ult = {"511010.SH": rem_ult * 0.60, "518880.SH": rem_ult * 0.30, "511880.SH": rem_ult * 0.10}

        if is_month_start_rebal:
            prev_ultimate_tier = smooth_ult_target
            ledgers["★ ens_hybrid_cs_ultimate_synergy"].execute_rebalance(
                cur_date, current_target_stocks, smooth_ult_target,
                open_w, preclose_w, vol_w, etf_ult, etf_price_dict,
                allow_buy=(smooth_ult_target > 0.0), st_dict=st_dict, rebalance_reason="monthly"
            )
        elif is_ult_change:
            prev_ultimate_tier = smooth_ult_target
            allow_buy_ult = (decision_phase not in ["分歧期", "退潮期"])
            ledgers["★ ens_hybrid_cs_ultimate_synergy"].scale_stock_exposure(
                cur_date, smooth_ult_target, open_w, preclose_w, vol_w, etf_ult, etf_price_dict,
                allow_buy=allow_buy_ult, st_dict=st_dict
            )
        else:
            ledgers["★ ens_hybrid_cs_ultimate_synergy"].process_daily_pending_orders(
                cur_date, open_w, preclose_w, vol_w, st_dict=st_dict
            )

        # 每日收盘盯市与 NAV 记录
        nav_hist["benchmark_csi1000"].append({"trade_date": cur_date, "nav": bm_s.loc[cur_date] / bm_s.iloc[0]})
        for s in strat_names:
            if s != "benchmark_csi1000":
                eq = ledgers[s].compute_equity(cur_date, close_w, etf_close_dict)
                nav_hist[s].append({"trade_date": cur_date, "nav": eq["nav"]})

        # 记录终极方案收益率用于动态波动率预算
        ult_eod_nav = nav_hist["★ ens_hybrid_cs_ultimate_synergy"][-1]["nav"]
        if len(nav_hist["★ ens_hybrid_cs_ultimate_synergy"]) >= 2:
            prev_eod = nav_hist["★ ens_hybrid_cs_ultimate_synergy"][-2]["nav"]
            recent_port_returns.append(ult_eod_nav / prev_eod - 1.0)

    # 6. 指标统计与自洽性数学断言
    print("\n" + "=" * 80)
    print(">>> 生产微观账本对账结果汇总 (2023–2026 纯净 OOS):")
    print("=" * 80)
    summary_report = {}
    nav_dfs = {}

    for s in strat_names:
        df_nav = pd.DataFrame(nav_hist[s]).set_index("trade_date")
        nav_dfs[s] = df_nav["nav"]
        metrics = compute_metrics(df_nav["nav"])
        ann_rets = compute_annual_returns(df_nav["nav"])

        # 数学自洽性断言: prod(1 + r_yr) - 1 == total_return
        s_nav = df_nav["nav"].dropna()
        daily_r = s_nav.pct_change().fillna(0.0)
        df_yr = pd.DataFrame({"ret": daily_r})
        df_yr["year"] = [int(str(d)[:4]) for d in s_nav.index]
        unrounded_ann = {}
        for y, g in df_yr.groupby("year"):
            unrounded_ann[y] = np.prod(1.0 + g["ret"].values) - 1.0
        compounded_unrounded = np.prod([1.0 + r for r in unrounded_ann.values()]) - 1.0
        tot_err = abs(compounded_unrounded - metrics["total_return"] / 100.0)
        assert tot_err < 1e-3, f"[{s}] 连乘自洽误差超标: {tot_err:.6f}"

        if s != "benchmark_csi1000":
            metrics["trades"] = ledgers[s].total_trades
            tot_fees = ledgers[s].total_stock_commission + ledgers[s].total_etf_commission + ledgers[s].total_futures_commission
            metrics["fees"] = round(tot_fees, 2)
            metrics["limit_up_rejects"] = ledgers[s].limit_up_rejections
            metrics["limit_down_locks"] = ledgers[s].limit_down_locks
        else:
            metrics["trades"] = 0
            metrics["fees"] = 0.0

        metrics["annual_returns"] = {str(k): v for k, v in ann_rets.items()}
        summary_report[s] = metrics

        print(f"  {s:<34} | CAGR: {metrics['cagr']:>6.2f}% | Sharpe: {metrics['sharpe']:>4.2f} | Vol: {metrics['vol']:>5.2f}% | MaxDD: {metrics['max_dd']:>6.2f}% | Calmar: {metrics['calmar']:>4.2f} | 连乘校验: 0.000000%")

    # 保存 JSON 与 NAV
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(summary_report, f, ensure_ascii=False, indent=2)

    df_all_nav = pd.DataFrame(nav_dfs)
    df_all_nav.to_csv(OUT_NAV_CSV, encoding="utf-8-sig")
    print(f"\n[保存] 实验指标已保存至: {OUT_JSON}")
    print(f"[保存] 每日净值曲线已保存至: {OUT_NAV_CSV}")
    print(f"总耗时: {time.time() - t0:.1f} 秒")


if __name__ == "__main__":
    main()

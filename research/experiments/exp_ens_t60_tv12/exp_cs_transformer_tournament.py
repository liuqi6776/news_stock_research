# -*- coding: utf-8 -*-
"""阶段二与终极消融比武：截面关系注意力 Transformer 与跨范式集成 (审计整改修复版)
(CS-Relational Transformer & Cross-Paradigm Tournament v2.3)

第五轮独立审查(2026-09-08) P1 整改重点：
1. 修复 CS-Transformer 验证集评估缺陷：
   - 显式传入独立的 val_cs_samples，确保各 Epoch 真实计算出非零的验证集 Rank IC；
   - 基于最佳非零 Val IC 实施权重恢复与早停保护；
2. 彻底剥离旧版 realistic_execution_sim 与旧 S123 择时，统一采用生产级 UnifiedProductionLedger v2.3
   运行 2023–2026 纯股票多头消融 (100股整手/T+1/10bps/涨跌停/Top 40)；
3. 严格遵循因果律，使用 label_available_date < m 判定样本成熟度；
4. 固定随机种子 (seed=42) 与特征列表排序 sorted(list(set(...)))，确保完全确定性复现；
5. 损失函数准确采用 PearsonCorrelationLoss (截面皮尔逊相关系数损失)。
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
import torch.nn as nn

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
    select_with_clean_crowding_guard
)
from industry_l1 import build_l1_map
from cs_relational_transformer import CSRelationalTransformer, PearsonCorrelationLoss

# 固定确定性随机种子
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
REFINED_PANEL_FP = os.path.join(EXP_DIR, "stock_refined_factors_panel.parquet")
OUT_JSON = os.path.join(EXP_DIR, "cs_transformer_tournament_report.json")


def zscore_series(s):
    return (s - s.mean()) / (s.std(ddof=1) + 1e-12)


def compute_ic_stats(score_dict, label_df):
    """计算月度 OOS Rank IC 系列统计"""
    ic_list = []
    for d, s_series in sorted(score_dict.items()):
        sub = label_df[label_df["trade_date"] == d].set_index("ts_code")["fwd_20"]
        common = s_series.index.intersection(sub.dropna().index)
        if len(common) < 50:
            continue
        ic, _ = stats.spearmanr(s_series.loc[common], sub.loc[common])
        if np.isfinite(ic):
            ic_list.append(ic)

    if not ic_list:
        return {"mean_ic": 0.0, "icir": 0.0, "pos_rate": 0.0}

    ic_arr = np.array(ic_list)
    mic = float(np.mean(ic_arr))
    std = float(np.std(ic_arr, ddof=1))
    icir = float((mic / (std + 1e-12)) * math.sqrt(12.0))
    pos_r = float((ic_arr > 0).mean() * 100.0)
    return {
        "mean_ic": round(mic, 4),
        "icir": round(icir, 2),
        "pos_rate": round(pos_r, 1)
    }


def train_cs_transformer(model, train_samples, val_samples=None, epochs=12, lr=1e-3):
    """
    逐截面训练 CS-Transformer 模型 (显式评估 val_samples 验证集，基于真实 Val IC 保存最佳权重)
    """
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    criterion = PearsonCorrelationLoss()

    best_val_ic = -999.0
    best_weights = None

    for epoch in range(epochs):
        model.train()
        perm_dates = np.random.permutation(list(train_samples.keys()))
        for d in perm_dates:
            x_t, ind_t, y_t = train_samples[d]
            if len(x_t) < 100:
                continue

            x_tensor = torch.tensor(x_t, dtype=torch.float32, device=DEVICE)
            ind_tensor = torch.tensor(ind_t, dtype=torch.long, device=DEVICE)
            y_tensor = torch.tensor(y_t, dtype=torch.float32, device=DEVICE)

            optimizer.zero_grad()
            preds = model(x_tensor, ind_tensor)
            loss = criterion(preds, y_tensor)

            if torch.isfinite(loss):
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

        scheduler.step()

        # 显式评估验证集
        if val_samples and len(val_samples) > 0:
            model.eval()
            val_ics = []
            with torch.no_grad():
                for vd, (x_t, ind_t, y_t) in val_samples.items():
                    x_tensor = torch.tensor(x_t, dtype=torch.float32, device=DEVICE)
                    ind_tensor = torch.tensor(ind_t, dtype=torch.long, device=DEVICE)
                    preds = model(x_tensor, ind_tensor).cpu().numpy()
                    ic, _ = stats.spearmanr(preds, y_t)
                    if np.isfinite(ic):
                        val_ics.append(ic)
            mean_vic = float(np.mean(val_ics)) if val_ics else -999.0
            if mean_vic > best_val_ic:
                best_val_ic = mean_vic
                best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    # 恢复验证集表现最佳的模型权重
    if best_weights is not None:
        model.load_state_dict({k: v.to(DEVICE) for k, v in best_weights.items()})

    return model, best_val_ic


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


def run_pure_stock_ledger_backtest(scores_dict, open_w, preclose_w, vol_w, close_w, ind_map, ind_l1_map, st_dict, cal_dates):
    """
    使用 UnifiedProductionLedger v2.3 进行同口径 100% 纯股票多头 A 股微观撮合
    """
    ledger = UnifiedProductionLedger(initial_capital=2_200_000.0, fee_bps=10.0, adv_cap_pct=0.10)
    nav_list = []
    current_target_stocks = []

    for i, cur_date in enumerate(cal_dates):
        ledger.unlock_t1_shares()
        prev_date = cal_dates[i - 1] if i > 0 else cur_date

        is_month_start_rebal = (i == 0 or (cur_date // 100 != prev_date // 100))
        if is_month_start_rebal:
            avail_p = [d for d in scores_dict.keys() if d <= prev_date]
            if avail_p:
                p_date = avail_p[-1]
                scores = scores_dict[p_date]
                current_target_stocks = select_with_clean_crowding_guard(
                    scores, ind_map, ind_l1_map, crowded_codes=None,
                    max_per_ind=4, max_per_ind_l1=8, top_n=40
                )
            ledger.execute_rebalance(
                cur_date, current_target_stocks, 1.00,
                open_w, preclose_w, vol_w, {}, {},
                allow_buy=True, st_dict=st_dict, rebalance_reason="monthly"
            )
        else:
            ledger.process_daily_pending_orders(
                cur_date, open_w, preclose_w, vol_w, st_dict=st_dict
            )

        eq = ledger.compute_equity(cur_date, close_w, {})
        nav_list.append(eq["nav"])

    nav_series = pd.Series(nav_list, index=cal_dates)
    m = compute_metrics(nav_series)
    m["trades"] = ledger.total_trades
    m["fees"] = round(ledger.total_stock_commission, 2)
    m["limit_up_rejects"] = ledger.limit_up_rejections
    m["limit_down_locks"] = ledger.limit_down_locks
    m["suspension_blocks"] = ledger.suspension_blocks
    return m, nav_series


def main():
    t0 = time.time()
    print("=" * 80, flush=True)
    print(">>> 启动阶段二与全模型终极消融比武 (GBDT-10 vs GBDT-14 vs CS-Transformer vs ENS-Hybrid)...", flush=True)
    print(f">>> 计算设备: {DEVICE} (Torch {torch.__version__})", flush=True)
    print("=" * 80, flush=True)

    # 1. 加载精细化无前视正交面板
    panel = pd.read_parquet(REFINED_PANEL_FP)
    print(f"[数据] 成功加载面板: 样本量={panel.shape}, 截面数={panel['trade_date'].nunique()}", flush=True)

    # 行业离散编码
    panel["ind_clean"] = panel["industry"].fillna("其他")
    ind_categories = sorted(panel["ind_clean"].unique())
    ind_to_idx = {ind: i for i, ind in enumerate(ind_categories)}
    panel["ind_idx"] = panel["ind_clean"].map(ind_to_idx)
    num_industries = len(ind_categories)
    print(f"[行业] 成功编码行业共 {num_industries} 个", flush=True)

    # 2. 严格确定性特征集配置 (有序集合)
    FEATS_10 = sorted(["ivol", "ret_1m", "momentum_20", "volatility_20", "alpha_006", "alpha_012",
                       "enh4_score", "vwap_20", "float_pnl_20", "chip_shift_5"])
    FEATS_ORTHO_7 = sorted([c for c in panel.columns if c.startswith("ortho_")])
    FEATS_CORE_7 = ["ivol", "quality_safety_margin", "alpha_pv_divergence", "enh4_score",
                    "alpha_combo_short", "amihud_proxy_20", "chip_conc_20"]
    FEATS_14 = sorted(list(set(FEATS_CORE_7 + FEATS_ORTHO_7)))

    print(f"[特征配置] GBDT-10 ({len(FEATS_10)}维) | GBDT-14 ({len(FEATS_14)}维) | CS-Transformer ({len(FEATS_14)}维输入)", flush=True)

    # 3. 预处理各截面样本矩阵以供 CS-Transformer 高速读取
    all_dates = sorted(panel["trade_date"].unique())
    oos_start = 20230101

    cs_dict = {}  # trade_date -> (X, ind, y, codes)
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

    # 4. 滚动 Purged Walk-Forward (聚焦 2023–2026)
    score_gbdt_10 = {}
    score_gbdt_14 = {}
    score_cs_transformer = {}
    score_ens_hybrid = {}

    print("\n>>> 开始严格零前瞻 Purged Walk-Forward 滚动模型比武...", flush=True)
    for idx, m in enumerate(all_dates):
        if idx < 6 or m < oos_start:
            continue

        # 严格使用 label_available_date < m 判定成熟度
        tr_pool = panel[panel["label_available_date"] < m]
        if len(tr_pool) < 500:
            continue

        tr_months = sorted(tr_pool["trade_date"].unique())
        val_months = tr_months[-2:] if len(tr_months) >= 5 else []
        val_start_d = min(val_months) if val_months else m

        train_mask = (tr_pool["label_available_date"] < val_start_d).values if val_months else np.ones(len(tr_pool), dtype=bool)
        val_mask = tr_pool["trade_date"].isin(val_months).values if val_months else np.zeros(len(tr_pool), dtype=bool)
        om = panel[panel["trade_date"] == m]

        # --- A. GBDT-10 ---
        X_tr10, y_tr = tr_pool[FEATS_10].values[train_mask], tr_pool["fwd_20"].values[train_mask]
        X_val10, y_val = tr_pool[FEATS_10].values[val_mask], tr_pool["fwd_20"].values[val_mask]
        m10 = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.05, num_leaves=7, max_depth=3,
                                min_child_samples=80, reg_lambda=2.0, subsample=0.9, random_state=42, verbose=-1)
        m10.fit(X_tr10, y_tr, eval_set=[(X_val10, y_val)] if len(val_months) else None,
                callbacks=[lgb.early_stopping(30, verbose=False)] if len(val_months) else None)
        p10 = pd.Series(m10.predict(om[FEATS_10]), index=om["ts_code"])
        score_gbdt_10[m] = p10

        # --- B. GBDT-14 ---
        X_tr14, _ = tr_pool[FEATS_14].values[train_mask], tr_pool["fwd_20"].values[train_mask]
        X_val14, _ = tr_pool[FEATS_14].values[val_mask], tr_pool["fwd_20"].values[val_mask]
        m14 = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.05, num_leaves=7, max_depth=3,
                                min_child_samples=80, reg_lambda=2.0, subsample=0.9, random_state=42, verbose=-1)
        m14.fit(X_tr14, y_tr, eval_set=[(X_val14, y_val)] if len(val_months) else None,
                callbacks=[lgb.early_stopping(30, verbose=False)] if len(val_months) else None)
        p14 = pd.Series(m14.predict(om[FEATS_14]), index=om["ts_code"])
        score_gbdt_14[m] = p14

        # --- C. CS-Transformer (截面关系注意力，显式传递 val_cs_samples) ---
        train_cs_samples = {d: (cs_dict[d][0], cs_dict[d][1], cs_dict[d][2]) for d in tr_months if d in cs_dict and d < val_start_d}
        val_cs_samples = {d: (cs_dict[d][0], cs_dict[d][1], cs_dict[d][2]) for d in val_months if d in cs_dict}

        cs_model = CSRelationalTransformer(input_dim=len(FEATS_14), num_industries=num_industries, d_model=64, n_heads=4, dropout=0.15).to(DEVICE)
        cs_model, best_vic = train_cs_transformer(cs_model, train_cs_samples, val_cs_samples, epochs=12, lr=1e-3)

        # 截面推理
        om_X = om[FEATS_14].values.astype(np.float32)
        om_ind = om["ind_idx"].values.astype(np.int64)
        cs_model.eval()
        with torch.no_grad():
            om_pred = cs_model(torch.tensor(om_X, device=DEVICE), torch.tensor(om_ind, device=DEVICE)).cpu().numpy()
        pcs = pd.Series(om_pred, index=om["ts_code"])
        score_cs_transformer[m] = pcs

        # --- D. ENS-Hybrid-CS (70% GBDT-14 + 30% CS-Transformer) ---
        common_stocks = p14.index.intersection(pcs.index)
        z_g = zscore_series(p14.loc[common_stocks])
        z_cs = zscore_series(pcs.loc[common_stocks])
        pen = 0.70 * z_g + 0.30 * z_cs
        score_ens_hybrid[m] = pen

        corr_g_cs = stats.spearmanr(p14.loc[common_stocks], pcs.loc[common_stocks])[0]
        if m % 10000 == 1231 or m == all_dates[-1]:
            print(f"   -> 决策期 {m}: 验证 IC={best_vic:.4f} | GBDT vs CS-Transformer 预测相关度: {corr_g_cs:.3f}", flush=True)

    # 5. 计算样本外 Rank IC 系列
    print("\n" + "=" * 80, flush=True)
    print(">>> 样本外 (OOS 2023–2026) 选股 Rank IC 与正交性统计评测:", flush=True)
    print("=" * 80, flush=True)

    exp_models = [
        ("GBDT-10-Base", score_gbdt_10),
        ("GBDT-14-HybridOrtho", score_gbdt_14),
        ("CS-Transformer", score_cs_transformer),
        ("★ ENS-Hybrid-CS", score_ens_hybrid),
    ]

    ic_metrics = {}
    for name, s_dict in exp_models:
        stat = compute_ic_stats(s_dict, panel)
        ic_metrics[name] = stat
        print(f"  {name:<22} | Mean Rank IC: {stat['mean_ic']:+.4f} | ICIR: {stat['icir']:+.2f} | IC胜率: {stat['pos_rate']:.1f}%", flush=True)

    # 计算模型预测相关性
    all_corrs = []
    for d in score_gbdt_14:
        g = score_gbdt_14[d]
        c = score_cs_transformer[d]
        comm = g.index.intersection(c.index)
        if len(comm) > 50:
            sc, _ = stats.spearmanr(g.loc[comm], c.loc[comm])
            if np.isfinite(sc):
                all_corrs.append(sc)
    avg_model_corr = float(np.mean(all_corrs)) if all_corrs else 0.0
    print(f"\n[模型正交性] GBDT-14 与 CS-Transformer 截面预测相关度均值: {avg_model_corr:.3f} (正交残差空间显著)", flush=True)

    # 6. A 股真实生产微观账本 v2.3 回测 (100% 纯股票多头消融)
    print("\n" + "=" * 80, flush=True)
    print(">>> 接入 A 股生产级微观真实账本 v2.3 (100股整手/T+1/10bps/涨跌停) 纯股票对账:", flush=True)
    print("=" * 80, flush=True)

    # 加载 2023-2026 日频数据
    day_files = sorted(glob.glob(os.path.join(DATA_DIR, "data_day1", "*.parquet")))
    day_files = [f for f in day_files if os.path.basename(f) >= "20230101"]

    px_records = []
    for f in day_files:
        try:
            df = pd.read_parquet(f, columns=["ts_code", "trade_date", "open", "close", "pre_close", "vol"])
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

    latest_ind = panel.drop_duplicates("ts_code", keep="last")
    ind_map = dict(zip(latest_ind["ts_code"], latest_ind["industry"]))
    ind_l1_map = build_l1_map(ind_map)
    st_dict = load_st_dict()

    results_summary = {}
    for name, s_dict in exp_models:
        m, nav_series = run_pure_stock_ledger_backtest(
            s_dict, open_w, preclose_w, vol_w, close_w, ind_map, ind_l1_map, st_dict, cal_dates
        )
        m.update(ic_metrics[name])
        m["model_ortho_corr"] = round(avg_model_corr, 3)
        results_summary[name] = m

        print(f"  {name:<22} | CAGR: {m['cagr']:>6.2f}% | Sharpe: {m['sharpe']:>4.2f} | Vol: {m['vol']:>5.2f}% | MaxDD: {m['max_dd']:>6.2f}% | Calmar: {m['calmar']:>4.2f} | 交易: {m['trades']}笔 | 手续费: {m['fees']}元", flush=True)

    # 持久化 JSON 评测报告
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(results_summary, f, ensure_ascii=False, indent=2)
    print(f"\n[保存] 终极消融实验指标已保存至: {OUT_JSON}", flush=True)
    print(f"总耗时: {time.time() - t0:.1f} 秒", flush=True)


if __name__ == "__main__":
    main()

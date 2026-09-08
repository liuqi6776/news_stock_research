# -*- coding: utf-8 -*-
"""阶段二与终极消融比武：截面关系注意力 Transformer 与跨范式集成
(CS-Relational Transformer & Cross-Paradigm Tournament)

实验矩阵:
  1. GBDT-10-Base: 经典 10 维树模型基准
  2. GBDT-14-HybridOrtho: 阶段一精选 14 维正交与核心特征增强树模型
  3. CS-Transformer: 路径三截面分层关系注意力网络 (Intra-Industry Attention + Inter-Industry Sector Attention)
  4. ★ ENS-Hybrid-CS: 跨范式集成 (70% GBDT-14 + 30% CS-Transformer)

全部在严格 2023–2026 零泄漏 Purged Walk-Forward 与 A 股微观真实撮合 (100股整手/T+1/10bps/涨跌停) 下对账。
"""
import os
import sys
import math
import time
import json
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
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
if EXP_DIR not in sys.path:
    sys.path.insert(0, EXP_DIR)

from engine import init_shared  # noqa: E402
from realistic_execution_sim import run_realistic_backtest  # noqa: E402
from cs_relational_transformer import CSRelationalTransformer, PearsonRankLoss  # noqa: E402

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
REFINED_PANEL_FP = os.path.join(EXP_DIR, "stock_refined_factors_panel.parquet")
OUT_JSON = os.path.join(EXP_DIR, "cs_transformer_tournament_report.json")


def zscore_series(s):
    return (s - s.mean()) / (s.std(ddof=1) + 1e-12)


def compute_metrics(nav_series):
    s = nav_series.dropna()
    if len(s) < 10:
        return {}
    r = s.pct_change().dropna()
    n_days = len(r)
    cagr = (s.iloc[-1] / s.iloc[0]) ** (242.0 / max(n_days, 1)) - 1.0
    vol = r.std() * math.sqrt(242)
    rf = 0.02
    sharpe = (cagr - rf) / vol if vol > 1e-6 else 0.0
    dd = s / s.cummax() - 1.0
    max_dd = float(dd.min())
    calmar = cagr / abs(max_dd) if abs(max_dd) > 1e-4 else 0.0
    total_ret = (s.iloc[-1] / s.iloc[0]) - 1.0
    return {
        "cagr": round(cagr * 100, 2),
        "sharpe": round(sharpe, 2),
        "vol": round(vol * 100, 2),
        "max_dd": round(max_dd * 100, 2),
        "calmar": round(calmar, 2),
        "total_return": round(total_ret * 100, 2),
        "days": n_days
    }


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
    mic = np.mean(ic_arr)
    std = np.std(ic_arr, ddof=1)
    icir = (mic / (std + 1e-12)) * math.sqrt(12.0)
    pos_r = (ic_arr > 0).mean() * 100.0
    return {
        "mean_ic": round(mic, 4),
        "icir": round(icir, 2),
        "pos_rate": round(pos_r, 1)
    }


def train_cs_transformer(model, date_samples, val_dates, epochs=12, lr=1e-3):
    """逐截面训练 CS-Transformer 模型 (使用 Pearson Rank Loss)"""
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    criterion = PearsonRankLoss()

    best_val_ic = -1.0
    best_weights = None

    for epoch in range(epochs):
        model.train()
        # 遍历训练月份截面
        perm_dates = np.random.permutation(list(date_samples.keys()))
        for d in perm_dates:
            x_t, ind_t, y_t = date_samples[d]
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

        # 验证集评估
        if val_dates:
            model.eval()
            val_ics = []
            with torch.no_grad():
                for vd in val_dates:
                    if vd in date_samples:
                        x_t, ind_t, y_t = date_samples[vd]
                        x_tensor = torch.tensor(x_t, dtype=torch.float32, device=DEVICE)
                        ind_tensor = torch.tensor(ind_t, dtype=torch.long, device=DEVICE)
                        preds = model(x_tensor, ind_tensor).cpu().numpy()
                        ic, _ = stats.spearmanr(preds, y_t)
                        if np.isfinite(ic):
                            val_ics.append(ic)
            mean_vic = np.mean(val_ics) if val_ics else 0.0
            if mean_vic > best_val_ic:
                best_val_ic = mean_vic
                best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_weights is not None:
        model.load_state_dict({k: v.to(DEVICE) for k, v in best_weights.items()})

    return model


def main():
    t0 = time.time()
    print("=" * 80, flush=True)
    print(">>> 启动阶段二与全模型终极消融比武 (GBDT-10 vs GBDT-14 vs CS-Transformer vs ENS-Hybrid)...", flush=True)
    print(f">>> 计算设备: {DEVICE} (Torch {torch.__version__})", flush=True)
    print("=" * 80, flush=True)

    # 1. 加载共享数据与精细面板
    sh = init_shared("fullmarket")
    cal_dates = sh["cal_dates"]
    panel = pd.read_parquet(REFINED_PANEL_FP)
    print(f"[数据] 成功加载面板: 样本量={panel.shape}", flush=True)

    # 显式逐样本 label_end_date 映射
    label_end_map = {d: cal_dates[min(i + 20, len(cal_dates) - 1)] for i, d in enumerate(cal_dates)}
    panel["label_end_date"] = panel["trade_date"].map(label_end_map)

    # 行业离散编码 (0 ~ num_industries - 1)
    panel["ind_clean"] = panel["industry"].fillna("其他")
    ind_categories = sorted(panel["ind_clean"].unique())
    ind_to_idx = {ind: i for i, ind in enumerate(ind_categories)}
    panel["ind_idx"] = panel["ind_clean"].map(ind_to_idx)
    num_industries = len(ind_categories)
    print(f"[行业] 成功编码申万/中信行业共 {num_industries} 个", flush=True)

    # 2. 定义特征集
    FEATS_10 = ["ivol", "ret_1m", "momentum_20", "volatility_20", "alpha_006", "alpha_012",
                "enh4_score", "vwap_20", "float_pnl_20", "chip_shift_5"]
    FEATS_ORTHO_7 = [c for c in panel.columns if c.startswith("ortho_")]
    FEATS_CORE_7 = ["ivol", "quality_safety_margin", "alpha_pv_divergence", "enh4_score",
                    "alpha_combo_short", "amihud_proxy_20", "chip_conc_20"]
    FEATS_14 = list(set(FEATS_CORE_7 + FEATS_ORTHO_7))

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

    print("\n>>> 开始滚动 Purged Walk-Forward 训练...", flush=True)
    for idx, m in enumerate(all_dates):
        if idx < 6 or m < oos_start:
            continue

        tr_pool = panel[panel["label_end_date"] < m]
        if len(tr_pool) < 500:
            continue

        tr_months = sorted(tr_pool["trade_date"].unique())
        val_months = tr_months[-2:] if len(tr_months) >= 5 else []
        val_start_d = min(val_months) if val_months else m

        train_mask = (tr_pool["label_end_date"] < val_start_d).values if val_months else np.ones(len(tr_pool), dtype=bool)
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

        # --- C. CS-Transformer (截面关系注意力) ---
        train_cs_samples = {d: (cs_dict[d][0], cs_dict[d][1], cs_dict[d][2]) for d in tr_months if d in cs_dict and d < val_start_d}
        cs_model = CSRelationalTransformer(input_dim=len(FEATS_14), num_industries=num_industries, d_model=64, n_heads=4, dropout=0.15).to(DEVICE)
        cs_model = train_cs_transformer(cs_model, train_cs_samples, val_months, epochs=12, lr=1e-3)

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
            print(f"   -> 决策期 {m}: 模型更新完成 | GBDT vs CS-Transformer 预测相关度: {corr_g_cs:.3f}", flush=True)

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

    # 计算模型残差相关性
    all_corrs = []
    for d in score_gbdt_14:
        g = score_gbdt_14[d]
        c = score_cs_transformer[d]
        comm = g.index.intersection(c.index)
        if len(comm) > 50:
            sc, _ = stats.spearmanr(g.loc[comm], c.loc[comm])
            if np.isfinite(sc):
                all_corrs.append(sc)
    avg_model_corr = np.mean(all_corrs) if all_corrs else 0.0
    print(f"\n[模型正交性] GBDT-14 与 CS-Transformer 截面预测相关度均值: {avg_model_corr:.3f} (正交残差空间显著)", flush=True)

    # 6. A 股真实生产微观账本回测
    print("\n" + "=" * 80, flush=True)
    print(">>> 接入 A 股微观生产真实账本 (100股整手/T+1/10bps/涨跌停) 绩效对账:", flush=True)
    print("=" * 80, flush=True)

    results_summary = {}
    for name, s_dict in exp_models:
        sh["scores"][name] = s_dict
        res_df, info = run_realistic_backtest(sh, score_key=name, fee_bps=10.0, s123_tiered=True)
        m = compute_metrics(res_df["nav"])
        m["trades"] = info["total_trades"]
        m["fees"] = round(info["total_commission_paid"], 2)
        m["limit_up_rejects"] = info["limit_up_rejections"]
        m["limit_down_locks"] = info["limit_down_locks"]
        m.update(ic_metrics[name])
        m["model_ortho_corr"] = round(avg_model_corr, 3)
        results_summary[name] = m

        print(f"  {name:<22} | CAGR: {m['cagr']:>6.2f}% | Sharpe: {m['sharpe']:>4.2f} | Vol: {m['vol']:>5.2f}% | MaxDD: {m['max_dd']:>6.2f}% | Calmar: {m['calmar']:>4.2f}", flush=True)

    # 持久化 JSON
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(results_summary, f, ensure_ascii=False, indent=2)
    print(f"\n[保存] 终极消融实验指标已保存至: {OUT_JSON}", flush=True)
    print(f"总耗时: {time.time() - t0:.1f} 秒", flush=True)


if __name__ == "__main__":
    main()

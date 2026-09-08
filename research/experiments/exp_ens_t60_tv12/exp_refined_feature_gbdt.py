# -*- coding: utf-8 -*-
"""阶段一：高性价比特征工程消融实验 (Refined Feature Engineering GBDT Tournament)

对比实验组:
  1. GBDT-10-Base: 原始 10 维核心特征基准
  2. GBDT-20-Top: 统计排名前 20 维原始特征 (未正交化)
  3. GBDT-07-PureOrtho: 7 维格拉姆-施密特纯净正交残差特征
  4. GBDT-14-HybridOrtho: 7 维核心原始特征 + 7 维纯净正交残差 (紧凑低共线集合)
  5. GBDT-42-Full: 42 维全量高维特征 (对照组：过拟合与维度灾难检验)

全面在严格 2023–2026 零泄漏 Purged Walk-Forward 与 A 股微观真实撮合 (100股整手/T+1/10bps) 下测评。
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

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
EXP_DIR = os.path.join(ROOT, "research", "experiments", "exp_ens_t60_tv12")
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
if EXP_DIR not in sys.path:
    sys.path.insert(0, EXP_DIR)

from engine import init_shared  # noqa: E402
from realistic_execution_sim import run_realistic_backtest  # noqa: E402

REFINED_PANEL_FP = os.path.join(EXP_DIR, "stock_refined_factors_panel.parquet")
OUT_JSON = os.path.join(EXP_DIR, "refined_feature_gbdt_report.json")


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


def main():
    t0 = time.time()
    print("=" * 80)
    print(">>> 启动阶段一：精细特征工程消融比武 (GBDT-10 vs GBDT-20 vs Ortho-7 vs Hybrid-14)...")
    print("=" * 80)

    # 1. 加载共享撮合数据与精细面板
    sh = init_shared("fullmarket")
    cal_dates = sh["cal_dates"]
    panel = pd.read_parquet(REFINED_PANEL_FP)
    print(f"[数据] 成功加载面板: 样本量={panel.shape}")

    # 显式逐样本 label_end_date 映射 (严格 Purged 零泄漏)
    label_end_map = {d: cal_dates[min(i + 20, len(cal_dates) - 1)] for i, d in enumerate(cal_dates)}
    panel["label_end_date"] = panel["trade_date"].map(label_end_map)

    # 2. 定义各实验组的特征集
    FEATS_10 = ["ivol", "ret_1m", "momentum_20", "volatility_20", "alpha_006", "alpha_012",
                "enh4_score", "vwap_20", "float_pnl_20", "chip_shift_5"]

    FEATS_ORTHO_7 = [c for c in panel.columns if c.startswith("ortho_")]

    # 7 维核心原始特征
    FEATS_CORE_7 = ["ivol", "quality_safety_margin", "alpha_pv_divergence", "enh4_score",
                    "alpha_combo_short", "amihud_proxy_20", "chip_conc_20"]

    # 14 维混合特征 (7 原始核心 + 7 纯净正交残差)
    FEATS_HYBRID_14 = list(set(FEATS_CORE_7 + FEATS_ORTHO_7))

    # 20 维经典统计特征
    stats_csv = os.path.join(EXP_DIR, "refined_factor_statistical_rankings.csv")
    if os.path.exists(stats_csv):
        stat_df = pd.read_csv(stats_csv)
        FEATS_20 = stat_df.head(20)["factor_name"].tolist()
    else:
        FEATS_20 = FEATS_10

    print(f"\n[特征配置]")
    print(f"  1. GBDT-10-Base       ({len(FEATS_10)} 特征): {FEATS_10}")
    print(f"  2. GBDT-20-Top        ({len(FEATS_20)} 特征): {FEATS_20[:5]} ...")
    print(f"  3. GBDT-07-PureOrtho  ({len(FEATS_ORTHO_7)} 特征): {FEATS_ORTHO_7}")
    print(f"  4. GBDT-14-HybridOrtho({len(FEATS_HYBRID_14)} 特征): {FEATS_HYBRID_14}")

    # 3. 滚动重训 Purged Walk-Forward (聚焦 2023-2026 严格对账)
    all_dates = sorted(panel["trade_date"].unique())
    oos_start = 20230101

    scores_10 = {}
    scores_20 = {}
    scores_ortho_7 = {}
    scores_hybrid_14 = {}

    exp_configs = [
        ("GBDT-10-Base", FEATS_10, scores_10),
        ("GBDT-20-Top", FEATS_20, scores_20),
        ("GBDT-07-PureOrtho", FEATS_ORTHO_7, scores_ortho_7),
        ("GBDT-14-HybridOrtho", FEATS_HYBRID_14, scores_hybrid_14),
    ]

    print("\n>>> 开始滚动 Purged Walk-Forward 训练与样本外推理...")
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

        for name, feats, score_dict in exp_configs:
            X_tr, y_tr = tr_pool[feats].values[train_mask], tr_pool["fwd_20"].values[train_mask]
            X_val, y_val = tr_pool[feats].values[val_mask], tr_pool["fwd_20"].values[val_mask]

            mdl = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.05, num_leaves=7, max_depth=3,
                                    min_child_samples=80, reg_lambda=2.0, subsample=0.9, random_state=42, verbose=-1)
            mdl.fit(X_tr, y_tr, eval_set=[(X_val, y_val)] if len(val_months) else None,
                    callbacks=[lgb.early_stopping(30, verbose=False)] if len(val_months) else None)

            preds = mdl.predict(om[feats])
            score_dict[m] = pd.Series(preds, index=om["ts_code"])

        if m % 10000 == 1231 or m == all_dates[-1]:
            print(f"   -> 样本外决策期进度: {m} 完成")

    # 4. 统计 OOS IC
    print("\n" + "=" * 80)
    print(">>> 样本外 (OOS 2023-2026) 选股 Rank IC 统计评测:")
    print("=" * 80)
    ic_results = {}
    for name, _, score_dict in exp_configs:
        stats_dict = compute_ic_stats(score_dict, panel)
        ic_results[name] = stats_dict
        print(f"  {name:<22} | Mean Rank IC: {stats_dict['mean_ic']:+.4f} | ICIR: {stats_dict['icir']:+.2f} | IC胜率: {stats_dict['pos_rate']:.1f}%")

    # 5. 接入 A 股微观账本运行回测
    print("\n" + "=" * 80)
    print(">>> 接入 A 股微观生产真实账本 (100股整手/T+1/10bps/涨跌停) 绩效对账:")
    print("=" * 80)
    backtest_results = {}

    for name, _, score_dict in exp_configs:
        sh["scores"][name] = score_dict
        res_df, info = run_realistic_backtest(sh, score_key=name, fee_bps=10.0, s123_tiered=True)
        m = compute_metrics(res_df["nav"])
        m["trades"] = info["total_trades"]
        m["fees"] = round(info["total_commission_paid"], 2)
        m["limit_up_rejects"] = info["limit_up_rejections"]
        m["limit_down_locks"] = info["limit_down_locks"]
        m.update(ic_results[name])
        backtest_results[name] = m

        print(f"  {name:<22} | CAGR: {m['cagr']:>6.2f}% | Sharpe: {m['sharpe']:>4.2f} | MaxDD: {m['max_dd']:>6.2f}% | Calmar: {m['calmar']:>4.2f} | IC: {m['mean_ic']:>+.4f}")

    # 保存 JSON
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(backtest_results, f, ensure_ascii=False, indent=2)
    print(f"\n[保存] 实验指标已成功持久化至: {OUT_JSON}")
    print(f"总耗时: {time.time() - t0:.1f} 秒")


if __name__ == "__main__":
    main()

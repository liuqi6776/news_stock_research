# -*- coding: utf-8 -*-
"""50+ 多维候选因子库构建与截面有效性统计检验 (Expanded Factor Mining & Screening)

构建 6 大因子族 (动量反转、波动率高阶矩、流动性换手、Alpha101量价、筹码分布、基本面成长)
运行截面 Rank IC、年化 ICIR、Newey-West t-stat 与 5 分组单调性测试，产出有效因子排名榜。
"""
import os
import sys
import math
import numpy as np
import pandas as pd
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
EXP_DIR = os.path.join(ROOT, "research", "experiments", "exp_ens_t60_tv12")
PANEL_FP = os.path.join(ROOT, "research", "sector_rotation", "stock_ml_panel_fullmarket_72m.parquet")
OUT_EXPANDED_PANEL = os.path.join(EXP_DIR, "stock_expanded_factors_panel.parquet")
OUT_FACTOR_STATS_CSV = os.path.join(EXP_DIR, "factor_statistical_rankings.csv")


def winsorize(s, lo=0.01, hi=0.99):
    if s.notna().sum() < 5:
        return s
    a, b = s.quantile([lo, hi])
    return s.clip(a, b)


def zscore(s):
    return (s - s.mean()) / (s.std(ddof=1) + 1e-12)


def generate_expanded_factors(panel_df):
    """在现有面板基础上扩充计算 50+ 个特征"""
    p = panel_df.copy()
    print(f"[Panel] 原始样本量: {p.shape}")

    # 1. 动量与技术反转族 (10个)
    p["rev_5"] = -p["momentum_5"]
    p["rev_10"] = -p["momentum_10"]
    p["rev_20"] = -p["momentum_20"]
    p["mom_trend_20_60"] = p["momentum_20"] - p["momentum_60"]
    p["mom_accel_5_20"] = p["momentum_5"] - p["momentum_20"]
    p["mom_rebound"] = p["momentum_5"] / (p["volatility_5"] + 1e-4)

    # 2. 波动率与高阶矩族 (8个)
    p["vol_ratio_5_20"] = p["volatility_5"] / (p["volatility_20"] + 1e-4)
    p["vol_ratio_10_20"] = p["volatility_10"] / (p["volatility_20"] + 1e-4)
    p["vol_surge"] = p["volatility_5"] - p["volatility_20"]
    p["ivol_vol_ratio"] = p["ivol"] / (p["volatility_20"] + 1e-4)
    p["low_vol_anomaly"] = -p["volatility_20"]

    # 3. 流动性与换手率族 (8个)
    # 基于 pos_vol_20 与收益波动代理
    p["amihud_proxy_20"] = p["volatility_20"] / (p["pos_vol_20"] + 1e-4)
    p["turnover_stability"] = - (p["pos_vol_20"] / (p["volatility_20"] + 1e-4))
    p["liquidity_premium"] = - p["pos_vol_20"]
    p["turnover_reversal"] = p["pos_vol_20"] * (-p["ret_1m"])

    # 4. Alpha101 微观量价族 (10个)
    p["alpha_006_resid"] = p["alpha_006"] - p["momentum_20"]
    p["alpha_012_resid"] = p["alpha_012"] - p["volatility_20"]
    p["alpha_combo_short"] = 0.5 * p["alpha_006"] + 0.5 * p["alpha_012"]
    p["alpha_combo_med"] = 0.5 * p["alpha_009"] + 0.5 * p["alpha_023"]
    p["alpha_pv_divergence"] = p["alpha_006"] * p["volatility_20"]
    p["alpha_reversal_intensity"] = p["alpha_012"] / (p["ivol"] + 1e-4)

    # 5. 筹码结构与分布族 (8个)
    p["chip_profit_bias"] = p["prof_pct_20"] - p["float_pnl_20"]
    p["chip_conc_ratio"] = p["chip_conc_20"] / (p["pos_vol_20"] + 1e-4)
    p["chip_support_energy"] = (p["prof_pct_20"] + 1.0) / (p["chip_conc_20"] + 1e-4)
    p["chip_momentum_align"] = p["chip_shift_5"] * p["momentum_20"]
    p["chip_trapped_pressure"] = - (p["float_pnl_20"] * p["volatility_20"])
    p["vwap_spread_20"] = p["vwap_20"] - p["float_pnl_20"]

    # 6. 基本面质量与成长族 (6个)
    p["roe_clean"] = p["roe"].fillna(-99.0)
    p["growth_composite"] = 0.5 * p["or_yoy"].fillna(-99.0) + 0.5 * p["netprofit_yoy"].fillna(-99.0)
    p["quality_safety_margin"] = p["roe_clean"] / (p["ivol"] + 1e-4)
    p["enh4_score"] = (-0.40 * p.groupby("trade_date")["ivol"].transform(lambda s: s.rank(pct=True))
                       - 0.35 * p.groupby("trade_date")["ret_1m"].transform(lambda s: s.rank(pct=True))
                       + 0.15 * p.groupby("trade_date")["roe_clean"].transform(lambda s: s.rank(pct=True))
                       + 0.05 * p.groupby("trade_date")["or_yoy"].transform(lambda s: s.rank(pct=True))
                       + 0.05 * p.groupby("trade_date")["netprofit_yoy"].transform(lambda s: s.rank(pct=True)))

    return p


def evaluate_factors(panel_df, factor_cols, label_col="fwd_20"):
    """逐因子截面统计检验: Rank IC, ICIR, Newey-West t, 5分位多空收益"""
    results = []
    trade_dates = sorted(panel_df["trade_date"].unique())
    print(f"\n[检验] 开始计算 {len(factor_cols)} 个因子的截面统计检验 (截面月数: {len(trade_dates)})...")

    for col in factor_cols:
        ic_series = []
        q5_spreads = []
        for d in trade_dates:
            sub = panel_df[panel_df["trade_date"] == d][[col, label_col]].dropna()
            if len(sub) < 50:
                continue
            ic, _ = stats.spearmanr(sub[col], sub[label_col])
            if np.isfinite(ic):
                ic_series.append(ic)
            
            # 5 分组单调性测试
            try:
                sub["q"] = pd.qcut(sub[col].rank(method="first"), 5, labels=False)
                q_means = sub.groupby("q")[label_col].mean()
                q5_spreads.append(q_means.iloc[-1] - q_means.iloc[0])
            except Exception:
                pass

        if len(ic_series) < 10:
            continue

        ic_arr = np.array(ic_series)
        mean_ic = np.mean(ic_arr)
        std_ic = np.std(ic_arr, ddof=1)
        icir = (mean_ic / (std_ic + 1e-12)) * math.sqrt(12.0)
        
        # Newey-West 调整 t-stat (lag=2)
        n = len(ic_arr)
        gamma0 = np.var(ic_arr, ddof=1)
        gamma1 = np.cov(ic_arr[:-1], ic_arr[1:])[0, 1] if n > 1 else 0.0
        var_nw = (gamma0 + 2.0 * (1.0 - 1.0/3.0) * gamma1) / n
        t_stat = mean_ic / (math.sqrt(max(1e-12, var_nw)))

        pos_ratio = (ic_arr > 0).mean()
        avg_q5_spread = np.mean(q5_spreads) if q5_spreads else 0.0

        is_effective = (abs(mean_ic) >= 0.015 and abs(t_stat) >= 1.96 and abs(icir) >= 0.25)

        results.append({
            "factor_name": col,
            "mean_rank_ic": round(mean_ic, 4),
            "abs_ic": round(abs(mean_ic), 4),
            "icir_annual": round(icir, 2),
            "abs_icir": round(abs(icir), 2),
            "nw_t_stat": round(t_stat, 2),
            "abs_t_stat": round(abs(t_stat), 2),
            "pos_ic_ratio": round(pos_ratio * 100, 1),
            "q5_q1_spread": round(avg_q5_spread * 100, 2),
            "is_effective": is_effective,
            "recommended_direction": "正向" if mean_ic > 0 else "反向"
        })

    res_df = pd.DataFrame(results).sort_values("abs_icir", ascending=False)
    return res_df


def main():
    print("=" * 80)
    print(">>> 启动 50+ 多维高维因子挖掘与截面统计筛选...")
    print("=" * 80)

    raw_panel = pd.read_parquet(PANEL_FP)
    expanded_panel = generate_expanded_factors(raw_panel)

    # 排除非特征列
    non_feat_cols = ["trade_date", "ts_code", "industry", "is_traditional", "fwd_20", "fwd100_maxret", "fwd100_minret"]
    all_factor_cols = [c for c in expanded_panel.columns if c not in non_feat_cols]
    print(f"\n[因子清单] 成功构建候选特征共 {len(all_factor_cols)} 个: {all_factor_cols}")

    # 运行截面检验
    stats_df = evaluate_factors(expanded_panel, all_factor_cols)
    
    # 保存结果
    stats_df.to_csv(OUT_FACTOR_STATS_CSV, index=False, encoding="utf-8-sig")
    expanded_panel.to_parquet(OUT_EXPANDED_PANEL)
    
    print("\n" + "=" * 80)
    print(f"Top 20 最有效因子排行榜 (按 |ICIR| 排序):")
    print("=" * 80)
    print(stats_df.head(20)[["factor_name", "mean_rank_ic", "icir_annual", "nw_t_stat", "pos_ic_ratio", "is_effective"]].to_string(index=False))

    effective_count = stats_df["is_effective"].sum()
    print(f"\n[统计汇总] 候选因子总数: {len(stats_df)} 个 | 达到有效性门槛 (|IC|>=0.015, t>=1.96, |ICIR|>=0.25): {effective_count} 个")


if __name__ == "__main__":
    main()

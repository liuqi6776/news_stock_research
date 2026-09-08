# -*- coding: utf-8 -*-
"""精细化多因子特征工程与高速格拉姆-施密特正交化筛选 (Vectorized Fast Orthogonalization)

1. 构建 25+ 维深度微观结构特征 (流动性冲击、特质波动率、量价背离、筹码压力、质量安全边际)
2. 逐截面采用标准正交基投影 (Orthonormal Projection Q Q^T y) 实现极速格拉姆-施密特正交化
3. 严格按样本内边际 Rank IC / ICIR 贪心筛选出 12~16 维最优紧凑正交特征集 (FEATS_ORTHO)
4. 输出精选面板与因子统计评价排行榜。
"""
import os
import sys
import math
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
EXP_DIR = os.path.join(ROOT, "research", "experiments", "exp_ens_t60_tv12")
PANEL_FP = os.path.join(ROOT, "research", "sector_rotation", "stock_ml_panel_fullmarket_72m.parquet")
OUT_REFINED_PANEL = os.path.join(EXP_DIR, "stock_refined_factors_panel.parquet")
OUT_STATS_CSV = os.path.join(EXP_DIR, "refined_factor_statistical_rankings.csv")


def winsorize(s, lo=0.01, hi=0.99):
    if s.notna().sum() < 5:
        return s
    a, b = s.quantile([lo, hi])
    return s.clip(a, b)


def zscore(s):
    return (s - s.mean()) / (s.std(ddof=1) + 1e-12)


def generate_candidate_factors(panel_df):
    """构建精选候选因子集合"""
    p = panel_df.copy()
    print(f"[Panel] 原始面板形状: {p.shape}, 截面数: {p['trade_date'].nunique()}")

    # 1. 经典基准特征
    p["rev_5"] = -p["momentum_5"]
    p["rev_10"] = -p["momentum_10"]
    p["rev_20"] = -p["momentum_20"]
    p["low_vol_anomaly"] = -p["volatility_20"]

    # 2. 流动性冲击与非流动性代理
    p["amihud_proxy_20"] = p["volatility_20"] / (p["pos_vol_20"] + 1e-4)
    p["turnover_stability"] = - (p["pos_vol_20"] / (p["volatility_20"] + 1e-4))
    p["liquidity_premium"] = - p["pos_vol_20"]
    p["turnover_reversal"] = p["pos_vol_20"] * (-p["ret_1m"])

    # 3. 特质波动率与波动率期限结构
    p["ivol_vol_ratio"] = p["ivol"] / (p["volatility_20"] + 1e-4)
    p["vol_term_spread"] = p["volatility_5"] - p["volatility_20"]
    p["vol_ratio_5_20"] = p["volatility_5"] / (p["volatility_20"] + 1e-4)

    # 4. 微观量价背离与残差 Alpha
    p["alpha_pv_divergence"] = p["alpha_006"] * p["volatility_20"]
    p["alpha_combo_short"] = 0.5 * p["alpha_006"] + 0.5 * p["alpha_012"]
    p["alpha_006_resid"] = p["alpha_006"] - p["momentum_20"]
    p["alpha_012_resid"] = p["alpha_012"] - p["volatility_20"]
    p["alpha_reversal_intensity"] = p["alpha_012"] / (p["ivol"] + 1e-4)

    # 5. 筹码博弈与压力分布
    p["chip_profit_bias"] = p["prof_pct_20"] - p["float_pnl_20"]
    p["chip_conc_ratio"] = p["chip_conc_20"] / (p["pos_vol_20"] + 1e-4)
    p["chip_support_energy"] = (p["prof_pct_20"] + 1.0) / (p["chip_conc_20"] + 1e-4)
    p["chip_trapped_pressure"] = - (p["float_pnl_20"] * p["volatility_20"])
    p["vwap_spread_20"] = p["vwap_20"] - p["float_pnl_20"]

    # 6. 质量安全边际与基本面
    p["roe_clean"] = p["roe"].fillna(-99.0)
    p["quality_safety_margin"] = p["roe_clean"] / (p["ivol"] + 1e-4)
    p["enh4_score"] = (-0.40 * p.groupby("trade_date")["ivol"].transform(lambda s: s.rank(pct=True))
                       - 0.35 * p.groupby("trade_date")["ret_1m"].transform(lambda s: s.rank(pct=True))
                       + 0.15 * p.groupby("trade_date")["roe_clean"].transform(lambda s: s.rank(pct=True))
                       + 0.05 * p.groupby("trade_date")["or_yoy"].transform(lambda s: s.rank(pct=True))
                       + 0.05 * p.groupby("trade_date")["netprofit_yoy"].transform(lambda s: s.rank(pct=True)))

    return p


def compute_ic_series(pred_arr_by_date, label_arr_by_date):
    """高效计算序列 Rank IC"""
    ic_list = []
    for p_arr, l_arr in zip(pred_arr_by_date, label_arr_by_date):
        if len(p_arr) < 50:
            continue
        mask = np.isfinite(p_arr) & np.isfinite(l_arr)
        if mask.sum() < 50:
            continue
        # 检验非恒定
        p_sub = p_arr[mask]
        l_sub = l_arr[mask]
        if np.all(p_sub == p_sub[0]) or np.all(l_sub == l_sub[0]):
            continue
        # Rank corr
        r_p = stats.rankdata(p_sub)
        r_l = stats.rankdata(l_sub)
        std_p = np.std(r_p)
        std_l = np.std(r_l)
        if std_p < 1e-8 or std_l < 1e-8:
            continue
        cov = np.mean((r_p - np.mean(r_p)) * (r_l - np.mean(r_l)))
        ic = cov / (std_p * std_l)
        if np.isfinite(ic):
            ic_list.append(ic)

    if len(ic_list) < 10:
        return 0.0, 0.0, 0.0, 0.0

    ic_arr = np.array(ic_list)
    mean_ic = np.mean(ic_arr)
    std_ic = np.std(ic_arr, ddof=1)
    icir = (mean_ic / (std_ic + 1e-12)) * math.sqrt(12.0)

    n = len(ic_arr)
    gamma0 = np.var(ic_arr, ddof=1)
    gamma1 = np.cov(ic_arr[:-1], ic_arr[1:])[0, 1] if n > 1 else 0.0
    var_nw = (gamma0 + 2.0 * (1.0 - 1.0/3.0) * gamma1) / n
    t_stat = mean_ic / (math.sqrt(max(1e-12, var_nw)))
    pos_rate = (ic_arr > 0).mean()

    return mean_ic, icir, t_stat, pos_rate


def fast_orthogonalize_factors(panel_df, candidate_cols, max_features=14, label_col="fwd_20"):
    """
    高速向量化格拉姆-施密特正交化 (Fast Vectorized Gram-Schmidt)
    在每个截面预构建标准正交基 Q_d (N_d x k)，每次将新候选向量快速投射消除共线性
    """
    p = panel_df.copy()
    trade_dates = sorted(p["trade_date"].unique())

    # 标准化
    for col in candidate_cols:
        p[col] = p.groupby("trade_date")[col].transform(winsorize)
        p[col] = p.groupby("trade_date")[col].transform(zscore).fillna(0.0)

    # 预按截面提取字典，极大加速计算
    date_data = {}
    for d in trade_dates:
        sub = p[p["trade_date"] == d]
        date_data[d] = {
            "indices": sub.index.values,
            "label": sub[label_col].values,
            "feats": {col: sub[col].values.astype(np.float64) for col in candidate_cols},
            "Q": None  # N x k 标准正交基 (包括常数项)
        }
        # 初始化常数项为正交基第一列
        n_d = len(sub)
        ones_col = np.ones((n_d, 1), dtype=np.float64) / math.sqrt(n_d)
        date_data[d]["Q"] = ones_col

    # 1. 评估所有候选因子的初始单因子表现
    labels_by_date = [date_data[d]["label"] for d in trade_dates]
    factor_stats = []
    for col in candidate_cols:
        preds_by_date = [date_data[d]["feats"][col] for d in trade_dates]
        mic, icir, t_stat, pos_r = compute_ic_series(preds_by_date, labels_by_date)
        factor_stats.append({
            "factor_name": col,
            "mean_ic": mic,
            "abs_ic": abs(mic),
            "icir": icir,
            "abs_icir": abs(icir),
            "t_stat": t_stat,
            "abs_t_stat": abs(t_stat),
            "pos_rate": pos_r
        })

    stat_df = pd.DataFrame(factor_stats).sort_values("abs_icir", ascending=False)
    print("\n[单因子排名 Top 10]")
    print(stat_df.head(10)[["factor_name", "mean_ic", "icir", "t_stat", "pos_rate"]].to_string(index=False))

    # 2. 贪心前向残差正交化
    selected_factors = []
    ortho_columns = []

    # 第一主因子
    first_factor = stat_df.iloc[0]["factor_name"]
    selected_factors.append(first_factor)
    ortho_name = f"ortho_01_{first_factor}"
    ortho_columns.append(ortho_name)
    p[ortho_name] = p[first_factor]

    # 将第一主因子投影并归一化更新到各截面 Q 矩阵
    for d in trade_dates:
        d_info = date_data[d]
        v = d_info["feats"][first_factor]
        Q = d_info["Q"]
        # v_proj = v - Q (Q^T v)
        proj = Q @ (Q.T @ v)
        v_ortho = v - proj
        norm = np.linalg.norm(v_ortho)
        if norm > 1e-6:
            q_new = (v_ortho / norm).reshape(-1, 1)
            d_info["Q"] = np.hstack([Q, q_new])

    remaining_candidates = [c for c in stat_df["factor_name"].tolist() if c != first_factor]
    print(f"\n[正交化迭代] 起始主因子: {first_factor} (|ICIR|={stat_df.iloc[0]['abs_icir']:.2f})")

    step = 2
    while len(selected_factors) < max_features and remaining_candidates:
        best_cand = None
        best_res_icir = -1.0
        best_res_stats = None
        best_res_by_date = None

        for cand in remaining_candidates:
            # 极速残差投影: r = v - Q (Q^T v)
            cand_res_by_date = []
            for d in trade_dates:
                d_info = date_data[d]
                v = d_info["feats"][cand]
                Q = d_info["Q"]
                r = v - Q @ (Q.T @ v)
                cand_res_by_date.append(r)

            mic, icir, t_stat, pos_r = compute_ic_series(cand_res_by_date, labels_by_date)

            if abs(icir) > best_res_icir and abs(t_stat) >= 1.96:
                best_res_icir = abs(icir)
                best_cand = cand
                best_res_stats = (mic, icir, t_stat, pos_r)
                best_res_by_date = cand_res_by_date

        if best_cand is None or best_res_icir < 0.30:
            print(f"[正交化终止] 边际残差有效性已饱和 (当前最优残差 |ICIR|={best_res_icir:.2f} < 0.30), 最终锁定 {len(selected_factors)} 个特征")
            break

        selected_factors.append(best_cand)
        remaining_candidates.remove(best_cand)
        final_col_name = f"ortho_{step:02d}_{best_cand}"
        ortho_columns.append(final_col_name)

        # 赋值并持久化到 DataFrame
        for d, res in zip(trade_dates, best_res_by_date):
            idx = date_data[d]["indices"]
            p.loc[idx, final_col_name] = res

            # 更新当前截面 Q 正交基
            Q = date_data[d]["Q"]
            norm = np.linalg.norm(res)
            if norm > 1e-6:
                q_new = (res / norm).reshape(-1, 1)
                date_data[d]["Q"] = np.hstack([Q, q_new])

        # 标准化该列
        p[final_col_name] = p.groupby("trade_date")[final_col_name].transform(winsorize)
        p[final_col_name] = p.groupby("trade_date")[final_col_name].transform(zscore).fillna(0.0)

        mic, icir, t_stat, pos_r = best_res_stats
        print(f"  Step {step:02d} | 选入特征: {best_cand:<25} | 残差 Rank IC={mic:+.4f} | 残差 ICIR={icir:+.2f} | t-stat={t_stat:+.2f}")
        step += 1

    return p, selected_factors, ortho_columns, stat_df


def main():
    print("=" * 80)
    print(">>> 阶段一：高性价比特征工程深化与格拉姆-施密特正交化筛选...")
    print("=" * 80)

    raw_panel = pd.read_parquet(PANEL_FP)
    panel = generate_candidate_factors(raw_panel)

    non_feat_cols = ["trade_date", "ts_code", "industry", "is_traditional", "fwd_20", "fwd100_maxret", "fwd100_minret"]
    all_candidate_feats = [c for c in panel.columns if c not in non_feat_cols]
    print(f"\n[候选清单] 总计包含 {len(all_candidate_feats)} 个候选特征")

    refined_panel, selected_raw, ortho_cols, stat_df = fast_orthogonalize_factors(
        panel, all_candidate_feats, max_features=14, label_col="fwd_20"
    )

    print("\n" + "=" * 80)
    print(f"成功锁定 {len(selected_raw)} 维最优紧凑正交特征集 (FEATS_ORTHO_{len(selected_raw)}):")
    print("=" * 80)
    for i, (raw, ortho) in enumerate(zip(selected_raw, ortho_cols), 1):
        print(f"  [{i:02d}] 原始因子: {raw:<25} -> 正交列名: {ortho}")

    print(f"\n[保存] 正在保存精选正交面板至: {OUT_REFINED_PANEL}")
    refined_panel.to_parquet(OUT_REFINED_PANEL)

    stat_df.to_csv(OUT_STATS_CSV, index=False, encoding="utf-8-sig")
    print(f"[保存] 因子统计评价表已保存至: {OUT_STATS_CSV}")
    print(">>> 阶段一特征工程与正交化完成！")


if __name__ == "__main__":
    main()

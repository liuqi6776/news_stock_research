# -*- coding: utf-8 -*-
"""精细化多因子特征工程与纯样本内格拉姆-施密特正交拟合器 (InSampleGramSchmidtTransformer)

第五轮独立审查(2026-09-08) P0 整改：
1. 彻底根除全样本未来标签前视泄漏：
   - 将特征筛选与正交基顺序确定严格限制在样本内已到期数据 (label_available_date < oos_start, 2017–2022)；
   - 严禁在全样本 (2023–2026) 上评估边际 IC/ICIR。
2. 数据源统一接入经过审计的 stock_ml_panel_fullmarket_2015.parquet，包含真实 point-in-time label_available_date；
3. 将正交化解耦为可复用的 InSampleGramSchmidtTransformer 类：
   - fit(train_df): 仅在训练期成熟样本上贪心搜索最优正交序列并冻结；
   - transform(any_df): 仅按已冻结的序列执行截面残差投影，零接触任何 label；
4. 导出版本化校验 manifest (orthogonal_factor_manifest.json)。
"""
import os
import sys
import math
import json
import hashlib
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
EXP_DIR = os.path.join(ROOT, "research", "experiments", "exp_ens_t60_tv12")
PANEL_FP = os.path.join(ROOT, "research", "sector_rotation", "stock_ml_panel_fullmarket_2015.parquet")
OUT_REFINED_PANEL = os.path.join(EXP_DIR, "stock_refined_factors_panel.parquet")
OUT_STATS_CSV = os.path.join(EXP_DIR, "refined_factor_statistical_rankings.csv")
OUT_MANIFEST = os.path.join(EXP_DIR, "orthogonal_factor_manifest.json")


def winsorize(s, lo=0.01, hi=0.99):
    if s.notna().sum() < 5:
        return s
    a, b = s.quantile([lo, hi])
    return s.clip(a, b)


def zscore(s):
    return (s - s.mean()) / (s.std(ddof=1) + 1e-12)


def generate_candidate_factors(panel_df):
    """构建精选候选因子集合 (仅基于同期截面与过去数据，无任何未来信息)"""
    p = panel_df.copy()
    print(f"[Panel] 原始面板形状: {p.shape}, 截面数: {p['trade_date'].nunique()}")

    # 1. 经典基准特征
    p["rev_5"] = -p["momentum_5"]
    p["rev_10"] = -p["momentum_10"]
    p["rev_20"] = -p["momentum_20"]
    p["low_vol_anomaly"] = -p["volatility_20"]

    # 2. 流动性冲击与特征比率 (注：原 amihud_proxy_20 系波动率与上涨成交量占比之比，非经典成交金额 Amihud 因子)
    p["vol_posvol_ratio_20"] = p["volatility_20"] / (p["pos_vol_20"] + 1e-4)
    p["amihud_proxy_20"] = p["vol_posvol_ratio_20"]  # 保留兼容历史模型与 manifest 命名
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
    """计算截面 Rank IC 序列与统计量 (Newey-West t-stat, ICIR)"""
    ic_list = []
    for p_arr, l_arr in zip(pred_arr_by_date, label_arr_by_date):
        mask = np.isfinite(p_arr) & np.isfinite(l_arr)
        if mask.sum() < 50:
            continue
        p_sub, l_sub = p_arr[mask], l_arr[mask]
        if np.all(p_sub == p_sub[0]) or np.all(l_sub == l_sub[0]):
            continue
        r_p = stats.rankdata(p_sub)
        r_l = stats.rankdata(l_sub)
        std_p, std_l = np.std(r_p), np.std(r_l)
        if std_p > 1e-8 and std_l > 1e-8:
            cov = np.mean((r_p - np.mean(r_p)) * (r_l - np.mean(r_l)))
            ic = cov / (std_p * std_l)
            if np.isfinite(ic):
                ic_list.append(ic)

    if len(ic_list) < 10:
        return 0.0, 0.0, 0.0, 0.0

    ic_arr = np.array(ic_list)
    mic = float(np.mean(ic_arr))
    std_ic = float(np.std(ic_arr, ddof=1))
    icir = float((mic / (std_ic + 1e-12)) * math.sqrt(12.0))

    n = len(ic_arr)
    gamma0 = np.var(ic_arr, ddof=1)
    gamma1 = np.cov(ic_arr[:-1], ic_arr[1:])[0, 1] if n > 1 else 0.0
    var_nw = (gamma0 + 2.0 * (1.0 - 1.0/3.0) * gamma1) / n
    t_stat = float(mic / (math.sqrt(max(1e-12, var_nw))))
    pos_rate = float((ic_arr > 0).mean())

    return mic, icir, t_stat, pos_rate


class InSampleGramSchmidtTransformer:
    """
    纯样本内格拉姆-施密特正交拟合器 (In-Sample Gram-Schmidt Transformer)

    严格遵循时序因果律：
    1. fit() 仅接收样本内已到期标签 (label_available_date < oos_start)，
       确定因子的最优前向残差正交顺序 [f_1, f_2, ..., f_K]；
    2. 拟合完成后冻结特征序列与超参数；
    3. transform() 对任意面板 (训练集、测试集或实盘推理截面) 纯粹执行截面正交投影，
       完全不接触任何 label，杜绝任何形式的未来信息渗透。
    """
    def __init__(self, max_features=14, min_icir=0.30, min_t_stat=1.96):
        self.max_features = max_features
        self.min_icir = min_icir
        self.min_t_stat = min_t_stat
        self.selected_factors_ = []
        self.ortho_column_names_ = []
        self.selection_steps_ = []
        self.single_factor_stats_ = None
        self.fitted_ = False
        self.manifest_ = {}

    def fit(self, df, candidate_cols, label_col="fwd_20", date_col="trade_date",
            label_avail_col="label_available_date", max_label_date=20230101, min_trade_date=20170101):
        """
        在严格样本内已到期数据上拟合因子正交选择序列。
        """
        print(f"[InSampleGramSchmidt] 启动样本内正交选择拟合: 决策区间 [{min_trade_date}, ...) 且 {label_avail_col} < {max_label_date}")
        
        # 1. 严格过滤样本内成熟切片
        in_mask = (df[date_col] >= min_trade_date) & (df[label_avail_col] < max_label_date) & df[label_col].notna()
        tr_df = df[in_mask].copy()
        trade_dates = sorted(tr_df[date_col].unique())
        print(f"[InSampleGramSchmidt] 样本内有效截面数: {len(trade_dates)} (从 {trade_dates[0]} 到 {trade_dates[-1]}), 样本量: {len(tr_df)}")

        # 2. 截面预标准化 (Winsorize & Z-score)
        for col in candidate_cols:
            tr_df[col] = tr_df.groupby(date_col)[col].transform(winsorize)
            tr_df[col] = tr_df.groupby(date_col)[col].transform(zscore).fillna(0.0)

        # 3. 构建快速截面基字典
        date_data = {}
        for d in trade_dates:
            sub = tr_df[tr_df[date_col] == d]
            n_d = len(sub)
            ones_col = np.ones((n_d, 1), dtype=np.float64) / math.sqrt(n_d)
            date_data[d] = {
                "label": sub[label_col].values,
                "feats": {c: sub[c].values.astype(np.float64) for c in candidate_cols},
                "Q": ones_col
            }

        labels_by_date = [date_data[d]["label"] for d in trade_dates]

        # 4. 计算所有候选因子的初始单因子表现
        factor_stats = []
        for col in candidate_cols:
            preds = [date_data[d]["feats"][col] for d in trade_dates]
            mic, icir, t_stat, pos_r = compute_ic_series(preds, labels_by_date)
            factor_stats.append({
                "factor_name": col,
                "mean_ic": round(mic, 4),
                "abs_ic": round(abs(mic), 4),
                "icir": round(icir, 2),
                "abs_icir": round(abs(icir), 2),
                "t_stat": round(t_stat, 2),
                "abs_t_stat": round(abs(t_stat), 2),
                "pos_rate": round(pos_r * 100, 1)
            })

        self.single_factor_stats_ = pd.DataFrame(factor_stats).sort_values("abs_icir", ascending=False).reset_index(drop=True)
        print("\n[样本内单因子表现 Top 10 (无前瞻)]:")
        print(self.single_factor_stats_.head(10)[["factor_name", "mean_ic", "icir", "t_stat", "pos_rate"]].to_string(index=False))

        # 5. 贪心前向残差正交化
        selected = []
        ortho_names = []
        steps_log = []

        first_f = self.single_factor_stats_.iloc[0]["factor_name"]
        first_icir = self.single_factor_stats_.iloc[0]["icir"]
        first_mic = self.single_factor_stats_.iloc[0]["mean_ic"]
        first_t = self.single_factor_stats_.iloc[0]["t_stat"]
        first_pos = self.single_factor_stats_.iloc[0]["pos_rate"]

        selected.append(first_f)
        ortho_name_1 = f"ortho_01_{first_f}"
        ortho_names.append(ortho_name_1)
        steps_log.append({
            "step": 1,
            "factor_name": first_f,
            "ortho_col": ortho_name_1,
            "residual_ic": first_mic,
            "residual_icir": first_icir,
            "t_stat": first_t,
            "pos_rate": first_pos
        })
        print(f"\n[正交化迭代] 起始主因子: {first_f:<25} | ICIR={first_icir:+.2f} | Rank IC={first_mic:+.4f}")

        # 将第一主因子投影并归一化更新到各截面 Q 矩阵
        for d in trade_dates:
            v = date_data[d]["feats"][first_f]
            Q = date_data[d]["Q"]
            r = v - Q @ (Q.T @ v)
            norm = np.linalg.norm(r)
            if norm > 1e-6:
                date_data[d]["Q"] = np.hstack([Q, (r / norm).reshape(-1, 1)])

        rem_candidates = [c for c in self.single_factor_stats_["factor_name"].tolist() if c != first_f]
        step = 2

        while len(selected) < self.max_features and rem_candidates:
            best_cand = None
            best_icir = -1.0
            best_stats = None

            for cand in rem_candidates:
                res_by_d = []
                for d in trade_dates:
                    v = date_data[d]["feats"][cand]
                    Q = date_data[d]["Q"]
                    r = v - Q @ (Q.T @ v)
                    res_by_d.append(r)

                mic, icir, t_stat, pos_r = compute_ic_series(res_by_d, labels_by_date)
                if abs(icir) > best_icir and abs(t_stat) >= self.min_t_stat:
                    best_icir = abs(icir)
                    best_cand = cand
                    best_stats = (mic, icir, t_stat, pos_r)

            if best_cand is None or best_icir < self.min_icir:
                print(f"[正交化终止] 边际残差有效性已饱和 (当前最优残差 |ICIR|={best_icir:.2f} < {self.min_icir:.2f}), 最终锁定 {len(selected)} 维正交特征")
                break

            selected.append(best_cand)
            rem_candidates.remove(best_cand)
            ortho_col = f"ortho_{step:02d}_{best_cand}"
            ortho_names.append(ortho_col)

            mic, icir, t_stat, pos_r = best_stats
            steps_log.append({
                "step": step,
                "factor_name": best_cand,
                "ortho_col": ortho_col,
                "residual_ic": round(mic, 4),
                "residual_icir": round(icir, 2),
                "t_stat": round(t_stat, 2),
                "pos_rate": round(pos_r * 100, 1)
            })
            print(f"  Step {step:02d} | 选入特征: {best_cand:<25} | 残差 ICIR={icir:+.2f} | 残差 Rank IC={mic:+.4f} | t-stat={t_stat:+.2f}")

            # 更新基矩阵
            for d in trade_dates:
                v = date_data[d]["feats"][best_cand]
                Q = date_data[d]["Q"]
                r = v - Q @ (Q.T @ v)
                norm = np.linalg.norm(r)
                if norm > 1e-6:
                    date_data[d]["Q"] = np.hstack([Q, (r / norm).reshape(-1, 1)])

            step += 1

        self.selected_factors_ = selected
        self.ortho_column_names_ = ortho_names
        self.selection_steps_ = steps_log
        self.fitted_ = True

        self.manifest_ = {
            "version": "2.3-anti-leakage",
            "date_fitted": 20260908,
            "min_trade_date": min_trade_date,
            "max_label_date": max_label_date,
            "num_in_sample_dates": len(trade_dates),
            "num_selected_features": len(self.selected_factors_),
            "selected_factors": self.selected_factors_,
            "ortho_columns": self.ortho_column_names_,
            "selection_steps": self.selection_steps_
        }
        return self

    def transform(self, df, date_col="trade_date"):
        """
        纯截面正交投影：将已拟合锁定的特征序列逐截面投影至正交子空间。
        严禁接触任何 label 字段，对任意样本区间具有完全的因果等价性。
        """
        if not self.fitted_:
            raise RuntimeError("InSampleGramSchmidtTransformer 尚未调用 fit() 拟合！")

        out_df = df.copy()
        dates = sorted(out_df[date_col].unique())

        # 标准化原始候选特征
        for col in self.selected_factors_:
            out_df[col] = out_df.groupby(date_col)[col].transform(winsorize)
            out_df[col] = out_df.groupby(date_col)[col].transform(zscore).fillna(0.0)

        # 预分配正交列
        for col in self.ortho_column_names_:
            out_df[col] = 0.0

        for d in dates:
            d_mask = (out_df[date_col] == d)
            sub = out_df.loc[d_mask]
            n_d = len(sub)
            if n_d < 5:
                continue

            Q = np.ones((n_d, 1), dtype=np.float64) / math.sqrt(n_d)

            for step_idx, factor_name in enumerate(self.selected_factors_):
                ortho_col = self.ortho_column_names_[step_idx]
                v = sub[factor_name].values.astype(np.float64)
                r = v - Q @ (Q.T @ v)
                out_df.loc[d_mask, ortho_col] = r

                norm = np.linalg.norm(r)
                if norm > 1e-6:
                    q_new = (r / norm).reshape(-1, 1)
                    Q = np.hstack([Q, q_new])

        # 对生成的正交列逐截面进行标准化
        for col in self.ortho_column_names_:
            out_df[col] = out_df.groupby(date_col)[col].transform(winsorize)
            out_df[col] = out_df.groupby(date_col)[col].transform(zscore).fillna(0.0)

        return out_df


def main():
    print("=" * 80)
    print(">>> 阶段一：纯样本内零前瞻特征工程与格拉姆-施密特正交拟合器...")
    print("=" * 80)

    raw_panel = pd.read_parquet(PANEL_FP)
    panel = generate_candidate_factors(raw_panel)

    non_feat_cols = [
        "trade_date", "ts_code", "industry", "is_traditional",
        "fwd_20", "label_available_date", "fwd100_maxret", "fwd100_minret"
    ]
    candidate_feats = [c for c in panel.columns if c not in non_feat_cols]
    print(f"\n[候选清单] 总计包含 {len(candidate_feats)} 个候选特征")

    # 严格在 20230101 样本外起点之前且已到期的样本上拟合
    transformer = InSampleGramSchmidtTransformer(max_features=14, min_icir=0.30, min_t_stat=1.96)
    transformer.fit(
        panel,
        candidate_cols=candidate_feats,
        label_col="fwd_20",
        date_col="trade_date",
        label_avail_col="label_available_date",
        max_label_date=20230101,
        min_trade_date=20170101
    )

    print("\n" + "=" * 80)
    print(f"成功锁定 {len(transformer.selected_factors_)} 维样本内正交特征集:")
    print("=" * 80)
    for step in transformer.selection_steps_:
        print(f"  [{step['step']:02d}] 原始因子: {step['factor_name']:<25} -> 正交列: {step['ortho_col']:<30} | 边际 ICIR={step['residual_icir']:+.2f}")

    print("\n[变换] 正在将冻结的正交基应用至全量面板 (零前瞻因果投影)...")
    refined_panel = transformer.transform(panel)

    print(f"[保存] 正在保存精选正交面板至: {OUT_REFINED_PANEL}")
    refined_panel.to_parquet(OUT_REFINED_PANEL)

    transformer.single_factor_stats_.to_csv(OUT_STATS_CSV, index=False, encoding="utf-8-sig")
    print(f"[保存] 因子统计评价表已保存至: {OUT_STATS_CSV}")

    # 保存版本化 Manifest
    with open(OUT_MANIFEST, "w", encoding="utf-8") as f:
        json.dump(transformer.manifest_, f, ensure_ascii=False, indent=2)
    print(f"[保存] 正交特征版本化 Manifest 已保存至: {OUT_MANIFEST}")
    print(">>> 阶段一特征工程与去前视正交化圆满完成！")


if __name__ == "__main__":
    main()

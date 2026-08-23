# -*- coding: utf-8 -*-
"""样本内嵌套特征选择与零泄漏 Purged Walk-Forward 引擎 (Purged Nested Factor Engine)

彻底消除全样本特征选择泄漏:
  1. 嵌套特征筛选: 在每个 Walk-Forward 折 T 中，仅使用 t <= T - 20 的历史数据计算因子 IC/ICIR。
  2. 动态特征池: 每一折根据该折历史截面 ICIR 动态选出 Top-20 因子，绝不提前使用 2023+ 未来收益。
  3. 严格 Purged 隔离: 确保每折训练集与验证集严格结清标签收益。
"""
import os
import sys
import numpy as np
import pandas as pd
import lightgbm as lgb
import warnings
warnings.filterwarnings("ignore")

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
EXP_DIR = os.path.join(ROOT, "research", "experiments", "exp_ens_t60_tv12")
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
if EXP_DIR not in sys.path:
    sys.path.insert(0, EXP_DIR)

from build_expanded_factors import generate_expanded_factors, winsorize, zscore  # noqa: E402


def generate_nested_purged_scores(shared, top_k_factors=20):
    """
    运行严格样本内嵌套特征选择的 Purged Walk-Forward 建模
    :param shared: 全局共享数据字典
    :param top_k_factors: 动态选择的最优因子数量 (默认 20)
    :return: dict: trade_date -> Series of prediction scores
    """
    cal_dates = shared["cal_dates"]
    raw_panel = shared["panel"]
    
    # 1. 扩充全量 50+ 候选因子特征
    panel = generate_expanded_factors(raw_panel)
    label_end_map = {d: cal_dates[min(i + 20, len(cal_dates) - 1)] for i, d in enumerate(cal_dates)}
    panel["label_end_date"] = panel["trade_date"].map(label_end_map)

    # 严格排除所有未来标签、收益率与非因子元数据字段
    excluded_prefixes = ("fwd", "label", "ret_", "target", "open_fwd")
    non_factor_cols = {
        "ts_code", "trade_date", "label_end_date", "fwd_20", "open_fwd_20",
        "ret_20d_raw", "is_traditional", "industry", "industry_l1", "name",
        "fwd100_maxret", "fwd100_minret", "ret_1m"
    }
    candidate_cols = [
        c for c in panel.columns 
        if c not in non_factor_cols 
        and not any(c.startswith(pfx) for pfx in excluded_prefixes)
        and pd.api.types.is_numeric_dtype(panel[c])
    ]

    # 截面标准化处理
    p = panel.copy()
    for c in candidate_cols:
        p[c] = p.groupby("trade_date")[c].transform(lambda s: winsorize(s))
        p[c] = p.groupby("trade_date")[c].transform(zscore).fillna(0.0)

    all_panel_dates = sorted(p["trade_date"].unique())
    score_gbdt_nested = {}
    score_hybrid_nested = {}
    score_enh4 = shared["scores"].get("ENH", {})

    print(f"[+] 正在启动嵌套样本内 Purged Walk-Forward 滚动建模 (候选因子库: {len(candidate_cols)} 维)...")

    for idx, m in enumerate(all_panel_dates):
        if idx < 6:
            continue
        
        # 1. 严格 Purged 训练集: label_end_date < m
        tr_pool = p[p["label_end_date"] < m]
        if len(tr_pool) < 500:
            continue
        assert (tr_pool["label_end_date"] < m).all()

        # 2. 【样本内嵌套特征选择】: 仅在当前历史切片 tr_pool 上计算各因子的截面 IC
        ic_records = []
        for feat in candidate_cols:
            df_sub = tr_pool[["trade_date", feat, "fwd_20"]].dropna()
            if len(df_sub) > 100:
                monthly_ic = df_sub.groupby("trade_date").apply(
                    lambda g: g[feat].corr(g["fwd_20"], method="spearman") if len(g) > 20 else np.nan
                ).dropna()
                if len(monthly_ic) >= 3:
                    mean_ic = monthly_ic.mean()
                    icir = mean_ic / (monthly_ic.std() + 1e-6)
                    ic_records.append({
                        "factor": feat,
                        "abs_ic": abs(mean_ic),
                        "icir": abs(icir)
                    })

        if len(ic_records) < top_k_factors:
            selected_features = candidate_cols[:top_k_factors]
        else:
            df_ic = pd.DataFrame(ic_records).sort_values("icir", ascending=False)
            selected_features = df_ic["factor"].head(top_k_factors).tolist()

        # 3. 划分样本内训练与早停验证集
        tr_months = sorted(tr_pool["trade_date"].unique())
        val_months = tr_months[-2:] if len(tr_months) >= 5 else []
        val_start_d = min(val_months) if val_months else m

        train_mask = (tr_pool["label_end_date"] < val_start_d).values if val_months else np.ones(len(tr_pool), dtype=bool)
        val_mask = tr_pool["trade_date"].isin(val_months).values if val_months else np.zeros(len(tr_pool), dtype=bool)
        om = p[p["trade_date"] == m]

        X_tr = tr_pool[selected_features].values[train_mask]
        y_tr = tr_pool["fwd_20"].values[train_mask]
        X_val = tr_pool[selected_features].values[val_mask]
        y_val = tr_pool["fwd_20"].values[val_mask]

        m_gbdt = lgb.LGBMRegressor(
            n_estimators=300, learning_rate=0.05, num_leaves=7, max_depth=3,
            min_child_samples=80, reg_lambda=2.0, subsample=0.9, random_state=42, verbose=-1
        )
        m_gbdt.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)] if len(val_months) else None,
            callbacks=[lgb.early_stopping(30, verbose=False)] if len(val_months) else None
        )

        s_g = pd.Series(m_gbdt.predict(om[selected_features]), index=om["ts_code"])
        score_gbdt_nested[m] = s_g

        # 4. 融合 ENH4 线性质量与估值底座
        s_enh = score_enh4.get(m, pd.Series(dtype=float))
        df_hyb = pd.DataFrame({"enh": s_enh, "gbdt": s_g}).dropna()
        if len(df_hyb) > 100:
            df_hyb_pct = df_hyb.rank(pct=True)
            score_hybrid_nested[m] = 0.50 * df_hyb_pct["enh"] + 0.50 * df_hyb_pct["gbdt"]
        else:
            score_hybrid_nested[m] = s_g

    print(f"[+] 嵌套特征选择与滚动重训完毕，覆盖 {len(score_hybrid_nested)} 个决策截面！")
    return score_gbdt_nested, score_hybrid_nested

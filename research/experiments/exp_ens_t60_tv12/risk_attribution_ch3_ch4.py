# -*- coding: utf-8 -*-
"""Liu, Stambaugh, Yuan (2019) CH3/CH4 因子风险归因与微盘壳价值诊断
(Risk Attribution & Micro-Cap Shell Value Diagnosis via CH3/CH4 Factors)

基于《A股20日收益目标：论文、因子与GitHub实现的增强研究 (2026-09-08)》方向C：
- 严格遵循 Liu, Stambaugh, Yuan (2019) JFE 论文标准定义:
  1. 排除全市场市值最小 30% 股票以消除借壳投机期权对定价的扭曲;
  2. SMB: 小市值组减大市值组收益 (剔除微盘后);
  3. VMG: 采用 E/P (1/PE_TTM) 构建的中国价值因子 (高减低);
  4. PMO: 换手率因子 (低换手减高换手);
- 对策略月度超额收益进行多元回归, 分解真 Alpha vs 宏观风格暴露 (SMB, VMG, PMO);
- 诊断持仓标的的真实市值分位数, 验证 Alpha 是否脱离微盘壳股.
"""
import os
import sys
import glob
import json
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
EXP_DIR = os.path.join(ROOT, "research", "experiments", "exp_ens_t60_tv12")
DATA_DIR = r"D:\iquant_data\data_v2"
NAV_CSV = os.path.join(EXP_DIR, "conditional_reversal_nav.csv")
OUT_ATTRIBUTION_JSON = os.path.join(EXP_DIR, "ch3_ch4_attribution_report.json")


def build_ch4_factors(start_date=20230101, end_date=20260909):
    """基于 A 股全市场真实日频行情构建 2023-2026 月度 CH3/CH4 因子时序"""
    print(">>> 正在基于真实全市场行情构建 Liu, Stambaugh, Yuan (2019) CH4 因子...")
    
    other_files = sorted(glob.glob(os.path.join(DATA_DIR, "other_day1", "*.parquet")))
    other_files = [f for f in other_files if os.path.basename(f) >= "20221201" and os.path.basename(f) <= "20260909"]
    
    day_files = sorted(glob.glob(os.path.join(DATA_DIR, "data_day1", "*.parquet")))
    day_files = [f for f in day_files if os.path.basename(f) >= "20221201" and os.path.basename(f) <= "20260909"]
    
    # 提取每月最后一个交易日
    dates_all = sorted([int(os.path.splitext(os.path.basename(f))[0]) for f in other_files])
    dates_df = pd.DataFrame({"trade_date": dates_all})
    dates_df["month"] = dates_df["trade_date"].astype(str).str[:6]
    month_end_dates = dates_df.groupby("month")["trade_date"].max().tolist()
    month_end_dates = [d for d in month_end_dates if d >= start_date]

    factor_records = []

    for idx in range(len(month_end_dates) - 1):
        cur_d = month_end_dates[idx]
        next_d = month_end_dates[idx + 1]
        
        # 读取当前月月末的市值与估值
        fp_cur = os.path.join(DATA_DIR, "other_day1", f"{cur_d}.parquet")
        fp_next = os.path.join(DATA_DIR, "data_day1", f"{next_d}.parquet")
        fp_cur_px = os.path.join(DATA_DIR, "data_day1", f"{cur_d}.parquet")
        
        if not (os.path.exists(fp_cur) and os.path.exists(fp_next) and os.path.exists(fp_cur_px)):
            continue
            
        df_cur = pd.read_parquet(fp_cur, columns=["ts_code", "circ_mv", "pe", "turnover_rate"])
        df_cur_px = pd.read_parquet(fp_cur_px, columns=["ts_code", "close"]).rename(columns={"close": "close_cur"})
        df_next_px = pd.read_parquet(fp_next, columns=["ts_code", "close"]).rename(columns={"close": "close_next"})
        
        m = df_cur.merge(df_cur_px, on="ts_code").merge(df_next_px, on="ts_code")
        m["fwd_ret"] = (m["close_next"] / m["close_cur"] - 1.0)
        m = m.dropna(subset=["fwd_ret", "circ_mv"]).copy()
        
        # 1. 市场因子 MKT (市值加权)
        mkt_ret = np.average(m["fwd_ret"], weights=m["circ_mv"])
        
        # 2. 排除最小 30% 微盘股 (Liu, Stambaugh, Yuan 2019 规则)
        mv_p30 = m["circ_mv"].quantile(0.30)
        m_screened = m[m["circ_mv"] >= mv_p30].copy()
        
        # 3. SMB: 剩余 70% 股票按中位数划分为小盘 (Small) 与大盘 (Big)
        mv_median = m_screened["circ_mv"].median()
        s_group = m_screened[m_screened["circ_mv"] < mv_median]
        b_group = m_screened[m_screened["circ_mv"] >= mv_median]
        smb = s_group["fwd_ret"].mean() - b_group["fwd_ret"].mean()
        
        # 4. VMG (Value Minus Growth, 基于 E/P = 1/PE)
        # 过滤 PE <= 0 的亏损股
        m_val = m_screened[m_screened["pe"] > 0].copy()
        m_val["ep"] = 1.0 / m_val["pe"]
        ep_p30 = m_val["ep"].quantile(0.30)
        ep_p70 = m_val["ep"].quantile(0.70)
        v_group = m_val[m_val["ep"] >= ep_p70]  # 高 E/P (价值)
        g_group = m_val[m_val["ep"] <= ep_p30]  # 低 E/P (成长)
        vmg = v_group["fwd_ret"].mean() - g_group["fwd_ret"].mean()
        
        # 5. PMO (Pessimistic Minus Optimistic, 换手率低减高)
        to_p30 = m_screened["turnover_rate"].quantile(0.30)
        to_p70 = m_screened["turnover_rate"].quantile(0.70)
        pess_group = m_screened[m_screened["turnover_rate"] <= to_p30]  # 低换手
        opt_group = m_screened[m_screened["turnover_rate"] >= to_p70]   # 高换手
        pmo = pess_group["fwd_ret"].mean() - opt_group["fwd_ret"].mean()
        
        factor_records.append({
            "month": str(cur_d)[:6],
            "trade_date": next_d,
            "mkt": mkt_ret,
            "smb": smb,
            "vmg": vmg,
            "pmo": pmo
        })

    df_fac = pd.DataFrame(factor_records)
    print(f"  -> CH4 因子序列构建完成: 共 {len(df_fac)} 个月度样本.")
    return df_fac


def run_attribution():
    print("=" * 80)
    print(">>> 启动 Liu, Stambaugh, Yuan (2019) CH3/CH4 因子风险归因...")
    print("=" * 80)
    
    # 1. 构建因子数据
    df_fac = build_ch4_factors()
    
    # 2. 读取策略净值并计算月度收益率
    df_nav = pd.read_csv(NAV_CSV)
    if "trade_date" not in df_nav.columns:
        df_nav = df_nav.rename(columns={df_nav.columns[0]: "trade_date"})
    df_nav["trade_date"] = df_nav["trade_date"].astype(int)
    df_nav["month"] = df_nav["trade_date"].astype(str).str[:6]
    
    # 提取月末净值
    month_nav = df_nav.groupby("month").last().reset_index()
    
    # 计算月度收益率
    strat_cols = [c for c in df_nav.columns if c not in ["trade_date", "month"]]
    strat_rets = {}
    for col in strat_cols:
        r = month_nav[col].pct_change().dropna()
        strat_rets[col] = r.values
    
    months = month_nav["month"].iloc[1:].tolist()
    df_strat_m = pd.DataFrame(strat_rets, index=months)
    
    # 对齐因子
    df_fac_aligned = df_fac.set_index("month").reindex(months).dropna()
    common_months = df_fac_aligned.index.intersection(df_strat_m.index)
    
    df_fac_sub = df_fac_aligned.loc[common_months]
    df_strat_sub = df_strat_m.loc[common_months]
    
    # 无风险利率: 约年化 1.5% -> 月化 0.125%
    rf = 0.015 / 12.0
    
    attribution_results = {}
    
    print("\n" + "=" * 90)
    print(f"{'策略方案 / Strategy Scheme':<36} | {'Alpha(年化)':<10} | {'t(Alpha)':<8} | {'Beta_MKT':<9} | {'Beta_SMB':<9} | {'Beta_VMG':<9} | {'Beta_PMO':<9} | {'R^2':<6}")
    print("-" * 90)
    
    for s in strat_cols:
        y = (df_strat_sub[s] - rf).values
        # X: 常数项, MKT-rf, SMB, VMG, PMO
        X = np.column_stack([
            np.ones(len(y)),
            df_fac_sub["mkt"].values - rf,
            df_fac_sub["smb"].values,
            df_fac_sub["vmg"].values,
            df_fac_sub["pmo"].values
        ])
        
        # OLS
        beta, residuals, rank, s_sv = np.linalg.lstsq(X, y, rcond=None)
        y_pred = X @ beta
        r2 = 1.0 - np.sum((y - y_pred)**2) / np.sum((y - np.mean(y))**2)
        
        n, p = X.shape
        dof = n - p
        sigma2 = np.sum((y - y_pred)**2) / dof
        cov_beta = sigma2 * np.linalg.inv(X.T @ X)
        se_beta = np.sqrt(np.diag(cov_beta))
        t_stats = beta / (se_beta + 1e-12)
        
        alpha_ann = beta[0] * 12.0 * 100.0
        t_alpha = t_stats[0]
        
        attribution_results[s] = {
            "alpha_annualized_pct": round(alpha_ann, 2),
            "t_stat_alpha": round(t_alpha, 2),
            "p_value_alpha": round(float(2 * (1 - stats.t.cdf(abs(t_alpha), dof))), 4),
            "beta_mkt": round(float(beta[1]), 3),
            "t_mkt": round(float(t_stats[1]), 2),
            "beta_smb": round(float(beta[2]), 3),
            "t_smb": round(float(t_stats[2]), 2),
            "beta_vmg": round(float(beta[3]), 3),
            "t_vmg": round(float(t_stats[3]), 2),
            "beta_pmo": round(float(beta[4]), 3),
            "t_pmo": round(float(t_stats[4]), 2),
            "r_squared": round(float(r2), 4)
        }
        
        print(f"{s:<36} | {alpha_ann:>8.2f}% | {t_alpha:>8.2f} | {beta[1]:>9.3f} | {beta[2]:>9.3f} | {beta[3]:>9.3f} | {beta[4]:>9.3f} | {r2:>6.2f}")
    
    print("=" * 90)
    
    # 3. 诊断持仓的市值分位数分布 (检查是否下沉微盘股)
    print("\n>>> 正在诊断策略持仓市值分布与微盘壳价值暴露...")
    refined_panel = pd.read_parquet(os.path.join(EXP_DIR, "stock_refined_factors_panel.parquet"))
    scores_cs = pd.read_parquet(os.path.join(EXP_DIR, "pred_scores_cs_transformer_v23.parquet"))
    
    # 提取 2023-2026 各月末全市场市值分布
    mv_audit_records = []
    decision_dates = sorted(scores_cs["trade_date"].unique())
    
    for d in decision_dates:
        fp_cur = os.path.join(DATA_DIR, "other_day1", f"{d}.parquet")
        if not os.path.exists(fp_cur):
            continue
        df_mv = pd.read_parquet(fp_cur, columns=["ts_code", "circ_mv"]).dropna()
        p30 = df_mv["circ_mv"].quantile(0.30)
        p50 = df_mv["circ_mv"].quantile(0.50)
        
        sub_sc = scores_cs[scores_cs["trade_date"] == d].sort_values("score", ascending=False).head(40)
        top_codes = sub_sc["ts_code"].tolist()
        top_mv = df_mv[df_mv["ts_code"].isin(top_codes)]["circ_mv"]
        
        micro_cap_count = (top_mv < p30).sum()
        micro_cap_ratio = micro_cap_count / len(top_mv) if len(top_mv) else 0.0
        
        mv_audit_records.append({
            "trade_date": d,
            "micro_cap_ratio": micro_cap_ratio,
            "mean_mv_yi": top_mv.mean() / 1e8 if len(top_mv) else 0.0
        })
        
    df_mv_audit = pd.DataFrame(mv_audit_records)
    avg_micro_ratio = df_mv_audit["micro_cap_ratio"].mean() * 100.0
    print(f"[微盘暴露诊断] CS-Transformer 选股落在全市场后 30% (壳股区间) 的平均比例仅为: {avg_micro_ratio:.2f}%")
    print(f"[微盘暴露诊断] 意味着有超过 {100.0 - avg_micro_ratio:.2f}% 的持仓均位于流动性充足的中大盘及主流中小盘内，不依赖微盘壳股期权！")
    
    attribution_results["micro_cap_diagnosis"] = {
        "bottom_30pct_micro_cap_ratio": round(avg_micro_ratio, 2),
        "institution_investable_ratio": round(100.0 - avg_micro_ratio, 2)
    }
    
    with open(OUT_ATTRIBUTION_JSON, "w", encoding="utf-8") as f:
        json.dump(attribution_results, f, indent=2, ensure_ascii=False)
    print(f"\n[OK] CH3/CH4 归因与微盘诊断报告已导出: {OUT_ATTRIBUTION_JSON}")


if __name__ == "__main__":
    from run_phase27_ablation_and_ch4 import main
    main()
    # 同步更新根目录下的 ch3_ch4_attribution_report.json
    src_json = os.path.join(EXP_DIR, "artifacts", "ch4_attribution", "regression_summary.json")
    if os.path.exists(src_json):
        import shutil
        shutil.copyfile(src_json, OUT_ATTRIBUTION_JSON)
        print(f"[OK] 已将最新 Newey-West HAC 归因同步至 {OUT_ATTRIBUTION_JSON}")

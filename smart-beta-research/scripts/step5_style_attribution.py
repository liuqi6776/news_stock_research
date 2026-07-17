"""
step5_style_attribution.py
对 A股多因子策略收益进行风格归因分析。
1. 自建日度风格因子：
   - 市场因子 (R_m)：全市场个股日度等权平均收益率。
   - 市值因子 (SMB)：每日按流通市值 (circ_mv) 排序，做多最小 30% 股票平均收益，做空最大 30% 股票平均收益。
   - 行业超额因子 (R_ind - R_m)：各行业每日个股等权平均收益率减去全市场等权平均收益，消除行业因子的共线性。
2. 线性回归 (OLS)：
   - 仅回归 市场 + 市值： R_strategy = alpha + beta_m * R_m + beta_s * SMB + e
   - 结合 市场 + 市值 + 行业超额： R_strategy = alpha + beta_m * R_m + beta_s * SMB + sum(beta_i * (R_ind_i - R_m)) + e
3. 统计检验与可视化：
   - 计算日度 Alpha、年化 Alpha 及其 t 统计量、p 值和 R-squared。
   - 绘制“纯 Alpha” (Residual Return) 累计收益率曲线。
4. 分年度业绩拆分。
"""
import os
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(PROJECT_DIR, 'data')
RESULTS_DIR = os.path.join(PROJECT_DIR, 'results')

NAV_FILE = os.path.join(RESULTS_DIR, 'portfolio_comparison_nav.csv')
FEATURES_FILE = os.path.join(DATA_DIR, 'features_longterm.parquet')
REPORT_FILE = os.path.join(RESULTS_DIR, 'style_attribution_report.txt')
PLOT_FILE = os.path.join(RESULTS_DIR, 'style_attribution_residual.png')

def run_attribution():
    print(">>> Starting Style Attribution Analysis (Fixing Multicollinearity)...", flush=True)
    
    # 1. 加载策略每日净值
    if not os.path.exists(NAV_FILE):
        raise FileNotFoundError(f"Strategy NAV file not found at {NAV_FILE}. Please run step4 first.")
    
    df_nav = pd.read_csv(NAV_FILE, index_col=0)
    df_nav.index = pd.to_datetime(df_nav.index)
    df_nav = df_nav.sort_index()
    
    # 计算策略日收益率 (以 Pure Multi-Factor 策略为准)
    df_nav['Strategy_Ret'] = df_nav['Strategy_Pure'].pct_change().fillna(0.0)
    df_nav['trade_date_str'] = df_nav.index.strftime('%Y%m%d')
    
    # 获取回测日期范围
    trade_dates = df_nav['trade_date_str'].tolist()
    start_date = trade_dates[0]
    end_date = trade_dates[-1]
    
    # 2. 从 parquet 中加载特征数据构建因子
    print(f"Loading stock features from {FEATURES_FILE} for range {start_date} to {end_date}...", flush=True)
    df_feat = pd.read_parquet(FEATURES_FILE, columns=['trade_date', 'ts_code', 'pct_chg', 'circ_mv', 'industry'])
    df_feat['trade_date'] = df_feat['trade_date'].astype(str)
    df_feat = df_feat[(df_feat['trade_date'] >= start_date) & (df_feat['trade_date'] <= end_date)].copy()
    
    df_feat['pct_chg'] = df_feat['pct_chg'].fillna(0.0)
    df_feat['circ_mv'] = pd.to_numeric(df_feat['circ_mv'], errors='coerce').fillna(0.0)
    
    # A. 市场等权因子 (R_m)
    print("Calculating daily Market Equal-Weight factor...", flush=True)
    df_mkt = df_feat.groupby('trade_date')['pct_chg'].mean().reset_index().rename(columns={'pct_chg': 'R_m'})
    
    # B. 市值因子 (SMB - Small Minus Big)
    print("Calculating daily SMB (Size) factor...", flush=True)
    def calc_smb(group):
        group = group[group['circ_mv'] > 0]
        n = len(group)
        if n < 10:
            return 0.0
        sorted_g = group.sort_values('circ_mv')
        n_cutoff = int(n * 0.3)
        r_small = sorted_g.iloc[:n_cutoff]['pct_chg'].mean()
        r_big = sorted_g.iloc[-n_cutoff:]['pct_chg'].mean()
        return r_small - r_big

    df_smb = df_feat.groupby('trade_date').apply(calc_smb).reset_index().rename(columns={0: 'SMB'})
    
    # C. 行业平均因子 (原始收益)
    print("Calculating daily Industry raw returns...", flush=True)
    df_ind = df_feat.groupby(['trade_date', 'industry'])['pct_chg'].mean().unstack(fill_value=0.0).reset_index()
    
    # D. 加载中证1000指数日收益率作为比对基准
    index_file = os.path.join(DATA_DIR, 'index_regime.csv')
    if os.path.exists(index_file):
        print("Loading CSI 1000 index returns...", flush=True)
        idx_df = pd.read_csv(index_file)
        idx_df['trade_date'] = idx_df['trade_date'].astype(str)
        idx_df = idx_df[idx_df['trade_date'].isin(trade_dates)].sort_values('trade_date').reset_index(drop=True)
        idx_df['CSI1000_Ret'] = idx_df['close'].pct_change().fillna(0.0)
        idx_map = idx_df.set_index('trade_date')['CSI1000_Ret'].to_dict()
    else:
        idx_map = {}
        
    # 3. 合并自变量与因变量
    df_reg = df_nav[['trade_date_str', 'Strategy_Ret']].merge(df_mkt, left_on='trade_date_str', right_on='trade_date', how='inner')
    df_reg = df_reg.merge(df_smb, on='trade_date', how='inner')
    df_reg = df_reg.merge(df_ind, on='trade_date', how='inner')
    
    df_reg['CSI1000_Ret'] = df_reg['trade_date'].map(idx_map).fillna(0.0)
    
    # 行业列
    ind_cols = [col for col in df_ind.columns if col not in ['trade_date', 'Unknown']]
    
    # 【核心修正】：将行业因子的自变量改为行业相对市场的超额收益率 (R_ind - R_m)
    # 从而完全解决 R_m 与所有行业因子相加等于全市场组合造成的 perfect multicollinearity 问题
    print("Converting industry raw returns to industry excess returns (R_ind - R_m) to solve multicollinearity...", flush=True)
    for col in ind_cols:
        df_reg[col] = df_reg[col] - df_reg['R_m']
        
    # 4. 执行 OLS 回归
    y = df_reg['Strategy_Ret']
    
    # A. Model 1: 仅考虑 市场(R_m) + 市值(SMB)
    X1 = df_reg[['R_m', 'SMB']]
    X1 = sm.add_constant(X1)
    model1 = sm.OLS(y, X1).fit()
    
    # B. Model 2: 考虑 市场(R_m) + 市值(SMB) + 行业超额因子 (控制了多重共线性)
    X2 = df_reg[['R_m', 'SMB'] + ind_cols]
    X2 = sm.add_constant(X2)
    model2 = sm.OLS(y, X2).fit()
    
    # 5. 输出报告
    report = []
    report.append("==========================================================================")
    report.append("                       Strategy Style Attribution Report                  ")
    report.append("==========================================================================")
    report.append(f"Backtest Period: {start_date} to {end_date} ({len(df_reg)} trading days)")
    report.append("==========================================================================")
    
    report.append("\n[Model 1: Market & Size Factor Regression (Standard)]")
    report.append(f"Formula: R_strategy = alpha + beta_m * R_m + beta_s * SMB + e")
    report.append("--------------------------------------------------------------------------")
    
    alpha_m1 = model1.params['const']
    alpha_ann_m1 = alpha_m1 * 252
    t_alpha_m1 = model1.tvalues['const']
    p_alpha_m1 = model1.pvalues['const']
    beta_m_m1 = model1.params['R_m']
    beta_s_m1 = model1.params['SMB']
    r2_m1 = model1.rsquared
    
    report.append(f"Daily Alpha (Intercept): {alpha_m1:+.6f}")
    report.append(f"Annualized Alpha:        {alpha_ann_m1:+.2%}")
    report.append(f"t-statistic of Alpha:    {t_alpha_m1:.4f} " + ("(SIGNIFICANT)" if abs(t_alpha_m1) >= 1.96 else "(NOT SIGNIFICANT)"))
    report.append(f"p-value of Alpha:        {p_alpha_m1:.6f}")
    report.append(f"Beta Market (beta_m):    {beta_m_m1:.4f} (t-stat: {model1.tvalues['R_m']:.2f})")
    report.append(f"Beta Size (beta_s):      {beta_s_m1:.4f} (t-stat: {model1.tvalues['SMB']:.2f})")
    report.append(f"R-squared:               {r2_m1:.4f} (Style explains {r2_m1:.2%} of variance)")
    
    report.append("\n==========================================================================")
    report.append("[Model 2: Market, Size & Orthogonalized Industry Excess Regression]")
    report.append(f"Formula: R_strategy = alpha + beta_m * R_m + beta_s * SMB + sum(beta_i * (R_ind_i - R_m)) + e")
    report.append("--------------------------------------------------------------------------")
    
    alpha_m2 = model2.params['const']
    alpha_ann_m2 = alpha_m2 * 252
    t_alpha_m2 = model2.tvalues['const']
    p_alpha_m2 = model2.pvalues['const']
    beta_m_m2 = model2.params['R_m']
    beta_s_m2 = model2.params['SMB']
    r2_m2 = model2.rsquared
    
    report.append(f"Daily Alpha (Intercept): {alpha_m2:+.6f}")
    report.append(f"Annualized Alpha:        {alpha_ann_m2:+.2%}")
    report.append(f"t-statistic of Alpha:    {t_alpha_m2:.4f} " + ("(SIGNIFICANT)" if abs(t_alpha_m2) >= 1.96 else "(NOT SIGNIFICANT)"))
    report.append(f"p-value of Alpha:        {p_alpha_m2:.6f}")
    report.append(f"Beta Market (beta_m):    {beta_m_m2:.4f} (t-stat: {model2.tvalues['R_m']:.2f})")
    report.append(f"Beta Size (beta_s):      {beta_s_m2:.4f} (t-stat: {model2.tvalues['SMB']:.2f})")
    report.append(f"R-squared:               {r2_m2:.4f} (Style explains {r2_m2:.2%} of variance)")
    
    # 提取显著的行业主动偏离暴露 (p < 0.05)
    report.append("\nSignificant Active Industry Exposures (p < 0.05):")
    sig_inds = []
    for col in ind_cols:
        p_val = model2.pvalues[col]
        beta_val = model2.params[col]
        t_val = model2.tvalues[col]
        if p_val < 0.05:
            sig_inds.append(f"  - {col}: beta = {beta_val:+.4f} (t-stat: {t_val:.2f}, p-val: {p_val:.4f})")
    if sig_inds:
        report.extend(sig_inds)
    else:
        report.append("  None")
        
    # 6. 分年度绩效拆分 (Year-by-Year Performance Breakdown)
    report.append("\n==========================================================================")
    report.append("                   Year-by-Year Strategy Performance                      ")
    report.append("==========================================================================")
    report.append(f"{'Year':<6} | {'Strategy':<10} | {'CSI 1000':<10} | {'Equal-Weight':<12} | {'Excess vs CSI':<13} | {'Excess vs EW':<12}")
    report.append("--------------------------------------------------------------------------")
    
    df_reg['year'] = df_reg['trade_date_str'].str[:4]
    years = sorted(df_reg['year'].unique())
    for y in years:
        df_year = df_reg[df_reg['year'] == y]
        strat_ret_y = (1 + df_year['Strategy_Ret']).prod() - 1
        csi1000_ret_y = (1 + df_year['CSI1000_Ret']).prod() - 1
        ew_ret_y = (1 + df_year['R_m']).prod() - 1
        
        excess_csi = strat_ret_y - csi1000_ret_y
        excess_ew = strat_ret_y - ew_ret_y
        report.append(f"{y:<6} | {strat_ret_y:<10.2%} | {csi1000_ret_y:<10.2%} | {ew_ret_y:<12.2%} | {excess_csi:<+13.2%} | {excess_ew:<+12.2%}")
        
    report.append("==========================================================================")
    
    report_text = "\n".join(report)
    print(report_text, flush=True)
    
    with open(REPORT_FILE, 'w', encoding='utf-8') as f:
        f.write(report_text)
    print(f"Saved attribution report to {REPORT_FILE}")
    
    # 7. 计算纯 Alpha 曲线 (Cumulative Residual Return)
    # 使用 Model 2 的残差来绘制纯净超额收益
    df_reg['Residual'] = model2.resid
    df_reg['Alpha_NAV'] = (1 + df_reg['Residual']).cumprod()
    
    # 绘图
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    fig, ax = plt.subplots(figsize=(12, 6))
    
    dates = pd.to_datetime(df_reg['trade_date_str'])
    
    ax.plot(dates, df_reg['Alpha_NAV'], label='Cumulative Residual (Pure Alpha)', color='#e65100', linewidth=2.5)
    df_nav_aligned = df_nav.loc[dates]
    ax.plot(dates, df_nav_aligned['Strategy_Pure'] / INITIAL_CAPITAL, label='Strategy (Pure Multi-Factor NAV)', color='#1565c0', linewidth=1.5, alpha=0.6)
    
    ax.set_title('Cumulative Residual Return (Clean Alpha after Market, Size & Industry Frictions)', fontsize=13, fontweight='bold', pad=15)
    ax.set_xlabel('Date', fontsize=11)
    ax.set_ylabel('Normalized Value', fontsize=11)
    ax.legend(loc='upper left', frameon=True, facecolor='white', framealpha=0.9, edgecolor='lightgray')
    
    plt.tight_layout()
    plt.savefig(PLOT_FILE, dpi=300)
    print(f"Saved cumulative residual plot to {PLOT_FILE}")
    plt.close()

if __name__ == '__main__':
    from step4_portfolio_backtest import INITIAL_CAPITAL
    run_attribution()

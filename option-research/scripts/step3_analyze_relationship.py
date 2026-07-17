"""
step3_analyze_relationship.py
统计检验“最大痛点阻力/引力”假说（严格版）：
1. 消除前视偏差：基于已计算的 T-1 持仓量 (oi_lag1) 痛点价格。
2. 控制反转效应：引入自变量 Ret_{t-5, t} (标的过去 5 交易日收益率) 作为控制变量，运行多元回归：
   R_{t->Expiry} = a + b * Dev_t + c * Ret_{t-5, t} + e
   如果控制反转后，Dev_t 的系数 b 依然显著为负，说明引力效应独立存在。
3. Placebo 检验：将痛点替换为“离现价最近的行权价”，检验其是否同样显著。若 Placebo 显著，则原效应为网格统计幻觉。
4. 样本拆分（盲测）：将样本划分为 IS (2022-2024) 和 OOS (2025-2026) 进行对比。
"""
import os
import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

# 目录配置
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(BASE_DIR, 'data')
PLOTS_DIR = os.path.join(BASE_DIR, 'plots')
os.makedirs(PLOTS_DIR, exist_ok=True)

INPUT_FILE = os.path.join(DATA_DIR, 'max_pain_history.csv')
RESULTS_CSV = os.path.join(DATA_DIR, 'regression_results.csv')

def run_numpy_ols(X_vars, y_var):
    """
    使用 NumPy 进行多元 OLS 回归，支持截距项，返回系数、标准差、t统计量、p值和R2。
    X_vars: list of np.ndarray
    y_var: np.ndarray
    """
    N = len(y_var)
    if N < 5:
        return None
        
    X = np.column_stack([np.ones(N)] + X_vars)
    K = X.shape[1]
    
    XTX = np.dot(X.T, X)
    try:
        XTX_inv = np.linalg.inv(XTX)
    except np.linalg.LinAlgError:
        return None
        
    beta = np.dot(XTX_inv, np.dot(X.T, y_var))
    residuals = y_var - np.dot(X, beta)
    ssr = np.dot(residuals.T, residuals)
    sigma2 = ssr / (N - K)
    
    cov_beta = sigma2 * XTX_inv
    # 防止由于极小负数或零导致平方根溢出
    diag_cov = np.diag(cov_beta).copy()
    diag_cov[diag_cov < 0] = 0.0
    se = np.sqrt(diag_cov)
    
    # 避免除以 0
    t_stats = np.where(se != 0, beta / se, 0.0)
    
    # p-values
    p_values = 2 * (1 - stats.norm.cdf(np.abs(t_stats)))
    
    # R2
    y_mean = np.mean(y_var)
    sst = np.sum((y_var - y_mean) ** 2)
    r_squared = 1.0 - (ssr / sst) if sst != 0 else 0.0
    
    return {
        'coefficients': beta,
        'se': se,
        't_stats': t_stats,
        'p_values': p_values,
        'r_squared': r_squared,
        'n_obs': N
    }

def main():
    print(">>> Starting Strict Regression & Placebo Analysis...", flush=True)
    if not os.path.exists(INPUT_FILE):
        raise FileNotFoundError(f"Input file {INPUT_FILE} not found. Please run step2 first.")
        
    df = pd.read_csv(INPUT_FILE)
    df['trade_date'] = df['trade_date'].astype(str)
    df['expiry_date'] = df['expiry_date'].astype(str)
    
    # 提取标的价格映射表
    price_lookup = df[['trade_date', 'underlying_code', 'underlying_close']].drop_duplicates()
    price_map = price_lookup.set_index(['trade_date', 'underlying_code'])['underlying_close'].to_dict()
    
    # 获取到期日的 ETF 价格
    df['underlying_close_at_expiry'] = df.apply(
        lambda r: price_map.get((r['expiry_date'], r['underlying_code']), np.nan), axis=1
    )
    
    df_clean = df.dropna(subset=['underlying_close_at_expiry', 'days_to_expiry']).copy()
    df_clean = df_clean[df_clean['days_to_expiry'] >= 0]
    
    # 按照标的分组计算过去 5 交易日的收益率作为控制变量 Ret_{t-5, t}
    df_clean = df_clean.sort_values(['underlying_code', 'trade_date']).reset_index(drop=True)
    df_clean['ret_5d'] = df_clean.groupby('underlying_code')['underlying_close'].pct_change(5)
    
    # 计算偏差和未来到期收益率
    # Max Pain 真实偏差
    df_clean['deviation'] = (df_clean['underlying_close'] - df_clean['max_pain_price']) / df_clean['underlying_close']
    # Placebo (最近邻行权价) 偏差
    df_clean['deviation_placebo'] = (df_clean['underlying_close'] - df_clean['placebo_price']) / df_clean['underlying_close']
    
    # 到期收益率
    df_clean['return_to_expiry'] = (df_clean['underlying_close_at_expiry'] - df_clean['underlying_close']) / df_clean['underlying_close']
    
    # 剔除由于计算 5 日收益导致前几行为 NaN 的记录
    df_clean = df_clean.dropna(subset=['ret_5d', 'deviation', 'deviation_placebo', 'return_to_expiry'])
    
    # 保存特征文件
    feature_file = os.path.join(DATA_DIR, 'max_pain_features.csv')
    df_clean.to_csv(feature_file, index=False)
    print(f"Saved cleaned features with 5d returns and placebo prices to {feature_file}")
    
    # ==================== 1. 价格收敛性重新统计 ====================
    print("\n--- 1. Convergence Analysis (Volatility of Deviation vs Trading DTE) ---")
    bins = [0, 2, 5, 10, 15, 20, 30, 100]
    labels = ['0-2 days', '3-5 days', '6-10 days', '11-15 days', '16-20 days', '21-30 days', '30+ days']
    df_clean['dte_bin'] = pd.cut(df_clean['days_to_expiry'], bins=bins, labels=labels, include_lowest=True)
    
    convergence_stats = []
    for und_code in ['510050.SH', '510300.SH']:
        print(f"\nAsset: {und_code}")
        df_sub = df_clean[df_clean['underlying_code'] == und_code]
        grouped = df_sub.groupby('dte_bin', observed=False)['deviation'].agg(['count', 'std', 'var'])
        print(grouped)
        for idx, row in grouped.iterrows():
            convergence_stats.append({
                'underlying_code': und_code,
                'dte_bin': idx,
                'count': row['count'],
                'std_dev': row['std'],
                'var_dev': row['var']
            })
            
    # ==================== 2. 回归预测统计 (全样本、样本内、样本外) ====================
    print("\n--- 2. Controlled Regressions & Placebo Tests ---")
    
    periods = {
        'Full Sample (2022-2026)': df_clean,
        'In-Sample (2022-2024)': df_clean[df_clean['trade_date'] < '20250101'],
        'Out-of-Sample (2025-2026)': df_clean[df_clean['trade_date'] >= '20250101']
    }
    
    dte_filters = [
        ('All DTE', lambda dte: True),
        ('DTE <= 10', lambda dte: dte <= 10),
        ('DTE <= 5', lambda dte: dte <= 5),
        ('DTE <= 3', lambda dte: dte <= 3)
    ]
    
    regression_records = []
    
    for period_name, df_period in periods.items():
        print(f"\n==================== Period: {period_name} ====================")
        
        for und_code in ['510050.SH', '510300.SH']:
            print(f"\n--- Asset: {und_code} ---")
            df_sub = df_period[df_period['underlying_code'] == und_code]
            
            for dte_name, filter_fn in dte_filters:
                df_reg = df_sub[df_sub['days_to_expiry'].apply(filter_fn)]
                if len(df_reg) < 10:
                    continue
                
                y = df_reg['return_to_expiry'].values
                x_dev = df_reg['deviation'].values
                x_ret5d = df_reg['ret_5d'].values
                x_placebo = df_reg['deviation_placebo'].values
                
                # Model A: Max Pain Single (未控制反转)
                m_single = run_numpy_ols([x_dev], y)
                # Model B: Max Pain Controlled (已控制反转)
                m_control = run_numpy_ols([x_dev, x_ret5d], y)
                # Model C: Placebo Controlled (最近邻行权价 + 控制反转)
                m_placebo = run_numpy_ols([x_placebo, x_ret5d], y)
                
                if m_single is None or m_control is None or m_placebo is None:
                    continue
                
                # 记录结果
                regression_records.append({
                    'period': period_name,
                    'underlying_code': und_code,
                    'window': dte_name,
                    'n_obs': len(df_reg),
                    'single_b': m_single['coefficients'][1],
                    'single_t': m_single['t_stats'][1],
                    'single_p': m_single['p_values'][1],
                    'single_r2': m_single['r_squared'],
                    'control_b_dev': m_control['coefficients'][1],
                    'control_t_dev': m_control['t_stats'][1],
                    'control_p_dev': m_control['p_values'][1],
                    'control_b_ret5d': m_control['coefficients'][2],
                    'control_t_ret5d': m_control['t_stats'][2],
                    'control_r2': m_control['r_squared'],
                    'placebo_b': m_placebo['coefficients'][1],
                    'placebo_t': m_placebo['t_stats'][1],
                    'placebo_p': m_placebo['p_values'][1],
                    'placebo_r2': m_placebo['r_squared']
                })
                
                # 打印对比
                print(f"  Window: {dte_name:<10} | Obs: {len(df_reg):<4}")
                print(f"    [MaxPain Single]     Slope: {m_single['coefficients'][1]:.4f} | t-stat: {m_single['t_stats'][1]:.2f} | R2: {m_single['r_squared']:.4f}")
                print(f"    [MaxPain Controlled] Dev t-stat: {m_control['t_stats'][1]:.2f} | Ret5d t-stat: {m_control['t_stats'][2]:.2f} | R2: {m_control['r_squared']:.4f}")
                print(f"    [Placebo Controlled] Placebo t: {m_placebo['t_stats'][1]:.2f} | Ret5d t-stat: {m_placebo['t_stats'][2]:.2f} | R2: {m_placebo['r_squared']:.4f}")
                print("-" * 80)
                
    df_results = pd.DataFrame(regression_records)
    df_results.to_csv(RESULTS_CSV, index=False)
    print(f"\nSaved strict regression results to {RESULTS_CSV}")
    
    # ==================== 3. 绘制收敛性与控制后的图表 ====================
    print("\n>>> Generating Analysis Plots...", flush=True)
    
    # Plot 1: 收敛图（标准差 vs Trading DTE）
    fig, ax = plt.subplots(figsize=(10, 6))
    df_conv = pd.DataFrame(convergence_stats)
    colors = {'510050.SH': '#1f77b4', '510300.SH': '#ff7f0e'}
    for und_code, group in df_conv.groupby('underlying_code'):
        group_sorted = group.iloc[::-1]
        ax.plot(group_sorted['dte_bin'], group_sorted['std_dev'] * 100, marker='o', linewidth=2, label=f"{und_code} Deviation Std (%)", color=colors[und_code])
    ax.set_title("Robust Price Convergence to Max Pain (Trading Days to Expiry)", fontsize=14, fontweight='bold', pad=15)
    ax.set_ylabel("Standard Deviation of Price Deviation (%)", fontsize=12)
    ax.set_xlabel("Trading Days to Expiry (Trading DTE) Bins", fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend(fontsize=11)
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'max_pain_convergence.png'), dpi=300)
    plt.close()
    
    # Plot 2: 预测图：展示回归系数 t 值的演变（随着 DTE 临近）
    # 我们绘制 Full Sample 下，Max Pain vs Placebo 的 t 统计量演变
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), sharey=True)
    windows = ['All DTE', 'DTE <= 10', 'DTE <= 5', 'DTE <= 3']
    x_ticks = np.arange(len(windows))
    
    for i, und_code in enumerate(['510050.SH', '510300.SH']):
        ax = axes[i]
        df_sub_res = df_results[(df_results['period'] == 'Full Sample (2022-2026)') & (df_results['underlying_code'] == und_code)]
        
        # 匹配对应窗口的值
        t_single = [df_sub_res[df_sub_res['window'] == w]['single_t'].values[0] for w in windows]
        t_controlled = [df_sub_res[df_sub_res['window'] == w]['control_t_dev'].values[0] for w in windows]
        t_placebo = [df_sub_res[df_sub_res['window'] == w]['placebo_t'].values[0] for w in windows]
        
        ax.plot(x_ticks, t_single, marker='o', color='gray', linestyle=':', label='Max Pain (No Control)')
        ax.plot(x_ticks, t_controlled, marker='s', color=colors[und_code], linewidth=2.5, label='Max Pain (Controlled)')
        ax.plot(x_ticks, t_placebo, marker='^', color='red', linestyle='--', label='Placebo (Nearest Strike)')
        
        # 显著性分界线 t = -1.96
        ax.axhline(-1.96, color='red', linestyle='-.', alpha=0.6, label='Significance Level (t = -1.96)')
        ax.axhline(0, color='black', linewidth=0.8)
        
        ax.set_title(f"{und_code} Regression t-stat Comparison", fontsize=13, fontweight='bold')
        ax.set_xticks(x_ticks)
        ax.set_xticklabels(windows)
        ax.set_xlabel("Trading Window (DTE)", fontsize=11)
        if i == 0:
            ax.set_ylabel("Coefficient t-statistic", fontsize=11)
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.legend(fontsize=10)
        
    plt.suptitle("Robust Hypothesis Verification: Max Pain vs Placebo (Full Sample)", fontsize=15, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'max_pain_predictability.png'), dpi=300)
    plt.close()
    
    print("\n>>> Robust Verification Complete! Saved plots to plots/ directory.")

if __name__ == '__main__':
    main()

"""
run_incremental_test.py
增量因子评估与风格归因对比主脚本。
1. 分别训练并预测两种特征配置（202409 至 202603）：
   - Config A (Baseline): 包含原有的所有特征 (包括新闻、舆情等)，但不包含 Vibe Alpha 因子。
   - Config B (Baseline + Vibe): 在 A 的基础上加入经过预处理的 Vibe Alpha 因子。
2. 运行高回真度组合回测，输出每日净值序列。
3. 对两种策略在相同的评估期（2024-09-01 至 2026-03-11）进行 Model 2 风格归因线性回归。
4. 比较它们的年化 Alpha、t 统计量、R-squared、总收益和夏普比率。
5. 生成详细对比报告。
"""
import os
import sys
import time
import pandas as pd
import numpy as np
import statsmodels.api as sm

# 确保脚本目录在 path 中以方便导入
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(PROJECT_DIR, 'data')
PRED_DIR = os.path.join(PROJECT_DIR, 'predictions')
RESULTS_DIR = os.path.join(PROJECT_DIR, 'results')
os.makedirs(PRED_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

sys.path.append(SCRIPT_DIR)

from step3_train_ranking_model import train_and_predict
from step4_portfolio_backtest import run_backtest, INITIAL_CAPITAL

# 文件定义
PRED_FILE_A = os.path.join(PRED_DIR, 'predictions_config_A.parquet')
PRED_FILE_B = os.path.join(PRED_DIR, 'predictions_config_B.parquet')

RES_DIR_A = os.path.join(RESULTS_DIR, 'config_A')
RES_DIR_B = os.path.join(RESULTS_DIR, 'config_B')

REPORT_FILE = os.path.join(RESULTS_DIR, 'incremental_test_report.txt')
FEATURES_FILE = os.path.join(DATA_DIR, 'features_longterm.parquet')

def run_regression_for_config(nav_file, start_date, end_date, df_mkt, df_smb, df_ind, ind_cols):
    """
    对指定配置的净值曲线进行 Model 2 风格归因回归
    """
    df_nav = pd.read_csv(nav_file, index_col=0)
    df_nav.index = pd.to_datetime(df_nav.index)
    df_nav = df_nav.sort_index()
    
    # 过滤时间段
    df_nav = df_nav[(df_nav.index >= start_date) & (df_nav.index <= end_date)].copy()
    
    # 计算日度收益率
    df_nav['Strategy_Ret'] = df_nav['Strategy_Pure'].pct_change().fillna(0.0)
    df_nav['trade_date_str'] = df_nav.index.strftime('%Y%m%d')
    
    # 合并因子自变量
    df_reg = df_nav[['trade_date_str', 'Strategy_Ret']].merge(df_mkt, left_on='trade_date_str', right_on='trade_date', how='inner')
    df_reg = df_reg.merge(df_smb, on='trade_date', how='inner')
    df_reg = df_reg.merge(df_ind, on='trade_date', how='inner')
    
    # 【核心】：转换为行业超额收益率，解决多重共线性
    for col in ind_cols:
        df_reg[col] = df_reg[col] - df_reg['R_m']
        
    y = df_reg['Strategy_Ret']
    X = df_reg[['R_m', 'SMB'] + ind_cols]
    X = sm.add_constant(X)
    
    model = sm.OLS(y, X).fit()
    
    # 计算表现指标
    years = len(df_nav) / 252.0
    final_val = df_nav['Strategy_Pure'].iloc[-1]
    init_val = df_nav['Strategy_Pure'].iloc[0]
    total_ret = final_val / init_val - 1
    
    daily_rets = df_nav['Strategy_Pure'].pct_change().dropna()
    ann_ret = daily_rets.mean() * 252
    ann_vol = daily_rets.std() * np.sqrt(252)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0
    max_dd = (df_nav['Strategy_Pure'] / df_nav['Strategy_Pure'].cummax() - 1).min()
    
    return {
        'total_return': total_ret,
        'sharpe': sharpe,
        'max_drawdown': max_dd,
        'alpha_ann': model.params['const'] * 252,
        't_stat': model.tvalues['const'],
        'p_value': model.pvalues['const'],
        'beta_m': model.params['R_m'],
        'beta_s': model.params['SMB'],
        'r2': model.rsquared
    }

def main():
    print("==========================================================================")
    print("                 Starting Incremental Factor Evaluation                   ")
    print("==========================================================================", flush=True)
    
    # 1. 运行 walk-forward 训练并输出预测数据 (评估期: 202409 - 202603)
    # 起始预测月份定为 202409 (使短样本因子均包含有效数据)
    print("\n>>> Step 1: Running monthly rolling Walk-Forward training for Configs...")
    train_and_predict(feature_config='A', output_file=PRED_FILE_A, start_month='202409')
    train_and_predict(feature_config='B', output_file=PRED_FILE_B, start_month='202409')
    
    # 2. 运行回测系统，输出每日 NAV 数据
    print("\n>>> Step 2: Running Portfolio Backtests for Configs...")
    run_backtest(pred_file=PRED_FILE_A, results_dir=RES_DIR_A, save_plot=False)
    run_backtest(pred_file=PRED_FILE_B, results_dir=RES_DIR_B, save_plot=False)
    
    # 3. 准备回归自变量
    print("\n>>> Step 3: Extracting daily control factors (R_m, SMB, Industries)...")
    start_date = '2024-09-02'
    end_date = '2026-03-11'
    start_date_str = '20240902'
    end_date_str = '20260311'
    
    df_feat = pd.read_parquet(FEATURES_FILE, columns=['trade_date', 'pct_chg', 'circ_mv', 'industry'])
    df_feat['trade_date'] = df_feat['trade_date'].astype(str)
    df_feat = df_feat[(df_feat['trade_date'] >= start_date_str) & (df_feat['trade_date'] <= end_date_str)].copy()
    
    df_feat['pct_chg'] = df_feat['pct_chg'].fillna(0.0)
    df_feat['circ_mv'] = pd.to_numeric(df_feat['circ_mv'], errors='coerce').fillna(0.0)
    
    # A. 市场因子 (R_m)
    df_mkt = df_feat.groupby('trade_date')['pct_chg'].mean().reset_index().rename(columns={'pct_chg': 'R_m'})
    
    # B. 市值因子 (SMB)
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
    
    # C. 行业平均因子
    df_ind = df_feat.groupby(['trade_date', 'industry'])['pct_chg'].mean().unstack(fill_value=0.0).reset_index()
    ind_cols = [col for col in df_ind.columns if col not in ['trade_date', 'Unknown']]
    
    # 4. 执行风格归因回归比较
    print("\n>>> Step 4: Performing Style Attribution regressions...")
    results = {}
    results['Config_A'] = run_regression_for_config(
        os.path.join(RES_DIR_A, 'portfolio_comparison_nav.csv'),
        start_date, end_date, df_mkt, df_smb, df_ind, ind_cols
    )
    results['Config_B'] = run_regression_for_config(
        os.path.join(RES_DIR_B, 'portfolio_comparison_nav.csv'),
        start_date, end_date, df_mkt, df_smb, df_ind, ind_cols
    )
    
    # 5. 输出对比报告
    report = []
    report.append("==========================================================================")
    report.append("                 Incremental Factor Regression Report                    ")
    report.append("==========================================================================")
    report.append(f"Comparison Period: {start_date} to {end_date} (Features Active)")
    report.append("Regresion Model: Model 2 (Market + SMB + Orthogonalized Industry Excess)")
    report.append("==========================================================================")
    report.append(f"{'Metric':<25} | {'Config A (Baseline)':<25} | {'Config B (+Vibe Alphas)':<25}")
    report.append("--------------------------------------------------------------------------")
    
    # 收益指标
    report.append(f"{'Total Return':<25} | {results['Config_A']['total_return']:<25.2%} | {results['Config_B']['total_return']:<25.2%}")
    report.append(f"{'Sharpe Ratio':<25} | {results['Config_A']['sharpe']:<25.2f} | {results['Config_B']['sharpe']:<25.2f}")
    report.append(f"{'Max Drawdown':<25} | {results['Config_A']['max_drawdown']:<25.2%} | {results['Config_B']['max_drawdown']:<25.2%}")
    
    # 回归指标
    report.append("--------------------------------------------------------------------------")
    report.append(f"{'Annualized Alpha':<25} | {results['Config_A']['alpha_ann']:<25.2%} | {results['Config_B']['alpha_ann']:<25.2%}")
    report.append(f"{'Alpha t-statistic':<25} | {results['Config_A']['t_stat']:<25.4f} | {results['Config_B']['t_stat']:<25.4f}")
    report.append(f"{'Alpha p-value':<25} | {results['Config_A']['p_value']:<25.6f} | {results['Config_B']['p_value']:<25.6f}")
    report.append(f"{'Beta Market (beta_m)':<25} | {results['Config_A']['beta_m']:<25.4f} | {results['Config_B']['beta_m']:<25.4f}")
    report.append(f"{'Beta Size (beta_s)':<25} | {results['Config_A']['beta_s']:<25.4f} | {results['Config_B']['beta_s']:<25.4f}")
    report.append(f"{'R-squared':<25} | {results['Config_A']['r2']:<25.2%} | {results['Config_B']['r2']:<25.2%}")
    report.append("==========================================================================")
    
    # 因子分析结论
    report.append("\nEmpirical Conclusions:")
    # A vs B
    vibe_alpha_diff = results['Config_B']['alpha_ann'] - results['Config_A']['alpha_ann']
    report.append(f"1. Preprocessed Vibe Alphas Incremental Value (Config B vs A):")
    report.append(f"   - Annualized Alpha change: {vibe_alpha_diff:+.2%}")
    report.append(f"   - Alpha t-stat: A = {results['Config_A']['t_stat']:.2f} ({'sig' if results['Config_A']['p_value'] < 0.05 else 'not sig'}) | B = {results['Config_B']['t_stat']:.2f} ({'sig' if results['Config_B']['p_value'] < 0.05 else 'not sig'})")
    
    r2_explain = results['Config_B']['r2']
    report.append(f"\n2. Style Explanation:")
    report.append(f"   - Config B's style/industry R2 is {r2_explain:.2%}, meaning style exposures explain the vast majority of daily return fluctuations.")
    report.append("==========================================================================")
    
    report_text = "\n".join(report)
    print(report_text, flush=True)
    
    with open(REPORT_FILE, 'w', encoding='utf-8') as f:
        f.write(report_text)
    print(f"\nSaved incremental report to {REPORT_FILE}")

if __name__ == '__main__':
    main()

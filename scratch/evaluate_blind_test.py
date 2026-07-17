import os
import pandas as pd
import numpy as np
import statsmodels.api as sm

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(PROJECT_DIR, 'longterm-research', 'results')
DATA_DIR = os.path.join(PROJECT_DIR, 'longterm-research', 'data')

NAV_FILE = os.path.join(RESULTS_DIR, 'portfolio_comparison_nav.csv')

def evaluate():
    if not os.path.exists(NAV_FILE):
        print(f"NAV file not found: {NAV_FILE}")
        return
        
    df_nav = pd.read_csv(NAV_FILE, index_col=0)
    df_nav.index = pd.to_datetime(df_nav.index)
    df_nav = df_nav.sort_index()
    
    # 截取盲测期 (2025-09-01 起)
    test_df = df_nav[df_nav.index >= '2025-09-01'].copy()
    print("==========================================================================")
    print(f"Blind Test Period: {test_df.index.min().strftime('%Y-%m-%d')} to {test_df.index.max().strftime('%Y-%m-%d')} ({len(test_df)} trading days)")
    print("==========================================================================")
    
    # 计算每日收益率
    for col in test_df.columns:
        ret = test_df[col].pct_change().fillna(0.0)
        ann_ret = ret.mean() * 252
        ann_std = ret.std() * np.sqrt(252)
        sharpe = ann_ret / ann_std if ann_std > 0 else 0.0
        max_dd = (test_df[col] / test_df[col].cummax() - 1).min()
        tot_ret = test_df[col].iloc[-1] / test_df[col].iloc[0] - 1
        print(f"{col:<25} | Tot Ret: {tot_ret:+.2%} | Ann Ret: {ann_ret:+.2%} | Sharpe: {sharpe:.2f} | Max DD: {max_dd:.2%}")
    print("==========================================================================\n")
    
    # 2. 运行盲测期的风格归因回归
    print(">>> Running Style Attribution for the Blind Test Period...")
    # 从 parquet 获取因子
    FEATURES_FILE = os.path.join(DATA_DIR, 'features_longterm.parquet')
    df_feat = pd.read_parquet(FEATURES_FILE, columns=['trade_date', 'pct_chg', 'circ_mv', 'industry'])
    df_feat['trade_date'] = df_feat['trade_date'].astype(str)
    
    trade_dates_str = test_df.index.strftime('%Y%m%d').tolist()
    df_feat = df_feat[df_feat['trade_date'].isin(trade_dates_str)].copy()
    
    df_feat['pct_chg'] = df_feat['pct_chg'].fillna(0.0)
    df_feat['circ_mv'] = pd.to_numeric(df_feat['circ_mv'], errors='coerce').fillna(0.0)
    
    # 计算市场因子与 SMB
    df_mkt = df_feat.groupby('trade_date')['pct_chg'].mean().reset_index().rename(columns={'pct_chg': 'R_m'})
    
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
    df_ind = df_feat.groupby(['trade_date', 'industry'])['pct_chg'].mean().unstack(fill_value=0.0).reset_index()
    
    # 合并
    df_reg = pd.DataFrame(index=test_df.index)
    df_reg['Strategy_Ret'] = test_df['Strategy_Pure'].pct_change().fillna(0.0)
    df_reg['trade_date_str'] = df_reg.index.strftime('%Y%m%d')
    
    df_reg = df_reg.merge(df_mkt, left_on='trade_date_str', right_on='trade_date', how='inner')
    df_reg = df_reg.merge(df_smb, on='trade_date', how='inner')
    df_reg = df_reg.merge(df_ind, on='trade_date', how='inner')
    
    ind_cols = [col for col in df_ind.columns if col not in ['trade_date', 'Unknown']]
    for col in ind_cols:
        df_reg[col] = df_reg[col] - df_reg['R_m']
        
    y = df_reg['Strategy_Ret']
    
    # Model 1
    X1 = df_reg[['R_m', 'SMB']]
    X1 = sm.add_constant(X1)
    model1 = sm.OLS(y, X1).fit()
    
    # Model 2
    X2 = df_reg[['R_m', 'SMB'] + ind_cols]
    X2 = sm.add_constant(X2)
    model2 = sm.OLS(y, X2).fit()
    
    print("[Model 1: Market & SMB]")
    print(f"Annualized Alpha: {model1.params['const']*252:+.2%}")
    print(f"t-stat:           {model1.tvalues['const']:.4f}")
    print(f"p-value:          {model1.pvalues['const']:.6f}")
    print(f"Beta Market:      {model1.params['R_m']:.4f}")
    print(f"Beta Size:        {model1.params['SMB']:.4f}")
    print(f"R-squared:        {model1.rsquared:.4f}")
    print()
    print("[Model 2: Market + SMB + Industry Excess]")
    print(f"Annualized Alpha: {model2.params['const']*252:+.2%}")
    print(f"t-stat:           {model2.tvalues['const']:.4f}")
    print(f"p-value:          {model2.pvalues['const']:.6f}")
    print(f"Beta Market:      {model2.params['R_m']:.4f}")
    print(f"Beta Size:        {model2.params['SMB']:.4f}")
    print(f"R-squared:        {model2.rsquared:.4f}")
    print("==========================================================================")

if __name__ == '__main__':
    evaluate()

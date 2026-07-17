import os
import pandas as pd
import numpy as np

# Define paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
RESULTS_DIR = os.path.join(PROJECT_DIR, 'results')

def analyze_behavior():
    print("Loading backtest history...")
    nav_df = pd.read_csv(os.path.join(RESULTS_DIR, 'nav_full_history.csv'))
    nav_df['trade_date'] = pd.to_datetime(nav_df['trade_date'])
    nav_df = nav_df.set_index('trade_date')
    
    # Load underlying indices to get trend status
    # Actually, nav_full_history.csv contains 'w_300', 'w_500', 'w_bond'. Let's use that!
    
    # Calculate daily returns
    nav_df['strat_ret'] = nav_df['nav'].pct_change().fillna(0.0)
    
    # Let's load the index daily returns to compute correlation
    hs300_daily = pd.read_csv(os.path.join(PROJECT_DIR, 'data/hs300_daily.csv'))
    hs300_daily['trade_date'] = pd.to_datetime(hs300_daily['trade_date'].astype(str))
    hs300_daily = hs300_daily.sort_values('trade_date').set_index('trade_date')
    hs300_daily['ret_300'] = hs300_daily['pct_chg'] / 100.0
    hs300_daily['ma250'] = hs300_daily['close'].rolling(250).mean()
    hs300_daily['uptrend'] = hs300_daily['close'] >= hs300_daily['ma250']
    
    zz500_daily = pd.read_csv(os.path.join(PROJECT_DIR, 'data/zz500_daily.csv'))
    zz500_daily['trade_date'] = pd.to_datetime(zz500_daily['trade_date'].astype(str))
    zz500_daily = zz500_daily.sort_values('trade_date').set_index('trade_date')
    zz500_daily['ret_500'] = zz500_daily['pct_chg'] / 100.0
    zz500_daily['ma250'] = zz500_daily['close'].rolling(250).mean()
    zz500_daily['uptrend'] = zz500_daily['close'] >= zz500_daily['ma250']

    # Merge
    merged = nav_df.join(hs300_daily[['ret_300', 'uptrend', 'close', 'ma250']], how='inner', rsuffix='_300')
    merged = merged.join(zz500_daily[['ret_500', 'uptrend']], how='inner', rsuffix='_500')
    
    # 1. Average Equity weight (HS300 + ZZ500)
    avg_eq_weight = (merged['w_300'] + merged['w_500']).mean()
    print(f"Overall average equity weight: {avg_eq_weight:.2%}")
    
    # 2. Average Equity weight in HS300 uptrend vs downtrend
    avg_eq_uptrend_300 = (merged[merged['uptrend']]['w_300'] + merged[merged['uptrend']]['w_500']).mean()
    avg_eq_downtrend_300 = (merged[~merged['uptrend']]['w_300'] + merged[~merged['uptrend']]['w_500']).mean()
    print(f"Average equity weight when HS300 is in Uptrend (Close >= MA250): {avg_eq_uptrend_300:.2%}")
    print(f"Average equity weight when HS300 is in Downtrend (Close < MA250): {avg_eq_downtrend_300:.2%}")
    
    # 3. Correlation with index returns during uptrends
    # Let's define market index return as the average of HS300 and ZZ500
    merged['mkt_ret'] = (merged['ret_300'] + merged['ret_500']) / 2.0
    
    # Correlation during HS300 uptrend
    uptrend_data = merged[merged['uptrend']]
    corr_uptrend = uptrend_data['strat_ret'].corr(uptrend_data['mkt_ret'])
    print(f"Correlation with market equal-weight return during HS300 Uptrend: {corr_uptrend:.4f}")
    
    # Correlation during HS300 downtrend
    downtrend_data = merged[~merged['uptrend']]
    corr_downtrend = downtrend_data['strat_ret'].corr(downtrend_data['mkt_ret'])
    print(f"Correlation with market equal-weight return during HS300 Downtrend: {corr_downtrend:.4f}")
    
    # Let's calculate Beta with market equal-weight during HS300 uptrend
    # Beta = Cov(Rp, Rm) / Var(Rm)
    cov = np.cov(uptrend_data['strat_ret'], uptrend_data['mkt_ret'])[0, 1]
    var_m = np.var(uptrend_data['mkt_ret'])
    beta_uptrend = cov / var_m if var_m > 0 else 0.0
    print(f"Beta with market equal-weight during HS300 Uptrend: {beta_uptrend:.4f}")
    
    # Let's see how much drawdown was compressed
    # Drawdown of the strategy
    cum_max_s = merged['nav'].cummax()
    dd_s = (merged['nav'] - cum_max_s) / cum_max_s
    print(f"Max Strategy Drawdown: {dd_s.min():.2%}")
    
    # Drawdown of the HS300 index
    cum_max_300 = merged['close'].cummax()
    dd_300 = (merged['close'] - cum_max_300) / cum_max_300
    print(f"Max HS300 Drawdown: {dd_300.min():.2%}")

if __name__ == "__main__":
    analyze_behavior()

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Define paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(PROJECT_DIR, 'data')
RESULTS_DIR = os.path.join(PROJECT_DIR, 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

def bs_put_price(S, K, T, r, sigma):
    """
    Black-Scholes Put Option Pricing Formula.
    S: Current asset price
    K: Strike price
    T: Time to expiration in years
    r: Risk-free rate (annualized, decimal)
    sigma: Implied volatility (annualized, decimal)
    """
    if T <= 0:
        return max(K - S, 0.0)
    if S <= 0 or K <= 0 or sigma <= 0:
        return 0.0
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    return max(price, 0.0)

def load_backtest_data(ma_window=50, val_window=1400):
    # Load daily quotes
    df_300_price = pd.read_csv(os.path.join(DATA_DIR, 'hs300_daily.csv'))
    df_500_price = pd.read_csv(os.path.join(DATA_DIR, 'zz500_daily.csv'))
    df_chinext_price = pd.read_csv(os.path.join(DATA_DIR, 'chinext_daily.csv'))
    df_div_price = pd.read_csv(os.path.join(DATA_DIR, 'div_low_vol_daily.csv'))
    df_gold_price = pd.read_csv(os.path.join(DATA_DIR, 'gold_etf_daily.csv'))
    df_nasdaq_price = pd.read_csv(os.path.join(DATA_DIR, 'nasdaq_etf_daily.csv'))
    df_bond_price = pd.read_csv(os.path.join(DATA_DIR, 'bond_etf_daily.csv'))
    
    # Load daily valuations
    df_300_val = pd.read_csv(os.path.join(DATA_DIR, 'hs300_valuation.csv'))
    df_500_val = pd.read_csv(os.path.join(DATA_DIR, 'zz500_valuation.csv'))
    df_chinext_val = pd.read_csv(os.path.join(DATA_DIR, 'chinext_valuation.csv'))
    df_sse50_val = pd.read_csv(os.path.join(DATA_DIR, 'sse50_valuation.csv'))
    
    # Load China 10-Year Bond Yield
    df_bond_yield = pd.read_csv(os.path.join(DATA_DIR, 'bond_yield_china.csv'))
    
    # Preprocess dates
    dfs = [df_300_price, df_500_price, df_chinext_price, df_div_price, df_gold_price, df_nasdaq_price, df_bond_price,
           df_300_val, df_500_val, df_chinext_val, df_sse50_val, df_bond_yield]
    for df in dfs:
        df['trade_date'] = pd.to_datetime(df['trade_date'].astype(str))
        df.sort_values('trade_date', inplace=True)
        df.reset_index(drop=True, inplace=True)
        
    # Merge bond yield into valuations for ERP calculation
    df_300_val = pd.merge(df_300_val, df_bond_yield, on='trade_date', how='left').ffill()
    df_500_val = pd.merge(df_500_val, df_bond_yield, on='trade_date', how='left').ffill()
    df_chinext_val = pd.merge(df_chinext_val, df_bond_yield, on='trade_date', how='left').ffill()
    df_sse50_val = pd.merge(df_sse50_val, df_bond_yield, on='trade_date', how='left').ffill()
    
    # Calculate fast MAs on prices
    df_300_price['ma'] = df_300_price['close'].rolling(ma_window).mean()
    df_500_price['ma'] = df_500_price['close'].rolling(ma_window).mean()
    df_chinext_price['ma'] = df_chinext_price['close'].rolling(ma_window).mean()
    df_div_price['ma'] = df_div_price['close'].rolling(ma_window).mean()
    df_gold_price['ma'] = df_gold_price['close'].rolling(ma_window).mean()
    df_nasdaq_price['ma'] = df_nasdaq_price['close'].rolling(ma_window).mean()
    
    # Calculate valuation quantiles (standard PE/PB rolling average)
    for val_df in [df_300_val, df_500_val, df_chinext_val, df_sse50_val]:
        val_df['pe_q'] = val_df['pe_ttm'].rolling(window=val_window, min_periods=250).rank(pct=True)
        val_df['pb_q'] = val_df['pb'].rolling(window=val_window, min_periods=250).rank(pct=True)
        val_df['val_q'] = (val_df['pe_q'] + val_df['pb_q']) / 2.0
        
    # Merge price and valuations
    m300 = pd.merge(df_300_price[['trade_date', 'close', 'pct_chg', 'ma']], df_300_val[['trade_date', 'val_q', 'yield_10y']], on='trade_date', how='inner')
    m500 = pd.merge(df_500_price[['trade_date', 'close', 'pct_chg', 'ma']], df_500_val[['trade_date', 'val_q', 'yield_10y']], on='trade_date', how='inner')
    mchinext = pd.merge(df_chinext_price[['trade_date', 'close', 'pct_chg', 'ma']], df_chinext_val[['trade_date', 'val_q', 'yield_10y']], on='trade_date', how='inner')
    mdiv = pd.merge(df_div_price[['trade_date', 'close', 'pct_chg', 'ma']], df_sse50_val[['trade_date', 'val_q', 'yield_10y']], on='trade_date', how='inner')
    
    mgold = df_gold_price[['trade_date', 'close', 'pct_chg', 'ma']].copy()
    mnasdaq = df_nasdaq_price[['trade_date', 'close', 'pct_chg', 'ma']].copy()
    mbond = df_bond_price[['trade_date', 'pct_chg']].copy()
    mbond['bond_ret'] = mbond['pct_chg'] / 100.0
    bond_map = mbond.set_index('trade_date')['bond_ret'].to_dict()
    
    trading_dates = m300['trade_date'].tolist()
    
    m300_dict = m300.set_index('trade_date').to_dict(orient='index')
    m500_dict = m500.set_index('trade_date').to_dict(orient='index')
    mchinext_dict = mchinext.set_index('trade_date').to_dict(orient='index')
    mdiv_dict = mdiv.set_index('trade_date').to_dict(orient='index')
    mgold_dict = mgold.set_index('trade_date').to_dict(orient='index')
    mnasdaq_dict = mnasdaq.set_index('trade_date').to_dict(orient='index')
    
    rows = []
    for dt in trading_dates:
        row300 = m300_dict.get(dt)
        row500 = m500_dict.get(dt)
        rowchinext = mchinext_dict.get(dt)
        rowdiv = mdiv_dict.get(dt)
        rowgold = mgold_dict.get(dt)
        rownasdaq = mnasdaq_dict.get(dt)
        
        if any(r is None for r in [row300, row500, rowchinext, rowdiv, rowgold, rownasdaq]):
            continue
            
        bond_ret = bond_map.get(dt, 0.03 / 242.0)
        if pd.isna(bond_ret):
            bond_ret = 0.03 / 242.0
            
        rows.append({
            'trade_date': dt,
            'close_300': row300['close'], 'ret_300': row300['pct_chg'] / 100.0, 'ma_300': row300['ma'], 'val_q_300': row300['val_q'], 'yield_10y_300': row300['yield_10y'],
            'close_500': row500['close'], 'ret_500': row500['pct_chg'] / 100.0, 'ma_500': row500['ma'], 'val_q_500': row500['val_q'], 'yield_10y_500': row500['yield_10y'],
            'close_chinext': rowchinext['close'], 'ret_chinext': rowchinext['pct_chg'] / 100.0, 'ma_chinext': rowchinext['ma'], 'val_q_chinext': rowchinext['val_q'], 'yield_10y_chinext': rowchinext['yield_10y'],
            'close_div': rowdiv['close'], 'ret_div': rowdiv['pct_chg'] / 100.0, 'ma_div': rowdiv['ma'], 'val_q_div': rowdiv['val_q'], 'yield_10y_div': rowdiv['yield_10y'],
            'close_gold': rowgold['close'], 'ret_gold': rowgold['pct_chg'] / 100.0, 'ma_gold': rowgold['ma'],
            'close_nasdaq': rownasdaq['close'], 'ret_nasdaq': rownasdaq['pct_chg'] / 100.0, 'ma_nasdaq': rownasdaq['ma'],
            'ret_bond': bond_ret
        })
        
    df_unified = pd.DataFrame(rows).sort_values('trade_date').reset_index(drop=True)
    
    # Load QVIX daily if exists
    qvix_path = os.path.join(DATA_DIR, 'qvix_daily.csv')
    if os.path.exists(qvix_path):
        df_qvix = pd.read_csv(qvix_path)
        df_qvix['trade_date'] = pd.to_datetime(df_qvix['date'].astype(str))
        df_qvix = df_qvix[['trade_date', 'close']].rename(columns={'close': 'qvix'})
        df_unified = pd.merge(df_unified, df_qvix, on='trade_date', how='left')
        df_unified['qvix'] = df_unified['qvix'].ffill().bfill().fillna(20.0)
    else:
        df_unified['qvix'] = 20.0
        
    return df_unified

def run_option_backtest(df_period, buy_put=False, strike_ratio=0.97, iv=0.20, initial_capital=1000000.0):
    if len(df_period) == 0:
        return None
        
    df_period = df_period.copy().reset_index(drop=True)
    
    # Portfolio holdings value
    val_300 = 0.0
    val_500 = 0.0
    val_chinext = 0.0
    val_div = 0.0
    val_gold = 0.0
    val_nasdaq = 0.0
    val_bond = initial_capital
    
    nav_history = []
    
    # Option tracking
    # Each item: {'expiry_idx': idx, 'asset': '300'/'500'/'chinext'/'div', 'purchase_val': float, 'purchase_price': float, 'strike_price': float}
    options_held = []
    
    for idx, row in df_period.iterrows():
        dt = row['trade_date']
        
        # 1. Update equity values with daily returns
        if idx > 0:
            val_300 *= (1.0 + row['ret_300'])
            val_500 *= (1.0 + row['ret_500'])
            val_chinext *= (1.0 + row['ret_chinext'])
            val_div *= (1.0 + row['ret_div'])
            val_gold *= (1.0 + row['ret_gold'])
            val_nasdaq *= (1.0 + row['ret_nasdaq'])
            val_bond *= (1.0 + row['ret_bond'])
            
        # 2. Check option expiries first (credited to cash/bond balance before rebalancing)
        payoff_today = 0.0
        active_options = []
        for opt in options_held:
            if idx >= opt['expiry_idx']:
                # Calculate payoff
                asset = opt['asset']
                close_price = row[f'close_{asset}']
                purchase_price = opt['purchase_price']
                strike_price = opt['strike_price']
                
                # Payoff = purchase_val * max(strike_price / purchase_price - close_price / purchase_price, 0)
                payoff = opt['purchase_val'] * max(strike_price / purchase_price - close_price / purchase_price, 0.0)
                payoff_today += payoff
            else:
                active_options.append(opt)
        options_held = active_options
        val_bond += payoff_today
        
        # Compute NAV before daily rebalancing
        nav = val_300 + val_500 + val_chinext + val_div + val_gold + val_nasdaq + val_bond
        
        # 3. Daily Rebalancing (Reactionary Fast exit)
        trend_300 = row['close_300'] >= row['ma_300'] if not pd.isna(row['ma_300']) else False
        trend_500 = row['close_500'] >= row['ma_500'] if not pd.isna(row['ma_500']) else False
        trend_chinext = row['close_chinext'] >= row['ma_chinext'] if not pd.isna(row['ma_chinext']) else False
        trend_div = row['close_div'] >= row['ma_div'] if not pd.isna(row['ma_div']) else False
        trend_gold = row['close_gold'] >= row['ma_gold'] if not pd.isna(row['ma_gold']) else False
        trend_nasdaq = row['close_nasdaq'] >= row['ma_nasdaq'] if not pd.isna(row['ma_nasdaq']) else False
        
        # Fast Momentum target weights:
        # Full equal-weight (20% each for valuation indices, 10% each for gold/nasdaq) if in uptrend, else 0%
        w_target_300 = 0.20 if trend_300 else 0.0
        w_target_500 = 0.20 if trend_500 else 0.0
        w_target_chinext = 0.20 if trend_chinext else 0.0
        w_target_div = 0.20 if trend_div else 0.0
        w_target_gold = 0.10 if trend_gold else 0.0
        w_target_nasdaq = 0.10 if trend_nasdaq else 0.0
        
        w_target_bond = 1.0 - (w_target_300 + w_target_500 + w_target_chinext + w_target_div + w_target_gold + w_target_nasdaq)
        
        # Current weights
        w_curr_300 = val_300 / nav
        w_curr_500 = val_500 / nav
        w_curr_chinext = val_chinext / nav
        w_curr_div = val_div / nav
        w_curr_gold = val_gold / nav
        w_curr_nasdaq = val_nasdaq / nav
        
        # Check deviations
        devs = [
            abs(w_curr_300 - w_target_300),
            abs(w_curr_500 - w_target_500),
            abs(w_curr_chinext - w_target_chinext),
            abs(w_curr_div - w_target_div),
            abs(w_curr_gold - w_target_gold),
            abs(w_curr_nasdaq - w_target_nasdaq)
        ]
        
        # Since we use daily "reactionary" exit, we trigger trade if status shifts (weights deviate > 5%) or first day
        if any(d > 0.05 for d in devs) or idx == 0:
            val_target_300 = nav * w_target_300
            val_target_500 = nav * w_target_500
            val_target_chinext = nav * w_target_chinext
            val_target_div = nav * w_target_div
            val_target_gold = nav * w_target_gold
            val_target_nasdaq = nav * w_target_nasdaq
            val_target_bond = nav * w_target_bond
            
            trade_vol = (abs(val_target_300 - val_300) + 
                         abs(val_target_500 - val_500) + 
                         abs(val_target_chinext - val_chinext) + 
                         abs(val_target_div - val_div) + 
                         abs(val_target_gold - val_gold) + 
                         abs(val_target_nasdaq - val_nasdaq) + 
                         abs(val_target_bond - val_bond))
            cost = trade_vol * 0.0005 # 0.05% cost
            
            nav -= cost
            val_300 = nav * w_target_300
            val_500 = nav * w_target_500
            val_chinext = nav * w_target_chinext
            val_div = nav * w_target_div
            val_gold = nav * w_target_gold
            val_nasdaq = nav * w_target_nasdaq
            val_bond = nav * w_target_bond
            
        # 4. Monthly Protective Put Purchase (Every 20 trading days)
        # Check if we should buy options today
        if buy_put and (idx % 20 == 0):
            # Time to expiry: 20 trading days ≈ 20/252 of a year
            T_years = 20.0 / 252.0
            
            # For each valuation asset, buy Put if Q > 70% AND we currently hold a long position in it
            assets = ['300', '500', 'chinext', 'div']
            for asset in assets:
                val_q = row[f'val_q_{asset}']
                val_holding = locals()[f'val_{asset}'] # Get current valuation holding value
                
                if (not pd.isna(val_q)) and (val_q > 0.70) and (val_holding > 0.0):
                    S0 = row[f'close_{asset}']
                    K = S0 * strike_ratio
                    r = (row[f'yield_10y_{asset}'] / 100.0) if not pd.isna(row[f'yield_10y_{asset}']) else 0.025
                    
                    # Compute BS Put Price
                    current_iv = (row['qvix'] / 100.0) if iv == 'qvix' else iv
                    put_price_per_share = bs_put_price(S0, K, T_years, r, current_iv)
                    pct_premium = put_price_per_share / S0
                    
                    # Option Cost
                    opt_premium_cost = val_holding * pct_premium
                    
                    # Deduct cost from cash balance
                    val_bond -= opt_premium_cost
                    nav -= opt_premium_cost
                    
                    # Store option contracts
                    options_held.append({
                        'expiry_idx': idx + 20,
                        'asset': asset,
                        'purchase_val': val_holding,
                        'purchase_price': S0,
                        'strike_price': K
                    })
                    
        nav_history.append({'trade_date': dt, 'nav': nav})
        
    return pd.DataFrame(nav_history).set_index('trade_date')

def compute_metrics(nav_series, initial_capital=1000000.0):
    total_ret = nav_series.iloc[-1] / initial_capital - 1
    years = (nav_series.index[-1] - nav_series.index[0]).days / 365.25
    cagr = (nav_series.iloc[-1] / initial_capital) ** (1.0 / years) - 1 if years > 0 else 0.0
    
    daily_rets = nav_series.pct_change().dropna()
    ann_vol = daily_rets.std() * np.sqrt(252)
    sharpe = (daily_rets.mean() * 252) / ann_vol if ann_vol > 0 else 0.0
    
    cum_max = nav_series.cummax()
    drawdowns = (nav_series - cum_max) / cum_max
    max_dd = drawdowns.min()
    
    return {
        'Total Return': f"{total_ret:.2%}",
        'CAGR': f"{cagr:.2%}",
        'Volatility': f"{ann_vol:.2%}",
        'Sharpe': f"{sharpe:.2f}",
        'Max Drawdown': f"{max_dd:.2%}"
    }

def main():
    print("Loading data...")
    # Using 50-day fast MA, 1400-day valuation window
    df_unified = load_backtest_data(ma_window=50, val_window=1400)
    
    # Split Dates
    is_start, is_end = "2015-01-01", "2024-02-05"
    oos_start, oos_end = "2024-02-06", "2026-03-13"
    
    df_is = df_unified[(df_unified['trade_date'] >= pd.to_datetime(is_start)) & (df_unified['trade_date'] <= pd.to_datetime(is_end))].copy()
    df_oos = df_unified[(df_unified['trade_date'] >= pd.to_datetime(oos_start)) & (df_unified['trade_date'] <= pd.to_datetime(oos_end))].copy()
    
    # Test configurations
    # 1. No Options Baseline (Fast Momentum only)
    # 2. Option Protected (ATM, Strike = 1.0)
    # 3. Option Protected (3% OTM, Strike = 0.97)
    # 4. Option Protected (5% OTM, Strike = 0.95)
    
    print("\n--- Running In-Sample Backtests (2015-01-01 to 2024-02-05) ---")
    nav_is_base = run_option_backtest(df_is, buy_put=False)
    nav_is_atm = run_option_backtest(df_is, buy_put=True, strike_ratio=1.00)
    nav_is_otm3 = run_option_backtest(df_is, buy_put=True, strike_ratio=0.97)
    nav_is_otm5 = run_option_backtest(df_is, buy_put=True, strike_ratio=0.95)
    
    metrics_is_base = compute_metrics(nav_is_base['nav'])
    metrics_is_atm = compute_metrics(nav_is_atm['nav'])
    metrics_is_otm3 = compute_metrics(nav_is_otm3['nav'])
    metrics_is_otm5 = compute_metrics(nav_is_otm5['nav'])
    
    # Benchmarks in IS
    hs300_nav_is = (1.0 + df_is['ret_300']).cumprod() * 1000000.0
    hs300_nav_is.index = df_is['trade_date']
    metrics_hs300_is = compute_metrics(hs300_nav_is)
    
    print("\n=== IN-SAMPLE OPTION-PROTECTION COMPARISON ===")
    summary_is = pd.DataFrame([
        {'Strategy': 'Fast Momentum Only (Base)', **metrics_is_base},
        {'Strategy': 'Option-Protected (ATM, 1.00)', **metrics_is_atm},
        {'Strategy': 'Option-Protected (3% OTM, 0.97)', **metrics_is_otm3},
        {'Strategy': 'Option-Protected (5% OTM, 0.95)', **metrics_is_otm5},
        {'Strategy': 'HS300 Buy & Hold', **metrics_hs300_is}
    ])
    print(summary_is.to_string(index=False))
    
    print("\n--- Running Out-of-Sample Backtests (2024-02-06 to 2026-03-13) ---")
    nav_oos_base = run_option_backtest(df_oos, buy_put=False, initial_capital=1000000.0)
    nav_oos_atm = run_option_backtest(df_oos, buy_put=True, strike_ratio=1.00, initial_capital=1000000.0)
    nav_oos_otm3 = run_option_backtest(df_oos, buy_put=True, strike_ratio=0.97, initial_capital=1000000.0)
    nav_oos_otm5 = run_option_backtest(df_oos, buy_put=True, strike_ratio=0.95, initial_capital=1000000.0)
    
    metrics_oos_base = compute_metrics(nav_oos_base['nav'])
    metrics_oos_atm = compute_metrics(nav_oos_atm['nav'])
    metrics_oos_otm3 = compute_metrics(nav_oos_otm3['nav'])
    metrics_oos_otm5 = compute_metrics(nav_oos_otm5['nav'])
    
    # Benchmarks in OOS
    hs300_nav_oos = (1.0 + df_oos['ret_300']).cumprod() * 1000000.0
    hs300_nav_oos.index = df_oos['trade_date']
    metrics_hs300_oos = compute_metrics(hs300_nav_oos)
    
    print("\n=== OUT-OF-SAMPLE OPTION-PROTECTION COMPARISON (BLIND TEST) ===")
    summary_oos = pd.DataFrame([
        {'Strategy': 'Fast Momentum Only (Base)', **metrics_oos_base},
        {'Strategy': 'Option-Protected (ATM, 1.00)', **metrics_oos_atm},
        {'Strategy': 'Option-Protected (3% OTM, 0.97)', **metrics_oos_otm3},
        {'Strategy': 'Option-Protected (5% OTM, 0.95)', **metrics_oos_otm5},
        {'Strategy': 'HS300 Buy & Hold', **metrics_hs300_oos}
    ])
    print(summary_oos.to_string(index=False))
    
    # Save CSVs
    summary_is.to_csv(os.path.join(RESULTS_DIR, 'option_protection_metrics_is.csv'), index=False)
    summary_oos.to_csv(os.path.join(RESULTS_DIR, 'option_protection_metrics_oos.csv'), index=False)
    
    # Plot IS Curves
    plt.figure(figsize=(12, 6))
    plt.plot(nav_is_base.index, nav_is_base['nav'] / 1e6, label='Fast Momentum Only (Base)', color='#d32f2f', linestyle=':')
    plt.plot(nav_is_atm.index, nav_is_atm['nav'] / 1e6, label='Option-Protected (ATM)', color='#1976d2', linewidth=2.0)
    plt.plot(nav_is_otm3.index, nav_is_otm3['nav'] / 1e6, label='Option-Protected (3% OTM)', color='#388e3c', linewidth=2.0)
    plt.plot(nav_is_otm5.index, nav_is_otm5['nav'] / 1e6, label='Option-Protected (5% OTM)', color='#f57c00', linewidth=1.5)
    plt.plot(hs300_nav_is.index, hs300_nav_is / 1e6, label='HS300 Buy & Hold', color='#757575', alpha=0.5)
    
    plt.title("In-Sample Option-Protection Performance Comparison (2015-01-01 to 2024-02-05)", fontsize=12, fontweight='bold')
    plt.xlabel("Date")
    plt.ylabel("NAV (Normalized)")
    plt.legend(loc='upper left')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'nav_options_comparison_is.png'), dpi=300)
    plt.close()
    
    # Plot OOS Curves
    plt.figure(figsize=(12, 6))
    plt.plot(nav_oos_base.index, nav_oos_base['nav'] / 1e6, label='Fast Momentum Only (Base)', color='#d32f2f', linestyle=':')
    plt.plot(nav_oos_atm.index, nav_oos_atm['nav'] / 1e6, label='Option-Protected (ATM)', color='#1976d2', linewidth=2.0)
    plt.plot(nav_oos_otm3.index, nav_oos_otm3['nav'] / 1e6, label='Option-Protected (3% OTM)', color='#388e3c', linewidth=2.0)
    plt.plot(nav_oos_otm5.index, nav_oos_otm5['nav'] / 1e6, label='Option-Protected (5% OTM)', color='#f57c00', linewidth=1.5)
    plt.plot(hs300_nav_oos.index, hs300_nav_oos / 1e6, label='HS300 Buy & Hold', color='#757575', alpha=0.5)
    
    plt.title("Out-of-Sample Option-Protection Performance Comparison (2024-02-06 to 2026-03-13)", fontsize=12, fontweight='bold')
    plt.xlabel("Date")
    plt.ylabel("NAV (Normalized)")
    plt.legend(loc='upper left')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'nav_options_comparison_oos.png'), dpi=300)
    plt.close()
    
    print("\nOption backtest complete. Metrics saved and charts saved in results/ directory.")

if __name__ == "__main__":
    main()

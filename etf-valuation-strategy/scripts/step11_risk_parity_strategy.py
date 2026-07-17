import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import warnings

warnings.filterwarnings('ignore')

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

def load_data_8assets(ma_window=200, val_window=1400, vol_lookback=60):
    print(f"Loading data with MA={ma_window}, ValWindow={val_window}, VolLookback={vol_lookback}...")
    
    # 1. Load price files
    files = {
        'hs300': 'hs300_daily.csv',
        'zz500': 'zz500_daily.csv',
        'chinext': 'chinext_daily.csv',
        'div': 'div_low_vol_daily.csv',
        'gold': 'gold_etf_daily.csv',
        'nasdaq': 'nasdaq_etf_daily.csv',
        'bond': 'bond_etf_daily.csv',
        'cbond': 'cbond_daily.csv'
    }
    
    price_dfs = {}
    for name, fname in files.items():
        path = os.path.join(DATA_DIR, fname)
        df = pd.read_csv(path)
        df['trade_date'] = pd.to_datetime(df['trade_date'].astype(str))
        df.sort_values('trade_date', inplace=True)
        df.reset_index(drop=True, inplace=True)
        
        # Adjust Nasdaq ETF split on 2022-01-14
        if name == 'nasdaq':
            split_date = pd.to_datetime('2022-01-14')
            adj_factor = 1.038 / 5.192
            mask = df['trade_date'] < split_date
            for col in ['close', 'open', 'high', 'low', 'pre_close']:
                df.loc[mask, col] *= adj_factor
        
        
        # Calculate daily return and volatility
        df['ret'] = df['pct_chg'] / 100.0
        df['vol'] = df['ret'].rolling(window=vol_lookback, min_periods=20).std()
        df['ma'] = df['close'].rolling(window=ma_window, min_periods=20).mean()
        price_dfs[name] = df
        
    # 2. Load valuations
    val_files = {
        'hs300': 'hs300_valuation.csv',
        'zz500': 'zz500_valuation.csv',
        'chinext': 'chinext_valuation.csv',
        'div': 'sse50_valuation.csv' # SSE50 valuation as a proxy for dividend low vol
    }
    
    df_bond_yield = pd.read_csv(os.path.join(DATA_DIR, 'bond_yield_china.csv'))
    df_bond_yield['trade_date'] = pd.to_datetime(df_bond_yield['trade_date'].astype(str))
    df_bond_yield.sort_values('trade_date', inplace=True)
    
    val_dfs = {}
    for name, fname in val_files.items():
        path = os.path.join(DATA_DIR, fname)
        df = pd.read_csv(path)
        df['trade_date'] = pd.to_datetime(df['trade_date'].astype(str))
        df.sort_values('trade_date', inplace=True)
        
        # Merge bond yield
        df = pd.merge(df, df_bond_yield, on='trade_date', how='left').ffill()
        
        # Calculate ERP and rolling ranking
        df['erp'] = 100.0 / df['pe_ttm'] - df['yield_10y']
        df['erp_rank_q'] = df['erp'].rolling(window=val_window, min_periods=250).rank(pct=True)
        df['val_q_erp'] = 1.0 - df['erp_rank_q']
        
        val_dfs[name] = df
        
    # 3. Load QVIX
    qvix_path = os.path.join(DATA_DIR, 'qvix_daily.csv')
    if os.path.exists(qvix_path):
        df_qvix = pd.read_csv(qvix_path)
        df_qvix['trade_date'] = pd.to_datetime(df_qvix['date'].astype(str))
        df_qvix = df_qvix[['trade_date', 'close']].rename(columns={'close': 'qvix'})
    else:
        df_qvix = pd.DataFrame(columns=['trade_date', 'qvix'])
        
    # 4. Sequentially merge all data by trade_date (inner join to align)
    # Start with HS300 price
    df_unified = price_dfs['hs300'][['trade_date', 'close', 'ret', 'vol', 'ma']].rename(
        columns={'close': 'close_hs300', 'ret': 'ret_hs300', 'vol': 'vol_hs300', 'ma': 'ma_hs300'}
    )
    
    # Merge other prices
    for name in ['zz500', 'chinext', 'div', 'gold', 'nasdaq', 'bond', 'cbond']:
        df_asset = price_dfs[name][['trade_date', 'close', 'ret', 'vol', 'ma']].rename(
            columns={'close': f'close_{name}', 'ret': f'ret_{name}', 'vol': f'vol_{name}', 'ma': f'ma_{name}'}
        )
        df_unified = pd.merge(df_unified, df_asset, on='trade_date', how='inner')
        
    # Merge valuations
    for name in ['hs300', 'zz500', 'chinext', 'div']:
        df_val = val_dfs[name][['trade_date', 'val_q_erp', 'yield_10y']].rename(
            columns={'val_q_erp': f'val_q_{name}', 'yield_10y': f'yield_10y_{name}'}
        )
        df_unified = pd.merge(df_unified, df_val, on='trade_date', how='inner')
        
    # Merge QVIX
    df_unified = pd.merge(df_unified, df_qvix, on='trade_date', how='left')
    df_unified['qvix'] = df_unified['qvix'].ffill().bfill().fillna(20.0)
    
    df_unified.sort_values('trade_date', inplace=True)
    df_unified.reset_index(drop=True, inplace=True)
    
    print(f"Unified dataset built. Shape: {df_unified.shape}, Dates: {df_unified['trade_date'].min().strftime('%Y-%m-%d')} to {df_unified['trade_date'].max().strftime('%Y-%m-%d')}")
    return df_unified

def run_backtest_risk_parity(
    df_period,
    vol_target=0.06,
    val_tilt=0.4,
    q_threshold=0.20,
    strike_ratio=0.97,
    trend_reduce_factor=0.5,
    vol_lookback=60,
    dev_threshold=0.05,
    initial_capital=1000000.0,
    rf_rate=0.02,
    buy_put=True
):
    if len(df_period) == 0:
        return None
        
    df_period = df_period.copy().reset_index(drop=True)
    
    # 8 Assets list
    assets = ['hs300', 'zz500', 'chinext', 'div', 'gold', 'nasdaq', 'bond', 'cbond']
    equity_assets = ['hs300', 'zz500', 'chinext', 'div']
    
    # Identify weekly rebalance dates
    df_period['year_week'] = df_period['trade_date'].dt.strftime('%Y-%U')
    rebalance_check_dates = set(df_period.groupby('year_week')['trade_date'].first())
    
    # Pre-calculate asset returns matrix for covariance estimation
    ret_cols = [f'ret_{a}' for a in assets]
    df_returns = df_period[ret_cols].copy()
    df_returns.columns = assets
    
    # Portfolio holdings value
    val = {a: 0.0 for a in assets}
    val_cash = initial_capital
    
    # Options held list
    options_held = []
    
    nav_history = []
    weight_history = []
    vol_history = []
    
    for idx, row in df_period.iterrows():
        dt = row['trade_date']
        
        # 1. Update holdings values with daily returns
        if idx > 0:
            for a in assets:
                val[a] *= (1.0 + row[f'ret_{a}'])
            val_cash *= (1.0 + rf_rate / 242.0)
            
        # 2. Check option expiries (credit payoffs to cash)
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
        val_cash += payoff_today
        
        # Calculate current NAV before any trading
        holdings_value = sum(val.values())
        nav = holdings_value + val_cash
        
        # 3. Calculate target weights
        # a. Risk Parity Base Weights (Volatility Inverse)
        vols = np.array([row[f'vol_{a}'] for a in assets])
        # Prevent division by zero
        vols = np.where(vols <= 0, 1e-4, vols)
        inv_vols = 1.0 / vols
        w_rp = inv_vols / inv_vols.sum()
        
        # Convert to dict for easier manipulation
        w_target = {a: w_rp[i] for i, a in enumerate(assets)}
        
        # b. Valuation Timing Overlay (for A-share equities)
        for a in equity_assets:
            val_q = row[f'val_q_{a}']
            if not pd.isna(val_q):
                # Underweight when expensive (val_q -> 1), overweight when cheap (val_q -> 0)
                w_target[a] *= (1.0 - val_tilt * (val_q - 0.5))
                
        # d. Trend Filter Overlay
        for a in assets:
            close_px = row[f'close_{a}']
            ma_px = row[f'ma_{a}']
            trend_up = close_px >= ma_px if not pd.isna(ma_px) else True
            
            # Apply trend rule
            if not trend_up:
                # Exemption for very cheap equities
                if a in equity_assets:
                    val_q = row[f'val_q_{a}']
                    if pd.isna(val_q) or val_q > q_threshold:
                        w_target[a] *= trend_reduce_factor
                else:
                    # Gold, Nasdaq, Bond, Convertible bond don't have valuation quantile
                    # Nasdaq and Gold are subject to trend filter
                    if a in ['gold', 'nasdaq']:
                        w_target[a] *= trend_reduce_factor
                        
        # Normalize target weights to sum to 1.0
        w_sum = sum(w_target.values())
        for a in assets:
            w_target[a] /= w_sum
            
        # e. Volatility Target Layer
        # Estimate rolling covariance matrix of returns over the vol lookback
        if idx >= vol_lookback:
            cov_matrix = df_returns.iloc[idx - vol_lookback + 1:idx + 1].cov().values
        else:
            # Fallback if we don't have enough history in this period
            cov_matrix = df_returns.iloc[0:idx + 1].cov().values if idx > 5 else np.eye(len(assets)) * (0.01 / 252.0)
            
        w_vector = np.array([w_target[a] for a in assets])
        port_variance = np.dot(w_vector, np.dot(cov_matrix, w_vector))
        port_vol = np.sqrt(port_variance * 252.0)
        
        # Calculate scaling factor
        sf = min(1.0, vol_target / max(port_vol, 1e-6))
        
        # Final weights after volatility scaling
        w_target_final = {a: w_target[a] * sf for a in assets}
        w_target_final['cash'] = 1.0 - sf
        
        # 4. Check if rebalancing is needed
        # Rebalance if:
        # - It is a weekly rebalance date
        # - Current weights deviate from targets by more than dev_threshold (e.g. daily risk exit)
        # - It is the very first day
        w_curr = {a: val[a] / nav if nav > 0 else 0.0 for a in assets}
        w_curr['cash'] = val_cash / nav if nav > 0 else 0.0
        
        # Calculate weight deviations
        devs = [abs(w_curr[a] - w_target_final[a]) for a in assets] + [abs(w_curr['cash'] - w_target_final['cash'])]
        max_dev = max(devs)
        
        is_rebal_day = (dt in rebalance_check_dates) or (max_dev > dev_threshold) or (idx == 0)
        
        if is_rebal_day:
            # Target values
            val_target = {a: nav * w_target_final[a] for a in assets}
            val_target_cash = nav * w_target_final['cash']
            
            # Calculate trade volume
            trade_vol = sum(abs(val_target[a] - val[a]) for a in assets) + abs(val_target_cash - val_cash)
            cost = trade_vol * 0.0005 # 0.05% execution cost
            
            # Deduct cost from cash and set values to targets
            nav -= cost
            val_cash = nav * w_target_final['cash']
            for a in assets:
                val[a] = nav * w_target_final[a]
                
        # 5. Options Protection Layer (monthly purchase, every 20 days)
        if buy_put and (idx % 20 == 0):
            # Time to expiry: 20 days ≈ 20/252 of a year
            T_years = 20.0 / 252.0
            
            for a in equity_assets:
                val_q = row[f'val_q_{a}']
                val_holding = val[a]
                
                # Buy Put option if asset is expensive (quantile > 0.7) and we hold a positive position
                if (not pd.isna(val_q)) and (val_q > 0.70) and (val_holding > 0.0):
                    S0 = row[f'close_{a}']
                    K = S0 * strike_ratio
                    r = (row[f'yield_10y_{a}'] / 100.0) if not pd.isna(row[f'yield_10y_{a}']) else 0.025
                    
                    # Use QVIX index or rolling volatility for Implied Volatility
                    current_iv = (row['qvix'] / 100.0) if 'qvix' in row else row[f'vol_{a}'] * np.sqrt(252.0)
                    if pd.isna(current_iv) or current_iv <= 0:
                        current_iv = 0.20
                        
                    # Black-Scholes price
                    put_price_per_share = bs_put_price(S0, K, T_years, r, current_iv)
                    pct_premium = put_price_per_share / S0
                    
                    # Option Premium Cost
                    opt_premium_cost = val_holding * pct_premium
                    
                    # Deduct from cash and NAV
                    val_cash -= opt_premium_cost
                    nav -= opt_premium_cost
                    
                    # Track option contracts
                    options_held.append({
                        'expiry_idx': idx + 20,
                        'asset': a,
                        'purchase_val': val_holding,
                        'purchase_price': S0,
                        'strike_price': K
                    })
                    
        # Record history
        nav_history.append({'trade_date': dt, 'nav': nav})
        
        hist_w = {f'w_{a}': val[a]/nav for a in assets}
        hist_w['w_cash'] = val_cash/nav
        hist_w['trade_date'] = dt
        weight_history.append(hist_w)
        
        vol_history.append({'trade_date': dt, 'portfolio_vol': port_vol, 'scaling_factor': sf})
        
    df_nav = pd.DataFrame(nav_history).set_index('trade_date')
    df_weights = pd.DataFrame(weight_history).set_index('trade_date')
    df_vols = pd.DataFrame(vol_history).set_index('trade_date')
    
    return df_nav, df_weights, df_vols

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
    
    calmar = cagr / abs(max_dd) if max_dd != 0 else 0.0
    
    return {
        'Total Return': total_ret,
        'CAGR': cagr,
        'Volatility': ann_vol,
        'Sharpe': sharpe,
        'Max Drawdown': max_dd,
        'Calmar': calmar
    }

def run_grid_search(df_is, df_oos):
    print("\n" + "="*50)
    print("  RUNNING PARAMETER ROBUSTNESS GRID SEARCH")
    print("="*50)
    
    # Search grid
    vol_targets = [0.04, 0.05, 0.06, 0.07, 0.08, 0.10]
    val_tilts = [0.0, 0.2, 0.4, 0.6, 0.8]
    strike_ratios = [0.95, 0.97, 1.00]
    
    results = []
    
    # Run loop
    for vt in vol_targets:
        for v_tilt in val_tilts:
            for sr in strike_ratios:
                # 1. In-sample
                df_nav_is, _, _ = run_backtest_risk_parity(
                    df_is, vol_target=vt, val_tilt=v_tilt, strike_ratio=sr, buy_put=True
                )
                metrics_is = compute_metrics(df_nav_is['nav'])
                
                # 2. Out-of-sample
                df_nav_oos, _, _ = run_backtest_risk_parity(
                    df_oos, vol_target=vt, val_tilt=v_tilt, strike_ratio=sr, buy_put=True
                )
                metrics_oos = compute_metrics(df_nav_oos['nav'])
                
                # Save result row
                results.append({
                    'vol_target': vt,
                    'val_tilt': v_tilt,
                    'strike_ratio': sr,
                    'is_cagr': metrics_is['CAGR'],
                    'is_mdd': metrics_is['Max Drawdown'],
                    'is_sharpe': metrics_is['Sharpe'],
                    'is_calmar': metrics_is['Calmar'],
                    'oos_cagr': metrics_oos['CAGR'],
                    'oos_mdd': metrics_oos['Max Drawdown'],
                    'oos_sharpe': metrics_oos['Sharpe'],
                    'oos_calmar': metrics_oos['Calmar'],
                })
                
    df_results = pd.DataFrame(results)
    df_results.to_csv(os.path.join(RESULTS_DIR, 'risk_parity_grid_search.csv'), index=False)
    print(f"Grid search completed. Saved {len(df_results)} combinations to results/risk_parity_grid_search.csv")
    return df_results

def main():
    # Load unified dataset
    df_unified = load_data_8assets(ma_window=200, val_window=1400, vol_lookback=60)
    
    # Split dates
    is_start, is_end = "2015-01-01", "2024-02-05"
    oos_start, oos_end = "2024-02-06", "2026-03-13"
    
    df_is = df_unified[(df_unified['trade_date'] >= pd.to_datetime(is_start)) & (df_unified['trade_date'] <= pd.to_datetime(is_end))].copy()
    df_oos = df_unified[(df_unified['trade_date'] >= pd.to_datetime(oos_start)) & (df_unified['trade_date'] <= pd.to_datetime(oos_end))].copy()
    
    # Run Grid Search for Robustness
    df_grid = run_grid_search(df_is, df_oos)
    
    # Find parameter set based ONLY on In-Sample performance to prevent OOS leakage
    # Target criteria: Maximize In-Sample CAGR subject to In-Sample MDD >= -12% (absolute MDD <= 12%)
    valid = df_grid[df_grid['is_mdd'] >= -0.12]
    
    if valid.empty:
        print("No parameter set met the In-Sample MDD constraint of -12%. Selecting the one with the highest In-Sample CAGR...")
        best_row = df_grid.sort_values(by='is_cagr', ascending=False).iloc[0]
    else:
        best_row = valid.sort_values(by='is_cagr', ascending=False).iloc[0]
        
    best_vt = best_row['vol_target']
    best_v_tilt = best_row['val_tilt']
    best_sr = best_row['strike_ratio']
    
    print("\n" + "="*50)
    print("  BEST ROBUST PARAMETER CONFIGURATION")
    print("="*50)
    print(f"  Volatility Target:   {best_vt:.1%}")
    print(f"  Valuation Tilt:     {best_v_tilt:.1f}")
    print(f"  Put Option Strike:  {best_sr:.2f}")
    print(f"  In-Sample CAGR:     {best_row['is_cagr']:.2%}")
    print(f"  In-Sample MDD:      {best_row['is_mdd']:.2%}")
    print(f"  Out-of-Sample CAGR:  {best_row['oos_cagr']:.2%}")
    print(f"  Out-of-Sample MDD:   {best_row['oos_mdd']:.2%}")
    print("="*50)
    
    # Run final backtests with best parameters
    print("\nRunning final backtests with chosen parameters...")
    df_nav_is, df_weights_is, df_vols_is = run_backtest_risk_parity(
        df_is, vol_target=best_vt, val_tilt=best_v_tilt, strike_ratio=best_sr, buy_put=True
    )
    df_nav_oos, df_weights_oos, df_vols_oos = run_backtest_risk_parity(
        df_oos, vol_target=best_vt, val_tilt=best_v_tilt, strike_ratio=best_sr, buy_put=True
    )
    
    # Save final NAV and weights
    df_nav_is.to_csv(os.path.join(RESULTS_DIR, 'nav_risk_parity_is.csv'))
    df_nav_oos.to_csv(os.path.join(RESULTS_DIR, 'nav_risk_parity_oos.csv'))
    df_weights_is.to_csv(os.path.join(RESULTS_DIR, 'weights_risk_parity_is.csv'))
    df_weights_oos.to_csv(os.path.join(RESULTS_DIR, 'weights_risk_parity_oos.csv'))
    
    # Calculate Benchmarks for comparison
    # 1. HS300 Buy & Hold
    hs300_is = (df_is['ret_hs300'] + 1.0).cumprod() * 1000000.0
    hs300_is.index = df_is['trade_date']
    hs300_oos = (df_oos['ret_hs300'] + 1.0).cumprod() * 1000000.0
    hs300_oos.index = df_oos['trade_date']
    
    # 2. Equal-Weight 8-asset Portfolio (Rebalanced weekly, transaction costs applied)
    eq_is_history = []
    assets = ['hs300', 'zz500', 'chinext', 'div', 'gold', 'nasdaq', 'bond', 'cbond']
    eq_val = {a: 1000000.0 / len(assets) for a in assets}
    df_is['year_week'] = df_is['trade_date'].dt.strftime('%Y-%U')
    rebalance_check_dates_is = set(df_is.groupby('year_week')['trade_date'].first())
    for idx, row in df_is.reset_index(drop=True).iterrows():
        dt = row['trade_date']
        if idx > 0:
            for a in assets:
                eq_val[a] *= (1.0 + row[f'ret_{a}'])
        nav = sum(eq_val.values())
        if dt in rebalance_check_dates_is:
            trade_vol = sum(abs(nav / len(assets) - eq_val[a]) for a in assets)
            cost = trade_vol * 0.0005
            nav -= cost
            eq_val = {a: nav / len(assets) for a in assets}
        eq_is_history.append({'trade_date': dt, 'nav': nav})
    df_eq_is = pd.DataFrame(eq_is_history).set_index('trade_date')
    
    eq_oos_history = []
    eq_val_oos = {a: 1000000.0 / len(assets) for a in assets}
    df_oos['year_week'] = df_oos['trade_date'].dt.strftime('%Y-%U')
    rebalance_check_dates_oos = set(df_oos.groupby('year_week')['trade_date'].first())
    for idx, row in df_oos.reset_index(drop=True).iterrows():
        dt = row['trade_date']
        if idx > 0:
            for a in assets:
                eq_val_oos[a] *= (1.0 + row[f'ret_{a}'])
        nav = sum(eq_val_oos.values())
        if dt in rebalance_check_dates_oos:
            trade_vol = sum(abs(nav / len(assets) - eq_val_oos[a]) for a in assets)
            cost = trade_vol * 0.0005
            nav -= cost
            eq_val_oos = {a: nav / len(assets) for a in assets}
        eq_oos_history.append({'trade_date': dt, 'nav': nav})
    df_eq_oos = pd.DataFrame(eq_oos_history).set_index('trade_date')
    
    # Generate Plots
    # Plot 1: In-Sample Curves
    plt.figure(figsize=(14, 8))
    plt.subplot(2, 1, 1)
    plt.plot(df_nav_is.index, df_nav_is['nav'] / 1e6, label='Risk Parity (VT + Timing + Put)', color='#1e88e5', linewidth=2.0)
    plt.plot(df_eq_is.index, df_eq_is['nav'] / 1e6, label='8-Asset Equal Weight', color='#43a047', alpha=0.7, linestyle='--')
    plt.plot(hs300_is.index, hs300_is / 1e6, label='HS300 Buy & Hold', color='#757575', alpha=0.5)
    plt.title("In-Sample Performance: 8-Asset Risk Parity Strategy (2015-01-01 to 2024-02-05)", fontsize=12, fontweight='bold')
    plt.ylabel("Normalized NAV")
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)
    
    # In-Sample Drawdowns
    plt.subplot(2, 1, 2)
    dd_rp = (df_nav_is['nav'] - df_nav_is['nav'].cummax()) / df_nav_is['nav'].cummax() * 100.0
    dd_eq = (df_eq_is['nav'] - df_eq_is['nav'].cummax()) / df_eq_is['nav'].cummax() * 100.0
    dd_hs = (hs300_is - hs300_is.cummax()) / hs300_is.cummax() * 100.0
    plt.fill_between(dd_rp.index, dd_rp, 0, label='Risk Parity Drawdown', color='#1e88e5', alpha=0.3)
    plt.plot(dd_eq.index, dd_eq, label='Equal Weight Drawdown', color='#43a047', alpha=0.5, linestyle='--')
    plt.plot(dd_hs.index, dd_hs, label='HS300 Drawdown', color='#757575', alpha=0.3)
    plt.ylabel("Drawdown (%)")
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'nav_risk_parity_is.png'), dpi=300)
    plt.close()
    
    # Plot 2: Out-of-Sample Curves
    plt.figure(figsize=(14, 8))
    plt.subplot(2, 1, 1)
    plt.plot(df_nav_oos.index, df_nav_oos['nav'] / 1e6, label='Risk Parity (VT + Timing + Put)', color='#1e88e5', linewidth=2.0)
    plt.plot(df_eq_oos.index, df_eq_oos['nav'] / 1e6, label='8-Asset Equal Weight', color='#43a047', alpha=0.7, linestyle='--')
    plt.plot(hs300_oos.index, hs300_oos / 1e6, label='HS300 Buy & Hold', color='#757575', alpha=0.5)
    plt.title("Out-of-Sample Performance (Blind Test): 8-Asset Risk Parity Strategy (2024-02-06 to 2026-03-13)", fontsize=12, fontweight='bold')
    plt.ylabel("Normalized NAV")
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)
    
    # Out-of-Sample Drawdowns
    plt.subplot(2, 1, 2)
    dd_rp_oos = (df_nav_oos['nav'] - df_nav_oos['nav'].cummax()) / df_nav_oos['nav'].cummax() * 100.0
    dd_eq_oos = (df_eq_oos['nav'] - df_eq_oos['nav'].cummax()) / df_eq_oos['nav'].cummax() * 100.0
    dd_hs_oos = (hs300_oos - hs300_oos.cummax()) / hs300_oos.cummax() * 100.0
    plt.fill_between(dd_rp_oos.index, dd_rp_oos, 0, label='Risk Parity Drawdown', color='#1e88e5', alpha=0.3)
    plt.plot(dd_eq_oos.index, dd_eq_oos, label='Equal Weight Drawdown', color='#43a047', alpha=0.5, linestyle='--')
    plt.plot(dd_hs_oos.index, dd_hs_oos, label='HS300 Drawdown', color='#757575', alpha=0.3)
    plt.ylabel("Drawdown (%)")
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'nav_risk_parity_oos.png'), dpi=300)
    plt.close()
    
    # Save final results summary
    metrics_rp_is = compute_metrics(df_nav_is['nav'])
    metrics_eq_is = compute_metrics(df_eq_is['nav'])
    metrics_hs_is = compute_metrics(hs300_is)
    
    metrics_rp_oos = compute_metrics(df_nav_oos['nav'])
    metrics_eq_oos = compute_metrics(df_eq_oos['nav'])
    metrics_hs_oos = compute_metrics(hs300_oos)
    
    summary = pd.DataFrame([
        {'Period': 'In-Sample', 'Strategy': 'Risk Parity (VT+Timing+Put)', 'CAGR': f"{metrics_rp_is['CAGR']:.2%}", 'Volatility': f"{metrics_rp_is['Volatility']:.2%}", 'Sharpe': f"{metrics_rp_is['Sharpe']:.2f}", 'Max Drawdown': f"{metrics_rp_is['Max Drawdown']:.2%}", 'Calmar': f"{metrics_rp_is['Calmar']:.2f}"},
        {'Period': 'In-Sample', 'Strategy': '8-Asset Equal Weight', 'CAGR': f"{metrics_eq_is['CAGR']:.2%}", 'Volatility': f"{metrics_eq_is['Volatility']:.2%}", 'Sharpe': f"{metrics_eq_is['Sharpe']:.2f}", 'Max Drawdown': f"{metrics_eq_is['Max Drawdown']:.2%}", 'Calmar': f"{metrics_eq_is['Calmar']:.2f}"},
        {'Period': 'In-Sample', 'Strategy': 'HS300 Buy & Hold', 'CAGR': f"{metrics_hs_is['CAGR']:.2%}", 'Volatility': f"{metrics_hs_is['Volatility']:.2%}", 'Sharpe': f"{metrics_hs_is['Sharpe']:.2f}", 'Max Drawdown': f"{metrics_hs_is['Max Drawdown']:.2%}", 'Calmar': f"{metrics_hs_is['Calmar']:.2f}"},
        {'Period': 'Out-of-Sample', 'Strategy': 'Risk Parity (VT+Timing+Put)', 'CAGR': f"{metrics_rp_oos['CAGR']:.2%}", 'Volatility': f"{metrics_rp_oos['Volatility']:.2%}", 'Sharpe': f"{metrics_rp_oos['Sharpe']:.2f}", 'Max Drawdown': f"{metrics_rp_oos['Max Drawdown']:.2%}", 'Calmar': f"{metrics_rp_oos['Calmar']:.2f}"},
        {'Period': 'Out-of-Sample', 'Strategy': '8-Asset Equal Weight', 'CAGR': f"{metrics_eq_oos['CAGR']:.2%}", 'Volatility': f"{metrics_eq_oos['Volatility']:.2%}", 'Sharpe': f"{metrics_eq_oos['Sharpe']:.2f}", 'Max Drawdown': f"{metrics_eq_oos['Max Drawdown']:.2%}", 'Calmar': f"{metrics_eq_oos['Calmar']:.2f}"},
        {'Period': 'Out-of-Sample', 'Strategy': 'HS300 Buy & Hold', 'CAGR': f"{metrics_hs_oos['CAGR']:.2%}", 'Volatility': f"{metrics_hs_oos['Volatility']:.2%}", 'Sharpe': f"{metrics_hs_oos['Sharpe']:.2f}", 'Max Drawdown': f"{metrics_hs_oos['Max Drawdown']:.2%}", 'Calmar': f"{metrics_hs_oos['Calmar']:.2f}"},
    ])
    
    summary.to_csv(os.path.join(RESULTS_DIR, 'risk_parity_final_comparison.csv'), index=False)
    print("\n" + "="*80)
    print("  FINAL COMPARISON METRICS")
    print("="*80)
    print(summary.to_string(index=False))
    print("="*80)
    print("\nStrategy run successful. Metrics and charts saved to results/ folder.")

if __name__ == "__main__":
    main()

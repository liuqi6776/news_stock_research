import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Define paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(PROJECT_DIR, 'data')
RESULTS_DIR = os.path.join(PROJECT_DIR, 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

def load_data_with_erp(ma_window, val_window):
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
    # We do it by merging on trade_date
    df_300_val = pd.merge(df_300_val, df_bond_yield, on='trade_date', how='left').ffill()
    df_500_val = pd.merge(df_500_val, df_bond_yield, on='trade_date', how='left').ffill()
    df_chinext_val = pd.merge(df_chinext_val, df_bond_yield, on='trade_date', how='left').ffill()
    df_sse50_val = pd.merge(df_sse50_val, df_bond_yield, on='trade_date', how='left').ffill()
    
    # Calculate MAs on prices
    df_300_price['ma'] = df_300_price['close'].rolling(ma_window).mean()
    df_500_price['ma'] = df_500_price['close'].rolling(ma_window).mean()
    df_chinext_price['ma'] = df_chinext_price['close'].rolling(ma_window).mean()
    df_div_price['ma'] = df_div_price['close'].rolling(ma_window).mean()
    df_gold_price['ma'] = df_gold_price['close'].rolling(ma_window).mean()
    df_nasdaq_price['ma'] = df_nasdaq_price['close'].rolling(ma_window).mean()
    
    # Calculate standard PE/PB average valuation quantile (Baseline Valuation Factor)
    for val_df in [df_300_val, df_500_val, df_chinext_val, df_sse50_val]:
        val_df['pe_q'] = val_df['pe_ttm'].rolling(window=val_window, min_periods=250).rank(pct=True)
        val_df['pb_q'] = val_df['pb'].rolling(window=val_window, min_periods=250).rank(pct=True)
        val_df['val_q_pepb'] = (val_df['pe_q'] + val_df['pb_q']) / 2.0
        
        # Calculate ERP: ERP = 100.0 / PE_ttm - Yield_10y
        val_df['erp'] = 100.0 / val_df['pe_ttm'] - val_df['yield_10y']
        val_df['erp_rank_q'] = val_df['erp'].rolling(window=val_window, min_periods=250).rank(pct=True)
        # Cheapness score: lower is cheap (highly undervalued). So val_q = 1.0 - erp_rank_q
        val_df['val_q_erp'] = 1.0 - val_df['erp_rank_q']
        
    # Merge price and valuations
    m300 = pd.merge(df_300_price[['trade_date', 'close', 'pct_chg', 'ma']], 
                    df_300_val[['trade_date', 'val_q_pepb', 'val_q_erp']], on='trade_date', how='inner')
    m500 = pd.merge(df_500_price[['trade_date', 'close', 'pct_chg', 'ma']], 
                    df_500_val[['trade_date', 'val_q_pepb', 'val_q_erp']], on='trade_date', how='inner')
    mchinext = pd.merge(df_chinext_price[['trade_date', 'close', 'pct_chg', 'ma']], 
                        df_chinext_val[['trade_date', 'val_q_pepb', 'val_q_erp']], on='trade_date', how='inner')
    mdiv = pd.merge(df_div_price[['trade_date', 'close', 'pct_chg', 'ma']], 
                    df_sse50_val[['trade_date', 'val_q_pepb', 'val_q_erp']], on='trade_date', how='inner')
    
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
            'close_300': row300['close'], 'ret_300': row300['pct_chg'] / 100.0, 'ma_300': row300['ma'], 
            'val_q_pepb_300': row300['val_q_pepb'], 'val_q_erp_300': row300['val_q_erp'],
            
            'close_500': row500['close'], 'ret_500': row500['pct_chg'] / 100.0, 'ma_500': row500['ma'], 
            'val_q_pepb_500': row500['val_q_pepb'], 'val_q_erp_500': row500['val_q_erp'],
            
            'close_chinext': rowchinext['close'], 'ret_chinext': rowchinext['pct_chg'] / 100.0, 'ma_chinext': rowchinext['ma'], 
            'val_q_pepb_chinext': rowchinext['val_q_pepb'], 'val_q_erp_chinext': rowchinext['val_q_erp'],
            
            'close_div': rowdiv['close'], 'ret_div': rowdiv['pct_chg'] / 100.0, 'ma_div': rowdiv['ma'], 
            'val_q_pepb_div': rowdiv['val_q_pepb'], 'val_q_erp_div': rowdiv['val_q_erp'],
            
            'close_gold': rowgold['close'], 'ret_gold': rowgold['pct_chg'] / 100.0, 'ma_gold': rowgold['ma'],
            'close_nasdaq': rownasdaq['close'], 'ret_nasdaq': rownasdaq['pct_chg'] / 100.0, 'ma_nasdaq': rownasdaq['ma'],
            'ret_bond': bond_ret
        })
        
    df_unified = pd.DataFrame(rows).sort_values('trade_date').reset_index(drop=True)
    return df_unified

def run_backtest_extended(df_period, valuation_mode='erp', val_coeff=0.4, q_threshold=0.20, dev_threshold=0.10, initial_capital=1000000.0):
    if len(df_period) == 0:
        return None
        
    df_period = df_period.copy().reset_index(drop=True)
    df_period['year_week'] = df_period['trade_date'].dt.strftime('%Y-%U')
    rebalance_check_dates = set(df_period.groupby('year_week')['trade_date'].first())
    
    val_300 = 0.0
    val_500 = 0.0
    val_chinext = 0.0
    val_div = 0.0
    val_gold = 0.0
    val_nasdaq = 0.0
    val_bond = initial_capital
    
    nav_history = []
    
    val_q_col_300 = 'val_q_erp_300' if valuation_mode == 'erp' else 'val_q_pepb_300'
    val_q_col_500 = 'val_q_erp_500' if valuation_mode == 'erp' else 'val_q_pepb_500'
    val_q_col_chinext = 'val_q_erp_chinext' if valuation_mode == 'erp' else 'val_q_pepb_chinext'
    val_q_col_div = 'val_q_erp_div' if valuation_mode == 'erp' else 'val_q_pepb_div'
    
    for idx, row in df_period.iterrows():
        dt = row['trade_date']
        
        if idx > 0:
            val_300 *= (1.0 + row['ret_300'])
            val_500 *= (1.0 + row['ret_500'])
            val_chinext *= (1.0 + row['ret_chinext'])
            val_div *= (1.0 + row['ret_div'])
            val_gold *= (1.0 + row['ret_gold'])
            val_nasdaq *= (1.0 + row['ret_nasdaq'])
            val_bond *= (1.0 + row['ret_bond'])
            
        nav = val_300 + val_500 + val_chinext + val_div + val_gold + val_nasdaq + val_bond
        
        if dt in rebalance_check_dates:
            trend_300 = row['close_300'] >= row['ma_300'] if not pd.isna(row['ma_300']) else False
            trend_500 = row['close_500'] >= row['ma_500'] if not pd.isna(row['ma_500']) else False
            trend_chinext = row['close_chinext'] >= row['ma_chinext'] if not pd.isna(row['ma_chinext']) else False
            trend_div = row['close_div'] >= row['ma_div'] if not pd.isna(row['ma_div']) else False
            trend_gold = row['close_gold'] >= row['ma_gold'] if not pd.isna(row['ma_gold']) else False
            trend_nasdaq = row['close_nasdaq'] >= row['ma_nasdaq'] if not pd.isna(row['ma_nasdaq']) else False
            
            q_300 = row[val_q_col_300]
            q_500 = row[val_q_col_500]
            q_chinext = row[val_q_col_chinext]
            q_div = row[val_q_col_div]
            
            w_val_300 = val_coeff * (1.0 - q_300) if not pd.isna(q_300) else 0.0
            w_val_500 = val_coeff * (1.0 - q_500) if not pd.isna(q_500) else 0.0
            w_val_chinext = val_coeff * (1.0 - q_chinext) if not pd.isna(q_chinext) else 0.0
            w_val_div = val_coeff * (1.0 - q_div) if not pd.isna(q_div) else 0.0
            
            w_target_300 = w_val_300 if (q_300 <= q_threshold) else (w_val_300 if trend_300 else w_val_300 * 0.5)
            w_target_500 = w_val_500 if (q_500 <= q_threshold) else (w_val_500 if trend_500 else w_val_500 * 0.5)
            w_target_chinext = w_val_chinext if (q_chinext <= q_threshold) else (w_val_chinext if trend_chinext else w_val_chinext * 0.5)
            w_target_div = w_val_div if (q_div <= q_threshold) else (w_val_div if trend_div else w_val_div * 0.5)
            
            w_target_gold = 0.10 if trend_gold else 0.0
            w_target_nasdaq = 0.10 if trend_nasdaq else 0.0
            
            total_eq = w_target_300 + w_target_500 + w_target_chinext + w_target_div + w_target_gold + w_target_nasdaq
            if total_eq > 1.0:
                w_target_300 /= total_eq
                w_target_500 /= total_eq
                w_target_chinext /= total_eq
                w_target_div /= total_eq
                w_target_gold /= total_eq
                w_target_nasdaq /= total_eq
                w_target_bond = 0.0
            else:
                w_target_bond = 1.0 - total_eq
                
            w_curr_300 = val_300 / nav if nav > 0 else 0.0
            w_curr_500 = val_500 / nav if nav > 0 else 0.0
            w_curr_chinext = val_chinext / nav if nav > 0 else 0.0
            w_curr_div = val_div / nav if nav > 0 else 0.0
            w_curr_gold = val_gold / nav if nav > 0 else 0.0
            w_curr_nasdaq = val_nasdaq / nav if nav > 0 else 0.0
            
            devs = [
                abs(w_curr_300 - w_target_300),
                abs(w_curr_500 - w_target_500),
                abs(w_curr_chinext - w_target_chinext),
                abs(w_curr_div - w_target_div),
                abs(w_curr_gold - w_target_gold),
                abs(w_curr_nasdaq - w_target_nasdaq)
            ]
            
            if any(d > dev_threshold for d in devs) or idx == 0:
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
                cost = trade_vol * 0.0005
                
                nav -= cost
                val_300 = nav * w_target_300
                val_500 = nav * w_target_500
                val_chinext = nav * w_target_chinext
                val_div = nav * w_target_div
                val_gold = nav * w_target_gold
                val_nasdaq = nav * w_target_nasdaq
                val_bond = nav * w_target_bond
                
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
    print("Running Valuation Factor Upgrade Step (PE/PB vs. ERP)...")
    
    # Use Robust Parameters
    ma, val_w, q, coeff = 200, 1400, 0.20, 0.4
    
    print("Loading data and computing indicators...")
    df_unified = load_data_with_erp(ma, val_w)
    
    # Split Dates
    is_start, is_end = "2015-01-01", "2024-02-05"
    oos_start, oos_end = "2024-02-06", "2026-03-13"
    
    # Filter by dates
    df_is = df_unified[(df_unified['trade_date'] >= pd.to_datetime(is_start)) & (df_unified['trade_date'] <= pd.to_datetime(is_end))].copy()
    df_oos = df_unified[(df_unified['trade_date'] >= pd.to_datetime(oos_start)) & (df_unified['trade_date'] <= pd.to_datetime(oos_end))].copy()
    
    # ------------------ IN-SAMPLE BACKTESTS ------------------
    print("\n--- Running In-Sample Backtests (2015-01-01 to 2024-02-05) ---")
    nav_pepb_is = run_backtest_extended(df_is, valuation_mode='pepb', val_coeff=coeff, q_threshold=q)
    nav_erp_is = run_backtest_extended(df_is, valuation_mode='erp', val_coeff=coeff, q_threshold=q)
    
    metrics_pepb_is = compute_metrics(nav_pepb_is['nav'])
    metrics_erp_is = compute_metrics(nav_erp_is['nav'])
    
    # Benchmark S&P 300
    hs300_nav_is = (1.0 + df_is['ret_300']).cumprod() * 1000000.0
    hs300_nav_is.index = df_is['trade_date']
    metrics_hs300_is = compute_metrics(hs300_nav_is)
    
    print("\n=== IN-SAMPLE VALUATION UPGRADE COMPARISON ===")
    summary_is = pd.DataFrame([
        {'Strategy': '6-Asset PE/PB Valuation', **metrics_pepb_is},
        {'Strategy': '6-Asset ERP Valuation (Upgraded)', **metrics_erp_is},
        {'Strategy': 'HS300 Buy & Hold', **metrics_hs300_is}
    ])
    print(summary_is.to_string(index=False))
    
    # ------------------ OUT-OF-SAMPLE BACKTESTS ------------------
    print("\n--- Running Out-of-Sample Backtests (2024-02-06 to 2026-03-13) ---")
    nav_pepb_oos = run_backtest_extended(df_oos, valuation_mode='pepb', val_coeff=coeff, q_threshold=q, initial_capital=1000000.0)
    nav_erp_oos = run_backtest_extended(df_oos, valuation_mode='erp', val_coeff=coeff, q_threshold=q, initial_capital=1000000.0)
    
    metrics_pepb_oos = compute_metrics(nav_pepb_oos['nav'])
    metrics_erp_oos = compute_metrics(nav_erp_oos['nav'])
    
    # Benchmark OOS
    hs300_nav_oos = (1.0 + df_oos['ret_300']).cumprod() * 1000000.0
    hs300_nav_oos.index = df_oos['trade_date']
    metrics_hs300_oos = compute_metrics(hs300_nav_oos)
    
    print("\n=== OUT-OF-SAMPLE VALUATION UPGRADE COMPARISON (BLIND TEST) ===")
    summary_oos = pd.DataFrame([
        {'Strategy': '6-Asset PE/PB Valuation', **metrics_pepb_oos},
        {'Strategy': '6-Asset ERP Valuation (Upgraded)', **metrics_erp_oos},
        {'Strategy': 'HS300 Buy & Hold', **metrics_hs300_oos}
    ])
    print(summary_oos.to_string(index=False))
    
    # Save CSVs
    summary_is.to_csv(os.path.join(RESULTS_DIR, 'erp_upgrade_metrics_is.csv'), index=False)
    summary_oos.to_csv(os.path.join(RESULTS_DIR, 'erp_upgrade_metrics_oos.csv'), index=False)
    
    # Save NAV histories to CSV for step 3 (leverage)
    nav_erp_is.to_csv(os.path.join(RESULTS_DIR, 'nav_erp_is_history.csv'))
    nav_erp_oos.to_csv(os.path.join(RESULTS_DIR, 'nav_erp_oos_history.csv'))
    
    # Plot OOS Curves
    plt.figure(figsize=(12, 6))
    plt.plot(nav_pepb_oos.index, nav_pepb_oos['nav'] / 1e6, label='6-Asset PE/PB Valuation', color='#f4511e', linestyle='--')
    plt.plot(nav_erp_oos.index, nav_erp_oos['nav'] / 1e6, label='6-Asset ERP Valuation (Upgraded)', color='#1e88e5', linewidth=2.5)
    plt.plot(hs300_nav_oos.index, hs300_nav_oos / 1e6, label='HS300 Buy & Hold', color='#757575', alpha=0.5)
    
    plt.title("Out-of-Sample Performance Upgrade: Valuation Factor to ERP (2024-02-06 to 2026-03-13)", fontsize=12, fontweight='bold')
    plt.xlabel("Date")
    plt.ylabel("NAV (Normalized)")
    plt.legend(loc='upper left')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'nav_erp_upgrade_comparison_oos.png'), dpi=300)
    plt.close()
    
    # Plot IS Curves
    plt.figure(figsize=(12, 6))
    plt.plot(nav_pepb_is.index, nav_pepb_is['nav'] / 1e6, label='6-Asset PE/PB Valuation', color='#f4511e', linestyle='--')
    plt.plot(nav_erp_is.index, nav_erp_is['nav'] / 1e6, label='6-Asset ERP Valuation (Upgraded)', color='#1e88e5', linewidth=2.5)
    plt.plot(hs300_nav_is.index, hs300_nav_is / 1e6, label='HS300 Buy & Hold', color='#757575', alpha=0.5)
    
    plt.title("In-Sample Performance Upgrade: Valuation Factor to ERP (2015-01-01 to 2024-02-05)", fontsize=12, fontweight='bold')
    plt.xlabel("Date")
    plt.ylabel("NAV (Normalized)")
    plt.legend(loc='upper left')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'nav_erp_upgrade_comparison_is.png'), dpi=300)
    plt.close()
    
    print("\nBacktest complete. Metrics saved and charts saved in results/ directory.")

if __name__ == "__main__":
    main()

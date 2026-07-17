import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Set style
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(PROJECT_DIR, 'data')
RESULTS_DIR = os.path.join(PROJECT_DIR, 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

def load_all_data(ma_window=250, val_window=1200):
    print("Loading all historical data files...")
    # Load A-share price and valuations
    df_1000_price = pd.read_csv(os.path.join(DATA_DIR, 'zz1000_daily.csv'))
    df_1000_val = pd.read_csv(os.path.join(DATA_DIR, 'zz1000_valuation.csv'))
    
    df_300_price = pd.read_csv(os.path.join(DATA_DIR, 'hs300_daily.csv'))
    df_300_val = pd.read_csv(os.path.join(DATA_DIR, 'hs300_valuation.csv'))
    
    df_500_price = pd.read_csv(os.path.join(DATA_DIR, 'zz500_daily.csv'))
    df_500_val = pd.read_csv(os.path.join(DATA_DIR, 'zz500_valuation.csv'))
    
    df_chinext_price = pd.read_csv(os.path.join(DATA_DIR, 'chinext_daily.csv'))
    df_chinext_val = pd.read_csv(os.path.join(DATA_DIR, 'chinext_valuation.csv'))
    
    df_div_price = pd.read_csv(os.path.join(DATA_DIR, 'div_low_vol_daily.csv'))
    df_sse50_val = pd.read_csv(os.path.join(DATA_DIR, 'sse50_valuation.csv'))
    
    # Load Nasdaq, Gold, Bond
    df_nasdaq_price = pd.read_csv(os.path.join(DATA_DIR, 'nasdaq_etf_daily.csv'))
    df_gold_price = pd.read_csv(os.path.join(DATA_DIR, 'gold_etf_daily.csv'))
    df_bond_price = pd.read_csv(os.path.join(DATA_DIR, 'bond_etf_daily.csv'))
    
    dfs = [df_1000_price, df_1000_val, df_300_price, df_300_val, df_500_price, df_500_val,
           df_chinext_price, df_chinext_val, df_div_price, df_sse50_val,
           df_nasdaq_price, df_gold_price, df_bond_price]
           
    for df in dfs:
        df['trade_date'] = pd.to_datetime(df['trade_date'].astype(str))
        df.sort_values('trade_date', inplace=True)
        df.reset_index(drop=True, inplace=True)
        
    # Nasdaq split adjustment on 2022-01-14
    split_date = pd.to_datetime('2022-01-14')
    if df_nasdaq_price['trade_date'].min() < split_date:
        adj_factor = 1.038 / 5.192
        mask = df_nasdaq_price['trade_date'] < split_date
        for col in ['close', 'open', 'high', 'low', 'pre_close']:
            if col in df_nasdaq_price.columns:
                df_nasdaq_price.loc[mask, col] *= adj_factor
                
    # Calculate MAs and PE/PB rolling quantiles
    for df_p in [df_1000_price, df_300_price, df_500_price, df_chinext_price, df_div_price, df_nasdaq_price, df_gold_price]:
        df_p['ma'] = df_p['close'].rolling(ma_window).mean()
        
    for df_v in [df_1000_val, df_300_val, df_500_val, df_chinext_val, df_sse50_val]:
        df_v['pe_q'] = df_v['pe_ttm'].rolling(window=val_window, min_periods=250).rank(pct=True)
        df_v['pb_q'] = df_v['pb'].rolling(window=val_window, min_periods=250).rank(pct=True)
        df_v['val_q'] = (df_v['pe_q'] + df_v['pb_q']) / 2.0
        
    # Merge price and valuations
    m1000 = pd.merge(df_1000_price[['trade_date', 'close', 'pct_chg', 'ma']], df_1000_val[['trade_date', 'val_q']], on='trade_date', how='inner')
    m300 = pd.merge(df_300_price[['trade_date', 'close', 'pct_chg', 'ma']], df_300_val[['trade_date', 'val_q']], on='trade_date', how='inner')
    m500 = pd.merge(df_500_price[['trade_date', 'close', 'pct_chg', 'ma']], df_500_val[['trade_date', 'val_q']], on='trade_date', how='inner')
    mchinext = pd.merge(df_chinext_price[['trade_date', 'close', 'pct_chg', 'ma']], df_chinext_val[['trade_date', 'val_q']], on='trade_date', how='inner')
    mdiv = pd.merge(df_div_price[['trade_date', 'close', 'pct_chg', 'ma']], df_sse50_val[['trade_date', 'val_q']], on='trade_date', how='inner')
    
    # Nasdaq, Gold, Bond returns mapping
    df_nasdaq_price['nasdaq_ret'] = df_nasdaq_price['pct_chg'] / 100.0
    df_gold_price['gold_ret'] = df_gold_price['pct_chg'] / 100.0
    df_bond_price['bond_ret'] = df_bond_price['pct_chg'] / 100.0
    
    nasdaq_ret_map = df_nasdaq_price.set_index('trade_date')['nasdaq_ret'].to_dict()
    gold_ret_map = df_gold_price.set_index('trade_date')['gold_ret'].to_dict()
    bond_ret_map = df_bond_price.set_index('trade_date')['bond_ret'].to_dict()
    
    nasdaq_close_map = df_nasdaq_price.set_index('trade_date')['close'].to_dict()
    nasdaq_ma_map = df_nasdaq_price.set_index('trade_date')['ma'].to_dict()
    
    gold_close_map = df_gold_price.set_index('trade_date')['close'].to_dict()
    gold_ma_map = df_gold_price.set_index('trade_date')['ma'].to_dict()
    
    # Align on index dates
    trading_dates = m1000['trade_date'].tolist()
    
    m1000_dict = m1000.set_index('trade_date').to_dict(orient='index')
    m300_dict = m300.set_index('trade_date').to_dict(orient='index')
    m500_dict = m500.set_index('trade_date').to_dict(orient='index')
    mchinext_dict = mchinext.set_index('trade_date').to_dict(orient='index')
    mdiv_dict = mdiv.set_index('trade_date').to_dict(orient='index')
    
    rows = []
    for dt in trading_dates:
        r1000 = m1000_dict.get(dt)
        r300 = m300_dict.get(dt)
        r500 = m500_dict.get(dt)
        rchinext = mchinext_dict.get(dt)
        rdiv = mdiv_dict.get(dt)
        
        # Check if all exist to build unified comparison frame
        if any(r is None for r in [r1000, r300, r500, rchinext, rdiv]):
            continue
            
        b_ret = bond_ret_map.get(dt, 0.03 / 242.0)
        if pd.isna(b_ret):
            b_ret = 0.03 / 242.0
            
        rows.append({
            'trade_date': dt,
            # CSI 1000
            'close_1000': r1000['close'], 'ret_1000': r1000['pct_chg'] / 100.0, 'ma_1000': r1000['ma'], 'val_q_1000': r1000['val_q'],
            # Baseline Assets
            'close_300': r300['close'], 'ret_300': r300['pct_chg'] / 100.0, 'ma_300': r300['ma'], 'val_q_300': r300['val_q'],
            'close_500': r500['close'], 'ret_500': r500['pct_chg'] / 100.0, 'ma_500': r500['ma'], 'val_q_500': r500['val_q'],
            'close_chinext': rchinext['close'], 'ret_chinext': rchinext['pct_chg'] / 100.0, 'ma_chinext': rchinext['ma'], 'val_q_chinext': rchinext['val_q'],
            'close_div': rdiv['close'], 'ret_div': rdiv['pct_chg'] / 100.0, 'ma_div': rdiv['ma'], 'val_q_div': rdiv['val_q'],
            # Nasdaq, Gold, Bond
            'close_nasdaq': nasdaq_close_map.get(dt, np.nan), 'ret_nasdaq': nasdaq_ret_map.get(dt, 0.0), 'ma_nasdaq': nasdaq_ma_map.get(dt, np.nan),
            'close_gold': gold_close_map.get(dt, np.nan), 'ret_gold': gold_ret_map.get(dt, 0.0), 'ma_gold': gold_ma_map.get(dt, np.nan),
            'ret_bond': b_ret
        })
        
    df_unified = pd.DataFrame(rows).sort_values('trade_date').reset_index(drop=True)
    print(f"Unified dataset built. Shape: {df_unified.shape}, Dates: {df_unified['trade_date'].min().strftime('%Y-%m-%d')} to {df_unified['trade_date'].max().strftime('%Y-%m-%d')}")
    return df_unified

def run_baseline_2asset(df_period, w_300=0.40, w_500=0.40, w_bond=0.20, dev_threshold=0.10, initial_capital=1000000.0):
    df_period = df_period.copy().reset_index(drop=True)
    df_period['year_week'] = df_period['trade_date'].dt.strftime('%Y-%U')
    rebalance_check_dates = set(df_period.groupby('year_week')['trade_date'].first())
    
    val_300 = 0.0
    val_500 = 0.0
    val_bond = initial_capital
    
    nav_history = []
    for idx, row in df_period.iterrows():
        dt = row['trade_date']
        
        if idx > 0:
            val_300 *= (1.0 + row['ret_300'])
            val_500 *= (1.0 + row['ret_500'])
            val_bond *= (1.0 + row['ret_bond'])
            
        nav = val_300 + val_500 + val_bond
        
        if dt in rebalance_check_dates or idx == 0:
            w_curr_300 = val_300 / nav if nav > 0 else 0.0
            w_curr_500 = val_500 / nav if nav > 0 else 0.0
            w_curr_bond = val_bond / nav if nav > 0 else 0.0
            
            devs = [abs(w_curr_300 - w_300), abs(w_curr_500 - w_500), abs(w_curr_bond - w_bond)]
            if any(d > dev_threshold for d in devs) or idx == 0:
                val_target_300 = nav * w_300
                val_target_500 = nav * w_500
                val_target_bond = nav * w_bond
                
                trade_vol = abs(val_target_300 - val_300) + abs(val_target_500 - val_500) + abs(val_target_bond - val_bond)
                cost = trade_vol * 0.0005
                
                nav -= cost
                val_300 = nav * w_300
                val_500 = nav * w_500
                val_bond = nav * w_bond
                
        nav_history.append({'trade_date': dt, 'nav': nav})
    return pd.DataFrame(nav_history).set_index('trade_date')['nav']

def run_baseline_6asset(df_period, val_coeff=0.4, q_threshold=0.20, dev_threshold=0.10, initial_capital=1000000.0):
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
        
        if dt in rebalance_check_dates or idx == 0:
            trend_300 = row['close_300'] >= row['ma_300'] if not pd.isna(row['ma_300']) else False
            trend_500 = row['close_500'] >= row['ma_500'] if not pd.isna(row['ma_500']) else False
            trend_chinext = row['close_chinext'] >= row['ma_chinext'] if not pd.isna(row['ma_chinext']) else False
            trend_div = row['close_div'] >= row['ma_div'] if not pd.isna(row['ma_div']) else False
            trend_gold = row['close_gold'] >= row['ma_gold'] if not pd.isna(row['ma_gold']) else False
            trend_nasdaq = row['close_nasdaq'] >= row['ma_nasdaq'] if not pd.isna(row['ma_nasdaq']) else False
            
            w_val_300 = val_coeff * (1.0 - row['val_q_300']) if not pd.isna(row['val_q_300']) else 0.0
            w_val_500 = val_coeff * (1.0 - row['val_q_500']) if not pd.isna(row['val_q_500']) else 0.0
            w_val_chinext = val_coeff * (1.0 - row['val_q_chinext']) if not pd.isna(row['val_q_chinext']) else 0.0
            w_val_div = val_coeff * (1.0 - row['val_q_div']) if not pd.isna(row['val_q_div']) else 0.0
            
            w_target_300 = w_val_300 if (row['val_q_300'] <= q_threshold) else (w_val_300 if trend_300 else w_val_300 * 0.5)
            w_target_500 = w_val_500 if (row['val_q_500'] <= q_threshold) else (w_val_500 if trend_500 else w_val_500 * 0.5)
            w_target_chinext = w_val_chinext if (row['val_q_chinext'] <= q_threshold) else (w_val_chinext if trend_chinext else w_val_chinext * 0.5)
            w_target_div = w_val_div if (row['val_q_div'] <= q_threshold) else (w_val_div if trend_div else w_val_div * 0.5)
            
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
            w_curr_bond = val_bond / nav if nav > 0 else 0.0
            
            devs = [
                abs(w_curr_300 - w_target_300), abs(w_curr_500 - w_target_500),
                abs(w_curr_chinext - w_target_chinext), abs(w_curr_div - w_target_div),
                abs(w_curr_gold - w_target_gold), abs(w_curr_nasdaq - w_target_nasdaq),
                abs(w_curr_bond - w_target_bond)
            ]
            
            if any(d > dev_threshold for d in devs) or idx == 0:
                val_target_300 = nav * w_target_300
                val_target_500 = nav * w_target_500
                val_target_chinext = nav * w_target_chinext
                val_target_div = nav * w_target_div
                val_target_gold = nav * w_target_gold
                val_target_nasdaq = nav * w_target_nasdaq
                val_target_bond = nav * w_target_bond
                
                trade_vol = (abs(val_target_300 - val_300) + abs(val_target_500 - val_500) + 
                             abs(val_target_chinext - val_chinext) + abs(val_target_div - val_div) + 
                             abs(val_target_gold - val_gold) + abs(val_target_nasdaq - val_nasdaq) + 
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
    return pd.DataFrame(nav_history).set_index('trade_date')['nav']

def run_zz1000_enhanced_bh(df_period, alpha_annual=0.11, initial_capital=1000000.0):
    df_period = df_period.copy().reset_index(drop=True)
    alpha_daily = (1.0 + alpha_annual) ** (1.0 / 242.0) - 1.0
    df_period['ret_enhanced'] = (1.0 + df_period['ret_1000']) * (1.0 + alpha_daily) - 1.0
    
    nav = initial_capital
    nav_history = []
    for idx, row in df_period.iterrows():
        if idx > 0:
            nav *= (1.0 + row['ret_enhanced'])
        nav_history.append({'trade_date': row['trade_date'], 'nav': nav})
    return pd.DataFrame(nav_history).set_index('trade_date')['nav']

def run_zz1000_bh(df_period, initial_capital=1000000.0):
    df_period = df_period.copy().reset_index(drop=True)
    nav = initial_capital
    nav_history = []
    for idx, row in df_period.iterrows():
        if idx > 0:
            nav *= (1.0 + row['ret_1000'])
        nav_history.append({'trade_date': row['trade_date'], 'nav': nav})
    return pd.DataFrame(nav_history).set_index('trade_date')['nav']

from step13_backtest_premium import run_premium_backtest, compute_metrics

def main():
    # Load unified dataset
    df_unified = load_all_data(ma_window=250, val_window=1200)
    
    # Split dates
    is_start, is_end = "2015-01-01", "2024-02-05"
    oos_start, oos_end = "2024-02-06", "2026-03-13"
    
    df_is = df_unified[(df_unified['trade_date'] >= pd.to_datetime(is_start)) & (df_unified['trade_date'] <= pd.to_datetime(is_end))].copy()
    df_oos = df_unified[(df_unified['trade_date'] >= pd.to_datetime(oos_start)) & (df_unified['trade_date'] <= pd.to_datetime(oos_end))].copy()
    
    # Running all backtests
    print("\nRunning backtests for In-Sample period...")
    nav_premium_is, _ = run_premium_backtest(df_is)
    nav_6asset_is = run_baseline_6asset(df_is)
    nav_2asset_is = run_baseline_2asset(df_is)
    nav_1000enh_is = run_zz1000_enhanced_bh(df_is)
    nav_1000bh_is = run_zz1000_bh(df_is)
    
    print("Running backtests for Out-of-Sample period...")
    nav_premium_oos, _ = run_premium_backtest(df_oos)
    nav_6asset_oos = run_baseline_6asset(df_oos)
    nav_2asset_oos = run_baseline_2asset(df_oos)
    nav_1000enh_oos = run_zz1000_enhanced_bh(df_oos)
    nav_1000bh_oos = run_zz1000_bh(df_oos)
    
    # Save combined NAV data for plotting
    df_nav_all_is = pd.DataFrame({
        'CSI 1000 B&H': nav_1000bh_is,
        'CSI 1000 Enhanced B&H': nav_1000enh_is,
        'Baseline 2-Asset': nav_2asset_is,
        'Baseline 6-Asset': nav_6asset_is,
        'Premium Multi-Asset': nav_premium_is['nav']
    })
    
    df_nav_all_oos = pd.DataFrame({
        'CSI 1000 B&H': nav_1000bh_oos,
        'CSI 1000 Enhanced B&H': nav_1000enh_oos,
        'Baseline 2-Asset': nav_2asset_oos,
        'Baseline 6-Asset': nav_6asset_oos,
        'Premium Multi-Asset': nav_premium_oos['nav']
    })
    
    df_nav_all_is.to_csv(os.path.join(RESULTS_DIR, 'nav_comparison_is.csv'))
    df_nav_all_oos.to_csv(os.path.join(RESULTS_DIR, 'nav_comparison_oos.csv'))
    
    # Compute metrics
    metrics_list = []
    strategies = [
        ('CSI 1000 B&H', nav_1000bh_is, nav_1000bh_oos),
        ('CSI 1000 Enhanced B&H', nav_1000enh_is, nav_1000enh_oos),
        ('Baseline 2-Asset', nav_2asset_is, nav_2asset_oos),
        ('Baseline 6-Asset', nav_6asset_is, nav_6asset_oos),
        ('Premium Multi-Asset', nav_premium_is['nav'], nav_premium_oos['nav'])
    ]
    
    for name, is_nav, oos_nav in strategies:
        m_is = compute_metrics(is_nav)
        m_oos = compute_metrics(oos_nav)
        
        metrics_list.append({
            'Strategy': name,
            'IS_Total_Ret': m_is['Total Return'], 'IS_CAGR': m_is['CAGR'], 'IS_Vol': m_is['Volatility'], 'IS_Sharpe': m_is['Sharpe'], 'IS_MaxDD': m_is['Max Drawdown'], 'IS_Calmar': m_is['Calmar'],
            'OOS_Total_Ret': m_oos['Total Return'], 'OOS_CAGR': m_oos['CAGR'], 'OOS_Vol': m_oos['Volatility'], 'OOS_Sharpe': m_oos['Sharpe'], 'OOS_MaxDD': m_oos['Max Drawdown'], 'OOS_Calmar': m_oos['Calmar']
        })
        
    df_metrics = pd.DataFrame(metrics_list)
    df_metrics.to_csv(os.path.join(RESULTS_DIR, 'metrics_comparison.csv'), index=False)
    
    # Render Markdown table
    print("\n" + "="*80)
    print("                      COMPARATIVE ANALYSIS SUMMARY")
    print("="*80)
    print(f"| Strategy | IS CAGR | IS Sharpe | IS MaxDD | OOS CAGR | OOS Sharpe | OOS MaxDD |")
    print(f"| :--- | :---: | :---: | :---: | :---: | :---: | :---: |")
    for _, r in df_metrics.iterrows():
        print(f"| {r['Strategy']:22} | {r['IS_CAGR']:.2%} | {r['IS_Sharpe']:.2f} | {r['IS_MaxDD']:.2%} | {r['OOS_CAGR']:.2%} | {r['OOS_Sharpe']:.2f} | {r['OOS_MaxDD']:.2%} |")
    print("="*80)
    
    # Generate Plots
    print("Generating performance visualization charts...")
    fig, axes = plt.subplots(2, 2, figsize=(16, 10), gridspec_kw={'height_ratios': [3, 1]})
    
    # Colors
    colors = {
        'CSI 1000 B&H': '#757575',
        'CSI 1000 Enhanced B&H': '#ff1744',
        'Baseline 2-Asset': '#ff9100',
        'Baseline 6-Asset': '#ffd600',
        'Premium Multi-Asset': '#00e676'
    }
    
    # Subplot 0, 0: In-Sample NAV
    for col in df_nav_all_is.columns:
        axes[0, 0].plot(df_nav_all_is.index, df_nav_all_is[col] / 1e6, label=col, color=colors[col], linewidth=1.5 if col != 'Premium Multi-Asset' else 2.5)
    axes[0, 0].set_title('In-Sample Cumulative Wealth (2015 - 2024)', fontsize=14, fontweight='bold')
    axes[0, 0].set_ylabel('Portfolio Value (Millions)', fontsize=12)
    axes[0, 0].legend(loc='upper left')
    
    # Subplot 1, 0: In-Sample Drawdown
    for col in df_nav_all_is.columns:
        nav = df_nav_all_is[col]
        dd = (nav - nav.cummax()) / nav.cummax()
        axes[1, 0].fill_between(df_nav_all_is.index, dd, 0, alpha=0.15, color=colors[col])
        axes[1, 0].plot(df_nav_all_is.index, dd, color=colors[col], linewidth=1)
    axes[1, 0].set_ylabel('Drawdown', fontsize=12)
    axes[1, 0].set_ylim(-0.6, 0.02)
    
    # Subplot 0, 1: Out-of-Sample NAV
    for col in df_nav_all_oos.columns:
        axes[0, 1].plot(df_nav_all_oos.index, df_nav_all_oos[col] / 1e6, label=col, color=colors[col], linewidth=1.5 if col != 'Premium Multi-Asset' else 2.5)
    axes[0, 1].set_title('Out-of-Sample Cumulative Wealth (2024 - 2026)', fontsize=14, fontweight='bold')
    axes[0, 1].legend(loc='upper left')
    
    # Subplot 1, 1: Out-of-Sample Drawdown
    for col in df_nav_all_oos.columns:
        nav = df_nav_all_oos[col]
        dd = (nav - nav.cummax()) / nav.cummax()
        axes[1, 1].fill_between(df_nav_all_oos.index, dd, 0, alpha=0.15, color=colors[col])
        axes[1, 1].plot(df_nav_all_oos.index, dd, color=colors[col], linewidth=1)
    axes[1, 1].set_ylim(-0.4, 0.02)
    
    # Adjust layouts
    for ax in axes.flatten():
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        
    plt.tight_layout()
    plot_path = os.path.join(RESULTS_DIR, 'premium_comparison.png')
    plt.savefig(plot_path, dpi=300)
    plt.savefig(plot_path.replace('.png', '.pdf'), dpi=300)
    plt.close()
    print(f"Comparison plot saved to {plot_path}.")

if __name__ == "__main__":
    main()

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

def load_2asset_data(ma_window, val_window):
    df_300_price = pd.read_csv(os.path.join(DATA_DIR, 'hs300_daily.csv'))
    df_500_price = pd.read_csv(os.path.join(DATA_DIR, 'zz500_daily.csv'))
    df_bond_price = pd.read_csv(os.path.join(DATA_DIR, 'bond_etf_daily.csv'))
    
    df_300_val = pd.read_csv(os.path.join(DATA_DIR, 'hs300_valuation.csv'))
    df_500_val = pd.read_csv(os.path.join(DATA_DIR, 'zz500_valuation.csv'))
    
    for df in [df_300_price, df_500_price, df_bond_price, df_300_val, df_500_val]:
        df['trade_date'] = pd.to_datetime(df['trade_date'].astype(str))
        df.sort_values('trade_date', inplace=True)
        df.reset_index(drop=True, inplace=True)
    
    df_300_price['ma'] = df_300_price['close'].rolling(ma_window).mean()
    df_300_val['pe_q'] = df_300_val['pe_ttm'].rolling(window=val_window, min_periods=250).rank(pct=True)
    df_300_val['pb_q'] = df_300_val['pb'].rolling(window=val_window, min_periods=250).rank(pct=True)
    df_300_val['val_q'] = (df_300_val['pe_q'] + df_300_val['pb_q']) / 2.0
    
    df_500_price['ma'] = df_500_price['close'].rolling(ma_window).mean()
    df_500_val['pe_q'] = df_500_val['pe_ttm'].rolling(window=val_window, min_periods=250).rank(pct=True)
    df_500_val['pb_q'] = df_500_val['pb'].rolling(window=val_window, min_periods=250).rank(pct=True)
    df_500_val['val_q'] = (df_500_val['pe_q'] + df_500_val['pb_q']) / 2.0
    
    m300 = pd.merge(df_300_price[['trade_date', 'close', 'pct_chg', 'ma']], 
                    df_300_val[['trade_date', 'val_q']], on='trade_date', how='inner')
    m500 = pd.merge(df_500_price[['trade_date', 'close', 'pct_chg', 'ma']], 
                    df_500_val[['trade_date', 'val_q']], on='trade_date', how='inner')
    
    df_bond_price['bond_ret'] = df_bond_price['pct_chg'] / 100.0
    bond_map = df_bond_price.set_index('trade_date')['bond_ret'].to_dict()
    
    trading_dates = m300['trade_date'].tolist()
    m300_dict = m300.set_index('trade_date').to_dict(orient='index')
    m500_dict = m500.set_index('trade_date').to_dict(orient='index')
    
    rows = []
    for dt in trading_dates:
        row300 = m300_dict.get(dt)
        row500 = m500_dict.get(dt)
        if row300 is None or row500 is None:
            continue
            
        bond_ret = bond_map.get(dt, 0.03 / 242.0)
        if pd.isna(bond_ret):
            bond_ret = 0.03 / 242.0
            
        rows.append({
            'trade_date': dt,
            'close_300': row300['close'], 'ret_300': row300['pct_chg'] / 100.0, 'ma_300': row300['ma'], 'val_q_300': row300['val_q'],
            'close_500': row500['close'], 'ret_500': row500['pct_chg'] / 100.0, 'ma_500': row500['ma'], 'val_q_500': row500['val_q'],
            'ret_bond': bond_ret
        })
        
    df_unified = pd.DataFrame(rows).sort_values('trade_date').reset_index(drop=True)
    return df_unified

def run_2asset_backtest(df_period, val_coeff=0.6, q_threshold=0.15, dev_threshold=0.10, initial_capital=1000000.0):
    if len(df_period) == 0:
        return None
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
        
        if dt in rebalance_check_dates:
            trend_300 = row['close_300'] >= row['ma_300'] if not pd.isna(row['ma_300']) else False
            trend_500 = row['close_500'] >= row['ma_500'] if not pd.isna(row['ma_500']) else False
            
            w_val_300 = val_coeff * (1.0 - row['val_q_300']) if not pd.isna(row['val_q_300']) else 0.0
            w_val_500 = val_coeff * (1.0 - row['val_q_500']) if not pd.isna(row['val_q_500']) else 0.0
            
            w_target_300 = w_val_300 if (row['val_q_300'] <= q_threshold) else (w_val_300 if trend_300 else w_val_300 * 0.5)
            w_target_500 = w_val_500 if (row['val_q_500'] <= q_threshold) else (w_val_500 if trend_500 else w_val_500 * 0.5)
            
            total_eq = w_target_300 + w_target_500
            if total_eq > 1.0:
                w_target_300 /= total_eq
                w_target_500 /= total_eq
                w_target_bond = 0.0
            else:
                w_target_bond = 1.0 - total_eq
            
            w_curr_300 = val_300 / nav if nav > 0 else 0.0
            w_curr_500 = val_500 / nav if nav > 0 else 0.0
            
            dev_300 = abs(w_curr_300 - w_target_300)
            dev_500 = abs(w_curr_500 - w_target_500)
            
            if dev_300 > dev_threshold or dev_500 > dev_threshold or idx == 0:
                val_target_300 = nav * w_target_300
                val_target_500 = nav * w_target_500
                val_target_bond = nav * w_target_bond
                
                trade_vol = abs(val_target_300 - val_300) + abs(val_target_500 - val_500) + abs(val_target_bond - val_bond)
                cost = trade_vol * 0.0005
                
                nav -= cost
                val_300 = nav * w_target_300
                val_500 = nav * w_target_500
                val_bond = nav * w_target_bond
                
        nav_history.append({'trade_date': dt, 'nav': nav})
    
    return pd.DataFrame(nav_history).set_index('trade_date')

def load_extended_data(ma_window, val_window):
    df_300_price = pd.read_csv(os.path.join(DATA_DIR, 'hs300_daily.csv'))
    df_500_price = pd.read_csv(os.path.join(DATA_DIR, 'zz500_daily.csv'))
    df_chinext_price = pd.read_csv(os.path.join(DATA_DIR, 'chinext_daily.csv'))
    df_div_price = pd.read_csv(os.path.join(DATA_DIR, 'div_low_vol_daily.csv'))
    df_gold_price = pd.read_csv(os.path.join(DATA_DIR, 'gold_etf_daily.csv'))
    df_nasdaq_price = pd.read_csv(os.path.join(DATA_DIR, 'nasdaq_etf_daily.csv'))
    df_bond_price = pd.read_csv(os.path.join(DATA_DIR, 'bond_etf_daily.csv'))
    
    df_300_val = pd.read_csv(os.path.join(DATA_DIR, 'hs300_valuation.csv'))
    df_500_val = pd.read_csv(os.path.join(DATA_DIR, 'zz500_valuation.csv'))
    df_chinext_val = pd.read_csv(os.path.join(DATA_DIR, 'chinext_valuation.csv'))
    df_sse50_val = pd.read_csv(os.path.join(DATA_DIR, 'sse50_valuation.csv'))
    
    dfs = [df_300_price, df_500_price, df_chinext_price, df_div_price, df_gold_price, df_nasdaq_price, df_bond_price,
           df_300_val, df_500_val, df_chinext_val, df_sse50_val]
    for df in dfs:
        df['trade_date'] = pd.to_datetime(df['trade_date'].astype(str))
        df.sort_values('trade_date', inplace=True)
        df.reset_index(drop=True, inplace=True)
        
    df_300_price['ma'] = df_300_price['close'].rolling(ma_window).mean()
    df_500_price['ma'] = df_500_price['close'].rolling(ma_window).mean()
    df_chinext_price['ma'] = df_chinext_price['close'].rolling(ma_window).mean()
    df_div_price['ma'] = df_div_price['close'].rolling(ma_window).mean()
    df_gold_price['ma'] = df_gold_price['close'].rolling(ma_window).mean()
    df_nasdaq_price['ma'] = df_nasdaq_price['close'].rolling(ma_window).mean()
    
    for val_df in [df_300_val, df_500_val, df_chinext_val, df_sse50_val]:
        val_df['pe_q'] = val_df['pe_ttm'].rolling(window=val_window, min_periods=250).rank(pct=True)
        val_df['pb_q'] = val_df['pb'].rolling(window=val_window, min_periods=250).rank(pct=True)
        val_df['val_q'] = (val_df['pe_q'] + val_df['pb_q']) / 2.0
        
    m300 = pd.merge(df_300_price[['trade_date', 'close', 'pct_chg', 'ma']], df_300_val[['trade_date', 'val_q']], on='trade_date', how='inner')
    m500 = pd.merge(df_500_price[['trade_date', 'close', 'pct_chg', 'ma']], df_500_val[['trade_date', 'val_q']], on='trade_date', how='inner')
    mchinext = pd.merge(df_chinext_price[['trade_date', 'close', 'pct_chg', 'ma']], df_chinext_val[['trade_date', 'val_q']], on='trade_date', how='inner')
    mdiv = pd.merge(df_div_price[['trade_date', 'close', 'pct_chg', 'ma']], df_sse50_val[['trade_date', 'val_q']], on='trade_date', how='inner')
    
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
            'close_300': row300['close'], 'ret_300': row300['pct_chg'] / 100.0, 'ma_300': row300['ma'], 'val_q_300': row300['val_q'],
            'close_500': row500['close'], 'ret_500': row500['pct_chg'] / 100.0, 'ma_500': row500['ma'], 'val_q_500': row500['val_q'],
            'close_chinext': rowchinext['close'], 'ret_chinext': rowchinext['pct_chg'] / 100.0, 'ma_chinext': rowchinext['ma'], 'val_q_chinext': rowchinext['val_q'],
            'close_div': rowdiv['close'], 'ret_div': rowdiv['pct_chg'] / 100.0, 'ma_div': rowdiv['ma'], 'val_q_div': rowdiv['val_q'],
            'close_gold': rowgold['close'], 'ret_gold': rowgold['pct_chg'] / 100.0, 'ma_gold': rowgold['ma'],
            'close_nasdaq': rownasdaq['close'], 'ret_nasdaq': rownasdaq['pct_chg'] / 100.0, 'ma_nasdaq': rownasdaq['ma'],
            'ret_bond': bond_ret
        })
        
    df_unified = pd.DataFrame(rows).sort_values('trade_date').reset_index(drop=True)
    return df_unified

def run_extended_backtest(df_period, val_coeff=0.4, q_threshold=0.20, dev_threshold=0.10, initial_capital=1000000.0):
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
            
            w_val_300 = val_coeff * (1.0 - row['val_q_300']) if not pd.isna(row['val_q_300']) else 0.0
            w_val_500 = val_coeff * (1.0 - row['val_q_500']) if not pd.isna(row['val_q_500']) else 0.0
            w_val_chinext = val_coeff * (1.0 - row['val_q_chinext']) if not pd.isna(row['val_q_chinext']) else 0.0
            w_val_div = val_coeff * (1.0 - row['val_q_div']) if not pd.isna(row['val_q_div']) else 0.0
            
            w_target_300 = w_val_300 if (row['val_q_300'] <= q_threshold) else (w_val_300 if trend_300 else w_val_300 * 0.5)
            w_target_500 = w_val_500 if (row['val_q_500'] <= q_threshold) else (w_val_500 if trend_500 else w_val_500 * 0.5)
            w_target_chinext = w_val_chinext if (row['val_q_chinext'] <= q_threshold) else (w_val_chinext if trend_chinext else w_val_chinext * 0.5)
            w_target_div = w_val_div if (row['val_q_div'] <= q_threshold) else (w_val_div if trend_div else w_val_div * 0.5)
            
            # Gold and Nasdaq allocation limit to 10% each
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
        'Max Drawdown': f"{max_dd:.2%}",
        'cagr_raw': cagr,
        'mdd_raw': max_dd,
        'sharpe_raw': sharpe
    }

def main():
    print("Running Asset Extension Step (Baseline vs. Extended Portfolio)...")
    
    # Define Parameter configurations
    base_ma, base_val_w, base_q, base_coeff = 250, 1210, 0.15, 0.6
    robust_ma, robust_val_w, robust_q, robust_coeff = 200, 1400, 0.20, 0.4
    
    # Load Unified Datasets
    print("Loading 2-asset data...")
    df_2asset_base = load_2asset_data(base_ma, base_val_w)
    df_2asset_robust = load_2asset_data(robust_ma, robust_val_w)
    
    print("Loading 6-asset extended data...")
    df_6asset_base = load_extended_data(base_ma, base_val_w)
    df_6asset_robust = load_extended_data(robust_ma, robust_val_w)
    
    # Split Dates
    is_start, is_end = "2015-01-01", "2024-02-05"
    oos_start, oos_end = "2024-02-06", "2026-03-13"
    
    # Filter datasets for IS and OOS
    def filter_dt(df, start, end):
        return df[(df['trade_date'] >= pd.to_datetime(start)) & (df['trade_date'] <= pd.to_datetime(end))].copy()
    
    df_2a_base_is = filter_dt(df_2asset_base, is_start, is_end)
    df_2a_base_oos = filter_dt(df_2asset_base, oos_start, oos_end)
    df_2a_robust_is = filter_dt(df_2asset_robust, is_start, is_end)
    df_2a_robust_oos = filter_dt(df_2asset_robust, oos_start, oos_end)
    
    df_6a_base_is = filter_dt(df_6asset_base, is_start, is_end)
    df_6a_base_oos = filter_dt(df_6asset_base, oos_start, oos_end)
    df_6a_robust_is = filter_dt(df_6asset_robust, is_start, is_end)
    df_6a_robust_oos = filter_dt(df_6asset_robust, oos_start, oos_end)
    
    # ------------------ IN-SAMPLE BACKTESTS ------------------
    print("\n--- Running In-Sample Backtests (2015-01-01 to 2024-02-05) ---")
    nav_2a_base_is = run_2asset_backtest(df_2a_base_is, val_coeff=base_coeff, q_threshold=base_q)
    nav_2a_robust_is = run_2asset_backtest(df_2a_robust_is, val_coeff=robust_coeff, q_threshold=robust_q)
    nav_6a_base_is = run_extended_backtest(df_6a_base_is, val_coeff=base_coeff, q_threshold=base_q)
    nav_6a_robust_is = run_extended_backtest(df_6a_robust_is, val_coeff=robust_coeff, q_threshold=robust_q)
    
    metrics_2a_base_is = compute_metrics(nav_2a_base_is['nav'])
    metrics_2a_robust_is = compute_metrics(nav_2a_robust_is['nav'])
    metrics_6a_base_is = compute_metrics(nav_6a_base_is['nav'])
    metrics_6a_robust_is = compute_metrics(nav_6a_robust_is['nav'])
    
    # Benchmarks in IS
    hs300_nav_is = (1.0 + df_2a_base_is['ret_300']).cumprod() * 1000000.0
    hs300_nav_is.index = df_2a_base_is['trade_date']
    metrics_hs300_is = compute_metrics(hs300_nav_is)
    
    static_50_50_is = []
    w300, w500 = 500000.0, 500000.0
    df_2a_base_is['year_week'] = df_2a_base_is['trade_date'].dt.strftime('%Y-%U')
    rebalance_dates_is = set(df_2a_base_is.groupby('year_week')['trade_date'].first())
    for idx, row in df_2a_base_is.iterrows():
        if idx > 0:
            w300 *= (1.0 + row['ret_300'])
            w500 *= (1.0 + row['ret_500'])
        nav = w300 + w500
        if row['trade_date'] in rebalance_dates_is:
            cost = (abs(nav*0.5 - w300) + abs(nav*0.5 - w500)) * 0.0005
            nav -= cost
            w300, w500 = nav*0.5, nav*0.5
        static_50_50_is.append(nav)
    static_nav_is = pd.Series(static_50_50_is, index=df_2a_base_is['trade_date'])
    metrics_static_is = compute_metrics(static_nav_is)
    
    # Print IS Table
    print("\n=== IN-SAMPLE PORTFOLIO COMPARISON ===")
    summary_is = pd.DataFrame([
        {'Strategy': '2-Asset Baseline (Params: Base)', **metrics_2a_base_is},
        {'Strategy': '2-Asset Baseline (Params: Robust)', **metrics_2a_robust_is},
        {'Strategy': '6-Asset Extended (Params: Base)', **metrics_6a_base_is},
        {'Strategy': '6-Asset Extended (Params: Robust)', **metrics_6a_robust_is},
        {'Strategy': 'HS300 Buy & Hold', **metrics_hs300_is},
        {'Strategy': 'Static 50/50', **metrics_static_is}
    ])
    print(summary_is.to_string(index=False))
    
    # ------------------ OUT-OF-SAMPLE BACKTESTS ------------------
    print("\n--- Running Out-of-Sample Backtests (2024-02-06 to 2026-03-13) ---")
    nav_2a_base_oos = run_2asset_backtest(df_2a_base_oos, val_coeff=base_coeff, q_threshold=base_q, initial_capital=1000000.0)
    nav_2a_robust_oos = run_2asset_backtest(df_2a_robust_oos, val_coeff=robust_coeff, q_threshold=robust_q, initial_capital=1000000.0)
    nav_6a_base_oos = run_extended_backtest(df_6a_base_oos, val_coeff=base_coeff, q_threshold=base_q, initial_capital=1000000.0)
    nav_6a_robust_oos = run_extended_backtest(df_6a_robust_oos, val_coeff=robust_coeff, q_threshold=robust_q, initial_capital=1000000.0)
    
    metrics_2a_base_oos = compute_metrics(nav_2a_base_oos['nav'])
    metrics_2a_robust_oos = compute_metrics(nav_2a_robust_oos['nav'])
    metrics_6a_base_oos = compute_metrics(nav_6a_base_oos['nav'])
    metrics_6a_robust_oos = compute_metrics(nav_6a_robust_oos['nav'])
    
    # Benchmarks in OOS
    hs300_nav_oos = (1.0 + df_2a_base_oos['ret_300']).cumprod() * 1000000.0
    hs300_nav_oos.index = df_2a_base_oos['trade_date']
    metrics_hs300_oos = compute_metrics(hs300_nav_oos)
    
    static_50_50_oos = []
    w300, w500 = 500000.0, 500000.0
    df_2a_base_oos['year_week'] = df_2a_base_oos['trade_date'].dt.strftime('%Y-%U')
    rebalance_dates_oos = set(df_2a_base_oos.groupby('year_week')['trade_date'].first())
    for idx, row in df_2a_base_oos.iterrows():
        if idx > 0:
            w300 *= (1.0 + row['ret_300'])
            w500 *= (1.0 + row['ret_500'])
        nav = w300 + w500
        if row['trade_date'] in rebalance_dates_oos:
            cost = (abs(nav*0.5 - w300) + abs(nav*0.5 - w500)) * 0.0005
            nav -= cost
            w300, w500 = nav*0.5, nav*0.5
        static_50_50_oos.append(nav)
    static_nav_oos = pd.Series(static_50_50_oos, index=df_2a_base_oos['trade_date'])
    metrics_static_oos = compute_metrics(static_nav_oos)
    
    # Print OOS Table
    print("\n=== OUT-OF-SAMPLE PORTFOLIO COMPARISON (BLIND TEST) ===")
    summary_oos = pd.DataFrame([
        {'Strategy': '2-Asset Baseline (Params: Base)', **metrics_2a_base_oos},
        {'Strategy': '2-Asset Baseline (Params: Robust)', **metrics_2a_robust_oos},
        {'Strategy': '6-Asset Extended (Params: Base)', **metrics_6a_base_oos},
        {'Strategy': '6-Asset Extended (Params: Robust)', **metrics_6a_robust_oos},
        {'Strategy': 'HS300 Buy & Hold', **metrics_hs300_oos},
        {'Strategy': 'Static 50/50', **metrics_static_oos}
    ])
    print(summary_oos.to_string(index=False))
    
    # Save CSVs
    summary_is.to_csv(os.path.join(RESULTS_DIR, 'extended_portfolio_metrics_is.csv'), index=False)
    summary_oos.to_csv(os.path.join(RESULTS_DIR, 'extended_portfolio_metrics_oos.csv'), index=False)
    
    # Plot Comparison Curves for OOS
    plt.figure(figsize=(12, 6))
    plt.plot(nav_2a_base_oos.index, nav_2a_base_oos['nav'] / 1e6, label='2-Asset Baseline (Base Params)', color='#e53935', linestyle=':')
    plt.plot(nav_2a_robust_oos.index, nav_2a_robust_oos['nav'] / 1e6, label='2-Asset Baseline (Robust Params)', color='#f4511e', linestyle='--')
    plt.plot(nav_6a_base_oos.index, nav_6a_base_oos['nav'] / 1e6, label='6-Asset Extended (Base Params)', color='#00897b', linewidth=2.0)
    plt.plot(nav_6a_robust_oos.index, nav_6a_robust_oos['nav'] / 1e6, label='6-Asset Extended (Robust Params)', color='#1e88e5', linewidth=2.5)
    plt.plot(hs300_nav_oos.index, hs300_nav_oos / 1e6, label='HS300 Buy & Hold', color='#757575', alpha=0.5)
    plt.plot(static_nav_oos.index, static_nav_oos / 1e6, label='Static 50/50', color='#43a047', linestyle='-.', alpha=0.6)
    
    plt.title("Out-of-Sample Performance Comparison: Asset Extension (2024-02-06 to 2026-03-13)", fontsize=12, fontweight='bold')
    plt.xlabel("Date")
    plt.ylabel("NAV (Normalized)")
    plt.legend(loc='upper left')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'nav_extended_comparison_oos.png'), dpi=300)
    plt.close()
    
    # Plot Comparison Curves for IS
    plt.figure(figsize=(12, 6))
    plt.plot(nav_2a_base_is.index, nav_2a_base_is['nav'] / 1e6, label='2-Asset Baseline (Base Params)', color='#e53935', linestyle=':')
    plt.plot(nav_2a_robust_is.index, nav_2a_robust_is['nav'] / 1e6, label='2-Asset Baseline (Robust Params)', color='#f4511e', linestyle='--')
    plt.plot(nav_6a_base_is.index, nav_6a_base_is['nav'] / 1e6, label='6-Asset Extended (Base Params)', color='#00897b', linewidth=2.0)
    plt.plot(nav_6a_robust_is.index, nav_6a_robust_is['nav'] / 1e6, label='6-Asset Extended (Robust Params)', color='#1e88e5', linewidth=2.5)
    plt.plot(hs300_nav_is.index, hs300_nav_is / 1e6, label='HS300 Buy & Hold', color='#757575', alpha=0.5)
    
    plt.title("In-Sample Performance Comparison: Asset Extension (2015-01-01 to 2024-02-05)", fontsize=12, fontweight='bold')
    plt.xlabel("Date")
    plt.ylabel("NAV (Normalized)")
    plt.legend(loc='upper left')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'nav_extended_comparison_is.png'), dpi=300)
    plt.close()
    
    print("\nBacktest complete. Metrics saved and charts saved in results/ directory.")

if __name__ == "__main__":
    main()

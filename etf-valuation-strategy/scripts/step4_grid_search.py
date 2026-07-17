import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import itertools

# Define paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(PROJECT_DIR, 'data')
RESULTS_DIR = os.path.join(PROJECT_DIR, 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

def load_data():
    df_300_price = pd.read_csv(os.path.join(DATA_DIR, 'hs300_daily.csv'))
    df_500_price = pd.read_csv(os.path.join(DATA_DIR, 'zz500_daily.csv'))
    df_bond_price = pd.read_csv(os.path.join(DATA_DIR, 'bond_etf_daily.csv'))
    
    df_300_val = pd.read_csv(os.path.join(DATA_DIR, 'hs300_valuation.csv'))
    df_500_val = pd.read_csv(os.path.join(DATA_DIR, 'zz500_valuation.csv'))
    
    for df in [df_300_price, df_500_price, df_bond_price, df_300_val, df_500_val]:
        df['trade_date'] = pd.to_datetime(df['trade_date'].astype(str))
    
    df_300_price = df_300_price.sort_values('trade_date').reset_index(drop=True)
    df_500_price = df_500_price.sort_values('trade_date').reset_index(drop=True)
    df_bond_price = df_bond_price.sort_values('trade_date').reset_index(drop=True)
    df_300_val = df_300_val.sort_values('trade_date').reset_index(drop=True)
    df_500_val = df_500_val.sort_values('trade_date').reset_index(drop=True)
    
    return df_300_price, df_500_price, df_bond_price, df_300_val, df_500_val

def build_unified_df(df_300_price, df_500_price, df_bond_price, df_300_val, df_500_val, ma_window, val_window):
    p300 = df_300_price.copy()
    p500 = df_500_price.copy()
    bprice = df_bond_price.copy()
    v300 = df_300_val.copy()
    v500 = df_500_val.copy()
    
    p300['ma'] = p300['close'].rolling(ma_window).mean()
    v300['pe_q'] = v300['pe_ttm'].rolling(window=val_window, min_periods=250).rank(pct=True)
    v300['pb_q'] = v300['pb'].rolling(window=val_window, min_periods=250).rank(pct=True)
    v300['val_q'] = (v300['pe_q'] + v300['pb_q']) / 2.0
    
    p500['ma'] = p500['close'].rolling(ma_window).mean()
    v500['pe_q'] = v500['pe_ttm'].rolling(window=val_window, min_periods=250).rank(pct=True)
    v500['pb_q'] = v500['pb'].rolling(window=val_window, min_periods=250).rank(pct=True)
    v500['val_q'] = (v500['pe_q'] + v500['pb_q']) / 2.0
    
    m300 = pd.merge(p300[['trade_date', 'close', 'open', 'pct_chg', 'ma']], 
                    v300[['trade_date', 'val_q']], on='trade_date', how='inner')
    m500 = pd.merge(p500[['trade_date', 'close', 'open', 'pct_chg', 'ma']], 
                    v500[['trade_date', 'val_q']], on='trade_date', how='inner')
    
    bprice['bond_ret'] = bprice['pct_chg'] / 100.0
    bond_map = bprice.set_index('trade_date')['bond_ret'].to_dict()
    
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
            'close_300': row300['close'],
            'ret_300': row300['pct_chg'] / 100.0,
            'ma_300': row300['ma'],
            'val_q_300': row300['val_q'],
            'close_500': row500['close'],
            'ret_500': row500['pct_chg'] / 100.0,
            'ma_500': row500['ma'],
            'val_q_500': row500['val_q'],
            'ret_bond': bond_ret
        })
        
    df_unified = pd.DataFrame(rows).sort_values('trade_date').reset_index(drop=True)
    return df_unified

def run_backtest(df_period, val_coeff=0.6, q_threshold=0.15, dev_threshold=0.10, initial_capital=1000000.0):
    if len(df_period) == 0:
        return None, None
        
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
            val_300 = val_300 * (1.0 + row['ret_300'])
            val_500 = val_500 * (1.0 + row['ret_500'])
            val_bond = val_bond * (1.0 + row['ret_bond'])
            
        nav = val_300 + val_500 + val_bond
        
        if dt in rebalance_check_dates:
            val_q_300 = row['val_q_300']
            val_q_500 = row['val_q_500']
            ma_300 = row['ma_300']
            ma_500 = row['ma_500']
            
            trend_300 = row['close_300'] >= ma_300 if not pd.isna(ma_300) else False
            trend_500 = row['close_500'] >= ma_500 if not pd.isna(ma_500) else False
            
            w_val_300 = val_coeff * (1.0 - val_q_300) if not pd.isna(val_q_300) else 0.0
            w_val_500 = val_coeff * (1.0 - val_q_500) if not pd.isna(val_q_500) else 0.0
            
            if pd.isna(val_q_300):
                w_target_300 = 0.0
            elif val_q_300 <= q_threshold:
                w_target_300 = w_val_300
            else:
                w_target_300 = w_val_300 if trend_300 else (w_val_300 * 0.5)
                
            if pd.isna(val_q_500):
                w_target_500 = 0.0
            elif val_q_500 <= q_threshold:
                w_target_500 = w_val_500
            else:
                w_target_500 = w_val_500 if trend_500 else (w_val_500 * 0.5)
                
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
                
                nav = nav - cost
                val_300 = nav * w_target_300
                val_500 = nav * w_target_500
                val_bond = nav * w_target_bond
                
        nav_history.append({
            'trade_date': dt,
            'nav': nav
        })
        
    nav_df = pd.DataFrame(nav_history).set_index('trade_date')
    return nav_df

def compute_performance_metrics(nav_series, initial_capital=1000000.0):
    if len(nav_series) < 2:
        return 0.0, 0.0, 0.0, 0.0
    total_ret = nav_series.iloc[-1] / initial_capital - 1
    years = (nav_series.index[-1] - nav_series.index[0]).days / 365.25
    cagr = (nav_series.iloc[-1] / initial_capital) ** (1.0 / years) - 1 if years > 0 else 0.0
    
    daily_rets = nav_series.pct_change().dropna()
    ann_vol = daily_rets.std() * np.sqrt(252)
    sharpe = (daily_rets.mean() * 252) / ann_vol if ann_vol > 0 else 0.0
    
    cum_max = nav_series.cummax()
    drawdowns = (nav_series - cum_max) / cum_max
    max_dd = drawdowns.min()
    
    return cagr, max_dd, sharpe, ann_vol

def main():
    print("Loading raw files...")
    df_300_price, df_500_price, df_bond_price, df_300_val, df_500_val = load_data()
    
    max_date = df_300_price['trade_date'].max()
    min_date = df_300_price['trade_date'].min()
    print(f"Data range available: {min_date.strftime('%Y-%m-%d')} to {max_date.strftime('%Y-%m-%d')}")
    
    ma_windows = [200, 220, 250, 280, 300]
    val_windows = [1000, 1100, 1210, 1300, 1400]
    q_thresholds = [0.10, 0.12, 0.15, 0.18, 0.20]
    val_coeffs = [0.4, 0.5, 0.6, 0.7, 0.8]
    
    is_start = "2015-01-01"
    is_end = "2024-02-05"
    oos_start = "2024-02-06"
    oos_end = max_date.strftime('%Y-%m-%d')
    
    print("\n--- Starting In-Sample Robustness Grid Search (2015-01-01 to 2024-02-05) ---")
    results = []
    data_cache = {}
    
    total_combinations = len(ma_windows) * len(val_windows) * len(q_thresholds) * len(val_coeffs)
    print(f"Total combinations to run: {total_combinations}")
    
    count = 0
    for ma in ma_windows:
        for val_w in val_windows:
            key = (ma, val_w)
            if key not in data_cache:
                df_unified = build_unified_df(df_300_price, df_500_price, df_bond_price, df_300_val, df_500_val, ma, val_w)
                df_is = df_unified[(df_unified['trade_date'] >= pd.to_datetime(is_start)) & 
                                   (df_unified['trade_date'] <= pd.to_datetime(is_end))].copy()
                df_oos = df_unified[(df_unified['trade_date'] >= pd.to_datetime(oos_start)) & 
                                    (df_unified['trade_date'] <= pd.to_datetime(oos_end))].copy()
                data_cache[key] = (df_is, df_oos)
            
            df_is, df_oos = data_cache[key]
            
            for q in q_thresholds:
                for coeff in val_coeffs:
                    count += 1
                    if count % 100 == 0:
                        print(f"Progress: {count}/{total_combinations}...")
                    
                    nav_is = run_backtest(df_is, val_coeff=coeff, q_threshold=q)
                    if nav_is is not None:
                        cagr, mdd, sharpe, vol = compute_performance_metrics(nav_is['nav'])
                        results.append({
                            'ma_window': ma,
                            'val_window': val_w,
                            'q_threshold': q,
                            'val_coeff': coeff,
                            'is_cagr': cagr,
                            'is_mdd': mdd,
                            'is_sharpe': sharpe,
                            'is_vol': vol
                        })
                        
    df_results = pd.DataFrame(results)
    df_results.to_csv(os.path.join(RESULTS_DIR, 'grid_search_is_results.csv'), index=False)
    print(f"Saved {len(df_results)} grid search results to results/grid_search_is_results.csv")
    
    print("\n--- Performing Neighborhood/Plateau Analysis ---")
    ma_map = {v: i for i, v in enumerate(ma_windows)}
    val_w_map = {v: i for i, v in enumerate(val_windows)}
    q_map = {v: i for i, v in enumerate(q_thresholds)}
    coeff_map = {v: i for i, v in enumerate(val_coeffs)}
    
    df_results['ma_idx'] = df_results['ma_window'].map(ma_map)
    df_results['val_w_idx'] = df_results['val_window'].map(val_w_map)
    df_results['q_idx'] = df_results['q_threshold'].map(q_map)
    df_results['coeff_idx'] = df_results['val_coeff'].map(coeff_map)
    
    coord_lookup = {}
    for idx, row in df_results.iterrows():
        coord = (int(row['ma_idx']), int(row['val_w_idx']), int(row['q_idx']), int(row['coeff_idx']))
        coord_lookup[coord] = {
            'cagr': row['is_cagr'],
            'mdd': row['is_mdd'],
            'sharpe': row['is_sharpe'],
            'row_idx': idx
        }
        
    neighbor_stats = []
    for idx, row in df_results.iterrows():
        c_ma, c_val_w, c_q, c_coeff = int(row['ma_idx']), int(row['val_w_idx']), int(row['q_idx']), int(row['coeff_idx'])
        
        neighbor_cagrs = []
        neighbor_mdds = []
        neighbor_sharpes = []
        
        for d_ma in [-1, 0, 1]:
            for d_val_w in [-1, 0, 1]:
                for d_q in [-1, 0, 1]:
                    for d_coeff in [-1, 0, 1]:
                        n_ma = c_ma + d_ma
                        n_val_w = c_val_w + d_val_w
                        n_q = c_q + d_q
                        n_coeff = c_coeff + d_coeff
                        
                        if (0 <= n_ma < len(ma_windows) and 
                            0 <= n_val_w < len(val_windows) and 
                            0 <= n_q < len(q_thresholds) and 
                            0 <= n_coeff < len(val_coeffs)):
                            n_coord = (n_ma, n_val_w, n_q, n_coeff)
                            if n_coord in coord_lookup:
                                neighbor_cagrs.append(coord_lookup[n_coord]['cagr'])
                                neighbor_mdds.append(coord_lookup[n_coord]['mdd'])
                                neighbor_sharpes.append(coord_lookup[n_coord]['sharpe'])
                                
        mean_cagr = np.mean(neighbor_cagrs)
        std_cagr = np.std(neighbor_cagrs)
        mean_mdd = np.mean(neighbor_mdds)
        std_mdd = np.std(neighbor_mdds)
        mean_sharpe = np.mean(neighbor_sharpes)
        
        robustness_score = mean_cagr - 0.5 * abs(mean_mdd) - 0.2 * std_cagr - 0.2 * std_mdd
        
        neighbor_stats.append({
            'row_idx': idx,
            'mean_cagr': mean_cagr,
            'std_cagr': std_cagr,
            'mean_mdd': mean_mdd,
            'std_mdd': std_mdd,
            'mean_sharpe': mean_sharpe,
            'num_neighbors': len(neighbor_cagrs),
            'robustness_score': robustness_score
        })
        
    df_neighbor = pd.DataFrame(neighbor_stats)
    df_results = df_results.merge(df_neighbor, left_index=True, right_on='row_idx').drop(columns=['row_idx'])
    
    df_results.to_csv(os.path.join(RESULTS_DIR, 'grid_search_is_robustness.csv'), index=False)
    
    baseline_mask = ((df_results['ma_window'] == 250) & 
                     (df_results['val_window'] == 1210) & 
                     (df_results['q_threshold'] == 0.15) & 
                     (df_results['val_coeff'] == 0.6))
    
    if baseline_mask.any():
        baseline_row = df_results[baseline_mask].iloc[0]
        print("\n=== Baseline In-Sample Metrics ===")
        print(f"Parameters: MA={baseline_row['ma_window']}, ValWindow={baseline_row['val_window']}, Q={baseline_row['q_threshold']:.2f}, Coeff={baseline_row['val_coeff']:.1f}")
        print(f"CAGR: {baseline_row['is_cagr']:.2%}")
        print(f"Max Drawdown: {baseline_row['is_mdd']:.2%}")
        print(f"Sharpe: {baseline_row['is_sharpe']:.2f}")
        print(f"Neighbor Mean CAGR: {baseline_row['mean_cagr']:.2%} (std: {baseline_row['std_cagr']:.2%})")
        print(f"Neighbor Mean Max Drawdown: {baseline_row['mean_mdd']:.2%} (std: {baseline_row['std_mdd']:.2%})")
    else:
        print("Warning: Baseline parameters not found in the grid search results.")
        baseline_row = None
        
    safe_candidates = df_results[df_results['mean_mdd'] >= -0.26].copy()
    if len(safe_candidates) == 0:
        safe_candidates = df_results.copy()
        
    top_robust = safe_candidates.sort_values('robustness_score', ascending=False).head(3)
    
    print("\n=== Top 3 Robust Parameter Sets (In-Sample) ===")
    top_sets = []
    for rank, (idx, row) in enumerate(top_robust.iterrows(), 1):
        print(f"Rank {rank}: MA={row['ma_window']}, ValWindow={row['val_window']}, Q={row['q_threshold']:.2f}, Coeff={row['val_coeff']:.1f}")
        print(f"  IS CAGR: {row['is_cagr']:.2%} (Neighbor Mean: {row['mean_cagr']:.2%}, std: {row['std_cagr']:.2%})")
        print(f"  IS MDD: {row['is_mdd']:.2%} (Neighbor Mean: {row['mean_mdd']:.2%}, std: {row['std_mdd']:.2%})")
        print(f"  IS Sharpe: {row['is_sharpe']:.2f} (Neighbor Mean: {row['mean_sharpe']:.2f})")
        top_sets.append(row)
        
    print(f"\n--- Starting Out-of-Sample Blind Test ({oos_start} to {oos_end}) ---")
    
    test_sets = []
    if baseline_row is not None:
        test_sets.append(('Baseline', baseline_row['ma_window'], baseline_row['val_window'], baseline_row['q_threshold'], baseline_row['val_coeff']))
    for i, row in enumerate(top_sets, 1):
        test_sets.append((f'Top Robust {i}', row['ma_window'], row['val_window'], row['q_threshold'], row['val_coeff']))
        
    oos_results = []
    
    for name, ma, val_w, q, coeff in test_sets:
        df_is_split, df_oos_split = data_cache[(ma, val_w)]
        nav_oos = run_backtest(df_oos_split, val_coeff=coeff, q_threshold=q, initial_capital=1000000.0)
        cagr, mdd, sharpe, vol = compute_performance_metrics(nav_oos['nav'])
        
        oos_results.append({
            'Name': name,
            'MA': ma,
            'ValWindow': val_w,
            'Q_Threshold': q,
            'Val_Coeff': coeff,
            'OOS_CAGR': cagr,
            'OOS_MDD': mdd,
            'OOS_Sharpe': sharpe,
            'OOS_Vol': vol,
            'nav_series': nav_oos['nav']
        })
        
    base_ma, base_val_w = 250, 1210
    _, df_oos_base = data_cache[(base_ma, base_val_w)]
    df_oos_base = df_oos_base.copy().reset_index(drop=True)
    
    hs300_nav = pd.Series((1.0 + df_oos_base['ret_300']).cumprod() * 1000000.0)
    hs300_nav.index = df_oos_base['trade_date']
    hs300_cagr, hs300_mdd, hs300_sharpe, hs300_vol = compute_performance_metrics(hs300_nav)
    oos_results.append({
        'Name': 'HS300 Buy & Hold',
        'MA': '-', 'ValWindow': '-', 'Q_Threshold': '-', 'Val_Coeff': '-',
        'OOS_CAGR': hs300_cagr, 'OOS_MDD': hs300_mdd, 'OOS_Sharpe': hs300_sharpe, 'OOS_Vol': hs300_vol,
        'nav_series': hs300_nav
    })
    
    zz500_nav = pd.Series((1.0 + df_oos_base['ret_500']).cumprod() * 1000000.0)
    zz500_nav.index = df_oos_base['trade_date']
    zz500_cagr, zz500_mdd, zz500_sharpe, zz500_vol = compute_performance_metrics(zz500_nav)
    oos_results.append({
        'Name': 'ZZ500 Buy & Hold',
        'MA': '-', 'ValWindow': '-', 'Q_Threshold': '-', 'Val_Coeff': '-',
        'OOS_CAGR': zz500_cagr, 'OOS_MDD': zz500_mdd, 'OOS_Sharpe': zz500_sharpe, 'OOS_Vol': zz500_vol,
        'nav_series': zz500_nav
    })
    
    static_nav_list = []
    val_300, val_500 = 500000.0, 500000.0
    df_oos_base['year_week'] = df_oos_base['trade_date'].dt.strftime('%Y-%U')
    rebalance_dates_oos = set(df_oos_base.groupby('year_week')['trade_date'].first())
    
    for idx, row in df_oos_base.iterrows():
        if idx > 0:
            val_300 = val_300 * (1.0 + row['ret_300'])
            val_500 = val_500 * (1.0 + row['ret_500'])
        nav = val_300 + val_500
        if row['trade_date'] in rebalance_dates_oos:
            cost = (abs(nav*0.5 - val_300) + abs(nav*0.5 - val_500)) * 0.0005
            nav -= cost
            val_300, val_500 = nav*0.5, nav*0.5
        static_nav_list.append(nav)
        
    static_nav_series = pd.Series(static_nav_list, index=df_oos_base['trade_date'])
    static_cagr, static_mdd, static_sharpe, static_vol = compute_performance_metrics(static_nav_series)
    oos_results.append({
        'Name': 'Static 50/50',
        'MA': '-', 'ValWindow': '-', 'Q_Threshold': '-', 'Val_Coeff': '-',
        'OOS_CAGR': static_cagr, 'OOS_MDD': static_mdd, 'OOS_Sharpe': static_sharpe, 'OOS_Vol': static_vol,
        'nav_series': static_nav_series
    })
    
    print("\n=== OUT-OF-SAMPLE BLIND TEST RESULTS (2024-02-06 to 2026-03-13) ===")
    df_oos_summary = pd.DataFrame([
        {
            'Portfolio/Strategy': r['Name'],
            'MA': r['MA'],
            'ValWindow': r['ValWindow'],
            'Q_Threshold': f"{r['Q_Threshold']:.2f}" if isinstance(r['Q_Threshold'], float) else r['Q_Threshold'],
            'Val_Coeff': f"{r['Val_Coeff']:.1f}" if isinstance(r['Val_Coeff'], float) else r['Val_Coeff'],
            'OOS CAGR': f"{r['OOS_CAGR']:.2%}",
            'OOS Max Drawdown': f"{r['OOS_MDD']:.2%}",
            'OOS Sharpe': f"{r['OOS_Sharpe']:.2f}",
            'OOS Volatility': f"{r['OOS_Vol']:.2%}"
        }
        for r in oos_results
    ])
    print(df_oos_summary.to_string(index=False))
    df_oos_summary.to_csv(os.path.join(RESULTS_DIR, 'oos_blind_test_results.csv'), index=False)
    
    plt.figure(figsize=(12, 6))
    for r in oos_results:
        name = r['Name']
        series = r['nav_series']
        dates = series.index
        
        if 'Hold' in name:
            alpha = 0.5
            style = '-'
            if '300' in name: color = '#e53935'
            else: color = '#ffb300'
        elif 'Static' in name:
            alpha = 0.7
            style = '--'
            color = '#4caf50'
        else:
            alpha = 1.0
            style = '-'
            if 'Baseline' in name: color = '#1a237e'
            elif '1' in name: color = '#0288d1'
            elif '2' in name: color = '#00796b'
            else: color = '#e65100'
            
        plt.plot(dates, series / 1e6, label=name, alpha=alpha, linestyle=style, color=color, linewidth=2.5 if 'Baseline' in name else 1.5)
        
    plt.title("Out-of-Sample Blind Test Performance (2024-02-06 to 2026-03-13)", fontsize=13, fontweight='bold')
    plt.xlabel("Date")
    plt.ylabel("NAV (Normalized)")
    plt.legend(loc='upper left')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'nav_oos_blind_test.png'), dpi=300)
    plt.close()
    print("\nOOS performance plot saved to results/nav_oos_blind_test.png")

if __name__ == "__main__":
    main()

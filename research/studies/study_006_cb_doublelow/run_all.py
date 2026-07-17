import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from backtest_cb_doublelow import load_data, run_backtest, compute_metrics

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
STUDY_DIR = SCRIPT_DIR
RESEARCH_DIR = os.path.dirname(os.path.dirname(STUDY_DIR))
CACHE_DIR = os.path.join(RESEARCH_DIR, 'cache')
RESULTS_DIR = os.path.join(RESEARCH_DIR, 'results')
ETF_DATA_DIR = os.path.join(os.path.dirname(RESEARCH_DIR), 'etf-valuation-strategy', 'data')

os.makedirs(RESULTS_DIR, exist_ok=True)

def run_grid_search(df_pit):
    print("\nRunning Multi-Factor Strategy Parameter Grid Search...")
    N_list = [15, 20, 25]
    freq_list = ['W', '2W']
    
    results = []
    
    # 1. Run Search for Multi-Factor Robust Strategy
    for N in N_list:
        for freq in freq_list:
            print(f"  Testing N={N}, Freq={freq} (Robust Multi-Factor)...")
            df_nav, _ = run_backtest(
                df_pit, N=N, rebalance_freq=freq,
                min_rating='A', min_size=1.0, min_maturity=0.5,
                use_multi_factor=True, use_risk_control=True, use_position_mgmt=True
            )
            metrics = compute_metrics(df_nav['nav'])
            metrics['N'] = N
            metrics['freq'] = freq
            metrics['type'] = 'Robust_Multi_Factor'
            results.append(metrics)
            
    # 2. Run Search for Baseline Double-Low (No Filters/风控/仓配)
    for N in N_list:
        for freq in freq_list:
            print(f"  Testing N={N}, Freq={freq} (Baseline Double-Low)...")
            df_nav, _ = run_backtest(
                df_pit, N=N, rebalance_freq=freq,
                min_rating='A', min_size=0.0, min_maturity=0.0,
                use_multi_factor=False, use_risk_control=False, use_position_mgmt=False
            )
            metrics = compute_metrics(df_nav['nav'])
            metrics['N'] = N
            metrics['freq'] = freq
            metrics['type'] = 'Baseline_Double_Low'
            results.append(metrics)
            
    df_results = pd.DataFrame(results)
    results_csv = os.path.join(RESULTS_DIR, 'cb_parameter_search.csv')
    df_results.to_csv(results_csv, index=False)
    print(f"Grid search complete. Results saved to {results_csv}")
    return df_results

def print_summary(df_results):
    print("\n" + "="*80)
    print("                      PARAMETER GRID SEARCH SUMMARY")
    print("="*80)
    print(f"| Type | N | Freq | CAGR | Sharpe | MaxDD | Calmar |")
    print(f"| :--- | :---: | :---: | :---: | :---: | :---: | :---: |")
    for _, r in df_results.sort_values(['type', 'Sharpe'], ascending=[True, False]).iterrows():
        print(f"| {r['type']:20} | {r['N']:2d} | {r['freq']:4s} | {r['CAGR']:.2%} | {r['Sharpe']:.2f} | {r['Max Drawdown']:.2%} | {r['Calmar']:.2f} |")
    print("="*80)

def generate_comparison_plots(df_pit):
    print("\nGenerating final comparative plots...")
    
    # 1. Baseline: N=20, Freq=2W, No Filters
    df_nav_base, _ = run_backtest(
        df_pit, N=20, rebalance_freq='2W',
        min_rating='A', min_size=0.0, min_maturity=0.0,
        use_multi_factor=False, use_risk_control=False, use_position_mgmt=False
    )
    
    # 2. Robust Multi-Factor: N=20, Freq=2W, With Filters, 风控, 仓配
    df_nav_robust, _ = run_backtest(
        df_pit, N=20, rebalance_freq='2W',
        min_rating='A', min_size=1.0, min_maturity=0.5,
        use_multi_factor=True, use_risk_control=True, use_position_mgmt=True
    )
    
    # 3. Load CSI 1000 Benchmark if available
    df_nav_bench = None
    bench_csv = os.path.join(ETF_DATA_DIR, 'zz1000_daily.csv')
    if os.path.exists(bench_csv):
        try:
            df_bench = pd.read_csv(bench_csv)
            df_bench['trade_date'] = pd.to_datetime(df_bench['trade_date'].astype(str))
            df_bench = df_bench[(df_bench['trade_date'] >= df_nav_base.index.min()) & 
                               (df_bench['trade_date'] <= df_nav_base.index.max())].copy()
            df_bench.sort_values('trade_date', inplace=True)
            first_close = df_bench.iloc[0]['close']
            df_bench['nav'] = (df_bench['close'] / first_close) * 1000000.0
            df_nav_bench = df_bench.set_index('trade_date')
            print("Successfully loaded CSI 1000 index benchmark.")
        except Exception as e:
            print(f"Error loading CSI 1000 benchmark: {e}")
            
    # Setup plots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True, gridspec_kw={'height_ratios': [3, 1]})
    
    # Plot NAV
    ax1.plot(df_nav_robust.index, df_nav_robust['nav'] / 1e6, label='Robust Multi-Factor (Filters+Risk+Position)', color='#2ca02c', linewidth=2.5)
    ax1.plot(df_nav_base.index, df_nav_base['nav'] / 1e6, label='Baseline Double-Low (No Filters)', color='#1f77b4', linewidth=1.5)
    if df_nav_bench is not None:
        ax1.plot(df_nav_bench.index, df_nav_bench['nav'] / 1e6, label='CSI 1000 Index B&H', color='#7f7f7f', linestyle='--', linewidth=1.2)
        
    ax1.set_title('Convertible Bond Rotation Strategy Comparison (2018 - 2026)', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Portfolio Wealth (Millions)', fontsize=12)
    ax1.legend(loc='upper left')
    
    # Plot Drawdown
    dd_robust = (df_nav_robust['nav'] - df_nav_robust['nav'].cummax()) / df_nav_robust['nav'].cummax()
    dd_base = (df_nav_base['nav'] - df_nav_base['nav'].cummax()) / df_nav_base['nav'].cummax()
    
    ax2.fill_between(df_nav_robust.index, dd_robust, 0, color='#2ca02c', alpha=0.15)
    ax2.plot(df_nav_robust.index, dd_robust, color='#2ca02c', linewidth=1.5)
    
    ax2.plot(df_nav_base.index, dd_base, color='#1f77b4', linewidth=1.0)
    
    if df_nav_bench is not None:
        dd_bench = (df_nav_bench['nav'] - df_nav_bench['nav'].cummax()) / df_nav_bench['nav'].cummax()
        ax2.plot(df_nav_bench.index, dd_bench, color='#7f7f7f', linestyle='--', linewidth=1.0)
        
    ax2.set_ylabel('Drawdown', fontsize=12)
    ax2.set_ylim(-0.8, 0.02)
    
    # Date formatting
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())
    plt.gcf().autofmt_xdate()
    
    plt.tight_layout()
    plot_path = os.path.join(RESULTS_DIR, 'cb_doublelow_comparison.png')
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"Comparison plot saved to: {plot_path}")

def main():
    try:
        df_pit = load_data()
        df_results = run_grid_search(df_pit)
        print_summary(df_results)
        generate_comparison_plots(df_pit)
        print("\nAll backtests and analysis tasks completed successfully!")
    except Exception as e:
        print(f"Error during orchestrator execution: {e}")

if __name__ == '__main__':
    main()

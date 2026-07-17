import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Add current directory to path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(SCRIPT_DIR)

from step9_backtest_options import load_backtest_data, run_option_backtest, compute_metrics

def main():
    print("Loading data and QVIX index...")
    # load_backtest_data will automatically merge qvix_daily.csv if it exists
    df_unified = load_backtest_data(ma_window=50, val_window=1400)
    
    # In-Sample period
    is_start, is_end = "2015-01-01", "2024-02-05"
    df_is = df_unified[(df_unified['trade_date'] >= pd.to_datetime(is_start)) & (df_unified['trade_date'] <= pd.to_datetime(is_end))].copy()
    
    # IV levels to test (including 'qvix' for real QVIX)
    iv_levels = [0.20, 0.25, 0.35, 0.45, 'qvix']
    # Strikes to test
    strikes = [0.97, 0.95] # 3% OTM and 5% OTM
    
    # No options baseline
    print("\n--- Running Baseline (No Options) ---")
    nav_base = run_option_backtest(df_is, buy_put=False)
    metrics_base = compute_metrics(nav_base['nav'])
    
    # Benchmarks
    hs300_nav_is = (1.0 + df_is['ret_300']).cumprod() * 1000000.0
    hs300_nav_is.index = df_is['trade_date']
    metrics_hs300 = compute_metrics(hs300_nav_is)
    
    results = []
    results.append({
        'Strategy': 'Fast Momentum Only (Base)',
        'IV Scenario': 'N/A',
        'Strike': 'N/A',
        **metrics_base
    })
    
    # Option-Protected backtests for different IVs and strikes
    nav_curves = {}
    nav_curves['Base'] = nav_base['nav']
    
    for iv in iv_levels:
        for strike in strikes:
            strike_name = "3% OTM" if strike == 0.97 else "5% OTM"
            iv_name = "Real QVIX" if iv == 'qvix' else f"{iv:.0%}"
            print(f"Running IV={iv_name}, Strike={strike_name}...")
            
            nav_opt = run_option_backtest(df_is, buy_put=True, strike_ratio=strike, iv=iv)
            metrics_opt = compute_metrics(nav_opt['nav'])
            
            results.append({
                'Strategy': f"Option-Protected ({strike_name})",
                'IV Scenario': iv_name,
                'Strike': f"{strike:.2f}",
                **metrics_opt
            })
            
            curve_key = f"IV_{int(iv*100)}_OTM{int((1-strike)*100)}" if isinstance(iv, float) else f"IV_QVIX_OTM{int((1-strike)*100)}"
            nav_curves[curve_key] = nav_opt['nav']
            
    results.append({
        'Strategy': 'HS300 Buy & Hold',
        'IV Scenario': 'N/A',
        'Strike': 'N/A',
        **metrics_hs300
    })
    
    df_results = pd.DataFrame(results)
    print("\n=== IN-SAMPLE IV SENSITIVITY & REAL QVIX COMPARISON ===")
    print(df_results.to_string(index=False))
    
    # Save CSV
    results_path = os.path.join(os.path.dirname(SCRIPT_DIR), 'results')
    os.makedirs(results_path, exist_ok=True)
    df_results.to_csv(os.path.join(results_path, 'option_iv_sensitivity_is.csv'), index=False)
    print(f"\nMetrics saved to results/option_iv_sensitivity_is.csv")
    
    # Plot IV Sensitivity Curves
    plt.figure(figsize=(14, 8))
    
    # Benchmarks
    plt.plot(nav_base.index, nav_base['nav'] / 1e6, label='Fast Momentum Only (Base)', color='#d32f2f', linestyle='--', linewidth=1.5)
    plt.plot(hs300_nav_is.index, hs300_nav_is / 1e6, label='HS300 Buy & Hold', color='#757575', alpha=0.5, linewidth=1.2)
    
    # 3% OTM curves
    plt.plot(nav_curves['IV_20_OTM3'].index, nav_curves['IV_20_OTM3'] / 1e6, label='3% OTM (IV=20%)', color='#4caf50', linewidth=1.5)
    plt.plot(nav_curves['IV_QVIX_OTM3'].index, nav_curves['IV_QVIX_OTM3'] / 1e6, label='3% OTM (Real QVIX)', color='#1b5e20', linewidth=2.0)
    plt.plot(nav_curves['IV_35_OTM3'].index, nav_curves['IV_35_OTM3'] / 1e6, label='3% OTM (IV=35%)', color='#a5d6a7', linewidth=1.0, linestyle=':')
    plt.plot(nav_curves['IV_45_OTM3'].index, nav_curves['IV_45_OTM3'] / 1e6, label='3% OTM (IV=45%)', color='#c8e6c9', linewidth=1.0, linestyle=':')
    
    # 5% OTM curves
    plt.plot(nav_curves['IV_20_OTM5'].index, nav_curves['IV_20_OTM5'] / 1e6, label='5% OTM (IV=20%)', color='#ff9800', linewidth=1.5)
    plt.plot(nav_curves['IV_QVIX_OTM5'].index, nav_curves['IV_QVIX_OTM5'] / 1e6, label='5% OTM (Real QVIX)', color='#e65100', linewidth=2.0)
    plt.plot(nav_curves['IV_35_OTM5'].index, nav_curves['IV_35_OTM5'] / 1e6, label='5% OTM (IV=35%)', color='#ffcc80', linewidth=1.0, linestyle=':')
    plt.plot(nav_curves['IV_45_OTM5'].index, nav_curves['IV_45_OTM5'] / 1e6, label='5% OTM (IV=45%)', color='#ffe0b2', linewidth=1.0, linestyle=':')
    
    plt.title("In-Sample Option-Protection IV Sensitivity & Real QVIX Calibration (2015-01-01 to 2024-02-05)", fontsize=12, fontweight='bold')
    plt.xlabel("Date")
    plt.ylabel("NAV (Normalized)")
    plt.legend(loc='upper left', ncol=2)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    
    plt.savefig(os.path.join(results_path, 'nav_options_sensitivity_is.png'), dpi=300)
    print(f"Chart saved to results/nav_options_sensitivity_is.png")

if __name__ == "__main__":
    main()

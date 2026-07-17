import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Define paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
RESULTS_DIR = os.path.join(PROJECT_DIR, 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

def run_leverage_simulation(nav_df, leverage_factor, financing_rate=0.045):
    # Calculate daily returns of the base portfolio
    daily_rets = nav_df['nav'].pct_change().fillna(0.0)
    
    # Daily financing cost
    daily_financing = financing_rate / 252.0
    
    # Leveraged daily return
    # R_lev = L * R_port - (L - 1) * R_fin
    leveraged_rets = leverage_factor * daily_rets - (leverage_factor - 1.0) * daily_financing
    
    # Cumulative NAV
    leveraged_nav = (1.0 + leveraged_rets).cumprod() * 1000000.0
    leveraged_nav.index = nav_df.index
    
    return leveraged_nav

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
    print("Running Moderate Leverage Simulation for 6-Asset ERP Strategy...")
    
    # Load history files
    nav_is = pd.read_csv(os.path.join(RESULTS_DIR, 'nav_erp_is_history.csv'))
    nav_is['trade_date'] = pd.to_datetime(nav_is['trade_date'])
    nav_is = nav_is.set_index('trade_date')
    
    nav_oos = pd.read_csv(os.path.join(RESULTS_DIR, 'nav_erp_oos_history.csv'))
    nav_oos['trade_date'] = pd.to_datetime(nav_oos['trade_date'])
    nav_oos = nav_oos.set_index('trade_date')
    
    leverage_factors = [1.0, 1.1, 1.2, 1.3, 1.4, 1.5]
    financing_rate = 0.045 # 4.5% annual financing cost
    
    # ------------------ IN-SAMPLE LEVERAGE ------------------
    print("\n--- Running In-Sample Leverage Simulation (2015-01-01 to 2024-02-05) ---")
    is_results = []
    is_curves = {}
    
    for L in leverage_factors:
        nav_lev = run_leverage_simulation(nav_is, L, financing_rate=financing_rate)
        is_curves[L] = nav_lev
        metrics = compute_metrics(nav_lev)
        is_results.append({
            'Leverage': f"{L:.1f}x",
            **metrics
        })
        
    df_is = pd.DataFrame(is_results)
    print(df_is.to_string(index=False))
    
    # ------------------ OUT-OF-SAMPLE LEVERAGE ------------------
    print("\n--- Running Out-of-Sample Leverage Simulation (2024-02-06 to 2026-03-13) ---")
    oos_results = []
    oos_curves = {}
    
    for L in leverage_factors:
        nav_lev = run_leverage_simulation(nav_oos, L, financing_rate=financing_rate)
        oos_curves[L] = nav_lev
        metrics = compute_metrics(nav_lev)
        oos_results.append({
            'Leverage': f"{L:.1f}x",
            **metrics
        })
        
    df_oos = pd.DataFrame(oos_results)
    print(df_oos.to_string(index=False))
    
    # Save CSVs
    df_is.to_csv(os.path.join(RESULTS_DIR, 'leverage_metrics_is.csv'), index=False)
    df_oos.to_csv(os.path.join(RESULTS_DIR, 'leverage_metrics_oos.csv'), index=False)
    
    # Plot IS curves
    plt.figure(figsize=(12, 6))
    for L in leverage_factors:
        plt.plot(is_curves[L].index, is_curves[L] / 1e6, label=f"Leverage {L:.1f}x")
    plt.title("In-Sample Leveraged Equity Curves (6-Asset ERP Portfolio, Cost=4.5%)", fontsize=12, fontweight='bold')
    plt.xlabel("Date")
    plt.ylabel("NAV (Normalized)")
    plt.legend(loc='upper left')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'nav_leverage_is.png'), dpi=300)
    plt.close()
    
    # Plot OOS curves
    plt.figure(figsize=(12, 6))
    for L in leverage_factors:
        plt.plot(oos_curves[L].index, oos_curves[L] / 1e6, label=f"Leverage {L:.1f}x")
    plt.title("Out-of-Sample Leveraged Equity Curves (6-Asset ERP Portfolio, Cost=4.5%)", fontsize=12, fontweight='bold')
    plt.xlabel("Date")
    plt.ylabel("NAV (Normalized)")
    plt.legend(loc='upper left')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'nav_leverage_oos.png'), dpi=300)
    plt.close()
    
    print("\nLeverage backtest complete. Metrics and charts saved in results/ directory.")

if __name__ == "__main__":
    main()

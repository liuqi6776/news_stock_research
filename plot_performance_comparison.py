import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

def generate_nav_curve(trades_file, initial_capital=100000):
    if not os.path.exists(trades_file):
        print(f"Warning: {trades_file} not found.")
        return None
    
    df = pd.read_csv(trades_file)
    sell_trades = df[df['action'] == 'SELL'].copy()
    sell_trades['sell_date'] = pd.to_datetime(sell_trades['sell_date'].astype(str))
    
    # Sort by date
    sell_trades = sell_trades.sort_values('sell_date')
    
    # Cumulative PNL
    sell_trades['cum_pnl'] = sell_trades['pnl'].cumsum()
    sell_trades['nav'] = initial_capital + sell_trades['cum_pnl']
    
    # Build a daily series
    nav_series = sell_trades.groupby('sell_date')['nav'].last()
    
    return nav_series

def plot_comparison():
    # Load curves
    oo_nav = generate_nav_curve('trades_open_open.csv')
    cc_nav = generate_nav_curve('trades_close_close.csv')
    
    plt.figure(figsize=(12, 7))
    
    if oo_nav is not None:
        plt.plot(oo_nav.index, oo_nav.values, label='Mode 1: Open-Open (Baseline)', color='#e74c3c', linewidth=2)
        
    if cc_nav is not None:
        plt.plot(cc_nav.index, cc_nav.values, label='Mode 2: Close-Close (Optimized)', color='#2ecc71', linewidth=2)
    
    plt.title('A-Stock T+1 Strategy Timing Optimization Comparison', fontsize=14, pad=20)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('NAV (Initial: 100,000)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(fontsize=11)
    
    # Highlight final values
    if oo_nav is not None:
        plt.annotate(f'{oo_nav.iloc[-1]:,.0f}', xy=(oo_nav.index[-1], oo_nav.iloc[-1]), 
                     xytext=(10, 0), textcoords='offset points', color='#c0392b', weight='bold')
    if cc_nav is not None:
        plt.annotate(f'{cc_nav.iloc[-1]:,.0f}', xy=(cc_nav.index[-1], cc_nav.iloc[-1]), 
                     xytext=(10, 5), textcoords='offset points', color='#27ae60', weight='bold')

    plt.tight_layout()
    plt.savefig('performance_comparison.png', dpi=150)
    print("Optimization comparison chart saved to performance_comparison.png")

if __name__ == "__main__":
    plot_comparison()

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os

def plot_equity(csv_path, output_path):
    print(f"Reading {csv_path}...")
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return

    if df.empty:
        print("CSV is empty.")
        return

    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date')

    # Plot
    plt.figure(figsize=(12, 6))
    
    # Plot Equity
    plt.plot(df.index, df['assets'], label='Total Assets', color='#1f77b4', linewidth=2)
    
    # Benchmark (Buy & Hold) - Optional, just drawing a line from start to end of Index if available
    # For now, just assets
    
    # Highlight Max Drawdown
    # Calculate drawdown again just to be sure
    roll_max = df['assets'].cummax()
    drawdown = (df['assets'] - roll_max) / roll_max
    
    # Fill drawdown area
    plt.fill_between(df.index, df['assets'], roll_max, color='red', alpha=0.1, label='Drawdown Area')

    # Formatting
    plt.title('Ptrade Strategy Equity Curve (2023-Now)', fontsize=14)
    plt.xlabel('Date')
    plt.ylabel('Total Assets (CNY)')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    # Add text for final stats
    final_assets = df['assets'].iloc[-1]
    ret = (final_assets - 1000000) / 1000000 * 100
    max_dd = drawdown.min() * 100
    
    stats_text = (
        f"Total Return: {ret:.2f}%\n"
        f"Final Assets: {final_assets:,.0f}\n"
        f"Max Drawdown: {max_dd:.2f}%"
    )
    plt.annotate(stats_text, xy=(0.02, 0.95), xycoords='axes fraction', 
                 bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.9))

    # Date formatting
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"Chart saved to {output_path}")

if __name__ == "__main__":
    csv_file = r"C:\Users\liuqi\quant_system_v2\backtesting\ptrade_equity.csv"
    # Save to artifacts dir
    out_file = r"C:\Users\liuqi\.gemini\antigravity\brain\02be0ea3-cfcb-4a7a-b56a-03b516b44ee9\ptrade_equity_curve_2023_2026.png"
    
    plot_equity(csv_file, out_file)

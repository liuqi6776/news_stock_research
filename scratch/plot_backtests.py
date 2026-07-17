import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
artifact_dir = r"C:\Users\liuqi\.gemini\antigravity\brain\43c45e4b-95ce-46fb-b48f-b2c549bd4814"

# Paths to the CSV files
path_dragon = os.path.join(ROOT_DIR, "dragon_daily_news_equity.csv")
# For option-enhanced strategy, we can load super_weekly_news_equity.csv or similar
path_option = os.path.join(ROOT_DIR, "super_weekly_news_equity.csv")

def analyze_equity(df, name):
    df = df.copy()
    date_col = 'date' if 'date' in df.columns else df.columns[0]
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col).reset_index(drop=True)
    
    # Capital
    nav = df['nav']
    initial_cap = nav.iloc[0]
    final_cap = nav.iloc[-1]
    
    # Returns
    df['ret'] = nav.pct_change()
    total_ret = final_cap / initial_cap - 1
    
    # Ann return
    n_days = len(df)
    years = n_days / 252.0
    ann_ret = (1 + total_ret) ** (1 / years) - 1 if years > 0 else 0
    
    # Vol & Sharpe
    vol = df['ret'].std() * np.sqrt(252)
    sharpe = ann_ret / vol if vol > 0 else 0
    
    # Max Drawdown
    cum_max = nav.cummax()
    dd = (nav - cum_max) / cum_max
    mdd = dd.min()
    
    print(f"[{name}]")
    print(f"  Total Return: {total_ret:.2%}")
    print(f"  Annual Return: {ann_ret:.2%}")
    print(f"  Volatility: {vol:.2%}")
    print(f"  Sharpe Ratio: {sharpe:.2f}")
    print(f"  Max Drawdown: {mdd:.2%}")
    print(f"  Trading Days: {n_days}")
    print(f"  Date Range: {df[date_col].min().strftime('%Y-%m-%d')} to {df[date_col].max().strftime('%Y-%m-%d')}")
    print("-" * 50)
    
    return df, {
        'Name': name,
        'Total Return': f"{total_ret:.2%}",
        'Annual Return': f"{ann_ret:.2%}",
        'Sharpe Ratio': f"{sharpe:.2f}",
        'Max Drawdown': f"{mdd:.2%}",
        'Days': n_days
    }

def main():
    results = []
    
    # 1. Load Dragon
    if os.path.exists(path_dragon):
        df_dragon = pd.read_csv(path_dragon)
        df_dragon, stats_dragon = analyze_equity(df_dragon, "Daily Dragon Strategy (THS Rank + News)")
        results.append((df_dragon, stats_dragon))
    else:
        print(f"Dragon CSV not found at {path_dragon}")
        
    # 2. Load Option Enhanced
    if os.path.exists(path_option):
        df_option = pd.read_csv(path_option)
        df_option, stats_option = analyze_equity(df_option, "Option-Enhanced Strategy (All-Market + Options)")
        results.append((df_option, stats_option))
    else:
        print(f"Option CSV not found at {path_option}")
        
    if len(results) < 2:
        print("Error: Could not load both backtests!")
        return
        
    # Plotting
    plt.figure(figsize=(12, 7))
    
    # Color palette
    colors = {
        "Daily Dragon Strategy (THS Rank + News)": "#E65100",  # Amber/Orange
        "Option-Enhanced Strategy (All-Market + Options)": "#1976D2"  # Blue
    }
    
    for df, stats in results:
        name = stats['Name']
        date_col = 'date' if 'date' in df.columns else df.columns[0]
        # Align starting point to 1.0 (or 100,000) for clean comparison
        nav_normalized = df['nav'] / df['nav'].iloc[0] * 100000.0
        plt.plot(df[date_col], nav_normalized, label=name, color=colors[name], linewidth=2.5)
        
    plt.title("A-Share Quantitative Strategy Backtest Comparison (2024 - 2026)", fontsize=14, fontweight='bold', pad=15)
    plt.xlabel("Date", fontsize=12)
    plt.ylabel("Normalized Net Asset Value (NAV)", fontsize=12)
    plt.legend(fontsize=11, loc="upper left")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"¥{x:,.0f}"))
    plt.tight_layout()
    
    # Save the comparison chart as an artifact image
    out_img = os.path.join(artifact_dir, "strategy_backtest_comparison.png")
    plt.savefig(out_img, dpi=150)
    print(f"\nSaved comparison chart to: {out_img}")
    
    # Print comparison table
    df_summary = pd.DataFrame([r[1] for r in results])
    print("\nSummary Table:")
    print(df_summary.to_string(index=False))

if __name__ == "__main__":
    main()

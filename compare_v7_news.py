import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

OUT_DIR = r'C:\Users\liuqi\quant_system_v2'

def calc_metrics(df):
    if len(df) < 2: return None
    df = df.copy()
    df['ret'] = df['nav'].pct_change()
    
    total_ret = df['nav'].iloc[-1] / df['nav'].iloc[0] - 1
    # assuming approx 252 trading days per year
    years = len(df) / 252.0
    if years <= 0: years = 1
    annual_ret = (1 + total_ret) ** (1 / years) - 1
    annual_vol = df['ret'].std() * np.sqrt(252)
    sharpe = annual_ret / annual_vol if annual_vol > 0 else 0
    
    df['cummax'] = df['nav'].cummax()
    df['drawdown'] = (df['nav'] - df['cummax']) / df['cummax']
    mdd = df['drawdown'].min()
    
    win_rate = (df['ret'] > 0).sum() / (df['ret'].notna().sum()) if df['ret'].notna().sum() > 0 else 0
    
    return {
        'Total Return': f"{total_ret:.2%}",
        'Annual Return': f"{annual_ret:.2%}",
        'Max Drawdown': f"{mdd:.2%}",
        'Sharpe': f"{sharpe:.2f}",
        'Win Rate': f"{win_rate:.1%}",
        'Days': len(df)
    }

def main():
    v7_path = os.path.join(OUT_DIR, 'super_weekly_equity.csv')
    news_path = os.path.join(OUT_DIR, 'super_weekly_news_equity.csv')
    
    if not os.path.exists(v7_path) or not os.path.exists(news_path):
        print(f"Missing required CSV files. checked: {v7_path}, {news_path}")
        return
        
    df_v7 = pd.read_csv(v7_path)
    df_v7['date'] = pd.to_datetime(df_v7['date'].astype(str))
    
    df_news = pd.read_csv(news_path)
    df_news['date'] = pd.to_datetime(df_news['date'].astype(str))
    
    # Filter to identical date range
    start_dt = max(df_v7['date'].min(), df_news['date'].min())
    end_dt = min(df_v7['date'].max(), df_news['date'].max())
    
    df_v7 = df_v7[(df_v7['date'] >= start_dt) & (df_v7['date'] <= end_dt)].reset_index(drop=True)
    df_news = df_news[(df_news['date'] >= start_dt) & (df_news['date'] <= end_dt)].reset_index(drop=True)
    
    # Rebase to starting value for fair comparison
    base_nav = 100000.0
    if not df_v7.empty:
        df_v7['nav'] = df_v7['nav'] / df_v7['nav'].iloc[0] * base_nav
    if not df_news.empty:
        df_news['nav'] = df_news['nav'] / df_news['nav'].iloc[0] * base_nav
    
    metrics_v7 = calc_metrics(df_v7)
    metrics_news = calc_metrics(df_news)
    
    print("=" * 100)
    print(f"{'Strategy':<30} | {'Total Ret':>10} | {'Annual Ret':>10} | {'Max DD':>10} | {'Sharpe':>8} | {'Win Rate':>8} | {'Days':>5}")
    print("=" * 100)
    if metrics_v7:
        print(f"{'V7 (super_weekly)':<30} | {metrics_v7['Total Return']:>10} | {metrics_v7['Annual Return']:>10} | {metrics_v7['Max Drawdown']:>10} | {metrics_v7['Sharpe']:>8} | {metrics_v7['Win Rate']:>8} | {metrics_v7['Days']:>5}")
    if metrics_news:
        print(f"{'Current (with News Features)':<30} | {metrics_news['Total Return']:>10} | {metrics_news['Annual Return']:>10} | {metrics_news['Max Drawdown']:>10} | {metrics_news['Sharpe']:>8} | {metrics_news['Win Rate']:>8} | {metrics_news['Days']:>5}")
    print("=" * 100)
    
    plt.figure(figsize=(10,6))
    if not df_v7.empty:
        plt.plot(df_v7['date'], df_v7['nav'], label='V7 (super_weekly)')
    if not df_news.empty:
        plt.plot(df_news['date'], df_news['nav'], label='Current (with News Features)')
        
    plt.axhline(y=base_nav, color='gray', linestyle='--', alpha=0.5)
    plt.title(f'Detailed Comparison: Current vs V7 ({start_dt.strftime("%Y-%m-%d")} to {end_dt.strftime("%Y-%m-%d")})')
    plt.xlabel('Date')
    plt.ylabel('Rebased NAV')
    plt.legend()
    plt.grid(True)
    out_img = os.path.join(OUT_DIR, 'v7_vs_current_comparison.png')
    plt.savefig(out_img)
    print(f"Comparison plot saved to {out_img}")
    
    # Save the output to a text file for the agent to easily read and summarize
    with open('v7_vs_current_metrics.txt', 'w', encoding='utf-8') as f:
        f.write("Strategy,Total Return,Annual Return,Max Drawdown,Sharpe,Win Rate\n")
        f.write(f"V7,{metrics_v7['Total Return']},{metrics_v7['Annual Return']},{metrics_v7['Max Drawdown']},{metrics_v7['Sharpe']},{metrics_v7['Win Rate']}\n")
        f.write(f"Current,{metrics_news['Total Return']},{metrics_news['Annual Return']},{metrics_news['Max Drawdown']},{metrics_news['Sharpe']},{metrics_news['Win Rate']}\n")

if __name__ == '__main__':
    main()

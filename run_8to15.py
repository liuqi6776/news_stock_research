
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, 'results_duobao')

def main():
    trades_csv = os.path.join(OUTPUT_DIR, 'preloaded_trades.csv')
    if not os.path.exists(trades_csv):
        print("Error: preloaded_trades.csv not found!")
        return
    
    trades_df = pd.read_csv(trades_csv)
    print(f"Loaded {len(trades_df)} trades")
    
    take_profit_list = [0.08, 0.09, 0.10, 0.11, 0.12, 0.13, 0.14, 0.15]
    results = []
    
    print("\n" + "="*80)
    print("Testing take-profit 8% - 15%")
    print("="*80)
    
    for tp in take_profit_list:
        initial_cap = 100000.0
        capital = initial_cap
        equity = []
        
        for date_t2, group in trades_df.groupby('date_t2', sort=True):
            alloc = capital / len(group)
            day_pnl = 0.0
            
            for _, trade in group.iterrows():
                buy_price = trade['open']
                
                if trade['high'] &gt;= buy_price * (1 + tp):
                    sell_price = buy_price * (1 + tp)
                else:
                    sell_price = trade['close']
                
                ret = (sell_price / buy_price) - 1
                ret -= 0.0015
                day_pnl += alloc * ret
            
            capital += day_pnl
            equity.append({'date': pd.to_datetime(date_t2), 'nav': capital})
        
        total_ret = capital / initial_cap - 1
        years = len(equity) / 252.0
        ann_ret = (1 + total_ret) ** (1 / years) - 1 if years &gt; 0 else 0
        eq_df = pd.DataFrame(equity)
        df_ret = eq_df['nav'].pct_change()
        mdd = ((eq_df['nav'] - eq_df['nav'].cummax()) / eq_df['nav'].cummax()).min()
        vol = df_ret.std() * np.sqrt(252)
        sharpe = ann_ret / vol if vol &gt; 0 else 0
        
        results.append({
            'take_profit_pct': tp,
            'total_ret': total_ret,
            'ann_ret': ann_ret,
            'mdd': mdd,
            'sharpe': sharpe,
            'num_trades': len(equity),
            'final_cap': capital,
            'equity_df': eq_df
        })
        print(f"\nTake-profit {tp*100:.0f}%:")
        print(f"  Total: {total_ret:+.2%}, Annual: {ann_ret:+.2%}")
        print(f"  Sharpe: {sharpe:.2f}, MDD: {mdd:.2%}")
        print(f"  Days: {len(equity)}, Final: ¥{capital:,.2f}")
    
    print("\n" + "="*80)
    print("Comparison Results")
    print("="*80)
    
    df_summary = pd.DataFrame([{
        'Take Profit': f"{r['take_profit_pct']*100:.0f}%",
        'Total Return': f"{r['total_ret']:+.2%}",
        'Annual Return': f"{r['ann_ret']:+.2%}",
        'Max Drawdown': f"{r['mdd']:.2%}",
        'Sharpe Ratio': f"{r['sharpe']:.2f}",
        'Trading Days': r['num_trades'],
        'Final Capital': f"¥{r['final_cap']:,.2f}"
    } for r in results])
    
    print(df_summary.to_string(index=False))
    
    summary_csv = os.path.join(OUTPUT_DIR, 'take_profit_comparison_8to15.csv')
    df_summary.to_csv(summary_csv, index=False)
    print(f"\nSaved to: {summary_csv}")
    
    plt.figure(figsize=(16, 10))
    for r in results:
        label = f"Take Profit {r['take_profit_pct']*100:.0f}%"
        plt.plot(r['equity_df']['date'], r['equity_df']['nav'], label=label, linewidth=2)
    plt.title('Take Profit Comparison (8% - 15%)', fontsize=16, fontweight='bold')
    plt.xlabel('Date', fontsize=14)
    plt.ylabel('Capital', fontsize=14)
    plt.legend(fontsize=12, loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    comparison_png = os.path.join(OUTPUT_DIR, 'take_profit_comparison_8to15.png')
    plt.savefig(comparison_png, dpi=150)
    print(f"Saved chart to: {comparison_png}")
    
    best_by_sharpe = max(results, key=lambda x: x['sharpe'])
    best_by_return = max(results, key=lambda x: x['total_ret'])
    
    print("\n" + "="*80)
    print("Best Strategy")
    print("="*80)
    print(f"Best by Sharpe: Take Profit {best_by_sharpe['take_profit_pct']*100:.0f}%")
    print(f"  Sharpe: {best_by_sharpe['sharpe']:.2f}, Total: {best_by_sharpe['total_ret']:+.2%}, MDD: {best_by_sharpe['mdd']:.2%}")
    print(f"\nBest by Return: Take Profit {best_by_return['take_profit_pct']*100:.0f}%")
    print(f"  Total: {best_by_return['total_ret']:+.2%}, Sharpe: {best_by_return['sharpe']:.2f}, MDD: {best_by_return['mdd']:.2%}")
    print("="*80)

if __name__ == "__main__":
    main()


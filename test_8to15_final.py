
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
        print("未找到预加载文件")
        return
    
    trades_df = pd.read_csv(trades_csv)
    print(f"已加载 {len(trades_df)} 条交易数据")
    
    take_profit_list = [0.08, 0.09, 0.10, 0.11, 0.12, 0.13, 0.14, 0.15]
    results = []
    
    print("\n" + "="*80)
    print("在预加载数据上测试止盈点 (8% - 15%)")
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
                
                if trade['high'] >= buy_price * (1 + tp):
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
        ann_ret = (1 + total_ret) ** (1 / years) - 1 if years > 0 else 0
        eq_df = pd.DataFrame(equity)
        df_ret = eq_df['nav'].pct_change()
        mdd = ((eq_df['nav'] - eq_df['nav'].cummax()) / eq_df['nav'].cummax()).min()
        vol = df_ret.std() * np.sqrt(252)
        sharpe = ann_ret / vol if vol > 0 else 0
        
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
        print(f"\n止盈 {tp*100:.0f}%:")
        print(f"  总收益: {total_ret:+.2%}, 年化: {ann_ret:+.2%}")
        print(f"  夏普: {sharpe:.2f}, 回撤: {mdd:.2%}")
        print(f"  交易天数: {len(equity)}, 最终资金: ¥{capital:,.2f}")
    
    print("\n" + "="*80)
    print("回测结果对比 (8% - 15%)")
    print("="*80)
    
    df_summary = pd.DataFrame([{
        '止盈点': f"{r['take_profit_pct']*100:.0f}%",
        '总收益率': f"{r['total_ret']:+.2%}",
        '年化收益率': f"{r['ann_ret']:+.2%}",
        '最大回撤': f"{r['mdd']:.2%}",
        '夏普比率': f"{r['sharpe']:.2f}",
        '交易天数': r['num_trades'],
        '最终资金': f"¥{r['final_cap']:,.2f}"
    } for r in results])
    
    print(df_summary.to_string(index=False))
    
    summary_csv = os.path.join(OUTPUT_DIR, 'take_profit_comparison_8to15.csv')
    df_summary.to_csv(summary_csv, index=False)
    print(f"\n对比结果已保存: {summary_csv}")
    
    plt.figure(figsize=(16, 10))
    
    for r in results:
        label = f"止盈 {r['take_profit_pct']*100:.0f}%"
        plt.plot(r['equity_df']['date'], r['equity_df']['nav'], label=label, linewidth=2)
    
    plt.title('不同止盈点策略净值对比 (8% - 15%)', fontsize=16, fontweight='bold')
    plt.xlabel('日期', fontsize=14)
    plt.ylabel('资金', fontsize=14)
    plt.legend(fontsize=12, loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    comparison_png = os.path.join(OUTPUT_DIR, 'take_profit_comparison_8to15.png')
    plt.savefig(comparison_png, dpi=150)
    print(f"对比图已保存: {comparison_png}")
    
    best_by_sharpe = max(results, key=lambda x: x['sharpe'])
    best_by_return = max(results, key=lambda x: x['total_ret'])
    
    print("\n" + "="*80)
    print("最优策略推荐 (8% - 15%)")
    print("="*80)
    print(f"按夏普比率最优: 止盈 {best_by_sharpe['take_profit_pct']*100:.0f}%")
    print(f"  夏普: {best_by_sharpe['sharpe']:.2f}, 总收益: {best_by_sharpe['total_ret']:+.2%}, 回撤: {best_by_sharpe['mdd']:.2%}")
    print(f"\n按总收益最优: 止盈 {best_by_return['take_profit_pct']*100:.0f}%")
    print(f"  总收益: {best_by_return['total_ret']:+.2%}, 夏普: {best_by_return['sharpe']:.2f}, 回撤: {best_by_return['mdd']:.2%}")
    print("="*80)

if __name__ == "__main__":
    main()


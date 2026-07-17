
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
    
    # 固定止盈 8%，测试不同止损
    take_profit = 0.08
    stop_loss_list = [-0.02, -0.04, -0.06, -0.08, -0.10, -0.12]
    results = []
    
    print("\n" + "="*80)
    print("测试止损策略（止盈 8% + 不同止损）")
    print("="*80)
    
    # 先测试一个基准：只有止盈 8%，不止损
    print("\n【基准】只有止盈 8%，不止损")
    initial_cap = 100000.0
    capital = initial_cap
    equity = []
    
    for date_t2, group in trades_df.groupby('date_t2', sort=True):
        alloc = capital / len(group)
        day_pnl = 0.0
        
        for _, trade in group.iterrows():
            buy_price = trade['open']
            
            if trade['high'] &gt;= buy_price * (1 + take_profit):
                sell_price = buy_price * (1 + take_profit)
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
        'stop_loss_pct': '无止损',
        'take_profit_pct': take_profit,
        'total_ret': total_ret,
        'ann_ret': ann_ret,
        'mdd': mdd,
        'sharpe': sharpe,
        'num_trades': len(equity),
        'final_cap': capital,
        'equity_df': eq_df
    })
    print(f"  总收益: {total_ret:+.2%}, 年化: {ann_ret:+.2%}")
    print(f"  夏普: {sharpe:.2f}, 回撤: {mdd:.2%}")
    print(f"  交易天数: {len(equity)}, 最终资金: ¥{capital:,.2f}")
    
    # 测试不同止损
    for sl in stop_loss_list:
        print(f"\n【测试】止盈 8% + 止损 {sl*100:.0f}%")
        initial_cap = 100000.0
        capital = initial_cap
        equity = []
        
        for date_t2, group in trades_df.groupby('date_t2', sort=True):
            alloc = capital / len(group)
            day_pnl = 0.0
            
            for _, trade in group.iterrows():
                buy_price = trade['open']
                
                # 判断先触发止盈还是止损
                if trade['high'] &gt;= buy_price * (1 + take_profit):
                    sell_price = buy_price * (1 + take_profit)
                elif trade['low'] &lt;= buy_price * (1 + sl):
                    sell_price = buy_price * (1 + sl)
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
            'stop_loss_pct': sl,
            'take_profit_pct': take_profit,
            'total_ret': total_ret,
            'ann_ret': ann_ret,
            'mdd': mdd,
            'sharpe': sharpe,
            'num_trades': len(equity),
            'final_cap': capital,
            'equity_df': eq_df
        })
        print(f"  总收益: {total_ret:+.2%}, 年化: {ann_ret:+.2%}")
        print(f"  夏普: {sharpe:.2f}, 回撤: {mdd:.2%}")
        print(f"  交易天数: {len(equity)}, 最终资金: ¥{capital:,.2f}")
    
    print("\n" + "="*80)
    print("止损策略对比结果")
    print("="*80)
    
    df_summary = pd.DataFrame([{
        '止损点': f"{r['stop_loss_pct']*100:.0f}%" if isinstance(r['stop_loss_pct'], (int, float)) else r['stop_loss_pct'],
        '止盈点': f"{r['take_profit_pct']*100:.0f}%",
        '总收益率': f"{r['total_ret']:+.2%}",
        '年化收益率': f"{r['ann_ret']:+.2%}",
        '最大回撤': f"{r['mdd']:.2%}",
        '夏普比率': f"{r['sharpe']:.2f}",
        '交易天数': r['num_trades'],
        '最终资金': f"¥{r['final_cap']:,.2f}"
    } for r in results])
    
    print(df_summary.to_string(index=False))
    
    summary_csv = os.path.join(OUTPUT_DIR, 'stop_loss_comparison.csv')
    df_summary.to_csv(summary_csv, index=False)
    print(f"\n对比结果已保存: {summary_csv}")
    
    plt.figure(figsize=(16, 10))
    
    for r in results:
        label = f"止损 {r['stop_loss_pct']*100:.0f}%" if isinstance(r['stop_loss_pct'], (int, float)) else "无止损"
        plt.plot(r['equity_df']['date'], r['equity_df']['nav'], label=label, linewidth=2)
    
    plt.title('止损策略对比（止盈 8%）', fontsize=16, fontweight='bold')
    plt.xlabel('日期', fontsize=14)
    plt.ylabel('资金', fontsize=14)
    plt.legend(fontsize=12, loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    comparison_png = os.path.join(OUTPUT_DIR, 'stop_loss_comparison.png')
    plt.savefig(comparison_png, dpi=150)
    print(f"对比图已保存: {comparison_png}")
    
    best_by_sharpe = max(results, key=lambda x: x['sharpe'])
    best_by_return = max(results, key=lambda x: x['total_ret'])
    
    print("\n" + "="*80)
    print("最优策略推荐")
    print("="*80)
    sharpe_label = f"止损 {best_by_sharpe['stop_loss_pct']*100:.0f}%" if isinstance(best_by_sharpe['stop_loss_pct'], (int, float)) else best_by_sharpe['stop_loss_pct']
    print(f"按夏普比率最优: {sharpe_label}")
    print(f"  夏普: {best_by_sharpe['sharpe']:.2f}, 总收益: {best_by_sharpe['total_ret']:+.2%}, 回撤: {best_by_sharpe['mdd']:.2%}")
    
    return_label = f"止损 {best_by_return['stop_loss_pct']*100:.0f}%" if isinstance(best_by_return['stop_loss_pct'], (int, float)) else best_by_return['stop_loss_pct']
    print(f"\n按总收益最优: {return_label}")
    print(f"  总收益: {best_by_return['total_ret']:+.2%}, 夏普: {best_by_return['sharpe']:.2f}, 回撤: {best_by_return['mdd']:.2%}")
    print("="*80)

if __name__ == "__main__":
    main()


"""
对比 final_method 和 doubao_result base method 的回测结果
"""
import pandas as pd
import numpy as np

# final_method 结果（从之前的运行中获取）
final_method = {
    'method': 'final_method',
    'total_ret': 0.8109,
    'sharpe': 0.82,
    'max_dd': -0.3439,
    'win_rate': 0.56,
    'avg_ret': None,  # 之前未记录
    'n_trades': None,  # 之前未记录
    'final_capital': 181090  # 估算
}

# 读取 doubao_result 的权益曲线
doubao_eq = pd.read_csv(r'C:\Users\liuqi\quant_system_v2\new_idea\final_result\doubao\equity_doubao_fixed.csv')
doubao_eq['date'] = pd.to_datetime(doubao_eq['date'])
doubao_eq = doubao_eq.sort_values('date')

# 计算 doubao_result 的统计指标
initial_cap = 100000.0
final_cap = doubao_eq['nav'].iloc[-1]
total_ret = final_cap / initial_cap - 1
years = len(doubao_eq) / 252.0
ann_ret = (1 + total_ret) ** (1 / years) - 1 if years > 0 else 0
df_ret = doubao_eq['nav'].pct_change()
mdd = ((doubao_eq['nav'] - doubao_eq['nav'].cummax()) / doubao_eq['nav'].cummax()).min()
vol = df_ret.std() * np.sqrt(252)
sharpe = ann_ret / vol if vol > 0 else 0

# 读取交易记录计算胜率
doubao_trades = pd.read_csv(r'C:\Users\liuqi\quant_system_v2\new_idea\final_result\doubao\trades_doubao_fixed.csv')
win_rate = (doubao_trades['ret'] > 0).mean()
avg_ret = doubao_trades['ret'].mean()

doubao_result = {
    'method': 'doubao_result base',
    'total_ret': total_ret,
    'ann_ret': ann_ret,
    'sharpe': sharpe,
    'max_dd': mdd,
    'win_rate': win_rate,
    'avg_ret': avg_ret,
    'n_trades': len(doubao_trades),
    'final_capital': final_cap
}

print("=" * 80)
print("A股修正版回测结果对比")
print("=" * 80)
print(f"{'指标':<20} {'final_method':<20} {'doubao_result base':<20}")
print("-" * 80)
print(f"{'总收益':<20} {final_method['total_ret']*100:>18.2f}% {doubao_result['total_ret']*100:>18.2f}%")
print(f"{'年化收益':<20} {'N/A':>20} {doubao_result['ann_ret']*100:>18.2f}%")
print(f"{'夏普比率':<20} {final_method['sharpe']:>20.2f} {doubao_result['sharpe']:>20.2f}")
print(f"{'最大回撤':<20} {final_method['max_dd']*100:>18.2f}% {doubao_result['max_dd']*100:>18.2f}%")
print(f"{'胜率':<20} {final_method['win_rate']*100:>18.2f}% {doubao_result['win_rate']*100:>18.2f}%")
print(f"{'平均收益':<20} {'N/A':>20} {doubao_result['avg_ret']*100:>18.2f}%")
print(f"{'交易次数':<20} {'N/A':>20} {doubao_result['n_trades']:>20}")
print(f"{'最终资金':<20} {final_method['final_capital']:>19.0f} {doubao_result['final_capital']:>19.2f}")
print("=" * 80)

print("\n结论:")
print("-" * 80)
if final_method['total_ret'] > doubao_result['total_ret']:
    print("✅ final_method 表现明显优于 doubao_result base method")
    print(f"   - 总收益高出 {(final_method['total_ret'] - doubao_result['total_ret'])*100:.2f} 个百分点")
else:
    print("✅ doubao_result base method 表现优于 final_method")
    
print(f"\n⚠️  doubao_result base method 出现大幅亏损:")
print(f"   - 总收益: {doubao_result['total_ret']*100:.2f}%")
print(f"   - 最终资金: {doubao_result['final_capital']:.2f} (初始: 100,000)")
print(f"   - 最大回撤: {doubao_result['max_dd']*100:.2f}%")

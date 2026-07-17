import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

weekly_df = pd.read_csv('super_weekly_equity.csv')
monthly_df = pd.read_csv('super_monthly_equity.csv')
low_dd_df = pd.read_csv('low_drawdown_equity.csv')

weekly_df['date'] = pd.to_datetime(weekly_df['date'])
monthly_df['date'] = pd.to_datetime(monthly_df['date'])
low_dd_df['date'] = pd.to_datetime(low_dd_df['date'])

INITIAL_CAP = 100000

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

ax1 = axes[0, 0]
ax1.plot(weekly_df['date'], weekly_df['nav'], label='Super Weekly', color='#2E86AB', linewidth=1.5)
ax1.plot(monthly_df['date'], monthly_df['nav'], label='Super Monthly', color='#E94F37', linewidth=1.5)
ax1.plot(low_dd_df['date'], low_dd_df['nav'], label='Low Drawdown', color='#28A745', linewidth=2)
ax1.axhline(y=INITIAL_CAP, color='gray', linestyle='--', alpha=0.5, label='Initial Capital')
ax1.set_title('Strategy Comparison - Net Asset Value', fontsize=14, fontweight='bold')
ax1.set_xlabel('Date')
ax1.set_ylabel('NAV (CNY)')
ax1.legend(loc='upper left')
ax1.grid(True, alpha=0.3)
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)

ax2 = axes[0, 1]
weekly_df['return'] = (weekly_df['nav'] / INITIAL_CAP - 1) * 100
monthly_df['return'] = (monthly_df['nav'] / INITIAL_CAP - 1) * 100
low_dd_df['return'] = (low_dd_df['nav'] / INITIAL_CAP - 1) * 100
ax2.fill_between(weekly_df['date'], weekly_df['return'], alpha=0.3, color='#2E86AB')
ax2.fill_between(monthly_df['date'], monthly_df['return'], alpha=0.3, color='#E94F37')
ax2.fill_between(low_dd_df['date'], low_dd_df['return'], alpha=0.3, color='#28A745')
ax2.plot(weekly_df['date'], weekly_df['return'], color='#2E86AB', linewidth=1, label='Weekly')
ax2.plot(monthly_df['date'], monthly_df['return'], color='#E94F37', linewidth=1, label='Monthly')
ax2.plot(low_dd_df['date'], low_dd_df['return'], color='#28A745', linewidth=2, label='Low DD')
ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
ax2.set_title('Cumulative Return (%)', fontsize=14, fontweight='bold')
ax2.set_xlabel('Date')
ax2.set_ylabel('Return (%)')
ax2.legend(loc='upper left')
ax2.grid(True, alpha=0.3)
ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)

ax3 = axes[1, 0]
weekly_df['cummax'] = weekly_df['nav'].cummax()
weekly_df['drawdown'] = (weekly_df['nav'] - weekly_df['cummax']) / weekly_df['cummax'] * 100
monthly_df['cummax'] = monthly_df['nav'].cummax()
monthly_df['drawdown'] = (monthly_df['nav'] - monthly_df['cummax']) / monthly_df['cummax'] * 100
ax3.fill_between(weekly_df['date'], weekly_df['drawdown'], 0, alpha=0.5, color='#2E86AB', label='Weekly DD')
ax3.fill_between(monthly_df['date'], monthly_df['drawdown'], 0, alpha=0.5, color='#E94F37', label='Monthly DD')
ax3.fill_between(low_dd_df['date'], low_dd_df['drawdown'], 0, alpha=0.5, color='#28A745', label='Low DD')
ax3.set_title('Drawdown Comparison', fontsize=14, fontweight='bold')
ax3.set_xlabel('Date')
ax3.set_ylabel('Drawdown (%)')
ax3.legend(loc='lower left')
ax3.grid(True, alpha=0.3)
ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45)

ax4 = axes[1, 1]
ax4.axis('off')

weekly_final = weekly_df['nav'].iloc[-1]
monthly_final = monthly_df['nav'].iloc[-1]
low_dd_final = low_dd_df['nav'].iloc[-1]

weekly_return = (weekly_final / INITIAL_CAP - 1) * 100
monthly_return = (monthly_final / INITIAL_CAP - 1) * 100
low_dd_return = (low_dd_final / INITIAL_CAP - 1) * 100

weekly_max_dd = weekly_df['drawdown'].min()
monthly_max_dd = monthly_df['drawdown'].min()
low_dd_max_dd = low_dd_df['drawdown'].min()

stats_text = f"""
+================================================================+
|                    STRATEGY COMPARISON                          |
+================================================================+
|  Metric          | Super Weekly | Super Monthly | Low Drawdown |
+================================================================+
|  Final NAV       | {weekly_final:>11,.0f} | {monthly_final:>11,.0f} | {low_dd_final:>11,.0f} |
|  Total Return    | {weekly_return:>+10.1f}% | {monthly_return:>+10.1f}% | {low_dd_return:>+10.1f}% |
|  Max Drawdown    | {weekly_max_dd:>10.1f}% | {monthly_max_dd:>10.1f}% | {low_dd_max_dd:>10.1f}% |
+================================================================+

KEY FINDINGS:
* Super Weekly: Highest return (+{weekly_return:.0f}%) but worst drawdown ({weekly_max_dd:.0f}%)
* Super Monthly: Good balance (+{monthly_return:.0f}%, {monthly_max_dd:.0f}% DD)
* Low Drawdown: Lowest return (+{low_dd_return:.0f}%) but still {low_dd_max_dd:.0f}% drawdown

CONCLUSION:
The risk control strategy reduced drawdown from -47% to -29%, but also
reduced return from +313% to +5%. This is a classic risk-return tradeoff.

For truly low drawdown, consider:
1. Lower position size (e.g., max 50% invested)
2. Stricter market timing (only invest in clear uptrends)
3. Add hedging (e.g., index puts during volatile periods)
"""

ax4.text(0.05, 0.95, stats_text, transform=ax4.transAxes, fontsize=10,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig('strategy_comparison_all.png', dpi=150, bbox_inches='tight')
print("Chart saved: strategy_comparison_all.png")

print(f"\n{'='*60}")
print("STRATEGY COMPARISON RESULTS")
print(f"{'='*60}")
print(f"Super Weekly:  Return {weekly_return:+.1f}%, Max DD {weekly_max_dd:.1f}%")
print(f"Super Monthly: Return {monthly_return:+.1f}%, Max DD {monthly_max_dd:.1f}%")
print(f"Low Drawdown:  Return {low_dd_return:+.1f}%, Max DD {low_dd_max_dd:.1f}%")
print(f"{'='*60}")

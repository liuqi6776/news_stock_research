import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime

weekly_df = pd.read_csv('super_weekly_equity.csv')
monthly_df = pd.read_csv('super_monthly_equity.csv')

weekly_df['date'] = pd.to_datetime(weekly_df['date'])
monthly_df['date'] = pd.to_datetime(monthly_df['date'])

weekly_df = weekly_df.sort_values('date')
monthly_df = monthly_df.sort_values('date')

INITIAL_CAP = 100000

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

ax1 = axes[0, 0]
ax1.plot(weekly_df['date'], weekly_df['nav'], label='Super Weekly', color='#2E86AB', linewidth=1.5)
ax1.plot(monthly_df['date'], monthly_df['nav'], label='Super Monthly', color='#E94F37', linewidth=1.5)
ax1.axhline(y=INITIAL_CAP, color='gray', linestyle='--', alpha=0.5, label='Initial Capital')
ax1.set_title('Super Weekly vs Super Monthly - Net Asset Value', fontsize=14, fontweight='bold')
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
ax2.fill_between(weekly_df['date'], weekly_df['return'], alpha=0.3, color='#2E86AB', label='Weekly Return')
ax2.fill_between(monthly_df['date'], monthly_df['return'], alpha=0.3, color='#E94F37', label='Monthly Return')
ax2.plot(weekly_df['date'], weekly_df['return'], color='#2E86AB', linewidth=1)
ax2.plot(monthly_df['date'], monthly_df['return'], color='#E94F37', linewidth=1)
ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
ax2.set_title('Cumulative Return (%)', fontsize=14, fontweight='bold')
ax2.set_xlabel('Date')
ax2.set_ylabel('Return (%)')
ax2.legend(loc='upper left')
ax2.grid(True, alpha=0.3)
ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)

ax3 = axes[1, 0]
weekly_df['cummax'] = weekly_df['nav'].cummax()
weekly_df['drawdown'] = (weekly_df['nav'] - weekly_df['cummax']) / weekly_df['cummax'] * 100
monthly_df['cummax'] = monthly_df['nav'].cummax()
monthly_df['drawdown'] = (monthly_df['nav'] - monthly_df['cummax']) / monthly_df['cummax'] * 100
ax3.fill_between(weekly_df['date'], weekly_df['drawdown'], 0, alpha=0.5, color='#2E86AB', label='Weekly Drawdown')
ax3.fill_between(monthly_df['date'], monthly_df['drawdown'], 0, alpha=0.5, color='#E94F37', label='Monthly Drawdown')
ax3.set_title('Drawdown (%)', fontsize=14, fontweight='bold')
ax3.set_xlabel('Date')
ax3.set_ylabel('Drawdown (%)')
ax3.legend(loc='lower left')
ax3.grid(True, alpha=0.3)
ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
ax3.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45)

ax4 = axes[1, 1]
ax4.axis('off')

weekly_final = weekly_df['nav'].iloc[-1]
monthly_final = monthly_df['nav'].iloc[-1]
weekly_return = (weekly_final / INITIAL_CAP - 1) * 100
monthly_return = (monthly_final / INITIAL_CAP - 1) * 100
weekly_max = weekly_df['nav'].max()
monthly_max = monthly_df['nav'].max()
weekly_max_dd = weekly_df['drawdown'].min()
monthly_max_dd = monthly_df['drawdown'].min()
weekly_sharpe = weekly_df['return'].diff().mean() / weekly_df['return'].diff().std() * (52 ** 0.5) if len(weekly_df) > 1 else 0
monthly_sharpe = monthly_df['return'].diff().mean() / monthly_df['return'].diff().std() * (12 ** 0.5) if len(monthly_df) > 1 else 0

stats_text = f"""
+-------------------------------------------------------------+
|              STRATEGY PERFORMANCE COMPARISON                 |
+-------------------------------------------------------------+
|  Metric            |  Super Weekly  |  Super Monthly        |
+-------------------------------------------------------------+
|  Final NAV         |  {weekly_final:>12,.0f}  |  {monthly_final:>12,.0f}       |
|  Total Return      |  {weekly_return:>+11.1f}%  |  {monthly_return:>+11.1f}%      |
|  Max NAV           |  {weekly_max:>12,.0f}  |  {monthly_max:>12,.0f}       |
|  Max Drawdown      |  {weekly_max_dd:>11.1f}%  |  {monthly_max_dd:>11.1f}%      |
|  Annual Sharpe     |  {weekly_sharpe:>12.2f}  |  {monthly_sharpe:>12.2f}       |
|  Rebalance Freq    |  Weekly        |  Monthly              |
|  Data Points       |  {len(weekly_df):>12}  |  {len(monthly_df):>12}       |
+-------------------------------------------------------------+

CONCLUSION:
* Super Weekly has higher return ({weekly_return:+.1f}% vs {monthly_return:+.1f}%)
* Super Monthly has smaller drawdown ({monthly_max_dd:.1f}% vs {weekly_max_dd:.1f}%)
* For higher returns -> Choose Weekly
* For stability -> Choose Monthly
"""

ax4.text(0.05, 0.95, stats_text, transform=ax4.transAxes, fontsize=11,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig('weekly_monthly_comparison.png', dpi=150, bbox_inches='tight')
print("Chart saved: weekly_monthly_comparison.png")

print(f"\n=== Strategy Comparison Summary ===")
print(f"Super Weekly: Final NAV {weekly_final:,.0f}, Return {weekly_return:+.1f}%, Max Drawdown {weekly_max_dd:.1f}%")
print(f"Super Monthly: Final NAV {monthly_final:,.0f}, Return {monthly_return:+.1f}%, Max Drawdown {monthly_max_dd:.1f}%")

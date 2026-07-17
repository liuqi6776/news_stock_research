import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

FINAL_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = r'D:\iquant_data\data_v2'
PRICE_DIR = os.path.join(DATA_DIR, 'data_day1')

def is_gem_or_star(ts_code):
    return ('300' in ts_code) or ('301' in ts_code) or ('688' in ts_code) or ('689' in ts_code)

def int_to_date(date_int):
    s = str(date_int)
    return datetime(int(s[:4]), int(s[4:6]), int(s[6:8]))

def get_next_trading_day(date_int, all_dates_set):
    current_dt = int_to_date(date_int)
    for i in range(1, 10):
        next_dt = current_dt + timedelta(days=i)
        next_int = int(next_dt.strftime('%Y%m%d'))
        if next_int in all_dates_set:
            return next_int
    return None

def backtest_with_tp(trades_df, take_profit, all_dates_set):
    initial_cap = 100000.0
    capital = initial_cap
    equity = []
    total_trades = 0
    cannot_sell_trades = 0
    for date_t2, group in trades_df.groupby('date_t2', sort=True):
        alloc = capital / len(group)
        day_pnl = 0.0
        for _, trade in group.iterrows():
            total_trades += 1
            ts_code = trade['ts_code']
            buy_price = trade['buy_price']
            sell_close = trade['sell_close']
            sell_high = trade['sell_high']
            sell_pre_close = trade['sell_pre_close']
            limit_down_pct = 0.8 if is_gem_or_star(ts_code) else 0.9
            limit_down_price = round(sell_pre_close * limit_down_pct, 2)
            is_cannot_sell = (sell_high == limit_down_price)
            if is_cannot_sell:
                cannot_sell_trades += 1
                date_t3 = get_next_trading_day(date_t2, all_dates_set)
                if date_t3:
                    p_t3 = os.path.join(PRICE_DIR, f"{date_t3}.parquet")
                    if os.path.exists(p_t3):
                        df_t3 = pd.read_parquet(p_t3, columns=['ts_code', 'open'])
                        t3_row = df_t3[df_t3['ts_code'] == ts_code]
                        if not t3_row.empty:
                            sell_price = t3_row.iloc[0]['open']
                        else:
                            sell_price = sell_close
                    else:
                        sell_price = sell_close
                else:
                    sell_price = sell_close
            elif take_profit and sell_high >= buy_price * (1 + take_profit):
                sell_price = buy_price * (1 + take_profit)
            else:
                sell_price = sell_close
            ret = (sell_price / buy_price) - 1 - 0.0015
            day_pnl += alloc * ret
        capital += day_pnl
        equity.append({'date': int_to_date(date_t2), 'nav': capital})
    eq_df = pd.DataFrame(equity)
    if len(eq_df) == 0:
        return {}, []
    total_ret = capital / initial_cap - 1
    years = len(eq_df) / 252.0
    ann_ret = (1 + total_ret) ** (1 / years) - 1 if years > 0 else 0
    df_ret = eq_df['nav'].pct_change()
    mdd = ((eq_df['nav'] - eq_df['nav'].cummax()) / eq_df['nav'].cummax()).min()
    vol = df_ret.std() * np.sqrt(252)
    sharpe = ann_ret / vol if vol > 0 else 0
    return {'total': total_ret, 'ann': ann_ret, 'sharpe': sharpe, 'mdd': mdd,
            'trades': total_trades, 'cannot_sell': cannot_sell_trades, 'final_nav': capital}, equity

def main():
    all_dates = sorted([f.replace('.parquet', '') for f in os.listdir(PRICE_DIR) if f.endswith('.parquet')])
    all_dates_set = set(int(d) for d in all_dates)

    strategies = {}
    for sname, sub in [('doubao_result', 'doubao'), ('NewIdea S2', 'NewIdea_S2'), ('NewIdea S3', 'NewIdea_S3')]:
        trades = pd.read_csv(os.path.join(FINAL_DIR, sub, 'trades.csv'))
        strategies[sname] = trades

    tp_list = [None, 0.08, 0.10, 0.12, 0.15, 0.18, 0.20]

    print("=" * 90)
    print(f"  止盈优化 - 各策略不同止盈点对比")
    print("=" * 90)

    results = {}
    for sname, trades in strategies.items():
        print(f"\n  {sname}:")
        results[sname] = {}
        for tp in tp_list:
            tp_label = f"TP={tp:.0%}" if tp else "No TP"
            stats, eq = backtest_with_tp(trades, tp, all_dates_set)
            if stats:
                results[sname][tp_label] = (stats, eq)
                print(f"    {tp_label:<12} Total={stats['total']:>9.2%}  Annual={stats['ann']:>9.2%}  Sharpe={stats['sharpe']:>6.2f}  MDD={stats['mdd']:>8.2%}  Final={stats['final_nav']:>12,.0f}")

    best_tp = {}
    for sname in strategies:
        best_sharpe = -999
        best_label = ""
        for tp_label, (stats, _) in results[sname].items():
            if stats['sharpe'] > best_sharpe:
                best_sharpe = stats['sharpe']
                best_label = tp_label
        best_tp[sname] = best_label

    print(f"\n{'='*90}")
    print(f"  最优止盈点")
    print(f"{'='*90}")
    for sname, tp_label in best_tp.items():
        stats = results[sname][tp_label][0]
        print(f"  {sname}: {tp_label} -> Total={stats['total']:.2%}, Sharpe={stats['sharpe']:.2f}")

    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

    fig, axes = plt.subplots(1, 3, figsize=(20, 7))
    colors = {None: '#1f77b4', 0.08: '#ff7f0e', 0.10: '#2ca02c', 0.12: '#d62728',
              0.15: '#9467bd', 0.18: '#8c564b', 0.20: '#e377c2'}

    for idx, (sname, trades) in enumerate(strategies.items()):
        ax = axes[idx]
        for tp in tp_list:
            tp_label = f"TP={tp:.0%}" if tp else "No TP"
            if tp_label in results[sname]:
                stats, eq = results[sname][tp_label]
                eq_df = pd.DataFrame(eq)
                eq_norm = eq_df['nav'] / eq_df['nav'].iloc[0]
                ax.plot(eq_df['date'], eq_norm, label=tp_label, linewidth=1.5, color=colors[tp])
        ax.set_title(sname, fontsize=13, fontweight='bold')
        ax.set_ylabel('NAV (normalized)')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)

    plt.suptitle('Take Profit Optimization - All Strategies', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FINAL_DIR, 'tp_optimization.png'), dpi=150, bbox_inches='tight')
    print(f"\n  Chart saved to tp_optimization.png")

    for sname in strategies:
        best_label = best_tp[sname]
        stats, eq = results[sname][best_label]
        sub = {'doubao_result': 'doubao', 'NewIdea S2': 'NewIdea_S2', 'NewIdea S3': 'NewIdea_S3'}[sname]
        eq_df = pd.DataFrame(eq)
        eq_df.to_csv(os.path.join(FINAL_DIR, sub, f'equity_{best_label.replace("=","_").replace("%","")}.csv'), index=False)

if __name__ == "__main__":
    main()

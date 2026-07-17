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

def backtest_filtered(trades_df, all_dates_set, prob_thresh=0.0, top_n=3, take_profit=None):
    if prob_thresh > 0:
        trades_df = trades_df[trades_df['prob'] >= prob_thresh]

    daily_groups = trades_df.groupby('date_t', sort=True)
    filtered_trades = []
    for date_t, group in daily_groups:
        top = group.nlargest(top_n, 'prob')
        filtered_trades.append(top)
    if not filtered_trades:
        return {}
    trades_df = pd.concat(filtered_trades)

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
    if not equity:
        return {}
    total_ret = capital / initial_cap - 1
    years = len(equity) / 252.0
    ann_ret = (1 + total_ret) ** (1 / years) - 1 if years > 0 else 0
    eq_s = pd.Series([e['nav'] for e in equity])
    df_ret = eq_s.pct_change()
    mdd = ((eq_s - eq_s.cummax()) / eq_s.cummax()).min()
    vol = df_ret.std() * np.sqrt(252)
    sharpe = ann_ret / vol if vol > 0 else 0
    return {'total': total_ret, 'ann': ann_ret, 'sharpe': sharpe, 'mdd': mdd,
            'trades': total_trades, 'cannot_sell': cannot_sell_trades, 'final_nav': capital}

def main():
    all_dates = sorted([f.replace('.parquet', '') for f in os.listdir(PRICE_DIR) if f.endswith('.parquet')])
    all_dates_set = set(int(d) for d in all_dates)

    s2_trades = pd.read_csv(os.path.join(FINAL_DIR, 'NewIdea_S2', 'trades.csv'))

    print("=" * 90)
    print(f"  概率阈值 + 止盈 组合优化 (基于S2全量交易数据)")
    print("=" * 90)

    print(f"\n--- 概率阈值优化 (No TP) ---")
    for thresh in [0.0, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
        stats = backtest_filtered(s2_trades, all_dates_set, prob_thresh=thresh, top_n=3)
        if stats:
            print(f"  prob>={thresh}: Total={stats['total']:>9.2%}  Sharpe={stats['sharpe']:>6.2f}  MDD={stats['mdd']:>8.2%}  Trades={stats['trades']:>5d}")

    print(f"\n--- 概率阈值 + TopN 组合 ---")
    for thresh in [0.0, 0.4, 0.5]:
        for top_n in [1, 2, 3, 5]:
            stats = backtest_filtered(s2_trades, all_dates_set, prob_thresh=thresh, top_n=top_n)
            if stats:
                print(f"  prob>={thresh}, Top{top_n}: Total={stats['total']:>9.2%}  Sharpe={stats['sharpe']:>6.2f}  MDD={stats['mdd']:>8.2%}  Trades={stats['trades']:>5d}")

    print(f"\n--- 最优组合: 概率阈值 + 止盈 ---")
    best_combo = None
    best_sharpe = -999
    for thresh in [0.0, 0.3, 0.4, 0.5]:
        for tp in [None, 0.12, 0.15, 0.18, 0.20]:
            for top_n in [1, 2, 3]:
                stats = backtest_filtered(s2_trades, all_dates_set, prob_thresh=thresh, top_n=top_n, take_profit=tp)
                if stats and stats['trades'] >= 50:
                    tp_label = f"TP={tp:.0%}" if tp else "No TP"
                    if stats['sharpe'] > best_sharpe:
                        best_sharpe = stats['sharpe']
                        best_combo = (thresh, tp, top_n, stats)
    if best_combo:
        thresh, tp, top_n, stats = best_combo
        tp_label = f"TP={tp:.0%}" if tp else "No TP"
        print(f"  最优: prob>={thresh}, Top{top_n}, {tp_label}")
        print(f"  Total={stats['total']:.2%}  Sharpe={stats['sharpe']:.2f}  MDD={stats['mdd']:.2%}  Trades={stats['trades']}")

    print(f"\n--- Top 10 组合 (by Sharpe) ---")
    all_combos = []
    for thresh in [0.0, 0.3, 0.4, 0.5]:
        for tp in [None, 0.12, 0.15, 0.18, 0.20]:
            for top_n in [1, 2, 3]:
                stats = backtest_filtered(s2_trades, all_dates_set, prob_thresh=thresh, top_n=top_n, take_profit=tp)
                if stats and stats['trades'] >= 50:
                    tp_label = f"TP={tp:.0%}" if tp else "No TP"
                    all_combos.append((thresh, tp, top_n, tp_label, stats))
    all_combos.sort(key=lambda x: x[4]['sharpe'], reverse=True)
    for thresh, tp, top_n, tp_label, stats in all_combos[:10]:
        print(f"  prob>={thresh}, Top{top_n}, {tp_label:<8} Total={stats['total']:>9.2%}  Sharpe={stats['sharpe']:>6.2f}  MDD={stats['mdd']:>8.2%}  Trades={stats['trades']:>5d}")

if __name__ == "__main__":
    main()

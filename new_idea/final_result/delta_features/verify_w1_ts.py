"""
Verify the W1_TS>0 scheme - check if Sharpe=67 is real or an artifact.
"""
import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = r'D:\iquant_data\data_v2'
PRICE_DIR = os.path.join(DATA_DIR, 'data_day1')

def is_gem_or_star(ts_code):
    return ('300' in ts_code) or ('301' in ts_code) or ('688' in ts_code) or ('689' in ts_code)

def int_to_date(date_int):
    s = str(int(date_int))
    return datetime(int(s[:4]), int(s[4:6]), int(s[6:8]))

def get_next_trading_day(date_int, all_dates_set):
    current_dt = int_to_date(date_int)
    for i in range(1, 10):
        next_dt = current_dt + timedelta(days=i)
        next_int = int(next_dt.strftime('%Y%m%d'))
        if next_int in all_dates_set:
            return next_int
    return None

all_dates = sorted([f.replace('.parquet', '') for f in os.listdir(PRICE_DIR) if f.endswith('.parquet')])
all_dates_set = set(int(d) for d in all_dates)

trades_ts = pd.read_csv(os.path.join(THIS_DIR, 'trades_with_ts_features.csv'))

wc = {'w_ret1d': -0.5, 'w_dwr': 3.0, 'w_dcc': -1.5, 'w_ret5d': -0.2, 'w_ma5': -0.3}
trades_ts['ts_score_v2'] = (
    trades_ts['ret_1d'].abs() * wc['w_ret1d'] +
    trades_ts['delta_winner_rate'] * wc['w_dwr'] +
    trades_ts['delta_chip_conc'].abs() * wc['w_dcc'] +
    trades_ts['ret_5d'].abs() * wc['w_ret5d'] +
    trades_ts['ma5_dist'].abs() * wc['w_ma5']
)

mask = trades_ts['ts_score_v2'] > 0
t = trades_ts[mask]
print(f"W1_TS>0: {len(t)} trades", flush=True)

print(f"\nts_score_v2 stats:", flush=True)
print(t['ts_score_v2'].describe(), flush=True)

print(f"\ndelta_winner_rate stats:", flush=True)
print(t['delta_winner_rate'].describe(), flush=True)

print(f"\nret_1d stats:", flush=True)
print(t['ret_1d'].describe(), flush=True)

print(f"\nDate distribution:", flush=True)
print(f"  Unique dates: {t['date_t2'].nunique()}", flush=True)
print(f"  Trades per date: mean={t.groupby('date_t2').size().mean():.1f}, max={t.groupby('date_t2').size().max()}", flush=True)

initial_cap = 100000.0
capital = initial_cap
equity = []
daily_returns = []

for date_t2, group in t.groupby('date_t2', sort=True):
    alloc = capital / len(group)
    day_pnl = 0.0
    for _, trade in group.iterrows():
        ts_code = trade['ts_code']
        buy_price = trade['buy_price']
        sell_close = trade['sell_close']
        sell_high = trade['sell_high']
        sell_pre_close = trade['sell_pre_close']

        limit_down_pct = 0.8 if is_gem_or_star(ts_code) else 0.9
        limit_down_price = round(sell_pre_close * limit_down_pct, 2)
        is_cannot_sell = (sell_high == limit_down_price)

        if is_cannot_sell:
            date_t3 = get_next_trading_day(date_t2, all_dates_set)
            if date_t3:
                p_t3 = os.path.join(PRICE_DIR, f"{date_t3}.parquet")
                if os.path.exists(p_t3):
                    df_t3 = pd.read_parquet(p_t3, columns=['ts_code', 'open'])
                    t3_row = df_t3[df_t3['ts_code'] == ts_code]
                    sell_price = t3_row.iloc[0]['open'] if not t3_row.empty else sell_close
                else:
                    sell_price = sell_close
            else:
                sell_price = sell_close
        else:
            sell_price = sell_close

        ret = (sell_price / buy_price) - 1 - 0.0015
        day_pnl += alloc * ret

    prev_capital = capital
    capital += day_pnl
    daily_ret = (capital / prev_capital) - 1
    daily_returns.append(daily_ret)
    equity.append({'date': int_to_date(date_t2), 'nav': capital, 'daily_ret': daily_ret})

eq_df = pd.DataFrame(equity)
total_ret = capital / initial_cap - 1
years = len(eq_df) / 252.0
ann_ret = (1 + total_ret) ** (1 / years) - 1
daily_rets = eq_df['daily_ret']

print(f"\n=== Detailed Backtest Results ===", flush=True)
print(f"Total Return: {total_ret:.2%}", flush=True)
print(f"Annual Return: {ann_ret:.2%}", flush=True)
print(f"Years: {years:.2f}", flush=True)
print(f"Trading days: {len(eq_df)}", flush=True)

vol = daily_rets.std() * np.sqrt(252)
sharpe = ann_ret / vol if vol > 0 else 0
mdd = ((eq_df['nav'] - eq_df['nav'].cummax()) / eq_df['nav'].cummax()).min()
calmar = ann_ret / abs(mdd) if mdd != 0 else 0

print(f"Daily Vol: {daily_rets.std():.6f}", flush=True)
print(f"Annual Vol: {vol:.4f}", flush=True)
print(f"Sharpe: {sharpe:.2f}", flush=True)
print(f"MDD: {mdd:.2%}", flush=True)
print(f"Calmar: {calmar:.2f}", flush=True)
print(f"Win Rate: {(daily_rets > 0).mean():.2%}", flush=True)

print(f"\nDaily return distribution:", flush=True)
print(f"  Mean: {daily_rets.mean():.4f}", flush=True)
print(f"  Std: {daily_rets.std():.4f}", flush=True)
print(f"  Min: {daily_rets.min():.4f}", flush=True)
print(f"  Max: {daily_rets.max():.4f}", flush=True)
print(f"  Skew: {daily_rets.skew():.4f}", flush=True)
print(f"  Kurtosis: {daily_rets.kurtosis():.4f}", flush=True)

print(f"\nNAV by year:", flush=True)
eq_df['year'] = eq_df['date'].dt.year
for year, yg in eq_df.groupby('year'):
    start_nav = yg['nav'].iloc[0]
    end_nav = yg['nav'].iloc[-1]
    yr_ret = end_nav / start_nav - 1
    print(f"  {year}: NAV {start_nav:.0f} -> {end_nav:.0f}, Return={yr_ret:.2%}", flush=True)

print(f"\nFirst 10 trades:", flush=True)
for _, tr in t.head(10).iterrows():
    ret = (tr['sell_close'] / tr['buy_price']) - 1 - 0.0015
    print(f"  {tr['date_t']} {tr['ts_code']} buy={tr['buy_price']:.2f} sell={tr['sell_close']:.2f} ret={ret:.2%} prob={tr['prob']:.4f} ts_score_v2={tr['ts_score_v2']:.2f}", flush=True)

print(f"\nLast 10 trades:", flush=True)
for _, tr in t.tail(10).iterrows():
    ret = (tr['sell_close'] / tr['buy_price']) - 1 - 0.0015
    print(f"  {tr['date_t']} {tr['ts_code']} buy={tr['buy_price']:.2f} sell={tr['sell_close']:.2f} ret={ret:.2%} prob={tr['prob']:.4f} ts_score_v2={tr['ts_score_v2']:.2f}", flush=True)

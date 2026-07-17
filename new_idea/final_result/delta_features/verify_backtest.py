import pandas as pd, numpy as np, os
from datetime import datetime, timedelta

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

trades_df = pd.read_csv(r'c:\Users\liuqi\quant_system_v2\new_idea\final_result\doubao\doubao_trades.csv')
print(f'Trades: {len(trades_df)}', flush=True)
print(f'date_t2 dtype: {trades_df["date_t2"].dtype}', flush=True)
print(f'Sample trades:', flush=True)
print(trades_df[['ts_code', 'buy_price', 'sell_close', 'sell_high', 'sell_pre_close']].head(5), flush=True)

# Check a few trades manually
for _, t in trades_df.head(5).iterrows():
    ret = (t['sell_close'] / t['buy_price']) - 1 - 0.0015
    print(f"  {t['ts_code']}: buy={t['buy_price']:.2f}, sell={t['sell_close']:.2f}, ret={ret:.4f}", flush=True)

# Run doubao's exact backtest
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
        else:
            sell_price = sell_close

        ret = (sell_price / buy_price) - 1 - 0.0015
        day_pnl += alloc * ret

    capital += day_pnl
    equity.append({'date': int_to_date(date_t2), 'nav': capital})

eq_df = pd.DataFrame(equity)
total_ret = capital / initial_cap - 1
years = len(eq_df) / 252.0
ann_ret = (1 + total_ret) ** (1 / years) - 1 if years > 0 else 0
df_ret = eq_df['nav'].pct_change()
mdd = ((eq_df['nav'] - eq_df['nav'].cummax()) / eq_df['nav'].cummax()).min()
vol = df_ret.std() * np.sqrt(252)
sharpe = ann_ret / vol if vol > 0 else 0

print(f'\nResults:', flush=True)
print(f'  Total: {total_ret:.2%}, Annual: {ann_ret:.2%}, Sharpe: {sharpe:.2f}, MDD: {mdd:.2%}', flush=True)
print(f'  Trades: {total_trades}, Cannot sell: {cannot_sell_trades}', flush=True)
print(f'  Final NAV: {capital:.2f}', flush=True)
print(f'  First 5 equity:', flush=True)
print(eq_df.head(5), flush=True)
print(f'  Last 5 equity:', flush=True)
print(eq_df.tail(5), flush=True)

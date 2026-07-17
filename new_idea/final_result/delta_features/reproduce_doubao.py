"""
Step 1: Reproduce doubao's exact backtest to verify correctness.
Step 2: Add TS re-ranking on top.
Memory optimization: specify columns when reading parquet, delete intermediates immediately.
"""
import os, sys, gc, traceback, time
import pandas as pd
import numpy as np
import joblib
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = r'D:\iquant_data\data_v2'
PRICE_DIR = os.path.join(DATA_DIR, 'data_day1')
RANK_DIR  = os.path.join(DATA_DIR, 'ths_rank1')
CHIP_DIR  = os.path.join(DATA_DIR, 'cyq1')
OTHER_DIR = os.path.join(DATA_DIR, 'other_day1')
NEWS_MAJOR_DIR = r"C:\Users\liuqi\clowspace\toolstock\news_major1"

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
FINAL_DIR = os.path.dirname(THIS_DIR)

BASE_FEATS = ['hot_rank_pct', 'chip_concentration', 'winner_rate',
              'news_market_impact', 'news_stock_impact']

CIRC_MV_LIMIT = 1000000

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

def process_news(news_dir):
    market_records, stock_records = [], []
    if not os.path.exists(news_dir):
        return pd.DataFrame(market_records), pd.DataFrame(stock_records)
    for filename in os.listdir(news_dir):
        if not filename.endswith('.json'):
            continue
        try:
            with open(os.path.join(news_dir, filename), 'r', encoding='utf-8') as f:
                data = json.load(f)
        except:
            continue
        date_str = data.get("article_date", "")
        if not date_str:
            continue
        trade_date = pd.to_datetime(date_str)
        market_records.append({'trade_date': trade_date, 'news_market_impact': float(data.get("market_impact", 0))})
        for s in data.get("stocks", []):
            code = s.get("stock_code")
            if not code:
                continue
            ts_code = f"{code}.SH" if code.startswith('6') else f"{code}.SZ" if (code.startswith('0') or code.startswith('3')) else code
            stock_records.append({'trade_date': trade_date, 'ts_code': ts_code, 'news_stock_impact': float(s.get("impact", 0))})
    return pd.DataFrame(market_records), pd.DataFrame(stock_records)

def add_news_features(df, d_curr, news_mkt, news_stk):
    if not news_mkt.empty:
        nm = news_mkt.copy()
        if pd.api.types.is_datetime64_any_dtype(nm['trade_date']):
            nm['trade_date'] = nm['trade_date'].dt.strftime('%Y%m%d')
        same_date = nm[nm['trade_date'] == d_curr]
        if not same_date.empty:
            df['news_market_impact'] = same_date['news_market_impact'].mean()
        else:
            df['news_market_impact'] = 0.0
    else:
        df['news_market_impact'] = 0.0
    if not news_stk.empty:
        ns = news_stk.copy()
        if pd.api.types.is_datetime64_any_dtype(ns['trade_date']):
            ns['trade_date'] = ns['trade_date'].dt.strftime('%Y%m%d')
        same_date = ns[ns['trade_date'] == d_curr]
        if not same_date.empty:
            df = pd.merge(df, same_date[['ts_code', 'news_stock_impact']], on='ts_code', how='left')
            df['news_stock_impact'] = df['news_stock_impact'].fillna(0.0)
        else:
            df['news_stock_impact'] = 0.0
    else:
        df['news_stock_impact'] = 0.0
    return df

def load_features(d_curr, news_mkt, news_stk):
    """Exact same as doubao's load_features"""
    p_rank = os.path.join(RANK_DIR, f"{d_curr}.parquet")
    p_chip = os.path.join(CHIP_DIR, f"{d_curr}.parquet")
    p_price = os.path.join(PRICE_DIR, f"{d_curr}.parquet")
    p_other = os.path.join(OTHER_DIR, f"{d_curr}.parquet")
    if not all(os.path.exists(p) for p in [p_rank, p_chip, p_price, p_other]):
        return None

    rank_df = pd.read_parquet(p_rank)
    rank_df['hot_rank_pct'] = rank_df['hot'].rank(pct=True)
    chip_df = pd.read_parquet(p_chip)
    chip_df['chip_concentration'] = (chip_df['cost_85pct'] - chip_df['cost_15pct']) / (chip_df['cost_50pct'] + 1e-8)
    price_df = pd.read_parquet(p_price, columns=['ts_code', 'close', 'pct_chg', 'amount', 'vol', 'pre_close', 'open', 'high', 'low'])
    other_df = pd.read_parquet(p_other, columns=['ts_code', 'circ_mv'])

    df = pd.merge(rank_df[['ts_code', 'hot_rank_pct']], price_df, on='ts_code')
    df = pd.merge(df, chip_df[['ts_code', 'chip_concentration', 'winner_rate']], on='ts_code')
    df = pd.merge(df, other_df, on='ts_code', how='left')
    df = df[~df['ts_code'].str.startswith('688')]
    df = df[df['circ_mv'] <= CIRC_MV_LIMIT]
    df = add_news_features(df, d_curr, news_mkt, news_stk)
    for f in BASE_FEATS:
        if f not in df.columns:
            df[f] = 0.0
    return df

def select(df, model, feats):
    """Exact same as doubao's select"""
    X = df[feats].fillna(0)
    if len(X) == 0:
        return pd.DataFrame()
    df = df.copy()
    df['prob'] = model.predict_proba(X)[:, 1]
    picks = df[df['prob'] > 0.8].sort_values('prob', ascending=False).head(3)
    if picks.empty:
        picks = df.sort_values('prob', ascending=False).head(1)
    return picks

def backtest(trades_df, all_dates_set, take_profit=None):
    if trades_df.empty:
        return pd.DataFrame(), {}
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
                        sell_price = t3_row.iloc[0]['open'] if not t3_row.empty else sell_close
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
        return eq_df, {}
    total_ret = capital / initial_cap - 1
    years = len(eq_df) / 252.0
    ann_ret = (1 + total_ret) ** (1 / years) - 1 if years > 0 else 0
    df_ret = eq_df['nav'].pct_change()
    mdd = ((eq_df['nav'] - eq_df['nav'].cummax()) / eq_df['nav'].cummax()).min()
    vol = df_ret.std() * np.sqrt(252)
    sharpe = ann_ret / vol if vol > 0 else 0
    calmar = ann_ret / abs(mdd) if mdd != 0 else 0
    win_rate = (df_ret > 0).mean()
    return eq_df, {'total': total_ret, 'ann': ann_ret, 'sharpe': sharpe, 'mdd': mdd,
                   'calmar': calmar, 'win_rate': win_rate, 'trades': total_trades,
                   'cannot_sell': cannot_sell_trades, 'final_nav': capital}

def main():
    print("=" * 90, flush=True)
    print("  Step 1: Reproduce doubao's exact backtest", flush=True)
    print("=" * 90, flush=True)

    all_dates = sorted([f.replace('.parquet', '') for f in os.listdir(PRICE_DIR) if f.endswith('.parquet')])
    all_dates_set = set(int(d) for d in all_dates)
    news_mkt, news_stk = process_news(NEWS_MAJOR_DIR)

    model, feats = joblib.load(os.path.join(FINAL_DIR, 'doubao', 'models', 'doubao_t1t2_model.joblib'))
    print("Model loaded", flush=True)

    trades = []
    for idx in range(1, len(all_dates) - 2):
        d_t = all_dates[idx]
        d_t1 = all_dates[idx + 1]
        d_t2 = all_dates[idx + 2]
        if d_t < '20230101' or d_t > '20260324':
            continue
        p_t1 = os.path.join(PRICE_DIR, f"{d_t1}.parquet")
        p_t2 = os.path.join(PRICE_DIR, f"{d_t2}.parquet")
        if not os.path.exists(p_t1) or not os.path.exists(p_t2):
            continue
        df = load_features(d_t, news_mkt, news_stk)
        if df is None:
            continue
        picks = select(df, model, feats)
        if picks.empty:
            continue
        df_t1 = pd.read_parquet(p_t1, columns=['ts_code', 'open', 'pre_close'])
        df_t2 = pd.read_parquet(p_t2, columns=['ts_code', 'open', 'high', 'low', 'close', 'pre_close'])
        for _, pick in picks.iterrows():
            ts_code = pick['ts_code']
            t1r = df_t1[df_t1['ts_code'] == ts_code]
            if t1r.empty:
                continue
            t1 = t1r.iloc[0]
            pre_close_t1 = t1['pre_close']
            up_limit = round(pre_close_t1 * 1.2, 2) if is_gem_or_star(ts_code) else round(pre_close_t1 * 1.1, 2)
            if pd.isna(t1['open']) or t1['open'] >= up_limit:
                continue
            t2r = df_t2[df_t2['ts_code'] == ts_code]
            if t2r.empty:
                continue
            t2 = t2r.iloc[0]
            trades.append({
                'date_t': d_t, 'date_t1': d_t1, 'date_t2': d_t2,
                'ts_code': ts_code, 'buy_price': t1['open'],
                'sell_open': t2['open'], 'sell_high': t2['high'],
                'sell_close': t2['close'], 'sell_pre_close': t2['pre_close'],
                'prob': pick['prob'],
            })
        del df, picks
        gc.collect()

        if len(trades) % 50 == 0 and len(trades) > 0:
            print(f"  {len(trades)} trades so far (date: {d_t})", flush=True)

    trades_df = pd.DataFrame(trades)
    print(f"\nTotal trades: {len(trades_df)}", flush=True)
    trades_df.to_csv(os.path.join(THIS_DIR, 'doubao_reproduced_trades.csv'), index=False)

    eq, stats = backtest(trades_df, all_dates_set)
    print(f"\nReproduced doubao results:", flush=True)
    print(f"  Total: {stats['total']:.2%}, Annual: {stats['ann']:.2%}, Sharpe: {stats['sharpe']:.2f}, MDD: {stats['mdd']:.2%}", flush=True)
    print(f"  Trades: {stats['trades']}, Cannot sell: {stats['cannot_sell']}", flush=True)

    eq.to_csv(os.path.join(THIS_DIR, 'doubao_reproduced_equity.csv'), index=False)
    print(f"Saved to {THIS_DIR}", flush=True)

if __name__ == "__main__":
    main()

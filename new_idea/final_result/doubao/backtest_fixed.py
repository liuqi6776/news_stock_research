"""
doubao_result base method 修正版回测
严格按照A股交易规则：
1. 涨停开盘不能买入
2. 跌停日按跌停价卖出
3. 扣除交易费用0.3%
4. 添加滑点0.2%
"""
import os
import sys
import pandas as pd
import numpy as np
import joblib
from tqdm import tqdm
from datetime import datetime, timedelta
import json

DATA_DIR = r'D:\iquant_data\data_v2'
PRICE_DIR = os.path.join(DATA_DIR, 'data_day1')
RANK_DIR = os.path.join(DATA_DIR, 'ths_rank1')
CHIP_DIR = os.path.join(DATA_DIR, 'cyq1')
OTHER_DIR = os.path.join(DATA_DIR, 'other_day1')
NEWS_MAJOR_DIR = r"C:\Users\liuqi\clowspace\toolstock\news_major1"

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(THIS_DIR, 'models', 'doubao_t1t2_model.joblib')

FEATS = ['hot_rank_pct', 'chip_concentration', 'winner_rate',
         'news_market_impact', 'news_stock_impact']

CIRC_MV_LIMIT = 1000000

# ============ A股交易规则配置 ============
COST_RATE = 0.003       # 交易费用 0.3% 双边
SLIPPAGE = 0.002        # 滑点 0.2%
LIMIT_THRESHOLD = 0.5   # 涨跌停阈值偏移 (10% -> 9.5%)


def is_gem_or_star(ts_code):
    return ('300' in ts_code) or ('301' in ts_code) or ('688' in ts_code) or ('689' in ts_code)


def get_limit_pct(ts_code):
    """获取股票的涨跌停幅度"""
    if ts_code.startswith('300') or ts_code.startswith('301'):
        return 20.0
    elif ts_code.startswith('688') or ts_code.startswith('689'):
        return 20.0
    elif ts_code.startswith('8') or ts_code.startswith('43'):
        return 30.0
    else:
        return 10.0


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


def process_news(news_dir):
    market_records = []
    stock_records = []
    if not os.path.exists(news_dir):
        return pd.DataFrame(market_records), pd.DataFrame(stock_records)
    for filename in os.listdir(news_dir):
        if not filename.endswith('.json'):
            continue
        filepath = os.path.join(news_dir, filename)
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except:
            continue
        date_str = data.get("article_date", "")
        if not date_str:
            continue
        trade_date = pd.to_datetime(date_str)
        market_impact = data.get("market_impact", 0)
        market_records.append({'trade_date': trade_date, 'news_market_impact': float(market_impact)})
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
    price_df = pd.read_parquet(p_price, columns=['ts_code', 'close', 'pre_close'])
    other_df = pd.read_parquet(p_other, columns=['ts_code', 'circ_mv'])
    df = pd.merge(rank_df[['ts_code', 'hot_rank_pct']], price_df, on='ts_code')
    df = pd.merge(df, chip_df[['ts_code', 'chip_concentration', 'winner_rate']], on='ts_code')
    df = pd.merge(df, other_df, on='ts_code', how='left')
    df = df[~df['ts_code'].str.startswith('688')]
    df = df[df['circ_mv'] <= CIRC_MV_LIMIT]
    df = add_news_features(df, d_curr, news_mkt, news_stk)
    for f in FEATS:
        if f not in df.columns:
            df[f] = 0.0
    return df


def select(df, model, feats):
    X = df[feats].fillna(0)
    if len(X) == 0:
        return pd.DataFrame()
    df['prob'] = model.predict_proba(X)[:, 1]
    picks = df[df['prob'] > 0.8].sort_values('prob', ascending=False).head(3)
    if picks.empty:
        picks = df.sort_values('prob', ascending=False).head(1)
    return picks


def backtest_fixed():
    print("=" * 60)
    print("doubao_result base method 修正版回测")
    print("=" * 60)
    print(f"交易费用: {COST_RATE*100:.2f}%")
    print(f"滑点: {SLIPPAGE*100:.2f}%")
    print(f"涨跌停阈值: {LIMIT_THRESHOLD:.1f}%")
    print("=" * 60)

    model, feats = joblib.load(MODEL_PATH)
    all_dates = sorted([f.replace('.parquet', '') for f in os.listdir(PRICE_DIR) if f.endswith('.parquet')])
    all_dates_set = set(int(d) for d in all_dates)
    news_mkt, news_stk = process_news(NEWS_MAJOR_DIR)

    trades = []
    skipped_limit_up = 0
    skipped_limit_down = 0

    for idx in tqdm(range(1, len(all_dates) - 2), desc="回测中"):
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

        df_t1 = pd.read_parquet(p_t1, columns=['ts_code', 'open', 'low', 'pre_close'])
        df_t2 = pd.read_parquet(p_t2, columns=['ts_code', 'open', 'high', 'low', 'close', 'pre_close'])

        for _, pick in picks.iterrows():
            ts_code = pick['ts_code']

            # 板块过滤（创业板/科创板）
            if is_gem_or_star(ts_code):
                continue

            t1r = df_t1[df_t1['ts_code'] == ts_code]
            if t1r.empty:
                continue
            t1 = t1r.iloc[0]
            t1_open = float(t1['open'])
            t1_pre = float(t1['pre_close'])

            # 获取涨跌停幅度
            limit_pct = get_limit_pct(ts_code)

            # ====== A股交易规则检查 ======

            # 1. 涨停检查 (不能买入)
            t1_open_chg = (t1_open - t1_pre) / t1_pre * 100
            if t1_open_chg >= (limit_pct - LIMIT_THRESHOLD):
                skipped_limit_up += 1
                continue

            t2r = df_t2[df_t2['ts_code'] == ts_code]
            if t2r.empty:
                continue
            t2 = t2r.iloc[0]
            t2_close = float(t2['close'])
            t2_low = float(t2['low'])
            t2_open = float(t2['open'])

            # 2. 跌停检查 (按跌停价卖出)
            t2_low_chg = (t2_low - t1_open) / t1_open * 100
            if t2_low_chg <= -(limit_pct - LIMIT_THRESHOLD):
                sell_price = min(t2_open, t1_open * (1 - limit_pct/100))
                skipped_limit_down += 1
            else:
                sell_price = t2_close

            # 3. 应用滑点
            buy_price = t1_open * (1 + SLIPPAGE)
            sell_price = sell_price * (1 - SLIPPAGE)

            # 4. 计算收益 (扣除费用)
            ret = sell_price / buy_price - 1 - COST_RATE

            trades.append({
                'date_t': d_t, 'date_t1': d_t1, 'date_t2': d_t2,
                'ts_code': ts_code, 'buy_price': buy_price, 'sell_price': sell_price,
                'ret': ret, 'prob': pick['prob'],
                't1_open_chg': t1_open_chg, 't2_low_chg': t2_low_chg,
            })

    if not trades:
        print("无交易!")
        return None, None, None

    trades_df = pd.DataFrame(trades)

    # 计算权益曲线
    initial_cap = 100000.0
    capital = initial_cap
    equity = []

    for date_t2, group in trades_df.groupby('date_t2', sort=True):
        alloc = capital / len(group)
        day_pnl = 0.0
        for _, trade in group.iterrows():
            day_pnl += alloc * trade['ret']
        capital += day_pnl
        equity.append({'date': int_to_date(date_t2), 'nav': capital})

    eq_df = pd.DataFrame(equity)

    print(f"\n{'='*60}")
    print("回测统计")
    print(f"{'='*60}")
    print(f"交易次数: {len(trades_df)}")
    print(f"跳过涨停买入: {skipped_limit_up}")
    print(f"跌停日卖出: {skipped_limit_down}")

    total_ret = capital / initial_cap - 1
    years = len(eq_df) / 252.0
    ann_ret = (1 + total_ret) ** (1 / years) - 1 if years > 0 else 0
    df_ret = eq_df['nav'].pct_change()
    mdd = ((eq_df['nav'] - eq_df['nav'].cummax()) / eq_df['nav'].cummax()).min()
    vol = df_ret.std() * np.sqrt(252)
    sharpe = ann_ret / vol if vol > 0 else 0
    win_rate = (trades_df['ret'] > 0).mean()
    avg_ret = trades_df['ret'].mean()

    print(f"\n总收益: {total_ret*100:.2f}%")
    print(f"年化收益: {ann_ret*100:.2f}%")
    print(f"夏普比率: {sharpe:.2f}")
    print(f"最大回撤: {mdd*100:.2f}%")
    print(f"胜率: {win_rate*100:.2f}%")
    print(f"平均收益: {avg_ret*100:.2f}%")
    print(f"最高单笔: {trades_df['ret'].max()*100:.2f}%")
    print(f"最低单笔: {trades_df['ret'].min()*100:.2f}%")

    # 保存结果
    trades_df.to_csv(os.path.join(THIS_DIR, 'trades_doubao_fixed.csv'), index=False)
    eq_df.to_csv(os.path.join(THIS_DIR, 'equity_doubao_fixed.csv'), index=False)
    print(f"\n结果已保存")

    return trades_df, eq_df, {
        'total_ret': total_ret,
        'ann_ret': ann_ret,
        'sharpe': sharpe,
        'mdd': mdd,
        'win_rate': win_rate,
        'avg_ret': avg_ret,
        'n_trades': len(trades_df)
    }


if __name__ == "__main__":
    backtest_fixed()

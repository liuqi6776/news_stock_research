"""
final_method 修正版回测
严格按照A股交易规则：
1. 涨停开盘不能买入
2. 跌停日按跌停价卖出
3. 扣除交易费用0.3%
4. 添加滑点0.2%
"""
import os
import sys
import json
import pandas as pd
import numpy as np
import joblib
from tqdm import tqdm

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = r'D:\iquant_data\data_v2'
PRICE_DIR = os.path.join(DATA_DIR, 'data_day1')
RANK_DIR = os.path.join(DATA_DIR, 'ths_rank1')
CHIP_DIR = os.path.join(DATA_DIR, 'cyq1')
OTHER_DIR = os.path.join(DATA_DIR, 'other_day1')
NEWS_MAJOR_DIR = os.path.join(THIS_DIR, "news_major1")
MODEL_PATH = os.path.join(THIS_DIR, 'models', 'doubao_t1t2_model.joblib')

FEATS = ['hot_rank_pct', 'chip_concentration', 'winner_rate',
         'news_market_impact', 'news_stock_impact']
CIRC_MV_LIMIT = 3000000
MIN_PROB_THRESHOLD = 0.6

# ============ A股交易规则配置 ============
COST_RATE = 0.003       # 交易费用 0.3% 双边
SLIPPAGE = 0.002        # 滑点 0.2%
LIMIT_THRESHOLD = 0.5   # 涨跌停阈值偏移 (10% -> 9.5%)


def is_main_board(ts_code):
    return ts_code.startswith('60') or ts_code.startswith('00')


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


def process_news(news_dir, target_date=None):
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
        date_formatted = trade_date.strftime('%Y%m%d')
        if target_date and date_formatted > target_date:
            continue
        market_impact = data.get("market_impact", 0)
        market_records.append({'trade_date': trade_date, 'news_market_impact': float(market_impact)})
        for s in data.get("stocks", []):
            code = s.get("stock_code", "")
            impact = float(s.get("impact", 0))
            ts_code = f"{code}.SH" if code.startswith('6') else f"{code}.SZ" if (
                    code.startswith('0') or code.startswith('3')) else code
            if ts_code:
                stock_records.append(
                    {'trade_date': trade_date, 'ts_code': ts_code, 'news_stock_impact': impact})
    return pd.DataFrame(market_records), pd.DataFrame(stock_records)


def add_news_features(df, news_mkt, news_stk):
    if not news_mkt.empty:
        nm = news_mkt.copy()
        df['news_market_impact'] = nm['news_market_impact'].max() if not nm.empty else 0.0
    else:
        df['news_market_impact'] = 0.0
    if not news_stk.empty and 'ts_code' in news_stk.columns:
        ns = news_stk.copy()
        ns_agg = ns.groupby('ts_code')['news_stock_impact'].max().reset_index()
        if not ns_agg.empty and 'ts_code' in ns_agg.columns:
            df = pd.merge(df, ns_agg[['ts_code', 'news_stock_impact']], on='ts_code', how='left')
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
    if not all(os.path.exists(p) for p in [p_rank, p_price, p_other]):
        return None
    if not os.path.exists(p_chip):
        all_chip_dates = sorted([f.replace('.parquet', '') for f in os.listdir(CHIP_DIR) if f.endswith('.parquet')],
                                reverse=True)
        for cd in all_chip_dates:
            if cd <= d_curr:
                p_chip = os.path.join(CHIP_DIR, f"{cd}.parquet")
                break
        else:
            return None
    rank_df = pd.read_parquet(p_rank)
    rank_df = rank_df.sort_values('hot', ascending=False).drop_duplicates(subset='ts_code', keep='first')
    rank_df['hot_rank_pct'] = rank_df['hot'].rank(pct=True)
    chip_df = pd.read_parquet(p_chip)
    chip_df['chip_concentration'] = (chip_df['cost_85pct'] - chip_df['cost_15pct']) / (
            chip_df['cost_50pct'] + 1e-8)
    price_df = pd.read_parquet(p_price, columns=['ts_code', 'close', 'pre_close'])
    other_df = pd.read_parquet(p_other, columns=['ts_code', 'circ_mv'])
    df = pd.merge(rank_df[['ts_code', 'hot_rank_pct']], price_df, on='ts_code')
    df = pd.merge(df, chip_df[['ts_code', 'chip_concentration', 'winner_rate']], on='ts_code', how='left')
    df['chip_concentration'] = df['chip_concentration'].fillna(
        df['chip_concentration'].median() if df['chip_concentration'].notna().any() else 0.1)
    df['winner_rate'] = df['winner_rate'].fillna(
        df['winner_rate'].median() if df['winner_rate'].notna().any() else 50.0)
    df = pd.merge(df, other_df, on='ts_code', how='left')
    df = df[df['ts_code'].apply(is_main_board)]
    if 'circ_mv' not in df.columns:
        df['circ_mv'] = 0
    df['circ_mv'] = df['circ_mv'].fillna(0)
    df = df[df['circ_mv'] <= CIRC_MV_LIMIT]
    if not df.empty and 'ts_code' in df.columns:
        df = add_news_features(df, news_mkt, news_stk)
    for f in FEATS:
        if f not in df.columns:
            df[f] = 0.0
    return df


def backtest_fixed(start_date='20230801', end_date='20260331'):
    print("=" * 60)
    print("final_method 修正版回测")
    print("=" * 60)
    print(f"交易费用: {COST_RATE*100:.2f}%")
    print(f"滑点: {SLIPPAGE*100:.2f}%")
    print(f"涨跌停阈值: {LIMIT_THRESHOLD:.1f}%")
    print("=" * 60)

    if not os.path.exists(MODEL_PATH):
        print(f"错误: 模型文件不存在: {MODEL_PATH}")
        return None, None, None

    loaded = joblib.load(MODEL_PATH)
    model = loaded[0] if isinstance(loaded, tuple) else loaded
    feats = loaded[1] if isinstance(loaded, tuple) else FEATS

    print(f"加载模型成功，特征: {feats}")

    all_dates = sorted([f.replace('.parquet', '') for f in os.listdir(PRICE_DIR) if f.endswith('.parquet')])
    date_idx = {d: i for i, d in enumerate(all_dates)}

    start_idx = date_idx.get(start_date, 0)
    end_idx = date_idx.get(end_date, len(all_dates) - 1)
    test_dates = all_dates[start_idx:end_idx - 2]

    print(f"回测期间: {test_dates[0]} ~ {test_dates[-1]}, 共{len(test_dates)}个交易日")

    # 处理新闻数据
    news_mkt, news_stk = process_news(NEWS_MAJOR_DIR, None)
    print(f"新闻数据: market={len(news_mkt)}条, stock={len(news_stk)}条")

    trades = []
    equity = [100000.0]
    skipped_limit_up = 0
    skipped_limit_down = 0
    no_signal_days = 0

    for d in tqdm(test_dates, desc="回测中"):
        curr_idx = date_idx[d]
        if curr_idx + 2 >= len(all_dates):
            break

        d_t1 = all_dates[curr_idx + 1]
        d_t2 = all_dates[curr_idx + 2]

        # 加载特征
        df = load_features(d, news_mkt, news_stk)
        if df is None or len(df) == 0:
            equity.append(equity[-1])
            continue

        # 预测
        X = df[feats].fillna(0)
        df['prob'] = model.predict_proba(X)[:, 1]

        # 选择最高概率股票
        best_prob = df['prob'].max()
        if best_prob < MIN_PROB_THRESHOLD:
            no_signal_days += 1
            equity.append(equity[-1])
            continue

        pick = df.loc[df['prob'].idxmax()]
        ts_code = pick['ts_code']

        # 加载价格数据
        p_t0 = os.path.join(PRICE_DIR, f"{d}.parquet")
        p_t1 = os.path.join(PRICE_DIR, f"{d_t1}.parquet")
        p_t2 = os.path.join(PRICE_DIR, f"{d_t2}.parquet")
        if not all(os.path.exists(p) for p in [p_t0, p_t1, p_t2]):
            equity.append(equity[-1])
            continue

        try:
            price_t0 = pd.read_parquet(p_t0, columns=['ts_code', 'close', 'pre_close'])
            price_t1 = pd.read_parquet(p_t1, columns=['ts_code', 'open', 'low', 'pre_close'])
            price_t2 = pd.read_parquet(p_t2, columns=['ts_code', 'close', 'low', 'open', 'pre_close'])
        except:
            equity.append(equity[-1])
            continue

        t0_row = price_t0[price_t0['ts_code'] == ts_code]
        t1_row = price_t1[price_t1['ts_code'] == ts_code]
        t2_row = price_t2[price_t2['ts_code'] == ts_code]

        if t0_row.empty or t1_row.empty or t2_row.empty:
            equity.append(equity[-1])
            continue

        t0_close = float(t0_row['close'].values[0])
        t1_open = float(t1_row['open'].values[0])
        t1_pre = float(t1_row['pre_close'].values[0]) if 'pre_close' in t1_row.columns else t0_close
        t2_close = float(t2_row['close'].values[0])
        t2_low = float(t2_row['low'].values[0])
        t2_open = float(t2_row['open'].values[0]) if 'open' in t2_row.columns else t2_close

        # 获取涨跌停幅度
        limit_pct = get_limit_pct(ts_code)

        # ====== A股交易规则检查 ======

        # 1. 涨停检查 (不能买入)
        t1_open_chg = (t1_open - t1_pre) / t1_pre * 100
        if t1_open_chg >= (limit_pct - LIMIT_THRESHOLD):
            skipped_limit_up += 1
            equity.append(equity[-1])
            continue

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
            'date': d,
            'ts_code': ts_code,
            'prob': pick['prob'],
            'buy_price': buy_price,
            'sell_price': sell_price,
            'ret': ret,
            't1_open_chg': t1_open_chg,
            't2_low_chg': t2_low_chg,
        })

        new_equity = equity[-1] * (1 + ret)
        equity.append(new_equity)

    equity = equity[1:]

    print(f"\n{'='*60}")
    print("回测统计")
    print(f"{'='*60}")
    print(f"总交易日: {len(test_dates)}")
    print(f"空仓天数: {no_signal_days}")
    print(f"交易次数: {len(trades)}")
    print(f"跳过涨停买入: {skipped_limit_up}")
    print(f"跌停日卖出: {skipped_limit_down}")

    if not trades:
        print("无交易!")
        return None, None, None

    trades_df = pd.DataFrame(trades)
    eq_arr = np.array(equity)
    rets = np.diff(eq_arr) / eq_arr[:-1]

    total_ret = eq_arr[-1] / eq_arr[0] - 1
    sharpe = np.mean(rets) / (np.std(rets) + 1e-8) * np.sqrt(252)
    max_dd = np.max(1 - eq_arr / np.maximum.accumulate(eq_arr))
    win_rate = (trades_df['ret'] > 0).mean()
    avg_ret = trades_df['ret'].mean()

    print(f"\n总收益: {total_ret*100:.2f}%")
    print(f"夏普比率: {sharpe:.2f}")
    print(f"最大回撤: {max_dd*100:.2f}%")
    print(f"胜率: {win_rate*100:.2f}%")
    print(f"平均收益: {avg_ret*100:.2f}%")
    print(f"最高单笔: {trades_df['ret'].max()*100:.2f}%")
    print(f"最低单笔: {trades_df['ret'].min()*100:.2f}%")

    # 保存结果
    trades_df.to_csv(os.path.join(THIS_DIR, 'trades_final_method_fixed.csv'), index=False)
    pd.DataFrame({'equity': equity}).to_csv(os.path.join(THIS_DIR, 'equity_final_method_fixed.csv'), index=False)
    print(f"\n结果已保存")

    return trades_df, equity, {
        'total_ret': total_ret,
        'sharpe': sharpe,
        'max_dd': max_dd,
        'win_rate': win_rate,
        'avg_ret': avg_ret,
        'n_trades': len(trades_df)
    }


if __name__ == "__main__":
    backtest_fixed()

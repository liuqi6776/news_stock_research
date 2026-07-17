"""
Step 4: Predict and select stocks with TS feature filtering
Uses doubao_t1t2_model to predict, then applies TS score filtering
Trading logic: T select -> T+1 buy (open) -> T+2 sell (close)
"""
import os
import json
import sys
import datetime
import pandas as pd
import numpy as np
import joblib

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

TS_WEIGHTS = {
    'w_ret1d': -0.5,
    'w_dwr': 3.0,
    'w_dcc': -1.5,
    'w_ret5d': -0.2,
    'w_ma5': -0.3,
}


def is_main_board(ts_code):
    return ts_code.startswith('60') or ts_code.startswith('00')


def process_news(news_dir, target_date=None):
    market_records = []
    stock_records = []
    name_code_map = {}
    if os.path.exists(RANK_DIR):
        for f in sorted(os.listdir(RANK_DIR), reverse=True)[:5]:
            if f.endswith('.parquet'):
                try:
                    rdf = pd.read_parquet(os.path.join(RANK_DIR, f), columns=['ts_code', 'ts_name'])
                    for _, r in rdf.drop_duplicates(subset='ts_code').iterrows():
                        if r['ts_name'] not in name_code_map:
                            name_code_map[r['ts_name']] = r['ts_code']
                except:
                    pass
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
            name = s.get("stock_name", "")
            impact = float(s.get("impact", 0))
            ts_code = ""
            if code:
                ts_code = f"{code}.SH" if code.startswith('6') else f"{code}.SZ" if (
                        code.startswith('0') or code.startswith('3')) else code
            elif name and name in name_code_map:
                ts_code = name_code_map[name]
            if ts_code:
                stock_records.append(
                    {'trade_date': trade_date, 'ts_code': ts_code, 'news_stock_impact': impact})
    return pd.DataFrame(market_records), pd.DataFrame(stock_records)


def add_news_features(df, d_curr, news_mkt, news_stk):
    if not news_mkt.empty:
        nm = news_mkt.copy()
        df['news_market_impact'] = nm['news_market_impact'].max() if not nm.empty else 0.0
    else:
        df['news_market_impact'] = 0.0
    if not news_stk.empty:
        ns = news_stk.copy()
        ns_agg = ns.groupby('ts_code')['news_stock_impact'].max().reset_index()
        df = pd.merge(df, ns_agg[['ts_code', 'news_stock_impact']], on='ts_code', how='left')
        df['news_stock_impact'] = df['news_stock_impact'].fillna(0.0)
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
                print(f"  筹码数据使用最近可用日期: {cd}")
                break
        else:
            print(f"  警告: 没有可用的筹码数据")
            return None
    rank_df = pd.read_parquet(p_rank)
    rank_df = rank_df.sort_values('hot', ascending=False).drop_duplicates(subset='ts_code', keep='first')
    rank_df['hot_rank_pct'] = rank_df['hot'].rank(pct=True)
    chip_df = pd.read_parquet(p_chip)
    chip_df['chip_concentration'] = (chip_df['cost_85pct'] - chip_df['cost_15pct']) / (
            chip_df['cost_50pct'] + 1e-8)
    price_df = pd.read_parquet(p_price,
                               columns=['ts_code', 'close', 'pct_chg', 'amount', 'vol', 'pre_close', 'open', 'high',
                                        'low'])
    other_df = pd.read_parquet(p_other, columns=['ts_code', 'circ_mv'])
    df = pd.merge(rank_df[['ts_code', 'hot_rank_pct']], price_df, on='ts_code')
    df = pd.merge(df, chip_df[['ts_code', 'chip_concentration', 'winner_rate']], on='ts_code', how='left')
    df['chip_concentration'] = df['chip_concentration'].fillna(df['chip_concentration'].median() if df['chip_concentration'].notna().any() else 0.1)
    df['winner_rate'] = df['winner_rate'].fillna(df['winner_rate'].median() if df['winner_rate'].notna().any() else 50.0)
    df = pd.merge(df, other_df, on='ts_code', how='left')
    df = df[df['ts_code'].apply(is_main_board)]
    df = df[df['circ_mv'] <= CIRC_MV_LIMIT]
    df = add_news_features(df, d_curr, news_mkt, news_stk)
    for f in FEATS:
        if f not in df.columns:
            df[f] = 0.0
    return df


def compute_ts_features(df, d_curr):
    all_dates = sorted([f.replace('.parquet', '') for f in os.listdir(PRICE_DIR) if f.endswith('.parquet')])
    date_idx = {d: i for i, d in enumerate(all_dates)}
    curr_idx = date_idx.get(d_curr)
    if curr_idx is None or curr_idx < 5:
        df['ret_1d'] = 0.0
        df['delta_winner_rate'] = 0.0
        df['delta_chip_conc'] = 0.0
        df['ret_5d'] = 0.0
        df['ma5_dist'] = 0.0
        df['ts_score'] = 0.0
        return df

    d_prev = all_dates[curr_idx - 1]
    d_prev5 = all_dates[curr_idx - 5]

    p_prev_price = os.path.join(PRICE_DIR, f"{d_prev}.parquet")
    p_prev5_price = os.path.join(PRICE_DIR, f"{d_prev5}.parquet")
    p_prev_chip = os.path.join(CHIP_DIR, f"{d_prev}.parquet")

    curr_close = df.set_index('ts_code')['close']

    if os.path.exists(p_prev_price):
        prev_price = pd.read_parquet(p_prev_price, columns=['ts_code', 'close'])
        prev_close = prev_price.set_index('ts_code')['close']
        common = curr_close.index.intersection(prev_close.index)
        df['ret_1d'] = 0.0
        df.loc[df['ts_code'].isin(common), 'ret_1d'] = (
                curr_close[common] / prev_close[common] - 1
        ).values
    else:
        df['ret_1d'] = 0.0

    if os.path.exists(p_prev_chip):
        prev_chip = pd.read_parquet(p_prev_chip, columns=['ts_code', 'winner_rate', 'cost_85pct', 'cost_15pct',
                                                           'cost_50pct'])
        prev_chip['chip_concentration'] = (prev_chip['cost_85pct'] - prev_chip['cost_15pct']) / (
                prev_chip['cost_50pct'] + 1e-8)
        curr_wr = df.set_index('ts_code')['winner_rate']
        curr_cc = df.set_index('ts_code')['chip_concentration']
        prev_wr = prev_chip.set_index('ts_code')['winner_rate']
        prev_cc = prev_chip.set_index('ts_code')['chip_concentration']
        common_wr = curr_wr.index.intersection(prev_wr.index)
        common_cc = curr_cc.index.intersection(prev_cc.index)
        df['delta_winner_rate'] = 0.0
        df.loc[df['ts_code'].isin(common_wr), 'delta_winner_rate'] = (
                curr_wr[common_wr] - prev_wr[common_wr]
        ).values
        df['delta_chip_conc'] = 0.0
        df.loc[df['ts_code'].isin(common_cc), 'delta_chip_conc'] = (
                curr_cc[common_cc] - prev_cc[common_cc]
        ).values
    else:
        df['delta_winner_rate'] = 0.0
        df['delta_chip_conc'] = 0.0

    if os.path.exists(p_prev5_price):
        prev5_price = pd.read_parquet(p_prev5_price, columns=['ts_code', 'close'])
        prev5_close = prev5_price.set_index('ts_code')['close']
        common5 = curr_close.index.intersection(prev5_close.index)
        df['ret_5d'] = 0.0
        df.loc[df['ts_code'].isin(common5), 'ret_5d'] = (
                curr_close[common5] / prev5_close[common5] - 1
        ).values
    else:
        df['ret_5d'] = 0.0

    ma5_dates = all_dates[curr_idx - 5:curr_idx]
    ma5_frames = []
    for md in ma5_dates:
        mp = os.path.join(PRICE_DIR, f"{md}.parquet")
        if os.path.exists(mp):
            mdf = pd.read_parquet(mp, columns=['ts_code', 'close'])
            ma5_frames.append(mdf.set_index('ts_code')['close'])
    if ma5_frames:
        ma5_df = pd.DataFrame(ma5_frames)
        ma5_mean = ma5_df.mean()
        common_ma5 = curr_close.index.intersection(ma5_mean.index)
        df['ma5_dist'] = 0.0
        df.loc[df['ts_code'].isin(common_ma5), 'ma5_dist'] = (
                curr_close[common_ma5] / ma5_mean[common_ma5] - 1
        ).values
    else:
        df['ma5_dist'] = 0.0

    df['ts_score'] = (
            df['ret_1d'].abs() * TS_WEIGHTS['w_ret1d']
            + df['delta_winner_rate'] * TS_WEIGHTS['w_dwr']
            + df['delta_chip_conc'].abs() * TS_WEIGHTS['w_dcc']
            + df['ret_5d'].abs() * TS_WEIGHTS['w_ret5d']
            + df['ma5_dist'].abs() * TS_WEIGHTS['w_ma5']
    )

    return df


def select(df, model, feats, use_ts_filter=True):
    X = df[feats].fillna(0)
    if len(X) == 0:
        return pd.DataFrame(), 'NO_DATA'
    df['prob'] = model.predict_proba(X)[:, 1]

    best_prob = df['prob'].max()
    print(f"  最高概率: {best_prob:.4f} (最低阈值: {MIN_PROB_THRESHOLD})")

    if best_prob < MIN_PROB_THRESHOLD:
        print(f"  [空仓] 最高概率 {best_prob:.4f} < {MIN_PROB_THRESHOLD}，无达标股票")
        print(f"  [空仓] 回测数据: prob<0.6 胜率<50%，建议今日空仓")
        return pd.DataFrame(), 'CASH'

    high_prob = df[df['prob'] > 0.8].sort_values('prob', ascending=False).head(3)
    if not high_prob.empty:
        picks = high_prob
        signal = 'STRONG'
    else:
        above_min = df[df['prob'] >= MIN_PROB_THRESHOLD].sort_values('prob', ascending=False).head(3)
        if not above_min.empty:
            news_in = above_min[above_min['news_stock_impact'] > 0]
            no_news_in = above_min[above_min['news_stock_impact'] <= 0]
            if not news_in.empty:
                picks = news_in.head(3)
            else:
                picks = above_min.head(3)
            signal = 'MODERATE'
        else:
            print(f"  [空仓] 无股票达到最低概率阈值 {MIN_PROB_THRESHOLD}")
            return pd.DataFrame(), 'CASH'

    if use_ts_filter and 'ts_score' in picks.columns:
        no_news = picks[picks['news_stock_impact'] <= 0]
        news_stocks = picks[picks['news_stock_impact'] > 0]
        ts_filtered_no_news = no_news[no_news['ts_score'] > 0]
        picks = pd.concat([news_stocks, ts_filtered_no_news]).drop_duplicates(subset='ts_code')
        if picks.empty:
            print("  [TS过滤] 所有候选股ts_score<=0，保留新闻正面股票")
            picks = df[(df['news_stock_impact'] > 0) & (df['prob'] >= MIN_PROB_THRESHOLD)].sort_values('prob', ascending=False).head(3)
            if picks.empty:
                print(f"  [空仓] TS过滤后无达标股票，建议空仓")
                return pd.DataFrame(), 'CASH'

    return picks, signal


def run_prediction(target_date, use_ts_filter=True):
    print(f"\n{'=' * 60}")
    print(f"--- [Step 4] 预测和选股: {target_date} ---")
    print(f"  TS过滤: {'开启' if use_ts_filter else '关闭'}")
    print(f"{'=' * 60}")

    if not os.path.exists(MODEL_PATH):
        print(f"错误: 模型文件不存在: {MODEL_PATH}")
        print("请先运行 3_train_model.py")
        return

    loaded = joblib.load(MODEL_PATH)
    model = loaded[0] if isinstance(loaded, tuple) else loaded
    feats = loaded[1] if isinstance(loaded, tuple) else FEATS

    print(f"加载模型成功，特征: {feats}")

    news_mkt, news_stk = process_news(NEWS_MAJOR_DIR, None)
    print(f"  新闻数据: market={len(news_mkt)}条, stock={len(news_stk)}条")
    if not news_stk.empty:
        ns_agg = news_stk.groupby('ts_code')['news_stock_impact'].max().reset_index()
        main_board_news = ns_agg[ns_agg['ts_code'].apply(is_main_board)]
        print(f"  主板新闻股票: {len(main_board_news)}只")
        for _, r in main_board_news.head(10).iterrows():
            print(f"    {r['ts_code']} impact={r['news_stock_impact']}")

    df = load_features(target_date, news_mkt, news_stk)
    if df is None:
        print(f"错误: 缺少 {target_date} 的数据文件")
        print(f"请先运行 2_process_data.py {target_date}")
        return

    print(f"加载 {target_date} 数据: {len(df)} 只股票")

    if use_ts_filter:
        print("计算TS特征...")
        df = compute_ts_features(df, target_date)
        print(f"  ret_1d: mean={df['ret_1d'].mean():.4f}")
        print(f"  delta_winner_rate: mean={df['delta_winner_rate'].mean():.4f}")
        print(f"  ts_score: mean={df['ts_score'].mean():.4f}")

    picks, signal = select(df, model, feats, use_ts_filter=use_ts_filter)

    if picks.empty or signal == 'CASH':
        print(f"\n{'=' * 60}")
        print(f"  🛑 今日建议: 空仓")
        print(f"{'=' * 60}")
        print(f"  原因: 无股票达到最低概率阈值 ({MIN_PROB_THRESHOLD})")
        print(f"  回测依据: prob < {MIN_PROB_THRESHOLD} 时胜率 < 50%，期望收益为负")
        print(f"  策略纪律: 宁可空仓，不强行推亏钱的票")

        result = {
            'date': target_date,
            'ts_filter': use_ts_filter,
            'signal': 'CASH',
            'reason': f'best_prob < {MIN_PROB_THRESHOLD}',
            'picks': []
        }
        result_file = os.path.join(THIS_DIR, f"prediction_{target_date}.json")
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"\n结果已保存: {result_file}")
        return

    signal_desc = {
        'STRONG': '强信号 (prob>0.8, 回测胜率>90%)',
        'MODERATE': '中等信号 (prob>=0.6, 回测胜率~60%)',
    }

    print(f"\n{'=' * 60}")
    print(f"--- 最终选股结果 ---")
    print(f"  信号强度: {signal_desc.get(signal, signal)}")
    print(f"{'=' * 60}")

    header = f"{'排名':<6}{'股票代码':<12}{'概率':<10}{'市值(亿)':<12}{'新闻影响':<10}"
    if use_ts_filter:
        header += f"{'ts_score':<12}{'ret_1d':<10}{'delta_WR':<12}"
    print(header)
    print('-' * len(header))

    for i, (idx, row) in enumerate(picks.iterrows()):
        line = f"{i + 1:<6}{row['ts_code']:<12}{row['prob']:.4f}    {row['circ_mv'] / 10000:<10.2f}    {row['news_stock_impact']:.2f}     "
        if use_ts_filter:
            line += f"{row.get('ts_score', 0):.4f}      {row.get('ret_1d', 0):.4f}    {row.get('delta_winner_rate', 0):.4f}"
        print(line)

    print(f"{'=' * 60}")

    result = {
        'date': target_date,
        'ts_filter': use_ts_filter,
        'signal': signal,
        'picks': []
    }
    for i, (idx, row) in enumerate(picks.iterrows()):
        pick_info = {
            'rank': i + 1,
            'ts_code': row['ts_code'],
            'prob': round(float(row['prob']), 4),
            'circ_mv_yi': round(float(row['circ_mv'] / 10000), 2),
            'news_stock_impact': round(float(row['news_stock_impact']), 2),
        }
        if use_ts_filter:
            pick_info['ts_score'] = round(float(row.get('ts_score', 0)), 4)
            pick_info['ret_1d'] = round(float(row.get('ret_1d', 0)), 4)
            pick_info['delta_winner_rate'] = round(float(row.get('delta_winner_rate', 0)), 4)
        result['picks'].append(pick_info)

    result_file = os.path.join(THIS_DIR, f"prediction_{target_date}.json")
    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"\n选股结果已保存: {result_file}")

    print(f"\n交易提醒:")
    print(f"  T+1日 ({target_date}的下一个交易日) 开盘买入上述股票")
    print(f"  T+2日 收盘前卖出")
    if use_ts_filter:
        print(f"  已启用TS综合评分过滤 (ts_score > 0)")


if __name__ == "__main__":
    use_ts = True
    date_arg = None
    for arg in sys.argv[1:]:
        if arg == '--no-ts':
            use_ts = False
        else:
            date_arg = arg

    if date_arg:
        target_date = date_arg
    else:
        target_date = datetime.datetime.now().strftime("%Y%m%d")

    run_prediction(target_date, use_ts_filter=use_ts)

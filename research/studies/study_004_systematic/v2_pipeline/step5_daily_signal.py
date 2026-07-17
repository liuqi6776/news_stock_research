"""
Step 5: 每日信号生成 (生产环境)

在每天收盘后运行，生成次日买入信号
使用最新的Walk-Forward模型和最优threshold/max_positions

输入: 当日原始数据 + models/latest_wf_model.joblib + results/optimized_params_v2.json
输出: signals/YYYYMMDD.json (次日买入信号)

耗时: 约1-2分钟
运行频率: 每个交易日收盘后 (15:30之后)
"""
import os
import sys
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))
from shared.data_loader import get_all_dates

from config import (
    RAW_PRICE_DIR, RAW_OTHER_DIR, RAW_NEWS_DIR, RAW_RANK_DIR,
    RAW_MONEYFLOW_DIR, RAW_THS_NEWS_DIR,
    MODELS_DIR, RESULTS_DIR, SIGNALS_DIR,
    OPT_RESULTS_FILE, MIN_LISTING_DAYS
)


def is_main_board(ts_code: str) -> bool:
    return ts_code.startswith(('60', '00', '002', '003'))


def load_today_features(date_str):
    p = os.path.join(RAW_PRICE_DIR, f"{date_str}.parquet")
    if not os.path.exists(p):
        print(f"错误: 价格数据不存在 {date_str}")
        return None

    df = pd.read_parquet(p)
    df = df[df['ts_code'].apply(is_main_board)]
    if df.empty:
        return None

    df['pct_chg'] = (df['close'] - df['pre_close']) / df['pre_close']
    df['amplitude'] = (df['high'] - df['low']) / df['pre_close']
    df['body_size'] = abs(df['close'] - df['open']) / df['pre_close']
    df['upper_shadow'] = (df['high'] - df[['close', 'open']].max(axis=1)) / df['pre_close']
    df['lower_shadow'] = (df[['close', 'open']].min(axis=1) - df['low']) / df['pre_close']
    df['is_yang'] = (df['close'] > df['open']).astype(int)
    df['gap'] = (df['open'] - df['pre_close']) / df['pre_close']
    df['close_to_high'] = (df['high'] - df['close']) / df['pre_close']
    df['close_to_low'] = (df['close'] - df['low']) / df['pre_close']
    df['vol_ratio'] = 1.0
    df['vol_amount'] = df['close'] * df['vol']

    fund_p = os.path.join(RAW_OTHER_DIR, f"{date_str}.parquet")
    if os.path.exists(fund_p):
        try:
            fund_df = pd.read_parquet(fund_p)
            cols = ['ts_code']
            for c in ['pe', 'pb', 'circ_mv', 'turnover_rate', 'volume_ratio']:
                if c in fund_df.columns:
                    cols.append(c)
            if len(cols) > 1:
                fund_subset = fund_df[cols].copy()
                if 'circ_mv' in fund_subset.columns:
                    fund_subset['log_circ_mv'] = np.log1p(fund_subset['circ_mv'])
                if 'pe' in fund_subset.columns:
                    fund_subset['pe'] = fund_subset['pe'].replace([np.inf, -np.inf], np.nan)
                    fund_subset['log_pe'] = np.log1p(fund_subset['pe'].abs())
                if 'pb' in fund_subset.columns:
                    fund_subset['pb'] = fund_subset['pb'].replace([np.inf, -np.inf], np.nan)
                    fund_subset['log_pb'] = np.log1p(fund_subset['pb'].abs())
                df = pd.merge(df, fund_subset, on='ts_code', how='left')
        except:
            pass

    news_p = os.path.join(RAW_NEWS_DIR, f"analysis_{date_str[:4]}-{date_str[4:6]}-{date_str[6:]}.json")
    if os.path.exists(news_p):
        try:
            with open(news_p, 'r', encoding='utf-8') as f:
                data = json.load(f)
            rows = []
            market_impact = data.get('market_impact', 0)
            for stock in data.get('stocks', []):
                code = stock.get('stock_code', '')
                if len(code) == 6:
                    ts_code = code + '.SH' if code.startswith('6') else code + '.SZ'
                    rows.append({
                        'ts_code': ts_code,
                        'news_stock_impact': stock.get('impact', 0),
                        'news_market_impact': market_impact,
                        'news_has_mention': 1,
                    })
            if rows:
                news_df = pd.DataFrame(rows)
                df = pd.merge(df, news_df, on='ts_code', how='left')
                df['news_stock_impact'] = df['news_stock_impact'].fillna(0)
                df['news_market_impact'] = df['news_market_impact'].fillna(0)
                df['news_has_mention'] = df['news_has_mention'].fillna(0)
        except:
            pass

    rank_p = os.path.join(RAW_RANK_DIR, f"{date_str}.parquet")
    if os.path.exists(rank_p):
        try:
            rank_df = pd.read_parquet(rank_p)
            if 'hot' in rank_df.columns and len(rank_df) > 0:
                rank_subset = rank_df[['ts_code', 'hot']].copy()
                rank_subset = rank_subset.rename(columns={'hot': 'ths_hot'})
                rank_subset['ths_hot_rank'] = rank_subset['ths_hot'].rank(ascending=False, method='min')
                df = pd.merge(df, rank_subset, on='ts_code', how='left')
                df['ths_hot'] = df['ths_hot'].fillna(0)
                df['ths_hot_rank'] = df['ths_hot_rank'].fillna(9999)
        except:
            pass

    mf_p = os.path.join(RAW_MONEYFLOW_DIR, f"{date_str}.parquet")
    if os.path.exists(mf_p):
        try:
            mf_df = pd.read_parquet(mf_p)
            cols = ['ts_code']
            for c in ['net_mf_amount', 'net_mf_vol', 'buy_lg_amount', 'sell_lg_amount',
                       'buy_elg_amount', 'sell_elg_amount']:
                if c in mf_df.columns:
                    cols.append(c)
            if len(cols) > 1:
                mf_subset = mf_df[cols].copy()
                if 'net_mf_amount' in mf_subset.columns:
                    mf_subset['net_mf_amount_norm'] = mf_subset['net_mf_amount'] / (
                        mf_subset['net_mf_amount'].abs().quantile(0.95) + 1e-8)
                df = pd.merge(df, mf_subset, on='ts_code', how='left')
        except:
            pass

    tn_p = os.path.join(RAW_THS_NEWS_DIR, f"{date_str}.parquet")
    if os.path.exists(tn_p):
        try:
            tn_df = pd.read_parquet(tn_p)
            cols = ['ts_code']
            for c in ['new_gs', 'new_bs', 'new_gi']:
                if c in tn_df.columns:
                    cols.append(c)
            if len(cols) > 1:
                df = pd.merge(df, tn_df[cols], on='ts_code', how='left')
        except:
            pass

    return df


def run(target_date=None):
    print("=" * 80)
    print("Step 5: 每日信号生成")
    print("=" * 80)

    model_path = os.path.join(MODELS_DIR, 'latest_wf_model.joblib')
    feats_path = os.path.join(MODELS_DIR, 'latest_wf_features.joblib')

    if not os.path.exists(model_path):
        print("错误: 模型不存在，请先运行 step2_walkforward_predict.py")
        return
    if not os.path.exists(OPT_RESULTS_FILE):
        print("错误: 最优参数不存在，请先运行 step3_optimize_threshold.py")
        return

    import joblib
    model = joblib.load(model_path)
    feature_cols = joblib.load(feats_path)

    with open(OPT_RESULTS_FILE, 'r') as f:
        opt_params = json.load(f)
    threshold = opt_params['best_params']['threshold']
    max_positions = opt_params['best_params']['max_positions']

    print(f"模型: {model_path}")
    print(f"最优参数: threshold={threshold:.2f}, max_positions={max_positions}")

    if target_date is None:
        all_dates = get_all_dates()
        today_str = datetime.now().strftime('%Y%m%d')
        target_date = None
        for d in reversed(all_dates):
            if d <= today_str:
                target_date = d
                break
        if target_date is None:
            print("错误: 无法确定最近交易日")
            return

    print(f"信号日期: {target_date}")

    df = load_today_features(target_date)
    if df is None:
        print("错误: 无法加载当日数据")
        return

    all_dates = get_all_dates()
    date_idx = {d: i for i, d in enumerate(all_dates)}
    curr_idx = date_idx.get(target_date, 0)

    first_seen = {}
    for d in all_dates:
        p = os.path.join(RAW_PRICE_DIR, f"{d}.parquet")
        if not os.path.exists(p):
            continue
        try:
            day_df = pd.read_parquet(p, columns=['ts_code'])
            for code in day_df['ts_code'].unique():
                if code not in first_seen:
                    first_seen[code] = d
        except:
            pass

    codes_to_keep = []
    for code in df['ts_code'].unique():
        if code in first_seen:
            list_date = first_seen[code]
            list_idx = date_idx.get(list_date, 0)
            days_since_list = curr_idx - list_idx
            if days_since_list >= MIN_LISTING_DAYS:
                codes_to_keep.append(code)
    df = df[df['ts_code'].isin(codes_to_keep)]
    print(f"过滤后股票数: {len(df)}")

    available_features = [c for c in feature_cols if c in df.columns]
    missing_features = [c for c in feature_cols if c not in df.columns]
    if missing_features:
        print(f"缺失特征 ({len(missing_features)}): {missing_features[:5]}...")
        for c in missing_features:
            df[c] = 0

    X = df[feature_cols].fillna(0)
    proba = model.predict_proba(X)[:, 1]
    df['prob'] = proba

    above = df[df['prob'] >= threshold].copy()
    above = above.sort_values('prob', ascending=False)
    selected = above.head(max_positions)

    signal = {
        'signal_date': target_date,
        'generated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'threshold': float(threshold),
        'max_positions': int(max_positions),
        'n_candidates': int(len(above)),
        'picks': [],
    }

    for _, row in selected.iterrows():
        pick = {
            'ts_code': row['ts_code'],
            'prob': float(row['prob']),
            'close': float(row['close']),
        }
        signal['picks'].append(pick)

    signal_path = os.path.join(SIGNALS_DIR, f'{target_date}.json')
    with open(signal_path, 'w', encoding='utf-8') as f:
        json.dump(signal, f, indent=2, ensure_ascii=False)

    print(f"\n信号已生成: {signal_path}")
    print(f"候选股票: {len(above)} 只 (prob >= {threshold:.2f})")
    print(f"选中股票: {len(selected)} 只 (top {max_positions})")
    for pick in signal['picks']:
        print(f"  {pick['ts_code']}: prob={pick['prob']:.4f}, close={pick['close']:.2f}")

    return signal


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--date', type=str, default=None, help='目标日期 YYYYMMDD')
    args = parser.parse_args()
    run(target_date=args.date)

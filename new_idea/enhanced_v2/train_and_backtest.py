"""
Enhanced v2: Train + Backtest with all new features
Compares: Baseline (5 features) vs Enhanced (5 + Chan + Lynch + Quant)
"""
import os
import sys
import json
import time
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import joblib
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from feature_engineering import (
    build_enhanced_features, download_fundamental_data,
    FUND_DIR, DATA_DIR, PRICE_DIR, OTHER_DIR, CHIP_DIR, RANK_DIR
)

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
NEWS_DIR = os.path.join(os.path.dirname(THIS_DIR), '..', 'final_method', 'news_major1')

BASE_FEATS = ['hot_rank_pct', 'chip_concentration', 'winner_rate',
              'news_market_impact', 'news_stock_impact']

CHAN_FEATS = ['chan_bi_count', 'chan_zhongshu_count', 'chan_zhongshu_width',
              'chan_macd_divergence', 'chan_bi_direction', 'chan_leave_zhongshu']

LYNCH_FEATS = ['lynch_peg', 'lynch_peg_rank', 'lynch_quality_score',
               'lynch_growth_value', 'lynch_roe_stability', 'lynch_earnings_momentum']

QUANT_FEATS = ['qf_mom_1d', 'qf_mom_3d', 'qf_mom_5d', 'qf_mom_10d', 'qf_mom_20d',
               'qf_reversal_1d', 'qf_reversal_3d', 'qf_realized_vol', 'qf_atr_pct',
               'qf_rsi_14', 'qf_bb_position', 'qf_ma_cross_5_10', 'qf_ma_cross_10_20',
               'qf_vol_ratio_5_20', 'qf_pv_corr']

ENHANCED_FEATS = BASE_FEATS + CHAN_FEATS + LYNCH_FEATS + QUANT_FEATS

LABEL_THRESHOLD = 0.04
CIRC_MV_LIMIT = 1000000
MIN_PROB = 0.6
COST_RATE = 0.003


def load_news_for_date(d_str):
    d_fmt = f"{d_str[:4]}-{d_str[4:6]}-{d_str[6:8]}"
    news_file = os.path.join(NEWS_DIR, f"analysis_{d_fmt}.json")
    if not os.path.exists(news_file):
        return None, None

    with open(news_file, 'r', encoding='utf-8') as f:
        news = json.load(f)

    mkt_impact = news.get('market_impact', 0)
    mkt_df = pd.DataFrame([{'news_market_impact': mkt_impact}])

    stocks = news.get('stocks', [])
    stk_rows = []
    for s in stocks:
        code = s.get('stock_code', '')
        name = s.get('stock_name', '')
        impact = s.get('impact', 0)
        if not code and name:
            stk_rows.append({'stock_name': name, 'news_stock_impact': float(impact)})
        elif code:
            if '.' not in code:
                if code.startswith('6'):
                    code = code + '.SH'
                elif code.startswith('0') or code.startswith('3'):
                    code = code + '.SZ'
            stk_rows.append({'ts_code': code, 'news_stock_impact': float(impact)})

    stk_df = pd.DataFrame(stk_rows) if stk_rows else pd.DataFrame()
    return mkt_df, stk_df


def build_label(d_curr, all_dates, date_idx):
    curr_idx = date_idx[d_curr]
    if curr_idx + 2 >= len(all_dates):
        return None

    d_t1 = all_dates[curr_idx + 1]
    d_t2 = all_dates[curr_idx + 2]

    p_t1 = os.path.join(PRICE_DIR, f"{d_t1}.parquet")
    p_t2 = os.path.join(PRICE_DIR, f"{d_t2}.parquet")
    if not os.path.exists(p_t1) or not os.path.exists(p_t2):
        return None

    price_t1 = pd.read_parquet(p_t1, columns=['ts_code', 'open'])
    price_t2 = pd.read_parquet(p_t2, columns=['ts_code', 'close'])

    label_df = pd.merge(price_t1, price_t2, on='ts_code', how='inner')
    label_df['ret'] = label_df['close'] / label_df['open'] - 1
    label_df['label'] = (label_df['ret'] > LABEL_THRESHOLD).astype(int)

    return label_df[['ts_code', 'ret', 'label']]


def build_training_dataset(start_date='20230801', end_date='20260331'):
    print("Building training dataset with enhanced features...")

    if not os.path.exists(os.path.join(FUND_DIR, 'fina_indicator_cache.parquet')):
        print("  Fundamental data not found, downloading...")
        download_fundamental_data()

    all_dates = sorted([f.replace('.parquet', '') for f in os.listdir(PRICE_DIR) if f.endswith('.parquet')])
    date_idx = {d: i for i, d in enumerate(all_dates)}

    start_idx = date_idx.get(start_date, 0)
    end_idx = date_idx.get(end_date, len(all_dates) - 1)
    train_dates = all_dates[start_idx:end_idx + 1]

    news_dates = set()
    if os.path.exists(NEWS_DIR):
        for f in os.listdir(NEWS_DIR):
            if f.startswith('analysis_') and f.endswith('.json'):
                d_part = f.replace('analysis_', '').replace('.json', '')
                d_str = d_part.replace('-', '')
                news_dates.add(d_str)

    if news_dates:
        news_train = [d for d in train_dates if d in news_dates]
        non_news_train = [d for d in train_dates if d not in news_dates]
        sample_step = max(1, len(non_news_train) // 100)
        sampled_non_news = non_news_train[::sample_step]
        train_dates = sorted(news_train + sampled_non_news)
        print(f"  Using {len(news_train)} news dates + {len(sampled_non_news)} sampled dates = {len(train_dates)} total")
    else:
        sample_step = max(1, len(train_dates) // 200)
        train_dates = train_dates[::sample_step]
        print(f"  Sampling {len(train_dates)} dates (every {sample_step} days)")

    if not train_dates:
        print("  ERROR: No training dates!")
        return None

    print(f"  Training period: {train_dates[0]} ~ {train_dates[-1]}")

    all_samples = []
    for d in tqdm(train_dates, desc="  Building features"):
        mkt_news, stk_news = load_news_for_date(d)
        features = build_enhanced_features(d, mkt_news, stk_news)
        if features is None:
            continue

        labels = build_label(d, all_dates, date_idx)
        if labels is None:
            continue

        features = pd.merge(features, labels, on='ts_code', how='inner')
        features = features[features['circ_mv'] <= CIRC_MV_LIMIT]
        features = features[~features['ts_code'].str.startswith('688')]
        features = features[~features['ts_code'].str.startswith('689')]

        all_samples.append(features)

    if not all_samples:
        print("  ERROR: No training samples!")
        return None

    df = pd.concat(all_samples, ignore_index=True)
    print(f"  Total samples: {len(df)}, positive: {df['label'].sum()} ({df['label'].mean():.3f})")
    return df


def train_xgboost(df, feats, model_name='enhanced'):
    from xgboost import XGBClassifier
    from sklearn.model_selection import cross_val_score

    X = df[feats].fillna(0)
    y = df['label']

    scale = (y == 0).sum() / max((y == 1).sum(), 1)

    model = XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.7,
        scale_pos_weight=scale,
        min_child_weight=5,
        gamma=0.1,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        eval_metric='logloss',
        use_label_encoder=False,
    )

    cv_scores = cross_val_score(model, X, y, cv=5, scoring='roc_auc')
    print(f"  [{model_name}] CV AUC: {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}")

    model.fit(X, y)

    fi = pd.DataFrame({'feature': feats, 'importance': model.feature_importances_})
    fi = fi.sort_values('importance', ascending=False)
    print(f"\n  [{model_name}] Feature Importance (Top 15):")
    for _, row in fi.head(15).iterrows():
        print(f"    {row['feature']:30s} {row['importance']:.4f}")

    return model, fi


def backtest(model, feats, start_date='20230801', end_date='20260331', top_n=1, prob_thresh=0.4):
    print(f"\nBacktesting with {len(feats)} features, Top{top_n}, prob>={prob_thresh}...")

    all_dates = sorted([f.replace('.parquet', '') for f in os.listdir(PRICE_DIR) if f.endswith('.parquet')])
    date_idx = {d: i for i, d in enumerate(all_dates)}

    start_idx = date_idx.get(start_date, 0)
    end_idx = date_idx.get(end_date, len(all_dates) - 1)
    test_dates = all_dates[start_idx:end_idx - 2]

    trades = []
    equity = [100000.0]
    cash = 100000.0

    skipped_limit_up = 0
    skipped_limit_down = 0
    skipped_cyb = 0

    for d in tqdm(test_dates, desc="  Backtesting"):
        curr_idx = date_idx[d]
        if curr_idx + 2 >= len(all_dates):
            break

        d_t1 = all_dates[curr_idx + 1]
        d_t2 = all_dates[curr_idx + 2]

        mkt_news, stk_news = load_news_for_date(d)
        features = build_enhanced_features(d, mkt_news, stk_news)
        if features is None or len(features) == 0:
            equity.append(equity[-1])
            continue

        X = features[feats].fillna(0)
        features['prob'] = model.predict_proba(X)[:, 1]

        candidates = features[features['prob'] >= prob_thresh].sort_values('prob', ascending=False)

        if candidates.empty:
            equity.append(equity[-1])
            continue

        picks = candidates.head(top_n)

        p_t0 = os.path.join(PRICE_DIR, f"{d}.parquet")
        p_t1 = os.path.join(PRICE_DIR, f"{d_t1}.parquet")
        p_t2 = os.path.join(PRICE_DIR, f"{d_t2}.parquet")
        if not os.path.exists(p_t0) or not os.path.exists(p_t1) or not os.path.exists(p_t2):
            equity.append(equity[-1])
            continue

        price_t0 = pd.read_parquet(p_t0, columns=['ts_code', 'close'])
        price_t1 = pd.read_parquet(p_t1, columns=['ts_code', 'open', 'low', 'pre_close'])
        price_t2 = pd.read_parquet(p_t2, columns=['ts_code', 'close', 'low', 'pre_close'])

        daily_ret = 0
        n_exec = 0
        for _, pick in picks.iterrows():
            ts_code = pick['ts_code']

            # 1. 过滤创业板 (300/301开头)
            if ts_code.startswith('300') or ts_code.startswith('301'):
                skipped_cyb += 1
                continue

            t0_row = price_t0[price_t0['ts_code'] == ts_code]
            t1_row = price_t1[price_t1['ts_code'] == ts_code]
            t2_row = price_t2[price_t2['ts_code'] == ts_code]

            if t0_row.empty or t1_row.empty or t2_row.empty:
                continue

            t0_close = float(t0_row['close'].values[0])
            t1_open = float(t1_row['open'].values[0])
            t1_low = float(t1_row['low'].values[0])
            t1_pre = float(t1_row['pre_close'].values[0]) if 'pre_close' in t1_row.columns else t0_close
            t2_close = float(t2_row['close'].values[0])
            t2_low = float(t2_row['low'].values[0])

            # 2. 涨跌停限制：主板10%
            limit_pct = 10.0

            # T+1开盘相对前收盘涨幅 > 9.5% -> 不能买入（涨停开盘买不到）
            t1_open_chg = (t1_open - t1_pre) / t1_pre * 100
            if t1_open_chg >= 9.5:
                skipped_limit_up += 1
                continue

            # 3. T+2跌停检查：如果T+2最低价相对T+1开盘跌超过9.5%，视为跌停无法卖出
            # 使用T+2的开盘价作为卖出价（跌停时只能以跌停价卖出）
            t2_low_chg_from_t1_open = (t2_low - t1_open) / t1_open * 100
            if t2_low_chg_from_t1_open <= -9.5:
                # 跌停日：假设以T+2开盘价卖出（如果开盘就跌停）或跌停价卖出
                t2_open = float(t2_row['open'].values[0]) if 'open' in t2_row.columns else t2_close
                sell_p = min(t2_open, t1_open * 0.905)  # 最多按跌停价卖出
                skipped_limit_down += 1
            else:
                sell_p = t2_close

            buy_p = t1_open
            ret = sell_p / buy_p - 1 - COST_RATE

            trades.append({
                'date': d,
                'ts_code': ts_code,
                'prob': pick['prob'],
                'buy_open': buy_p,
                'sell_close': sell_p,
                'ret': ret,
                't1_open_chg': t1_open_chg,
                't2_low_chg': t2_low_chg_from_t1_open,
            })
            daily_ret += ret
            n_exec += 1

        if n_exec > 0:
            daily_ret = daily_ret / n_exec
        new_equity = equity[-1] * (1 + daily_ret)
        equity.append(new_equity)

    equity = equity[1:]
    print(f"  Skipped: limit_up={skipped_limit_up}, limit_down={skipped_limit_down}, cyb={skipped_cyb}")

    if not trades:
        print("  No trades!")
        return [], [], {}

    trades_df = pd.DataFrame(trades)
    eq_arr = np.array(equity)
    rets = np.diff(eq_arr) / eq_arr[:-1]

    n_trades = len(trades_df)
    win_rate = (trades_df['ret'] > 0).mean()
    avg_ret = trades_df['ret'].mean()
    total_ret = eq_arr[-1] / eq_arr[0] - 1
    sharpe = np.mean(rets) / (np.std(rets) + 1e-8) * np.sqrt(252)
    max_dd = np.max(1 - eq_arr / np.maximum.accumulate(eq_arr))

    stats = {
        'n_trades': n_trades,
        'win_rate': win_rate,
        'avg_ret': avg_ret,
        'total_ret': total_ret,
        'sharpe': sharpe,
        'max_dd': max_dd,
        'final_equity': eq_arr[-1],
    }

    print(f"\n  Results:")
    print(f"    Trades: {n_trades}")
    print(f"    Win Rate: {win_rate:.2%}")
    print(f"    Avg Return: {avg_ret:.2%}")
    print(f"    Total Return: {total_ret:.2%}")
    print(f"    Sharpe: {sharpe:.2f}")
    print(f"    Max Drawdown: {max_dd:.2%}")
    print(f"    Final Equity: {eq_arr[-1]:,.0f}")

    return trades_df, equity, stats


def run_comparison():
    print("=" * 60)
    print("Enhanced v2 Strategy Research")
    print("Comparing: Baseline (5 feats) vs Enhanced (5+Chan+Lynch+Quant)")
    print("=" * 60)

    df = build_training_dataset()
    if df is None:
        return

    print(f"\n{'=' * 60}")
    print("Training Baseline Model (5 features)")
    print("=" * 60)
    base_model, base_fi = train_xgboost(df, BASE_FEATS, 'Baseline')

    print(f"\n{'=' * 60}")
    print("Training Enhanced Model (all features)")
    print("=" * 60)
    enh_model, enh_fi = train_xgboost(df, ENHANCED_FEATS, 'Enhanced')

    print(f"\n{'=' * 60}")
    print("Training Chan-only Model (5 + Chan)")
    print("=" * 60)
    chan_feats = BASE_FEATS + CHAN_FEATS
    chan_model, chan_fi = train_xgboost(df, chan_feats, 'Chan')

    print(f"\n{'=' * 60}")
    print("Training Lynch-only Model (5 + Lynch)")
    print("=" * 60)
    lynch_feats = BASE_FEATS + LYNCH_FEATS
    lynch_model, lynch_fi = train_xgboost(df, lynch_feats, 'Lynch')

    print(f"\n{'=' * 60}")
    print("Training Quant-only Model (5 + Quant)")
    print("=" * 60)
    quant_feats = BASE_FEATS + QUANT_FEATS
    quant_model, quant_fi = train_xgboost(df, quant_feats, 'Quant')

    # Backtest all models
    print(f"\n{'=' * 60}")
    print("BACKTESTING ALL MODELS")
    print("=" * 60)

    results = {}

    for name, model, feats in [
        ('Baseline_5feat', base_model, BASE_FEATS),
        ('Enhanced_All', enh_model, ENHANCED_FEATS),
        ('Chan_Only', chan_model, chan_feats),
        ('Lynch_Only', lynch_model, lynch_feats),
        ('Quant_Only', quant_model, quant_feats),
    ]:
        print(f"\n--- {name} ---")
        trades, equity, stats = backtest(model, feats, top_n=1, prob_thresh=0.4)
        if stats:
            results[name] = stats
            trades_df = pd.DataFrame(trades)
            trades_df.to_csv(os.path.join(THIS_DIR, f'trades_{name}.csv'), index=False)
            pd.DataFrame({'equity': equity}).to_csv(
                os.path.join(THIS_DIR, f'equity_{name}.csv'), index=False)

    # Summary comparison
    print(f"\n{'=' * 60}")
    print("COMPARISON SUMMARY")
    print("=" * 60)
    summary = pd.DataFrame(results).T
    summary = summary[['n_trades', 'win_rate', 'avg_ret', 'total_ret', 'sharpe', 'max_dd']]
    print(summary.to_string())

    summary.to_csv(os.path.join(THIS_DIR, 'comparison_summary.csv'))

    # Save best model
    best_name = summary['sharpe'].idxmax()
    best_sharpe = summary.loc[best_name, 'sharpe']
    print(f"\nBest model by Sharpe: {best_name} (Sharpe={best_sharpe:.2f})")

    model_map = {
        'Baseline_5feat': (base_model, BASE_FEATS),
        'Enhanced_All': (enh_model, ENHANCED_FEATS),
        'Chan_Only': (chan_model, chan_feats),
        'Lynch_Only': (lynch_model, lynch_feats),
        'Quant_Only': (quant_model, quant_feats),
    }
    best_model, best_feats = model_map[best_name]
    joblib.dump(best_model, os.path.join(THIS_DIR, 'best_model.joblib'))
    joblib.dump(best_feats, os.path.join(THIS_DIR, 'best_feats.joblib'))
    print(f"Best model saved: best_model.joblib, features: {best_feats}")

    # Feature importance for best model
    enh_fi.to_csv(os.path.join(THIS_DIR, 'feature_importance_enhanced.csv'), index=False)

    return results


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == 'download_fund':
        download_fundamental_data()
    else:
        run_comparison()

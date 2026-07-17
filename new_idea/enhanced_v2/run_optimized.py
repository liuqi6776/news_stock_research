"""
Optimized research: Pre-compute features once, then train & backtest
"""
import sys
import os
import time
import pandas as pd
import numpy as np
import joblib
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from feature_engineering import (
    build_enhanced_features, download_fundamental_data,
    FUND_DIR, DATA_DIR, PRICE_DIR, OTHER_DIR, CHIP_DIR, RANK_DIR,
    _get_all_dates, _get_price_hist, _get_fina_df,
    compute_chan_features, compute_lynch_features, compute_quant_factors,
)
from train_and_backtest import (
    build_label, BASE_FEATS, CHAN_FEATS, LYNCH_FEATS, QUANT_FEATS, ENHANCED_FEATS,
    CIRC_MV_LIMIT, LABEL_THRESHOLD, THIS_DIR, NEWS_DIR
)

CACHE_DIR = os.path.join(THIS_DIR, "feature_cache")
os.makedirs(CACHE_DIR, exist_ok=True)


def build_training_dataset_fast(start_date='20230801', end_date='20260331'):
    print("Building training dataset (optimized with caching)...")

    if not os.path.exists(os.path.join(FUND_DIR, 'fina_indicator_cache.parquet')):
        print("  Fundamental data not found, downloading...")
        download_fundamental_data()

    all_dates, date_idx = _get_all_dates()

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
        sample_step = max(1, len(non_news_train) // 60)
        sampled_non_news = non_news_train[::sample_step]
        train_dates = sorted(news_train + sampled_non_news)
        print("  Using %d news dates + %d sampled dates = %d total" % (
            len(news_train), len(sampled_non_news), len(train_dates)))
    else:
        sample_step = max(1, len(train_dates) // 120)
        train_dates = train_dates[::sample_step]
        print("  Sampling %d dates (every %d days)" % (len(train_dates), sample_step))

    if not train_dates:
        print("  ERROR: No training dates!")
        return None

    print("  Training period: %s ~ %s" % (train_dates[0], train_dates[-1]))

    all_samples = []
    for d in tqdm(train_dates, desc="  Building features"):
        cache_file = os.path.join(CACHE_DIR, "feat_%s.parquet" % d)
        if os.path.exists(cache_file):
            features = pd.read_parquet(cache_file)
        else:
            features = build_enhanced_features(d, None, None)
            if features is not None:
                features.to_parquet(cache_file, index=False)

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
    print("  Total samples: %d, positive: %d (%.3f)" % (
        len(df), df['label'].sum(), df['label'].mean()))
    return df


def train_xgboost(df, feats, model_name='model'):
    from xgboost import XGBClassifier

    X = df[feats].fillna(0).replace([np.inf, -np.inf], 0)
    y = df['label']

    pos_count = y.sum()
    neg_count = len(y) - pos_count
    scale_pos = neg_count / (pos_count + 1)

    model = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos,
        eval_metric='logloss',
        use_label_encoder=False,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X, y)

    fi = pd.DataFrame({'feature': feats, 'importance': model.feature_importances_})
    fi = fi.sort_values('importance', ascending=False)

    print("  %s trained: %d samples, top feat: %s (%.4f)" % (
        model_name, len(df), fi.iloc[0]['feature'], fi.iloc[0]['importance']))
    return model, fi


def backtest_model(model, feats, top_n=1, prob_thresh=0.4):
    all_dates, date_idx = _get_all_dates()

    test_dates = all_dates[-60:]

    trades = []
    equity = [1.0]

    for d in tqdm(test_dates, desc="  Backtesting"):
        cache_file = os.path.join(CACHE_DIR, "feat_%s.parquet" % d)
        if os.path.exists(cache_file):
            features = pd.read_parquet(cache_file)
        else:
            features = build_enhanced_features(d, None, None)
            if features is not None:
                features.to_parquet(cache_file, index=False)

        if features is None:
            equity.append(equity[-1])
            continue

        features = features[features['circ_mv'] <= CIRC_MV_LIMIT]
        features = features[~features['ts_code'].str.startswith('688')]
        features = features[~features['ts_code'].str.startswith('689')]

        X = features[feats].fillna(0).replace([np.inf, -np.inf], 0)
        if len(X) == 0:
            equity.append(equity[-1])
            continue
        try:
            proba = model.predict_proba(X)
            if proba.shape[1] > 1:
                features['prob'] = proba[:, 1]
            else:
                features['prob'] = 0
        except Exception:
            equity.append(equity[-1])
            continue

        selected = features[features['prob'] >= prob_thresh].nlargest(top_n, 'prob')

        if len(selected) == 0:
            equity.append(equity[-1])
            continue

        d_idx = date_idx.get(d)
        if d_idx is None or d_idx + 2 >= len(all_dates):
            equity.append(equity[-1])
            continue

        d_t1 = all_dates[d_idx + 1]
        d_t2 = all_dates[d_idx + 2]

        p_t1 = os.path.join(PRICE_DIR, "%s.parquet" % d_t1)
        p_t2 = os.path.join(PRICE_DIR, "%s.parquet" % d_t2)

        if not os.path.exists(p_t1) or not os.path.exists(p_t2):
            equity.append(equity[-1])
            continue

        price_t1 = pd.read_parquet(p_t1, columns=['ts_code', 'open'])
        price_t2 = pd.read_parquet(p_t2, columns=['ts_code', 'close'])

        day_ret = 0
        for _, row in selected.iterrows():
            tc = row['ts_code']
            t1_open = price_t1.loc[price_t1['ts_code'] == tc, 'open']
            t2_close = price_t2.loc[price_t2['ts_code'] == tc, 'close']

            if len(t1_open) > 0 and len(t2_close) > 0:
                ret = t2_close.values[0] / t1_open.values[0] - 1
                day_ret += ret
                trades.append({
                    'date': d, 'ts_code': tc, 'prob': row['prob'],
                    'ret': ret
                })

        day_ret /= len(selected)
        equity.append(equity[-1] * (1 + day_ret))

    stats = {}
    if len(trades) > 0:
        trades_df = pd.DataFrame(trades)
        eq_arr = np.array(equity)
        rets = np.diff(eq_arr) / eq_arr[:-1]
        stats = {
            'sharpe': np.mean(rets) / (np.std(rets) + 1e-8) * np.sqrt(252),
            'total_ret': eq_arr[-1] / eq_arr[0] - 1,
            'win_rate': (trades_df['ret'] > 0).mean(),
            'avg_ret': trades_df['ret'].mean(),
            'n_trades': len(trades),
            'max_dd': np.max(1 - eq_arr / np.maximum.accumulate(eq_arr)),
        }

    return trades, equity, stats


if __name__ == "__main__":
    print("=" * 60)
    print("Enhanced v2 Strategy Research (Optimized)")
    print("=" * 60)

    df = build_training_dataset_fast()
    if df is None:
        sys.exit(1)

    print("\n--- Training Models ---")
    models = {}
    feat_sets = {
        'Baseline_5feat': BASE_FEATS,
        'Enhanced_All': ENHANCED_FEATS,
        'Chan_Only': BASE_FEATS + CHAN_FEATS,
        'Lynch_Only': BASE_FEATS + LYNCH_FEATS,
        'Quant_Only': BASE_FEATS + QUANT_FEATS,
    }

    for name, feats in feat_sets.items():
        print("\nTraining %s (%d features)..." % (name, len(feats)))
        model, fi = train_xgboost(df, feats, name)
        models[name] = (model, feats, fi)

    print("\n--- Backtesting Models ---")
    results = {}
    for name, (model, feats, fi) in models.items():
        print("\nBacktesting %s..." % name)
        trades, equity, stats = backtest_model(model, feats, top_n=1, prob_thresh=0.4)
        if stats:
            results[name] = stats
            pd.DataFrame(trades).to_csv(
                os.path.join(THIS_DIR, "trades_%s.csv" % name), index=False)
            pd.DataFrame({"equity": equity}).to_csv(
                os.path.join(THIS_DIR, "equity_%s.csv" % name), index=False)

    print("\n" + "=" * 70)
    print("COMPARISON SUMMARY")
    print("=" * 70)
    for name, stats in sorted(results.items(), key=lambda x: -x[1]['sharpe']):
        print("  %-20s: Sharpe=%6.2f, Total=%8.2f%%, WR=%5.2f%%, AvgRet=%5.2f%%, Trades=%3d, MaxDD=%5.2f%%" % (
            name, stats['sharpe'], stats['total_ret']*100, stats['win_rate']*100,
            stats['avg_ret']*100, stats['n_trades'], stats['max_dd']*100))

    best_name = max(results, key=lambda k: results[k]['sharpe']) if results else None
    if best_name:
        print("\nBest model: %s (Sharpe=%.2f)" % (best_name, results[best_name]['sharpe']))
        model, feats, fi = models[best_name]
        joblib.dump(model, os.path.join(THIS_DIR, "best_model.joblib"))
        joblib.dump(feats, os.path.join(THIS_DIR, "best_feats.joblib"))
        print("Saved best_model.joblib and best_feats.joblib")

        fi.to_csv(os.path.join(THIS_DIR, "feature_importance_%s.csv" % best_name), index=False)

        print("\nTop 15 Feature Importance (%s):" % best_name)
        for _, row in fi.head(15).iterrows():
            print("  %-25s: %.4f" % (row['feature'], row['importance']))

    summary = pd.DataFrame(results).T
    summary.to_csv(os.path.join(THIS_DIR, "comparison_summary.csv"))
    print("\nSummary saved to comparison_summary.csv")

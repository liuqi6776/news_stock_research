"""
step2_train_models.py
基于 features_005.parquet，进行按月滚动的 Walk-Forward 训练。
1. 样本时间衰减: sample_weight = exp(-ln(2) * (days_diff / 252))
2. 双模型:
   - xgb_up: predict target_up (> 1%)
   - xgb_crash: predict target_crash_bin (T+1 intraday < -3%)
输出 -> predictions/predictions_005_wf.parquet
"""
import os, sys, time, gc, warnings
import pandas as pd
import numpy as np
warnings.filterwarnings('ignore')
from xgboost import XGBClassifier

sys.stdout.reconfigure(line_buffering=True)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
STUDY_DIR = os.path.dirname(SCRIPT_DIR)
FEATURES_FILE = os.path.join(STUDY_DIR, 'data', 'features_005.parquet')
OUTPUT = os.path.join(STUDY_DIR, 'predictions', 'predictions_005_wf.parquet')
os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)

TRAIN_START = '20200101'
HALF_LIFE_DAYS = 252.0  # 半衰期 1 年 (约252个交易日)

def get_feature_cols(df):
    exclude_cols = {
        'ts_code', 'trade_date', 'ds', 'industry',
        'open', 'high', 'low', 'close', 'pre_close',
        'entry_price', 'next_open',
        'exit_price_1d', 'return_1d', 'return_1d_open',
        'exit_price_5d', 'return_5d', 'return_5d_open',
        'exit_price_28d', 'return_28d', 'return_28d_open',
        'exit_28d_close',
        'calc_ret5d', 'return_5d_from_open', 'return_28d_from_open',
        'entry_vs_close',
        'return_1d_open_old', 'actual_return',
        't1_intraday_return', 'target_crash_bin', 'target_up_bin',
        'index_ma20_bias'
    }
    return [c for c in df.columns
            if c not in exclude_cols
            and not c.startswith('hist_')
            and df[c].dtype in ('float64', 'float32', 'int64', 'int32')]

def run():
    t0 = time.time()
    print("Loading features 005...", flush=True)
    df = pd.read_parquet(FEATURES_FILE)
    df['ds'] = df['trade_date'].astype(str)
    
    # 构造 label_up
    df['label_up'] = df['target_up_bin'].fillna(0).astype(np.int8)
    
    # 如果 target_crash_bin 有 NaN，处理一下
    df['label_crash'] = df['target_crash_bin'].fillna(0).astype(np.int8)

    feature_cols = get_feature_cols(df)
    print(f"Rows: {len(df)}, Features: {len(feature_cols)}", flush=True)

    months = sorted(df['ds'].str[:6].unique())
    pred_months = [m for m in months if m >= '202201']
    print(f"Predict months: {len(pred_months)} ({pred_months[0]}-{pred_months[-1]})", flush=True)

    all_preds = []
    
    for i, month in enumerate(pred_months):
        mt0 = time.time()
        train_end = str(int(month) - 1)
        if train_end.endswith('00'):
            train_end = f"{int(train_end[:4])-1}12"

        # 12-month rolling training window (1 year) to keep training size highly optimal and adapt to fast regime shifts
        year = int(train_end[:4])
        month_val = int(train_end[4:6])
        start_year = year - 1
        start_month = month_val + 1
        if start_month > 12:
            start_month -= 12
            start_year += 1
        rolling_start = f"{start_year}{start_month:02d}01"

        train_mask = (df['ds'] >= rolling_start) & (df['ds'].str[:6] <= train_end) & df['return_1d_open'].notna()
        pred_mask = df['ds'].str[:6] == month

        train_df = df.loc[train_mask, feature_cols + ['ds', 'label_up', 'label_crash']].copy()
        pred_df = df.loc[pred_mask, feature_cols + ['trade_date', 'ts_code', 'return_1d_open', 'next_open', 'industry']].copy()

        if len(train_df) < 10000 or len(pred_df) == 0:
            continue

        X_train = train_df[feature_cols].fillna(0)
        y_up = train_df['label_up']
        y_crash = train_df['label_crash']
        
        # Time-decayed weights calculated directly on train_df
        train_dates = pd.to_datetime(train_df['ds'], format='%Y%m%d')
        ref_date = train_dates.max()
        days_diff = (ref_date - train_dates).dt.days
        decay_lambda = np.log(2) / HALF_LIFE_DAYS
        # High-performance Series mapping aligned with indices
        sample_weights_series = pd.Series(np.exp(-decay_lambda * days_diff).values, index=train_df.index)
        sample_weights_series = sample_weights_series / sample_weights_series.mean()

        # Downsample Negatives for UP Model (1 pos : 2 negs) - instant index mapping
        pos_up_idx = y_up[y_up == 1].index
        neg_up_idx = y_up[y_up == 0].index
        if len(pos_up_idx) > 0 and len(neg_up_idx) > 0:
            sampled_neg_up = y_up[neg_up_idx].sample(min(len(pos_up_idx) * 2, len(neg_up_idx)), random_state=42).index
            train_up_idx = pos_up_idx.union(sampled_neg_up)
            X_train_up = X_train.loc[train_up_idx]
            y_up_bal = y_up.loc[train_up_idx]
            weights_up = sample_weights_series.loc[train_up_idx].values
            weights_up = weights_up / weights_up.mean()
        else:
            X_train_up, y_up_bal, weights_up = X_train, y_up, sample_weights_series.values

        # Downsample Negatives for CRASH Model (1 pos : 2 negs) - instant index mapping
        pos_crash_idx = y_crash[y_crash == 1].index
        neg_crash_idx = y_crash[y_crash == 0].index
        if len(pos_crash_idx) > 0 and len(neg_crash_idx) > 0:
            sampled_neg_crash = y_crash[neg_crash_idx].sample(min(len(pos_crash_idx) * 2, len(neg_crash_idx)), random_state=43).index
            train_crash_idx = pos_crash_idx.union(sampled_neg_crash)
            X_train_crash = X_train.loc[train_crash_idx]
            y_crash_bal = y_crash.loc[train_crash_idx]
            weights_crash = sample_weights_series.loc[train_crash_idx].values
            weights_crash = weights_crash / weights_crash.mean()
        else:
            X_train_crash, y_crash_bal, weights_crash = X_train, y_crash, sample_weights_series.values

        # Model 1: UP Model (trained on balanced data)
        model_up = XGBClassifier(
            n_estimators=100, max_depth=5, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8,
            random_state=42, n_jobs=-1, eval_metric='logloss',
            tree_method='hist'
        )
        model_up.fit(X_train_up, y_up_bal, sample_weight=weights_up, verbose=False)
        
        # Model 2: CRASH Model (trained on balanced data)
        model_crash = XGBClassifier(
            n_estimators=80, max_depth=4, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8,
            random_state=43, n_jobs=-1, eval_metric='logloss',
            tree_method='hist'
        )
        model_crash.fit(X_train_crash, y_crash_bal, sample_weight=weights_crash, verbose=False)

        X_pred = pred_df[feature_cols].fillna(0)
        prob_up = model_up.predict_proba(X_pred)[:, 1]
        prob_crash = model_crash.predict_proba(X_pred)[:, 1]

        mp = pd.DataFrame({
            'trade_date': pred_df['trade_date'].values,
            'ts_code': pred_df['ts_code'].values,
            'industry': pred_df['industry'].values,
            'prob_up': prob_up,
            'prob_crash': prob_crash,
            'actual_return': pred_df['return_1d_open'].values,
            'entry_price': pred_df['next_open'].values if 'next_open' in pred_df.columns else np.nan,
        })
        all_preds.append(mp)

        elapsed = time.time() - mt0
        total = time.time() - t0
        remaining = (total / (i + 1)) * (len(pred_months) - i - 1)
        print(f"[{i+1}/{len(pred_months)}] {month}: train_up={len(X_train_up)}, train_crash={len(X_train_crash)}, "
              f"up_pos={y_up.mean():.1%}, crash_pos={y_crash.mean():.1%}, "
              f"elapsed={elapsed:.0f}s, rem~{remaining/60:.0f}m", flush=True)

        del train_df, pred_df, X_train, y_up, y_crash, model_up, model_crash, X_pred, mp
        gc.collect()

    if all_preds:
        combined = pd.concat(all_preds, ignore_index=True)
        combined.to_parquet(OUTPUT)
        print(f"\nSaved: {OUTPUT} ({len(combined)} rows)", flush=True)
    print(f"Total: {time.time()-t0:.0f}s", flush=True)

if __name__ == '__main__':
    run()

"""
Step 2: Monthly Walk-Forward Prediction

For each month from 2022-01 to latest:
  - Train XGBoost on all data up to previous month
  - Predict probabilities for current month
  - Store predictions with actual returns

Runtime: ~5-6 hours (51+ months x train+predict)

Output: predictions/predictions_1d_wf_monthly.parquet
  Columns: trade_date, ts_code, prob, target, actual_return
"""
import os
import sys
import pandas as pd
import numpy as np
import time
import warnings

warnings.filterwarnings('ignore')
sys.stdout.reconfigure(line_buffering=True)

RELEASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(RELEASE_DIR, 'data')
PREDICTIONS_DIR = os.path.join(RELEASE_DIR, 'predictions')
os.makedirs(PREDICTIONS_DIR, exist_ok=True)

FEATURES_FILE = os.path.join(DATA_DIR, 'all_features_v2.parquet')
OUTPUT_FILE = os.path.join(PREDICTIONS_DIR, 'predictions_1d_wf_monthly.parquet')

TRAIN_START = '20200101'
TARGET_THRESHOLD = 0.015
MIN_TRAIN_SAMPLES = 50000


def run():
    print("=" * 90)
    print("Walk-Forward Monthly Prediction (retrain each month)")
    print("=" * 90)

    if not os.path.exists(FEATURES_FILE):
        print(f"ERROR: Feature file not found: {FEATURES_FILE}")
        print("Please run step1_build_features.py first.")
        return

    print("Loading features...")
    features_df = pd.read_parquet(FEATURES_FILE)
    features_df['ds'] = features_df['trade_date'].astype(str)
    print(f"Data: {len(features_df)} rows, {len(features_df.columns)} columns")
    print(f"Date range: {features_df['ds'].min()} - {features_df['ds'].max()}")

    return_col = 'return_1d'
    if return_col not in features_df.columns:
        print(f"ERROR: Missing {return_col} column")
        return

    exclude_cols = ['ts_code', 'trade_date', 'ds', 'entry_price',
                    'exit_price_1d', 'return_1d',
                    'exit_price_5d', 'return_5d',
                    'exit_price_28d', 'return_28d']
    feature_cols = [c for c in features_df.columns if c not in exclude_cols and
                    not c.startswith('hist_') and
                    features_df[c].dtype in ['float64', 'float32', 'int64', 'int32']]
    print(f"Available features: {len(feature_cols)}")

    months = sorted(features_df['ds'].str[:6].unique())
    pred_months = [m for m in months if m >= '202201']
    print(f"Prediction months: {len(pred_months)} ({pred_months[0]} - {pred_months[-1]})")

    all_predictions = []
    total_start = time.time()

    from xgboost import XGBClassifier

    for i, month in enumerate(pred_months):
        month_start = time.time()

        train_end_month = str(int(month) - 1)
        if train_end_month.endswith('00'):
            year = int(train_end_month[:4]) - 1
            train_end_month = f"{year}12"

        train_mask = (features_df['ds'] >= TRAIN_START) & (features_df['ds'].str[:6] <= train_end_month)
        pred_mask = features_df['ds'].str[:6] == month

        train_df = features_df[train_mask & features_df[return_col].notna()].copy()
        pred_df = features_df[pred_mask].copy()

        if len(train_df) < MIN_TRAIN_SAMPLES:
            print(f"  [{i + 1}/{len(pred_months)}] {month}: Insufficient training data ({len(train_df)}), skip")
            continue
        if len(pred_df) == 0:
            print(f"  [{i + 1}/{len(pred_months)}] {month}: No prediction data, skip")
            continue

        train_df['label'] = (train_df[return_col] > TARGET_THRESHOLD).astype(int)
        pos_rate = train_df['label'].mean()

        X_train = train_df[feature_cols].fillna(0)
        y_train = train_df['label']

        model = XGBClassifier(
            n_estimators=100, max_depth=5, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8,
            random_state=42, n_jobs=-1, eval_metric='logloss'
        )
        model.fit(X_train, y_train)

        X_pred = pred_df[feature_cols].fillna(0)
        proba = model.predict_proba(X_pred)[:, 1]

        month_pred = pred_df[['trade_date', 'ts_code']].copy()
        month_pred['prob'] = proba
        month_pred['target'] = '1d'

        if return_col in pred_df.columns:
            month_pred['actual_return'] = pred_df[return_col].values

        all_predictions.append(month_pred)

        elapsed = time.time() - month_start
        total_elapsed = time.time() - total_start
        avg_per_month = total_elapsed / (i + 1)
        remaining = avg_per_month * (len(pred_months) - i - 1)

        n_above_58 = (proba >= 0.58).sum()
        n_above_64 = (proba >= 0.64).sum()
        print(f"  [{i + 1}/{len(pred_months)}] {month}: train={len(train_df)}, pred={len(pred_df)}, "
              f"pos_rate={pos_rate:.1%}, prob>=0.58={n_above_58}, prob>=0.64={n_above_64}, "
              f"elapsed={elapsed:.0f}s, remaining~{remaining / 60:.0f}min")

    if not all_predictions:
        print("ERROR: No predictions generated")
        return

    combined = pd.concat(all_predictions, ignore_index=True)
    combined.to_parquet(OUTPUT_FILE)
    total_time = (time.time() - total_start) / 60
    print(f"\nMonthly WF predictions saved: {OUTPUT_FILE}")
    print(f"Total predictions: {len(combined)} rows")
    print(f"Total time: {total_time:.1f} minutes")
    print("Done!")


if __name__ == '__main__':
    run()

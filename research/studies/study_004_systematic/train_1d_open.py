import os, sys, time, gc, warnings
import pandas as pd
import numpy as np
warnings.filterwarnings('ignore')
from xgboost import XGBClassifier

sys.stdout.reconfigure(line_buffering=True)

STUDY_DIR = os.path.dirname(os.path.abspath(__file__))
FEATURES_FILE = os.path.join(STUDY_DIR, 'data', 'all_features_v2.parquet')
OUTPUT = os.path.join(STUDY_DIR, 'predictions', 'predictions_1d_open_wf_monthly.parquet')
TRAIN_START = '20200101'

def get_feature_cols(df):
    exclude_cols = {'ts_code', 'trade_date', 'ds',
                    'open', 'high', 'low', 'close', 'pre_close',
                    'entry_price', 'next_open',
                    'exit_price_1d', 'return_1d', 'return_1d_open',
                    'exit_price_5d', 'return_5d', 'return_5d_open',
                    'exit_price_28d', 'return_28d', 'return_28d_open',
                    'exit_28d_close',
                    'calc_ret5d', 'return_5d_from_open', 'return_28d_from_open',
                    'entry_vs_close',
                    'return_1d_open_old', 'actual_return'}
    return [c for c in df.columns
            if c not in exclude_cols
            and not c.startswith('hist_')
            and df[c].dtype in ('float64', 'float32', 'int64', 'int32')]

def run():
    t0 = time.time()
    print("Loading features...", flush=True)
    df = pd.read_parquet(FEATURES_FILE)
    df['ds'] = df['trade_date'].astype(str)
    feature_cols = get_feature_cols(df)
    print(f"Rows: {len(df)}, Features: {len(feature_cols)}", flush=True)

    df['label_1d_open'] = (df['return_1d_open'] > 0.01).astype(np.int8)

    months = sorted(df['ds'].str[:6].unique())
    pred_months = [m for m in months if m >= '202201']
    print(f"Predict months: {len(pred_months)} ({pred_months[0]}-{pred_months[-1]})", flush=True)

    all_preds = []
    for i, month in enumerate(pred_months):
        mt0 = time.time()
        train_end = str(int(month) - 1)
        if train_end.endswith('00'):
            train_end = f"{int(train_end[:4])-1}12"

        train_mask = (df['ds'] >= TRAIN_START) & (df['ds'].str[:6] <= train_end) & df['return_1d_open'].notna()
        pred_mask = df['ds'].str[:6] == month

        train_df = df.loc[train_mask, feature_cols + ['label_1d_open']].copy()
        pred_df = df.loc[pred_mask, feature_cols + ['trade_date', 'ts_code', 'return_1d_open', 'next_open']].copy()

        if len(train_df) < 50000 or len(pred_df) == 0:
            continue

        X_train = train_df[feature_cols].fillna(0)
        y_train = train_df['label_1d_open']

        model = XGBClassifier(
            n_estimators=100, max_depth=5, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8,
            random_state=42, n_jobs=4, eval_metric='logloss',
            tree_method='hist'
        )
        model.fit(X_train, y_train, verbose=False)

        X_pred = pred_df[feature_cols].fillna(0)
        proba = model.predict_proba(X_pred)[:, 1]

        mp = pd.DataFrame({
            'trade_date': pred_df['trade_date'].values,
            'ts_code': pred_df['ts_code'].values,
            'prob': proba,
            'target': '1d_open',
            'actual_return': pred_df['return_1d_open'].values,
            'entry_price': pred_df['next_open'].values if 'next_open' in pred_df.columns else np.nan,
        })
        all_preds.append(mp)

        elapsed = time.time() - mt0
        total = time.time() - t0
        remaining = (total / (i + 1)) * (len(pred_months) - i - 1)
        print(f"[{i+1}/{len(pred_months)}] {month}: train={len(train_df)}, "
              f"pos={y_train.mean():.1%}, prob>=0.5={(proba>=0.5).sum()}, "
              f"elapsed={elapsed:.0f}s, remaining~{remaining/60:.0f}min", flush=True)

        del train_df, pred_df, X_train, y_train, model, X_pred, mp
        gc.collect()

    if all_preds:
        combined = pd.concat(all_preds, ignore_index=True)
        combined.to_parquet(OUTPUT)
        print(f"\nSaved: {OUTPUT} ({len(combined)} rows)", flush=True)
    print(f"Total: {time.time()-t0:.0f}s", flush=True)

if __name__ == '__main__':
    run()

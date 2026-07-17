"""
Walk-Forward 月级别预测 - 正确target版本

关键修正:
  旧target: return_5d = (T+5_close - T_close) / T_close  ← 收盘价买入，不可能实现
  新target: return_5d_open = (T+5_close - T+1_open) / T+1_open  ← 次日开盘买入，真实可执行
  新target: return_28d_open = (T+28_close - T+1_open) / T+1_open  ← 30日持有

同时训练5d和28d(30d)两个模型，输出两套预测

运行:
  cd study_004_systematic
  python -u run_wf_monthly_v2.py
"""
import os
import sys
import pandas as pd
import numpy as np
import json
import time
import warnings
warnings.filterwarnings('ignore')

sys.stdout.reconfigure(line_buffering=True)

STUDY_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(STUDY_DIR, 'data')
PREDICTIONS_DIR = os.path.join(STUDY_DIR, 'predictions')
os.makedirs(PREDICTIONS_DIR, exist_ok=True)

FEATURES_FILE = os.path.join(DATA_DIR, 'all_features_v2.parquet')
OUTPUT_5D = os.path.join(PREDICTIONS_DIR, 'predictions_5d_open_wf_monthly.parquet')
OUTPUT_28D = os.path.join(PREDICTIONS_DIR, 'predictions_28d_open_wf_monthly.parquet')

TRAIN_START = '20200101'
MIN_TRAIN_SAMPLES = 50000

TARGETS = [
    {
        'name': '5d_open',
        'return_col': 'return_5d_open',
        'threshold': 0.03,
        'output': OUTPUT_5D,
        'entry_price_col': 'next_open',
        'exit_price_col': 'exit_price_5d',
    },
    {
        'name': '28d_open',
        'return_col': 'return_28d_open',
        'threshold': 0.08,
        'output': OUTPUT_28D,
        'entry_price_col': 'next_open',
        'exit_price_col': 'exit_28d_close',
    },
]


def get_feature_cols(df):
    exclude_cols = {'ts_code', 'trade_date', 'ds',
                    'open', 'high', 'low', 'close', 'pre_close',
                    'entry_price', 'next_open',
                    'exit_price_1d', 'return_1d',
                    'exit_price_5d', 'return_5d', 'return_5d_open',
                    'exit_price_28d', 'return_28d', 'return_28d_open',
                    'exit_28d_close',
                    'calc_ret5d', 'return_5d_from_open', 'return_28d_from_open',
                    'entry_vs_close'}
    return [c for c in df.columns
            if c not in exclude_cols
            and not c.startswith('hist_')
            and df[c].dtype in ('float64', 'float32', 'int64', 'int32')]


def run():
    print("=" * 90, flush=True)
    print("Walk-Forward 月级别预测 v2 (正确target: 从T+1开盘计算收益)", flush=True)
    print("=" * 90, flush=True)

    print("加载特征数据...", flush=True)
    features_df = pd.read_parquet(FEATURES_FILE)
    features_df['ds'] = features_df['trade_date'].astype(str)
    print(f"数据: {len(features_df)} 行, {len(features_df.columns)} 列", flush=True)
    print(f"日期范围: {features_df['ds'].min()} - {features_df['ds'].max()}", flush=True)

    for target_info in TARGETS:
        rc = target_info['return_col']
        if rc not in features_df.columns:
            print(f"ERROR: 缺少 {rc} 列，请先运行 build_new_targets.py", flush=True)
            return

        n_valid = features_df[rc].notna().sum()
        n_pos = (features_df[rc] > target_info['threshold']).sum()
        print(f"{rc}: {n_valid} 有效值, {n_pos} 正样本 ({n_pos/n_valid:.1%} > {target_info['threshold']:.0%})", flush=True)

    feature_cols = get_feature_cols(features_df)
    print(f"可用特征: {len(feature_cols)}", flush=True)

    months = sorted(features_df['ds'].str[:6].unique())
    pred_months = [m for m in months if m >= '202201']
    print(f"预测月数: {len(pred_months)} ({pred_months[0]} - {pred_months[-1]})", flush=True)

    from xgboost import XGBClassifier

    for target_info in TARGETS:
        print(f"\n{'='*90}", flush=True)
        print(f"Training target: {target_info['name']} ({target_info['return_col']})", flush=True)
        print(f"Positive threshold: {target_info['threshold']:.0%}", flush=True)
        print(f"{'='*90}", flush=True)

        return_col = target_info['return_col']
        all_predictions = []
        total_start = time.time()

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
                print(f"  [{i+1}/{len(pred_months)}] {month}: 训练数据不足 ({len(train_df)}), 跳过", flush=True)
                continue
            if len(pred_df) == 0:
                print(f"  [{i+1}/{len(pred_months)}] {month}: 预测数据为空, 跳过", flush=True)
                continue

            train_df['label'] = (train_df[return_col] > target_info['threshold']).astype(int)
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
            month_pred['target'] = target_info['name']

            if return_col in pred_df.columns:
                month_pred['actual_return'] = pred_df[return_col].values

            if 'next_open' in pred_df.columns:
                month_pred['entry_price'] = pred_df['next_open'].values

            if target_info['exit_price_col'] in pred_df.columns:
                month_pred['exit_price'] = pred_df[target_info['exit_price_col']].values
            elif 'exit_price_5d' in pred_df.columns and target_info['name'] == '5d_open':
                month_pred['exit_price'] = pred_df['exit_price_5d'].values

            all_predictions.append(month_pred)

            elapsed = time.time() - month_start
            total_elapsed = time.time() - total_start
            avg_per_month = total_elapsed / (i + 1)
            remaining = avg_per_month * (len(pred_months) - i - 1)

            n_above_05 = (proba >= 0.50).sum()
            n_above_06 = (proba >= 0.60).sum()
            print(f"  [{i+1}/{len(pred_months)}] {month}: train={len(train_df)}, pred={len(pred_df)}, "
                  f"pos={pos_rate:.1%}, prob>=0.5={n_above_05}, prob>=0.6={n_above_06}, "
                  f"耗时={elapsed:.0f}s, 剩余~{remaining/60:.0f}min", flush=True)

        if not all_predictions:
            print(f"ERROR: {target_info['name']} 未生成任何预测", flush=True)
            continue

        combined = pd.concat(all_predictions, ignore_index=True)
        combined.to_parquet(target_info['output'])
        total_time = (time.time() - total_start) / 60
        print(f"\n{target_info['name']} 预测已保存: {target_info['output']}", flush=True)
        print(f"总预测: {len(combined)} 行, 总耗时: {total_time:.1f} 分钟", flush=True)

        n_with_return = combined['actual_return'].notna().sum()
        n_pos_return = (combined['actual_return'] > 0).sum()
        print(f"有收益数据: {n_with_return}, 正收益: {n_pos_return} ({n_pos_return/n_with_return:.1%})", flush=True)

        prob_bins = [0, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0]
        print(f"\n概率分箱 vs 实际收益率 ({target_info['name']}):", flush=True)
        print(f"  {'概率区间':<12} {'样本数':>8} {'正样本率':>10} {'平均收益':>12}", flush=True)
        valid = combined.dropna(subset=['actual_return'])
        for j in range(len(prob_bins) - 1):
            lo, hi = prob_bins[j], prob_bins[j + 1]
            bin_df = valid[(valid['prob'] >= lo) & (valid['prob'] < hi)]
            if len(bin_df) > 0:
                pos_rate = (bin_df['actual_return'] > target_info['threshold']).mean()
                avg_ret = bin_df['actual_return'].mean()
                print(f"  [{lo:.1f}, {hi:.1f})  {len(bin_df):>8} {pos_rate:>9.1%} {avg_ret:>11.2%}", flush=True)

    print(f"\n{'='*90}", flush=True)
    print("所有模型训练完成!", flush=True)
    print(f"{'='*90}", flush=True)


if __name__ == '__main__':
    run()

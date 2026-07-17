"""
5日持有期 Walk-Forward 月级别预测 (每月重训练)

与1d版本的关键区别:
  - 目标: return_5d (5日收益率)
  - 正样本阈值: 3% (5日持有期合理收益)
  - 输出: predictions_5d_wf_monthly.parquet

严格防止的问题:
  1. 无 np.clip() - 所有收益基于实际价格
  2. 月度WF - 每月独立训练，完全样本外预测
  3. 不使用未来数据 - 训练截止到预测月前一个月

运行命令:
  cd study_004_systematic
  python -u run_5d_wf_monthly.py 2>&1 | tee wf_5d_monthly_log.txt
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
RESULTS_DIR = os.path.join(STUDY_DIR, 'results')
PREDICTIONS_DIR = os.path.join(STUDY_DIR, 'predictions')
os.makedirs(PREDICTIONS_DIR, exist_ok=True)

FEATURES_FILE = os.path.join(DATA_DIR, 'all_features_v2.parquet')
OUTPUT_FILE = os.path.join(PREDICTIONS_DIR, 'predictions_5d_wf_monthly.parquet')

TRAIN_START = '20200101'
TARGET_THRESHOLD = 0.03
MIN_TRAIN_SAMPLES = 50000


def log(msg):
    print(msg, flush=True)


def get_feature_cols(df):
    exclude_cols = {'ts_code', 'trade_date', 'ds',
                    'entry_price',
                    'exit_price_1d', 'return_1d',
                    'exit_price_5d', 'return_5d',
                    'exit_price_28d', 'return_28d'}
    return [c for c in df.columns
            if c not in exclude_cols
            and not c.startswith('hist_')
            and df[c].dtype in ('float64', 'float32', 'int64', 'int32')]


def run():
    log("=" * 90)
    log("5日持有期 Walk-Forward 月级别预测 (每月重训练)")
    log("=" * 90)
    log(f"目标: return_5d > {TARGET_THRESHOLD:.0%}")
    log(f"训练起始: {TRAIN_START}")
    log("")

    if not os.path.exists(FEATURES_FILE):
        log("ERROR: 请先运行 step1_build_features.py 生成特征数据")
        return

    log("加载特征数据...")
    features_df = pd.read_parquet(FEATURES_FILE)
    features_df['ds'] = features_df['trade_date'].astype(str)
    log(f"数据: {len(features_df)} 行, {len(features_df.columns)} 列")
    log(f"日期范围: {features_df['ds'].min()} - {features_df['ds'].max()}")

    return_col = 'return_5d'
    if return_col not in features_df.columns:
        log(f"ERROR: 缺少 {return_col} 列")
        return

    n_valid = features_df[return_col].notna().sum()
    n_pos = (features_df[return_col] > TARGET_THRESHOLD).sum()
    log(f"return_5d: {n_valid} 有效值, {n_pos} 正样本 ({n_pos/n_valid:.1%} > {TARGET_THRESHOLD:.0%})")

    feature_cols = get_feature_cols(features_df)
    log(f"可用特征: {len(feature_cols)}")

    months = sorted(features_df['ds'].str[:6].unique())
    pred_months = [m for m in months if m >= '202201']
    log(f"预测月数: {len(pred_months)} ({pred_months[0]} - {pred_months[-1]})")
    log("")

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
            log(f"  [{i+1}/{len(pred_months)}] {month}: 训练数据不足 ({len(train_df)}), 跳过")
            continue
        if len(pred_df) == 0:
            log(f"  [{i+1}/{len(pred_months)}] {month}: 预测数据为空, 跳过")
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
        month_pred['target'] = '5d'

        if return_col in pred_df.columns:
            month_pred['actual_return'] = pred_df[return_col].values

        if 'entry_price' in pred_df.columns:
            month_pred['entry_price'] = pred_df['entry_price'].values

        if 'exit_price_5d' in pred_df.columns:
            month_pred['exit_price_5d'] = pred_df['exit_price_5d'].values

        all_predictions.append(month_pred)

        elapsed = time.time() - month_start
        total_elapsed = time.time() - total_start
        avg_per_month = total_elapsed / (i + 1)
        remaining = avg_per_month * (len(pred_months) - i - 1)

        n_above_05 = (proba >= 0.50).sum()
        n_above_06 = (proba >= 0.60).sum()
        log(f"  [{i+1}/{len(pred_months)}] {month}: train={len(train_df)}, pred={len(pred_df)}, "
            f"pos={pos_rate:.1%}, prob>=0.5={n_above_05}, prob>=0.6={n_above_06}, "
            f"耗时={elapsed:.0f}s, 剩余~{remaining/60:.0f}min")

    if not all_predictions:
        log("ERROR: 未生成任何预测")
        return

    combined = pd.concat(all_predictions, ignore_index=True)
    combined.to_parquet(OUTPUT_FILE)
    total_time = (time.time() - total_start) / 60
    log(f"\n5d月级别WF预测已保存: {OUTPUT_FILE}")
    log(f"总预测: {len(combined)} 行")
    log(f"总耗时: {total_time:.1f} 分钟")

    n_with_return = combined['actual_return'].notna().sum()
    n_pos_return = (combined['actual_return'] > 0).sum()
    log(f"有收益数据: {n_with_return}, 正收益: {n_pos_return} ({n_pos_return/n_with_return:.1%})")

    prob_bins = [0, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0]
    log(f"\n概率分箱 vs 实际5d收益率:")
    log(f"  {'概率区间':<12} {'样本数':>8} {'正样本率':>10} {'平均5d收益':>12}")
    log(f"  {'-'*46}")
    valid = combined.dropna(subset=['actual_return'])
    for j in range(len(prob_bins) - 1):
        lo, hi = prob_bins[j], prob_bins[j + 1]
        bin_df = valid[(valid['prob'] >= lo) & (valid['prob'] < hi)]
        if len(bin_df) > 0:
            pos_rate = (bin_df['actual_return'] > TARGET_THRESHOLD).mean()
            avg_ret = bin_df['actual_return'].mean()
            log(f"  [{lo:.1f}, {hi:.1f})  {len(bin_df):>8} {pos_rate:>9.1%} {avg_ret:>11.2%}")

    log(f"\n{'='*90}")
    log("5d月度WF预测完成! 下一步运行 backtest_5d_t1.py 进行T+1约束回测")
    log(f"{'='*90}")


if __name__ == '__main__':
    run()

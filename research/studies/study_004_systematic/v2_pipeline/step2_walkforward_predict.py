"""
Step 2: Walk-Forward 训练 + 预测

输入: data/all_features_v2.parquet
输出: predictions/predictions_1d_wf.parquet, models/latest_wf_model.joblib

耗时: 约30分钟
运行频率: 每月1次 或 Step 1 更新后
"""
import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime
import json
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

from config import (
    FEATURES_FILE, WF_PREDICTIONS_FILE, MODELS_DIR,
    TRAIN_START, WALK_FORWARD_YEARS, MIN_TRAIN_SAMPLES,
    TARGET_RETURN_THRESHOLD, RESULTS_DIR
)


def is_main_board(ts_code: str) -> bool:
    return ts_code.startswith(('60', '00', '002', '003'))


def run():
    print("=" * 80)
    print("Step 2: Walk-Forward 训练 + 预测")
    print("=" * 80)

    if not os.path.exists(FEATURES_FILE):
        print("错误: 请先运行 step1_build_features.py")
        return None

    print("加载特征数据...")
    features_df = pd.read_parquet(FEATURES_FILE)
    features_df['ds'] = features_df['trade_date'].astype(str)
    print(f"数据: {len(features_df)} 行, {len(features_df.columns)} 列")

    exclude_cols = ['ts_code', 'trade_date', 'ds', 'entry_price',
                    'exit_price_1d', 'actual_return']
    feature_cols = [c for c in features_df.columns if c not in exclude_cols and
                    not c.startswith('hist_') and
                    features_df[c].dtype in ['float64', 'float32', 'int64', 'int32']]
    print(f"可用特征: {len(feature_cols)}")

    return_col = 'actual_return'
    if return_col not in features_df.columns:
        print("错误: 缺少 actual_return 列")
        return None

    all_predictions = []
    latest_model = None
    latest_feature_cols = None

    for year in WALK_FORWARD_YEARS:
        print(f"\n--- {year}年: 训练(<{year}) -> 预测({year}) ---")

        train_end = f'{year}0101'
        pred_start = f'{year}0101'
        pred_end = f'{year}1231'

        train_mask = (features_df['ds'] >= TRAIN_START) & (features_df['ds'] < train_end)
        pred_mask = (features_df['ds'] >= pred_start) & (features_df['ds'] <= pred_end)

        train_df = features_df[train_mask & features_df[return_col].notna()].copy()
        pred_df = features_df[pred_mask].copy()

        if len(train_df) < MIN_TRAIN_SAMPLES:
            print(f"  训练数据不足: {len(train_df)}")
            continue
        if len(pred_df) == 0:
            print(f"  预测数据为空")
            continue

        train_df['label'] = (train_df[return_col] > TARGET_RETURN_THRESHOLD).astype(int)

        print(f"  训练: {len(train_df)} 行, 正样本: {train_df['label'].mean():.2%}")
        print(f"  预测: {len(pred_df)} 行")

        from xgboost import XGBClassifier

        X_train = train_df[feature_cols].fillna(0)
        y_train = train_df['label']

        model = XGBClassifier(
            n_estimators=100, max_depth=5, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8,
            random_state=42, n_jobs=-1, eval_metric='logloss'
        )
        model.fit(X_train, y_train)

        importance = pd.DataFrame({
            'feature': feature_cols,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        print(f"  Top 5 特征: {importance.head(5)['feature'].tolist()}")

        X_pred = pred_df[feature_cols].fillna(0)
        proba = model.predict_proba(X_pred)[:, 1]

        year_pred = pred_df[['trade_date', 'ts_code']].copy()
        year_pred['prob'] = proba
        year_pred['target'] = '1d'

        if return_col in pred_df.columns:
            year_pred['actual_return'] = pred_df[return_col].values

        all_predictions.append(year_pred)
        print(f"  {year}年预测完成: {len(year_pred)} 行, prob均值={proba.mean():.4f}")

        latest_model = model
        latest_feature_cols = feature_cols

    if not all_predictions:
        print("错误: 未生成任何预测")
        return None

    combined = pd.concat(all_predictions, ignore_index=True)
    combined.to_parquet(WF_PREDICTIONS_FILE)
    print(f"\nWalk-Forward预测已保存: {WF_PREDICTIONS_FILE}")
    print(f"总预测: {len(combined)} 行")

    if latest_model is not None:
        import joblib
        joblib.dump(latest_model, os.path.join(MODELS_DIR, 'latest_wf_model.joblib'))
        joblib.dump(latest_feature_cols, os.path.join(MODELS_DIR, 'latest_wf_features.joblib'))
        print(f"最新模型已保存: {MODELS_DIR}/latest_wf_model.joblib")

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    summary = {
        'step': 'step2_walkforward',
        'timestamp': timestamp,
        'n_features': len(feature_cols),
        'walk_forward_years': WALK_FORWARD_YEARS,
        'total_predictions': len(combined),
        'date_range': f"{combined['trade_date'].min()} - {combined['trade_date'].max()}",
    }
    summary_path = os.path.join(RESULTS_DIR, f'step2_summary_{timestamp}.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    return combined


if __name__ == '__main__':
    run()

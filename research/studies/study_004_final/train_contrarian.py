"""
Step 2: Walk-Forward 训练 + 预测（28日目标）

输入: data/contrarian_features.parquet
输出: predictions/contrarian_predictions.parquet, models/contrarian_model.joblib

耗时: 约30-60分钟
"""
import os
import sys
import pandas as pd
import numpy as np
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (
    FEATURES_FILE, PREDICTIONS_FILE, MODEL_FILE, FEATS_FILE, MODELS_DIR, RESULTS_DIR,
    WALK_FORWARD_YEARS, MIN_TRAIN_SAMPLES, TARGET_RETURN_THRESHOLD
)


def get_feature_columns(df):
    """获取特征列（排除非特征列）"""
    exclude = ['ts_code', 'trade_date', 'open', 'high', 'low', 'close', 'pre_close',
                'change', 'pct_chg', 'vol', 'amount', 'entry_price', 'exit_price_1d',
                'exit_price_5d', 'exit_price_28d', 'actual_return', 'return_1d',
                'return_5d', 'return_28d', 'return_1d_open', 'return_5d_open',
                'return_28d_open', 'return_1d_open_old', 'next_open', 'exit_28d_close',
                'target_return', 'label', 'ds']
    feature_cols = []
    for c in df.columns:
        if c in exclude:
            continue
        if df[c].dtype not in ['float64', 'float32', 'int64', 'int32']:
            continue
        if df[c].isna().all():
            continue
        feature_cols.append(c)
    return feature_cols


def run():
    print("=" * 80)
    print("Step 2: Walk-Forward 训练 + 预测（28日目标）")
    print("=" * 80)

    if not os.path.exists(FEATURES_FILE):
        print("错误: 请先运行 build_contrarian_features.py")
        return

    print("加载特征数据...")
    df = pd.read_parquet(FEATURES_FILE)
    df['ds'] = df['trade_date'].astype(str)
    print(f"数据: {len(df)} 行, {len(df.columns)} 列")

    feature_cols = get_feature_columns(df)
    print(f"可用特征: {len(feature_cols)} 列")
    print(f"Top 10: {feature_cols[:10]}")

    return_col = 'target_return'
    if return_col not in df.columns:
        print("错误: 缺少目标收益列")
        return

    all_predictions = []
    latest_model = None
    latest_feature_cols = None

    for year in WALK_FORWARD_YEARS:
        print(f"\n--- {year}年: 训练(<{year}) -> 预测({year}) ---")

        train_end = f'{year}0101'
        pred_start = f'{year}0101'
        pred_end = f'{year}1231'

        train_mask = (df['ds'] < train_end) & (df[return_col].notna())
        pred_mask = (df['ds'] >= pred_start) & (df['ds'] <= pred_end)

        train_df = df[train_mask].copy()
        pred_df = df[pred_mask].copy()

        if len(train_df) < MIN_TRAIN_SAMPLES:
            print(f"  训练数据不足: {len(train_df)}")
            continue
        if len(pred_df) == 0:
            print(f"  预测数据为空")
            continue

        print(f"  训练: {len(train_df)} 行, 正样本: {train_df['label'].mean():.2%}")
        print(f"  预测: {len(pred_df)} 行")

        from xgboost import XGBClassifier
        import joblib

        X_train = train_df[feature_cols].fillna(0)
        y_train = train_df['label']

        model = XGBClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            random_state=42, n_jobs=-1, eval_metric='logloss'
        )
        model.fit(X_train, y_train)

        importance = pd.DataFrame({
            'feature': feature_cols,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        print(f"  Top 5 特征: {importance.head(5)['feature'].tolist()}")
        print(f"  Top 5 重要性: {importance.head(5)['importance'].tolist()}")

        X_pred = pred_df[feature_cols].fillna(0)
        proba = model.predict_proba(X_pred)[:, 1]

        year_pred = pred_df[['trade_date', 'ts_code']].copy()
        year_pred['prob'] = proba
        year_pred['target'] = '28d'

        if return_col in pred_df.columns:
            year_pred['actual_return'] = pred_df[return_col].values
        if 'label' in pred_df.columns:
            year_pred['label'] = pred_df['label'].values

        all_predictions.append(year_pred)
        print(f"  {year}年预测完成: {len(year_pred)} 行, prob均值={proba.mean():.4f}")

        latest_model = model
        latest_feature_cols = feature_cols

    if not all_predictions:
        print("错误: 未生成任何预测")
        return

    combined = pd.concat(all_predictions, ignore_index=True)
    combined.to_parquet(PREDICTIONS_FILE)
    print(f"\nWalk-Forward预测已保存: {PREDICTIONS_FILE}")
    print(f"总预测: {len(combined)} 行")
    print(f"日期范围: {combined['trade_date'].min()} - {combined['trade_date'].max()}")

    if latest_model is not None:
        import joblib
        joblib.dump(latest_model, MODEL_FILE)
        joblib.dump(latest_feature_cols, FEATS_FILE)
        print(f"模型已保存: {MODEL_FILE}")
        print(f"特征列表已保存: {FEATS_FILE}")

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    summary = {
        'step': 'train_contrarian',
        'timestamp': timestamp,
        'n_features': len(feature_cols),
        'feature_names': feature_cols,
        'walk_forward_years': WALK_FORWARD_YEARS,
        'total_predictions': len(combined),
        'date_range': f"{combined['trade_date'].min()} - {combined['trade_date'].max()}",
    }
    summary_path = os.path.join(RESULTS_DIR, f'train_summary_{timestamp}.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    return combined


if __name__ == '__main__':
    run()

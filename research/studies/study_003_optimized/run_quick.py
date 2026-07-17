"""
Study 003: 快速验证版本
"""
import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from shared.data_loader import get_all_dates, PRICE_DIR
from config import *

# 加载已保存的特征数据
features_path = os.path.join(DATA_DIR, f'features_h{PREDICT_HORIZON}.parquet')
print(f"加载特征数据: {features_path}")
features_df = pd.read_parquet(features_path)
print(f"特征数据: {len(features_df)} 行")

# 选择特征列
exclude_cols = ['ts_code', 'trade_date', 't1_open', f't{PREDICT_HORIZON}_close', 
                'label_ret', 'label', 'hist_vol_mean']
feature_cols = [c for c in features_df.columns if c not in exclude_cols]
print(f"特征数量: {len(feature_cols)}")

# 分割数据
train_mask = features_df['trade_date'].astype(str).str[:4].isin(['2022', '2023'])
test_mask = features_df['trade_date'].astype(str).str[:4] == '2024'

train_df = features_df[train_mask]
test_df = features_df[test_mask]

print(f"训练数据: {len(train_df)} 行")
print(f"测试数据: {len(test_df)} 行")

# 准备标签
train_df = train_df.copy()
train_df['label'] = (train_df['label_ret'] > LABEL_THRESHOLD).astype(int)
print(f"正样本比例: {train_df['label'].mean():.2%}")

# 训练模型（简化版）
print("\n训练模型...")
from xgboost import XGBClassifier

# 采样加速
if len(train_df) > 500000:
    train_sample = train_df.sample(n=500000, random_state=42)
    print(f"使用采样数据: {len(train_sample)} 行")
else:
    train_sample = train_df

X_train = train_sample[feature_cols].fillna(0)
y_train = train_sample['label']

model = XGBClassifier(
    n_estimators=50,  # 减少树的数量
    max_depth=4,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1,
    eval_metric='logloss'
)

model.fit(X_train, y_train)
print("模型训练完成")

# 特征重要性
importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print("\nTop 10 特征:")
print(importance.head(10))

# 预测
print("\n生成预测...")
X_test = test_df[feature_cols].fillna(0)
test_df = test_df.copy()
test_df['prob'] = model.predict_proba(X_test)[:, 1]

# 保存预测
pred_df = test_df[['trade_date', 'ts_code', 'prob', 'label_ret']]
pred_path = os.path.join(DATA_DIR, f'predictions_h{PREDICT_HORIZON}.parquet')
pred_df.to_parquet(pred_path)
print(f"预测数据已保存: {pred_path}")

print("\n预测统计:")
print(f"  平均概率: {test_df['prob'].mean():.4f}")
print(f"  最大概率: {test_df['prob'].max():.4f}")
print(f"  最小概率: {test_df['prob'].min():.4f}")

print("\n✅ 快速验证完成！")

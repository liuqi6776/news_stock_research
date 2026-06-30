#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Study B — DOS + 情感强度作为新特征加入方向模型
====================================================
核心假设: 把 DOS、总情感强度、预测跳空作为特征加入方向模型,
让模型学到交互(如"高DOS+资金流入=强;高DOS+资金流出=危险"),
提升方向预测 AUC/胜率。

因环境中无 sklearn/xgboost, 本脚本用 numpy 手动实现:
  1. 逐日截面标准化
  2. 逻辑回归(sigmoid + 梯度下降)
  3. walk-forward: 用过去 N 天训练, 预测明天

对比:
  基线模型: 价格形态 + 动量 + 资金流 + 筹码
  增强模型: 基线 + DOS + total_sentiment + pred_gap + 交互项
"""

import pandas as pd
import numpy as np
import os
import json

SAVE_DIR = 'C:/Users/liuqi/quant_system_v2'

# ========== 1. 加载数据 ==========
print("Loading data...")
df = pd.read_csv(f'{SAVE_DIR}/study_a_features_v3.csv')
df['trade_date'] = df['trade_date'].astype(int)
df = df.sort_values(['ts_code', 'trade_date']).reset_index(drop=True)

print(f"Total: {len(df)} rows, {df['trade_date'].nunique()} dates, {df['ts_code'].nunique()} stocks")

# ========== 2. 构建新特征 ==========
print("Building features...")

# total_sentiment: 情感强度 (positive + |negative|)
df['total_sentiment'] = df['positive'] + df['negative'].abs()

# pred_gap: 基于DOS和历史波动的跳空预测 (简单线性组合)
# 研究A已证明: DOS对|跳空|有显著预测力 (NW-t=4.66)
df['pred_gap'] = (
    0.6 * df['dos'] +
    0.3 * df['vol_5d'].fillna(0) +
    0.1 * df['vol_20d'].fillna(0)
)

# 交互项: DOS × 资金流 (高DOS + 资金流入 = 强; 高DOS + 资金流出 = 危险)
df['dos_x_mf'] = df['dos'] * df['mf_ratio_5d'].fillna(0)
df['dos_x_mom'] = df['dos'] * df['mom_5d'].fillna(0)

# 目标: 次日是否上涨
df['target'] = (df['next_return'] > 0).astype(int)

# ========== 3. 定义特征集 ==========
BASELINE_FEATURES = [
    'returns_5d', 'returns_10d', 'returns_20d',
    'mom_5d', 'mom_10d', 'mom_20d',
    'price_vs_ma5', 'price_vs_ma20',
    'vol_5d', 'vol_ratio',
    'mf_ratio', 'mf_ratio_5d',
    'winner_rate', 'profit_pressure',
]

ENHANCED_FEATURES = BASELINE_FEATURES + [
    'dos', 'total_sentiment', 'pred_gap',
    'dos_x_mf', 'dos_x_mom',
]

# 确保所有特征存在
BASELINE_FEATURES = [f for f in BASELINE_FEATURES if f in df.columns]
ENHANCED_FEATURES = [f for f in ENHANCED_FEATURES if f in df.columns]

print(f"Baseline features: {len(BASELINE_FEATURES)}")
print(f"Enhanced features: {len(ENHANCED_FEATURES)} (+{len(ENHANCED_FEATURES)-len(BASELINE_FEATURES)} new)")

# ========== 4. 手动实现逻辑回归 ==========
class SimpleLogisticRegressor:
    """用numpy手动实现的逻辑回归,支持L2正则化。"""
    def __init__(self, lr=0.01, n_iter=100, l2=0.01):
        self.lr = lr
        self.n_iter = n_iter
        self.l2 = l2
        self.w = None
        self.b = None
        
    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))
    
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.w = np.zeros(n_features)
        self.b = 0
        
        for _ in range(self.n_iter):
            z = X @ self.w + self.b
            pred = self._sigmoid(z)
            
            # 梯度
            dw = (X.T @ (pred - y)) / n_samples + self.l2 * self.w
            db = np.mean(pred - y)
            
            self.w -= self.lr * dw
            self.b -= self.lr * db
            
    def predict_proba(self, X):
        z = X @ self.w + self.b
        return self._sigmoid(z)
    
    def predict(self, X):
        return (self.predict_proba(X) > 0.5).astype(int)


# ========== 5. Walk-forward 回测 ==========
def walk_forward_backtest(df, features, label='baseline'):
    """Walk-forward: 用过去60天训练,预测次日。"""
    
    # 只保留有完整特征的行
    valid = df[features + ['target', 'trade_date', 'ts_code', 'next_return', 'next_open_pct']].dropna()
    
    dates = sorted(valid['trade_date'].unique())
    train_window = 60  # 用过去60天训练
    
    predictions = []
    
    for i in range(train_window, len(dates)):
        train_dates = dates[i-train_window:i]
        test_date = dates[i]
        
        train = valid[valid['trade_date'].isin(train_dates)]
        test = valid[valid['trade_date'] == test_date]
        
        if len(train) < 100 or len(test) < 10:
            continue
        
        # 标准化 (用训练集统计量)
        X_train = train[features].values
        y_train = train['target'].values
        X_test = test[features].values
        
        mean = X_train.mean(axis=0)
        std = X_train.std(axis=0) + 1e-8
        X_train_norm = (X_train - mean) / std
        X_test_norm = (X_test - mean) / std
        
        # 训练
        model = SimpleLogisticRegressor(lr=0.05, n_iter=200, l2=0.001)
        model.fit(X_train_norm, y_train)
        
        # 预测
        proba = model.predict_proba(X_test_norm)
        
        for idx, (_, row) in enumerate(test.iterrows()):
            predictions.append({
                'trade_date': test_date,
                'ts_code': row['ts_code'],
                'proba': proba[idx],
                'target': row['target'],
                'next_return': row['next_return'],
                'next_open_pct': row['next_open_pct'],
                'model': label,
            })
    
    return pd.DataFrame(predictions)


print("\n" + "="*70)
print("Running walk-forward backtest...")
print("="*70)

# 基线模型
print("\n[1/2] Baseline model...")
pred_base = walk_forward_backtest(df, BASELINE_FEATURES, 'baseline')

# 增强模型
print("[2/2] Enhanced model (with DOS features)...")
pred_enh = walk_forward_backtest(df, ENHANCED_FEATURES, 'enhanced')

# 合并
predictions = pd.concat([pred_base, pred_enh], ignore_index=True)

# ========== 6. 评估 ==========
print("\n" + "="*70)
print("EVALUATION")
print("="*70)

def evaluate(pred_df, label):
    """评估模型表现。"""
    # AUC (手动计算)
    pred_df = pred_df.sort_values('proba', ascending=False).reset_index(drop=True)
    n_pos = pred_df['target'].sum()
    n_neg = len(pred_df) - n_pos
    
    # 简单AUC近似
    pos_ranks = pred_df[pred_df['target']==1].index
    auc = (pos_ranks.sum() - n_pos*(n_pos-1)/2) / (n_pos * n_neg) if n_pos > 0 and n_neg > 0 else 0.5
    auc = 1 - auc  # 修正方向
    
    # 胜率 (Top-20 预测)
    daily_win = []
    daily_ret = []
    daily_gap = []
    
    for date in sorted(pred_df['trade_date'].unique()):
        day = pred_df[pred_df['trade_date'] == date]
        if len(day) < 20:
            continue
        day = day.sort_values('proba', ascending=False)
        top20 = day.head(20)
        daily_win.append((top20['target'] > 0).mean())
        daily_ret.append(top20['next_return'].mean())
        daily_gap.append(top20['next_open_pct'].mean())
    
    win_rate = np.mean(daily_win)
    avg_ret = np.mean(daily_ret)
    avg_gap = np.mean(daily_gap)
    sharpe = avg_ret / np.std(daily_ret) if np.std(daily_ret) > 0 else 0
    
    return {
        'model': label,
        'auc': auc,
        'win_rate': win_rate,
        'avg_ret': avg_ret,
        'avg_gap': avg_gap,
        'sharpe': sharpe,
        'n_days': len(daily_win),
    }

base_eval = evaluate(pred_base, 'baseline')
enh_eval = evaluate(pred_enh, 'enhanced')

for ev in [base_eval, enh_eval]:
    print(f"\n{ev['model'].upper()}:")
    print(f"  AUC:       {ev['auc']:.4f}")
    print(f"  Win Rate:  {ev['win_rate']:.2%}")
    print(f"  Avg Ret:   {ev['avg_ret']:.4f}")
    print(f"  Avg Gap:   {ev['avg_gap']:.4f}")
    print(f"  Sharpe:    {ev['sharpe']:.3f}")
    print(f"  Days:      {ev['n_days']}")

# 改善
print("\n" + "="*70)
print("IMPROVEMENT (Enhanced vs Baseline)")
print("="*70)
print(f"  AUC:      {enh_eval['auc'] - base_eval['auc']:+.4f}")
print(f"  Win Rate: {enh_eval['win_rate'] - base_eval['win_rate']:+.2%}")
print(f"  Avg Ret:  {enh_eval['avg_ret'] - base_eval['avg_ret']:+.4f}")
print(f"  Sharpe:   {enh_eval['sharpe'] - base_eval['sharpe']:+.3f}")

# ========== 7. 特征重要性 (系数绝对值) ==========
print("\n" + "="*70)
print("FEATURE IMPORTANCE (Enhanced Model Coefficients)")
print("="*70)

# 重新训练最后一个模型来获取系数
valid = df[ENHANCED_FEATURES + ['target', 'trade_date']].dropna()
train_dates = sorted(valid['trade_date'].unique())[-60:]
train = valid[valid['trade_date'].isin(train_dates)]
X_train = train[ENHANCED_FEATURES].values
y_train = train['target'].values
mean = X_train.mean(axis=0)
std = X_train.std(axis=0) + 1e-8
X_train_norm = (X_train - mean) / std

model = SimpleLogisticRegressor(lr=0.05, n_iter=200, l2=0.001)
model.fit(X_train_norm, y_train)

importance = pd.DataFrame({
    'feature': ENHANCED_FEATURES,
    'coef': model.w,
    'abs_coef': np.abs(model.w),
}).sort_values('abs_coef', ascending=False)

print(importance.to_string(index=False))

# 标记新特征
new_features = set(ENHANCED_FEATURES) - set(BASELINE_FEATURES)
print(f"\nNew features (DOS-related): {', '.join(new_features)}")
new_importance = importance[importance['feature'].isin(new_features)]
print(f"New features rank/importance:")
print(new_importance.to_string(index=False))

# ========== 8. 保存结果 ==========
results = {
    'baseline': base_eval,
    'enhanced': enh_eval,
    'improvement': {
        'auc': enh_eval['auc'] - base_eval['auc'],
        'win_rate': enh_eval['win_rate'] - base_eval['win_rate'],
        'avg_ret': enh_eval['avg_ret'] - base_eval['avg_ret'],
        'sharpe': enh_eval['sharpe'] - base_eval['sharpe'],
    },
    'feature_importance': importance.to_dict('records'),
    'new_features': list(new_features),
}

with open(f'{SAVE_DIR}/study_b_results.json', 'w', encoding='utf-8') as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

predictions.to_csv(f'{SAVE_DIR}/study_b_predictions.csv', index=False, encoding='utf-8-sig')

print(f"\n\nSaved results to study_b_results.json")
print(f"Saved predictions to study_b_predictions.csv")

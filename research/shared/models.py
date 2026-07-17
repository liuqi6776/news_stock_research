"""
模型模块 - 定义和训练模型
"""
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from typing import List, Tuple


def train_xgboost(X_train: pd.DataFrame, y_train: pd.Series,
                  feature_cols: List[str],
                  n_estimators: int = 100,
                  max_depth: int = 5,
                  learning_rate: float = 0.1) -> XGBClassifier:
    """
    训练XGBoost模型
    
    Args:
        X_train: 训练特征
        y_train: 训练标签
        feature_cols: 特征列名
        n_estimators: 树的数量
        max_depth: 最大深度
        learning_rate: 学习率
    
    Returns:
        训练好的模型
    """
    model = XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        eval_metric='logloss'
    )
    
    model.fit(X_train[feature_cols], y_train)
    return model


def predict_proba(model: XGBClassifier, 
                  X: pd.DataFrame, 
                  feature_cols: List[str]) -> np.ndarray:
    """
    预测概率
    
    Args:
        model: 训练好的模型
        X: 特征数据
        feature_cols: 特征列名
    
    Returns:
        预测概率
    """
    return model.predict_proba(X[feature_cols])[:, 1]


def get_feature_importance(model: XGBClassifier, 
                          feature_cols: List[str]) -> pd.DataFrame:
    """
    获取特征重要性
    
    Args:
        model: 训练好的模型
        feature_cols: 特征列名
    
    Returns:
        特征重要性DataFrame
    """
    importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    return importance

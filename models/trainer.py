"""
模型训练模块

提供 XGBoost 模型训练和保存功能
"""

import os
import re
from typing import List, Tuple, Optional, Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib

try:
    import xgboost as xgb
except ImportError:
    xgb = None


class ModelTrainer:
    """
    模型训练器
    
    基于 XGBoost 的股票预测模型训练
    
    Example:
        trainer = ModelTrainer()
        trainer.train(df, features)
        trainer.save('model_v1')
    """
    
    def __init__(self, model_params: dict = None):
        """
        初始化训练器
        
        Args:
            model_params: XGBoost 模型参数
        """
        self.model_params = model_params or {
            'n_estimators': 100,
            'max_depth': 6,
            'learning_rate': 0.1,
            'random_state': 42
        }
        
        self.model = None
        self.scaler = None
        self.features: List[str] = []
    
    def prepare_features(self, df: pd.DataFrame) -> List[str]:
        """
        自动选择特征列
        
        排除非特征列和中文列名
        
        Args:
            df: 数据 DataFrame
        
        Returns:
            特征列名列表
        """
        # 排除的列
        exclude_cols = [
            'ts_code', 'trade_date', 'ts_name', 'concept', 
            'ann_date', 'f_ann_date', 'Future_Change', 'Y',
            'growth_rate', 'quarter_end_month', 'quarter',
            'year', 'future_date', 'future_close', 'growth_percentage',
            'tb_label', 'return_label', 'pct_label', 'data_quality'
        ]
        
        # 过滤中文列名
        pattern = re.compile(r'[\u4e00-\u9fff]+')
        
        features = []
        for col in df.columns:
            if col in exclude_cols:
                continue
            if pattern.search(col):
                continue
            if df[col].dtype in ['object', 'datetime64[ns]']:
                continue
            features.append(col)
        
        return features
    
    def train(
        self,
        df: pd.DataFrame,
        features: List[str] = None,
        target_col: str = 'tb_label',
        train_end_date: str = None
    ) -> Tuple[Any, Any]:
        """
        训练模型
        
        Args:
            df: 训练数据
            features: 特征列表，None 则自动选择
            target_col: 目标列
            train_end_date: 训练数据截止日期
        
        Returns:
            (scaler, model) 元组
        """
        if xgb is None:
            raise ImportError("请安装 xgboost: pip install xgboost")
        
        # 自动选择特征
        if features is None:
            features = self.prepare_features(df)
        
        self.features = features
        
        # 划分训练集
        if train_end_date:
            train_df = df[df['trade_date'] <= train_end_date].copy()
        else:
            train_df = df.copy()
        
        # 准备数据
        X = train_df[features].copy()
        y = train_df[target_col].copy()
        
        # 处理缺失值
        X = X.fillna(0)
        
        # 转换数据类型
        for col in X.columns:
            X[col] = pd.to_numeric(X[col], errors='coerce').fillna(0)
        
        # 标准化
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        # 训练模型
        print(f"开始训练模型，特征数: {len(features)}, 样本数: {len(X)}")
        
        self.model = xgb.XGBClassifier(**self.model_params)
        self.model.fit(X_scaled, y)
        
        print("模型训练完成")
        
        return self.scaler, self.model
    
    def save(self, name: str = 'model', save_dir: str = '.'):
        """
        保存模型、标准化器和特征列表
        
        Args:
            name: 模型名称前缀
            save_dir: 保存目录
        """
        if self.model is None:
            raise ValueError("模型尚未训练")
        
        os.makedirs(save_dir, exist_ok=True)
        
        joblib.dump(self.model, os.path.join(save_dir, f'{name}_model.joblib'))
        joblib.dump(self.scaler, os.path.join(save_dir, f'{name}_scaler.joblib'))
        joblib.dump(self.features, os.path.join(save_dir, f'{name}_features.joblib'))
        
        print(f"模型已保存到 {save_dir}")
    
    def load(self, name: str = 'model', load_dir: str = '.'):
        """
        加载模型
        
        Args:
            name: 模型名称前缀
            load_dir: 加载目录
        """
        self.model = joblib.load(os.path.join(load_dir, f'{name}_model.joblib'))
        self.scaler = joblib.load(os.path.join(load_dir, f'{name}_scaler.joblib'))
        self.features = joblib.load(os.path.join(load_dir, f'{name}_features.joblib'))
        
        print(f"模型已加载，特征数: {len(self.features)}")
    
    def get_feature_importance(self, top_n: int = 20) -> pd.DataFrame:
        """
        获取特征重要性
        
        Args:
            top_n: 返回前 N 个重要特征
        
        Returns:
            特征重要性 DataFrame
        """
        if self.model is None:
            raise ValueError("模型尚未训练")
        
        importance = pd.DataFrame({
            'feature': self.features,
            'importance': self.model.feature_importances_
        })
        
        return importance.sort_values('importance', ascending=False).head(top_n)

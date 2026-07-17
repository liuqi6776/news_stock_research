"""
模型预测模块

提供模型加载和预测功能
"""

import os
from typing import List, Optional

import numpy as np
import pandas as pd
import joblib


class Predictor:
    """
    预测器
    
    加载训练好的模型进行预测
    
    Example:
        predictor = Predictor()
        predictor.load('model_v1')
        result = predictor.predict(df)
    """
    
    def __init__(self):
        """初始化预测器"""
        self.model = None
        self.scaler = None
        self.features: List[str] = []
    
    def load(self, name: str = 'model', load_dir: str = '.'):
        """
        加载模型
        
        Args:
            name: 模型名称前缀
            load_dir: 加载目录
        """
        model_path = os.path.join(load_dir, f'{name}_model.joblib')
        scaler_path = os.path.join(load_dir, f'{name}_scaler.joblib')
        features_path = os.path.join(load_dir, f'{name}_features.joblib')
        
        # 兼容旧格式
        if not os.path.exists(model_path):
            model_path = os.path.join(load_dir, 'xgboost_model.joblib')
            scaler_path = os.path.join(load_dir, 'scaler.joblib')
            features_path = os.path.join(load_dir, 'feas.joblib')
        
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
        self.features = joblib.load(features_path)
        
        print(f"模型已加载，特征数: {len(self.features)}")
    
    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        进行预测
        
        Args:
            df: 预测数据，需包含特征列
        
        Returns:
            添加预测结果列的 DataFrame
        """
        if self.model is None:
            raise ValueError("模型尚未加载")
        
        result_df = df.copy()
        
        # 准备特征
        X = result_df[self.features].copy()
        X = X.fillna(0)
        
        # 转换数据类型
        for col in X.columns:
            X[col] = pd.to_numeric(X[col], errors='coerce').fillna(0)
        
        # 标准化
        X_scaled = self.scaler.transform(X)
        
        # 预测
        result_df['y_pred'] = self.model.predict(X_scaled)
        
        # 预测概率（如果模型支持）
        if hasattr(self.model, 'predict_proba'):
            proba = self.model.predict_proba(X_scaled)
            if proba.shape[1] > 1:
                result_df['y_proba'] = proba[:, 1]  # 正类概率
        
        return result_df
    
    def get_top_stocks(
        self,
        df: pd.DataFrame,
        date: str = None,
        top_n: int = 10,
        min_proba: float = 0.5
    ) -> pd.DataFrame:
        """
        获取推荐的股票
        
        Args:
            df: 预测后的数据
            date: 指定日期，None 则使用最新日期
            top_n: 返回前 N 只股票
            min_proba: 最小概率阈值
        
        Returns:
            推荐股票 DataFrame
        """
        if date is None:
            date = df['trade_date'].max()
        
        day_df = df[df['trade_date'] == date].copy()
        
        # 按预测概率排序
        if 'y_proba' in day_df.columns:
            day_df = day_df[day_df['y_proba'] >= min_proba]
            day_df = day_df.sort_values('y_proba', ascending=False)
        else:
            day_df = day_df[day_df['y_pred'] == 1]
        
        # 返回结果
        result_cols = ['ts_code', 'trade_date', 'close']
        if 'y_pred' in day_df.columns:
            result_cols.append('y_pred')
        if 'y_proba' in day_df.columns:
            result_cols.append('y_proba')
        
        return day_df[result_cols].head(top_n)

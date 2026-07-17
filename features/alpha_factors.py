"""
Alpha 101 因子计算模块

实现 WorldQuant Alpha 101 核心因子
"""

import numpy as np
import pandas as pd
from typing import List, Optional


def calculate_alpha101_factors(df: pd.DataFrame, group_col: str = 'ts_code') -> pd.DataFrame:
    """
    计算 WorldQuant Alpha 101 核心因子
    
    选取适合 A股的反转、波动率和流动性因子
    
    Args:
        df: 包含 OHLCV 数据的 DataFrame
        group_col: 分组列名（股票代码）
    
    Returns:
        添加了 Alpha 因子列的 DataFrame
    
    因子说明:
        - alpha_006: 开盘价与成交量相关性（日内走势预测）
        - alpha_009: 简化版动量/反转因子
        - alpha_012: 量价反转
        - alpha_023: 均值回归检测
        - volatility_20: 20日波动率（风控用）
    """
    print("正在计算 Alpha 101 因子...")
    
    df = df.copy()
    
    # 确保数据按日期排序并重置索引
    df = df.sort_values([group_col, 'trade_date']).reset_index(drop=True)
    
    # Alpha 6: (-1 * correlation(open, volume, 10))
    def calc_alpha_006(group):
        corr = group['open'].rolling(window=10).corr(group['vol'])
        return corr * -1
    
    df['alpha_006'] = df.groupby(group_col).apply(calc_alpha_006).reset_index(level=0, drop=True)
    
    # Alpha 9: 简化版动量/反转因子
    def calc_alpha_009(group):
        delta_close = group['close'].diff(1)
        cond1 = delta_close.rolling(5).min() > 0
        cond2 = delta_close.rolling(5).max() < 0
        return np.where(cond1, delta_close, np.where(cond2, delta_close, -1 * delta_close))
    
    df['alpha_009'] = df.groupby(group_col).apply(calc_alpha_009).reset_index(level=0, drop=True)
    
    # Alpha 12: sign(delta(volume, 1)) * (-1 * delta(close, 1))
    def calc_alpha_012(group):
        return np.sign(group['vol'].diff(1)) * (-1 * group['close'].diff(1))
    
    df['alpha_012'] = df.groupby(group_col).apply(calc_alpha_012).reset_index(level=0, drop=True)
    
    # Alpha 23: ((sum(high, 20) / 20) < high) ? (-1 * delta(high, 2)) : 0
    def calc_alpha_023(group):
        mean_high = group['high'].rolling(20).mean()
        return np.where(mean_high < group['high'], -1 * group['high'].diff(2), 0)
    
    df['alpha_023'] = df.groupby(group_col).apply(calc_alpha_023).reset_index(level=0, drop=True)
    
    # 增加波动率特征 (用于风控)
    def calc_volatility(group):
        return group['close'].pct_change().rolling(20).std()
    
    df['volatility_20'] = df.groupby(group_col).apply(calc_volatility).reset_index(level=0, drop=True)
    
    # 填充因子计算产生的 NaN
    fill_cols = ['alpha_006', 'alpha_009', 'alpha_012', 'alpha_023', 'volatility_20']
    df[fill_cols] = df[fill_cols].fillna(0)
    
    print("Alpha 因子计算完成")
    return df


def calculate_momentum_factors(df: pd.DataFrame, windows: List[int] = [5, 10, 20, 60]) -> pd.DataFrame:
    """
    计算动量因子
    
    Args:
        df: 包含 close 列的 DataFrame
        windows: 计算窗口列表
    
    Returns:
        添加动量因子的 DataFrame
    """
    df = df.copy()
    grouped = df.groupby('ts_code')
    
    for w in windows:
        df[f'momentum_{w}'] = grouped['close'].pct_change(w)
    
    return df


def calculate_volatility_factors(df: pd.DataFrame, windows: List[int] = [5, 10, 20]) -> pd.DataFrame:
    """
    计算波动率因子
    
    Args:
        df: 包含收益率数据的 DataFrame
        windows: 计算窗口列表
    
    Returns:
        添加波动率因子的 DataFrame
    """
    df = df.copy()
    grouped = df.groupby('ts_code')
    
    returns = grouped['close'].pct_change()
    
    for w in windows:
        df[f'volatility_{w}'] = returns.rolling(w).std()
    
    return df

"""
数据清洗模块

提供 ST股过滤、股票代码筛选、标准化等功能
"""

import warnings
from typing import List, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm


def remove_st_stocks(df: pd.DataFrame, stock_basic_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """
    剔除 ST 股票
    
    避免引入不可控的波动风险
    
    Args:
        df: 股票数据 DataFrame
        stock_basic_df: 股票基础信息表（需包含 ts_code, name）
    
    Returns:
        过滤后的 DataFrame
    """
    print("正在执行 ST 股票过滤...")
    
    if 'name' not in df.columns and stock_basic_df is None:
        warnings.warn("无法过滤 ST 股票：DataFrame 中缺少 'name' 列且未提供基础信息表")
        return df
    
    target_df = df.copy()
    
    if 'name' not in df.columns:
        # 合并股票名称
        target_df = pd.merge(df, stock_basic_df[['ts_code', 'name']], on='ts_code', how='left')
    
    # 过滤包含 ST, *ST 的股票
    condition = ~target_df['name'].str.contains('ST', na=False)
    filtered_df = target_df[condition]
    
    if 'name' not in df.columns:
        filtered_df = filtered_df.drop(columns=['name'])
    
    removed_count = len(target_df['ts_code'].unique()) - len(filtered_df['ts_code'].unique())
    print(f"ST 过滤完成，移除股票数量: {removed_count}")
    
    return filtered_df


def filter_stock_codes(df: pd.DataFrame, patterns: List[str] = ['^60', '^00']) -> pd.DataFrame:
    """
    根据股票代码前缀筛选
    
    Args:
        df: 股票数据 DataFrame
        patterns: 保留的股票代码前缀模式列表
    
    Returns:
        过滤后的 DataFrame
    """
    if 'ts_code' not in df.columns:
        warnings.warn("DataFrame 中缺少 'ts_code' 列，无法筛选股票代码")
        return df
    
    # 构建正则模式
    pattern = '|'.join([f'({p})' for p in patterns])
    
    return df[df['ts_code'].str.match(pattern)]


def apply_expanding_standardization(
    df: pd.DataFrame,
    cols_to_standardize: List[str],
    group_col: str = 'ts_code',
    min_periods: int = 60,
    suffix: str = '_norm'
) -> pd.DataFrame:
    """
    使用扩展窗口进行标准化，消除前视偏差
    
    z_t = (x_t - mean_{0:t-1}) / std_{0:t-1}
    
    Args:
        df: 数据 DataFrame
        cols_to_standardize: 需要标准化的列名列表
        group_col: 分组列名
        min_periods: 最小计算周期
        suffix: 标准化列名后缀
    
    Returns:
        添加标准化列的 DataFrame
    """
    print("正在应用扩展窗口标准化(消除前视偏差)...")
    
    df = df.copy()
    
    # 仅对存在的列进行标准化
    valid_cols = [c for c in cols_to_standardize if c in df.columns]
    
    if not valid_cols:
        warnings.warn("没有找到需要标准化的列")
        return df
    
    grouped = df.groupby(group_col)
    
    for col in tqdm(valid_cols, desc="标准化特征"):
        df[f'{col}{suffix}'] = grouped[col].transform(
            lambda x: (x - x.expanding(min_periods=min_periods).mean()) / 
                     (x.expanding(min_periods=min_periods).std() + 1e-8)
        )
        
        # 缺失值填充
        df[f'{col}{suffix}'] = df[f'{col}{suffix}'].fillna(0)
    
    return df


def remove_outliers(
    df: pd.DataFrame,
    columns: List[str],
    method: str = 'zscore',
    threshold: float = 3.0
) -> pd.DataFrame:
    """
    移除异常值
    
    Args:
        df: 数据 DataFrame
        columns: 检查的列
        method: 方法 ('zscore' 或 'iqr')
        threshold: 阈值
    
    Returns:
        过滤后的 DataFrame
    """
    df = df.copy()
    
    for col in columns:
        if col not in df.columns:
            continue
        
        if method == 'zscore':
            z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
            df = df[z_scores < threshold]
        elif method == 'iqr':
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            df = df[(df[col] >= Q1 - threshold * IQR) & (df[col] <= Q3 + threshold * IQR)]
    
    return df

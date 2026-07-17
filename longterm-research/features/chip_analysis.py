"""
筹码分析模块

提供筹码分布特征计算功能
"""

import pandas as pd
import numpy as np
from typing import Optional


def analyze_chip_features(chips_df: pd.DataFrame) -> pd.DataFrame:
    """
    分析筹码分布特征
    
    Args:
        chips_df: 筹码分布数据，需包含 cost_15pct, cost_50pct, cost_85pct, winner_rate, weight_avg
    
    Returns:
        添加了筹码特征列的 DataFrame
    
    计算的特征:
        - chip_price_range_ratio: 价格区间比率
        - chip_symmetry_ratio: 对称性比率
        - chip_concentration_score: 筹码集中度评分
        - chip_is_single_peak: 是否单峰分布
        - chip_near_avg_price: 是否接近平均成本
    """
    if chips_df.empty:
        return chips_df
    
    result_df = chips_df.copy()
    
    # 计算价格区间比率
    result_df['chip_price_range_ratio'] = (
        (result_df['cost_85pct'] - result_df['cost_15pct']) / 
        result_df['cost_50pct'].replace(0, np.nan)
    )
    
    # 计算对称性比率
    upper_range = result_df['cost_85pct'] - result_df['cost_50pct']
    lower_range = result_df['cost_50pct'] - result_df['cost_15pct']
    max_range = np.maximum(upper_range, lower_range).replace(0, np.nan)
    min_range = np.minimum(upper_range, lower_range)
    result_df['chip_symmetry_ratio'] = min_range / max_range
    
    # 计算筹码集中度评分
    result_df['chip_concentration_score'] = (
        (1 - result_df['chip_price_range_ratio'].clip(upper=0.5)) * 
        result_df['winner_rate'] * 
        result_df['chip_symmetry_ratio']
    )
    
    # 判断是否单峰分布
    result_df['chip_is_single_peak'] = (
        (result_df['chip_price_range_ratio'] < 0.3) & 
        (result_df['winner_rate'] > 0.4) & 
        (result_df['chip_symmetry_ratio'] > 0.7)
    )
    
    # 判断是否接近平均成本
    weight_avg = result_df['weight_avg'].replace(0, np.nan)
    price_deviation = abs(result_df['weight_avg'] - weight_avg) / weight_avg
    result_df['chip_near_avg_price'] = price_deviation < 0.05
    
    # 填充 NaN
    result_df = result_df.fillna(0)
    
    return result_df


def enhance_chips_with_price_data(chips_df: pd.DataFrame, price_df: pd.DataFrame) -> pd.DataFrame:
    """
    用价格数据增强筹码数据
    
    Args:
        chips_df: 筹码分布数据
        price_df: 价格数据，需包含 ts_code, trade_date, close
    
    Returns:
        合并后的 DataFrame
    """
    if chips_df.empty or price_df.empty:
        return chips_df
    
    merged_df = pd.merge(
        chips_df,
        price_df[['ts_code', 'trade_date', 'close']],
        on=['ts_code', 'trade_date'],
        how='left'
    )
    
    return merged_df


def process_chips_with_features(chips_df: pd.DataFrame, price_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """
    处理筹码数据并计算特征
    
    Args:
        chips_df: 筹码分布数据
        price_df: 价格数据（可选）
    
    Returns:
        处理后的 DataFrame
    """
    if chips_df.empty:
        return chips_df
    
    # 如果提供了价格数据，先合并
    if price_df is not None:
        enhanced_chips = enhance_chips_with_price_data(chips_df, price_df)
    else:
        enhanced_chips = chips_df.copy()
    
    # 计算筹码特征
    result_df = analyze_chip_features(enhanced_chips)
    
    return result_df

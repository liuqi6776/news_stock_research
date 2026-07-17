"""
标签生成模块

提供多种标签生成方法，包括三阻碍法(Triple Barrier Method)
"""

import numpy as np
import pandas as pd
from typing import Optional


def triple_barrier_labeling(
    df: pd.DataFrame,
    volatility_col: str = 'volatility_20',
    horizon: int = 5,
    pt: float = 2.0,
    sl: float = 1.0,
    group_col: str = 'ts_code'
) -> pd.DataFrame:
    """
    三阻碍法标签生成 (Triple Barrier Method)
    
    根据未来价格路径判断是先触碰止盈线还是止损线，或者时间到期
    
    Args:
        df: 包含 close 和波动率列的 DataFrame
        volatility_col: 动态波动率列名
        horizon: 时间终止界（天）
        pt: 止盈倍数 (Profit Take) * volatility
        sl: 止损倍数 (Stop Loss) * volatility
        group_col: 分组列名
    
    Returns:
        添加 tb_label 列的 DataFrame
        - 1: 止盈
        - 2: 止损（便于多分类）
        - 0: 时间到期/无动作
    """
    print("正在生成三阻碍标签...")
    
    df = df.copy()
    
    # 确保按时间排序
    df = df.sort_values([group_col, 'trade_date'])
    
    def get_barrier_outcome(group: pd.DataFrame) -> pd.Series:
        """计算单只股票的标签"""
        closes = group['close'].values
        vols = group[volatility_col].values if volatility_col in group.columns else np.full(len(closes), 0.02)
        outcomes = np.zeros(len(closes))
        
        for i in range(len(closes) - horizon):
            current_price = closes[i]
            current_vol = vols[i]
            
            # 如果波动率太低或为 NaN，设置默认值
            if np.isnan(current_vol) or current_vol < 0.001:
                current_vol = 0.02
            
            upper_barrier = current_price * (1 + pt * current_vol)
            lower_barrier = current_price * (1 - sl * current_vol)
            
            # 未来窗口内的价格路径
            future_prices = closes[i + 1: i + 1 + horizon]
            
            # 检查是否触碰上界
            hit_upper = np.where(future_prices >= upper_barrier)[0]
            first_upper = hit_upper[0] if len(hit_upper) > 0 else horizon + 1
            
            # 检查是否触碰下界
            hit_lower = np.where(future_prices <= lower_barrier)[0]
            first_lower = hit_lower[0] if len(hit_lower) > 0 else horizon + 1
            
            if first_upper < first_lower and first_upper < horizon:
                outcomes[i] = 1  # 止盈
            elif first_lower < first_upper and first_lower < horizon:
                outcomes[i] = 2  # 止损
            else:
                outcomes[i] = 0  # 时间终止
        
        # 最后几天无法计算标签
        outcomes[-horizon:] = 0
        
        return pd.Series(outcomes, index=group.index)
    
    # 对每个股票应用
    df['tb_label'] = df.groupby(group_col).apply(get_barrier_outcome).reset_index(level=0, drop=True)
    
    print("三阻碍标签生成完成")
    return df


def simple_return_labeling(
    df: pd.DataFrame,
    forward_days: int = 5,
    threshold: float = 0.05,
    group_col: str = 'ts_code',
    price_col: str = 'open'  # 默认用 Open 价 (匹配 T+1 执行逻辑)
) -> pd.DataFrame:
    """
    简单收益率标签
    
    根据未来N天收益率分类
    
    Args:
        df: 包含 close/open 列的 DataFrame
        forward_days: 前瞻天数
        threshold: 分类阈值
        group_col: 分组列名
        price_col: 用于计算收益的价格列 ('open' 匹配 T+1, 'close' 为传统)
    
    Returns:
        添加 return_label 列的 DataFrame
        - 1: 上涨超过阈值
        - 0: 涨跌在阈值内
        - -1: 下跌超过阈值
    """
    df = df.copy()
    df = df.sort_values([group_col, 'trade_date'])
    
    # 计算未来收益率 (使用 price_col 列)
    df['future_return'] = df.groupby(group_col)[price_col].pct_change(forward_days).shift(-forward_days)
    
    # 生成标签
    df['return_label'] = 0
    df.loc[df['future_return'] > threshold, 'return_label'] = 1
    df.loc[df['future_return'] < -threshold, 'return_label'] = -1
    
    return df


def percentile_labeling(
    df: pd.DataFrame,
    forward_days: int = 5,
    top_pct: float = 0.2,
    bottom_pct: float = 0.2,
    group_col: str = 'ts_code'
) -> pd.DataFrame:
    """
    百分位标签
    
    根据同期所有股票的收益率排名分类
    
    Args:
        df: 包含 close 列的 DataFrame
        forward_days: 前瞻天数
        top_pct: 顶部百分位阈值
        bottom_pct: 底部百分位阈值
        group_col: 分组列名
    
    Returns:
        添加 pct_label 列的 DataFrame
    """
    df = df.copy()
    df = df.sort_values([group_col, 'trade_date'])
    
    # 计算未来收益率
    df['future_return'] = df.groupby(group_col)['close'].pct_change(forward_days).shift(-forward_days)
    
    # 按日期分组计算百分位排名
    df['return_rank'] = df.groupby('trade_date')['future_return'].rank(pct=True)
    
    # 生成标签
    df['pct_label'] = 0
    df.loc[df['return_rank'] >= (1 - top_pct), 'pct_label'] = 1
    df.loc[df['return_rank'] <= bottom_pct, 'pct_label'] = -1
    
    return df

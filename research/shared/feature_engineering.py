"""
特征工程模块 - 计算所有特征
"""
import pandas as pd
import numpy as np
from typing import List


def is_main_board(ts_code: str) -> bool:
    """判断是否主板股票"""
    return ts_code.startswith(('60', '00', '002', '003'))


def calculate_technical_features(df: pd.DataFrame) -> pd.DataFrame:
    """计算技术指标特征"""
    df = df.copy()
    
    # 基础价格特征
    df['pct_chg'] = (df['close'] - df['pre_close']) / df['pre_close']
    df['price_change'] = df['close'] - df['pre_close']
    df['close_to_high'] = (df['high'] - df['close']) / df['pre_close']
    df['close_to_low'] = (df['close'] - df['low']) / df['pre_close']
    df['high_change'] = (df['high'] - df['pre_close']) / df['pre_close']
    df['low_change'] = (df['low'] - df['pre_close']) / df['pre_close']
    df['amplitude'] = (df['high'] - df['low']) / df['pre_close']
    df['body_size'] = abs(df['close'] - df['open']) / df['pre_close']
    df['upper_shadow'] = (df['high'] - df[['close', 'open']].max(axis=1)) / df['pre_close']
    df['lower_shadow'] = (df[['close', 'open']].min(axis=1) - df['low']) / df['pre_close']
    df['is_yang'] = (df['close'] > df['open']).astype(int)
    df['gap'] = (df['open'] - df['pre_close']) / df['pre_close']
    df['vol_amount'] = df['close'] * df['vol']
    df['vol_ratio_day'] = df['vol'] / (df['vol'].mean() + 1e-8)
    
    return df


def calculate_momentum_features(df: pd.DataFrame) -> pd.DataFrame:
    """计算动量特征"""
    df = df.copy()
    
    for w in [5, 10, 20, 60]:
        df[f'mom_{w}d'] = df['close'].pct_change(w)
    
    return df


def calculate_volatility_features(df: pd.DataFrame) -> pd.DataFrame:
    """计算波动率特征"""
    df = df.copy()
    
    for w in [5, 10, 20, 60]:
        df[f'vol_{w}d'] = df['close'].pct_change().rolling(w).std() * (252 ** 0.5)
    
    return df


def calculate_all_features(price_df: pd.DataFrame, 
                          news_df: pd.DataFrame = None,
                          rank_df: pd.DataFrame = None,
                          factor_df: pd.DataFrame = None) -> pd.DataFrame:
    """
    计算所有特征
    
    Args:
        price_df: 行情数据
        news_df: 新闻数据 (可选)
        rank_df: 排名数据 (可选)
        factor_df: 因子数据 (可选)
    
    Returns:
        包含所有特征的DataFrame
    """
    df = price_df.copy()
    
    # 技术特征
    df = calculate_technical_features(df)
    
    # 合并新闻特征
    if news_df is not None and not news_df.empty:
        news_cols = [c for c in news_df.columns if c not in ['ts_code', 'trade_date']]
        df = pd.merge(df, news_df[['ts_code'] + news_cols], on='ts_code', how='left')
    
    # 合并排名特征
    if rank_df is not None and not rank_df.empty:
        rank_cols = [c for c in rank_df.columns if c not in ['ts_code', 'trade_date']]
        df = pd.merge(df, rank_df[['ts_code'] + rank_cols], on='ts_code', how='left')
    
    # 合并因子特征
    if factor_df is not None and not factor_df.empty:
        factor_cols = [c for c in factor_df.columns if c not in ['ts_code', 'trade_date']]
        df = pd.merge(df, factor_df[['ts_code'] + factor_cols], on='ts_code', how='left')
    
    return df


def prepare_training_data(start_date: str, end_date: str, 
                         all_dates: List[str],
                         price_dir: str) -> pd.DataFrame:
    """
    准备训练数据 - 计算目标变量和特征
    
    Args:
        start_date: 开始日期
        end_date: 结束日期
        all_dates: 所有交易日列表
        price_dir: 价格数据目录
    
    Returns:
        训练数据DataFrame
    """
    import os
    
    start_idx = all_dates.index(start_date)
    end_idx = all_dates.index(end_date)
    
    training_data = []
    
    for i in range(start_idx, end_idx + 1):
        d_curr = all_dates[i]
        
        # 需要t+1和t+2的数据
        if i + 2 >= len(all_dates):
            break
            
        d_t1 = all_dates[i + 1]
        d_t2 = all_dates[i + 2]
        
        # 加载数据
        p_curr = os.path.join(price_dir, f"{d_curr}.parquet")
        p_t1 = os.path.join(price_dir, f"{d_t1}.parquet")
        p_t2 = os.path.join(price_dir, f"{d_t2}.parquet")
        
        if not all(os.path.exists(p) for p in [p_curr, p_t1, p_t2]):
            continue
        
        df_curr = pd.read_parquet(p_curr)
        df_t1 = pd.read_parquet(p_t1)
        df_t2 = pd.read_parquet(p_t2)
        
        # 过滤主板
        df_curr = df_curr[df_curr['ts_code'].apply(is_main_board)]
        
        if df_curr.empty:
            continue
        
        # 计算特征
        features = calculate_all_features(df_curr)
        
        # 计算目标变量 (t+1开盘买入, t+2收盘卖出)
        t1_data = df_t1[['ts_code', 'open']].rename(columns={'open': 't1_open'})
        t2_data = df_t2[['ts_code', 'close']].rename(columns={'close': 't2_close'})
        
        features = pd.merge(features, t1_data, on='ts_code', how='left')
        features = pd.merge(features, t2_data, on='ts_code', how='left')
        
        # 计算收益率
        features['label_ret'] = features['t2_close'] / features['t1_open'] - 1
        features['trade_date'] = d_curr
        
        training_data.append(features)
    
    if training_data:
        return pd.concat(training_data, ignore_index=True)
    return pd.DataFrame()

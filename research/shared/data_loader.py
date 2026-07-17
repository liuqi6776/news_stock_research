"""
数据加载模块 - 统一加载原始数据
"""
import pandas as pd
import os
from typing import List, Optional

# 数据路径
PRICE_DIR = r'D:\iquant_data\data_v2\data_day1'
NEWS_DIR = r'D:\iquant_data\data_v2\data_day_news'
RANK_DIR = r'D:\iquant_data\data_v2\data_day_ths_rank'
FACTOR_DIR = r'D:\iquant_data\data_v2\data_day_factor'


def get_all_dates() -> List[str]:
    """获取所有交易日列表"""
    return sorted([f.replace('.parquet', '') for f in os.listdir(PRICE_DIR)])


def load_price_data(date: str) -> Optional[pd.DataFrame]:
    """加载某一天的行情数据"""
    p = os.path.join(PRICE_DIR, f"{date}.parquet")
    if os.path.exists(p):
        return pd.read_parquet(p)
    return None


def load_news_data(date: str) -> Optional[pd.DataFrame]:
    """加载某一天的新闻数据"""
    p = os.path.join(NEWS_DIR, f"{date}.parquet")
    if os.path.exists(p):
        return pd.read_parquet(p)
    return None


def load_rank_data(date: str) -> Optional[pd.DataFrame]:
    """加载某一天的同花顺排名数据"""
    p = os.path.join(RANK_DIR, f"{date}.parquet")
    if os.path.exists(p):
        return pd.read_parquet(p)
    return None


def load_factor_data(date: str) -> Optional[pd.DataFrame]:
    """加载某一天的因子数据"""
    p = os.path.join(FACTOR_DIR, f"{date}.parquet")
    if os.path.exists(p):
        return pd.read_parquet(p)
    return None


def load_price_range(start_date: str, end_date: str) -> pd.DataFrame:
    """加载一段时间的价格数据"""
    all_dates = get_all_dates()
    
    # 找到范围内的日期
    start_idx = all_dates.index(start_date) if start_date in all_dates else 0
    end_idx = all_dates.index(end_date) if end_date in all_dates else len(all_dates)
    dates = all_dates[start_idx:end_idx+1]
    
    dfs = []
    for d in dates:
        df = load_price_data(d)
        if df is not None:
            df['trade_date'] = d
            dfs.append(df)
    
    if dfs:
        return pd.concat(dfs, ignore_index=True)
    return pd.DataFrame()

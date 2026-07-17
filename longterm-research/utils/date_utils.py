"""
日期处理工具模块

提供统一的日期格式转换和处理函数
"""

import datetime
from typing import Union, List, Optional
import pandas as pd


def parse_date(date_input: Union[str, int, datetime.date, pd.Timestamp]) -> datetime.date:
    """
    解析各种格式的日期输入为 datetime.date
    
    Args:
        date_input: 日期输入，支持格式:
            - 字符串: '20240101', '2024-01-01', '2024/01/01'
            - 整数: 20240101
            - datetime.date 对象
            - pd.Timestamp 对象
    
    Returns:
        datetime.date 对象
    
    Example:
        >>> parse_date('20240101')
        datetime.date(2024, 1, 1)
        >>> parse_date(20240101)
        datetime.date(2024, 1, 1)
    """
    if isinstance(date_input, datetime.date):
        return date_input
    
    if isinstance(date_input, pd.Timestamp):
        return date_input.date()
    
    if isinstance(date_input, (int, float)):
        date_input = str(int(date_input))
    
    date_str = str(date_input).strip()
    
    # 尝试不同的日期格式
    formats = ['%Y%m%d', '%Y-%m-%d', '%Y/%m/%d']
    for fmt in formats:
        try:
            return datetime.datetime.strptime(date_str, fmt).date()
        except ValueError:
            continue
    
    raise ValueError(f"无法解析日期: {date_input}")


def format_date(date_input: Union[str, int, datetime.date, pd.Timestamp], 
                output_format: str = '%Y%m%d') -> str:
    """
    格式化日期为指定格式的字符串
    
    Args:
        date_input: 日期输入
        output_format: 输出格式，默认 '%Y%m%d'
    
    Returns:
        格式化后的日期字符串
    
    Example:
        >>> format_date(datetime.date(2024, 1, 1))
        '20240101'
        >>> format_date('2024-01-01', '%Y-%m-%d')
        '2024-01-01'
    """
    date_obj = parse_date(date_input)
    return date_obj.strftime(output_format)


def to_datetime(date_series: pd.Series) -> pd.Series:
    """
    将 Pandas Series 中的日期转换为 datetime 类型
    
    Args:
        date_series: 包含日期的 Series
    
    Returns:
        转换后的 datetime Series
    """
    return pd.to_datetime(date_series.astype(str), format='%Y%m%d', errors='coerce')


def get_today(fmt: str = '%Y%m%d') -> str:
    """
    获取今天的日期字符串
    
    Args:
        fmt: 日期格式
    
    Returns:
        今天的日期字符串
    """
    return datetime.datetime.today().strftime(fmt)


def get_date_range(start_date: str, end_date: str) -> List[str]:
    """
    获取日期范围内的所有日期
    
    Args:
        start_date: 开始日期 (格式: YYYYMMDD)
        end_date: 结束日期 (格式: YYYYMMDD)
    
    Returns:
        日期字符串列表
    """
    start = parse_date(start_date)
    end = parse_date(end_date)
    
    dates = []
    current = start
    while current <= end:
        dates.append(format_date(current))
        current += datetime.timedelta(days=1)
    
    return dates


def get_latest_date_file(directory: str, pattern: str = '*.parquet') -> Optional[str]:
    """
    获取目录下最新的日期文件名
    
    Args:
        directory: 目录路径
        pattern: 文件匹配模式
    
    Returns:
        最新的日期字符串，如无则返回 None
    """
    import os
    
    if not os.path.exists(directory):
        return None
    
    candidates = []
    for f in os.listdir(directory):
        if f.endswith('.parquet') and len(f) >= 8:
            date_part = f[:8]
            if date_part.isdigit():
                candidates.append(date_part)
    
    return max(candidates) if candidates else None

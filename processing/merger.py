"""
数据合并模块

提供 DataFrame 合并和日期扩展功能
"""

import warnings
from typing import List, Optional
from functools import reduce

import pandas as pd


def merge_dataframes(
    df_list: List[pd.DataFrame],
    on: List[str] = ['trade_date', 'ts_code'],
    how: str = 'outer',
    drop_duplicates: bool = True
) -> pd.DataFrame:
    """
    合并多个 DataFrame
    
    Args:
        df_list: DataFrame 列表
        on: 合并键
        how: 合并方式
        drop_duplicates: 是否删除重复列
    
    Returns:
        合并后的 DataFrame
    """
    if not isinstance(df_list, list) or len(df_list) == 0:
        raise ValueError("df_list 必须是包含至少一个 DataFrame 的列表")
    
    def standardize_dates(df: pd.DataFrame) -> pd.DataFrame:
        """标准化日期格式"""
        if 'trade_date' in df.columns:
            df['trade_date'] = pd.to_datetime(df['trade_date'], errors='coerce')
        return df
    
    # 预处理
    processed_dfs = []
    for df in df_list:
        if df.empty:
            continue
        
        df = standardize_dates(df.copy())
        
        # 处理中国波指数据（只有 trade_date 列，没有 ts_code）
        if 'ts_code' not in df.columns and 'trade_date' in df.columns:
            # 为中国波指数据添加 ts_code 列，值为 '000130.SH'
            df['ts_code'] = '000130.SH'
        
        # 处理融资融券数据（可能有不同的列名）
        if 'ts_code' in df.columns:
            processed_dfs.append(df)
        else:
            # 对于没有 ts_code 的数据，只按日期合并
            if 'trade_date' in df.columns:
                # 重命名列以避免冲突
                df = df.rename(columns={col: f'margin_{col}' for col in df.columns if col != 'trade_date'})
                processed_dfs.append(df)
    
    df_list = [df.drop_duplicates(subset=on if 'ts_code' in df.columns else ['trade_date'], keep='first') 
               for df in processed_dfs]
    df_list = [df for df in df_list if not df.empty]
    
    if not df_list:
        return pd.DataFrame(columns=on)
    
    # 逐个合并
    merged_df = df_list[0]
    for i in range(1, len(df_list)):
        right_df = df_list[i]
        # 确定合并键
        if 'ts_code' in right_df.columns:
            merge_on = on
        else:
            merge_on = ['trade_date']
        
        merged_df = pd.merge(merged_df, right_df, on=merge_on, how=how, suffixes=('', '_dup'))
    
    # 删除重复列
    if drop_duplicates:
        dup_cols = [col for col in merged_df.columns if col.endswith('_dup')]
        merged_df = merged_df.drop(columns=dup_cols)
        merged_df = merged_df.drop_duplicates(subset=on)
    
    merged_df = merged_df.sort_values(by=on).reset_index(drop=True)
    
    return merged_df


def extend_to_current_date(
    df: pd.DataFrame,
    id_col: str = 'ts_code',
    date_col: str = 'trade_date'
) -> pd.DataFrame:
    """
    将数据扩展到当前日期
    
    用于处理季度数据等需要向前填充的情况
    
    Args:
        df: 数据 DataFrame
        id_col: ID 列名
        date_col: 日期列名
    
    Returns:
        扩展后的 DataFrame
    """
    if df.empty:
        return df
    
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    
    today = pd.Timestamp.today().normalize()
    
    # 获取每个 ID 的最新数据
    latest_data = df.sort_values([id_col, date_col]).groupby(id_col).tail(1)
    
    extended_rows = []
    for _, row in latest_data.iterrows():
        max_date = row[date_col]
        if max_date >= today:
            continue
        
        # 生成缺失日期
        date_range = pd.date_range(start=max_date + pd.Timedelta(days=1), end=today)
        
        for date in date_range:
            new_row = row.copy()
            new_row[date_col] = date
            extended_rows.append(new_row)
    
    if extended_rows:
        extended_df = pd.DataFrame(extended_rows)
        df = pd.concat([df, extended_df], ignore_index=True)
    
    return df


def process_broker_recommend(df: pd.DataFrame) -> pd.DataFrame:
    """
    处理券商推荐数据，将月度数据展开到每日
    
    Args:
        df: 券商推荐数据
    
    Returns:
        展开后的 DataFrame
    """
    import calendar
    
    if df.empty:
        return pd.DataFrame()
    
    def get_dates_from_month(month_str: str) -> pd.DatetimeIndex:
        """获取月份内的所有日期"""
        year, month = int(month_str[:4]), int(month_str[-2:])
        _, last_day = calendar.monthrange(year, month)
        return pd.date_range(start=f'{year}-{month}-01', end=f'{year}-{month}-{last_day}')
    
    expanded_rows = []
    for _, row in df.iterrows():
        if 'month' not in row:
            continue
        dates = get_dates_from_month(row['month'])
        for date in dates:
            expanded_rows.append({
                'trade_date': date,
                'ts_code': row['ts_code'],
                'broker': row.get('broker', '')
            })
    
    return pd.DataFrame(expanded_rows)

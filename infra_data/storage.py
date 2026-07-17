"""
数据存储模块

提供统一的数据读取和存储接口
"""

import os
from typing import List, Optional, Callable
import warnings

import pandas as pd
from tqdm import tqdm

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import settings
from utils.date_utils import to_datetime


class DataStorage:
    """
    数据存储管理器
    
    提供读取和存储 parquet 数据文件的统一接口
    """
    
    def __init__(self, base_path: str = None):
        """
        初始化存储管理器
        
        Args:
            base_path: 数据基础路径，默认从配置读取
        """
        self.base_path = base_path or settings.DATA_PATH
    
    def read_parquet_files(
        self,
        data_dir: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        date_column: str = 'trade_date',
        parse_date: bool = True,
        show_progress: bool = True
    ) -> pd.DataFrame:
        """
        读取目录下的所有 parquet 文件并合并
        
        Args:
            data_dir: 数据目录路径
            start_date: 开始日期过滤（基于文件名）
            end_date: 结束日期过滤（基于文件名）
            date_column: 日期列名
            parse_date: 是否解析日期列为 datetime
            show_progress: 是否显示进度条
        
        Returns:
            合并后的 DataFrame
        """
        if not os.path.exists(data_dir):
            warnings.warn(f"目录不存在: {data_dir}")
            return pd.DataFrame()
        
        # 获取文件列表
        file_list = [f for f in os.listdir(data_dir) 
                     if f.endswith('.parquet') and not f.startswith('.')]
        
        # 日期过滤
        if start_date or end_date:
            filtered_files = []
            for f in file_list:
                file_date = f.split('.')[0]
                if len(file_date) >= 6:  # 至少是月份格式
                    if start_date and file_date < start_date[:len(file_date)]:
                        continue
                    if end_date and file_date > end_date[:len(file_date)]:
                        continue
                filtered_files.append(f)
            file_list = filtered_files
        
        if not file_list:
            return pd.DataFrame()
        
        # 读取文件
        dfs = []
        iterator = tqdm(file_list, desc=f"读取 {os.path.basename(data_dir)}") if show_progress else file_list
        
        for filename in iterator:
            try:
                df = pd.read_parquet(os.path.join(data_dir, filename))
                
                # 如果没有 trade_date 列，从文件名提取
                if date_column not in df.columns:
                    file_date = filename.split('.')[0]
                    df[date_column] = file_date
                
                dfs.append(df)
            except Exception as e:
                warnings.warn(f"读取文件失败 {filename}: {str(e)}")
        
        if not dfs:
            return pd.DataFrame()
        
        result = pd.concat(dfs, ignore_index=True)
        
        # 解析日期列
        if parse_date and date_column in result.columns:
            result[date_column] = to_datetime(result[date_column])
        
        return result
    
    def load_daily_data(self, start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """加载日线行情数据"""
        return self.read_parquet_files(settings.daily_data_path, start_date, end_date)
    
    def load_daily_basic(self, start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """加载每日指标数据"""
        return self.read_parquet_files(settings.other_data_path, start_date, end_date)
    
    def load_technical_factors(self, start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """加载技术因子数据"""
        return self.read_parquet_files(settings.skill_data_path, start_date, end_date)
    
    def load_chip_data(self, start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """加载筹码分布数据"""
        return self.read_parquet_files(settings.cyq_path, start_date, end_date)
    
    def load_money_flow(self, start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """加载资金流向数据"""
        return self.read_parquet_files(settings.moneyflow_path, start_date, end_date)
    
    def load_ths_rank(self, start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """加载同花顺热榜数据"""
        return self.read_parquet_files(settings.ths_rank_path, start_date, end_date)
    
    def load_board_data(self, start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """加载行业板块数据"""
        return self.read_parquet_files(settings.board_path, start_date, end_date)
    
    def load_vix_data(self, start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """加载中国波指数据"""
        return self.read_parquet_files(settings.vix_data_path, start_date, end_date)
    
    def load_margin_data(self, start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """加载融资融券数据"""
        return self.read_parquet_files(settings.margin_data_path, start_date, end_date)
    
    def load_news_data(self, start_date: str = None, end_date: str = None, valid_dates: pd.Series = None) -> tuple:
        """加载新闻情绪数据"""
        try:
            from processing.news_processor import load_and_process_news
            news_dir = r'D:\iquant_data\data_v2\news_major1'
            industry_map_path = os.path.join(self.base_path, 'stock_industry_map_cached.parquet')
            return load_and_process_news(news_dir, start_date, end_date, industry_map_path, valid_dates)
        except Exception as e:
            warnings.warn(f"加载新闻数据失败: {str(e)}")
            import pandas as pd
            return pd.DataFrame(), pd.DataFrame()
    
    def save_dataframe(self, df: pd.DataFrame, filepath: str):
        """保存 DataFrame 为 parquet 文件"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        df.to_parquet(filepath, index=False)

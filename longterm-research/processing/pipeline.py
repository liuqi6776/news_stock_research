"""
数据处理流水线模块

提供完整的数据处理流程
"""

import os
from typing import List, Optional, Tuple

import pandas as pd
import joblib

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import settings
from infra_data.storage import DataStorage
from features.alpha_factors import calculate_alpha101_factors
from features.chip_analysis import process_chips_with_features
from features.labeling import triple_barrier_labeling
from .cleaner import remove_st_stocks, filter_stock_codes, apply_expanding_standardization
from .merger import merge_dataframes, extend_to_current_date, process_broker_recommend


class DataPipeline:
    """
    数据处理流水线
    
    集成数据加载、清洗、特征工程、标签生成的完整流程
    
    Example:
        pipeline = DataPipeline()
        df = pipeline.run(start_date='20200103', end_date='20251118')
    """
    
    def __init__(self, data_path: str = None):
        """
        初始化流水线
        
        Args:
            data_path: 数据路径，默认从配置读取
        """
        self.data_path = data_path or settings.DATA_PATH
        self.storage = DataStorage(self.data_path)
    
    def load_all_data(self, start_date: str, end_date: str) -> List[pd.DataFrame]:
        """
        加载所有数据源
        
        Args:
            start_date: 开始日期
            end_date: 结束日期
        
        Returns:
            DataFrame 列表
        """
        print("Step 1: 加载数据...")
        
        data_all = self.storage.load_daily_data(start_date, end_date)
        other_all = self.storage.load_daily_basic(start_date, end_date)
        skill_all = self.storage.load_technical_factors(start_date, end_date)
        chips_all = self.storage.load_chip_data(start_date, end_date)
        moneyflow_all = self.storage.load_money_flow(start_date, end_date)
        rank_all = self.storage.load_ths_rank(start_date, end_date)
        vix_all = self.storage.load_vix_data(start_date, end_date)
        margin_all = self.storage.load_margin_data(start_date, end_date)
        
        # 处理券商推荐数据
        roc_path = settings.recommend_data_path
        roc_all = self.storage.read_parquet_files(roc_path, start_date[:6])
        if not roc_all.empty:
            roc_all = process_broker_recommend(roc_all)
            roc_all = pd.get_dummies(roc_all, columns=['broker']).drop_duplicates()
            
        # 提取交易日历以将新闻影响顺延一个交易日
        valid_dates_series = None
        if not data_all.empty:
            valid_dates_series = pd.Series(sorted(data_all['trade_date'].unique()))
            
        # 加载新闻数据
        news_market_df, news_stock_sector_df = self.storage.load_news_data(start_date, end_date, valid_dates_series)
        
        return [data_all, roc_all, other_all, skill_all, rank_all, chips_all, moneyflow_all, vix_all, margin_all, news_market_df, news_stock_sector_df]
    
    def process_chips(self, dfs: List[pd.DataFrame]) -> List[pd.DataFrame]:
        """处理筹码数据"""
        if len(dfs) > 5 and not dfs[5].empty and not dfs[0].empty:
            print("处理筹码分布特征...")
            dfs[5] = process_chips_with_features(dfs[5], dfs[0])
        return dfs
    
    def run(
        self,
        start_date: str,
        end_date: str,
        apply_st_filter: bool = True,
        apply_alpha_factors: bool = True,
        apply_standardization: bool = True,
        apply_labeling: bool = True,
        save_intermediate: bool = True
    ) -> pd.DataFrame:
        """
        运行完整的数据处理流水线
        
        Args:
            start_date: 开始日期
            end_date: 结束日期
            apply_st_filter: 是否过滤 ST 股票
            apply_alpha_factors: 是否计算 Alpha 因子
            apply_standardization: 是否应用标准化
            apply_labeling: 是否生成标签
            save_intermediate: 是否保存中间结果
        
        Returns:
            处理后的 DataFrame
        """
        # 1. 加载数据
        dfs = self.load_all_data(start_date, end_date)
        
        # 2. 处理筹码特征
        dfs = self.process_chips(dfs)
        
        # 保存中间结果
        if save_intermediate:
            joblib.dump(dfs, "old_list.pkl")
        
        # 3. 合并数据
        print("Step 2: 合并数据表...")
        merged_df = merge_dataframes(dfs)
        
        # 填充新闻特征的空值
        news_cols = ['news_market_impact', 'news_stock_impact', 'news_sector_impact']
        for col in news_cols:
            if col in merged_df.columns:
                merged_df[col] = merged_df[col].fillna(0.0)
        
        # 4. ST 股过滤
        if apply_st_filter:
            merged_df = remove_st_stocks(merged_df)
        
        # 5. Alpha 因子计算
        if apply_alpha_factors:
            print("Step 3: 计算 Alpha 因子...")
            merged_df = calculate_alpha101_factors(merged_df)
        
        # 6. 标准化
        if apply_standardization:
            print("Step 4: 应用标准化...")
            cols_to_std = [
                'turnover_rate', 'volume_ratio', 'pe', 'pb', 'circ_mv',
                'macd', 'rsi_6', 'rsi_12', 'rsi_24', 'cci', 'kdj_k', 'kdj_d', 'kdj_j',
                'alpha_006', 'alpha_009', 'alpha_012', 'alpha_023', 'volatility_20'
            ]
            merged_df = apply_expanding_standardization(merged_df, cols_to_std)
        
        # 7. 标签生成
        if apply_labeling:
            print("Step 5: 生成标签...")
            merged_df = triple_barrier_labeling(merged_df)
        
        # 保存合并结果
        if save_intermediate:
            merged_df.to_parquet("merged_df.parquet")
        
        # 8. 数据筛选
        print("Step 6: 数据筛选与导出...")
        filtered_df = self._filter_data(merged_df)
        
        if save_intermediate:
            filtered_df.to_parquet("filtered_df.parquet")
        
        print("数据处理完成!")
        return filtered_df
    
    def _filter_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """数据筛选"""
        mdate = df['trade_date'].max()
        
        # 只保留最新有数据的股票
        valid_codes = df.groupby('ts_code').apply(
            lambda x: x['trade_date'].max() == mdate
        ).reset_index()
        valid_codes = valid_codes[valid_codes[0]]['ts_code']
        
        # 数据量要求
        num_df = pd.DataFrame(df.groupby('ts_code').count()['open'] > settings.MIN_DATA_POINTS)
        new_codes = num_df[num_df['open'] == True].index
        
        # 应用筛选
        filtered_df = df[
            df['ts_code'].isin(valid_codes) & 
            df['ts_code'].isin(new_codes)
        ].copy()
        
        # 数据质量过滤
        filtered_df['data_quality'] = (
            filtered_df.groupby('ts_code')['open'].transform('count') / 
            len(df['trade_date'].unique())
        )
        filtered_df = filtered_df[filtered_df['data_quality'] >= settings.DATA_QUALITY_THRESHOLD]
        
        # 股票代码筛选（只保留主板股票）
        filtered_df = filter_stock_codes(filtered_df)
        
        # 去重
        filtered_df = filtered_df.drop_duplicates()
        filtered_df = filtered_df.sort_values('trade_date').groupby(
            ['trade_date', 'ts_code']
        ).tail(1).reset_index(drop=True)
        
        # 打印空值统计
        null_ratios = (filtered_df.isnull().sum() / len(filtered_df)) * 100
        null_ratios = null_ratios.round(2).sort_values(ascending=False)
        print("各列空值比例 (%):")
        print(null_ratios.head(10))
        
        return filtered_df

#!/usr/bin/env python3
"""
测试新添加的中国波指和融资融券数据功能
"""

import os
import sys
import pandas as pd

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from infra_data.fetcher import DataFetcher
from processing.pipeline import DataPipeline
from config.settings import settings

def test_data_fetching():
    """测试数据获取功能"""
    print("测试数据获取功能...")
    
    # 初始化数据获取器
    fetcher = DataFetcher()
    
    # 获取最近30天的交易日期
    end_date = pd.Timestamp.today().strftime('%Y%m%d')
    start_date = (pd.Timestamp.today() - pd.Timedelta(days=30)).strftime('%Y%m%d')
    
    # 测试获取中国波指数据
    print(f"\n获取中国波指数据 ({start_date} - {end_date})...")
    vix_failed = fetcher.fetch_vix_data(start_date, end_date)
    print(f"中国波指数据获取完成，失败日期: {vix_failed}")
    
    # 测试获取融资融券数据
    dates = fetcher.get_trading_dates(start_date, end_date)
    print(f"\n获取融资融券数据 ({start_date} - {end_date})...")
    margin_failed = fetcher.fetch_margin_data(dates)
    print(f"融资融券数据获取完成，失败日期: {margin_failed}")

def test_data_processing():
    """测试数据处理功能"""
    print("\n\n测试数据处理功能...")
    
    # 初始化数据处理流水线
    pipeline = DataPipeline()
    
    # 处理最近90天的数据
    end_date = pd.Timestamp.today().strftime('%Y%m%d')
    start_date = (pd.Timestamp.today() - pd.Timedelta(days=90)).strftime('%Y%m%d')
    
    print(f"处理数据 ({start_date} - {end_date})...")
    try:
        df = pipeline.run(
            start_date=start_date,
            end_date=end_date,
            apply_st_filter=True,
            apply_alpha_factors=True,
            apply_standardization=True,
            apply_labeling=True,
            save_intermediate=True
        )
        
        print(f"\n数据处理完成，处理后的数据形状: {df.shape}")
        print(f"数据列名: {list(df.columns)}")
        
        # 检查是否包含中国波指和融资融券相关列
        vix_columns = [col for col in df.columns if 'vix' in col.lower() or '000188' in col]
        margin_columns = [col for col in df.columns if 'margin' in col.lower()]
        
        print(f"\n中国波指相关列: {vix_columns}")
        print(f"融资融券相关列: {margin_columns}")
        
        # 显示前几行数据
        print("\n前5行数据:")
        print(df.head())
        
    except Exception as e:
        print(f"数据处理失败: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # 验证配置
    try:
        settings.validate()
        print("配置验证通过")
    except Exception as e:
        print(f"配置验证失败: {str(e)}")
        sys.exit(1)
    
    # 测试数据获取
    test_data_fetching()
    
    # 测试数据处理
    test_data_processing()
    
    print("\n\n测试完成！")

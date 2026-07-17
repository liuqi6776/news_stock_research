#!/usr/bin/env python3
"""
简化版测试脚本，用于验证新feature的集成情况
"""

import os
import sys
import pandas as pd

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config.settings import settings
from infra_data.fetcher import DataFetcher
from processing.merger import merge_dataframes

def test_data_fetching():
    """测试数据获取功能"""
    print("测试数据获取功能...")
    
    # 初始化数据获取器
    fetcher = DataFetcher()
    
    # 处理最近7天的数据
    end_date = pd.Timestamp.today().strftime('%Y%m%d')
    start_date = (pd.Timestamp.today() - pd.Timedelta(days=7)).strftime('%Y%m%d')
    
    # 获取交易日期
    dates = fetcher.get_trading_dates(start_date, end_date)
    print(f"获取数据 ({start_date} - {end_date})...")
    print(f"交易日期: {dates}")
    
    try:
        # 获取融资融券数据
        print("1. 获取融资融券数据...")
        failed_margin = fetcher.fetch_margin_data(dates)
        print(f"   融资融券数据获取完成，失败日期: {failed_margin}")
        
        # 获取中国波指数据
        print("2. 获取中国波指数据...")
        failed_vix = fetcher.fetch_vix_data(start_date, end_date)
        print(f"   中国波指数据获取完成，失败日期: {failed_vix}")
        
        # 加载保存的数据
        from infra_data.storage import DataStorage
        storage = DataStorage()
        
        print("3. 加载保存的数据...")
        margin_data = storage.load_margin_data(start_date, end_date)
        print(f"   加载的融资融券数据形状: {margin_data.shape}")
        print(f"   加载的融资融券数据列: {list(margin_data.columns)}")
        
        vix_data = storage.load_vix_data(start_date, end_date)
        print(f"   加载的中国波指数据形状: {vix_data.shape}")
        print(f"   加载的中国波指数据列: {list(vix_data.columns)}")
        
        return margin_data, vix_data
        
    except Exception as e:
        print(f"数据获取失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, None

def test_data_merging(margin_data, vix_data):
    """测试数据合并功能"""
    if margin_data is None or vix_data is None:
        print("数据获取失败，跳过合并测试")
        return None
    
    print("\n测试数据合并功能...")
    
    try:
        # 合并数据
        merged_df = merge_dataframes([margin_data, vix_data])
        print(f"合并后的数据形状: {merged_df.shape}")
        print(f"合并后的数据列: {list(merged_df.columns)}")
        
        # 显示前几行数据
        print("\n前5行数据:")
        print(merged_df.head())
        
        # 保存合并后的数据
        output_path = "merged_features.parquet"
        merged_df.to_parquet(output_path)
        print(f"\n合并后的数据已保存到: {output_path}")
        
        return merged_df
        
    except Exception as e:
        print(f"数据合并失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def main():
    """主函数"""
    print("开始测试新feature的集成情况...")
    
    # 测试数据获取
    margin_data, vix_data = test_data_fetching()
    
    # 测试数据合并
    merged_df = test_data_merging(margin_data, vix_data)
    
    print("\n测试完成！")

if __name__ == "__main__":
    main()

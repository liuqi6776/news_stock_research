#!/usr/bin/env python3
"""
测试000183.SH数据获取
"""

import os
import sys
import pandas as pd

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from infra_data.fetcher import DataFetcher
from config.settings import settings

def test_vix_data():
    """测试000183.SH数据获取"""
    print("测试000183.SH数据获取...")
    
    # 初始化数据获取器
    fetcher = DataFetcher()
    
    # 获取最近30天的交易日期
    end_date = pd.Timestamp.today().strftime('%Y%m%d')
    start_date = (pd.Timestamp.today() - pd.Timedelta(days=30)).strftime('%Y%m%d')
    
    # 测试获取000183.SH数据
    print(f"\n获取000183.SH数据 ({start_date} - {end_date})...")
    vix_failed = fetcher.fetch_vix_data(start_date, end_date)
    print(f"000183.SH数据获取完成，失败日期: {vix_failed}")

if __name__ == "__main__":
    # 验证配置
    try:
        settings.validate()
        print("配置验证通过")
    except Exception as e:
        print(f"配置验证失败: {str(e)}")
        sys.exit(1)
    
    # 测试数据获取
    test_vix_data()
    
    print("\n测试完成！")

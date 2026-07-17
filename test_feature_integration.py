#!/usr/bin/env python3
"""
测试新feature的集成情况
"""

import os
import sys
import pandas as pd

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from processing.pipeline import DataPipeline
from config.settings import settings

def test_feature_integration():
    """测试新feature的集成情况"""
    print("测试新feature的集成情况...")
    
    # 初始化数据处理流水线
    pipeline = DataPipeline()
    
    # 处理最近30天的数据
    end_date = pd.Timestamp.today().strftime('%Y%m%d')
    start_date = (pd.Timestamp.today() - pd.Timedelta(days=30)).strftime('%Y%m%d')
    
    print(f"处理数据 ({start_date} - {end_date})...")
    
    try:
        # 运行数据处理流水线
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
        
        # 检查是否包含融资融券相关列
        margin_columns = [col for col in df.columns if 'margin' in col.lower() or 'rzye' in col or 'rzmre' in col or 'rqye' in col]
        print(f"\n融资融券相关列: {margin_columns}")
        
        # 显示前几行数据
        print("\n前5行数据:")
        print(df.head())
        
        # 保存集成后的数据
        output_path = "integrated_features.parquet"
        df.to_parquet(output_path)
        print(f"\n集成后的数据已保存到: {output_path}")
        
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
    
    # 测试数据集成
    test_feature_integration()
    
    print("\n测试完成！")

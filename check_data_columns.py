
import os
import sys
import pandas as pd
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

PRICE_DIR = r'D:\iquant_data\data_v2\data_day1'

def main():
    # 检查几个数据文件的列名
    sample_files = [
        '20230101.parquet',
        '20240101.parquet',
        '20250101.parquet',
        '20230821.parquet'
    ]
    
    for filename in sample_files:
        filepath = os.path.join(PRICE_DIR, filename)
        if os.path.exists(filepath):
            print(f"\n文件: {filename}")
            df = pd.read_parquet(filepath)
            print("列名:")
            for col in df.columns:
                print(f"  - {col}")
            print(f"数据样本（前3行）:")
            print(df.head(3))
        else:
            print(f"\n文件不存在: {filename}")

if __name__ == "__main__":
    main()


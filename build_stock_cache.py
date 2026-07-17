
import os
import sys
import pandas as pd
import pickle
from datetime import datetime, timedelta
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, 'results_duobao')
PRICE_DIR = r'D:\iquant_data\data_v2\data_day1'

def main():
    print("开始构建股票历史数据缓存...")
    
    # 首先获取所有交易数据文件的日期列表
    all_files = os.listdir(PRICE_DIR)
    date_files = [f for f in all_files if f.endswith('.parquet')]
    dates = sorted([int(f.replace('.parquet', '')) for f in date_files])
    
    print(f"找到 {len(dates)} 个交易日数据文件")
    print(f"日期范围: {dates[0]} 到 {dates[-1]}")
    
    # 构建股票出现日期字典
    stock_dates = {}
    error_files = []
    
    print("\n开始扫描数据...")
    for i, date_int in enumerate(dates):
        if (i + 1) % 100 == 0:
            print(f"  进度: {i+1}/{len(dates)} ({((i+1)/len(dates)*100):.1f}%)")
        
        try:
            df = pd.read_parquet(os.path.join(PRICE_DIR, f"{date_int}.parquet"))
            if 'ts_code' in df.columns:
                for ts_code in df['ts_code'].unique():
                    if ts_code not in stock_dates:
                        stock_dates[ts_code] = []
                    stock_dates[ts_code].append(date_int)
        except Exception as e:
            error_files.append((date_int, str(e)))
            continue
    
    # 对每个股票的日期进行排序
    for ts_code in stock_dates:
        stock_dates[ts_code].sort()
    
    print(f"\n扫描完成！共处理 {len(stock_dates)} 只股票")
    
    if error_files:
        print(f"\n有 {len(error_files)} 个文件处理失败:")
        for date_int, error in error_files[:10]:
            print(f"  - {date_int}.parquet: {error}")
        if len(error_files) > 10:
            print(f"  ... 还有 {len(error_files)-10} 个")
    
    # 保存缓存
    cache_file = os.path.join(BASE_DIR, 'trade_stock_dates_cache.pkl')
    with open(cache_file, 'wb') as f:
        pickle.dump(stock_dates, f)
    
    print(f"\n缓存已保存至: {cache_file}")

if __name__ == "__main__":
    main()


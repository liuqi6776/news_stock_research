
import os
import sys
import pandas as pd
import pickle
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, 'results_duobao')
PRICE_DIR = r'D:\iquant_data\data_v2\data_day1'

def main():
    print("开始构建交易数据中股票的历史缓存...")
    
    # 首先读取预加载的交易数据，获取所有股票代码
    trades_csv = os.path.join(OUTPUT_DIR, 'preloaded_trades.csv')
    if not os.path.exists(trades_csv):
        print("未找到预加载文件")
        return
    
    trades_df = pd.read_csv(trades_csv)
    target_stocks = set(trades_df['ts_code'].unique())
    print(f"交易数据中包含 {len(target_stocks)} 只股票")
    
    # 获取所有日期文件
    all_files = os.listdir(PRICE_DIR)
    date_files = [f for f in all_files if f.endswith('.parquet')]
    dates = sorted([int(f.replace('.parquet', '')) for f in date_files])
    
    print(f"找到 {len(dates)} 个交易日数据文件")
    
    # 构建股票出现日期字典
    stock_dates = {}
    
    print("\n开始扫描数据...")
    for i, date_int in enumerate(dates):
        if (i + 1) % 100 == 0:
            print(f"  进度: {i+1}/{len(dates)} ({((i+1)/len(dates)*100):.1f}%)")
        
        try:
            df = pd.read_parquet(os.path.join(PRICE_DIR, f"{date_int}.parquet"))
            if 'ts_code' in df.columns:
                # 只处理目标股票
                for ts_code in df['ts_code'].unique():
                    if ts_code in target_stocks:
                        if ts_code not in stock_dates:
                            stock_dates[ts_code] = []
                        stock_dates[ts_code].append(date_int)
        except Exception as e:
            continue
    
    # 对每个股票的日期进行排序
    for ts_code in stock_dates:
        stock_dates[ts_code].sort()
    
    print(f"\n扫描完成！共处理 {len(stock_dates)} 只股票")
    
    # 保存缓存
    cache_file = os.path.join(OUTPUT_DIR, 'trade_stock_dates_cache.pkl')
    with open(cache_file, 'wb') as f:
        pickle.dump(stock_dates, f)
    
    print(f"\n缓存已保存至: {cache_file}")

if __name__ == "__main__":
    main()


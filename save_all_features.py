import os
import sys
import pandas as pd
import numpy as np
import joblib
from tqdm import tqdm
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from infra_data.storage import DataStorage

DATA_DIR = r'D:\iquant_data\data_v2'
PRICE_DIR = os.path.join(DATA_DIR, 'data_day1')
RANK_DIR  = os.path.join(DATA_DIR, 'ths_rank1')
CHIP_DIR  = os.path.join(DATA_DIR, 'cyq1')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, 'new_idea')

def get_all_dates():
    return sorted([f.replace('.parquet', '') for f in os.listdir(PRICE_DIR) if f.endswith('.parquet')])

def load_features_for_date(date_str):
    p_rank = os.path.join(RANK_DIR, f"{date_str}.parquet")
    p_chip = os.path.join(CHIP_DIR, f"{date_str}.parquet")
    p_price = os.path.join(PRICE_DIR, f"{date_str}.parquet")
    
    if not all(os.path.exists(p) for p in [p_rank, p_chip, p_price]):
        return None
    
    rank_df = pd.read_parquet(p_rank)
    rank_df['hot_rank_pct'] = rank_df['hot'].rank(pct=True)
    
    chip_df = pd.read_parquet(p_chip)
    chip_df['chip_concentration'] = (chip_df['cost_85pct'] - chip_df['cost_15pct']) / (chip_df['cost_50pct'] + 1e-8)
    
    price_df = pd.read_parquet(p_price, columns=['ts_code', 'close', 'pct_chg', 'amount', 'vol', 'pre_close'])
    
    df = pd.merge(rank_df[['ts_code', 'hot_rank_pct']], price_df, on='ts_code')
    df = pd.merge(df, chip_df[['ts_code', 'chip_concentration', 'winner_rate']], on='ts_code')
    
    return df

def load_news_data(start_date, end_date, dates_list):
    storage = DataStorage()
    valid_dates = pd.Series([pd.to_datetime(d) for d in dates_list if start_date <= d <= end_date]).sort_values()
    news_market_df, news_stock_sector_df = storage.load_news_data(start_date, end_date, valid_dates)
    
    if not news_market_df.empty:
        news_market_df['trade_date'] = news_market_df['trade_date'].dt.strftime('%Y%m%d')
    if not news_stock_sector_df.empty:
        news_stock_sector_df['trade_date'] = news_stock_sector_df['trade_date'].dt.strftime('%Y%m%d')
    
    return news_market_df, news_stock_sector_df

def save_all_features_with_prob():
    dates = get_all_dates()
    
    TEST_START = '20220101'
    TEST_END = '20260324'
    
    test_dates = [d for d in dates if TEST_START <= d <= TEST_END]
    all_news_dates = [d for d in dates if '20200101' <= d <= TEST_END]
    
    print(f"总测试日期数: {len(test_dates)}")
    
    model_path = os.path.join(BASE_DIR, 'daily_dragon_news_model.joblib')
    if not os.path.exists(model_path):
        print(f"模型不存在: {model_path}")
        return None
    
    model, feats = joblib.load(model_path)
    print(f"已加载模型: {model_path}")
    
    print(f"\n正在加载新闻数据...")
    news_market_df, news_stock_sector_df = load_news_data('20200101', TEST_END, all_news_dates)
    print(f"  news_market_df: {len(news_market_df)} 行")
    print(f"  news_stock_sector_df: {len(news_stock_sector_df)} 行")
    
    all_features = []
    
    print(f"\n正在生成所有日期的特征和 prob...")
    for i in tqdm(range(len(test_dates)-2)):
        d_t = test_dates[i]
        d_t1 = test_dates[i+1]
        d_t2 = test_dates[i+2]
        
        df_t = load_features_for_date(d_t)
        if df_t is None:
            continue
        
        df_t['date_t'] = d_t
        df_t['date_t2'] = d_t2
        df_t['trade_date'] = d_t1
        
        if not news_market_df.empty:
            df_t = pd.merge(df_t, news_market_df, on='trade_date', how='left')
        else:
            df_t['news_market_impact'] = 0.0
            
        if not news_stock_sector_df.empty:
            df_t = pd.merge(df_t, news_stock_sector_df, on=['trade_date', 'ts_code'], how='left')
        else:
            df_t['news_stock_impact'] = 0.0
            df_t['news_sector_impact'] = 0.0
            
        df_t[['news_market_impact', 'news_stock_impact', 'news_sector_impact']] = \
            df_t[['news_market_impact', 'news_stock_impact', 'news_sector_impact']].fillna(0.0)
        
        X = df_t[feats].fillna(0)
        
        try:
            df_t['prob'] = model.predict_proba(X)[:, 1]
        except Exception:
            continue
        
        all_features.append(df_t)
    
    all_features_df = pd.concat(all_features, ignore_index=True)
    output_path = os.path.join(OUTPUT_DIR, 'all_features_with_prob.csv')
    all_features_df.to_csv(output_path, index=False)
    print(f"\n✓ 已保存所有特征和 prob 到: {output_path}")
    print(f"  总记录数: {len(all_features_df)}")
    return all_features_df

if __name__ == "__main__":
    save_all_features_with_prob()

import os
import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
from tqdm import tqdm
import sys

# Ensure this script can import quant_system_v2 modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from infra_data.storage import DataStorage

DATA_DIR = r'D:\iquant_data\data_v2'
PRICE_DIR = os.path.join(DATA_DIR, 'data_day1')
RANK_DIR  = os.path.join(DATA_DIR, 'ths_rank1')
CHIP_DIR  = os.path.join(DATA_DIR, 'cyq1')

def train_daily_model(start_date='20220101', end_date='20231231', model_path=None):
    dates = sorted([f.replace('.parquet', '') for f in os.listdir(PRICE_DIR) if f.endswith('.parquet')])
    train_dates = [d for d in dates if start_date <= d <= end_date]
    if len(train_dates) < 2:
        print(f"Warning: Not enough training dates between {start_date} and {end_date}")
        return None, None
        
    storage = DataStorage()
    valid_dates_series = pd.Series([pd.to_datetime(d) for d in train_dates]).sort_values()
    news_market_df, news_stock_sector_df = storage.load_news_data(start_date, end_date, valid_dates_series)
    
    if not news_market_df.empty:
        news_market_df['trade_date'] = news_market_df['trade_date'].dt.strftime('%Y%m%d')
    if not news_stock_sector_df.empty:
        news_stock_sector_df['trade_date'] = news_stock_sector_df['trade_date'].dt.strftime('%Y%m%d')

    all_data = []
    
    for i in tqdm(range(len(train_dates)-1), desc=f"Loading T+1 Training [{start_date}-{end_date}]", leave=False):
        d_curr, d_next = train_dates[i], train_dates[i+1]
        
        p_rank = os.path.join(RANK_DIR, f"{d_curr}.parquet")
        p_chip = os.path.join(CHIP_DIR, f"{d_curr}.parquet")
        p_price = os.path.join(PRICE_DIR, f"{d_curr}.parquet")
        
        if not all(os.path.exists(p) for p in [p_rank, p_chip, p_price]):
            continue
            
        rank_df = pd.read_parquet(p_rank)
        # Process ranking into percentile
        rank_df['hot_rank_pct'] = rank_df['hot'].rank(pct=True)
        
        chip_df = pd.read_parquet(p_chip)
        chip_df['chip_concentration'] = (chip_df['cost_85pct'] - chip_df['cost_15pct']) / (chip_df['cost_50pct'] + 1e-8)
        
        price_df = pd.read_parquet(p_price, columns=['ts_code', 'close', 'pct_chg', 'amount', 'vol'])
        
        # Next day features for labelling
        p_next = os.path.join(PRICE_DIR, f"{d_next}.parquet")
        if not os.path.exists(p_next):
            continue
        next_df = pd.read_parquet(p_next, columns=['ts_code', 'open', 'high', 'close'])
        
        # Label: If MAX intraday return > 4%, we classify as 1
        next_df['label_ret'] = (next_df['high'] / (next_df['open'] + 1e-8)) - 1
        next_df['label'] = (next_df['label_ret'] > 0.04).astype(int)
        
        m = pd.merge(rank_df[['ts_code', 'hot_rank_pct']], price_df, on='ts_code')
        m = pd.merge(m, chip_df[['ts_code', 'chip_concentration', 'winner_rate']], on='ts_code')
        m = pd.merge(m, next_df[['ts_code', 'label']], on='ts_code')
        
        # Align trade_date to d_next to perfectly match T+0 zero-delay mapping
        m['trade_date'] = d_next
        
        all_data.append(m)
        
    df = pd.concat(all_data, ignore_index=True)
    
    # Merge Clean News
    if not news_market_df.empty:
        df = pd.merge(df, news_market_df, on='trade_date', how='left')
    else:
        df['news_market_impact'] = 0.0
        
    # EXPLICITLY Drop Sector impact, ONLY merge Stock impact to avoid follower trap
    if not news_stock_sector_df.empty:
        df = pd.merge(df, news_stock_sector_df[['trade_date', 'ts_code', 'news_stock_impact']], on=['trade_date', 'ts_code'], how='left')
    else:
        df['news_stock_impact'] = 0.0
        
    df[['news_market_impact', 'news_stock_impact']] = df[['news_market_impact', 'news_stock_impact']].fillna(0.0)
    
    FEATURE_COLS = [
        'hot_rank_pct', 'chip_concentration', 'winner_rate',
        'news_market_impact', 'news_stock_impact' # NO SECTOR!
    ]
    
    X = df[FEATURE_COLS].fillna(0)
    y = df['label']
    
    print(f"\nTraining sample size: {len(X)}")
    print(f"Positive labels: {y.sum()} ({y.sum()/len(y):.2%})")
    print(f"Features used: {FEATURE_COLS}")
    
    model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric='auc',
        n_jobs=-1,
        tree_method='hist'
    )
    
    print(f"Training XGBoost [{start_date} to {end_date}], samples={len(X)}")
    model.fit(X, y)
    
    if model_path:
        output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), model_path)
        joblib.dump((model, FEATURE_COLS), output_path)
        print(f"Model saved to {output_path}")
        
    return model, FEATURE_COLS

if __name__ == "__main__":
    train_daily_model(model_path='daily_t1_model.joblib')

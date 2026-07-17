import os
import pandas as pd
import numpy as np
import xgboost as xgb
from tqdm import tqdm
import joblib
import sys
from dotenv import load_dotenv

# Ensure environment variables are loaded
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(ROOT_DIR)
load_dotenv(os.path.join(ROOT_DIR, ".env"))

from infra_data.storage import DataStorage

DATA_DIR = r'D:\iquant_data\data_v2'
PRICE_DIR = os.path.join(DATA_DIR, 'data_day1')
RANK_DIR  = os.path.join(DATA_DIR, 'ths_rank1')
CHIP_DIR  = os.path.join(DATA_DIR, 'cyq1')

def load_data_for_date_range(start_date, end_date, dates):
    storage = DataStorage()
    valid_dates_series = pd.Series([pd.to_datetime(d) for d in dates 
                                     if start_date <= d <= end_date]).sort_values()
    news_market_df, news_stock_sector_df = storage.load_news_data(start_date, end_date, valid_dates_series)
    
    if not news_market_df.empty:
        news_market_df['trade_date'] = news_market_df['trade_date'].dt.strftime('%Y%m%d')
    if not news_stock_sector_df.empty:
        news_stock_sector_df['trade_date'] = news_stock_sector_df['trade_date'].dt.strftime('%Y%m%d')
    
    return news_market_df, news_stock_sector_df

def load_options_features():
    print("[INFO] Loading Options PCR & QVIX features...")
    pcr_csv = r"D:\iquant_data\data_v2\qiquan\historical_pcr.csv"
    if not os.path.exists(pcr_csv):
        print(f"[WARNING] Options PCR CSV not found at: {pcr_csv}")
        return pd.DataFrame()
    try:
        df_pcr = pd.read_csv(pcr_csv)
        df_pcr['date'] = pd.to_datetime(df_pcr['date'])
        df_pcr['trade_date'] = df_pcr['date'].dt.strftime('%Y%m%d')
        df_pcr_clean = df_pcr[['trade_date', 'pcr_50', 'oi_pcr_50']].rename(columns={
            'pcr_50': 'opt_pcr_vol_50',
            'oi_pcr_50': 'opt_pcr_oi_50'
        })
    except Exception as e:
        print(f"[ERROR] Failed to load PCR: {e}")
        return pd.DataFrame()

    import akshare as ak
    try:
        df_qvix = ak.index_option_50etf_qvix()
        df_qvix['date'] = pd.to_datetime(df_qvix['date'])
        df_qvix['trade_date'] = df_qvix['date'].dt.strftime('%Y%m%d')
        df_qvix['opt_qvix_close'] = df_qvix['close']
        df_qvix['opt_qvix_ma'] = df_qvix['close'].rolling(20).mean()
        df_qvix['opt_qvix_std'] = df_qvix['close'].rolling(20).std()
        df_qvix['opt_qvix_zscore'] = (df_qvix['close'] - df_qvix['opt_qvix_ma']) / df_qvix['opt_qvix_std']
        df_qvix_clean = df_qvix[['trade_date', 'opt_qvix_close', 'opt_qvix_zscore']].fillna(0)
    except Exception as e:
        print(f"[ERROR] Failed to load QVIX: {e}")
        df_qvix_clean = pd.DataFrame()

    if df_qvix_clean.empty:
        return df_pcr_clean
    merged = pd.merge(df_pcr_clean, df_qvix_clean, on='trade_date', how='outer').sort_values('trade_date').reset_index(drop=True)
    merged[['opt_pcr_vol_50', 'opt_pcr_oi_50', 'opt_qvix_close', 'opt_qvix_zscore']] = \
        merged[['opt_pcr_vol_50', 'opt_pcr_oi_50', 'opt_qvix_close', 'opt_qvix_zscore']].ffill().fillna(0)
    return merged

def prepare_features_for_date(d_curr, d_next, news_market_df, news_stock_sector_df, options_df=None):
    p_rank = os.path.join(RANK_DIR, f"{d_curr}.parquet")
    p_chip = os.path.join(CHIP_DIR, f"{d_curr}.parquet")
    p_price = os.path.join(PRICE_DIR, f"{d_curr}.parquet")
    
    if not all(os.path.exists(p) for p in [p_rank, p_chip, p_price]):
        return None
    
    rank_df = pd.read_parquet(p_rank)
    rank_df['hot_rank_pct'] = rank_df['hot'].rank(pct=True)
    
    chip_df = pd.read_parquet(p_chip)
    chip_df['chip_concentration'] = (chip_df['cost_85pct'] - chip_df['cost_15pct']) / (chip_df['cost_50pct'] + 1e-8)
    
    price_df = pd.read_parquet(p_price, columns=['ts_code', 'close', 'pct_chg', 'amount', 'vol', 'pre_close'])
    
    df = pd.merge(rank_df[['ts_code', 'hot_rank_pct']], price_df, on='ts_code')
    df = pd.merge(df, chip_df[['ts_code', 'chip_concentration', 'winner_rate']], on='ts_code')
    df['trade_date'] = d_next
    
    if not news_market_df.empty:
        df = pd.merge(df, news_market_df, on='trade_date', how='left')
    else:
        df['news_market_impact'] = 0.0
        
    if not news_stock_sector_df.empty:
        df = pd.merge(df, news_stock_sector_df, on=['trade_date', 'ts_code'], how='left')
    else:
        df['news_stock_impact'] = 0.0
        df['news_sector_impact'] = 0.0
        
    df[['news_market_impact', 'news_stock_impact', 'news_sector_impact']] = \
        df[['news_market_impact', 'news_stock_impact', 'news_sector_impact']].fillna(0.0)
        
    if options_df is not None and not options_df.empty:
        opt_row = options_df[options_df['trade_date'] == d_curr]
        if not opt_row.empty:
            for col in ['opt_pcr_vol_50', 'opt_pcr_oi_50', 'opt_qvix_close', 'opt_qvix_zscore']:
                df[col] = float(opt_row[col].values[0])
        else:
            for col in ['opt_pcr_vol_50', 'opt_pcr_oi_50', 'opt_qvix_close', 'opt_qvix_zscore']:
                df[col] = 0.0
    else:
        for col in ['opt_pcr_vol_50', 'opt_pcr_oi_50', 'opt_qvix_close', 'opt_qvix_zscore']:
            df[col] = 0.0
            
    return df

def train_daily_dragon_model(train_start='20220101', train_end='20231231'):
    dates = sorted([f.replace('.parquet', '') for f in os.listdir(PRICE_DIR) if f.endswith('.parquet')])
    
    cache_path = r'D:\iquant_data\data_v2\qiquan\dragon_features_cache.parquet'
    if not os.path.exists(cache_path):
        print(f"[ERROR] Cache parquet not found at: {cache_path}")
        return None, None
        
    df_all = pd.read_parquet(cache_path)
    
    try:
        start_idx = dates.index(train_start)
        min_trade_date = dates[start_idx+1]
    except Exception:
        min_trade_date = train_start
        
    try:
        end_idx = dates.index(train_end)
        max_trade_date = dates[end_idx+1] if end_idx+1 < len(dates) else dates[-1]
    except Exception:
        max_trade_date = train_end
        
    df_train = df_all[(df_all['trade_date'] >= min_trade_date) & (df_all['trade_date'] <= max_trade_date)]
    
    feats = [
        'hot_rank_pct', 'pct_chg', 'amount', 'vol', 'chip_concentration', 'winner_rate', 
        'news_market_impact', 'news_stock_impact', 'news_sector_impact',
        'opt_pcr_vol_50', 'opt_pcr_oi_50', 'opt_qvix_close', 'opt_qvix_zscore'
    ]
    
    X = df_train[feats].fillna(0)
    y = df_train['label']
    
    # Balance classes
    pos = df_train[y == 1]
    neg = df_train[y == 0].sample(min(len(pos)*2, len(df_train)-len(pos)), random_state=42)
    train_bal = pd.concat([pos, neg])
    
    model = xgb.XGBClassifier(
        n_estimators=200, 
        max_depth=5, 
        learning_rate=0.08, 
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42, 
        n_jobs=-1, 
        tree_method='hist', 
        eval_metric='logloss'
    )
    model.fit(train_bal[feats], train_bal['label'])
    
    print(f"[SUCCESS] Model training complete! Pos class samples: {len(pos)}")
    return model, feats

if __name__ == "__main__":
    model, feats = train_daily_dragon_model(train_start='20220101', train_end='20231231')
    if model is not None:
        joblib.dump((model, feats), 'daily_dragon_news_model.joblib')
        print("[SUCCESS] Static model successfully trained and saved to daily_dragon_news_model.joblib!")

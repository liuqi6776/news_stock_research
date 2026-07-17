import os
import pandas as pd
import datetime
import sys
import joblib
import json
import warnings

# Ensure relative imports work
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from train_model import train_daily_model
from processing.news_processor import load_and_process_news

DATA_DIR  = r'D:\iquant_data\data_v2'
PRICE_DIR = os.path.join(DATA_DIR, 'data_day1')
RANK_DIR  = os.path.join(DATA_DIR, 'ths_rank1')
CHIP_DIR  = os.path.join(DATA_DIR, 'cyq1')
OTHER_DIR = os.path.join(DATA_DIR, 'other_day1')
NEWS_MAJOR_DIR = os.path.join(DATA_DIR, 'news_major1')
BASE_PATH = r'C:\Users\liuqi\quant_system_v2' # Adjust if needed

def run_daily_prediction():
    """
    1. Determine latest available data date.
    2. Load News JSONs from news_major and process via industry/concept mapping.
    3. Retrain/Update model (Expanding WFO).
    4. Apply Filters (No-688, Cap < 500亿).
    5. Output top 3 picks.
    """
    # Find most recent price data date
    all_dates = sorted([f.replace('.parquet','') for f in os.listdir(PRICE_DIR) if f.endswith('.parquet')])
    if not all_dates:
        print("Error: No data found in PRICE_DIR.")
        return
    
    latest_date = all_dates[-1]
    print(f"--- Running Prediction for Next Trading Day based on: {latest_date} ---")
    
    # 1. Update/Train Model (Expanding Window)
    model_file = os.path.join(os.path.dirname(__file__), 'daily_t1_model.joblib')
    print(f"Step 1: Updating model with data up to {latest_date}...")
    model, feats = train_daily_model('20220101', latest_date, model_path=model_file)
    
    # 2. Process News Data (Mapping JSON to Stocks)
    print(f"Step 2: Processing News JSONs from news_major...")
    industry_map_path = os.path.join(BASE_PATH, 'stock_industry_map_cached.parquet')
    
    # We treat latest_date as the target for our prediction's perspective
    # load_and_process_news returns (market_df, stock_sector_df)
    # Using a 1-day window for the latest news
    m_news, s_news = load_and_process_news(
        NEWS_MAJOR_DIR, 
        start_date=latest_date, 
        end_date=latest_date,
        industry_map_path=industry_map_path
    )
    
    # 3. Extract Features for latest_date
    p_rank = os.path.join(RANK_DIR, f"{latest_date}.parquet")
    p_chip = os.path.join(CHIP_DIR, f"{latest_date}.parquet")
    p_price= os.path.join(PRICE_DIR, f"{latest_date}.parquet")
    p_other= os.path.join(OTHER_DIR, f"{latest_date}.parquet")
    
    if not all(os.path.exists(p) for p in [p_rank, p_chip, p_price, p_other]):
        print(f"Error: Missing feature files for {latest_date}.")
        return
    
    rank_df = pd.read_parquet(p_rank)
    rank_df['hot_rank_pct'] = rank_df['hot'].rank(pct=True)
    
    chip_df = pd.read_parquet(p_chip)
    chip_df['chip_concentration'] = (chip_df['cost_85pct'] - chip_df['cost_15pct']) / (chip_df['cost_50pct'] + 1e-8)
    
    price_df = pd.read_parquet(p_price, columns=['ts_code', 'close', 'pct_chg', 'pre_close'])
    other_df = pd.read_parquet(p_other, columns=['ts_code', 'circ_mv'])
    
    # Merge Features
    df = pd.merge(rank_df[['ts_code', 'hot_rank_pct']], price_df, on='ts_code')
    df = pd.merge(df, chip_df[['ts_code', 'chip_concentration', 'winner_rate']], on='ts_code')
    df = pd.merge(df, other_df, on='ts_code', how='left')
    
    # 4. Filter Universe (Optimized Mid-Cap Dragon)
    df = df[~df['ts_code'].str.startswith('688')]
    df = df[df['circ_mv'] <= 500000] # 500亿 Ceiling
    
    # 5. Integrate News Impacts
    if not m_news.empty:
        # Take the average impact if multiple news items exist
        df['news_market_impact'] = m_news['news_market_impact'].mean()
    else:
        df['news_market_impact'] = 0.0
        
    if not s_news.empty:
        # s_news contains 'ts_code', 'news_stock_impact', 'news_sector_impact'
        df = pd.merge(df, s_news[['ts_code', 'news_stock_impact']], on='ts_code', how='left')
    else:
        df['news_stock_impact'] = 0.0
    
    df[['news_market_impact', 'news_stock_impact']] = df[['news_market_impact', 'news_stock_impact']].fillna(0.0)
    
    # 6. Prediction
    X = df[feats].fillna(0)
    df['prob'] = model.predict_proba(X)[:, 1]
    
    # 7. Output Result
    top_picks = df[df['prob'] > 0.8].sort_values('prob', ascending=False).head(3)
    if top_picks.empty:
        top_picks = df.sort_values('prob', ascending=False).head(3)
        
    print("\n" + "="*60)
    print(f"🔥 TOP DRAGON PICKS FOR NEXT TRADING DAY ({latest_date} context) 🔥")
    print("="*60)
    if top_picks.empty:
        print("No high-probability signals found.")
    else:
        for i, (idx, row) in enumerate(top_picks.iterrows()):
            print(f"Rank {i+1}: {row['ts_code']} | Prob: {row['prob']:.4f} | MktCap: {row['circ_mv']/10000:.2f}亿 | NewsImpact: {row['news_stock_impact']:.2f}")
    print("="*60)
    print("Rules: T+1, Buy Open, Target +4% Profit, STAR market excluded.")

if __name__ == "__main__":
    run_daily_prediction()

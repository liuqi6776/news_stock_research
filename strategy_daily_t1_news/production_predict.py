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

# Import specific news processor
from processing.news_processor import load_and_process_news

DATA_DIR  = r'D:\iquant_data\data_v2'
PRICE_DIR = os.path.join(DATA_DIR, 'data_day1')
RANK_DIR  = os.path.join(DATA_DIR, 'ths_rank1')
CHIP_DIR  = os.path.join(DATA_DIR, 'cyq1')
OTHER_DIR = os.path.join(DATA_DIR, 'other_day1')
NEWS_MAJOR_DIR = os.path.join(DATA_DIR, 'news_major1')
BASE_PATH = r'C:\Users\liuqi\quant_system_v2'

def run_prediction_for_date(target_date, model_file='daily_t1_model.joblib'):
    """
    Optimized Production Prediction:
    1. Uses FIXED pre-trained model (no retraining).
    2. Predicts for SPECIFIED target_date.
    3. Incorporates News JSON (stock/sector mapping).
    """
    print(f"\n--- [T+1 News Strategy] Prediction for Date: {target_date} ---")
    
    # 1. Load Pre-trained Model
    if not os.path.exists(model_file):
        print(f"Error: Model file {model_file} not found.")
        return
    
    print(f"Loading pre-trained model: {model_file}...")
    model, feats = joblib.load(model_file)
    
    # 2. Process News JSONs
    print(f"Processing news JSONs for {target_date}...")
    industry_map_path = os.path.join(BASE_PATH, 'stock_industry_map_cached.parquet')
    
    # Fuzzy Date Matching: If no news on target_date, look back up to 10 days
    m_news, s_news = pd.DataFrame(), pd.DataFrame()
    for lookback in range(0, 11):
        check_date = (pd.to_datetime(target_date) - pd.Timedelta(days=lookback)).strftime('%Y%m%d')
        m_news, s_news = load_and_process_news(
            NEWS_MAJOR_DIR, 
            start_date=check_date, 
            end_date=check_date,
            industry_map_path=industry_map_path
        )
        if not m_news.empty or not s_news.empty:
            print(f"Found latest available news from: {check_date} (Lookback: {lookback} days)")
            break
            
    if m_news.empty and s_news.empty:
        print("Warning: No news found in the last 10 days. Results will have 0 news impact.")
    
    # 3. Load Feature Files
    p_rank = os.path.join(RANK_DIR, f"{target_date}.parquet")
    p_chip = os.path.join(CHIP_DIR, f"{target_date}.parquet")
    p_price= os.path.join(PRICE_DIR, f"{target_date}.parquet")
    p_other= os.path.join(OTHER_DIR, f"{target_date}.parquet")
    
    if not all(os.path.exists(p) for p in [p_rank, p_chip, p_price, p_other]):
        print(f"Error: Required feature files for {target_date} are missing.")
        print("Please run data_fetcher.py first.")
        return
    
    rank_df = pd.read_parquet(p_rank)
    rank_df['hot_rank_pct'] = rank_df['hot'].rank(pct=True)
    
    chip_df = pd.read_parquet(p_chip)
    chip_df['chip_concentration'] = (chip_df['cost_85pct'] - chip_df['cost_15pct']) / (chip_df['cost_50pct'] + 1e-8)
    
    price_df = pd.read_parquet(p_price, columns=['ts_code', 'close', 'pct_chg', 'pre_close'])
    other_df = pd.read_parquet(p_other, columns=['ts_code', 'circ_mv'])
    
    # Merge
    df = pd.merge(rank_df[['ts_code', 'hot_rank_pct']], price_df, on='ts_code')
    df = pd.merge(df, chip_df[['ts_code', 'chip_concentration', 'winner_rate']], on='ts_code')
    df = pd.merge(df, other_df, on='ts_code', how='left')
    
    # Apply Optimized Filters
    df = df[~df['ts_code'].str.startswith('688')] # No STAR
    df = df[df['circ_mv'] <= 500000] # <= 500亿
    
    # News Impact Integration
    if not m_news.empty:
        df['news_market_impact'] = m_news['news_market_impact'].mean()
    else:
        df['news_market_impact'] = 0.0
        
    if not s_news.empty:
        df = pd.merge(df, s_news[['ts_code', 'news_stock_impact']], on='ts_code', how='left')
    else:
        df['news_stock_impact'] = 0.0
    
    df[['news_market_impact', 'news_stock_impact']] = df[['news_market_impact', 'news_stock_impact']].fillna(0.0)
    
    # 4. Predict
    X = df[feats].fillna(0)
    df['prob'] = model.predict_proba(X)[:, 1]
    
    # 5. Output
    top_picks = df[df['prob'] > 0.8].sort_values('prob', ascending=False).head(10)
    if top_picks.empty:
        top_picks = df.sort_values('prob', ascending=False).head(10)
        
    print("\n" + "="*60)
    print(f"--- FINAL DRAGON SELECTIONS (TOP 10) FOR {target_date} ---")
    print("="*60)
    for i, (idx, row) in enumerate(top_picks.iterrows()):
        print(f"Rank {i+1}: {row['ts_code']} | Score: {row['prob']:.4f} | MktCap: {row['circ_mv']/10000:.2f}亿 | NewsImpact: {row['news_stock_impact']:.2f}")
    print("="*60)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        date = sys.argv[1]
    else:
        date = datetime.datetime.now().strftime("%Y%m%d")
    
    run_prediction_for_date(date)

import os
import sys
import json
import pandas as pd
import numpy as np
from tqdm import tqdm
from datetime import datetime

# Adjust paths if necessary based on your structure
DATA_DIR  = r'D:\iquant_data\data_v2'
PRICE_DIR = os.path.join(DATA_DIR, 'data_day1')
RANK_DIR  = os.path.join(DATA_DIR, 'ths_rank1')
CHIP_DIR  = os.path.join(DATA_DIR, 'cyq1')
OTHER_DIR = os.path.join(DATA_DIR, 'other_day1')
MAJOR1_DIR = os.path.join(DATA_DIR, 'news_major1')
MAJOR_DIR  = os.path.join(DATA_DIR, 'news_major')

def load_news_from_json(news_dir):
    """
    Load specialized JSON news into a raw DataFrame preserving article_date.
    """
    if not os.path.exists(news_dir):
        return pd.DataFrame()
        
    stock_records = []
    for filename in os.listdir(news_dir):
        if not filename.endswith('.json'): continue
        filepath = os.path.join(news_dir, filename)
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception: 
            continue
            
        date_str = data.get("article_date", "")
        if not date_str: continue
            
        trade_date = pd.to_datetime(date_str).strftime('%Y%m%d')
        
        for s in data.get("stocks", []):
            code = s.get("stock_code")
            if not code: continue
            # Convert to internal ts_code format
            ts_code = f"{code}.SH" if code.startswith('6') else \
                      f"{code}.SZ" if code.startswith(('0','3')) else \
                      f"{code}.BJ" if code.startswith(('4','8')) else code
            stock_records.append({
                'article_date': trade_date, 
                'ts_code': ts_code, 
                'news_stock_impact': float(s.get("impact", 0.0))
            })
            
        for s in data.get("sectors", []):
            # Currently ignoring sector processing to speed up, if needed later we add.
            pass
            
    if not stock_records:
        return pd.DataFrame()
        
    df = pd.DataFrame(stock_records)
    # Aggregate in case there are multiple news items for the same stock on the same day
    return df.groupby(['article_date', 'ts_code'], as_index=False).mean()

def build_mega_dataset(start_date='20180101', end_date='20261231', out_path='data/super_dataset.parquet'):
    print(f"Building Dataset from {start_date} to {end_date}...")
    
    dates = sorted([f.replace('.parquet', '') for f in os.listdir(PRICE_DIR) if f.endswith('.parquet')])
    valid_dates = [d for d in dates if start_date <= d <= end_date]
    if len(valid_dates) < 3:
        raise ValueError("Not enough dates found!")
        
    print("Loading News Data...")
    df_major = load_news_from_json(MAJOR_DIR)      # Post-market news
    df_major1 = load_news_from_json(MAJOR1_DIR)    # Pre-market news
    
    # We will build dictionary mapping to avoid slow merges inside loop
    # dict key: (article_date, ts_code) -> score
    lookup_major = df_major.set_index(['article_date', 'ts_code'])['news_stock_impact'].to_dict() if not df_major.empty else {}
    lookup_major1 = df_major1.set_index(['article_date', 'ts_code'])['news_stock_impact'].to_dict() if not df_major1.empty else {}

    all_data = []
    
    # We iterate until len-2 because we need T+1 and T+2 for T+1 backtesting labels
    for i in tqdm(range(len(valid_dates)-2), desc="Processing cross-sections"):
        t_curr = valid_dates[i]       # Base signal day (T)
        t_buy  = valid_dates[i+1]     # Execution day   (T+1)
        t_sell = valid_dates[i+2]     # Exit day        (T+2)
        
        # 1. Base files for day T
        p_rank = os.path.join(RANK_DIR, f"{t_curr}.parquet")
        p_chip = os.path.join(CHIP_DIR, f"{t_curr}.parquet")
        p_price = os.path.join(PRICE_DIR, f"{t_curr}.parquet")
        p_other = os.path.join(OTHER_DIR, f"{t_curr}.parquet")
        
        if not all(os.path.exists(p) for p in [p_rank, p_chip, p_price, p_other]):
            continue
            
        rank_df = pd.read_parquet(p_rank)
        chip_df = pd.read_parquet(p_chip)
        price_df = pd.read_parquet(p_price, columns=['ts_code', 'close', 'pct_chg', 'amount', 'vol'])
        other_df = pd.read_parquet(p_other, columns=['ts_code', 'circ_mv'])
        
        # Merge basic cross-section T
        df_sec = pd.merge(price_df, rank_df[['ts_code', 'hot']], on='ts_code', how='left')
        df_sec = pd.merge(df_sec, chip_df[['ts_code', 'cost_15pct', 'cost_50pct', 'cost_85pct', 'winner_rate']], on='ts_code', how='left')
        df_sec = pd.merge(df_sec, other_df, on='ts_code', how='left')
        
        # 2. Add features
        df_sec['hot_rank_pct'] = df_sec['hot'].rank(pct=True)
        df_sec['chip_concentration'] = (df_sec['cost_85pct'] - df_sec['cost_15pct']) / (df_sec['cost_50pct'] + 1e-8)
        
        df_sec['trade_date'] = t_curr
        df_sec['buy_date'] = t_buy
        df_sec['sell_date'] = t_sell
        
        # 3. Add News Features
        # Rule: news-major (Post market) from day T
        # Rule: news-major1 (Pre market) from day T+1
        df_sec['news_major_impact'] = df_sec['ts_code'].apply(lambda c: lookup_major.get((t_curr, c), 0.0))
        df_sec['news_major1_impact'] = df_sec['ts_code'].apply(lambda c: lookup_major1.get((t_buy, c), 0.0))
        
        # 4. Add Future Execution Price & Labels
        buy_price_df = pd.read_parquet(os.path.join(PRICE_DIR, f"{t_buy}.parquet"), columns=['ts_code', 'open', 'high', 'close', 'pre_close'])
        sell_price_df = pd.read_parquet(os.path.join(PRICE_DIR, f"{t_sell}.parquet"), columns=['ts_code', 'open', 'pre_close'])
        
        # Rename for clarity
        buy_price_df.rename(columns={'open': 'entry_open', 'high': 'entry_high', 'close': 'entry_close', 'pre_close': 'entry_pre_close'}, inplace=True)
        sell_price_df.rename(columns={'open': 'exit_open', 'pre_close': 'exit_pre_close'}, inplace=True)
        
        df_sec = pd.merge(df_sec, buy_price_df, on='ts_code', how='inner')
        df_sec = pd.merge(df_sec, sell_price_df, on='ts_code', how='inner')
        
        # Label 1: Intra-day return on T+1 (For training similar to old train_model logic: intra_high > 4%)
        df_sec['label_intra_ret'] = (df_sec['entry_high'] / (df_sec['entry_open'] + 1e-8)) - 1
        df_sec['label'] = (df_sec['label_intra_ret'] > 0.04).astype(int)
        
        # Label 2: True T+1 Open to T+2 Open Return (For backtesting and strict reality checks)
        # We explicitly calculate Intraday * Overnight gaps to avoid dividend splits artifacts
        entry_intraday_ret = df_sec['entry_close'] / (df_sec['entry_open'] + 1e-8)
        exit_overnight_ret = df_sec['exit_open'] / (df_sec['exit_pre_close'] + 1e-8)
        df_sec['true_t1_ret'] = (entry_intraday_ret * exit_overnight_ret) - 1
        
        # Keep only necessary memory footprint
        required_cols = [
            'trade_date', 'buy_date', 'sell_date', 'ts_code', 'circ_mv', 
            'hot_rank_pct', 'chip_concentration', 'winner_rate',
            'news_major_impact', 'news_major1_impact',
            'entry_open', 'entry_pre_close', 'entry_close', 'exit_open', 'exit_pre_close', 'label_intra_ret', 'true_t1_ret', 'label'
        ]
        
        df_sec = df_sec[required_cols]
        all_data.append(df_sec)
        
    print("\nConcatenating everything...")
    final_df = pd.concat(all_data, ignore_index=True)
    
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    print(f"Saving {len(final_df)} rows to {out_path}...")
    final_df.to_parquet(out_path, index=False)
    print("Done!")

if __name__ == "__main__":
    out_dir = os.path.dirname(os.path.abspath(__file__))
    out_file = os.path.join(out_dir, 'data', 'super_dataset.parquet')
    build_mega_dataset('20210101', '20261231', out_file)

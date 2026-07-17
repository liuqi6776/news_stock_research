import os
import pandas as pd
import numpy as np
import akshare as ak

def main():
    print("==================================================")
    print("     Building Features with Options Indicators     ")
    print("==================================================")

    # 1. Load base features
    feat_path = "research/study_005_1d_advanced/data/features_005.parquet"
    if not os.path.exists(feat_path):
        print(f"Error: Base feature file not found at {feat_path}")
        return
        
    print("Loading base features...")
    df = pd.read_parquet(feat_path)
    df['trade_date'] = df['trade_date'].astype(str)
    print(f"Loaded base features. Rows: {len(df)}")

    # 2. Fetch QVIX data
    print("Fetching 50ETF QVIX...")
    try:
        df_qvix = ak.index_option_50etf_qvix()
        df_qvix['date'] = pd.to_datetime(df_qvix['date'])
        df_qvix['trade_date'] = df_qvix['date'].dt.strftime('%Y%m%d')
        
        # Calculate features
        df_qvix['opt_qvix_close'] = df_qvix['close']
        df_qvix['opt_qvix_change'] = df_qvix['close'].pct_change()
        df_qvix['opt_qvix_ma'] = df_qvix['close'].rolling(20).mean()
        df_qvix['opt_qvix_std'] = df_qvix['close'].rolling(20).std()
        df_qvix['opt_qvix_zscore'] = (df_qvix['close'] - df_qvix['opt_qvix_ma']) / df_qvix['opt_qvix_std']
        
        # Clean columns
        df_qvix_clean = df_qvix[['trade_date', 'opt_qvix_close', 'opt_qvix_change', 'opt_qvix_zscore']]
        print(f"QVIX features calculated. Range: {df_qvix_clean['trade_date'].min()} to {df_qvix_clean['trade_date'].max()}")
    except Exception as e:
        print("Error fetching QVIX:", e)
        return

    # 3. Load historical PCR data
    pcr_path = r"D:\iquant_data\data_v2\qiquan\historical_pcr.csv"
    if not os.path.exists(pcr_path):
        print(f"Error: PCR file not found at {pcr_path}")
        return
        
    print("Loading Put-Call Ratio data...")
    df_pcr = pd.read_csv(pcr_path)
    df_pcr['date'] = pd.to_datetime(df_pcr['date'])
    df_pcr['trade_date'] = df_pcr['date'].dt.strftime('%Y%m%d')
    
    # Rename columns for clarity in model features
    df_pcr_clean = df_pcr[['trade_date', 'pcr_50', 'oi_pcr_50', 'pcr_300', 'oi_pcr_300']].rename(columns={
        'pcr_50': 'opt_pcr_vol_50',
        'oi_pcr_50': 'opt_pcr_oi_50',
        'pcr_300': 'opt_pcr_vol_300',
        'oi_pcr_300': 'opt_pcr_oi_300'
    })
    print(f"PCR features loaded. Range: {df_pcr_clean['trade_date'].min()} to {df_pcr_clean['trade_date'].max()}")

    # 4. Merge Options and PCR data with stock features
    print("Merging Options and PCR data with stock features...")
    df_merged = pd.merge(df, df_qvix_clean, on='trade_date', how='left')
    df_merged = pd.merge(df_merged, df_pcr_clean, on='trade_date', how='left')
    
    # Forward fill the options features per ts_code to make sure no gaps (e.g. if options data has holiday differences)
    print("Forward filling options features...")
    opt_cols = ['opt_qvix_close', 'opt_qvix_change', 'opt_qvix_zscore', 
                'opt_pcr_vol_50', 'opt_pcr_oi_50', 'opt_pcr_vol_300', 'opt_pcr_oi_300']
                
    # Since all stocks share the same trade date option features, we can sort by trade date,
    # fill options data, and then merge back, which is 100x faster than doing groupby(ts_code)!
    # Let's create a trade-date level options lookup and forward fill it, then merge!
    df_opts_unique = df_merged[['trade_date'] + opt_cols].drop_duplicates().sort_values('trade_date').reset_index(drop=True)
    df_opts_unique[opt_cols] = df_opts_unique[opt_cols].ffill().fillna(0)
    
    # Drop original options columns from df_merged and merge the filled unique ones
    df_merged = df_merged.drop(columns=opt_cols, errors='ignore')
    df_merged = pd.merge(df_merged, df_opts_unique, on='trade_date', how='left')
    
    # Fill remaining NaNs or Infs
    df_merged[opt_cols] = df_merged[opt_cols].fillna(0).replace([np.inf, -np.inf], 0)
    
    # Save new feature parquet
    out_path = "research/study_005_1d_advanced/data/features_005_options.parquet"
    print(f"Saving merged features to {out_path}...")
    df_merged.to_parquet(out_path)
    print(f"Successfully saved {len(df_merged)} rows to {out_path}")

if __name__ == "__main__":
    main()

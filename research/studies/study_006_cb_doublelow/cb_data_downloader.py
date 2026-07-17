import os
import time
import pandas as pd
import akshare as ak
import numpy as np

# Define paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
STUDY_DIR = SCRIPT_DIR
RESEARCH_DIR = os.path.dirname(os.path.dirname(STUDY_DIR))
CACHE_DIR = os.path.join(RESEARCH_DIR, 'cache')
RAW_DIR = os.path.join(CACHE_DIR, 'cb_raw')

os.makedirs(RAW_DIR, exist_ok=True)

def download_master_list():
    print("Fetching convertible bond master list from Eastmoney...")
    try:
        df_master = ak.bond_zh_cov()
        df_master.to_csv(os.path.join(CACHE_DIR, 'cb_master_list.csv'), index=False, encoding='utf-8-sig')
        print(f"Master list fetched successfully. Total bonds: {len(df_master)}")
        return df_master
    except Exception as e:
        print(f"Error fetching master list: {e}")
        if os.path.exists(os.path.join(CACHE_DIR, 'cb_master_list.csv')):
            print("Loading cached master list...")
            return pd.read_csv(os.path.join(CACHE_DIR, 'cb_master_list.csv'))
        else:
            raise e

def download_daily_valuations(df_master):
    # Fetch each bond's daily valuation history
    bond_codes = df_master['债券代码'].astype(str).str.zfill(6).tolist()
    total = len(bond_codes)
    
    print("Downloading daily valuation histories...")
    failed_codes = []
    
    for idx, code in enumerate(bond_codes, 1):
        raw_csv_path = os.path.join(RAW_DIR, f"{code}.csv")
        
        # Check if already cached
        if os.path.exists(raw_csv_path):
            if os.path.getsize(raw_csv_path) > 100: # Verify not empty/corrupted
                continue
                
        print(f"[{idx}/{total}] Downloading valuation history for bond {code}...")
        success = False
        retries = 3
        
        for attempt in range(retries):
            try:
                # Eastmoney valuation history details (dates, close, pure bond value, conversion value, premium, etc.)
                df_val = ak.bond_zh_cov_value_analysis(symbol=code)
                if not df_val.empty:
                    df_val.to_csv(raw_csv_path, index=False, encoding='utf-8-sig')
                    success = True
                    break
                else:
                    print(f"  Warning: Empty data returned for bond {code}.")
                    break
            except Exception as e:
                print(f"  Error on attempt {attempt+1}/{retries} for bond {code}: {e}")
                time.sleep(1.0)
                
        if not success and not os.path.exists(raw_csv_path):
            failed_codes.append(code)
            
    if failed_codes:
        print(f"Completed download. Failed to download {len(failed_codes)} codes: {failed_codes}")
    else:
        print("All downloads completed successfully!")

def compile_pit_data(df_master):
    print("Compiling raw files into unified Point-In-Time dataset...")
    
    # Create mapping dictionaries from master list
    df_master['债券代码_str'] = df_master['债券代码'].astype(str).str.zfill(6)
    
    stock_code_map = df_master.set_index('债券代码_str')['正股代码'].to_dict()
    stock_name_map = df_master.set_index('债券代码_str')['正股简称'].to_dict()
    rating_map = df_master.set_index('债券代码_str')['信用评级'].to_dict()
    scale_map = df_master.set_index('债券代码_str')['发行规模'].to_dict()
    listing_map = df_master.set_index('债券代码_str')['上市时间'].to_dict()
    name_map = df_master.set_index('债券代码_str')['债券简称'].to_dict()
    
    all_dfs = []
    raw_files = [f for f in os.listdir(RAW_DIR) if f.endswith('.csv')]
    
    for idx, filename in enumerate(raw_files, 1):
        code = filename.replace('.csv', '')
        file_path = os.path.join(RAW_DIR, filename)
        
        try:
            df = pd.read_csv(file_path)
            if df.empty or '日期' not in df.columns:
                continue
                
            # Rename columns
            df = df.rename(columns={
                '日期': 'trade_date',
                '收盘价': 'close',
                '纯债价值': 'pure_bond_value',
                '转股价值': 'convert_value',
                '纯债溢价率': 'pure_bond_premium',
                '转股溢价率': 'premium'
            })
            
            # Format trade_date to datetime
            df['trade_date'] = pd.to_datetime(df['trade_date'])
            
            # Keep rows with valid date and drop rows with empty values (e.g. final delisted row which Eastmoney leaves empty)
            df = df.dropna(subset=['trade_date', 'close', 'premium'])
            
            if df.empty:
                continue
                
            # Add metadata columns
            df['ts_code'] = code
            df['name'] = name_map.get(code, 'Unknown')
            df['stock_code'] = str(stock_code_map.get(code, ''))
            df['stock_name'] = stock_name_map.get(code, 'Unknown')
            df['rating'] = rating_map.get(code, 'Unknown')
            df['issue_size'] = pd.to_numeric(scale_map.get(code, np.nan), errors='coerce')
            df['list_date'] = pd.to_datetime(listing_map.get(code, pd.NaT))
            
            all_dfs.append(df)
            
        except Exception as e:
            print(f"Error processing file {filename}: {e}")
            
    if not all_dfs:
        print("No valid bond price files found. Aborting compile.")
        return
        
    df_pit = pd.concat(all_dfs, ignore_index=True)
    
    # Sort by date and code
    df_pit = df_pit.sort_values(['trade_date', 'ts_code']).reset_index(drop=True)
    
    # Output to parquet (efficient storage) and CSV (debugging)
    parquet_path = os.path.join(CACHE_DIR, 'cb_pit_daily.parquet')
    csv_path = os.path.join(CACHE_DIR, 'cb_pit_daily.csv')
    
    df_pit.to_parquet(parquet_path, index=False)
    # Save a small subset or full to CSV for manual validation
    df_pit.to_csv(csv_path, index=False, encoding='utf-8-sig')
    
    print(f"Compilation complete!")
    print(f"  Unified rows: {len(df_pit)}")
    print(f"  Parquet saved to: {parquet_path}")
    print(f"  CSV saved to: {csv_path}")
    
def main():
    df_master = download_master_list()
    download_daily_valuations(df_master)
    compile_pit_data(df_master)

if __name__ == '__main__':
    main()

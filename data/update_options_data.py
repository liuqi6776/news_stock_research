import akshare as ak
import pandas as pd
import numpy as np
import time
import os
import sys

def main():
    print("==================================================")
    print("   Incremental Options PCR Data Sync Running      ")
    print("==================================================")

    # 1. Paths Configuration
    csv_dir = r"D:\iquant_data\data_v2\qiquan"
    csv_path = os.path.join(csv_dir, "historical_pcr.csv")
    
    os.makedirs(csv_dir, exist_ok=True)
    
    # 2. Get Trading Days from SSE Index
    print("[INFO] Fetching SSE Index trading dates...")
    try:
        df_index = ak.stock_zh_index_daily(symbol="sh000001")
        df_index['date'] = pd.to_datetime(df_index['date'])
        
        # Filter trading dates from 2022-01-01 to today
        today_str = pd.Timestamp.now().strftime('%Y-%m-%d')
        df_filtered = df_index[(df_index['date'] >= '2022-01-01') & (df_index['date'] <= today_str)]
        trade_dates = df_filtered['date'].dt.strftime('%Y%m%d').tolist()
        print(f"[INFO] SSE trading days from 2022-01-01 to today: {len(trade_dates)}")
    except Exception as e:
        print(f"[ERROR] Failed to fetch index dates: {e}")
        return

    # 3. Load Existing PCR Data
    pcr_records = []
    existing_dates = set()
    if os.path.exists(csv_path):
        try:
            df_exist = pd.read_csv(csv_path)
            df_exist['date'] = pd.to_datetime(df_exist['date'])
            pcr_records = df_exist.to_dict('records')
            existing_dates = set(df_exist['date'].dt.strftime('%Y%m%d').tolist())
            print(f"[INFO] Loaded existing database: {len(existing_dates)} trading days.")
        except Exception as e:
            print(f"[WARNING] Could not parse existing CSV: {e}. Starting fresh.")
    else:
        print(f"[INFO] No existing CSV found at {csv_path}. Initializing fresh dataset.")

    # 4. Identify Missing Dates
    dates_to_fetch = [d for d in trade_dates if d not in existing_dates]
    
    # Filter out today if market is not closed yet (before 16:00 to ensure option stats are compiled)
    current_hour = pd.Timestamp.now().hour
    today_yyyymmdd = pd.Timestamp.now().strftime('%Y%m%d')
    if today_yyyymmdd in dates_to_fetch and current_hour < 16:
        dates_to_fetch.remove(today_yyyymmdd)
        print(f"[INFO] Excluding today ({today_yyyymmdd}) as option stats compile after 16:00 local time.")

    print(f"[INFO] Found {len(dates_to_fetch)} missing dates to download.")
    
    if not dates_to_fetch:
        print("[SUCCESS] All options PCR data is already up-to-date!")
        return

    # 5. Fetch Options Daily Statistics Incrementally
    print(f"[INFO] Fetching options PCR for {len(dates_to_fetch)} dates...")
    start_time = time.time()
    batch_save_size = 10
    success_count = 0
    
    for idx, date_str in enumerate(dates_to_fetch):
        # Save every few records to avoid data loss if interrupted
        if idx > 0 and idx % batch_save_size == 0 and success_count > 0:
            df_temp = pd.DataFrame(pcr_records)
            df_temp = df_temp.sort_values('date').reset_index(drop=True)
            df_temp.to_csv(csv_path, index=False)
            print(f"  [SAVE] Saved batch! Progress: {idx}/{len(dates_to_fetch)} | Time: {time.time() - start_time:.1f}s")

        try:
            # Fetch options statistics for SSE
            df_day = ak.option_daily_stats_sse(date=date_str)
            if df_day is not None and not df_day.empty:
                # SSE 50ETF Option
                row_50 = df_day[df_day.iloc[:, 0].astype(str) == '510050']
                # SSE 300ETF Option
                row_300 = df_day[df_day.iloc[:, 0].astype(str) == '510300']
                
                record = {'date': pd.to_datetime(date_str, format='%Y%m%d')}
                
                if not row_50.empty:
                    record['pcr_50'] = float(row_50.iloc[0, 7]) / 100.0
                    oi_call = float(row_50.iloc[0, 9])
                    oi_put = float(row_50.iloc[0, 10])
                    record['oi_pcr_50'] = oi_put / oi_call if oi_call > 0 else np.nan
                    record['vol_50'] = float(row_50.iloc[0, 4])
                    
                if not row_300.empty:
                    record['pcr_300'] = float(row_300.iloc[0, 7]) / 100.0
                    oi_call_300 = float(row_300.iloc[0, 9])
                    oi_put_300 = float(row_300.iloc[0, 10])
                    record['oi_pcr_300'] = oi_put_300 / oi_call_300 if oi_call_300 > 0 else np.nan
                    record['vol_300'] = float(row_300.iloc[0, 4])
                    
                pcr_records.append(record)
                success_count += 1
        except Exception as e:
            # Ignore and skip non-trading days or API temporary failures
            pass
        
        # Polite API delay to respect Akshare rate limits
        time.sleep(0.1)

    # 6. Final Save
    if success_count > 0:
        df_temp = pd.DataFrame(pcr_records)
        df_temp = df_temp.sort_values('date').reset_index(drop=True)
        df_temp.to_csv(csv_path, index=False)
        print(f"[SUCCESS] Incremental PCR Sync Complete! Downloaded {success_count} new dates.")
        print(f"[INFO] Database saved at: {csv_path}. Total records: {len(df_temp)}.")
    else:
        print("[INFO] No new dates were fetched successfully.")

if __name__ == "__main__":
    main()

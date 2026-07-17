import akshare as ak
import pandas as pd
import numpy as np
import time
import os

def main():
    print("==================================================")
    print("   Downloading Historical A-Share Options PCR     ")
    print("==================================================")

    # 1. Fetch SSE Index dates
    try:
        df_index = ak.stock_zh_index_daily(symbol="sh000001")
        df_index['date'] = pd.to_datetime(df_index['date'])
        # Filter for dates from 2022-01-01 to 2026-05-28
        df_filtered = df_index[(df_index['date'] >= '2022-01-01') & (df_index['date'] <= '2026-05-28')]
        trade_dates = df_filtered['date'].dt.strftime('%Y%m%d').tolist()
        print(f"Total trading days to download: {len(trade_dates)}")
    except Exception as e:
        print("Error fetching index dates:", e)
        return

    os.makedirs("research/期权/data", exist_ok=True)
    csv_path = "research/期权/data/historical_pcr.csv"
    
    # Load existing if any, to resume
    existing_dates = set()
    pcr_records = []
    if os.path.exists(csv_path):
        try:
            df_exist = pd.read_csv(csv_path)
            df_exist['date'] = pd.to_datetime(df_exist['date'])
            pcr_records = df_exist.to_dict('records')
            existing_dates = set(df_exist['date'].dt.strftime('%Y%m%d').tolist())
            print(f"Resuming download. Found {len(existing_dates)} dates already saved.")
        except Exception:
            pass

    # Filter out already downloaded dates
    dates_to_fetch = [d for d in trade_dates if d not in existing_dates]
    print(f"Dates remaining to fetch: {len(dates_to_fetch)}")
    
    if not dates_to_fetch:
        print("All data already downloaded!")
        return

    start_time = time.time()
    batch_size = 50
    
    for idx, date_str in enumerate(dates_to_fetch):
        if idx > 0 and idx % batch_size == 0:
            # Periodically save to disk to prevent data loss
            df_temp = pd.DataFrame(pcr_records)
            df_temp = df_temp.sort_values('date').reset_index(drop=True)
            df_temp.to_csv(csv_path, index=False)
            elapsed = time.time() - start_time
            print(f"  Saved batch! Progress: {idx}/{len(dates_to_fetch)} | Time elapsed: {elapsed:.1f}s")
            
        try:
            df_day = ak.option_daily_stats_sse(date=date_str)
            if df_day is not None and not df_day.empty:
                row_50 = df_day[df_day.iloc[:, 0].astype(str) == '510050']
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
        except Exception:
            # Ignore errors for non-trading or corrupted days
            pass
        # Very small delay to respect rate limit without slowing down too much
        time.sleep(0.02)
        
    # Final save
    df_temp = pd.DataFrame(pcr_records)
    df_temp = df_temp.sort_values('date').reset_index(drop=True)
    df_temp.to_csv(csv_path, index=False)
    print(f"Completed! Saved all {len(df_temp)} records to {csv_path} in {time.time() - start_time:.1f}s.")

if __name__ == "__main__":
    main()

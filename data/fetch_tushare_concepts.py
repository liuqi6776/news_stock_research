import tushare as ts
import pandas as pd
import os
import time
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()
token = os.getenv('TUSHARE_TOKEN')
if not token:
    raise ValueError("TUSHARE_TOKEN not found in .env")

pro = ts.pro_api(token)

def fetch_and_cache_concepts(save_path='tushare_concept_map_cached.parquet'):
    print("Fetching concept list...")
    try:
        concepts = pro.concept()
    except Exception as e:
        print(f"Fatal error fetching concepts: {e}")
        return

    if concepts is None or concepts.empty:
        print("Failed to fetch concept list - returned empty.")
        return
        
    print(f"Total concepts found: {len(concepts)}")
    
    all_details = []
    
    for i, row in tqdm(concepts.iterrows(), total=len(concepts), desc="Fetching concept details"):
        c_id = row['code']
        c_name = row['name']
        try:
            df_detail = pro.concept_detail(id=c_id)
            if not df_detail.empty:
                df_detail['concept_name'] = c_name
                all_details.append(df_detail[['concept_name', 'ts_code']])
        except Exception as e:
            print(f"Error fetching {c_name} ({c_id}): {e}")
            time.sleep(2) 
            continue
            
        time.sleep(0.12)
        
    if all_details:
        final_df = pd.concat(all_details, ignore_index=True)
        final_df = final_df.drop_duplicates()
        final_df.to_parquet(save_path)
        print(f"Successfully saved {len(final_df)} concept-stock pairs to {save_path}")
    else:
        print("No concept details fetched.")

if __name__ == '__main__':
    OUT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    save_path = os.path.join(OUT_DIR, 'tushare_concept_map_cached.parquet')
    fetch_and_cache_concepts(save_path)

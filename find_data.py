import os
import pandas as pd
from tqdm import tqdm

def find_columns_in_parquet(root_dir, target_cols):
    found_files = []
    print(f"Searching in {root_dir}...")
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith('.parquet'):
                path = os.path.join(root, file)
                try:
                    # Only read the metadata (columns) to be fast
                    df_sample = pd.read_parquet(path, columns=[]) # Empty columns might not work for checking presence
                    # Better way:
                    import pyarrow.parquet as pq
                    schema = pq.read_schema(path)
                    cols = schema.names
                    if any(c in cols for c in target_cols):
                        print(f"Found in: {path}")
                        found_files.append((path, cols))
                except Exception as e:
                    continue
    return found_files

if __name__ == "__main__":
    targets = ['rzye', 'vix_close', '000188.SH', '000188']
    results = find_columns_in_parquet(r'D:\iquant_data\data_v2', targets)
    if not results:
        results = find_columns_in_parquet(r'C:\Users\liuqi\quant_system_v2', targets)
    
    for path, cols in results:
        print(f"FILE: {path}\nCOLS: {cols}\n")

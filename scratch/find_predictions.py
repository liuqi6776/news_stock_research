import os
import pandas as pd

def find_parquet_files(root_dir):
    for root, dirs, files in os.walk(root_dir):
        for f in files:
            if f.endswith('.parquet') and ('prediction' in f or 'pred' in f):
                path = os.path.join(root, f)
                try:
                    df = pd.read_parquet(path)
                    date_col = None
                    for col in ['trade_date', 'date', 'ds']:
                        if col in df.columns:
                            date_col = col
                            break
                    if date_col:
                        dates = df[date_col].astype(str)
                        print(f"Path: {path}")
                        print(f"  Rows: {len(df)}, Date Col: {date_col}")
                        print(f"  Min Date: {dates.min()}, Max Date: {dates.max()}")
                    else:
                        print(f"Path: {path} (No date column found, cols={list(df.columns)})")
                except Exception as e:
                    print(f"Error reading {path}: {e}")

if __name__ == "__main__":
    find_parquet_files("c:\\Users\\liuqi\\quant_system_v2")

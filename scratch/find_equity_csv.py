import os
import pandas as pd

def find_csv_files(root_dir):
    for root, dirs, files in os.walk(root_dir):
        for f in files:
            if f.endswith('.csv') and ('equity' in f or 'trades' in f or 'nav' in f):
                path = os.path.join(root, f)
                try:
                    df = pd.read_csv(path)
                    print(f"Path: {path}")
                    print(f"  Rows: {len(df)}, Columns: {list(df.columns)}")
                    date_col = None
                    for c in ['date', 'trade_date', 'date_t2', 'Date']:
                        if c in df.columns:
                            date_col = c
                            break
                    if date_col:
                        dates = pd.to_datetime(df[date_col]).dt.strftime('%Y-%m-%d')
                        print(f"  Range: {dates.min()} to {dates.max()}")
                except Exception as e:
                    print(f"Error reading {path}: {e}")

if __name__ == "__main__":
    find_csv_files("c:\\Users\\liuqi\\quant_system_v2")

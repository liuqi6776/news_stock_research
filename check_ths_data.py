import pandas as pd
import os

dates = ['20260407', '20260403', '20260402', '20260401', '20260331']
THS_DIR = r"D:\iquant_data\data_v2\ths_rank1"

for date in dates:
    filepath = os.path.join(THS_DIR, f"{date}.parquet")
    if os.path.exists(filepath):
        try:
            df = pd.read_parquet(filepath)
            print(f"{date}: {len(df)} 行")
            if len(df) > 0:
                print(df.head(3))
        except Exception as e:
            print(f"{date}: 错误 - {e}")
    else:
        print(f"{date}: 文件不存在")

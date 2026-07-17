import pandas as pd

try:
    df = pd.read_parquet(r'D:\iquant_data\data_v2\income1\20241201.parquet')
    print('Income cols:', list(df.columns))
    print(df.head(2))
except Exception as e:
    print('Income error:', e)

try:
    df2 = pd.read_parquet(r'D:\iquant_data\data_v2\other_day1\20241201.parquet')
    print('Other cols:', list(df2.columns))
    print(df2.head(2))
except Exception as e:
    print('Other error:', e)

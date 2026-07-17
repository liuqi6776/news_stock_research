import os
import pandas as pd
import numpy as np

# Define paths
DATA_DIR = r"c:\Users\liuqi\quant_system_v2\etf-valuation-strategy\data"

# Download 000832.CSI data
import tushare as ts
TOKEN = "421ff94dd31be789aa7f95e61ad6fad5bcefa250a0b2c4d298224aa"
pro = ts.pro_api(TOKEN)

print("Downloading 000832.CSI index daily data...")
df_cbond = pro.index_daily(ts_code="000832.CSI", start_date="20100101", end_date="20260315")
cbond_path = os.path.join(DATA_DIR, 'cbond_daily.csv')
df_cbond.to_csv(cbond_path, index=False)
print(f"Convertible bond data saved to {cbond_path}")

# Load all 8 assets
files = {
    'hs300': 'hs300_daily.csv',
    'zz500': 'zz500_daily.csv',
    'chinext': 'chinext_daily.csv',
    'div_low_vol': 'div_low_vol_daily.csv',
    'gold': 'gold_etf_daily.csv',
    'nasdaq': 'nasdaq_etf_daily.csv',
    'bond': 'bond_etf_daily.csv',
    'cbond': 'cbond_daily.csv'
}

data = {}
for name, fname in files.items():
    path = os.path.join(DATA_DIR, fname)
    df = pd.read_csv(path)
    df['trade_date'] = pd.to_datetime(df['trade_date'].astype(str))
    df = df.sort_values('trade_date').reset_index(drop=True)
    df['ret'] = df['pct_chg'] / 100.0
    data[name] = df.set_index('trade_date')['ret']

# Combine returns
df_rets = pd.DataFrame(data).dropna()
print(f"\nCombined returns shape: {df_rets.shape}")

# Correlation matrix
corr = df_rets.corr()
print("\nCorrelation Matrix of the 8 Assets:")
print(corr.round(2))

# Average correlation
upper_tri = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
avg_corr = upper_tri.stack().mean()
print(f"\nAverage Correlation between assets: {avg_corr:.3f}")

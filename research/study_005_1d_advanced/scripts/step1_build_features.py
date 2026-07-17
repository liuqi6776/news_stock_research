"""
step1_build_features.py
基于 004 的特征文件，构建 005 需要的增强版特征文件。
新增：
1. 行业映射 (industry)
2. 宽基指数 MA20 距离 (regime_filter)
3. 双重 Target：
   - target_up: (T+2 close - T+1 open) / T+1 open
   - target_crash: (T+1 close - T+1 open) / T+1 open < -0.03
"""
import os
import pandas as pd
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
STUDY_DIR = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(STUDY_DIR, 'data')

# 原始 004 数据路径
SRC_FEAT = r'C:\Users\liuqi\quant_system_v2\research\study_004_1d_release\data\all_features_v2.parquet'

OUT_FEAT = os.path.join(DATA_DIR, 'features_005.parquet')

def build_features():
    print(f"Loading base features from {SRC_FEAT}...")
    df = pd.read_parquet(SRC_FEAT)
    df['trade_date'] = df['trade_date'].astype(str)
    
    # 1. 加载行业映射
    ind_file = os.path.join(DATA_DIR, 'industry_map.csv')
    if os.path.exists(ind_file):
        print("Merging industry map...")
        ind_df = pd.read_csv(ind_file)
        df = df.merge(ind_df[['ts_code', 'industry']], on='ts_code', how='left')
        df['industry'] = df['industry'].fillna('Unknown')
    else:
        df['industry'] = 'Unknown'

    # 我们将直接在回测中使用 news_market_impact 作为择时过滤

    print("Calculating targets...")
    df = df.sort_values(['ts_code', 'trade_date'])
    
    # T+1 开盘, 收盘, T+2 收盘
    df['next_open'] = df.groupby('ts_code')['open'].shift(-1)
    df['next_close'] = df.groupby('ts_code')['close'].shift(-1)
    df['d2_close'] = df.groupby('ts_code')['close'].shift(-2)
    
    df['next_high'] = df.groupby('ts_code')['high'].shift(-1)
    df['d2_high'] = df.groupby('ts_code')['high'].shift(-2)
    df['next_low'] = df.groupby('ts_code')['low'].shift(-1)
    df['d2_low'] = df.groupby('ts_code')['low'].shift(-2)
    
    # 目标 1: return_1d_open (T+2收盘 - T+1开盘) / T+1开盘 (保留做回归或分析)
    df['return_1d_open'] = (df['d2_close'] - df['next_open']) / df['next_open']
    
    # 实际高低价计算
    max_high = df[['next_high', 'd2_high']].max(axis=1)
    min_low = df[['next_low', 'd2_low']].min(axis=1)
    
    # 构建物理止盈/止损二分类目标 (T+1开盘买入，T+2收盘前是否触及+6%或-5%)
    df['target_up_bin'] = ((max_high - df['next_open']) / df['next_open'] >= 0.06).astype(int)
    df['target_crash_bin'] = ((min_low - df['next_open']) / df['next_open'] <= -0.05).astype(int)
    
    # 处理边界 NaNs
    df.loc[df['return_1d_open'].isna(), ['target_up_bin', 'target_crash_bin']] = np.nan
    
    df = df.drop(columns=['next_close', 'd2_close', 'next_high', 'd2_high', 'next_low', 'd2_low'])
    
    print("Saving to", OUT_FEAT)
    df.to_parquet(OUT_FEAT)
    
    print(f"Total rows: {len(df)}")
    print(f"Target crash positive rate: {df['target_crash_bin'].mean():.2%}")

if __name__ == '__main__':
    build_features()

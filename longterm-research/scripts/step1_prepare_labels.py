"""
step1_prepare_labels.py
基于 004 基础特征文件构建长期多因子研究的收益率标签与超额收益率（Market-relative & Industry-relative）。
同时获取并合并 QVIX 与期权 Put-Call Ratio (PCR) 指标。
支持的持有期限：5日、10日、20日。
"""
import os
import pandas as pd
import numpy as np
import akshare as ak

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(PROJECT_DIR, 'data')

import sys
sys.path.append(PROJECT_DIR)
from features.alpha_factors import calculate_alpha101_factors
from features.vibe_alpha_zoo import calculate_vibe_alphas

# 原始 004 数据路径
SRC_FEAT = r'C:\Users\liuqi\quant_system_v2\research\study_004_1d_release\data\all_features_v2.parquet'
OUT_FEAT = os.path.join(DATA_DIR, 'features_longterm.parquet')

def prepare_labels():
    print(f"Loading base features from {SRC_FEAT}...")
    df = pd.read_parquet(SRC_FEAT)
    df['trade_date'] = df['trade_date'].astype(str)
    
    print("Deduplicating features and correcting ths_hot_rank daily...")
    df['ths_hot'] = pd.to_numeric(df['ths_hot'], errors='coerce').fillna(0.0)
    max_hot = df.groupby(['trade_date', 'ts_code'])['ths_hot'].max().reset_index()
    df_dedup = df.drop_duplicates(subset=['trade_date', 'ts_code'], keep='first').copy()
    df_dedup = df_dedup.drop(columns=['ths_hot', 'ths_hot_rank'], errors='ignore')
    df_dedup = df_dedup.merge(max_hot, on=['trade_date', 'ts_code'], how='left')
    
    daily_max_hot = df_dedup.groupby('trade_date')['ths_hot'].transform('max')
    valid_day_mask = daily_max_hot > 0
    
    df_dedup['ths_hot_rank'] = np.nan
    hot_mask = valid_day_mask & (df_dedup['ths_hot'] > 0)
    df_dedup.loc[hot_mask, 'ths_hot_rank'] = df_dedup[hot_mask].groupby('trade_date')['ths_hot'].rank(ascending=False, method='min')
    
    cold_mask = valid_day_mask & (df_dedup['ths_hot'] == 0)
    df_dedup.loc[cold_mask, 'ths_hot_rank'] = 9999.0
    
    df = df_dedup
    
    # 排序以保证 shift 逻辑的正确性
    print("Sorting data by ts_code and trade_date...")
    df = df.sort_values(['ts_code', 'trade_date']).reset_index(drop=True)
    
    # 计算 Alpha 101 因子
    print("Calculating Alpha 101 factors...")
    df = calculate_alpha101_factors(df)
    
    # 计算 Vibe-Trading Alpha Zoo
    print("Calculating Vibe-Trading Alpha Zoo factors...")
    df = calculate_vibe_alphas(df, num_factors=40)
    
    # 加载行业映射
    ind_file = os.path.join(DATA_DIR, 'industry_map.csv')
    if os.path.exists(ind_file):
        print("Merging industry map...")
        ind_df = pd.read_csv(ind_file)
        if 'industry' in df.columns:
            df = df.drop(columns=['industry'])
        df = df.merge(ind_df[['ts_code', 'industry']], on='ts_code', how='left')
        df['industry'] = df['industry'].fillna('Unknown')
    else:
        print("⚠️ Industry map file not found, using 'Unknown' for all stocks.")
        df['industry'] = 'Unknown'

    # 3. 获取并合并 QVIX 指标
    print("Fetching 50ETF QVIX from Akshare...")
    try:
        df_qvix = ak.index_option_50etf_qvix()
        df_qvix['date'] = pd.to_datetime(df_qvix['date'])
        df_qvix['trade_date'] = df_qvix['date'].dt.strftime('%Y%m%d')
        
        # 计算 QVIX 衍生指标
        df_qvix['opt_qvix_close'] = df_qvix['close']
        df_qvix['opt_qvix_change'] = df_qvix['close'].pct_change()
        df_qvix['opt_qvix_ma'] = df_qvix['close'].rolling(20).mean()
        df_qvix['opt_qvix_std'] = df_qvix['close'].rolling(20).std()
        df_qvix['opt_qvix_zscore'] = (df_qvix['close'] - df_qvix['opt_qvix_ma']) / df_qvix['opt_qvix_std']
        
        df_qvix_clean = df_qvix[['trade_date', 'opt_qvix_close', 'opt_qvix_change', 'opt_qvix_zscore']]
        print(f"  QVIX fetched: {len(df_qvix_clean)} rows.")
    except Exception as e:
        print("⚠️ Failed to fetch QVIX from Akshare, creating empty placeholders:", e)
        df_qvix_clean = pd.DataFrame(columns=['trade_date', 'opt_qvix_close', 'opt_qvix_change', 'opt_qvix_zscore'])

    # 4. 加载历史期权 PCR 指标
    pcr_path = r"D:\iquant_data\data_v2\qiquan\historical_pcr.csv"
    if os.path.exists(pcr_path):
        print("Loading PCR data from local file...")
        df_pcr = pd.read_csv(pcr_path)
        df_pcr['date'] = pd.to_datetime(df_pcr['date'])
        df_pcr['trade_date'] = df_pcr['date'].dt.strftime('%Y%m%d')
        
        df_pcr_clean = df_pcr[['trade_date', 'pcr_50', 'oi_pcr_50', 'pcr_300', 'oi_pcr_300']].rename(columns={
            'pcr_50': 'opt_pcr_vol_50',
            'oi_pcr_50': 'opt_pcr_oi_50',
            'pcr_300': 'opt_pcr_vol_300',
            'oi_pcr_300': 'opt_pcr_oi_300'
        })
        print(f"  PCR loaded: {len(df_pcr_clean)} rows.")
    else:
        print("⚠️ PCR file not found, creating empty placeholders.")
        df_pcr_clean = pd.DataFrame(columns=['trade_date', 'opt_pcr_vol_50', 'opt_pcr_oi_50', 'opt_pcr_vol_300', 'opt_pcr_oi_300'])

    # 5. 合并期权指标
    print("Merging Options and PCR data with stock features...")
    df = df.merge(df_qvix_clean, on='trade_date', how='left')
    df = df.merge(df_pcr_clean, on='trade_date', how='left')
    
    # 填充 options columns
    opt_cols = ['opt_qvix_close', 'opt_qvix_change', 'opt_qvix_zscore', 
                'opt_pcr_vol_50', 'opt_pcr_oi_50', 'opt_pcr_vol_300', 'opt_pcr_oi_300']
                
    print("Forward filling options features...")
    df_opts_unique = df[['trade_date'] + opt_cols].drop_duplicates().sort_values('trade_date').reset_index(drop=True)
    df_opts_unique[opt_cols] = df_opts_unique[opt_cols].ffill().fillna(0)
    
    df = df.drop(columns=opt_cols, errors='ignore')
    df = df.merge(df_opts_unique, on='trade_date', how='left')
    df[opt_cols] = df[opt_cols].fillna(0).replace([np.inf, -np.inf], 0)
        
    print("Calculating forward returns for 5d, 10d, and 20d...")
    # T+1 开盘买入
    df['next_open'] = df.groupby('ts_code')['open'].shift(-1)
    
    for N in [5, 10, 20]:
        print(f"Calculating {N}-day forward return...")
        # T+N 收盘卖出
        df[f'close_T{N}'] = df.groupby('ts_code')['close'].shift(-N)
        
        # 收益率计算： (close_t+N - open_t+1) / open_t+1
        df[f'ret_{N}d'] = (df[f'close_T{N}'] - df['next_open']) / df['next_open']
        
        # 1. 市场超额收益率：相对于每日所有股票平均收益率的超额
        print(f"Calculating market-relative excess return for {N}d...")
        mkt_mean = df.groupby('trade_date')[f'ret_{N}d'].transform('mean')
        df[f'mkt_excess_ret_{N}d'] = df[f'ret_{N}d'] - mkt_mean
        
        # 2. 行业超额收益率：相对于每日同行业股票平均收益率的超额
        print(f"Calculating industry-relative excess return for {N}d...")
        ind_mean = df.groupby(['trade_date', 'industry'])[f'ret_{N}d'].transform('mean')
        df[f'ind_excess_ret_{N}d'] = df[f'ret_{N}d'] - ind_mean
        
        # 打印部分统计数据
        valid_rows = df[f'ret_{N}d'].notna().sum()
        print(f"  {N}d valid rows: {valid_rows}, Mean return: {df[f'ret_{N}d'].mean():.4f}, Std: {df[f'ret_{N}d'].std():.4f}")
        
    print("Saving enriched feature file to", OUT_FEAT)
    df.to_parquet(OUT_FEAT, index=False)
    print("Enriched feature file saved successfully!")

if __name__ == '__main__':
    prepare_labels()

"""
日频 Alpha 诊断：情绪 (ths_rank) 与 筹码 (cyq) 的 T+1 预测能力分析
目标：验证这些因子是否能支撑年化 50% 的单日交易
"""
import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm

DATA_DIR = r'D:\iquant_data\data_v2'
PRICE_DIR = os.path.join(DATA_DIR, 'data_day1')
RANK_DIR  = os.path.join(DATA_DIR, 'ths_rank1')
CHIP_DIR  = os.path.join(DATA_DIR, 'cyq1')

def diagnose_daily_alpha(start_date='20240101', end_date='20240930'):
    dates = sorted([f.replace('.parquet', '') for f in os.listdir(PRICE_DIR) if f.endswith('.parquet')])
    dates = [d for d in dates if start_date <= d <= end_date]
    
    results = []
    
    for i in tqdm(range(len(dates)-1), desc="分析日频 Alpha"):
        d_curr = dates[i]
        d_next = dates[i+1]
        
        # 1. 加载当日情绪与筹码
        p_rank = os.path.join(RANK_DIR, f"{d_curr}.parquet")
        p_chip = os.path.join(CHIP_DIR, f"{d_curr}.parquet")
        p_price = os.path.join(PRICE_DIR, f"{d_curr}.parquet")
        if not all(os.path.exists(p) for p in [p_rank, p_chip, p_price]):
            continue
            
        rank_df = pd.read_parquet(p_rank)
        chip_df = pd.read_parquet(p_chip)
        price_df = pd.read_parquet(p_price, columns=['ts_code', 'close', 'pct_chg'])
        
        # 2. 计算 T+1 收益 (用 T+1 的数据)
        p_next = os.path.join(PRICE_DIR, f"{d_next}.parquet")
        next_df = pd.read_parquet(p_next, columns=['ts_code', 'open', 'high', 'close'])
        
        # T+1 收益率 = (T+1 高价 / T+1 开盘价) - 1  (捕捉日内冲高机会)
        next_df['ret_intraday'] = (next_df['high'] / next_df['open']) - 1
        # T+1 隔夜收益率 = (T+1 收盘 / T+1 开盘) - 1
        next_df['ret_close'] = (next_df['close'] / next_df['open']) - 1
        
        # 3. 合并特征
        # 情绪特征：热度排名
        rank_df['hot_rank'] = rank_df['hot'].rank(ascending=False)
        # 筹码特征：85% 成本区间宽度 (越窄越集中)
        chip_df['chip_width'] = (chip_df['cost_85pct'] - chip_df['cost_15pct']) / chip_df['cost_50pct']
        
        merged = pd.merge(rank_df[['ts_code', 'hot', 'hot_rank']], next_df[['ts_code', 'ret_intraday', 'ret_close']], on='ts_code')
        merged = pd.merge(merged, chip_df[['ts_code', 'chip_width', 'winner_rate']], on='ts_code')
        
        if merged.empty: continue
        
        # 分组统计
        # 热度前 20 
        top20 = merged[merged['hot_rank'] <= 20]
        # 热度 20-100
        mid100 = merged[(merged['hot_rank'] > 20) & (merged['hot_rank'] <= 100)]
        
        results.append({
            'date': d_curr,
            'top20_ret_hit': top20['ret_intraday'].mean(),
            'top20_ret_close': top20['ret_close'].mean(),
            'mid100_ret_hit': mid100['ret_intraday'].mean(),
            'chip_narrow_ret': merged[merged['chip_width'] < 0.15]['ret_intraday'].mean()
        })
        
    res_df = pd.DataFrame(results)
    print("\n--- 日频 Alpha 诊断结果 ---")
    print(res_df.describe().to_string())
    
    # 统计胜率 (>2% 冲高机会)
    print(f"\n热度 Top20 日均冲高幅度: {res_df.top20_ret_hit.mean()*100:.2f}%")
    print(f"热度 Top20 收盘平均表现: {res_df.top20_ret_close.mean()*100:.2f}%")
    print(f"筹码集中股日均冲高幅度: {res_df.chip_narrow_ret.mean()*100:.2f}%")
    
    # 绘图
    plt.figure(figsize=(10, 6))
    plt.plot(res_df['top20_ret_hit'].rolling(5).mean(), label='Top20 Hot Intraday Peak (5D MA)')
    plt.axhline(0.02, color='r', linestyle='--', label='2% Target')
    plt.title('Daily Sentiment Alpha: Intraday Peak Potential')
    plt.legend()
    plt.savefig('daily_alpha_sentiment.png')
    print("图表已保存: daily_alpha_sentiment.png")

if __name__ == "__main__":
    diagnose_daily_alpha()

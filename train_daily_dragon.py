"""
高收益日频 Alpha 模型 (Dragon-Sentiment Hybrid)
目标：结合同花顺热度、筹码分布与连板基因，预测 T+1 的爆发性机会 (>50% 年化)
"""
import os
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import precision_score
from tqdm import tqdm

DATA_DIR = r'D:\iquant_data\data_v2'
PRICE_DIR = os.path.join(DATA_DIR, 'data_day1')
RANK_DIR  = os.path.join(DATA_DIR, 'ths_rank1')
CHIP_DIR  = os.path.join(DATA_DIR, 'cyq1')

def train_daily_dragon_model(train_start='20220101', train_end='20231231'):
    dates = sorted([f.replace('.parquet', '') for f in os.listdir(PRICE_DIR) if f.endswith('.parquet')])
    train_dates = [d for d in dates if train_start <= d <= train_end]
    
    all_data = []
    
    for i in tqdm(range(len(train_dates)-1), desc="准备训练集"):
        d_curr, d_next = train_dates[i], train_dates[i+1]
        
        p_rank = os.path.join(RANK_DIR, f"{d_curr}.parquet")
        p_chip = os.path.join(CHIP_DIR, f"{d_curr}.parquet")
        p_price = os.path.join(PRICE_DIR, f"{d_curr}.parquet")
        if not all(os.path.exists(p) for p in [p_rank, p_chip, p_price]): continue
        
        # 特征 1: 情绪 (ths_rank)
        rank_df = pd.read_parquet(p_rank)
        rank_df['hot_rank_pct'] = rank_df['hot'].rank(pct=True)
        
        # 特征 2: 筹码 (cyq)
        chip_df = pd.read_parquet(p_chip)
        chip_df['chip_concentration'] = (chip_df['cost_85pct'] - chip_df['cost_15pct']) / chip_df['cost_50pct']
        
        # 特征 3: 价格与成交量
        price_df = pd.read_parquet(p_price, columns=['ts_code', 'close', 'pct_chg', 'amount', 'vol'])
        
        # 标签: T+1 收益率 > 3% 为正类 (为了高收益)
        p_next = os.path.join(PRICE_DIR, f"{d_next}.parquet")
        next_df = pd.read_parquet(p_next, columns=['ts_code', 'open', 'high', 'close'])
        # 实际收益逻辑：开盘买，最高价卖 (模拟 Ptrade 止盈)
        next_df['label_ret'] = (next_df['high'] / next_df['open']) - 1
        next_df['label'] = (next_df['label_ret'] > 0.04).astype(int) 
        
        # 合并
        m = pd.merge(rank_df[['ts_code', 'hot_rank_pct']], price_df, on='ts_code')
        m = pd.merge(m, chip_df[['ts_code', 'chip_concentration', 'winner_rate']], on='ts_code')
        m = pd.merge(m, next_df[['ts_code', 'label']], on='ts_code')
        
        all_data.append(m)
        
    df = pd.concat(all_data, ignore_index=True)
    
    # 训练模型
    feats = ['hot_rank_pct', 'pct_chg', 'amount', 'vol', 'chip_concentration', 'winner_rate']
    X = df[feats].fillna(0)
    y = df['label']
    
    # 平衡样本
    pos = df[df.label == 1]
    neg = df[df.label == 0].sample(len(pos))
    train_bal = pd.concat([pos, neg])
    
    model = xgb.XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.1, eval_metric='logloss')
    model.fit(train_bal[feats], train_bal['label'])
    
    print(f"模型训练完成。正类样本: {len(pos)}。")
    return model, feats

if __name__ == "__main__":
    model, feats = train_daily_dragon_model()
    # 保存模型
    import joblib
    joblib.dump((model, feats), 'daily_dragon_model.joblib')

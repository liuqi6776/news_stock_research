import os
import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
from tqdm import tqdm
import sys

DATA_DIR = r'D:\iquant_data\data_v2'
PRICE_DIR = os.path.join(DATA_DIR, 'data_day1')
RANK_DIR  = os.path.join(DATA_DIR, 'ths_rank1')
CHIP_DIR  = os.path.join(DATA_DIR, 'cyq1')
NEWS_MAJOR_DIR = r"C:\Users\liuqi\clowspace\toolstock\news_major1"
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models', 'daily_t1_model.joblib')

def process_news(news_dir):
    """处理 news_major1 新闻数据"""
    market_records = []
    stock_records = []
    
    for filename in os.listdir(news_dir):
        if not filename.endswith('.json'): continue
        filepath = os.path.join(news_dir, filename)
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except: continue
            
        date_str = data.get("article_date", "")
        if not date_str: continue
            
        trade_date = pd.to_datetime(date_str)
        date_formatted = trade_date.strftime('%Y%m%d')
            
        market_impact = data.get("market_impact", 0)
        market_records.append({'trade_date': trade_date, 'news_market_impact': float(market_impact)})
        
        for s in data.get("stocks", []):
            code = s.get("stock_code")
            if not code: continue
            ts_code = f"{code}.SH" if code.startswith('6') else f"{code}.SZ" if (code.startswith('0') or code.startswith('3')) else code
            stock_records.append({'trade_date': trade_date, 'ts_code': ts_code, 'news_stock_impact': float(s.get("impact", 0))})
    
    return pd.DataFrame(market_records), pd.DataFrame(stock_records)

def train_daily_model(start_date='20220101', end_date=None, model_path=None):
    """
    训练 XGBoost T+1 模型
    标签：次日高点 > 开盘价 +4%
    特征：热度分位数、筹码集中度、获利盘、新闻影响
    """
    print(f"--- [Step 3] 训练模型: {start_date} 至 {end_date} ---")
    
    dates = sorted([f.replace('.parquet', '') for f in os.listdir(PRICE_DIR) if f.endswith('.parquet')])
    
    if end_date is None:
        end_date = dates[-1]
    
    train_dates = [d for d in dates if start_date <= d <= end_date]
    if len(train_dates) < 2:
        print(f"警告: 训练数据不足，需要至少 2 天")
        return None, None
        
    news_mkt, news_stk = process_news(NEWS_MAJOR_DIR)
    if not news_mkt.empty:
        news_mkt['trade_date'] = news_mkt['trade_date'].dt.strftime('%Y%m%d')
    if not news_stk.empty:
        news_stk['trade_date'] = news_stk['trade_date'].dt.strftime('%Y%m%d')

    all_data = []
    
    for i in tqdm(range(len(train_dates)-1), desc=f"加载训练数据 [{start_date}-{end_date}]", leave=False):
        d_curr, d_next = train_dates[i], train_dates[i+1]
        
        p_rank = os.path.join(RANK_DIR, f"{d_curr}.parquet")
        p_chip = os.path.join(CHIP_DIR, f"{d_curr}.parquet")
        p_price = os.path.join(PRICE_DIR, f"{d_curr}.parquet")
        
        if not all(os.path.exists(p) for p in [p_rank, p_chip, p_price]):
            continue
            
        rank_df = pd.read_parquet(p_rank)
        rank_df['hot_rank_pct'] = rank_df['hot'].rank(pct=True)
        
        chip_df = pd.read_parquet(p_chip)
        chip_df['chip_concentration'] = (chip_df['cost_85pct'] - chip_df['cost_15pct']) / (chip_df['cost_50pct'] + 1e-8)
        
        price_df = pd.read_parquet(p_price, columns=['ts_code', 'close', 'pct_chg', 'amount', 'vol'])
        
        p_next = os.path.join(PRICE_DIR, f"{d_next}.parquet")
        if not os.path.exists(p_next):
            continue
        next_df = pd.read_parquet(p_next, columns=['ts_code', 'open', 'high', 'close'])
        
        next_df['label_ret'] = (next_df['high'] / (next_df['open'] + 1e-8)) - 1
        next_df['label'] = (next_df['label_ret'] > 0.04).astype(int)
        
        m = pd.merge(rank_df[['ts_code', 'hot_rank_pct']], price_df, on='ts_code')
        m = pd.merge(m, chip_df[['ts_code', 'chip_concentration', 'winner_rate']], on='ts_code')
        m = pd.merge(m, next_df[['ts_code', 'label']], on='ts_code')
        m['trade_date'] = d_next
        
        all_data.append(m)
        
    df = pd.concat(all_data, ignore_index=True)
    
    if not news_mkt.empty:
        df = pd.merge(df, news_mkt, on='trade_date', how='left')
    else:
        df['news_market_impact'] = 0.0
        
    if not news_stk.empty:
        df = pd.merge(df, news_stk[['trade_date', 'ts_code', 'news_stock_impact']], on=['trade_date', 'ts_code'], how='left')
    else:
        df['news_stock_impact'] = 0.0
        
    df[['news_market_impact', 'news_stock_impact']] = df[['news_market_impact', 'news_stock_impact']].fillna(0.0)
    
    FEATURE_COLS = [
        'hot_rank_pct', 'chip_concentration', 'winner_rate',
        'news_market_impact', 'news_stock_impact'
    ]
    
    X = df[FEATURE_COLS].fillna(0)
    y = df['label']
    
    print(f"\n训练样本数: {len(X)}")
    print(f"正样本数: {y.sum()} ({y.sum()/len(y):.2%})")
    print(f"使用特征: {FEATURE_COLS}")
    
    model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric='auc',
        n_jobs=-1,
        tree_method='hist'
    )
    
    print(f"训练 XGBoost 模型...")
    model.fit(X, y)
    
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump((model, FEATURE_COLS), model_path)
    print(f"✓ 模型保存成功: {model_path}")
        
    return model, FEATURE_COLS

if __name__ == "__main__":
    import json
    train_daily_model(model_path=MODEL_PATH)

import os
import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm
import sys

# Ensure imports work
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from infra_data.storage import DataStorage

DATA_DIR = r'D:\iquant_data\data_v2'
PRICE_DIR = os.path.join(DATA_DIR, 'data_day1')
RANK_DIR  = os.path.join(DATA_DIR, 'ths_rank1')
CHIP_DIR  = os.path.join(DATA_DIR, 'cyq1')

def load_dataset(start_date, end_date):
    """Loads price, rank, cyq and merges with news features."""
    storage = DataStorage()
    
    dates = sorted([f.replace('.parquet', '') for f in os.listdir(PRICE_DIR) if f.endswith('.parquet')])
    valid_dates = [d for d in dates if start_date <= d <= end_date]
    valid_dates_series = pd.Series([pd.to_datetime(d) for d in valid_dates]).sort_values()
    
    # 1. Load News
    news_market_df, news_stock_sector_df = storage.load_news_data(start_date, end_date, valid_dates_series)
    
    if not news_market_df.empty:
        news_market_df['trade_date'] = news_market_df['trade_date'].dt.strftime('%Y%m%d')
    if not news_stock_sector_df.empty:
        news_stock_sector_df['trade_date'] = news_stock_sector_df['trade_date'].dt.strftime('%Y%m%d')
        
    all_data = []
    
    for i in tqdm(range(len(valid_dates)-1), desc=f"Loading valid rows {start_date}-{end_date}"):
        d_curr, d_next = valid_dates[i], valid_dates[i+1]
        
        p_rank = os.path.join(RANK_DIR, f"{d_curr}.parquet")
        p_chip = os.path.join(CHIP_DIR, f"{d_curr}.parquet")
        p_price = os.path.join(PRICE_DIR, f"{d_curr}.parquet")
        if not all(os.path.exists(p) for p in [p_rank, p_chip, p_price]): 
            continue
            
        rank_df = pd.read_parquet(p_rank)
        rank_df['hot_rank_pct'] = rank_df['hot'].rank(pct=True)
        
        chip_df = pd.read_parquet(p_chip)
        chip_df['chip_concentration'] = (chip_df['cost_85pct'] - chip_df['cost_15pct']) / (chip_df['cost_50pct'] + 1e-8)
        
        price_df = pd.read_parquet(p_price, columns=['ts_code', 'open', 'close', 'pre_close', 'pct_chg', 'amount', 'vol'])
        
        # Next day returns for labels and backtest simulation
        p_next = os.path.join(PRICE_DIR, f"{d_next}.parquet")
        next_df = pd.read_parquet(p_next, columns=['ts_code', 'open', 'high', 'close', 'pre_close'])
        next_df['label_ret'] = (next_df['high'] / (next_df['open'] + 1e-8)) - 1
        next_df['label'] = (next_df['label_ret'] > 0.04).astype(int) 
        
        m = pd.merge(rank_df[['ts_code', 'hot_rank_pct']], price_df, on='ts_code')
        m = pd.merge(m, chip_df[['ts_code', 'chip_concentration', 'winner_rate']], on='ts_code')
        m = pd.merge(m, next_df[['ts_code', 'open', 'high', 'close', 'pre_close', 'label']], on='ts_code', suffixes=('', '_next'))
        m['trade_date'] = d_curr
        m['next_date'] = d_next
        
        all_data.append(m)
        
    if not all_data:
        return pd.DataFrame()
        
    df = pd.concat(all_data, ignore_index=True)
    
    # 2. Merge News Features
    if not news_market_df.empty:
        df = pd.merge(df, news_market_df, on='trade_date', how='left')
    else:
        df['news_market_impact'] = 0.0
        
    if not news_stock_sector_df.empty:
        df = pd.merge(df, news_stock_sector_df, on=['trade_date', 'ts_code'], how='left')
    else:
        df['news_stock_impact'] = 0.0
        df['news_sector_impact'] = 0.0
        
    df[['news_market_impact', 'news_stock_impact', 'news_sector_impact']] = df[['news_market_impact', 'news_stock_impact', 'news_sector_impact']].fillna(0.0)
    
    return df

def train_and_backtest(use_news_features=False):
    # Determine train and test periods
    train_start, train_end = '20221031', '20230315'
    test_start, test_end   = '20230316', '20260316'
    
    print(f"\n--- Running evaluation (News Features = {use_news_features}) ---")
    
    # We load everything matching train and test
    df_train = load_dataset(train_start, train_end)
    df_test = load_dataset(test_start, test_end)
    
    base_feats = ['hot_rank_pct', 'pct_chg', 'amount', 'vol', 'chip_concentration', 'winner_rate']
    news_feats = ['news_market_impact', 'news_stock_impact', 'news_sector_impact']
    
    feats = base_feats + news_feats if use_news_features else base_feats
    
    # Train
    print("Training model...")
    X_train = df_train[feats].fillna(0)
    y_train = df_train['label']
    
    pos = df_train[y_train == 1]
    neg = df_train[y_train == 0].sample(min(len(pos)*2, len(df_train)-len(pos)), random_state=42)
    train_bal = pd.concat([pos, neg])
    
    model = xgb.XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.1, eval_metric='logloss')
    model.fit(train_bal[feats], train_bal['label'])
    
    # Backtest
    print("Running Backtest...")
    initial_cap = 100000.0
    capital = initial_cap
    equity = []
    
    test_dates = sorted(df_test['trade_date'].unique())
    
    for d_curr in test_dates:
        day_df = df_test[df_test['trade_date'] == d_curr].copy()
        if day_df.empty:
            continue
            
        X_test = day_df[feats].fillna(0)
        day_df['prob'] = model.predict_proba(X_test)[:, 1]
        
        # Select top 3
        picks = day_df.sort_values('prob', ascending=False).head(3)
        
        day_pnl = 0
        cash_per_stock = capital / 3
        
        for _, row in picks.iterrows():
            # T+1 open and limit logic
            up_limit = round(row['pre_close'] * 1.1, 2)
            if row['ts_code'].startswith(('30', '68')):
                up_limit = round(row['pre_close'] * 1.2, 2)
                
            open_next = row['open_next']
            high_next = row['high']
            close_next = row['close_next']
            
            if pd.isna(open_next) or open_next >= up_limit:
                continue
                
            buy_px = open_next * 1.001
            # exit logic
            if (high_next / open_next - 1) > 0.035:
                sell_px = open_next * 1.03 * 0.9985 
            else:
                sell_px = close_next * 0.9985
                
            pnl = (sell_px - buy_px) / buy_px * cash_per_stock
            day_pnl += pnl
            
        capital += day_pnl
        equity.append({'date': d_curr, 'nav': capital})
        
    eq_df = pd.DataFrame(equity)
    total_ret = (capital - initial_cap) / initial_cap
    print(f"Final Capital: {capital:,.2f}")
    print(f"Total Return: {total_ret*100:.2f}%\n")
    
    if use_news_features:
        eq_df.to_csv('news_impact_equity.csv', index=False)
        
    return eq_df

if __name__ == '__main__':
    print("Evaluating Strategy...")
    # Baseline
    eq_base = train_and_backtest(use_news_features=False)
    # With News Features
    eq_news = train_and_backtest(use_news_features=True)
    
    # Plotting comparison
    plt.figure(figsize=(10, 6))
    if not eq_base.empty:
        plt.plot(pd.to_datetime(eq_base['date']), eq_base['nav'], label='Baseline (No News)')
    if not eq_news.empty:
        plt.plot(pd.to_datetime(eq_news['date']), eq_news['nav'], label='With News Features')
    
    plt.title('Backtest Comparison (2023-03-16 to 2026-03-16)')
    plt.legend()
    plt.grid(True)
    plt.savefig('news_impact_comparison.png')
    print("Evaluation completed. Saved plot to news_impact_comparison.png")

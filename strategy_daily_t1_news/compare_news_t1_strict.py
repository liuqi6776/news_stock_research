import os
import sys
import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.preprocessing import RobustScaler

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from infra_data.storage import DataStorage
from panqian_processor import process_panqian_news

# Paths
DATA_DIR  = r'D:\iquant_data\data_v2'
PRICE_DIR = os.path.join(DATA_DIR, 'data_day1')
RANK_DIR  = os.path.join(DATA_DIR, 'ths_rank1')
CHIP_DIR  = os.path.join(DATA_DIR, 'cyq1')
OTHER_DIR = os.path.join(DATA_DIR, 'other_day1')

# Configuration
INITIAL_CAP = 1000000.0
TOP_N = 3
SLIPPAGE = 0.001
COMMISSION = 0.0003
STAMP_DUTY = 0.0005
MARKET_CAP_LIMIT = 5000000 # 50 Billion (Unit: 10k)

def train_model(train_df, features):
    sub = train_df.dropna(subset=features + ['label']).copy()
    if len(sub) < 100: return None, None
    X = sub[features].replace([np.inf, -np.inf], 0).fillna(0)
    y = sub['label']
    scaler = RobustScaler()
    X_s = scaler.fit_transform(X)
    model = xgb.XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.05, n_jobs=-1, eval_metric='logloss')
    model.fit(X_s, y)
    return model, scaler

def run_strict_t1_backtest(news_dir, label_name):
    print(f"\n>>> Running Strict T+1 Backtest for: {label_name}")
    start_date, end_date = '20240101', '20260327'
    train_start = '20220101'
    
    dates = sorted([f.replace('.parquet', '') for f in os.listdir(PRICE_DIR) if f.endswith('.parquet')])
    test_dates = [d for d in dates if start_date <= d <= end_date]
    
    # Pre-process News
    news_mkt, news_stk = process_panqian_news(news_dir, '20220101', end_date)
    news_stk['trade_date_str'] = news_stk['trade_date'].dt.strftime('%Y%m%d')
    # Handle duplicate news entries for the same stock on the same day
    news_stk = news_stk.groupby(['trade_date_str', 'ts_code'])[['news_stock_impact', 'news_sector_impact']].mean().reset_index()
    news_lookup = news_stk.set_index(['trade_date_str', 'ts_code'])[['news_stock_impact', 'news_sector_impact']].to_dict('index')

    capital = INITIAL_CAP
    equity = []
    features = ['hot_rank_pct', 'chip_concentration', 'winner_rate', 'news_stock_impact', 'news_sector_impact']
    
    # Monthly WFO
    current_model, cur_scaler = None, None
    last_train_month = -1

    for i in tqdm(range(len(test_dates)-2), desc=f"Backtesting {label_name}"):
        d_curr, d_buy, d_sell = test_dates[i], test_dates[i+1], test_dates[i+2]
        
        # 1. Training (Monthly)
        curr_dt = pd.to_datetime(d_curr)
        if curr_dt.month != last_train_month:
            # Re-train using data before d_curr
            t_end = d_curr
            # For simplicity, we use a fixed window here or pre-calculate labels
            # To be efficient in this script, we'll try to load a segment
            # but for a true WFO we'd need a larger loop.
            # Here we skip retraining every step for performance but update monthly.
            last_train_month = curr_dt.month
        
        # 2. Daily Prediction (on d_curr close for d_buy entry)
        p_rank = os.path.join(RANK_DIR, f"{d_curr}.parquet")
        p_chip = os.path.join(CHIP_DIR, f"{d_curr}.parquet")
        p_price = os.path.join(PRICE_DIR, f"{d_curr}.parquet")
        p_other = os.path.join(OTHER_DIR, f"{d_curr}.parquet")
        
        if not all(os.path.exists(p) for p in [p_rank, p_chip, p_price, p_other]):
            equity.append({'date': pd.to_datetime(d_buy), 'nav': capital})
            continue

        rank_df = pd.read_parquet(p_rank)
        rank_df['hot_rank_pct'] = rank_df['hot'].rank(pct=True)
        chip_df = pd.read_parquet(p_chip)
        chip_df['chip_concentration'] = (chip_df['cost_85pct'] - chip_df['cost_15pct']) / (chip_df['cost_50pct'] + 1e-8)
        price_df = pd.read_parquet(p_price, columns=['ts_code', 'close', 'pre_close'])
        other_df = pd.read_parquet(p_other, columns=['ts_code', 'circ_mv'])
        
        df = pd.merge(rank_df[['ts_code', 'hot_rank_pct']], price_df, on='ts_code')
        df = pd.merge(df, chip_df[['ts_code', 'chip_concentration', 'winner_rate']], on='ts_code')
        df = pd.merge(df, other_df, on='ts_code')
        
        # FILTERS: < 50B, Non-688
        df = df[(df['circ_mv'] <= MARKET_CAP_LIMIT) & (~df['ts_code'].str.startswith('688'))]
        
        # Determine the correct news lookup date based on news source type
        # news-major1 (Panqian) logic: article date is the same as the trading date (d_buy)
        # news-major (Post-market) logic: article date is the day before the trading date (d_curr)
        news_date = d_buy if 'Major1' in label_name else d_curr
        
        df['news_stock_impact'] = df['ts_code'].apply(lambda x: news_lookup.get((news_date, x), {}).get('news_stock_impact', 0.0))
        df['news_sector_impact'] = df['ts_code'].apply(lambda x: news_lookup.get((news_date, x), {}).get('news_sector_impact', 0.0))
        
        # Simple Scoring for this comparison (or pre-trained model)
        # Since training a full WFO in one script is slow, let's use a weighted rank 
        # specifically focused on the news impact to see the delta.
        df['score'] = df['news_stock_impact'] * 2.0 + df['news_sector_impact'] * 1.0 + df['hot_rank_pct'] * 0.5
        picks = df.sort_values('score', ascending=False).head(TOP_N)
        
        # 3. Execution (Buy d_buy Open, Sell d_sell Open)
        next_df = pd.read_parquet(os.path.join(PRICE_DIR, f"{d_buy}.parquet"), columns=['ts_code', 'open', 'pre_close'])
        sell_df = pd.read_parquet(os.path.join(PRICE_DIR, f"{d_sell}.parquet"), columns=['ts_code', 'open'])
        
        day_pnl = 0
        alloc = capital / TOP_N
        
        for _, row in picks.iterrows():
            code = row['ts_code']
            b_px = next_df[next_df['ts_code']==code]
            s_px = sell_df[sell_df['ts_code']==code]
            
            if b_px.empty or s_px.empty: continue
            
            open_buy = b_px.iloc[0]['open']
            pre_close = b_px.iloc[0]['pre_close']
            open_sell = s_px.iloc[0]['open']
            
            # Limit up filter
            limit_ratio = 1.195 if code.startswith('300') else 1.095
            if open_buy >= round(pre_close * limit_ratio, 2): continue
            
            # Simple T+1 Return
            ret = (open_sell / open_buy) - 1
            ret -= (SLIPPAGE * 2 + COMMISSION * 2 + STAMP_DUTY) # Rough round-trip fee
            day_pnl += alloc * ret
            
        capital += day_pnl
        equity.append({'date': pd.to_datetime(d_buy), 'nav': capital})
        
    return pd.DataFrame(equity)

if __name__ == "__main__":
    major_path = r'D:\iquant_data\data_v2\news_major'
    major1_path = r'D:\iquant_data\data_v2\news_major1'
    
    eq_major = run_strict_t1_backtest(major_path, "News Major")
    eq_major1 = run_strict_t1_backtest(major1_path, "News Major1")
    
    plt.figure(figsize=(12, 7))
    plt.plot(eq_major['date'], eq_major['nav'], label='News Major (Strict T+1)', color='royalblue')
    plt.plot(eq_major1['date'], eq_major1['nav'], label='News Major1 (Strict T+1)', color='tomato')
    plt.title('Strict T+1 Comparison: News Major vs Major1 (<50B, Non-688)')
    plt.xlabel('Date')
    plt.ylabel('NAV')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    out_dir = r'c:\Users\liuqi\quant_system_v2\strategy_daily_t1_news'
    plt.savefig(os.path.join(out_dir, 'news_t1_comparison_strict.png'), dpi=150)
    print(f"\nFinal major NAV: {eq_major.iloc[-1]['nav']:.2f}")
    print(f"Final major1 NAV: {eq_major1.iloc[-1]['nav']:.2f}")
    print(f"Plot saved to {os.path.join(out_dir, 'news_t1_comparison_strict.png')}")

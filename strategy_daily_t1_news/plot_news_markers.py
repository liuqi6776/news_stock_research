"""
Backtest Annotation: Mark News-Impact Days on Top-10 Equity Curve
- Period: 2024-01-01 to 2026-03-26
- Track days where at least one stock in Top-10 has news_stock_impact > 0
"""
import os
import sys
import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from infra_data.storage import DataStorage
from train_model import train_daily_model

DATA_DIR  = r'D:\iquant_data\data_v2'
PRICE_DIR = os.path.join(DATA_DIR, 'data_day1')
RANK_DIR  = os.path.join(DATA_DIR, 'ths_rank1')
CHIP_DIR  = os.path.join(DATA_DIR, 'cyq1')
OTHER_DIR = os.path.join(DATA_DIR, 'other_day1')

def run_annotated_backtest(start_date='20240101', end_date='20260326', train_start='20220101', step_months=3):
    print("=== Annotated Top-10 Backtest (Marking News Impact) ===")
    dates = sorted([f.replace('.parquet', '') for f in os.listdir(PRICE_DIR) if f.endswith('.parquet')])
    
    current_test_start = pd.to_datetime(start_date)
    final_end_date = pd.to_datetime(end_date)
    fixed_train_start_dt = pd.to_datetime(train_start)
    
    storage = DataStorage()
    initial_cap = 100000.0
    capital = initial_cap
    equity = []
    news_driven_dates = []

    while current_test_start <= final_end_date:
        current_test_end = current_test_start + pd.DateOffset(months=step_months) - pd.Timedelta(days=1)
        if current_test_end > final_end_date: current_test_end = final_end_date
        
        t0, t1 = fixed_train_start_dt.strftime('%Y%m%d'), (current_test_start - pd.Timedelta(days=1)).strftime('%Y%m%d')
        s0, s1 = current_test_start.strftime('%Y%m%d'), current_test_end.strftime('%Y%m%d')
        
        model, feats = train_daily_model(t0, t1, model_path=None)
        if model is None: 
            current_test_start += pd.DateOffset(months=step_months)
            continue
            
        test_dates = [d for d in dates if s0 <= d <= s1]
        test_series = pd.Series([pd.to_datetime(d) for d in test_dates]).sort_values()
        news_mkt, news_stk = storage.load_news_data(s0, s1, test_series)
        if not news_stk.empty: news_stk['trade_date'] = news_stk['trade_date'].dt.strftime('%Y%m%d')

        for i in range(len(test_dates)-1):
            d_curr, d_next = test_dates[i], test_dates[i+1]
            p_price = os.path.join(PRICE_DIR, f"{d_curr}.parquet")
            p_rank = os.path.join(RANK_DIR, f"{d_curr}.parquet")
            p_chip = os.path.join(CHIP_DIR, f"{d_curr}.parquet")
            p_other = os.path.join(OTHER_DIR, f"{d_curr}.parquet")
            
            if not all(os.path.exists(p) for p in [p_price, p_rank, p_chip, p_other]):
                continue
            
            # Load features
            rank_df = pd.read_parquet(p_rank)
            rank_df['hot_rank_pct'] = rank_df['hot'].rank(pct=True)
            chip_df = pd.read_parquet(p_chip)
            chip_df['chip_concentration'] = (chip_df['cost_85pct'] - chip_df['cost_15pct']) / (chip_df['cost_50pct'] + 1e-8)
            price_df = pd.read_parquet(p_price, columns=['ts_code', 'close', 'pct_chg', 'amount', 'vol', 'pre_close'])
            other_df = pd.read_parquet(p_other, columns=['ts_code', 'circ_mv'])
            
            df = pd.merge(rank_df[['ts_code', 'hot_rank_pct']], price_df, on='ts_code')
            df = pd.merge(df, chip_df[['ts_code', 'chip_concentration', 'winner_rate']], on='ts_code')
            df = pd.merge(df, other_df, on='ts_code', how='left')
            df = df[(~df['ts_code'].str.startswith('688')) & (df['circ_mv'] <= 500000)]
            
            if not news_stk.empty:
                df = pd.merge(df, news_stk[news_stk['trade_date']==d_next][['ts_code', 'news_stock_impact']], on='ts_code', how='left')
            else: df['news_stock_impact'] = 0.0
            
            df['news_market_impact'] = 0.0 
            X = df[feats].fillna(0)
            try:
                df['prob'] = model.predict_proba(X)[:, 1]
            except Exception as e:
                df['prob'] = 0
            
            # Pick Top 10
            picks = df.sort_values('prob', ascending=False).head(10)
            
            # Check if any pick has news impact
            if not picks.empty and (picks['news_stock_impact'] > 0).any():
                news_driven_dates.append(pd.to_datetime(d_next))
            
            # Calculate PnL
            next_df = pd.read_parquet(os.path.join(PRICE_DIR, f"{d_next}.parquet"), columns=['ts_code', 'open', 'high', 'close', 'pre_close'])
            day_pnl = 0
            for _, row in picks.iterrows():
                nxt = next_df[next_df['ts_code'] == row['ts_code']]
                if nxt.empty or nxt.iloc[0]['open'] >= nxt.iloc[0]['pre_close'] * 1.095: continue
                buy_p, high_p, close_p = nxt.iloc[0]['open'], nxt.iloc[0]['high'], nxt.iloc[0]['close']
                sell_p = buy_p * 1.04 if high_p >= buy_p * 1.04 else close_p
                day_pnl += (capital/len(picks)) * ((sell_p/buy_p)-1-0.0015)
            capital += day_pnl
            equity.append({'date': pd.to_datetime(d_next), 'nav': capital})

        current_test_start += pd.DateOffset(months=step_months)

    # Plotting
    eq_df = pd.DataFrame(equity)
    plt.figure(figsize=(15, 8))
    plt.plot(eq_df['date'], eq_df['nav'], label='Top-10 Equity Curve', color='royalblue', alpha=0.8)
    
    # Mark news-driven days
    news_navs = eq_df[eq_df['date'].isin(news_driven_dates)]
    plt.scatter(news_navs['date'], news_navs['nav'], color='red', s=15, label='News Impact Days (Top 10 has News)', zorder=5)
    
    plt.title('Top-10 Strategy: Equity Curve with News-Impact Days Highlighted', fontsize=14)
    plt.xlabel('Date')
    plt.ylabel('Portfolio Value')
    plt.legend()
    plt.grid(True, alpha=0.3)
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'news_annotated_equity_top10.png')
    plt.savefig(out_path, dpi=150)
    print(f"Annotated chart saved to: {out_path}")
    print(f"Total News-Impact Days: {len(news_driven_dates)}")

if __name__ == "__main__":
    run_annotated_backtest()

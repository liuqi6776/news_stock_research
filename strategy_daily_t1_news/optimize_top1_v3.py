"""
Enhanced Optimization Script for Top-1 Filtering
- Saves NAV data to CSV for verification
- Prints detailed trade statistics
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
from panqian_processor import process_panqian_news

DATA_DIR  = r'D:\iquant_data\data_v2'
PRICE_DIR = os.path.join(DATA_DIR, 'data_day1')
RANK_DIR  = os.path.join(DATA_DIR, 'ths_rank1')
CHIP_DIR  = os.path.join(DATA_DIR, 'cyq1')
OTHER_DIR = os.path.join(DATA_DIR, 'other_day1')
NEWS_DIR  = os.path.join(DATA_DIR, 'news_major1')

def run_filtered_top1_v3():
    print(f"=== Top-1 Filtering Analysis v3 ===")
    start_date, end_date = '20240101', '20260327'
    train_start = '20220101'
    step_months = 3
    
    dates = sorted([f.replace('.parquet', '') for f in os.listdir(PRICE_DIR) if f.endswith('.parquet')])
    current_test_start = pd.to_datetime(start_date)
    final_end_date = pd.to_datetime(end_date)
    fixed_train_start_dt = pd.to_datetime(train_start)
    
    # Use more inclusive thresholds to ENSURE trades happen
    thresholds = [
        (0.5, 0),   # Benchmark
        (0.5, 1),   # Filter only by News
        (0.52, 1),  # Very light filter
        (0.55, 1),  # Recommended
    ]
    
    capitals = {t: 100000.0 for t in thresholds}
    equities = {t: [] for t in thresholds}
    trades_log = {t: 0 for t in thresholds}
    
    feats = ['hot_rank_pct', 'chip_concentration', 'winner_rate', 'news_market_impact', 'news_stock_impact', 'news_sector_impact']

    while current_test_start <= final_end_date:
        current_test_end = current_test_start + pd.DateOffset(months=step_months) - pd.Timedelta(days=1)
        if current_test_end > final_end_date: current_test_end = final_end_date
        
        t0, t1 = fixed_train_start_dt.strftime('%Y%m%d'), (current_test_start - pd.Timedelta(days=1)).strftime('%Y%m%d')
        s0, s1 = current_test_start.strftime('%Y%m%d'), current_test_end.strftime('%Y%m%d')
        
        model, _ = train_daily_model(t0, t1, model_path=None)
        if model is None:
            current_test_start += pd.DateOffset(months=step_months)
            continue
            
        news_mkt, news_stk = process_panqian_news(NEWS_DIR, s0, s1)
        if not news_stk.empty: 
            news_stk['trade_date'] = news_stk['trade_date'].dt.strftime('%Y%m%d')

        test_dates = [d for d in dates if s0 <= d <= s1]
        for i in range(len(test_dates)-1):
            d_curr, d_next = test_dates[i], test_dates[i+1]
            p_price = os.path.join(PRICE_DIR, f"{d_curr}.parquet")
            p_rank = os.path.join(RANK_DIR, f"{d_curr}.parquet")
            p_chip = os.path.join(CHIP_DIR, f"{d_curr}.parquet")
            p_other = os.path.join(OTHER_DIR, f"{d_curr}.parquet")
            
            if not all(os.path.exists(p) for p in [p_price, p_rank, p_chip, p_other]): continue
            
            # Load features
            rank_df = pd.read_parquet(p_rank); rank_df['hot_rank_pct'] = rank_df['hot'].rank(pct=True)
            chip_df = pd.read_parquet(p_chip); chip_df['chip_concentration'] = (chip_df['cost_85pct'] - chip_df['cost_15pct']) / (chip_df['cost_50pct'] + 1e-8)
            price_df = pd.read_parquet(p_price, columns=['ts_code', 'close', 'pct_chg'])
            other_df = pd.read_parquet(p_other, columns=['ts_code', 'circ_mv'])
            
            df = pd.merge(rank_df[['ts_code', 'hot_rank_pct']], price_df, on='ts_code')
            df = pd.merge(df, chip_df[['ts_code', 'chip_concentration', 'winner_rate']], on='ts_code')
            df = pd.merge(df, other_df, on='ts_code', how='left')
            df = df[(~df['ts_code'].str.startswith('688')) & (df['circ_mv'] <= 500000)]
            
            if not news_stk.empty:
                daily_news = news_stk[news_stk['trade_date']==d_next]
                df = pd.merge(df, daily_news[['ts_code', 'news_stock_impact', 'news_sector_impact']], on='ts_code', how='left')
            else:
                df['news_stock_impact'] = 0.0; df['news_sector_impact'] = 0.0
            
            df = df.fillna(0)
            df['news_market_impact'] = 0.0
            
            # Model features (must be exact 5)
            model_feats = ['hot_rank_pct', 'chip_concentration', 'winner_rate', 'news_market_impact', 'news_stock_impact']
            df['prob'] = 0.0
            if not df.empty:
                X = df[model_feats]
                try:
                    probs = model.predict_proba(X)
                    if probs.shape[1] > 1:
                        df['prob'] = probs[:, 1]
                    else:
                        df['prob'] = probs[:, 0] # fallback if only one class
                except Exception as e:
                    print(f"Prediction error on {d_curr}: {e}")
            
            # Filtering features (can include sector)
            df['news_max'] = df[['news_stock_impact', 'news_sector_impact']].max(axis=1)
            
            next_df = pd.read_parquet(os.path.join(PRICE_DIR, f"{d_next}.parquet"), columns=['ts_code', 'open', 'close', 'pre_close'])

            for t_prob, t_news in thresholds:
                filtered = df[(df['prob'] >= t_prob) & (df['news_max'] >= t_news)]
                pick = filtered.sort_values('prob', ascending=False).head(1)
                
                day_pnl = 0
                if not pick.empty:
                    nxt = next_df[next_df['ts_code'] == pick.iloc[0]['ts_code']]
                    if not nxt.empty and nxt.iloc[0]['open'] < nxt.iloc[0]['pre_close'] * 1.095:
                        ret = (nxt.iloc[0]['close'] / nxt.iloc[0]['open']) - 1 - 0.0015
                        day_pnl = capitals[(t_prob, t_news)] * ret
                        trades_log[(t_prob, t_news)] += 1
                
                capitals[(t_prob, t_news)] += day_pnl
                equities[(t_prob, t_news)].append({'date': d_next, 'nav': capitals[(t_prob, t_news)]})

        current_test_start += pd.DateOffset(months=step_months)

    # Save and Plot
    plt.figure(figsize=(15, 8))
    summary_data = []
    for (tp, tn), data in equities.items():
        if data:
            edf = pd.DataFrame(data)
            edf.to_csv(f"panqianjiyao_daily/nav_P{tp}_N{tn}.csv", index=False)
            label = f"Prob>{tp}, News>{tn} (Trades:{trades_log[(tp, tn)]})"
            plt.plot(pd.to_datetime(edf['date']), edf['nav'], label=label)
            summary_data.append(label)
    
    plt.title('Top-1 Filtered Strategy Comparison v3')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('panqianjiyao_daily/top1_filtered_v3.png', dpi=150)
    print("\n".join(summary_data))
    print("Files saved to panqianjiyao_daily/")

if __name__ == "__main__":
    run_filtered_top1_v3()

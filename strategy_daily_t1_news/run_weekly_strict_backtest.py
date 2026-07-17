"""
Weekly Strictly Filtered WFO Backtest (Top-10)
- Weekly Retraining
- No 688 (Star Market)
- No Buy if Open at Limit-Up (GEM 20%, Main Board 10%)
- 1-day Prediction Offset
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
from train_model import train_daily_model
from panqian_processor import process_panqian_news

DATA_DIR  = r'D:\iquant_data\data_v2'
PRICE_DIR = os.path.join(DATA_DIR, 'data_day1')
RANK_DIR  = os.path.join(DATA_DIR, 'ths_rank1')
CHIP_DIR  = os.path.join(DATA_DIR, 'cyq1')
OTHER_DIR = os.path.join(DATA_DIR, 'other_day1')
NEWS_DIR  = os.path.join(DATA_DIR, 'news_major1')

def run_weekly_strict_backtest():
    print(f"=== Weekly Strict WFO Backtest - TopN Comparison ===")
    start_date, end_date = '20240101', '20260327'
    train_start = '20220101'
    step_weeks = 1
    
    dates = sorted([f.replace('.parquet', '') for f in os.listdir(PRICE_DIR) if f.endswith('.parquet')])
    current_test_start_orig = pd.to_datetime(start_date)
    final_end_date = pd.to_datetime(end_date)
    fixed_train_start_dt = pd.to_datetime(train_start)
    
    top_n_list = [1, 10]
    all_equities = {}
    
    feats = ['hot_rank_pct', 'chip_concentration', 'winner_rate', 'news_market_impact', 'news_stock_impact']

    for top_n in top_n_list:
        print(f"\n--- Testing Top {top_n} ---")
        capital = 1000000.0
        equity = []
        current_test_start = current_test_start_orig

        while current_test_start <= final_end_date:
            current_test_end = current_test_start + pd.DateOffset(weeks=step_weeks) - pd.Timedelta(days=1)
            if current_test_end > final_end_date: current_test_end = final_end_date
            
            t0, t1 = fixed_train_start_dt.strftime('%Y%m%d'), (current_test_start - pd.Timedelta(days=1)).strftime('%Y%m%d')
            s0, s1 = current_test_start.strftime('%Y%m%d'), current_test_end.strftime('%Y%m%d')
            
            model, _ = train_daily_model(t0, t1, model_path=None)
            if model is None:
                current_test_start += pd.DateOffset(weeks=step_weeks)
                continue
                
            news_mkt, news_stk = process_panqian_news(NEWS_DIR, s0, s1)
            if not news_stk.empty: news_stk['trade_date'] = news_stk['trade_date'].dt.strftime('%Y%m%d')

            test_dates = [d for d in dates if s0 <= d <= s1]
            for i in range(len(test_dates)-1):
                d_curr, d_next = test_dates[i], test_dates[i+1]
                p_price = os.path.join(PRICE_DIR, f"{d_curr}.parquet")
                p_rank = os.path.join(RANK_DIR, f"{d_curr}.parquet")
                p_chip = os.path.join(CHIP_DIR, f"{d_curr}.parquet")
                p_other = os.path.join(OTHER_DIR, f"{d_curr}.parquet")
                if not all(os.path.exists(p) for p in [p_price, p_rank, p_chip, p_other]): continue
                
                # Prep
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
                
                df = df.fillna(0); df['news_market_impact'] = 0.0
                X = df[feats]
                try:
                    probs = model.predict_proba(X)
                    df['prob'] = probs[:, 1] if probs.shape[1] > 1 else 0
                except: df['prob'] = 0
                
                # Execution
                next_df = pd.read_parquet(os.path.join(PRICE_DIR, f"{d_next}.parquet"), columns=['ts_code', 'open', 'close', 'pre_close'])
                picks = df.sort_values('prob', ascending=False).head(top_n + 20) 
                final_picks = []
                for _, row in picks.iterrows():
                    if len(final_picks) >= top_n: break
                    code = row['ts_code']
                    nxt = next_df[next_df['ts_code'] == code]
                    if nxt.empty: continue
                    pre_close, open_px = nxt.iloc[0]['pre_close'], nxt.iloc[0]['open']
                    limit_ratio = 1.195 if code.startswith('300') else 1.095
                    if open_px >= round(pre_close * limit_ratio, 2): continue
                    final_picks.append({'ret': (nxt.iloc[0]['close'] / open_px) - 1 - 0.0015})
                
                if final_picks: capital += capital * (len(final_picks) / top_n) * np.mean([p['ret'] for p in final_picks])
                equity.append({'date': pd.to_datetime(d_next), 'nav': capital})

            current_test_start += pd.DateOffset(weeks=step_weeks)
        
        all_equities[top_n] = pd.DataFrame(equity)

    # Final Plot
    plt.figure(figsize=(15, 8))
    colors = {1: 'gold', 10: 'royalblue'}
    for n in top_n_list:
        df_eq = all_equities[n]
        plt.plot(df_eq['date'], df_eq['nav'], label=f'Weekly Strict Top-{n}', color=colors[n])
        df_eq.to_csv(f'panqianjiyao_daily/weekly_strict_nav_top{n}.csv', index=False)
    
    plt.title('Weekly WFO Strict Comparison: Top-1 vs Top-10 (No 688, No Limit-up Open)')
    plt.legend(); plt.grid(True, alpha=0.3)
    plt.savefig('panqianjiyao_daily/weekly_strict_comparison_top1.png', dpi=150)
    print("Optimization Plot Saved.")

if __name__ == "__main__":
    run_weekly_strict_backtest()

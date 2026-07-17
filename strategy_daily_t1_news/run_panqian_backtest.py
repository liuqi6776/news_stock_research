"""
Backtest for Panqian Jiyao (news_major1)
- Period: 2024-01-01 to 2026-03-27
- Model retraining: every 3 months
- Offset: News on Date T is used to predict stocks for Date T (using features from T-1).
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

def run_panqian_comparison(top_n_list=(1, 3, 5, 10, 20)):
    print(f"=== Panqian Jiyao Backtest (news_major1) - Top {top_n_list} ===")
    start_date, end_date = '20240101', '20260327'
    train_start = '20220101'
    step_months = 3
    
    dates = sorted([f.replace('.parquet', '') for f in os.listdir(PRICE_DIR) if f.endswith('.parquet')])
    current_test_start = pd.to_datetime(start_date)
    final_end_date = pd.to_datetime(end_date)
    fixed_train_start_dt = pd.to_datetime(train_start)
    
    initial_cap = 100000.0
    capitals = {n: initial_cap for n in top_n_list}
    equities = {n: [] for n in top_n_list}
    
    feats = ['hot_rank_pct', 'chip_concentration', 'winner_rate', 'news_market_impact', 'news_stock_impact', 'news_sector_impact']

    while current_test_start <= final_end_date:
        current_test_end = current_test_start + pd.DateOffset(months=step_months) - pd.Timedelta(days=1)
        if current_test_end > final_end_date: current_test_end = final_end_date
        
        t0, t1 = fixed_train_start_dt.strftime('%Y%m%d'), (current_test_start - pd.Timedelta(days=1)).strftime('%Y%m%d')
        s0, s1 = current_test_start.strftime('%Y%m%d'), current_test_end.strftime('%Y%m%d')
        
        print(f"\n>>> [WFO] Train: {t0}~{t1} | Test: {s0}~{s1}")
        model, _ = train_daily_model(t0, t1, model_path=None)
        if model is None:
            current_test_start += pd.DateOffset(months=step_months)
            continue
            
        # Load all panqian news for this test segment
        news_mkt, news_stk = process_panqian_news(NEWS_DIR, s0, s1)
        if not news_stk.empty: news_stk['trade_date'] = news_stk['trade_date'].dt.strftime('%Y%m%d')

        test_dates = [d for d in dates if s0 <= d <= s1]
        
        for i in tqdm(range(len(test_dates)-1), desc=f"Testing {s0}"):
            d_curr, d_next = test_dates[i], test_dates[i+1]
            
            # Prediction for d_next using features from d_curr and News for d_next
            p_price = os.path.join(PRICE_DIR, f"{d_curr}.parquet")
            p_rank = os.path.join(RANK_DIR, f"{d_curr}.parquet")
            p_chip = os.path.join(CHIP_DIR, f"{d_curr}.parquet")
            p_other = os.path.join(OTHER_DIR, f"{d_curr}.parquet")
            
            if not all(os.path.exists(p) for p in [p_price, p_rank, p_chip, p_other]): continue
            
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
            
            # Map News for d_next (the day we are predicting)
            if not news_stk.empty:
                daily_news = news_stk[news_stk['trade_date']==d_next]
                df = pd.merge(df, daily_news[['ts_code', 'news_stock_impact', 'news_sector_impact']], on='ts_code', how='left')
            else:
                df['news_stock_impact'] = 0.0
                df['news_sector_impact'] = 0.0
            
            df['news_market_impact'] = 0.0 
            X = df[feats].fillna(0)
            try:
                df['prob'] = model.predict_proba(X)[:, 1]
            except: df['prob'] = 0
            
            for n in top_n_list:
                picks = df.sort_values('prob', ascending=False).head(n)
                # Next day pnl
                next_df = pd.read_parquet(os.path.join(PRICE_DIR, f"{d_next}.parquet"), columns=['ts_code', 'open', 'high', 'close', 'pre_close'])
                day_pnl = 0
                for _, row in picks.iterrows():
                    nxt = next_df[next_df['ts_code'] == row['ts_code']]
                    if nxt.empty or nxt.iloc[0]['open'] >= nxt.iloc[0]['pre_close'] * 1.095: continue
                    ret = (nxt.iloc[0]['close'] / nxt.iloc[0]['open']) - 1 - 0.0015
                    day_pnl += (capitals[n]/n) * ret
                capitals[n] += day_pnl
                equities[n].append({'date': pd.to_datetime(d_next), 'nav': capitals[n]})

        current_test_start += pd.DateOffset(months=step_months)

    # Plot
    plt.figure(figsize=(15, 8))
    colors = ['gold', 'royalblue', 'green', 'tomato', 'purple']
    for i, n in enumerate(top_n_list):
        if equities[n]:
            df_eq = pd.DataFrame(equities[n])
            plt.plot(df_eq['date'], df_eq['nav'], label=f'Top-{n} (news_major1)', color=colors[i])
            
    plt.title('Panqian Jiyao (news_major1) Backtest Comparison (2024-2026)')
    plt.xlabel('Date')
    plt.ylabel('NAV')
    plt.legend()
    plt.grid(True, alpha=0.3)
    out_path = os.path.join('panqianjiyao_daily', 'panqian_comparison.png')
    plt.savefig(out_path, dpi=150)
    print(f"Comparison chart saved to: {out_path}")

if __name__ == "__main__":
    run_panqian_comparison()

import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import sys
import matplotlib.pyplot as plt

# Ensure this script can import quant_system_v2 modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from infra_data.storage import DataStorage
from train_model import train_daily_model

DATA_DIR = r'D:\iquant_data\data_v2'
PRICE_DIR = os.path.join(DATA_DIR, 'data_day1')
RANK_DIR  = os.path.join(DATA_DIR, 'ths_rank1')
CHIP_DIR  = os.path.join(DATA_DIR, 'cyq1')
OTHER_DIR = os.path.join(DATA_DIR, 'other_day1')

def run_wfo_backtest(start_date='20240101', end_date='20261231', train_start='20220101', step_months=1):
    dates = sorted([f.replace('.parquet', '') for f in os.listdir(PRICE_DIR) if f.endswith('.parquet')])
    
    current_test_start = pd.to_datetime(start_date)
    final_end_date = pd.to_datetime(end_date)
    fixed_train_start_dt = pd.to_datetime(train_start)
    
    storage = DataStorage()
    
    initial_cap = 100000.0
    capital = initial_cap
    equity = []
    
    print(f"=== Starting Walk-Forward Optimization (Expanding Window) ===")
    print(f"Test Period: {start_date} to {end_date}")
    print(f"Rolling Rules: Expand training data infinitely from {train_start}, predict forward {step_months} month(s).")
    
    while current_test_start <= final_end_date:
        current_test_end = current_test_start + pd.DateOffset(months=step_months) - pd.Timedelta(days=1)
        if current_test_end > final_end_date:
            current_test_end = final_end_date
            
        train_start_dt = fixed_train_start_dt
        train_end_dt = current_test_start - pd.Timedelta(days=1)
        
        train_start_str = train_start_dt.strftime('%Y%m%d')
        train_end_str = train_end_dt.strftime('%Y%m%d')
        test_start_str = current_test_start.strftime('%Y%m%d')
        test_end_str = current_test_end.strftime('%Y%m%d')
        
        print(f"\n>>> [WFO Segment] Next Test Window: {test_start_str} to {test_end_str}")
        
        # 1. Retrain Model
        model, feats = train_daily_model(train_start_str, train_end_str, model_path=None)
        if model is None:
            print(f"Skipping segment {test_start_str}-{test_end_str} due to missing training data.")
            current_test_start = current_test_start + pd.DateOffset(months=step_months)
            continue
            
        # 2. Execute Test Segment
        test_dates = [d for d in dates if test_start_str <= d <= test_end_str]
        if len(test_dates) < 2:
            print("No test dates available in segment. Skipping.")
            current_test_start = current_test_start + pd.DateOffset(months=step_months)
            continue
            
        test_valid_series = pd.Series([pd.to_datetime(d) for d in test_dates]).sort_values()
        news_market_df, news_stock_sector_df = storage.load_news_data(test_start_str, test_end_str, test_valid_series)
        
        if not news_market_df.empty:
            news_market_df['trade_date'] = news_market_df['trade_date'].dt.strftime('%Y%m%d')
        if not news_stock_sector_df.empty:
            news_stock_sector_df['trade_date'] = news_stock_sector_df['trade_date'].dt.strftime('%Y%m%d')
            
        for i in tqdm(range(len(test_dates)-1), desc=f"Predicting & Trading [{test_start_str}-{test_end_str}]", leave=False):
            d_curr, d_next = test_dates[i], test_dates[i+1]
            
            p_rank = os.path.join(RANK_DIR, f"{d_curr}.parquet")
            p_chip = os.path.join(CHIP_DIR, f"{d_curr}.parquet")
            p_price = os.path.join(PRICE_DIR, f"{d_curr}.parquet")
            p_other = os.path.join(OTHER_DIR, f"{d_curr}.parquet")
            if not all(os.path.exists(p) for p in [p_rank, p_chip, p_price, p_other]): continue
            
            rank_df = pd.read_parquet(p_rank)
            rank_df['hot_rank_pct'] = rank_df['hot'].rank(pct=True)
            
            chip_df = pd.read_parquet(p_chip)
            chip_df['chip_concentration'] = (chip_df['cost_85pct'] - chip_df['cost_15pct']) / (chip_df['cost_50pct'] + 1e-8)
            
            price_df = pd.read_parquet(p_price, columns=['ts_code', 'close', 'pct_chg', 'amount', 'vol', 'pre_close'])
            
            df = pd.merge(rank_df[['ts_code', 'hot_rank_pct']], price_df, on='ts_code')
            df = pd.merge(df, chip_df[['ts_code', 'chip_concentration', 'winner_rate']], on='ts_code')
            
            # Market Cap Filter: Ceiling at 50 billion CNY (500,000 units)
            other_df = pd.read_parquet(p_other, columns=['ts_code', 'circ_mv'])
            df = pd.merge(df, other_df, on='ts_code', how='left')
            df = df[df['circ_mv'] <= 500000]
            
            df['trade_date'] = d_next
            
            if not news_market_df.empty:
                df = pd.merge(df, news_market_df, on='trade_date', how='left')
            else:
                df['news_market_impact'] = 0.0
                
            # STRICTLY NO SECTOR
            if not news_stock_sector_df.empty:
                df = pd.merge(df, news_stock_sector_df[['trade_date', 'ts_code', 'news_stock_impact']], on=['trade_date', 'ts_code'], how='left')
            else:
                df['news_stock_impact'] = 0.0
                
            df[['news_market_impact', 'news_stock_impact']] = df[['news_market_impact', 'news_stock_impact']].fillna(0.0)
            
            X = df[feats].fillna(0)
            try:
                df['prob'] = model.predict_proba(X)[:, 1]
            except Exception:
                df['prob'] = 0
                
            # STRICT EXCLUSION: STAR Market (688)
            df = df[~df['ts_code'].str.startswith('688')]
                
            picks = df[df['prob'] > 0.8].sort_values('prob', ascending=False).head(3)
            if picks.empty:
                picks = df.sort_values('prob', ascending=False).head(1)
                
            p_next = os.path.join(PRICE_DIR, f"{d_next}.parquet")
            if not os.path.exists(p_next): break
            next_df = pd.read_parquet(p_next, columns=['ts_code', 'open', 'high', 'close', 'pre_close'])
            
            day_pnl = 0
            alloc = capital / max(1, len(picks))
            
            for _, row in picks.iterrows():
                ts_code = row['ts_code']
                nxt = next_df[next_df['ts_code'] == ts_code]
                if nxt.empty: continue
                
                n_row = nxt.iloc[0]
                open_p, high_p, close_p, pre_close_p = n_row['open'], n_row['high'], n_row['close'], n_row['pre_close']
                
                is_20_pct = ts_code.startswith('300') or ts_code.startswith('688')
                up_limit = round(pre_close_p * 1.2, 2) if is_20_pct else round(pre_close_p * 1.1, 2)
                lockup_threshold = pre_close_p * 1.195 if is_20_pct else pre_close_p * 1.095
                
                # Anti-Lockup filter to simulate realistic execution inability on gap-ups
                if pd.isna(open_p) or open_p >= up_limit or open_p >= lockup_threshold:
                    continue
                    
                buy_price = open_p
                if high_p >= buy_price * 1.04:
                    sell_price = buy_price * 1.04
                else:
                    sell_price = close_p
                    
                ret = (sell_price / buy_price) - 1
                ret -= 0.0015 # fee
                day_pnl += alloc * ret
                
            capital += day_pnl
            equity.append({'date': pd.to_datetime(d_next), 'nav': capital})
            
        current_test_start = current_test_start + pd.DateOffset(months=step_months)
        
    out_dir = os.path.dirname(os.path.abspath(__file__))
    eq_df = pd.DataFrame(equity)
    total_ret = capital / initial_cap - 1
    years = len(eq_df) / 252.0 if len(eq_df) > 0 else 1
    ann_ret = (1+total_ret)**(1/years)-1 if years > 0 else 0
    df_ret = eq_df['nav'].pct_change()
    mdd = ((eq_df['nav'] - eq_df['nav'].cummax()) / eq_df['nav'].cummax()).min()
    vol = df_ret.std() * np.sqrt(252)
    sharpe = ann_ret / vol if vol > 0 else 0
    
    print("\n" + "="*50)
    print("Walk-Forward Optimization (WFO) Results")
    print(f"Final Capital: {capital:,.2f}")
    print(f"Total Return: {total_ret:.2%}")
    print(f"Annual Return: {ann_ret:.2%}")
    print(f"Max Drawdown: {mdd:.2%}")
    print(f"Sharpe Ratio: {sharpe:.2f}")
    print("="*50)
    
    eq_df.to_csv(os.path.join(out_dir, 'nav_equity.csv'), index=False)
    
    plt.figure(figsize=(10,6))
    plt.plot(eq_df['date'], eq_df['nav'], label='Optimized News WFO (No-688, <500亿)')
    plt.title('Daily T+1 News Strategy: Optimized Mid-Cap WFO')
    plt.xlabel('Date')
    plt.ylabel('Capital')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(out_dir, 'equity_curve.png'))
    print(f"Successfully generated equity_curve.png in {out_dir}")

if __name__ == "__main__":
    run_wfo_backtest(start_date='20240101', end_date='20261231')

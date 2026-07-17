import os
import pandas as pd
import numpy as np
import joblib
from tqdm import tqdm
import sys
import matplotlib.pyplot as plt
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from infra_data.storage import DataStorage

DATA_DIR = r'D:\iquant_data\data_v2'
PRICE_DIR = os.path.join(DATA_DIR, 'data_day1')
RANK_DIR  = os.path.join(DATA_DIR, 'ths_rank1')
CHIP_DIR  = os.path.join(DATA_DIR, 'cyq1')

def run_backtest(start_date='20240101', end_date='20261231'):
    model, feats = joblib.load('daily_dragon_news_model.joblib')
    dates = sorted([f.replace('.parquet', '') for f in os.listdir(PRICE_DIR) if f.endswith('.parquet')])
    test_dates = [d for d in dates if start_date <= d <= end_date]
    
    storage = DataStorage()
    valid_dates_series = pd.Series([pd.to_datetime(d) for d in test_dates]).sort_values()
    news_market_df, news_stock_sector_df = storage.load_news_data(start_date, end_date, valid_dates_series)
    
    if not news_market_df.empty:
        news_market_df['trade_date'] = news_market_df['trade_date'].dt.strftime('%Y%m%d')
    if not news_stock_sector_df.empty:
        news_stock_sector_df['trade_date'] = news_stock_sector_df['trade_date'].dt.strftime('%Y%m%d')

    initial_cap = 100000.0
    capital = initial_cap
    equity = []
    
    for i in tqdm(range(len(test_dates)-1), desc="Running Daily Strict Backtest (News)"):
        d_curr, d_next = test_dates[i], test_dates[i+1]
        
        p_rank = os.path.join(RANK_DIR, f"{d_curr}.parquet")
        p_chip = os.path.join(CHIP_DIR, f"{d_curr}.parquet")
        p_price = os.path.join(PRICE_DIR, f"{d_curr}.parquet")
        if not all(os.path.exists(p) for p in [p_rank, p_chip, p_price]): continue
        
        rank_df = pd.read_parquet(p_rank)
        rank_df['hot_rank_pct'] = rank_df['hot'].rank(pct=True)
        
        chip_df = pd.read_parquet(p_chip)
        chip_df['chip_concentration'] = (chip_df['cost_85pct'] - chip_df['cost_15pct']) / (chip_df['cost_50pct'] + 1e-8)
        
        price_df = pd.read_parquet(p_price, columns=['ts_code', 'close', 'pct_chg', 'amount', 'vol', 'pre_close'])
        
        df = pd.merge(rank_df[['ts_code', 'hot_rank_pct']], price_df, on='ts_code')
        df = pd.merge(df, chip_df[['ts_code', 'chip_concentration', 'winner_rate']], on='ts_code')
        df['trade_date'] = d_next
        
        # Merge news
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
        
        X = df[feats].fillna(0)
        try:
            df['prob'] = model.predict_proba(X)[:, 1]
        except Exception:
            df['prob'] = 0
            
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
            
            up_limit = round(pre_close_p * 1.2, 2) if '300' in ts_code or '688' in ts_code else round(pre_close_p * 1.1, 2)
            if pd.isna(open_p) or open_p >= up_limit:
                continue
                
            buy_price = open_p
            # T+1 logic: close at 4% profit or EOD
            if high_p >= buy_price * 1.04:
                sell_price = buy_price * 1.04
            else:
                sell_price = close_p
                
            ret = (sell_price / buy_price) - 1
            ret -= 0.0015 # fee
            
            day_pnl += alloc * ret
            
        capital += day_pnl
        equity.append({'date': pd.to_datetime(d_next), 'nav': capital})
        
    eq_df = pd.DataFrame(equity)
    total_ret = capital / initial_cap - 1
    years = len(eq_df) / 252.0 if len(eq_df) > 0 else 1
    ann_ret = (1+total_ret)**(1/years)-1 if years > 0 else 0
    df_ret = eq_df['nav'].pct_change()
    mdd = ((eq_df['nav'] - eq_df['nav'].cummax()) / eq_df['nav'].cummax()).min()
    vol = df_ret.std() * np.sqrt(252)
    sharpe = ann_ret / vol if vol > 0 else 0
    
    print("="*40)
    print("Daily Strict (With News Features) results:")
    print(f"Final Capital: {capital:,.2f}")
    print(f"Total Return: {total_ret:.2%}")
    print(f"Annual Return: {ann_ret:.2%}")
    print(f"Max Drawdown: {mdd:.2%}")
    print(f"Sharpe Ratio: {sharpe:.2f}")
    print("="*40)
    
    eq_df.to_csv('dragon_daily_news_equity.csv', index=False)
    
    plt.figure(figsize=(10,6))
    plt.plot(eq_df['date'], eq_df['nav'], label='Dragon Daily (News)')
    plt.title('Dragon Daily Strict with News Features')
    plt.xlabel('Date')
    plt.ylabel('Capital')
    plt.legend()
    plt.grid(True)
    plt.savefig('dragon_daily_news_equity.png')
    print("Saved plot to dragon_daily_news_equity.png")

if __name__ == "__main__":
    run_backtest('20240101', '20261231')

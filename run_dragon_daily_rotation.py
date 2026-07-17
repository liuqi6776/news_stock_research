"""
龙系列日频爆发策略 (A-Share T+1 Rotation)
逻辑：
1. T日 14:50 (接近收盘) 根据模型信号买入。
2. T+1日 (次日) 盘中冲高止盈 或 收盘卖出。
3. 严格涨跌停：T日收盘涨停买不到，T+1日跌停卖不出。
"""
import os
import pandas as pd
import numpy as np
import joblib
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

DATA_DIR = r'D:\iquant_data\data_v2'
PRICE_DIR = os.path.join(DATA_DIR, 'data_day1')
RANK_DIR  = os.path.join(DATA_DIR, 'ths_rank1')
CHIP_DIR  = os.path.join(DATA_DIR, 'cyq1')

def get_limit_price(code, pre_close, direction='up'):
    ratio = 0.2 if code.startswith(('30', '68')) else 0.1
    if direction == 'up':
        return round(pre_close * (1 + ratio), 2)
    return round(pre_close * (1 - ratio), 2)

def run_rotation_backtest():
    # 1. 加载模型
    model, feats = joblib.load('daily_dragon_model.joblib')
    
    dates = sorted([f.replace('.parquet', '') for f in os.listdir(PRICE_DIR) if f.endswith('.parquet')])
    test_dates = [d for d in dates if '20240101' <= d <= '20241231']
    
    capital = 100000.0
    initial_cap = capital
    equity = []
    
    # 我们模拟 T 日闭市前买，T+1 日离场
    for i in tqdm(range(len(test_dates) - 1), desc="Rotation Backtest"):
        d_curr = test_dates[i]       # 选股并买入的日子 (T)
        d_next = test_dates[i+1]     # 卖出的日子 (T+1)
        
        # --- A. 选股 (T日 14:50) ---
        p_rank = os.path.join(RANK_DIR, f"{d_curr}.parquet")
        p_chip = os.path.join(CHIP_DIR, f"{d_curr}.parquet")
        p_price = os.path.join(PRICE_DIR, f"{d_curr}.parquet")
        if not all(os.path.exists(p) for p in [p_rank, p_chip, p_price]):
            equity.append({'date': d_curr, 'nav': capital}); continue
            
        rank_df = pd.read_parquet(p_rank)
        rank_df['hot_rank_pct'] = rank_df['hot'].rank(pct=True)
        chip_df = pd.read_parquet(p_chip)
        chip_df['chip_concentration'] = (chip_df['cost_85pct'] - chip_df['cost_15pct']) / chip_df['cost_50pct']
        price_df = pd.read_parquet(p_price, columns=['ts_code', 'close', 'pre_close', 'high', 'low', 'amount', 'pct_chg', 'vol'])
        
        # 严格买入检查：T日收盘没涨停
        def is_limit_up(row):
            up_price = get_limit_price(row['ts_code'], row['pre_close'], 'up')
            return row['close'] >= up_price
            
        price_df['is_limit_up'] = price_df.apply(is_limit_up, axis=1)
        test_df = pd.merge(rank_df[['ts_code', 'hot_rank_pct']], price_df, on='ts_code')
        test_df = pd.merge(test_df, chip_df[['ts_code', 'chip_concentration', 'winner_rate']], on='ts_code')
        
        # 仅选没涨停的热门股 (Top 100)
        hot_stocks = rank_df.sort_values('hot', ascending=False).head(100)['ts_code'].tolist()
        test_df = test_df[test_df['ts_code'].isin(hot_stocks) & (test_df['is_limit_up'] == False)]
        
        if test_df.empty:
            equity.append({'date': d_curr, 'nav': capital}); continue
            
        X = test_df[feats].fillna(0)
        test_df['prob'] = model.predict_proba(X)[:, 1]
        picks = test_df.sort_values('prob', ascending=False).head(3)
        
        # --- B. 计算 T+1 日卖出收益 ---
        p_next_file = os.path.join(PRICE_DIR, f"{d_next}.parquet")
        next_prices = pd.read_parquet(p_next_file, columns=['ts_code', 'open', 'high', 'low', 'close', 'pre_close'])
        
        trade_results = pd.merge(picks[['ts_code', 'close', 'prob']], next_prices, on='ts_code', suffixes=('_buy', '_sell'))
        
        if trade_results.empty:
            equity.append({'date': d_curr, 'nav': capital}); continue
            
        day_pnl = 0
        cash_per_stock = capital / 3
        
        for _, row in trade_results.iterrows():
            # T 日买入价 = T 日收盘价
            buy_px = row['close_buy'] * 1.0005 # 费率
            
            # T+1 日检查卖出
            down_limit = get_limit_price(row['ts_code'], row['pre_close'], 'down')
            if row['open'] <= down_limit and row['high'] <= down_limit:
                # 跌停全天卖不出，假设按收盘跌停价卖出 (实际可能更惨，但此处模拟平仓)
                sell_px = down_limit * 0.9985
            else:
                # 止盈逻辑：如果在 T+1 盘中冲高 > 3.5%
                if (row['high'] / row['open'] - 1) > 0.035:
                    sell_px = row['open'] * 1.03 * 0.9985
                else:
                    sell_px = row['close_sell'] * 0.9985
            
            pnl = (sell_px - buy_px) / buy_px * cash_per_stock
            day_pnl += pnl
            
        capital += day_pnl
        equity.append({'date': d_next, 'nav': capital})
        
    eq_df = pd.DataFrame(equity)
    total_ret = (capital - initial_cap) / initial_cap
    print(f"\n--- T+1 Rotation (Buy T Close, Sell T+1 Peak) Results ---")
    print(f"最终净值: {capital:,.2f}")
    print(f"累计收益率: {total_ret*100:.2f}%")
    
    plt.figure(figsize=(10, 6))
    plt.plot(pd.to_datetime(eq_df['date']), eq_df['nav'])
    plt.title('Daily Rotation Strategy (Realistic T+1)')
    plt.savefig('dragon_daily_rotation.png')

if __name__ == "__main__":
    run_rotation_backtest()

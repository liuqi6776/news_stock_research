"""
龙系列日频爆发策略 (Strict A-Share T+1)
1. 严格 T+1 制度：T日选股，T+1日开盘买，T+2日离场。
2. 严格涨跌停：开盘涨停无法买入，全天跌停（或开盘跌停）无法卖出。
3. 盘中监控：模拟 Ptrade T+2 盘中冲高止盈。
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

def run_strict_t1_backtest():
    # 1. 加载模型
    model, feats = joblib.load('daily_dragon_model.joblib')
    
    dates = sorted([f.replace('.parquet', '') for f in os.listdir(PRICE_DIR) if f.endswith('.parquet')])
    test_dates = [d for d in dates if '20240101' <= d <= '20241231']
    
    capital = 100000.0
    initial_cap = capital
    equity = []
    
    # 仓位管理
    holdings = [] # {'ts_code': str, 'shares': int, 'buy_px': float}
    
    for i in tqdm(range(len(test_dates) - 2), desc="Strict T+1 Backtest"):
        d_curr = test_dates[i]       # 模型计算分数的日子 (T)
        d_next = test_dates[i+1]     # 买入的日子 (T+1)
        d_sell = test_dates[i+2]     # 离场的日子 (T+2)
        
        # --- A. 卖出 T-1 日买入的股票 (如果有) ---
        if holdings:
            p_sell_file = os.path.join(PRICE_DIR, f"{d_sell}.parquet")
            sell_data = pd.read_parquet(p_sell_file, columns=['ts_code', 'open', 'high', 'low', 'close', 'pre_close'])
            
            for pos in list(holdings):
                # 找到对应股票表现
                stock_perf = sell_data[sell_data['ts_code'] == pos['ts_code']]
                if stock_perf.empty:
                    # 停牌或缺失，假设按原价持有（实际会有损耗）
                    continue
                
                row = stock_perf.iloc[0]
                down_limit = get_limit_price(row['ts_code'], row['pre_close'], 'down')
                
                # 如果开盘即跌停且全天没打开，无法卖出 (简化逻辑：全天 Low == DownLimit)
                if row['open'] <= down_limit and row['high'] <= down_limit:
                    # 无法成交，继续持有至下一周期
                    continue
                
                # 止盈逻辑：如果在 T+2 盘中冲高 > 2.5% (T+2 冲高概率通常低于 T+1)
                # 我们假设 Ptrade 监控止盈
                if (row['high'] / row['open'] - 1) > 0.03:
                    sell_px = row['open'] * 1.025 * 0.9985
                else:
                    sell_px = row['close'] * 0.9985
                
                revenue = pos['shares'] * sell_px
                capital += revenue
                holdings.remove(pos)
        
        # --- B. 计算今日分数并准备明日买入 ---
        p_rank = os.path.join(RANK_DIR, f"{d_curr}.parquet")
        p_chip = os.path.join(CHIP_DIR, f"{d_curr}.parquet")
        p_price = os.path.join(PRICE_DIR, f"{d_curr}.parquet")
        if not all(os.path.exists(p) for p in [p_rank, p_chip, p_price]):
            equity.append({'date': d_curr, 'nav': capital}); continue
            
        rank_df = pd.read_parquet(p_rank)
        rank_df['hot_rank_pct'] = rank_df['hot'].rank(pct=True)
        chip_df = pd.read_parquet(p_chip)
        chip_df['chip_concentration'] = (chip_df['cost_85pct'] - chip_df['cost_15pct']) / chip_df['cost_50pct']
        price_df = pd.read_parquet(p_price, columns=['ts_code', 'close', 'pct_chg', 'amount', 'vol'])
        
        hot_stocks = rank_df.sort_values('hot', ascending=False).head(100)['ts_code'].tolist()
        test_df = pd.merge(rank_df[['ts_code', 'hot_rank_pct']], price_df, on='ts_code')
        test_df = pd.merge(test_df, chip_df[['ts_code', 'chip_concentration', 'winner_rate']], on='ts_code')
        test_df = test_df[test_df['ts_code'].isin(hot_stocks)]
        
        if test_df.empty:
            equity.append({'date': d_curr, 'nav': capital}); continue
            
        X = test_df[feats].fillna(0)
        test_df['prob'] = model.predict_proba(X)[:, 1]
        picks = test_df.sort_values('prob', ascending=False).head(3)
        
        # --- C. 执行买入 (T+1 开盘) ---
        p_next_file = os.path.join(PRICE_DIR, f"{d_next}.parquet")
        next_prices = pd.read_parquet(p_next_file, columns=['ts_code', 'open', 'pre_close'])
        
        buy_candidates = pd.merge(picks[['ts_code']], next_prices, on='ts_code')
        if not buy_candidates.empty:
            cash_per_stock = capital / 3
            for _, row in buy_candidates.iterrows():
                up_limit = get_limit_price(row['ts_code'], row['pre_close'], 'up')
                if row['open'] >= up_limit:
                    continue # 涨停买不进
                
                buy_px = row['open'] * 1.001
                shares = int(cash_per_stock / buy_px / 100) * 100
                if shares >= 100:
                    cost = shares * buy_px
                    capital -= cost
                    holdings.append({'ts_code': row['ts_code'], 'shares': shares, 'buy_px': buy_px})
                    
        equity.append({'date': d_curr, 'nav': capital + sum(p['shares'] * p['buy_px'] for p in holdings)})
        
    eq_df = pd.DataFrame(equity)
    total_ret = (equity[-1]['nav'] - initial_cap) / initial_cap
    print(f"\n--- Strict A-Share T+1 Results ---")
    print(f"最终净值: {equity[-1]['nav']:,.2f}")
    print(f"累计收益率: {total_ret*100:.2f}%")
    
    plt.figure(figsize=(10, 6))
    plt.plot(pd.to_datetime(eq_df['date']), eq_df['nav'])
    plt.title('Strict T+1 Dragon Strategy (Buy T+1, Sell T+2)')
    plt.savefig('dragon_daily_strict_t1.png')

if __name__ == "__main__":
    run_strict_t1_backtest()

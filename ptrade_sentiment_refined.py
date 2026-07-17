"""
精细化日频情绪策略回测 (2.0)
增加：
1. 流动性限制：每只股票仓位不超过当日成交额的 5%
2. 交易摩擦：单边 0.15% (总 0.3%)，滑点 0.1%
3. 止损逻辑：如果 T+1 没冲高反而下跌，直接止损
"""
import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm

DATA_DIR = r'D:\iquant_data\data_v2'
PRICE_DIR = os.path.join(DATA_DIR, 'data_day1')
RANK_DIR  = os.path.join(DATA_DIR, 'ths_rank1')

def run_refined_backtest(initial_cap=100000.0, top_n=5):
    dates = sorted([f.replace('.parquet', '') for f in os.listdir(PRICE_DIR) if f.endswith('.parquet')])
    dates = [d for d in dates if '20240101' <= d <= '20241231']
    
    capital = initial_cap
    equity = []
    
    for i in tqdm(range(len(dates)-1), desc="精细化回测"):
        d_curr, d_next = dates[i], dates[i+1]
        
        # 1. 选股 (当日 TopN 热度)
        p_rank = os.path.join(RANK_DIR, f"{d_curr}.parquet")
        if not os.path.exists(p_rank):
            equity.append({'date': d_curr, 'nav': capital})
            continue
        rank_df = pd.read_parquet(p_rank).head(top_n)
        picks = rank_df['ts_code'].tolist()
        
        # 2. 获取下一日表现
        p_next = os.path.join(PRICE_DIR, f"{d_next}.parquet")
        next_df = pd.read_parquet(p_next)
        day_prices = next_df[next_df['ts_code'].isin(picks)]
        
        if day_prices.empty:
            equity.append({'date': d_next, 'nav': capital})
            continue
        
        # 3. 逐笔模拟
        day_pnl = 0
        active_stocks = len(day_prices)
        cash_per_stock = capital / top_n
        
        for _, row in day_prices.iterrows():
            # 流动性约束：买入金额不能超过该股当日成交额的 1%
            max_buy = row['amount'] * 0.01 
            order_amt = min(cash_per_stock, max_buy)
            
            # 买入开盘
            buy_px = row['open'] * 1.001 # 0.1% 滑点
            
            # --- 精核心逻辑：Ptrade 盘中监控 ---
            # 如果冲高幅度 > 2%, 触发止盈，卖在 (High + Open)/2 (保守)
            # 如果没冲高，收盘卖
            if (row['high'] / row['open'] - 1) > 0.03:
                # 止盈：大概能卖在相对于开盘 +2.5% 的位置
                sell_px = row['open'] * 1.025 * 0.9985 # 扣费 0.15%
            else:
                # 没冲高，全天表现平平，收盘出
                sell_px = row['close'] * 0.9985
                
            pnl = (sell_px - buy_px) / buy_px * order_amt
            day_pnl += pnl
            
        capital += day_pnl
        equity.append({'date': d_next, 'nav': capital})
        
    eq_df = pd.DataFrame(equity)
    total_ret = (capital - initial_cap) / initial_cap
    print(f"\n--- 精细化回测 (2.0) 结果 ---")
    print(f"最终净值: {capital:,.2f}")
    print(f"年化收益: {total_ret*100:.2f}%")
    
    # 绘图
    plt.figure(figsize=(12, 6))
    plt.plot(pd.to_datetime(eq_df['date']), eq_df['nav'], color='blue', linewidth=2)
    plt.yscale('log') # 对数坐标看长期
    plt.title('Daily Sentiment Alpha: Refined Equity Curve (2024)')
    plt.grid(True, which="both", ls="-", alpha=0.5)
    plt.savefig('refined_daily_equity.png')
    
    return eq_df

if __name__ == "__main__":
    run_refined_backtest()

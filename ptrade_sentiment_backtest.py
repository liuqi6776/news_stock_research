"""
Ptrade 实时交易逻辑模板 (模拟版)
目标：每天开盘集合竞价后，根据热度排名选股，盘中监控卖出。

实战逻辑：
1. 每日 9:25: 获取同花顺热度 Top 50 股票。
2. 每日 9:30: 市价/限价买入热度上升最快、且筹码集中的前 3-5 只。
3. 盘中监控: 
   - 达到 +3% 到 +5% 的止盈区间，分批卖出。
   - 如果价格跌破开盘价 -2%，止损卖出。
   - 如果 14:50 还没卖出，市价卖出（回血准备明天交易）。
"""

def ptrade_on_timer(context, data):
    # --- 1. 获取热度数据 (Ptrade API 示例) ---
    # 实际 Ptrade 中可能需要通过 get_ths_hot_rank() 获取
    # 这里用我们系统计算好的结果模拟
    candidates = get_hot_sentiment_leaders(top_n=20)
    
    # --- 2. 筛选筹码与波动 ---
    final_picks = []
    for code in candidates:
        # 获取 5 分钟 K 线，看开盘强度
        # bars = get_bars(code, count=1, unit='5m', fields=['close', 'open'])
        # if bars['close'] > bars['open']: 
        final_picks.append(code)
    
    # --- 3. 执行买入 ---
    # order_target_value(code, context.portfolio.cash / 5)
    
    pass

def ptrade_handle_tick(context, tick):
    # --- 4. 盘中高频止盈 (HFT Logic) ---
    # if tick.last >= pos.entry_price * 1.04:
    #    close_position(tick.code)
    pass

# ============================================================
# 以下为系统回测代码，模拟上述逻辑
# ============================================================
import os
import pandas as pd
import numpy as np
from tqdm import tqdm

DATA_DIR = r'D:\iquant_data\data_v2'
PRICE_DIR = os.path.join(DATA_DIR, 'data_day1')
RANK_DIR  = os.path.join(DATA_DIR, 'ths_rank1')

def run_sentiment_daily_backtest(initial_cap=100000.0, top_n=3):
    dates = sorted([f.replace('.parquet', '') for f in os.listdir(PRICE_DIR) if f.endswith('.parquet')])
    dates = [d for d in dates if '20240101' <= d <= '20241231']
    
    capital = initial_cap
    equity = []
    
    for i in tqdm(range(len(dates)-1), desc="模拟日频交易"):
        d_curr, d_next = dates[i], dates[i+1]
        
        # 加载特征
        p_rank = os.path.join(RANK_DIR, f"{d_curr}.parquet")
        if not os.path.exists(p_rank):
            equity.append({'date': d_curr, 'nav': capital})
            continue
            
        rank_df = pd.read_parquet(p_rank).sort_values('hot', ascending=False)
        picks = rank_df.head(top_n)['ts_code'].tolist()
        
        # 加载价格
        p_next = os.path.join(PRICE_DIR, f"{d_next}.parquet")
        next_df = pd.read_parquet(p_next)
        next_df = next_df[next_df['ts_code'].isin(picks)]
        
        if next_df.empty:
            equity.append({'date': d_curr, 'nav': capital})
            continue
            
        # 模拟交易逻辑：
        # 买入价 = Open * (1 + 0.001滑点)
        # 卖出价 = 动态。我们假设能够抓到 (High + Open)/2 的中值（保守估计）
        # 或者模拟 50% 概率抓到 High，50% 概率抓到 Close
        
        day_pnl = 0
        cash_per_stock = capital / top_n
        
        for _, row in next_df.iterrows():
            buy_px = row['open'] * 1.001
            # 保守估计：卖出价为 (High + Close) / 2
            sell_px = (row['high'] + row['close']) / 2 * 0.999 # 扣除费率
            
            pnl = (sell_px - buy_px) / buy_px * cash_per_stock
            day_pnl += pnl
            
        capital += day_pnl
        equity.append({'date': d_next, 'nav': capital})
        
    eq_df = pd.DataFrame(equity)
    total_ret = (capital - initial_cap) / initial_cap
    print(f"\n日频情绪策略回测结果:")
    print(f"最终净值: {capital:,.2f}")
    print(f"累计收益率: {total_ret*100:.2f}%")
    
    return eq_df

if __name__ == "__main__":
    run_sentiment_daily_backtest()

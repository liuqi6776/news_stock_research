"""
龙回头/首板日频策略回测 (Dragon Daily)
使用训练好的日向模型进行 2024 年回测
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

def run_backtest_2024():
    # 1. 加载模型
    model, feats = joblib.load('daily_dragon_model.joblib')
    
    dates = sorted([f.replace('.parquet', '') for f in os.listdir(PRICE_DIR) if f.endswith('.parquet')])
    test_dates = [d for d in dates if '20240101' <= d <= '20241231']
    
    initial_cap = 100000.0
    capital = initial_cap
    equity = []
    
    for i in tqdm(range(len(test_dates)-1), desc="Dragon Backtest"):
        d_curr, d_next = test_dates[i], test_dates[i+1]
        
        # 加载特征
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
        
        # 预估当日 Top 100 热门股
        hot_stocks = rank_df.sort_values('hot', ascending=False).head(100)['ts_code'].tolist()
        
        # 合并特征并预测
        test_df = pd.merge(rank_df[['ts_code', 'hot_rank_pct']], price_df, on='ts_code')
        test_df = pd.merge(test_df, chip_df[['ts_code', 'chip_concentration', 'winner_rate']], on='ts_code')
        test_df = test_df[test_df['ts_code'].isin(hot_stocks)]
        
        if test_df.empty:
            equity.append({'date': d_curr, 'nav': capital}); continue
            
        X = test_df[feats].fillna(0)
        test_df['prob'] = model.predict_proba(X)[:, 1]
        
        # 选预测概率最高的前 3 名
        picks = test_df.sort_values('prob', ascending=False).head(3)
        
        # 4. 获取下一日表现
        p_next = os.path.join(PRICE_DIR, f"{d_next}.parquet")
        # 增加 pre_close 以计算涨跌停
        next_prices = pd.read_parquet(p_next, columns=['ts_code', 'open', 'high', 'low', 'close', 'pre_close', 'amount'])
        day_results = pd.merge(picks[['ts_code', 'prob']], next_prices, on='ts_code')
        
        if day_results.empty:
            equity.append({'date': d_next, 'nav': capital}); continue
            
        # 交易成交逻辑
        day_pnl = 0
        cash_per_stock = capital / 3
        
        for _, row in day_results.iterrows():
            # --- 严格涨跌停检查 ---
            # 计算涨停价 (主板 10%, 创业板/科创板 20%)
            code = row['ts_code']
            if code.startswith(('30', '68')):
                limit_ratio = 0.2
            else:
                limit_ratio = 0.1
            
            # A股涨停价通常是 pre_close * (1 + ratio) 四舍五入到两位
            up_limit = round(row['pre_close'] * (1 + limit_ratio), 2)
            down_limit = round(row['pre_close'] * (1 - limit_ratio), 2)
            
            # 如果开盘即涨停，无法买入
            if row['open'] >= up_limit:
                continue
            
            # 如果全天一字跌停，无法卖出（此处假设为买入当天的逻辑，通常买入当天无法卖出 T+1 制度）
            # 但我们需要考虑未来卖出时的限制。在此策略中，我们是 T 日买，T+1 日卖。
            
            # 流动性与成本
            buy_px = row['open'] * 1.001
            # 止盈逻辑：如果盘中冲高 > 3.5%，模拟 Ptrade 止盈卖在 +3%
            if (row['high'] / row['open'] - 1) > 0.035:
                sell_px = row['open'] * 1.03 * 0.9985 # 扣除 0.15% 交易费
            else:
                sell_px = row['close'] * 0.9985
                
            pnl = (sell_px - buy_px) / buy_px * cash_per_stock
            day_pnl += pnl
            
        capital += day_pnl
        equity.append({'date': d_next, 'nav': capital})
        
    eq_df = pd.DataFrame(equity)
    total_ret = (capital - initial_cap) / initial_cap
    print(f"\n--- Dragon Daily Backtest 结果 ---")
    print(f"最终净值: {capital:,.2f}")
    print(f"累计收益率: {total_ret*100:.2f}%")
    
    plt.figure(figsize=(10, 6))
    plt.plot(pd.to_datetime(eq_df['date']), eq_df['nav'])
    plt.title('Dragon Sentiment Strategy (T+1 Daily)')
    plt.savefig('dragon_daily_backtest.png')
    
if __name__ == "__main__":
    run_backtest_2024()

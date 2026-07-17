import os
import sys
import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# 添加项目根目录到 Path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from strategy_daily_t1_news.panqian_processor import process_panqian_news

# 配置路径
DATA_DIR  = r'D:\iquant_data\data_v2'
PRICE_DIR = os.path.join(DATA_DIR, 'data_day1')
OTHER_DIR = os.path.join(DATA_DIR, 'other_day1')
OUT_DIR   = r'c:\Users\liuqi\quant_system_v2\strategy_daily_t1_news'

# 核心回测参数
INITIAL_CAP = 1000000.0
SLIPPAGE = 0.001
COMMISSION = 0.0003
STAMP_DUTY = 0.0005  # A股印花税单边
MARKET_CAP_LIMIT = 5000000 # 500亿 (单位:万元)

def run_strict_t1_v2(news_dir, dataset_name, start_date='20240101', end_date='20260327'):
    print(f"\n[回测执行] 数据集: {dataset_name} | 模式: 严格 T+1")
    
    # 1. 加载交易日期列表
    dates = sorted([f.replace('.parquet', '') for f in os.listdir(PRICE_DIR) if f.endswith('.parquet')])
    trade_dates = [d for d in dates if start_date <= d <= end_date]
    date_to_idx = {d: i for i, d in enumerate(dates)}
    
    # 2. 解析新闻数据
    # 使用 panqian_processor 提取基础 impact
    news_mkt, news_stk = process_panqian_news(news_dir, '20220101', end_date)
    news_stk['article_date_str'] = news_stk['trade_date'].dt.strftime('%Y%m%d')
    
    # 处理日期对齐逻辑
    # news_major: article_date 是预测的前一天 -> 匹配到 article_date 的下一个交易日
    # news_major1: article_date 是预测的当天 -> 匹配到 article_date 当日
    
    if dataset_name == "news_major":
        def get_next_trade_date(d_str):
            if d_str not in dates: 
                # 如果 article_date 不是交易日，寻找它之后的第一个交易日
                future_dates = [dt for dt in dates if dt > d_str]
                return future_dates[0] if future_dates else None
            idx = date_to_idx[d_str]
            return dates[idx + 1] if (idx + 1) < len(dates) else None
        
        news_stk['trade_date_v2'] = news_stk['article_date_str'].apply(get_next_trade_date)
    else: # news_major1
        news_stk['trade_date_v2'] = news_stk['article_date_str']
    
    # 过滤掉无法匹配日期的记录
    news_stk = news_stk.dropna(subset=['trade_date_v2'])
    
    # 聚合同一天同一标的的信号
    news_agg = news_stk.groupby(['trade_date_v2', 'ts_code'])[['news_stock_impact', 'news_sector_impact']].mean().reset_index()
    news_lookup = news_agg.set_index(['trade_date_v2', 'ts_code']).to_dict('index')

    # 3. 模拟账户
    capital = INITIAL_CAP
    equity = []
    trades = []
    
    # 回测主循环
    for i, d_buy in enumerate(tqdm(trade_dates, desc=f"回测进展 {dataset_name}")):
        # 严格 T+1 需要卖出日 d_sell (d_buy 的下一个交易日)
        current_idx = date_to_idx[d_buy]
        if current_idx + 1 >= len(dates): break
        d_sell = dates[current_idx + 1]
        
        # 加载市值数据
        p_other = os.path.join(OTHER_DIR, f"{d_buy}.parquet")
        if not os.path.exists(p_other):
            equity.append({'date': pd.to_datetime(d_buy), 'nav': capital})
            continue
        other_df = pd.read_parquet(p_other, columns=['ts_code', 'circ_mv'])
        
        # 筛选符合今日买入条件的标的
        # 模式：只打分值最高的 Top 3
        daily_signals = []
        for (t_date, ts_code), impacts in news_lookup.items():
            if t_date == d_buy:
                # 排除科创板
                if ts_code.startswith('688'): continue
                # 市值过滤
                mv = other_df[other_df['ts_code']==ts_code]['circ_mv'].values
                if len(mv) > 0 and mv[0] <= MARKET_CAP_LIMIT:
                    score = impacts['news_stock_impact'] * 2.0 + impacts['news_sector_impact']
                    if impacts['news_stock_impact'] >= 3.0: # 严格阈值
                        daily_signals.append({'ts_code': ts_code, 'score': score})
        
        if not daily_signals:
            equity.append({'date': pd.to_datetime(d_buy), 'nav': capital})
            continue
            
        picks = sorted(daily_signals, key=lambda x: x['score'], reverse=True)[:3]
        
        # 执行买入 (Open) 与 次日卖出 (Open)
        b_df = pd.read_parquet(os.path.join(PRICE_DIR, f"{d_buy}.parquet"), columns=['ts_code', 'open', 'pre_close'])
        s_df = pd.read_parquet(os.path.join(PRICE_DIR, f"{d_sell}.parquet"), columns=['ts_code', 'open'])
        
        day_pnl = 0
        active_slots = len(picks)
        alloc = capital / 3.0 # 假设固定 3 个仓位
        
        for p in picks:
            code = p['ts_code']
            b_px = b_df[b_df['ts_code']==code]
            s_px = s_df[s_df['ts_code']==code]
            if b_px.empty or s_px.empty: continue
            
            b_val = b_px.iloc[0]['open']
            pre_close = b_px.iloc[0]['pre_close']
            s_val = s_px.iloc[0]['open']
            
            # 涨停板限制 (无法买入)
            limit_r = 1.195 if code.startswith('300') else 1.095
            if b_val >= round(pre_close * limit_r, 2): continue
            
            # T+1 真实收益计算
            ret = (s_val / b_val) - 1
            # 扣除双边所有费用
            fee = SLIPPAGE * 2 + COMMISSION * 2 + STAMP_DUTY
            net_ret = ret - fee
            day_pnl += alloc * net_ret
            
            trades.append({'buy_date': d_buy, 'sell_date': d_sell, 'code': code, 'ret': net_ret})
            
        capital += day_pnl
        equity.append({'date': pd.to_datetime(d_buy), 'nav': capital})
        
    return pd.DataFrame(equity), pd.DataFrame(trades)

if __name__ == "__main__":
    major_p  = r'D:\iquant_data\data_v2\news_major'
    major1_p = r'D:\iquant_data\data_v2\news_major1'
    
    eq_major, t_major = run_strict_t1_v2(major_p, "news_major")
    eq_major1, t_major1 = run_strict_t1_v2(major1_p, "news_major1")
    
    # 汇总绩效
    def get_stats(df, trades):
        ret = df['nav'].iloc[-1] / INITIAL_CAP - 1
        win_rate = (trades['ret'] > 0).mean() if not trades.empty else 0
        avg_ret = trades['ret'].mean() if not trades.empty else 0
        return f"总收益: {ret:.2%}, 胜率: {win_rate:.2%}, 平均涨幅: {avg_ret:.2%}"

    print("\n" + "="*50)
    print(f"News Major  指标: {get_stats(eq_major, t_major)}")
    print(f"News Major1 指标: {get_stats(eq_major1, t_major1)}")
    print("="*50)
    
    # 绘图
    plt.figure(figsize=(12, 6))
    plt.plot(eq_major['date'], eq_major['nav'], label='News Major (Strict T+1)', color='blue')
    plt.plot(eq_major1['date'], eq_major1['nav'], label='News Major1 (Strict T+1)', color='red')
    plt.axhline(INITIAL_CAP, color='black', linestyle='--')
    plt.title('Correct T+1 News Strategy Comparison: Major vs Major1')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(OUT_DIR, 'news_v2_comparison_strict.png'))
    print(f"\n对比图已保存至: {os.path.join(OUT_DIR, 'news_v2_comparison_strict.png')}")

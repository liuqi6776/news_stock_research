import os
import sys
import pandas as pd
import numpy as np
import joblib
from tqdm import tqdm
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from infra_data.storage import DataStorage

DATA_DIR = r'D:\iquant_data\data_v2'
PRICE_DIR = os.path.join(DATA_DIR, 'data_day1')
RANK_DIR  = os.path.join(DATA_DIR, 'ths_rank1')
CHIP_DIR  = os.path.join(DATA_DIR, 'cyq1')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, 'results_duobao')

FEATURE_COLS = ['hot_rank_pct', 'chip_concentration', 'winner_rate',
               'news_market_impact', 'news_stock_impact', 'news_sector_impact']

def get_all_dates():
    return sorted([f.replace('.parquet', '') for f in os.listdir(PRICE_DIR) if f.endswith('.parquet')])

def load_features_for_date(date_str):
    p_rank = os.path.join(RANK_DIR, f"{date_str}.parquet")
    p_chip = os.path.join(CHIP_DIR, f"{date_str}.parquet")
    p_price = os.path.join(PRICE_DIR, f"{date_str}.parquet")
    
    if not all(os.path.exists(p) for p in [p_rank, p_chip, p_price]):
        return None
    
    rank_df = pd.read_parquet(p_rank)
    rank_df['hot_rank_pct'] = rank_df['hot'].rank(pct=True)
    
    chip_df = pd.read_parquet(p_chip)
    chip_df['chip_concentration'] = (chip_df['cost_85pct'] - chip_df['cost_15pct']) / (chip_df['cost_50pct'] + 1e-8)
    
    price_df = pd.read_parquet(p_price, columns=['ts_code', 'close', 'pct_chg', 'amount', 'vol', 'pre_close'])
    
    df = pd.merge(rank_df[['ts_code', 'hot_rank_pct']], price_df, on='ts_code')
    df = pd.merge(df, chip_df[['ts_code', 'chip_concentration', 'winner_rate']], on='ts_code')
    
    return df

def load_news_data(start_date, end_date, dates_list):
    storage = DataStorage()
    valid_dates = pd.Series([pd.to_datetime(d) for d in dates_list if start_date <= d <= end_date]).sort_values()
    news_market_df, news_stock_sector_df = storage.load_news_data(start_date, end_date, valid_dates)
    
    if not news_market_df.empty:
        news_market_df['trade_date'] = news_market_df['trade_date'].dt.strftime('%Y%m%d')
    if not news_stock_sector_df.empty:
        news_stock_sector_df['trade_date'] = news_stock_sector_df['trade_date'].dt.strftime('%Y%m%d')
    
    return news_market_df, news_stock_sector_df

def run_backtest(take_profit_pct):
    dates = get_all_dates()
    
    TEST_START = '20220101'
    TEST_END = '20260324'
    
    test_dates = [d for d in dates if TEST_START <= d <= TEST_END]
    all_news_dates = [d for d in dates if '20200101' <= d <= TEST_END]
    
    print(f"测试日期范围: {test_dates[0]} ~ {test_dates[-1]}, 共 {len(test_dates)} 天")
    
    model_path = os.path.join(BASE_DIR, 'daily_dragon_news_model.joblib')
    model, feats = joblib.load(model_path)
    print(f"已加载模型")
    
    news_market_df, news_stock_sector_df = load_news_data('20200101', TEST_END, all_news_dates)
    print(f"已加载新闻数据")
    
    initial_cap = 100000.0
    capital = initial_cap
    
    print(f"\n开始回测止盈点: {take_profit_pct*100:.0f}%")
    
    for i in tqdm(range(len(test_dates)-2)):
        d_t = test_dates[i]
        d_t1 = test_dates[i+1]
        d_t2 = test_dates[i+2]
        
        df_t = load_features_for_date(d_t)
        if df_t is None:
            continue
        
        df_t['trade_date'] = d_t1
        
        if not news_market_df.empty:
            df_t = pd.merge(df_t, news_market_df, on='trade_date', how='left')
        else:
            df_t['news_market_impact'] = 0.0
            
        if not news_stock_sector_df.empty:
            df_t = pd.merge(df_t, news_stock_sector_df, on=['trade_date', 'ts_code'], how='left')
        else:
            df_t['news_stock_impact'] = 0.0
            df_t['news_sector_impact'] = 0.0
            
        df_t[['news_market_impact', 'news_stock_impact', 'news_sector_impact']] = \
            df_t[['news_market_impact', 'news_stock_impact', 'news_sector_impact']].fillna(0.0)
        
        X = df_t[feats].fillna(0)
        
        try:
            df_t['prob'] = model.predict_proba(X)[:, 1]
        except Exception:
            continue
        
        picks = df_t[df_t['prob'] > 0.8].sort_values('prob', ascending=False).head(3)
        if picks.empty:
            picks = df_t.sort_values('prob', ascending=False).head(1)
        
        p_t2 = os.path.join(PRICE_DIR, f"{d_t2}.parquet")
        if not os.path.exists(p_t2):
            continue
        df_t2 = pd.read_parquet(p_t2, columns=['ts_code', 'open', 'high', 'close', 'pre_close'])
        
        day_pnl = 0.0
        alloc = capital / len(picks)
        
        for _, pick in picks.iterrows():
            ts_code = pick['ts_code']
            t2_row = df_t2[df_t2['ts_code'] == ts_code]
            
            if t2_row.empty:
                continue
            
            t2 = t2_row.iloc[0]
            pre_close = t2['pre_close']
            
            up_limit = round(pre_close * 1.2, 2) if ('300' in ts_code or '688' in ts_code) else round(pre_close * 1.1, 2)
            
            if pd.isna(t2['open']) or t2['open'] >= up_limit:
                continue
            
            buy_price = t2['open']
            
            if t2['high'] >= buy_price * (1 + take_profit_pct):
                sell_price = buy_price * (1 + take_profit_pct)
            else:
                sell_price = t2['close']
            
            ret = (sell_price / buy_price) - 1
            ret -= 0.0015
            
            day_pnl += alloc * ret
        
        capital += day_pnl
    
    total_ret = capital / initial_cap - 1
    print(f"\n止盈 {take_profit_pct*100:.0f}%: 总收益 {total_ret:+.2%}, 最终资金 ¥{capital:,.2f}")
    return take_profit_pct, total_ret, capital

if __name__ == "__main__":
    take_profit_list = [0.03, 0.04, 0.05, 0.06, 0.07, 0.08]
    results = []
    
    print("="*80)
    print("快速止盈点测试")
    print("="*80)
    
    for tp in take_profit_list:
        tp_pct, ret, cap = run_backtest(tp)
        results.append((tp_pct, ret, cap))
    
    print("\n" + "="*80)
    print("测试结果汇总")
    print("="*80)
    for tp_pct, ret, cap in results:
        print(f"止盈 {tp_pct*100:.0f}%: 总收益 {ret:+.2%}, 最终资金 ¥{cap:,.2f}")
    
    best = max(results, key=lambda x: x[1])
    print("\n最优结果:")
    print(f"止盈 {best[0]*100:.0f}%: 总收益 {best[1]:+.2%}, 最终资金 ¥{best[2]:,.2f}")

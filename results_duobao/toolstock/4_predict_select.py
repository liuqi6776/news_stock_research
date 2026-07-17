import os
import pandas as pd
import datetime
import sys
import joblib
import json
import warnings
import pickle

DATA_DIR  = r'D:\iquant_data\data_v2'
PRICE_DIR = os.path.join(DATA_DIR, 'data_day1')
RANK_DIR  = os.path.join(DATA_DIR, 'ths_rank1')
CHIP_DIR  = os.path.join(DATA_DIR, 'cyq1')
OTHER_DIR = os.path.join(DATA_DIR, 'other_day1')
NEWS_MAJOR_DIR = r"C:\Users\liuqi\clowspace\toolstock\news_major1"
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models', 'daily_t1_model.joblib')
STOCK_CACHE_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'trade_stock_dates_cache.pkl')

def load_stock_dates_cache():
    """加载股票历史交易日期缓存"""
    if os.path.exists(STOCK_CACHE_PATH):
        try:
            with open(STOCK_CACHE_PATH, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            print(f"警告: 加载股票缓存失败: {e}")
    return {}

def is_new_stock(ts_code, date_t, stock_dates, min_days=10):
    """判断是否为新股（T日前历史交易数据少于min_days天）"""
    if ts_code not in stock_dates:
        return True
    dates = stock_dates[ts_code]
    count = sum(1 for d in dates if d < date_t)
    return count < min_days

def process_news(news_dir, target_date):
    """处理 news_major1 新闻数据用于预测"""
    market_records = []
    stock_records = []
    
    for filename in os.listdir(news_dir):
        if not filename.endswith('.json'): continue
        filepath = os.path.join(news_dir, filename)
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except: continue
            
        date_str = data.get("article_date", "")
        if not date_str: continue
            
        trade_date = pd.to_datetime(date_str)
        date_formatted = trade_date.strftime('%Y%m%d')
        
        if date_formatted > target_date:
            continue
            
        market_impact = data.get("market_impact", 0)
        market_records.append({'trade_date': trade_date, 'news_market_impact': float(market_impact)})
        
        for s in data.get("stocks", []):
            code = s.get("stock_code")
            if not code: continue
            ts_code = f"{code}.SH" if code.startswith('6') else f"{code}.SZ" if (code.startswith('0') or code.startswith('3')) else code
            stock_records.append({'trade_date': trade_date, 'ts_code': ts_code, 'news_stock_impact': float(s.get("impact", 0))})
    
    news_mkt = pd.DataFrame(market_records)
    news_stk = pd.DataFrame(stock_records)
    
    if not news_mkt.empty:
        news_mkt['trade_date'] = news_mkt['trade_date'].dt.strftime('%Y%m%d')
    if not news_stk.empty:
        news_stk['trade_date'] = news_stk['trade_date'].dt.strftime('%Y%m%d')
        
    return news_mkt, news_stk

def run_prediction_for_date(target_date):
    """
    运行预测并输出选股结果
    """
    print(f"\n{'='*60}")
    print(f"--- [Step 4] 预测和选股: {target_date} ---")
    print(f"{'='*60}")
    
    if not os.path.exists(MODEL_PATH):
        print(f"错误: 模型文件不存在: {MODEL_PATH}")
        print("请先运行 3_train_model.py")
        return
    
    print(f"加载预训练模型...")
    model, feats = joblib.load(MODEL_PATH)
    
    print(f"加载股票历史交易缓存...")
    stock_dates = load_stock_dates_cache()
    date_int = int(target_date)
    
    print(f"处理新闻数据...")
    m_news, s_news = process_news(NEWS_MAJOR_DIR, target_date)
    
    p_rank = os.path.join(RANK_DIR, f"{target_date}.parquet")
    p_chip = os.path.join(CHIP_DIR, f"{target_date}.parquet")
    p_price= os.path.join(PRICE_DIR, f"{target_date}.parquet")
    p_other= os.path.join(OTHER_DIR, f"{target_date}.parquet")
    
    if not all(os.path.exists(p) for p in [p_rank, p_chip, p_price, p_other]):
        print(f"错误: 缺少数据文件")
        print(f"请先运行 2_process_data.py {target_date}")
        return
    
    rank_df = pd.read_parquet(p_rank)
    rank_df['hot_rank_pct'] = rank_df['hot'].rank(pct=True)
    
    chip_df = pd.read_parquet(p_chip)
    chip_df['chip_concentration'] = (chip_df['cost_85pct'] - chip_df['cost_15pct']) / (chip_df['cost_50pct'] + 1e-8)
    
    price_df = pd.read_parquet(p_price, columns=['ts_code', 'close', 'pct_chg', 'pre_close'])
    other_df = pd.read_parquet(p_other, columns=['ts_code', 'circ_mv'])
    
    df = pd.merge(rank_df[['ts_code', 'hot_rank_pct']], price_df, on='ts_code')
    df = pd.merge(df, chip_df[['ts_code', 'chip_concentration', 'winner_rate']], on='ts_code')
    df = pd.merge(df, other_df, on='ts_code', how='left')
    
    df = df[~df['ts_code'].str.startswith('688')]
    df = df[df['circ_mv'] <= 500000]
    
    print(f"新股过滤中...")
    original_count = len(df)
    df = df[~df['ts_code'].apply(lambda x: is_new_stock(x, date_int, stock_dates, 10))]
    new_count = len(df)
    print(f"  新股过滤前: {original_count} 只，过滤后: {new_count} 只，剔除 {original_count - new_count} 只新股")
    
    if not m_news.empty:
        df['news_market_impact'] = m_news['news_market_impact'].mean()
    else:
        df['news_market_impact'] = 0.0
        
    if not s_news.empty:
        df = pd.merge(df, s_news[['ts_code', 'news_stock_impact']], on='ts_code', how='left')
    else:
        df['news_stock_impact'] = 0.0
    
    df[['news_market_impact', 'news_stock_impact']] = df[['news_market_impact', 'news_stock_impact']].fillna(0.0)
    
    X = df[feats].fillna(0)
    df['prob'] = model.predict_proba(X)[:, 1]
    
    top_picks = df[df['prob'] > 0.8].sort_values('prob', ascending=False).head(10)
    if top_picks.empty:
        top_picks = df.sort_values('prob', ascending=False).head(10)
        
    print(f"\n{'='*60}")
    print(f"--- 最终选股结果 (TOP 10) ---")
    print(f"{'='*60}")
    print(f"{'排名':<6} {'股票代码':<12} {'概率':<10} {'市值(亿)':<12} {'新闻影响':<10}")
    print(f"{'-'*60}")
    
    for i, (idx, row) in enumerate(top_picks.iterrows()):
        print(f"{i+1:<6} {row['ts_code']:<12} {row['prob']:.4f}      {row['circ_mv']/10000:<10.2f}    {row['news_stock_impact']:.2f}")
    
    print(f"{'='*60}")
    print(f"\n建议: 次日开盘时买入排名靠前的股票，严格 T+1 交易！")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        date = sys.argv[1]
    else:
        date = datetime.datetime.now().strftime("%Y%m%d")
    
    run_prediction_for_date(date)

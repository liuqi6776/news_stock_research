import os
import sys
import pandas as pd
import numpy as np
import joblib
import pickle
from datetime import datetime

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = r'D:\iquant_data\data_v2'
PRICE_DIR = os.path.join(DATA_DIR, 'data_day1')
RANK_DIR  = os.path.join(DATA_DIR, 'ths_rank1')
CHIP_DIR  = os.path.join(DATA_DIR, 'cyq1')
OTHER_DIR = os.path.join(DATA_DIR, 'other_day1')
NEWS_DIR  = os.path.join(DATA_DIR, 'news_major1')
MODEL_PATH = 'daily_dragon_news_model.joblib'
STOCK_CACHE_PATH = 'trade_stock_dates_cache.pkl'

def load_stock_dates_cache():
    if os.path.exists(STOCK_CACHE_PATH):
        try:
            with open(STOCK_CACHE_PATH, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            print(f"警告: 加载股票缓存失败: {e}")
    return {}

def is_new_stock(ts_code, date_t, stock_dates, min_days=10):
    if ts_code not in stock_dates:
        return True
    dates = stock_dates[ts_code]
    count = sum(1 for d in dates if d < date_t)
    return count < min_days

def get_latest_date():
    files = [f for f in os.listdir(PRICE_DIR) if f.endswith('.parquet')]
    dates = sorted([f.replace('.parquet', '') for f in files])
    return dates[-1]

def load_news_data(target_date):
    news_market_impact = 0.0
    news_stock_dict = {}
    
    if not os.path.exists(NEWS_DIR):
        return news_market_impact, news_stock_dict
    
    for filename in os.listdir(NEWS_DIR):
        if not filename.endswith('.json'):
            continue
        filepath = os.path.join(NEWS_DIR, filename)
        try:
            import json
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            date_str = data.get("article_date", "")
            if not date_str:
                continue
            trade_date = pd.to_datetime(date_str).strftime('%Y%m%d')
            if trade_date > target_date:
                continue
            
            market_impact = data.get("market_impact", 0)
            news_market_impact = float(market_impact)
            
            for s in data.get("stocks", []):
                code = s.get("stock_code")
                if not code:
                    continue
                ts_code = f"{code}.SH" if code.startswith('6') else f"{code}.SZ" if (code.startswith('0') or code.startswith('3')) else code
                news_stock_dict[ts_code] = float(s.get("impact", 0))
        except Exception as e:
            continue
    
    return news_market_impact, news_stock_dict

def load_options_features():
    pcr_csv = r"D:\iquant_data\data_v2\qiquan\historical_pcr.csv"
    if not os.path.exists(pcr_csv):
        return pd.DataFrame()
    try:
        df_pcr = pd.read_csv(pcr_csv)
        df_pcr['date'] = pd.to_datetime(df_pcr['date'])
        df_pcr['trade_date'] = df_pcr['date'].dt.strftime('%Y%m%d')
        df_pcr_clean = df_pcr[['trade_date', 'pcr_50', 'oi_pcr_50']].rename(columns={
            'pcr_50': 'opt_pcr_vol_50',
            'oi_pcr_50': 'opt_pcr_oi_50'
        })
    except Exception:
        return pd.DataFrame()

    import akshare as ak
    try:
        df_qvix = ak.index_option_50etf_qvix()
        df_qvix['date'] = pd.to_datetime(df_qvix['date'])
        df_qvix['trade_date'] = df_qvix['date'].dt.strftime('%Y%m%d')
        df_qvix['opt_qvix_close'] = df_qvix['close']
        df_qvix['opt_qvix_ma'] = df_qvix['close'].rolling(20).mean()
        df_qvix['opt_qvix_std'] = df_qvix['close'].rolling(20).std()
        df_qvix['opt_qvix_zscore'] = (df_qvix['close'] - df_qvix['opt_qvix_ma']) / df_qvix['opt_qvix_std']
        df_qvix_clean = df_qvix[['trade_date', 'opt_qvix_close', 'opt_qvix_zscore']].fillna(0)
    except Exception:
        df_qvix_clean = pd.DataFrame()

    if df_qvix_clean.empty:
        return df_pcr_clean
    merged = pd.merge(df_pcr_clean, df_qvix_clean, on='trade_date', how='outer').sort_values('trade_date').reset_index(drop=True)
    merged[['opt_pcr_vol_50', 'opt_pcr_oi_50', 'opt_qvix_close', 'opt_qvix_zscore']] = \
        merged[['opt_pcr_vol_50', 'opt_pcr_oi_50', 'opt_qvix_close', 'opt_qvix_zscore']].ffill().fillna(0)
    return merged

def predict_for_date(target_date):
    print("\n" + "="*70)
    print(f"  【Daily Dragon Strategy】选股推荐 ({target_date})")
    print("="*70)
    
    if not os.path.exists(MODEL_PATH):
        print(f"错误: 模型文件不存在: {MODEL_PATH}")
        return None, None
    
    print(f"加载模型: {MODEL_PATH}...")
    model, feats = joblib.load(MODEL_PATH)
    print(f"模型特征: {feats}")
    
    print(f"\n加载股票历史交易缓存...")
    stock_dates = load_stock_dates_cache()
    date_int = int(target_date)
    
    print(f"\n加载新闻数据...")
    news_market_impact, news_stock_dict = load_news_data(target_date)
    print(f"  大盘新闻影响: {news_market_impact}")
    print(f"  个股新闻数量: {len(news_stock_dict)}")
    
    p_rank = os.path.join(RANK_DIR, f"{target_date}.parquet")
    if not os.path.exists(p_rank):
        print(f"警告: 今日同花顺热度数据不存在，尝试使用前一日热度数据")
        dates = sorted([f.replace('.parquet', '') for f in os.listdir(RANK_DIR) if f.endswith('.parquet') and f < target_date])
        if dates:
            p_rank = os.path.join(RANK_DIR, f"{dates[-1]}.parquet")
            print(f"  使用历史热度数据: {dates[-1]}")
        else:
            print("错误: 没有可用的历史热度数据")
            return None, None

    p_chip = os.path.join(CHIP_DIR, f"{target_date}.parquet")
    p_price = os.path.join(PRICE_DIR, f"{target_date}.parquet")
    p_other = os.path.join(OTHER_DIR, f"{target_date}.parquet")
    
    if not all(os.path.exists(p) for p in [p_price, p_other]):
        print(f"错误: 缺少核心行情或基本面数据文件")
        return None, None
    
    if not os.path.exists(p_chip):
        print(f"警告: 今日筹码数据不存在，使用前一日数据")
        dates = sorted([f.replace('.parquet', '') for f in os.listdir(CHIP_DIR) if f.endswith('.parquet')])
        if dates:
            p_chip = os.path.join(CHIP_DIR, f"{dates[-1]}.parquet")
            print(f"  使用: {dates[-1]}")
        else:
            print(f"错误: 没有可用的筹码数据")
            return None, None
    
    print(f"\n加载行情数据...")
    rank_df = pd.read_parquet(p_rank)
    rank_df = rank_df.drop_duplicates(subset=['ts_code'], keep='first')
    rank_df['hot_rank_pct'] = rank_df['hot'].rank(pct=True)
    print(f"  热度数据: {len(rank_df)} 只股票")
    
    chip_df = pd.read_parquet(p_chip)
    chip_df['chip_concentration'] = (chip_df['cost_85pct'] - chip_df['cost_15pct']) / (chip_df['cost_50pct'] + 1e-8)
    print(f"  筹码数据: {len(chip_df)} 只股票")
    
    price_df = pd.read_parquet(p_price, columns=['ts_code', 'close', 'pct_chg', 'amount', 'vol', 'pre_close'])
    other_df = pd.read_parquet(p_other, columns=['ts_code', 'circ_mv'])
    
    df = pd.merge(rank_df[['ts_code', 'hot_rank_pct']], price_df, on='ts_code')
    df = pd.merge(df, chip_df[['ts_code', 'chip_concentration', 'winner_rate']], on='ts_code')
    df = pd.merge(df, other_df, on='ts_code', how='left')
    
    df['news_market_impact'] = news_market_impact
    df['news_stock_impact'] = df['ts_code'].map(news_stock_dict).fillna(0.0)
    df['news_sector_impact'] = 0.0

    print(f"\n加载并对齐期权指标特征...")
    options_df = load_options_features()
    if not options_df.empty:
        opt_row = options_df[options_df['trade_date'] == target_date]
        if not opt_row.empty:
            for col in ['opt_pcr_vol_50', 'opt_pcr_oi_50', 'opt_qvix_close', 'opt_qvix_zscore']:
                df[col] = float(opt_row[col].values[0])
        else:
            latest_row = options_df.iloc[-1]
            for col in ['opt_pcr_vol_50', 'opt_pcr_oi_50', 'opt_qvix_close', 'opt_qvix_zscore']:
                df[col] = float(latest_row[col])
    else:
        for col in ['opt_pcr_vol_50', 'opt_pcr_oi_50', 'opt_qvix_close', 'opt_qvix_zscore']:
            df[col] = 0.0
    
    print(f"\n初步过滤...")
    original_count = len(df)
    df = df[~df['ts_code'].str.startswith('688')]
    df = df[df['circ_mv'] <= 500000]
    after_filter_count = len(df)
    print(f"  过滤前: {original_count} 只")
    print(f"  过滤后(不含科创板+市值≤500亿): {after_filter_count} 只")
    
    print(f"\n新股过滤中...")
    before_new_stock = len(df)
    df = df[~df['ts_code'].apply(lambda x: is_new_stock(x, date_int, stock_dates, 10))]
    after_new_stock = len(df)
    print(f"  新股过滤前: {before_new_stock} 只，过滤后: {after_new_stock} 只，剔除 {before_new_stock - after_new_stock} 只新股")
    
    if after_new_stock == 0:
        print("警告: 所有股票都被新股过滤剔除了，暂时关闭新股过滤...")
        df = pd.merge(rank_df[['ts_code', 'hot_rank_pct']], price_df, on='ts_code')
        df = pd.merge(df, chip_df[['ts_code', 'chip_concentration', 'winner_rate']], on='ts_code')
        df = pd.merge(df, other_df, on='ts_code', how='left')
        df['news_market_impact'] = news_market_impact
        df['news_stock_impact'] = df['ts_code'].map(news_stock_dict).fillna(0.0)
        df['news_sector_impact'] = 0.0
        if not options_df.empty:
            opt_row = options_df[options_df['trade_date'] == target_date]
            if not opt_row.empty:
                for col in ['opt_pcr_vol_50', 'opt_pcr_oi_50', 'opt_qvix_close', 'opt_qvix_zscore']:
                    df[col] = float(opt_row[col].values[0])
            else:
                latest_row = options_df.iloc[-1]
                for col in ['opt_pcr_vol_50', 'opt_pcr_oi_50', 'opt_qvix_close', 'opt_qvix_zscore']:
                    df[col] = float(latest_row[col])
        else:
            for col in ['opt_pcr_vol_50', 'opt_pcr_oi_50', 'opt_qvix_close', 'opt_qvix_zscore']:
                df[col] = 0.0
        df = df[~df['ts_code'].str.startswith('688')]
        df = df[df['circ_mv'] <= 500000]
    
    print(f"\n预测中...")
    X = df[feats].fillna(0)
    try:
        df['prob'] = model.predict_proba(X)[:, 1]
    except Exception as e:
        print(f"预测失败: {e}")
        return None, None
    
    picks_top3 = df[df['prob'] > 0.8].sort_values('prob', ascending=False).head(3)
    picks_top10 = df.sort_values('prob', ascending=False).head(10)
    
    print("\n" + "="*70)
    print("  【首选推荐】概率 > 0.8")
    print("="*70)
    if not picks_top3.empty:
        print(f"{'排名':<4} {'股票代码':<12} {'概率':<10} {'收盘价':<8} {'涨跌':<8} {'市值(亿)':<10} {'新闻影响':<10}")
        print("-"*70)
        for i, (_, row) in enumerate(picks_top3.iterrows(), 1):
            print(f"{i:<4} {row['ts_code']:<12} {row['prob']:.4f}      {row['close']:<8.2f} {row['pct_chg']:>7.2f}%    {row['circ_mv']/10000:<10.2f}    {row['news_stock_impact']:.2f}")
    else:
        print("  没有概率 > 0.8 的股票，使用备选方案（Top 1）")
        top1 = picks_top10.head(1)
        if not top1.empty:
            row = top1.iloc[0]
            print(f"{'排名':<4} {'股票代码':<12} {'概率':<10} {'收盘价':<8} {'涨跌':<8} {'市值(亿)':<10} {'新闻影响':<10}")
            print("-"*70)
            print(f"{'1':<4} {row['ts_code']:<12} {row['prob']:.4f}      {row['close']:<8.2f} {row['pct_chg']:>7.2f}%    {row['circ_mv']/10000:<10.2f}    {row['news_stock_impact']:.2f}")
    
    print("\n" + "="*70)
    print("  【完整推荐】TOP 10")
    print("="*70)
    print(f"{'排名':<4} {'股票代码':<12} {'概率':<10} {'收盘价':<8} {'涨跌':<8} {'市值(亿)':<10} {'新闻影响':<10}")
    print("-"*70)
    for i, (_, row) in enumerate(picks_top10.iterrows(), 1):
        flag = "★" if row['prob'] > 0.8 else ""
        print(f"{i:<4} {row['ts_code']:<12} {row['prob']:.4f} {flag:<3} {row['close']:<8.2f} {row['pct_chg']:>7.2f}%    {row['circ_mv']/10000:<10.2f}    {row['news_stock_impact']:.2f}")
    
    print("\n" + "="*70)
    print("  【交易规则】")
    print("="*70)
    print("  买入: 次日（T+1日）9:30 以开盘价买入")
    print("        - 若开盘涨停（主板>9.5% / 创业板>19.5%）放弃")
    print("  卖出:")
    print("        - T+2日 盘中触及买入价 +8% 自动止盈")
    print("        - 若未触发止盈，T+2日收盘前（14:50~14:55）全仓卖出")
    print("  选股标准:")
    print("        - 不含科创板（688开头）")
    print("        - 市值 ≤ 500 亿")
    print("        - 排除新股（T日前历史交易数据少于10天）")
    print("="*70)
    
    return picks_top3, picks_top10

if __name__ == "__main__":
    if len(sys.argv) > 1:
        target_date = sys.argv[1]
    else:
        target_date = get_latest_date()
    
    predict_for_date(target_date)

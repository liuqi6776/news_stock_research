"""
Step 3: Train XGBoost model for doubao strategy
Label: (T+2 close / T+1 open - 1) > 0.04
Features: hot_rank_pct, chip_concentration, winner_rate, news_market_impact, news_stock_impact
Filter: circ_mv <= 1000000, exclude 688/689 (STAR market)
Model saved to: models/doubao_t1t2_model.joblib
"""
import os
import json
import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
from tqdm import tqdm
import sys

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = r'D:\iquant_data\data_v2'
PRICE_DIR = os.path.join(DATA_DIR, 'data_day1')
RANK_DIR = os.path.join(DATA_DIR, 'ths_rank1')
CHIP_DIR = os.path.join(DATA_DIR, 'cyq1')
OTHER_DIR = os.path.join(DATA_DIR, 'other_day1')
NEWS_MAJOR_DIR = os.path.join(THIS_DIR, "news_major1")
MODEL_PATH = os.path.join(THIS_DIR, 'models', 'doubao_t1t2_model.joblib')

FEATS = ['hot_rank_pct', 'chip_concentration', 'winner_rate',
         'news_market_impact', 'news_stock_impact']
CIRC_MV_LIMIT = 1000000


def process_news(news_dir):
    market_records = []
    stock_records = []
    if not os.path.exists(news_dir):
        return pd.DataFrame(market_records), pd.DataFrame(stock_records)
    for filename in os.listdir(news_dir):
        if not filename.endswith('.json'):
            continue
        filepath = os.path.join(news_dir, filename)
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except:
            continue
        date_str = data.get("article_date", "")
        if not date_str:
            continue
        trade_date = pd.to_datetime(date_str)
        market_impact = data.get("market_impact", 0)
        market_records.append({'trade_date': trade_date, 'news_market_impact': float(market_impact)})
        for s in data.get("stocks", []):
            code = s.get("stock_code")
            if not code:
                continue
            ts_code = f"{code}.SH" if code.startswith('6') else f"{code}.SZ" if (
                    code.startswith('0') or code.startswith('3')) else code
            stock_records.append(
                {'trade_date': trade_date, 'ts_code': ts_code, 'news_stock_impact': float(s.get("impact", 0))})
    return pd.DataFrame(market_records), pd.DataFrame(stock_records)


def train_model(start_date='20220101', end_date=None):
    print(f"--- [Step 3] 训练模型 ---")
    print(f"  标签: (T+2 close / T+1 open - 1) > 0.04")
    print(f"  特征: {FEATS}")
    print(f"  过滤: circ_mv <= {CIRC_MV_LIMIT}, 排除688/689")

    dates = sorted([f.replace('.parquet', '') for f in os.listdir(PRICE_DIR) if f.endswith('.parquet')])
    if end_date is None:
        end_date = dates[-1]

    train_dates = [d for d in dates if start_date <= d <= end_date]
    if len(train_dates) < 3:
        print(f"训练数据不足，需要至少 3 天")
        return None, None

    news_mkt, news_stk = process_news(NEWS_MAJOR_DIR)
    if not news_mkt.empty:
        news_mkt['trade_date'] = news_mkt['trade_date'].dt.strftime('%Y%m%d')
    if not news_stk.empty:
        news_stk['trade_date'] = news_stk['trade_date'].dt.strftime('%Y%m%d')

    all_data = []
    for i in tqdm(range(len(train_dates) - 2), desc="加载训练数据"):
        d_curr = train_dates[i]
        d_t1 = train_dates[i + 1]
        d_t2 = train_dates[i + 2]

        p_rank = os.path.join(RANK_DIR, f"{d_curr}.parquet")
        p_chip = os.path.join(CHIP_DIR, f"{d_curr}.parquet")
        p_price = os.path.join(PRICE_DIR, f"{d_curr}.parquet")
        p_other = os.path.join(OTHER_DIR, f"{d_curr}.parquet")
        p_t1 = os.path.join(PRICE_DIR, f"{d_t1}.parquet")
        p_t2 = os.path.join(PRICE_DIR, f"{d_t2}.parquet")

        if not all(os.path.exists(p) for p in [p_rank, p_chip, p_price, p_other, p_t1, p_t2]):
            continue

        rank_df = pd.read_parquet(p_rank)
        rank_df['hot_rank_pct'] = rank_df['hot'].rank(pct=True)

        chip_df = pd.read_parquet(p_chip)
        chip_df['chip_concentration'] = (chip_df['cost_85pct'] - chip_df['cost_15pct']) / (
                chip_df['cost_50pct'] + 1e-8)

        price_df = pd.read_parquet(p_price, columns=['ts_code', 'close', 'pct_chg', 'amount', 'vol'])
        other_df = pd.read_parquet(p_other, columns=['ts_code', 'circ_mv'])

        t1_df = pd.read_parquet(p_t1, columns=['ts_code', 'open'])
        t2_df = pd.read_parquet(p_t2, columns=['ts_code', 'close'])

        t1_df.rename(columns={'open': 't1_open'}, inplace=True)
        t2_df.rename(columns={'close': 't2_close'}, inplace=True)

        m = pd.merge(rank_df[['ts_code', 'hot_rank_pct']], price_df, on='ts_code')
        m = pd.merge(m, chip_df[['ts_code', 'chip_concentration', 'winner_rate']], on='ts_code')
        m = pd.merge(m, other_df, on='ts_code', how='left')
        m = m[~m['ts_code'].str.startswith('688')]
        m = m[~m['ts_code'].str.startswith('689')]
        m = m[m['circ_mv'] <= CIRC_MV_LIMIT]

        m = pd.merge(m, t1_df, on='ts_code')
        m = pd.merge(m, t2_df, on='ts_code')

        m['label_ret'] = m['t2_close'] / m['t1_open'] - 1
        m['label'] = (m['label_ret'] > 0.04).astype(int)
        m['trade_date'] = d_curr

        all_data.append(m)

    if not all_data:
        print("没有训练数据！")
        return None, None

    df = pd.concat(all_data, ignore_index=True)

    if not news_mkt.empty:
        df = pd.merge(df, news_mkt, on='trade_date', how='left')
    else:
        df['news_market_impact'] = 0.0
    if not news_stk.empty:
        df = pd.merge(df, news_stk[['trade_date', 'ts_code', 'news_stock_impact']],
                      on=['trade_date', 'ts_code'], how='left')
    else:
        df['news_stock_impact'] = 0.0
    df[['news_market_impact', 'news_stock_impact']] = df[['news_market_impact', 'news_stock_impact']].fillna(0.0)

    X = df[FEATS].fillna(0)
    y = df['label']

    print(f"\n训练样本数: {len(X)}")
    print(f"正样本数: {y.sum()} ({y.sum() / len(y):.2%})")
    print(f"使用特征: {FEATS}")

    model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric='auc',
        n_jobs=-1,
        tree_method='hist'
    )

    print("训练 XGBoost 模型...")
    model.fit(X, y)

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump((model, FEATS), MODEL_PATH)
    print(f"模型保存成功: {MODEL_PATH}")

    return model, FEATS


if __name__ == "__main__":
    train_model()

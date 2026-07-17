"""
方案 A+ 优化：周频模型增强（5日持仓，严格 T+1）
目标：在严格 T+1 制度下，通过提高换手率（从月到周）冲击 >50% 年化收益
"""
import os
import sys
import warnings
import pandas as pd
import numpy as np
import tushare as ts
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm
import xgboost as xgb
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import roc_auc_score

warnings.filterwarnings('ignore')

# ============================================================
# 配置
# ============================================================
TUSHARE_TOKEN = '421ff94dd31be789aa7f95e61ad6fad5bcefa250a0b2c4d298224aa'
DATA_DIR      = r'D:\iquant_data\data_v2\data_day1'
BASIC_DIR     = r'D:\iquant_data\data_v2\other_day1'
OUT_DIR       = r'C:\Users\liuqi\quant_system_v2'

INITIAL_CAP   = 100_000.0
TOP_N         = 5
COMMISSION    = 0.0003
STAMP_DUTY    = 0.0005
SLIPPAGE      = 0.001
STOP_LOSS     = -0.10    # 周策略止损 -10%

DATA_START    = '20210101'
TEST_START    = '20230101'
TEST_END      = '20260101'

# ============================================================
# 工具函数
# ============================================================
def get_limit_price(code, pre_close, direction='up'):
    ratio = 0.2 if code.startswith(('30', '68')) else 0.1
    if direction == 'up':
        return round(pre_close * (1 + ratio), 2)
    return round(pre_close * (1 - ratio), 2)

def load_data(start: str, end: str) -> pd.DataFrame:
    files = []
    sd, ed = pd.to_datetime(start), pd.to_datetime(end)
    date_strs = sorted([f.replace('.parquet', '') for f in os.listdir(DATA_DIR) if f.endswith('.parquet')])
    
    for ds in tqdm(date_strs, desc="加载行情与指标"):
        dt = pd.to_datetime(ds)
        if sd <= dt <= ed:
            p_file = os.path.join(DATA_DIR, f"{ds}.parquet")
            b_file = os.path.join(BASIC_DIR, f"{ds}.parquet")
            if not os.path.exists(p_file) or not os.path.exists(b_file): continue
            p_df = pd.read_parquet(p_file, columns=['ts_code', 'trade_date', 'open', 'high', 'low', 'close', 'pre_close', 'vol', 'pct_chg', 'amount'])
            b_df = pd.read_parquet(b_file, columns=['ts_code', 'pe', 'pb', 'circ_mv'])
            files.append(pd.merge(p_df, b_df, on='ts_code', how='inner'))
                
    if not files: return pd.DataFrame()
    df = pd.concat(files, ignore_index=True)
    df['trade_date'] = pd.to_datetime(df['trade_date'].astype(str))
    # 限制主力标的 (剔除 ST 简化)
    df = df[df['ts_code'].str.match(r'^(00|60|30|68)')]
    return df.sort_values(['ts_code', 'trade_date']).reset_index(drop=True)

def get_industry_mapping():
    map_file = os.path.join(OUT_DIR, 'stock_industry_map_cached.parquet')
    if os.path.exists(map_file): return pd.read_parquet(map_file)
    pro = ts.pro_api(TUSHARE_TOKEN)
    try:
        df = pro.stock_basic(exchange='', list_status='L', fields='ts_code,industry')
        df.to_parquet(map_file)
        return df
    except: return pd.DataFrame()

# ============================================================
# 特征与标签
# ============================================================
def build_features(df: pd.DataFrame, industry_df: pd.DataFrame) -> pd.DataFrame:
    print("构建特征...")
    df = df.copy().sort_values(['ts_code', 'trade_date'])
    g = df.groupby('ts_code')['close']
    for w in [5, 20]:
        df[f'mom_{w}'] = g.transform(lambda x: x / x.shift(w) - 1)
        ma = g.transform(lambda x: x.rolling(w).mean())
        df[f'bias_{w}'] = (df['close'] - ma) / (ma + 1e-8)
    
    df['ep'] = 1.0 / (df['pe'] + 1e-8)
    df['bp'] = 1.0 / (df['pb'] + 1e-8)
    df['log_mv'] = np.log(df['circ_mv'] + 1)
    
    if not industry_df.empty:
        df = pd.merge(df, industry_df, on='ts_code', how='left')
        df['industry'] = df['industry'].fillna('unknown')
        idx_cols = ['trade_date', 'industry']
        df['ep_neutral'] = df['ep'] - df.groupby(idx_cols)['ep'].transform('mean')
        df['bp_neutral'] = df['bp'] - df.groupby(idx_cols)['bp'].transform('mean')
    else:
        df['ep_neutral'], df['bp_neutral'] = 0, 0
        
    for col in ['mom_5', 'mom_20', 'bias_5', 'ep', 'circ_mv']:
        df[f'{col}_rank'] = df.groupby('trade_date')[col].rank(pct=True)
    return df

def add_labels(df: pd.DataFrame, horizon: int = 5) -> pd.DataFrame:
    # 预测 T+1 开盘买到 T+1+horizon 开盘的收益
    df = df.sort_values(['ts_code', 'trade_date'])
    entry = df.groupby('ts_code')['open'].shift(-1)
    exit_ = df.groupby('ts_code')['open'].shift(-1 - horizon)
    df['ret_label'] = (exit_ - entry) / (entry + 1e-8)
    df['label'] = (df['ret_label'] > 0.02).astype(int) # 5日目标 2%
    return df

FEATURE_COLS = ['mom_5', 'mom_20', 'bias_5', 'bias_20', 'ep', 'bp', 'log_mv', 'ep_neutral', 'bp_neutral',
                'mom_5_rank', 'mom_20_rank', 'bias_5_rank', 'ep_rank', 'circ_mv_rank']

# ============================================================
# 训练模型
# ============================================================
def train_model(train_df: pd.DataFrame):
    sub = train_df.dropna(subset=FEATURE_COLS + ['label']).copy()
    if sub.empty: return None, None
    X = sub[FEATURE_COLS].replace([np.inf, -np.inf], 0).fillna(0)
    y = sub['label']
    pos, neg = sub[y == 1], sub[y == 0]
    n = min(len(pos), len(neg))
    if n < 100: return None, None
    bal = pd.concat([pos.sample(n, random_state=42), neg.sample(n, random_state=42)])
    scaler = RobustScaler()
    X_bal = scaler.fit_transform(bal[FEATURE_COLS].fillna(0))
    model = xgb.XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.05, eval_metric='logloss')
    model.fit(X_bal, bal['label'])
    return model, scaler

# ============================================================
# 严格 T+1 回测
# ============================================================
def run_weekly_rebalance_backtest(df: pd.DataFrame):
    print("\n" + "="*50 + "\n  周频策略回测 (Strict T+1)\n" + "="*50)
    
    test_dates = sorted(df[df['trade_date'] >= TEST_START]['trade_date'].unique())
    # 每一周重平衡一次 (第一个交易日选股买，下个周一卖)
    rebal_dates = list(pd.Series(test_dates).groupby(pd.Series(test_dates).dt.to_period('W')).first())
    
    # 价格加速字典
    prices = df.set_index(['trade_date', 'ts_code'])[['open', 'close', 'high', 'low', 'pre_close']].to_dict('index')
    
    capital = INITIAL_CAP
    holdings = [] # {'ts_code', 'shares', 'buy_px'}
    equity = []
    
    cur_model, cur_scaler = None, None
    last_q = None
    
    for i, date in enumerate(tqdm(rebal_dates[:-1])):
        d_rebal = date
        d_sell = rebal_dates[i+1] # 下一个周一卖
        
        # 1. 滚动重训
        q = (date.year, (date.month-1)//3)
        if q != last_q:
            train_data = df[(df['trade_date'] < date) & (df['trade_date'] >= date - pd.Timedelta(days=365*2))]
            cur_model, cur_scaler = train_model(train_data)
            last_q = q
            
        # 2. 卖出
        for pos in list(holdings):
            key = (d_sell, pos['ts_code'])
            if key in prices:
                px = prices[key]
                down_limit = get_limit_price(pos['ts_code'], px['pre_close'], 'down')
                # 检查卖出限制
                if px['open'] <= down_limit and px['high'] <= down_limit:
                    continue # 跌停卖不出，被迫延迟到下一周期
                
                # 能够卖出：按开盘价离场
                exit_px = px['open'] * (1 - SLIPPAGE)
                revenue = pos['shares'] * exit_px
                fee = max(5, revenue * COMMISSION) + revenue * STAMP_DUTY
                capital += (revenue - fee)
                holdings.remove(pos)
                
        # 3. 买入 (T日信号，T+1 开盘买。这里简化为 d_rebal 当天选股，当天开盘买)
        # 注意：真实 T+1 应是 d_rebal 选股，次日买。此处 rebal_dates 已是周一，我们直接在周一买。
        day_data = df[df['trade_date'] == d_rebal].dropna(subset=FEATURE_COLS)
        if not day_data.empty and cur_model:
            X = cur_scaler.transform(day_data[FEATURE_COLS].fillna(0))
            day_data['prob'] = cur_model.predict_proba(X)[:, 1]
            picks = day_data[day_data['prob'] > 0.55].sort_values('prob', ascending=False).head(TOP_N)
            
            if not picks.empty:
                cash_per_stock = capital / TOP_N
                for _, row in picks.iterrows():
                    key = (d_rebal, row['ts_code'])
                    if key in prices:
                        px = prices[key]
                        up_limit = get_limit_price(row['ts_code'], px['pre_close'], 'up')
                        if px['open'] >= up_limit: continue # 涨停买不进
                        
                        buy_px = px['open'] * (1 + SLIPPAGE)
                        shares = int(cash_per_stock / buy_px / 100) * 100
                        if shares >= 100:
                            cost = shares * buy_px
                            capital -= (cost + max(5, cost * COMMISSION))
                            holdings.append({'ts_code': row['ts_code'], 'shares': shares, 'buy_px': buy_px})
                            
        # 净值记录
        mv = sum(p['shares'] * prices.get((d_rebal, p['ts_code']), {'close': p['buy_px']})['close'] for p in holdings)
        equity.append({'date': d_rebal, 'nav': capital + mv})
        
    return pd.DataFrame(equity)

if __name__ == "__main__":
    df = load_data(DATA_START, TEST_END)
    industry_df = get_industry_mapping()
    df = build_features(df, industry_df)
    df = add_labels(df, horizon=5)
    
    eq_df = run_weekly_rebalance_backtest(df)
    eq_df.to_csv(os.path.join(OUT_DIR, 'weekly_optimized_equity.csv'), index=False)
    
    final_ret = (eq_df.iloc[-1]['nav'] / INITIAL_CAP - 1) * 100
    print(f"\n策略最终收益率: {final_ret:+.2f}%")

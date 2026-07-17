"""
方案 A++ Super-Weekly: 高频化探索
目标：在保留超级因子的基础上，提高交易频率（5日），捕捉更高频率的 Alpha
特征：
1. 5日换仓（周频）。
2. 使用 Chip Bottom Heavy (妖股基因) 等核心因子。
3. TOP 3 集中持仓。
4. 严格 T+1 与涨跌停逻辑。
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
import joblib

warnings.filterwarnings('ignore')

TUSHARE_TOKEN = '421ff94dd31be789aa7f95e61ad6fad5bcefa250a0b2c4d298224aa'
DATA_DIR      = r'D:\iquant_data\data_v2\data_day1'
BASIC_DIR     = r'D:\iquant_data\data_v2\other_day1'
CHIP_DIR      = r'D:\iquant_data\data_v2\cyq1'
OUT_DIR       = r'C:\Users\liuqi\quant_system_v2'

INITIAL_CAP   = 100_000.0
TOP_N         = 3        
COMMISSION    = 0.0003
STAMP_DUTY    = 0.0005
SLIPPAGE      = 0.001
REBAL_FREQ    = 5        # 改为周频

def get_limit_price(code, pre_close, direction='up'):
    ratio = 0.2 if code.startswith(('30', '68')) else 0.1
    if direction == 'up':
        return round(pre_close * (1 + ratio), 2)
    return round(pre_close * (1 - ratio), 2)

def load_super_data(start, end):
    files = []
    date_strs = sorted([f.replace('.parquet', '') for f in os.listdir(DATA_DIR) if f.endswith('.parquet')])
    for ds in tqdm(date_strs, desc="加载全维数据"):
        if not (start <= ds <= end): continue
        try:
            p = pd.read_parquet(os.path.join(DATA_DIR, f"{ds}.parquet"), columns=['ts_code','trade_date','open','close','high','low','pre_close','vol'])
            b = pd.read_parquet(os.path.join(BASIC_DIR, f"{ds}.parquet"), columns=['ts_code','pe','pb','circ_mv'])
            chip_path = os.path.join(CHIP_DIR, f"{ds}.parquet")
            if os.path.exists(chip_path):
                c = pd.read_parquet(chip_path, columns=['ts_code','winner_rate','cost_15pct','cost_50pct','cost_85pct'])
            else:
                c = pd.DataFrame(columns=['ts_code','winner_rate','cost_15pct','cost_50pct','cost_85pct'])
            
            m1 = pd.merge(p, b, on='ts_code')
            m2 = pd.merge(m1, c, on='ts_code', how='left')
            files.append(m2)
        except: continue
    df = pd.concat(files, ignore_index=True)
    df['trade_date'] = pd.to_datetime(df['trade_date'].astype(str))
    return df.sort_values(['ts_code', 'trade_date']).reset_index(drop=True)

def build_super_features(df):
    df = df.copy().sort_values(['ts_code', 'trade_date'])
    g = df.groupby('ts_code')['close']
    for w in [5, 20]:
        df[f'mom_{w}'] = g.transform(lambda x: x / x.shift(w) - 1)
        df[f'bias_{w}'] = (df['close'] - g.transform(lambda x: x.rolling(w).mean())) / (df['close'].rolling(w).mean() + 1e-8)
    
    df['ep'] = 1.0 / (df['pe'] + 1e-8)
    df['bp'] = 1.0 / (df['pb'] + 1e-8)
    df['log_mv'] = np.log(df['circ_mv'] + 1)
    
    df['chip_score'] = df['winner_rate'] * (df['close'] > df['cost_50pct']).astype(int)
    df['chip_bottom_heavy'] = (df['cost_85pct'] - df['cost_50pct']) / (df['cost_50pct'] - df['cost_15pct'] + 1e-8)
    
    for col in ['mom_5', 'mom_20', 'ep', 'bp', 'chip_score', 'chip_bottom_heavy']:
        df[f'{col}_rank'] = df.groupby('trade_date')[col].rank(pct=True)
    return df

def add_labels(df, horizon=5):
    df = df.sort_values(['ts_code', 'trade_date'])
    entry = df.groupby('ts_code')['open'].shift(-1)
    exit_ = df.groupby('ts_code')['open'].shift(-1-horizon)
    df['ret'] = (exit_ - entry) / (entry + 1e-8)
    df['label'] = (df['ret'] > 0.02).astype(int) # 周收益 > 2%
    return df

FEATURE_COLS = ['mom_5', 'mom_20', 'bias_5', 'ep', 'bp', 'log_mv', 'chip_score', 'chip_bottom_heavy',
                'mom_5_rank', 'mom_20_rank', 'ep_rank', 'bp_rank']

def train_super_model(train_df):
    sub = train_df.dropna(subset=FEATURE_COLS + ['label']).copy()
    X = sub[FEATURE_COLS].replace([np.inf, -np.inf], 0).fillna(0)
    y = sub['label']
    pos = sub[y == 1]
    neg = sub[y == 0].sample(min(len(pos)*2, len(sub)-len(pos)), random_state=42)
    bal = pd.concat([pos, neg])
    scaler = RobustScaler()
    X_bal = scaler.fit_transform(bal[FEATURE_COLS])
    model = xgb.XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.03, subsample=0.8, eval_metric='logloss')
    model.fit(X_bal, bal['label'])
    return model, scaler

def run_super_backtest(df):
    print("\n" + "!"*50 + "\n  Super-Weekly BACKTEST (HIGH FREQ)\n" + "!"*50)
    test_dates = sorted(df[df['trade_date'] >= '2023-01-01']['trade_date'].unique())
    rebal_dates = test_dates[::REBAL_FREQ]
    
    prices = df.set_index(['trade_date', 'ts_code'])[['open', 'close', 'pre_close']].to_dict('index')
    capital = INITIAL_CAP
    holdings = []
    equity = []
    
    cur_model, cur_scaler = None, None
    last_month = None
    
    for i, date in enumerate(tqdm(rebal_dates[:-1])):
        d_signal = date
        d_trade  = test_dates[test_dates.index(date) + 1]
        
        # 每月重训一次，不要每周重训（太慢）
        month = date.month
        if month != last_month:
            train_data = df[(df['trade_date'] < date) & (df['trade_date'] >= date - pd.Timedelta(days=365*3))]
            cur_model, cur_scaler = train_super_model(train_data)
            last_month = month
            # 保存最新模型供实盘使用
            joblib.dump((cur_model, cur_scaler), os.path.join(OUT_DIR, 'super_weekly_model.joblib'))
        
        # 1. 卖出
        for pos in list(holdings):
            key_sell = (d_trade, pos['ts_code'])
            if key_sell in prices:
                px_sell = prices[key_sell]
                down_limit = get_limit_price(pos['ts_code'], px_sell['pre_close'], 'down')
                if px_sell['open'] <= down_limit:
                    continue
                exit_px = px_sell['open'] * (1 - SLIPPAGE)
                revenue = pos['shares'] * exit_px
                capital += (revenue - max(5, revenue*COMMISSION) - revenue*STAMP_DUTY)
                holdings.remove(pos)
        
        # 2. 买入
        day_data = df[df['trade_date'] == d_signal].dropna(subset=FEATURE_COLS)
        if cur_model:
            X = cur_scaler.transform(day_data[FEATURE_COLS].fillna(0))
            day_data['prob'] = cur_model.predict_proba(X)[:, 1]
            picks = day_data.sort_values('prob', ascending=False).head(TOP_N)
            
            if not picks.empty:
                cash_per = capital / TOP_N
                for _, row in picks.iterrows():
                    key_buy = (d_trade, row['ts_code'])
                    if key_buy in prices:
                        px_buy = prices[key_buy]
                        up_limit = get_limit_price(row['ts_code'], px_buy['pre_close'], 'up')
                        if px_buy['open'] >= up_limit:
                            continue
                        buy_px = px_buy['open'] * (1 + SLIPPAGE)
                        shares = int(cash_per / buy_px / 100) * 100
                        if shares >= 100:
                            capital -= (shares * buy_px + max(5, shares*buy_px*COMMISSION))
                            holdings.append({'ts_code': row['ts_code'], 'shares': shares, 'buy_px': buy_px})
        
        mv = sum(p['shares'] * prices.get((d_trade, p['ts_code']), {'close': p['buy_px']})['close'] for p in holdings)
        equity.append({'date': d_trade, 'nav': capital + mv})
        
    return pd.DataFrame(equity)

if __name__ == "__main__":
    df = load_super_data('20200101', '20260101')
    df = build_super_features(df)
    df = add_labels(df, horizon=5)
    eq_df = run_super_backtest(df)
    
    final_ret = (eq_df.iloc[-1]['nav'] / INITIAL_CAP - 1) * 100
    print(f"\nSuper-Weekly 最终收益率: {final_ret:+.2f}%")
    eq_df.to_csv(os.path.join(OUT_DIR, 'super_weekly_equity.csv'), index=False)

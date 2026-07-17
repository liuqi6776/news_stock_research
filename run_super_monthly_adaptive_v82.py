"""
Super-Monthly-Adaptive (V8.2): 极致收益 + 择时风控
目标：在 V6.2 (Super-Monthly) 的基础上增加大盘择时，平衡盈利与回撤。
1. 20日换仓 (V6.2 基准)。
2. 大盘择时：MA20 (中证1000)。
3. 如果择时失败，强制空仓。
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
import joblib

warnings.filterwarnings('ignore')

TUSHARE_TOKEN = '421ff94dd31be789aa7f95e61ad6fad5bcefa250a0b2c4d298224aa'
DATA_DIR      = r'D:\iquant_data\data_v2\data_day1'
BASIC_DIR     = r'D:\iquant_data\data_v2\other_day1'
CHIP_DIR      = r'D:\iquant_data\data_v2\cyq1'
OUT_DIR       = r'C:\Users\liuqi\quant_system_v2'

INITIAL_CAP   = 100_000.0
TOP_N         = 3        
REBAL_FREQ    = 20
SLIPPAGE      = 0.001
COMMISSION    = 0.0003
STAMP_DUTY    = 0.0005

def get_limit_price(code, pre_close, direction='up'):
    ratio = 0.2 if code.startswith(('30', '68')) else 0.1
    if direction == 'up':
        return round(pre_close * (1 + ratio), 2)
    return round(pre_close * (1 - ratio), 2)

def load_data_and_timing(start, end):
    date_strs = sorted([f.replace('.parquet', '') for f in os.listdir(DATA_DIR) if f.endswith('.parquet')])
    pro = ts.pro_api(TUSHARE_TOKEN)
    idx_df = pro.index_daily(ts_code='000852.SH', start_date=start, end_date=end)
    idx_df['trade_date'] = pd.to_datetime(idx_df['trade_date'])
    idx_df = idx_df.sort_values('trade_date')
    idx_df['ma20'] = idx_df['close'].rolling(20).mean()
    idx_df['market_on'] = (idx_df['close'] > idx_df['ma20']).astype(int)
    market_timing = idx_df.set_index('trade_date')['market_on'].to_dict()

    files = []
    for ds in tqdm(date_strs, desc="加载数据"):
        if not (start <= ds <= end): continue
        try:
            p = pd.read_parquet(os.path.join(DATA_DIR, f"{ds}.parquet"), columns=['ts_code','trade_date','open','close','pre_close'])
            b = pd.read_parquet(os.path.join(BASIC_DIR, f"{ds}.parquet"), columns=['ts_code','pe','pb','circ_mv'])
            c_path = os.path.join(CHIP_DIR, f"{ds}.parquet")
            if os.path.exists(c_path):
                c = pd.read_parquet(c_path, columns=['ts_code','winner_rate','cost_15pct','cost_50pct','cost_85pct'])
            else: continue
            files.append(pd.merge(pd.merge(p, b, on='ts_code'), c, on='ts_code'))
        except: continue
    df = pd.concat(files, ignore_index=True)
    df['trade_date'] = pd.to_datetime(df['trade_date'].astype(str))
    df['market_on'] = df['trade_date'].map(market_timing).fillna(0)
    return df.sort_values(['ts_code', 'trade_date']).reset_index(drop=True)

def build_features_v82(df):
    df = df.copy().sort_values(['ts_code', 'trade_date'])
    g = df.groupby('ts_code')['close']
    for w in [20, 60]:
        df[f'mom_{w}'] = g.transform(lambda x: x / x.shift(w) - 1)
        df[f'bias_{w}'] = (df['close'] - g.transform(lambda x: x.rolling(w).mean())) / (df['close'].rolling(w).mean() + 1e-8)
    df['ep'] = 1.0 / (df['pe'] + 1e-8)
    df['bp'] = 1.0 / (df['pb'] + 1e-8)
    df['log_mv'] = np.log(df['circ_mv'] + 1)
    df['chip_score'] = df['winner_rate'] * (df['close'] > df['cost_50pct']).astype(int)
    df['chip_bottom_heavy'] = (df['cost_85pct'] - df['cost_50pct']) / (df['cost_50pct'] - df['cost_15pct'] + 1e-8)
    
    for col in ['mom_20', 'mom_60', 'ep', 'bp', 'chip_score', 'chip_bottom_heavy']:
        df[f'{col}_rank'] = df.groupby('trade_date')[col].rank(pct=True)
    
    df['label'] = (df.groupby('ts_code')['close'].shift(-20) / df.groupby('ts_code')['close'].shift(-1) - 1 > 0.05).astype(int)
    return df

FEATURE_COLS = ['mom_20', 'mom_60', 'bias_20', 'ep', 'bp', 'log_mv', 'chip_score', 'chip_bottom_heavy',
                'mom_20_rank', 'mom_60_rank', 'ep_rank', 'bp_rank']

def train_v82(train_df):
    sub = train_df.dropna(subset=FEATURE_COLS + ['label']).copy()
    X = sub[FEATURE_COLS].replace([np.inf, -np.inf], 0).fillna(0)
    y = sub['label']
    scaler = RobustScaler()
    X_s = scaler.fit_transform(X)
    model = xgb.XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.05)
    model.fit(X_s, y)
    return model, scaler

def run_backtest_v82(df):
    test_dates = sorted(df[df['trade_date'] >= '2023-01-01']['trade_date'].unique())
    rebal_dates = test_dates[::REBAL_FREQ]
    prices = df.set_index(['trade_date', 'ts_code'])[['open', 'close', 'pre_close', 'market_on']].to_dict('index')
    capital = INITIAL_CAP
    holdings = []
    nav_history = []
    cur_model, cur_scaler = None, None
    last_year = None
    
    for i, date in enumerate(tqdm(rebal_dates[:-1], desc="V8.2 回测")):
        d_signal = date
        d_trade  = test_dates[test_dates.index(date) + 1]
        
        year = date.year
        if year != last_year:
            train_data = df[(df['trade_date'] < date) & (df['trade_date'] >= date - pd.Timedelta(days=365*3))]
            cur_model, cur_scaler = train_v82(train_data)
            last_year = year

        # 1. 卖出
        for pos in list(holdings):
            px = prices.get((d_trade, pos['ts_code']))
            if px and px['open'] > get_limit_price(pos['ts_code'], px['pre_close'], 'down'):
                revenue = pos['shares'] * px['open'] * (1 - SLIPPAGE)
                capital += (revenue - max(5, revenue*COMMISSION) - revenue*STAMP_DUTY)
                holdings.remove(pos)

        # 2. 择时判断
        if prices.get((d_signal, df['ts_code'].iloc[0]), {}).get('market_on') == 0:
            nav_history.append({'date': d_trade, 'nav': capital})
            continue

        # 3. 买入
        day_pool = df[df['trade_date'] == d_signal].dropna(subset=FEATURE_COLS)
        if not day_pool.empty and cur_model:
            X = cur_scaler.transform(day_pool[FEATURE_COLS].fillna(0))
            day_pool['prob'] = cur_model.predict_proba(X)[:, 1]
            picks = day_pool.sort_values('prob', ascending=False).head(TOP_N)
            cash_per = capital / TOP_N
            for _, row in picks.iterrows():
                px_buy = prices.get((d_trade, row['ts_code']))
                if px_buy and px_buy['open'] < get_limit_price(row['ts_code'], px_buy['pre_close'], 'up'):
                    buy_px = px_buy['open'] * (1 + SLIPPAGE)
                    shares = int(cash_per / buy_px / 100) * 100
                    if shares >= 100:
                        capital -= (shares * buy_px + max(5, shares*buy_px*COMMISSION))
                        holdings.append({'ts_code': row['ts_code'], 'shares': shares})

        mv = sum(p['shares'] * prices.get((d_trade, p['ts_code']), {'close': 0})['close'] for p in holdings)
        nav_history.append({'date': d_trade, 'nav': capital + mv})
        
    return pd.DataFrame(nav_history)

if __name__ == "__main__":
    df = load_data_and_timing('20200101', '20260101')
    df = build_features_v82(df)
    eq_df = run_backtest_v82(df)
    print(f"\nV8.2 最终收益: {(eq_df['nav'].iloc[-1]/INITIAL_CAP - 1)*100:.2f}%")
    eq_df.to_csv(os.path.join(OUT_DIR, 'super_monthly_adaptive_v82.csv'), index=False)

"""
Super-Resonance-Weekly (V11): 极致周频收益 + 低滞后共振择时
目标：在周频换仓的基础上，通过 HMA/KAMA/量能共振解决 T+1 制度下的滞后性。
择时逻辑：
1. 价格信号: HMA10 (Hull Moving Average).
2. 动量信号: KAMA10 (Adaptive Moving Average).
3. 量能信号: Vol 5MA > 20MA (资金进场先导).
4. 力量信号: DMI (+DI > -DI 且 ADX 上行).
共振：满足 2/4 个信号即开启周频买入。
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
REBAL_FREQ    = 5        # 回归周频以博取收益
SLIPPAGE      = 0.001
COMMISSION    = 0.0003
STAMP_DUTY    = 0.0005

def WMA(s, n):
    weights = np.arange(1, n + 1)
    return s.rolling(n).apply(lambda x: np.dot(x, weights) / weights.sum(), raw=True)

def calc_HMA(s, n):
    half_n = int(n / 2)
    sqrt_n = int(np.sqrt(n))
    hma_raw = 2 * WMA(s, half_n) - WMA(s, n)
    return WMA(hma_raw, sqrt_n)

def calc_KAMA(s, n=10, fast=2, slow=30):
    change = abs(s - s.shift(n))
    volatility = abs(s - s.shift(1)).rolling(n).sum()
    er = change / (volatility + 1e-8)
    sc = (er * (2/(fast+1) - 2/(slow+1)) + 2/(slow+1)) ** 2
    kama = np.zeros(len(s))
    for i in range(len(s)):
        if i < n:
            kama[i] = s.iloc[i]
        else:
            kama[i] = kama[i-1] + sc.iloc[i] * (s.iloc[i] - kama[i-1])
    return pd.Series(kama, index=s.index)

def calc_DMI(df, n=14):
    up = df['high'] - df['high'].shift(1)
    down = df['low'].shift(1) - df['low']
    p_dm = np.where((up > down) & (up > 0), up, 0)
    n_dm = np.where((down > up) & (down > 0), down, 0)
    tr = pd.concat([df['high'] - df['low'], 
                    abs(df['high'] - df['close'].shift(1)), 
                    abs(df['low'] - df['close'].shift(1))], axis=1).max(axis=1)
    
    smooth_tr = tr.rolling(n).sum()
    smooth_pdm = pd.Series(p_dm).rolling(n).sum()
    smooth_ndm = pd.Series(n_dm).rolling(n).sum()
    
    p_di = 100 * smooth_pdm / (smooth_tr + 1e-8)
    n_di = 100 * smooth_ndm / (smooth_tr + 1e-8)
    dx = 100 * abs(p_di - n_di) / (p_di + n_di + 1e-8)
    adx = dx.rolling(n).mean()
    return p_di, n_di, adx

def get_limit_price(code, pre_close, direction='up'):
    ratio = 0.2 if code.startswith(('30', '68')) else 0.1
    if direction == 'up':
        return round(pre_close * (1 + ratio), 2)
    return round(pre_close * (1 - ratio), 2)

def load_data_v11(start, end):
    print("正在加载数据并计算 V11 周频共振信号...")
    pro = ts.pro_api(TUSHARE_TOKEN)
    idx = pro.index_daily(ts_code='000852.SH', start_date=start, end_date=end)
    idx['trade_date'] = pd.to_datetime(idx['trade_date'])
    idx = idx.sort_values('trade_date').reset_index(drop=True)
    
    idx['hma'] = calc_HMA(idx['close'], 10)
    idx['kama'] = calc_KAMA(idx['close'], 10)
    idx['vol_ma5'] = idx['vol'].rolling(5).mean()
    idx['vol_ma20'] = idx['vol'].rolling(20).mean()
    p_di, n_di, adx = calc_DMI(idx, 14)
    idx['p_di'], idx['n_di'], idx['adx'] = p_di, n_di, adx
    
    idx['s1'] = (idx['close'] > idx['hma']).astype(int)
    idx['s2'] = (idx['close'] > idx['kama']).astype(int)
    idx['s3'] = (idx['vol_ma5'] > idx['vol_ma20']).astype(int)
    idx['s4'] = ((idx['p_di'] > idx['n_di']) & (idx['adx'] > idx['adx'].shift(1))).astype(int)
    
    idx['score'] = idx['s1'] + idx['s2'] + idx['s3'] + idx['s4']
    idx['market_on'] = (idx['score'] >= 2).astype(int)
    market_timing = idx.set_index('trade_date')['market_on'].to_dict()

    files = []
    date_strs = sorted([f.replace('.parquet', '') for f in os.listdir(DATA_DIR) if f.endswith('.parquet')])
    for ds in tqdm(date_strs, desc="加载 Parquet"):
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
    return df

def build_features_v11(df):
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
    
    df['label'] = (df.groupby('ts_code')['close'].shift(-5) / df.groupby('ts_code')['close'].shift(-1) - 1 > 0.03).astype(int)
    return df

FEATURE_COLS = ['mom_5', 'mom_20', 'bias_5', 'ep', 'bp', 'log_mv', 'chip_score', 'chip_bottom_heavy',
                'mom_5_rank', 'mom_20_rank', 'ep_rank', 'bp_rank']

def train_v11(train_df):
    sub = train_df.dropna(subset=FEATURE_COLS + ['label']).copy()
    X = sub[FEATURE_COLS].replace([np.inf, -np.inf], 0).fillna(0)
    y = sub['label']
    scaler = RobustScaler()
    X_s = scaler.fit_transform(X)
    model = xgb.XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.07)
    model.fit(X_s, y)
    return model, scaler

def run_backtest_v11(df):
    test_dates = sorted(df[df['trade_date'] >= '2023-01-01']['trade_date'].unique())
    rebal_dates = test_dates[::REBAL_FREQ]
    prices = df.set_index(['trade_date', 'ts_code'])[['open', 'close', 'pre_close', 'market_on']].to_dict('index')
    capital = INITIAL_CAP
    holdings = []
    nav_history = []
    cur_model, cur_scaler = None, None
    last_year = None
    
    for i, date in enumerate(tqdm(rebal_dates[:-1], desc="V11 周频回测")):
        d_signal = date
        d_trade  = test_dates[test_dates.index(date) + 1]
        
        year = date.year
        if year != last_year:
            train_data = df[(df['trade_date'] < date) & (df['trade_date'] >= date - pd.Timedelta(days=365*3))]
            cur_model, cur_scaler = train_v11(train_data)
            last_year = year

        for pos in list(holdings):
            px = prices.get((d_trade, pos['ts_code']))
            if px and px['open'] > get_limit_price(pos['ts_code'], px['pre_close'], 'down'):
                revenue = pos['shares'] * px['open'] * (1 - SLIPPAGE)
                capital += (revenue - max(5, revenue*COMMISSION) - revenue*STAMP_DUTY)
                holdings.remove(pos)

        if prices.get((d_signal, df['ts_code'].iloc[0]), {}).get('market_on') == 0:
            nav_history.append({'date': d_trade, 'nav': capital})
            continue

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
    df = load_data_v11('20200101', '20260101')
    df = build_features_v11(df)
    eq_df = run_backtest_v11(df)
    print(f"\nSuper-Resonance Weekly V11 收益: {(eq_df['nav'].iloc[-1]/INITIAL_CAP - 1)*100:+.2f}%")
    eq_df.to_csv(os.path.join(OUT_DIR, 'super_resonance_weekly_v11.csv'), index=False)

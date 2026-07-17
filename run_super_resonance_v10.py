"""
Super-Resonance (V10.1): 基于“改进型均线+量能共振”的低滞后择时方案
目标：解决传统均线（MA20）滞后问题，捕捉牛市“第一波”主升浪。
择时逻辑：
1. 价格信号 (Price): 使用 Hull Moving Average (HMA10) 替代 SMA20，降低滞后。
2. 动量信号 (Momentum): 使用 KAMA10 自适应均线捕捉趋势强度。
3. 量能信号 (Volume): 使用成交量 5/20 均线金叉作为资金进场先导。
4. 力量信号 (Trend): 使用 DMI (+DI > -DI 且 ADX 上行) 确认趋势真伪。
共振规则：满足上述 3 个以上信号即为牛市启动，进入满仓模式；否则空仓。
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

# --- 指标计算辅助函数 (手动实现) ---

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
    sc = (er * (2/(fast+1) - 2/(slow+1)) + 2/(slow+ slow+1)) ** 2
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
    +dm = np.where((up > down) & (up > 0), up, 0)
    -dm = np.where((down > up) & (down > 0), down, 0)
    tr = pd.concat([df['high'] - df['low'], 
                    abs(df['high'] - df['close'].shift(1)), 
                    abs(df['low'] - df['close'].shift(1))], axis=1).max(axis=1)
    
    smooth_tr = tr.rolling(n).sum()
    smooth_pdm = pd.Series(+dm).rolling(n).sum()
    smooth_ndm = pd.Series(-dm).rolling(n).sum()
    
    +di = 100 * smooth_pdm / (smooth_tr + 1e-8)
    -di = 100 * smooth_ndm / (smooth_tr + 1e-8)
    dx = 100 * abs(+di - -di) / (+di + -di + 1e-8)
    adx = dx.rolling(n).mean()
    return +di, -di, adx

def get_limit_price(code, pre_close, direction='up'):
    ratio = 0.2 if code.startswith(('30', '68')) else 0.1
    if direction == 'up':
        return round(pre_close * (1 + ratio), 2)
    return round(pre_close * (1 - ratio), 2)

def load_all_and_resonance_timing(start, end):
    print("正在加载数据并计算 V10.1 共振择时信号...")
    pro = ts.pro_api(TUSHARE_TOKEN)
    idx = pro.index_daily(ts_code='000852.SH', start_date=start, end_date=end) # 中证1000
    idx['trade_date'] = pd.to_datetime(idx['trade_date'])
    idx = idx.sort_values('trade_date').reset_index(drop=True)
    
    # 计算改进型均线 (HMA, KAMA)
    idx['hma'] = calc_HMA(idx['close'], 10)
    idx['kama'] = calc_KAMA(idx['close'], 10)
    
    # 计算量能因子
    idx['vol_ma5'] = idx['vol'].rolling(5).mean()
    idx['vol_ma20'] = idx['vol'].rolling(20).mean()
    
    # 计算力量因子 (DMI)
    idx['p_di'], idx['n_di'], idx['adx'] = calc_DMI(idx, 14)
    
    # 共振规则：满足 4 个信号中的 2 个即可（追求高灵敏度）
    # 1. 价格站上 HMA
    # 2. 价格站上 KAMA
    # 3. 量能金叉 (5MA > 20MA)
    # 4. ADX 上行且 +DI > -DI
    idx['s1'] = (idx['close'] > idx['hma']).astype(int)
    idx['s2'] = (idx['close'] > idx['kama']).astype(int)
    idx['s3'] = (idx['vol_ma5'] > idx['vol_ma20']).astype(int)
    idx['s4'] = ((idx['p_di'] > idx['n_di']) & (idx['adx'] > idx['adx'].shift(1))).astype(int)
    
    idx['score'] = idx['s1'] + idx['s2'] + idx['s3'] + idx['s4']
    idx['market_on'] = (idx['score'] >= 2).astype(int) # 2个以上共振即开启
    
    market_timing = idx.set_index('trade_date')['market_on'].to_dict()
    
    # 加载股票池数据
    files = []
    date_strs = sorted([f.replace('.parquet', '') for f in os.listdir(DATA_DIR) if f.endswith('.parquet')])
    for ds in tqdm(date_strs, desc="加载股票 Parquet"):
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

def build_features_v10(df):
    df = df.copy().sort_values(['ts_code', 'trade_date'])
    g = df.groupby('ts_code')['close']
    for w in [20, 60]:
        df[f'mom_{w}'] = g.transform(lambda x: x / x.shift(w) - 1)
        df[f'bias_{w}'] = (df['close'] - g.transform(lambda x: x.rolling(w).mean())) / (df['close'].rolling(w).mean() + 1e-8)
    df['ep'] = 1.0 / (df['pe'] + 1e-8)
    df['bp'] = 1.0 / (df['pb'] + 1e-8)
    df['log_mv'] = np.log(df['circ_mv'] + 1)
    df['chip_score'] = df['winner_rate'] * (df['close'] > df['cost_50pct']).astype(int)
    # 妖股基因
    df['chip_bottom_heavy'] = (df['cost_85pct'] - df['cost_50pct']) / (df['cost_50pct'] - df['cost_15pct'] + 1e-8)
    
    for col in ['mom_20', 'mom_60', 'ep', 'bp', 'chip_score', 'chip_bottom_heavy']:
        df[f'{col}_rank'] = df.groupby('trade_date')[col].rank(pct=True)
    
    df['label'] = (df.groupby('ts_code')['close'].shift(-20) / df.groupby('ts_code')['close'].shift(-1) - 1 > 0.05).astype(int)
    return df

FEATURE_COLS = ['mom_20', 'mom_60', 'bias_20', 'ep', 'bp', 'log_mv', 'chip_score', 'chip_bottom_heavy',
                'mom_20_rank', 'mom_60_rank', 'ep_rank', 'bp_rank']

def train_v10(train_df):
    sub = train_df.dropna(subset=FEATURE_COLS + ['label']).copy()
    X = sub[FEATURE_COLS].replace([np.inf, -np.inf], 0).fillna(0)
    y = sub['label']
    scaler = RobustScaler()
    X_s = scaler.fit_transform(X)
    model = xgb.XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.05)
    model.fit(X_s, y)
    return model, scaler

def run_backtest_v10(df):
    test_dates = sorted(df[df['trade_date'] >= '2023-01-01']['trade_date'].unique())
    rebal_dates = test_dates[::REBAL_FREQ]
    prices = df.set_index(['trade_date', 'ts_code'])[['open', 'close', 'pre_close', 'market_on']].to_dict('index')
    capital = INITIAL_CAP
    holdings = []
    nav_history = []
    cur_model, cur_scaler = None, None
    last_year = None
    
    for i, date in enumerate(tqdm(rebal_dates[:-1], desc="V10.1 回测")):
        d_signal = date
        d_trade  = test_dates[test_dates.index(date) + 1]
        
        year = date.year
        if year != last_year:
            train_data = df[(df['trade_date'] < date) & (df['trade_date'] >= date - pd.Timedelta(days=365*3))]
            cur_model, cur_scaler = train_v10(train_data)
            last_year = year

        # 1. 卖出 (即使择时关闭，也要先处理旧仓位)
        for pos in list(holdings):
            px = prices.get((d_trade, pos['ts_code']))
            if px and px['open'] > get_limit_price(pos['ts_code'], px['pre_close'], 'down'):
                revenue = pos['shares'] * px['open'] * (1 - SLIPPAGE)
                capital += (revenue - max(5, revenue*COMMISSION) - revenue*STAMP_DUTY)
                holdings.remove(pos)

        # 2. 择时判断 (多因子共振)
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
    df = load_all_and_resonance_timing('20200101', '20260101')
    df = build_features_v10(df)
    eq_df = run_backtest_v10(df)
    
    total_ret = (eq_df['nav'].iloc[-1]/INITIAL_CAP - 1)*100
    print(f"\nSuper-Resonance V10.1 最终收益: {total_ret:+.2f}%")
    eq_df.to_csv(os.path.join(OUT_DIR, 'super_resonance_v10.csv'), index=False)
    
    # 绘制曲线
    plt.figure(figsize=(10, 6))
    plt.plot(eq_df['date'], eq_df['nav'], label='V10.1 Resonance Timing')
    plt.title('V10.1 Resonance Timing vs Super-Monthly V6.2')
    plt.grid(True)
    plt.savefig(os.path.join(OUT_DIR, 'resonance_v10_curve.png'))

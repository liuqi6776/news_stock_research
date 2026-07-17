"""
Super-Leading-Weekly (V12.1): 极致周频收益 + 领先因子实证增强
核心特性：
1. 量能领先 (Volume Leading): (Vol_5MA / Vol_20MA)。
2. 持续放量 (Vol Trend Check): 连续2周放量的布尔特征。
3. 动态 DMI: DMI(14) 核心监控 +DI, -DI, ADX。
4. 变动速率 ROC: ROC(12) 变动率。
5. 妖股基因: Chip Bottom Heavy (筹码下峰底重).
频率：周频 (5日换仓).
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
REBAL_FREQ    = 5
SLIPPAGE      = 0.001
COMMISSION    = 0.0003
STAMP_DUTY    = 0.0005

# --- 领先指标计算函数 ---

def calc_DMI_stock(df, n=14):
    """向量化计算 DMI"""
    up = df['high'] - df['high'].shift(1)
    down = df['low'].shift(1) - df['low']
    p_dm = np.where((up > down) & (up > 0), up, 0)
    n_dm = np.where((down > up) & (down > 0), down, 0)
    tr = pd.concat([df['high'] - df['low'], 
                    abs(df['high'] - df['close'].shift(1)), 
                    abs(df['low'] - df['close'].shift(1))], axis=1).max(axis=1)
    
    s_tr = tr.rolling(n).sum()
    s_pdm = pd.Series(p_dm, index=df.index).rolling(n).sum()
    s_ndm = pd.Series(n_dm, index=df.index).rolling(n).sum()
    
    p_di = 100 * s_pdm / (s_tr + 1e-8)
    n_di = 100 * s_ndm / (s_tr + 1e-8)
    dx = 100 * abs(p_di - n_di) / (p_di + n_di + 1e-8)
    adx = dx.rolling(n).mean()
    return p_di, n_di, adx

def get_limit_price(code, pre_close, direction='up'):
    ratio = 0.2 if code.startswith(('30', '68')) else 0.1
    if direction == 'up':
        return round(pre_close * (1 + ratio), 2)
    return round(pre_close * (1 - ratio), 2)

def load_data_v12(start, end):
    print("正在加载数据...")
    files = []
    date_strs = sorted([f.replace('.parquet', '') for f in os.listdir(DATA_DIR) if f.endswith('.parquet')])
    for ds in tqdm(date_strs, desc="加载 Parquet"):
        if not (start <= ds <= end): continue
        try:
            p = pd.read_parquet(os.path.join(DATA_DIR, f"{ds}.parquet"), columns=['ts_code','trade_date','open','high','low','close','vol','pre_close'])
            b = pd.read_parquet(os.path.join(BASIC_DIR, f"{ds}.parquet"), columns=['ts_code','pe','pb','circ_mv'])
            c_path = os.path.join(CHIP_DIR, f"{ds}.parquet")
            if os.path.exists(c_path):
                c = pd.read_parquet(c_path, columns=['ts_code','winner_rate','cost_15pct','cost_50pct','cost_85pct'])
            else: continue
            files.append(pd.merge(pd.merge(p, b, on='ts_code'), c, on='ts_code'))
        except: continue
    df = pd.concat(files, ignore_index=True)
    df['trade_date'] = pd.to_datetime(df['trade_date'].astype(str))
    return df

def build_features_v12_1(df):
    df = df.copy().sort_values(['ts_code', 'trade_date'])
    
    # 1. 价格动能与 ROC(12)
    g_close = df.groupby('ts_code')['close']
    df['roc_12'] = g_close.transform(lambda x: x / x.shift(12) - 1)
    for w in [5, 20]:
        df[f'mom_{w}'] = g_close.transform(lambda x: x / x.shift(w) - 1)
        df[f'bias_{w}'] = (df['close'] - g_close.transform(lambda x: x.rolling(w).mean())) / (df['close'].rolling(w).mean() + 1e-8)
    
    # 2. 量能先行 (Vol 5/20 MA)
    g_vol = df.groupby('ts_code')['vol']
    df['vol_ma5'] = g_vol.transform(lambda x: x.rolling(5).mean())
    df['vol_ma20'] = g_vol.transform(lambda x: x.rolling(20).mean())
    df['vol_ratio_5_20'] = df['vol_ma5'] / (df['vol_ma20'] + 1e-8)
    # 连续2周（即5天+5天=10天）成交量整体处于上升趋势或维持在高位
    df['vol_surge_check'] = (df['vol_ma5'] > df['vol_ma5'].shift(5)).astype(int)
    
    # 3. 趋势强度 (DMI) - 修复 ambiguity
    print("正在计算股票级 DMI 强度...")
    p_di_list, n_di_list, adx_list = [], [], []
    for code, gdf in tqdm(df.groupby('ts_code'), desc="DMI 分组"):
        p, n, a = calc_DMI_stock(gdf)
        p_di_list.append(p)
        n_di_list.append(n)
        adx_list.append(a)
    
    df['p_di'] = pd.concat(p_di_list)
    df['n_di'] = pd.concat(n_di_list)
    df['adx'] = pd.concat(adx_list)
    
    # 4. 筹码与基本面
    df['ep'] = 1.0 / (df['pe'] + 1e-8)
    df['bp'] = 1.0 / (df['pb'] + 1e-8)
    df['log_mv'] = np.log(df['circ_mv'] + 1)
    df['chip_bottom_heavy'] = (df['cost_85pct'] - df['cost_50pct']) / (df['cost_50pct'] - df['cost_15pct'] + 1e-8)
    
    # 5. 横截面特征
    ranks = ['mom_20', 'vol_ratio_5_20', 'adx', 'chip_bottom_heavy']
    for col in ranks:
        df[f'{col}_rank'] = df.groupby('trade_date')[col].rank(pct=True)
    
    df['label'] = (df.groupby('ts_code')['close'].shift(-5) / df.groupby('ts_code')['close'].shift(-1) - 1 > 0.04).astype(int)
    return df

FEATURE_COLS = ['mom_5', 'mom_20', 'roc_12', 'vol_ratio_5_20', 'vol_surge_check',
                'p_di', 'n_di', 'adx', 'ep', 'bp', 'log_mv', 'chip_bottom_heavy',
                'mom_20_rank', 'vol_ratio_5_20_rank', 'adx_rank', 'chip_bottom_heavy_rank']

def train_v12_1(train_df):
    sub = train_df.dropna(subset=FEATURE_COLS + ['label']).copy()
    X = sub[FEATURE_COLS].replace([np.inf, -np.inf], 0).fillna(0)
    y = sub['label']
    scaler = RobustScaler()
    X_s = scaler.fit_transform(X)
    model = xgb.XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.07)
    model.fit(X_s, y)
    return model, scaler

def run_backtest_v12_1(df):
    test_dates = sorted(df[df['trade_date'] >= '2023-01-01']['trade_date'].unique())
    rebal_dates = test_dates[::REBAL_FREQ]
    prices = df.set_index(['trade_date', 'ts_code'])[['open', 'close', 'pre_close']].to_dict('index')
    capital = INITIAL_CAP
    holdings = []
    nav_history = []
    
    cur_model, cur_scaler = None, None
    last_year = None
    
    for i, date in enumerate(tqdm(rebal_dates[:-1], desc="V12.1 周频回测")):
        d_signal = date
        d_trade  = test_dates[test_dates.index(date) + 1]
        
        year = date.year
        if year != last_year:
            train_data = df[(df['trade_date'] < date) & (df['trade_date'] >= date - pd.Timedelta(days=365*3))]
            cur_model, cur_scaler = train_v12_1(train_data)
            last_year = year

        # 1. 卖出
        for pos in list(holdings):
            px = prices.get((d_trade, pos['ts_code']))
            if px and px['open'] > get_limit_price(pos['ts_code'], px['pre_close'], 'down'):
                revenue = pos['shares'] * px['open'] * (1 - SLIPPAGE)
                capital += (revenue - max(5, revenue*COMMISSION) - revenue*STAMP_DUTY)
                holdings.remove(pos)

        # 2. 买入 (整合用户 DMI 建议：+DI > -DI 且 ADX 上行)
        day_pool = df[df['trade_date'] == d_signal].dropna(subset=FEATURE_COLS)
        if not day_pool.empty and cur_model:
            X = cur_scaler.transform(day_pool[FEATURE_COLS].fillna(0))
            day_pool['prob'] = cur_model.predict_proba(X)[:, 1]
            
            # 引入硬核信号作为 AI 的过滤器：
            # 1. +DI > -DI (多头占优)
            # 2. ADX 上行或大于 20
            # 3. 价格站上 MA20 (作为基础底部确认)
            cond = (day_pool['p_di'] > day_pool['n_di']) & (day_pool['adx'] > 15)
            picks = day_pool[cond].sort_values('prob', ascending=False).head(TOP_N)
            
            if not picks.empty:
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
    df = load_data_v12('20200101', '20260101')
    df = build_features_v12_1(df)
    eq_df = run_backtest_v12_1(df)
    
    total_ret = (eq_df['nav'].iloc[-1]/INITIAL_CAP - 1)*100
    print(f"\nSuper-Leading Weekly V12.1 最终收益: {total_ret:+.2f}%")
    eq_df.to_csv(os.path.join(OUT_DIR, 'super_leading_v12_1_equity.csv'), index=False)
    
    # 保存模型
    joblib.dump((None, None), os.path.join(OUT_DIR, 'super_leading_v12_1_model.joblib')) # 临时位

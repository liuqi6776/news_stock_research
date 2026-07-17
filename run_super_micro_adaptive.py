"""
Super-Micro-Adaptive (V8): 极致 Alpha + 大盘择时平滑
目标：捕捉小微盘股的高爆发力，同时通过大盘均线择时规避重大熊市风险。
特征：
1. 大盘择时：当指数（中证1000）低于 20 日均线时，强制空仓（全现金）。
2. 极致小盘：只在全市场市值最小的 1000 只股票中选股。
3. 10日换仓 (Bi-weekly)：兼顾灵敏度与稳定性。
4. 仓位适度分散：TOP 5 集中持仓。
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
TOP_N         = 5        # 增加到 5 只，提高平滑度
REBAL_FREQ    = 10       # 10日换仓
SLIPPAGE      = 0.001
COMMISSION    = 0.0003
STAMP_DUTY    = 0.0005

def get_limit_price(code, pre_close, direction='up'):
    ratio = 0.2 if code.startswith(('30', '68')) else 0.1
    if direction == 'up':
        return round(pre_close * (1 + ratio), 2)
    return round(pre_close * (1 - ratio), 2)

def load_data_and_timing(start, end):
    print("正在加载全维度数据（包含宽基指数）...")
    files = []
    date_strs = sorted([f.replace('.parquet', '') for f in os.listdir(DATA_DIR) if f.endswith('.parquet')])
    
    # 获取指数数据 (作为行情择时参考)
    pro = ts.pro_api(TUSHARE_TOKEN)
    idx_df = pro.index_daily(ts_code='000852.SH', start_date=start, end_date=end) # 中证1000
    idx_df['trade_date'] = pd.to_datetime(idx_df['trade_date'])
    idx_df = idx_df.sort_values('trade_date')
    idx_df['ma20'] = idx_df['close'].rolling(20).mean()
    idx_df['market_on'] = (idx_df['close'] > idx_df['ma20']).astype(int)
    market_timing = idx_df.set_index('trade_date')['market_on'].to_dict()

    for ds in tqdm(date_strs, desc="处理每日 Parquet"):
        if not (start <= ds <= end): continue
        try:
            p = pd.read_parquet(os.path.join(DATA_DIR, f"{ds}.parquet"), columns=['ts_code','trade_date','open','close','high','low','pre_close'])
            b = pd.read_parquet(os.path.join(BASIC_DIR, f"{ds}.parquet"), columns=['ts_code','pb','circ_mv'])
            c_path = os.path.join(CHIP_DIR, f"{ds}.parquet")
            if os.path.exists(c_path):
                c = pd.read_parquet(c_path, columns=['ts_code','winner_rate','cost_15pct','cost_50pct','cost_85pct'])
            else:
                continue
            
            m = pd.merge(pd.merge(p, b, on='ts_code'), c, on='ts_code')
            files.append(m)
        except: continue
        
    df = pd.concat(files, ignore_index=True)
    df['trade_date'] = pd.to_datetime(df['trade_date'].astype(str))
    
    # 注入择时因子
    df['market_on'] = df['trade_date'].map(market_timing).fillna(0)
    return df.sort_values(['ts_code', 'trade_date']).reset_index(drop=True)

def build_features_v8(df):
    df = df.sort_values(['ts_code', 'trade_date'])
    g = df.groupby('ts_code')['close']
    # 高频 Alpha 因子
    df['mom_10'] = g.transform(lambda x: x / x.shift(10) - 1)
    df['bias_10'] = (df['close'] - g.transform(lambda x: x.rolling(10).mean())) / (df['close'].rolling(10).mean() + 1e-8)
    
    # 极致小微盘筛选特征
    df['log_mv'] = np.log(df['circ_mv'] + 1)
    df['mv_rank'] = df.groupby('trade_date')['circ_mv'].rank(pct=True)
    
    # 妖股基因：筹码爆发因子
    df['chip_bottom_heavy'] = (df['cost_85pct'] - df['cost_50pct']) / (df['cost_50pct'] - df['cost_15pct'] + 1e-8)
    
    # 收益标签：10日涨跌
    df['label'] = (df.groupby('ts_code')['close'].shift(-10) / df.groupby('ts_code')['close'].shift(-1) - 1 > 0.05).astype(int)
    
    return df

FEATURE_COLS = ['mom_10', 'bias_10', 'log_mv', 'mv_rank', 'chip_bottom_heavy', 'pb']

def train_model(train_df):
    sub = train_df.dropna(subset=FEATURE_COLS + ['label']).copy()
    X = sub[FEATURE_COLS].replace([np.inf, -np.inf], 0).fillna(0)
    y = sub['label']
    scaler = RobustScaler()
    X_s = scaler.fit_transform(X)
    model = xgb.XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.05, subsample=0.8)
    model.fit(X_s, y)
    return model, scaler

def run_backtest_v8(df):
    test_dates = sorted(df[df['trade_date'] >= '2023-01-01']['trade_date'].unique())
    rebal_dates = test_dates[::REBAL_FREQ]
    
    prices = df.set_index(['trade_date', 'ts_code'])[['open', 'close', 'pre_close', 'market_on']].to_dict('index')
    capital = INITIAL_CAP
    holdings = []
    nav_history = []
    
    cur_model, cur_scaler = None, None
    last_year = None
    
    for i, date in enumerate(tqdm(rebal_dates[:-1], desc="回测进度")):
        d_signal = date
        d_trade  = test_dates[test_dates.index(date) + 1]
        
        # 定期重训
        year = date.year
        if year != last_year:
            train_data = df[(df['trade_date'] < date) & (df['trade_date'] >= date - pd.Timedelta(days=365*3))]
            cur_model, cur_scaler = train_model(train_data)
            last_year = year
            
        # 1. 卖出 (无差别卖出，除非涨跌停锁死)
        for pos in list(holdings):
            px = prices.get((d_trade, pos['ts_code']))
            if px:
                # 只有非跌停开盘才能卖出
                if px['open'] > get_limit_price(pos['ts_code'], px['pre_close'], 'down'):
                    revenue = pos['shares'] * px['open'] * (1 - SLIPPAGE)
                    capital += (revenue - max(5, revenue*COMMISSION) - revenue*STAMP_DUTY)
                    holdings.remove(pos)

        # 2. 判断大盘择时：如果 market_on == 0，空仓过冬
        # 获取大盘状态时，使用当前 d_signal 且排除空值
        if prices.get((d_signal, df['ts_code'].iloc[0]), {}).get('market_on') == 0:
            # 即便空仓，也要把还没能卖出的持仓卖掉
            for pos in list(holdings):
                px = prices.get((d_trade, pos['ts_code']))
                if px and px['open'] > get_limit_price(pos['ts_code'], px['pre_close'], 'down'):
                    revenue = pos['shares'] * px['open'] * (1 - SLIPPAGE)
                    capital += (revenue - max(5, revenue*COMMISSION) - revenue*STAMP_DUTY)
                    holdings.remove(pos)
            nav_history.append({'date': d_trade, 'nav': capital})
            continue

        # 3. 买入 (只在 market_on == 1 时买入)
        # 优化筛选：全市场市值排名在 1000-2000 之间的“准小盘股”，避开流动性枯竭的“壳股”
        day_pool = df[(df['trade_date'] == d_signal)].sort_values('circ_mv').iloc[1000:2000].dropna(subset=FEATURE_COLS)
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
                        holdings.append({'ts_code': row['ts_code'], 'shares': shares, 'val_at_buy': buy_px * shares})

        # 计算当日净值
        mv = sum(p['shares'] * prices.get((d_trade, p['ts_code']), {'close': p['val_at_buy']/p['shares']})['close'] for p in holdings)
        nav_history.append({'date': d_trade, 'nav': capital + mv})
        
    return pd.DataFrame(nav_history)

if __name__ == "__main__":
    df = load_data_and_timing('20200101', '20260101')
    df = build_features_v8(df)
    eq_df = run_backtest_v8(df)
    
    total_ret = (eq_df['nav'].iloc[-1] / INITIAL_CAP - 1) * 100
    print(f"\nAdaptive Micro-cap V8 最终收益率: {total_ret:+.2f}%")
    eq_df.to_csv(os.path.join(OUT_DIR, 'super_micro_adaptive_nav.csv'), index=False)
    
    # 绘制曲线
    plt.figure(figsize=(10, 6))
    plt.plot(eq_df['date'], eq_df['nav'], label='V8 Adaptive Micro-cap')
    plt.title('V8 Equity Curve (With Market Timing)')
    plt.grid(True)
    plt.savefig(os.path.join(OUT_DIR, 'micro_adaptive_curve.png'))

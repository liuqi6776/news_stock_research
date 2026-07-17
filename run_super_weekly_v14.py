"""
Super-Weekly V14: 高频 Alpha + 动态风险管理 + 新Feature集成
目标：在周频换仓的基础上，集成中国波指和融资融券数据，提高策略表现。
核心逻辑：
1. 5日周期换仓 (XGBoost 选股)。
2. 每日监控：
   - 止损 (Stop-Loss): -10% (相对于买入价)。
   - 止盈 (Take-Profit): +30% (锁定爆发利润)。
   - 最大持仓时间 (Max Hold): 10个交易日 (强制轮换，避免资金长时间被动)。
3. 严格 T+1 与涨跌停逻辑。
4. 集成中国波指和融资融券数据作为新Feature。
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
VIX_DIR       = r'D:\iquant_data\data_v2\vix_data'
MARGIN_DIR    = r'D:\iquant_data\data_v2\margin_data'
OUT_DIR       = r'C:\Users\liuqi\quant_system_v2'

INITIAL_CAP   = 100_000.0
TOP_N         = 3        
REBAL_FREQ    = 5        # 换仓周期
SLIPPAGE      = 0.001
COMMISSION    = 0.0003
STAMP_DUTY    = 0.0005

# 风险管理参数
STOP_LOSS      = -0.10   # -10% 止损
TAKE_PROFIT    = 0.30    # +30% 止盈
MAX_HOLD_DAYS  = 10      # 最大持仓 10 天

def get_limit_price(code, pre_close, direction='up'):
    ratio = 0.2 if code.startswith(('30', '68')) else 0.1
    if direction == 'up':
        return round(pre_close * (1 + ratio), 2)
    return round(pre_close * (1 - ratio), 2)

def load_vix_data(start, end):
    """加载中国波指数据"""
    files = []
    date_strs = sorted([f.replace('.parquet', '') for f in os.listdir(VIX_DIR) if f.endswith('.parquet')])
    for ds in date_strs:
        if not (start <= ds <= end): continue
        try:
            df = pd.read_parquet(os.path.join(VIX_DIR, f"{ds}.parquet"))
            df['trade_date'] = ds
            files.append(df)
        except:
            continue
    if files:
        vix_df = pd.concat(files, ignore_index=True)
        vix_df['trade_date'] = pd.to_datetime(vix_df['trade_date'])
        vix_df = vix_df.rename(columns={'close': 'vix_close', 'pct_chg': 'vix_pct_chg'})
        return vix_df[['trade_date', 'vix_close', 'vix_pct_chg']]
    return pd.DataFrame(columns=['trade_date', 'vix_close', 'vix_pct_chg'])

def load_margin_data(start, end):
    """加载融资融券数据"""
    files = []
    date_strs = sorted([f.replace('.parquet', '') for f in os.listdir(MARGIN_DIR) if f.endswith('.parquet')])
    for ds in date_strs:
        if not (start <= ds <= end): continue
        try:
            df = pd.read_parquet(os.path.join(MARGIN_DIR, f"{ds}.parquet"))
            df['trade_date'] = ds
            files.append(df)
        except:
            continue
    if files:
        margin_df = pd.concat(files, ignore_index=True)
        margin_df['trade_date'] = pd.to_datetime(margin_df['trade_date'])
        margin_df = margin_df.rename(columns={'rzye': 'margin_rzye', 'rzmre': 'margin_rzmre', 'rqye': 'margin_rqye', 'rzrqye': 'margin_rzrqye'})
        return margin_df[['trade_date', 'margin_rzye', 'margin_rzmre', 'margin_rqye', 'margin_rzrqye']]
    return pd.DataFrame(columns=['trade_date', 'margin_rzye', 'margin_rzmre', 'margin_rqye', 'margin_rzrqye'])

def load_data(start, end):
    files = []
    date_strs = sorted([f.replace('.parquet', '') for f in os.listdir(DATA_DIR) if f.endswith('.parquet')])
    for ds in tqdm(date_strs, desc="加载全维数据"):
        if not (start <= ds <= end): continue
        try:
            p = pd.read_parquet(os.path.join(DATA_DIR, f"{ds}.parquet"), columns=['ts_code','trade_date','open','close','high','low','pre_close'])
            b = pd.read_parquet(os.path.join(BASIC_DIR, f"{ds}.parquet"), columns=['ts_code','pe','pb','circ_mv'])
            c_path = os.path.join(CHIP_DIR, f"{ds}.parquet")
            if os.path.exists(c_path):
                c = pd.read_parquet(c_path, columns=['ts_code','winner_rate','cost_15pct','cost_50pct','cost_85pct'])
            else: continue
            files.append(pd.merge(pd.merge(p, b, on='ts_code'), c, on='ts_code'))
        except: continue
    df = pd.concat(files, ignore_index=True)
    df['trade_date'] = pd.to_datetime(df['trade_date'].astype(str))
    
    # 加载中国波指数据
    vix_df = load_vix_data(start, end)
    if not vix_df.empty:
        df = pd.merge(df, vix_df, on='trade_date', how='left')
    
    # 加载融资融券数据
    margin_df = load_margin_data(start, end)
    if not margin_df.empty:
        df = pd.merge(df, margin_df, on='trade_date', how='left')
    
    return df.sort_values(['ts_code', 'trade_date']).reset_index(drop=True)

def build_features(df):
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
    
    # 新增中国波指相关特征
    if 'vix_close' in df.columns:
        df['vix_rank'] = df.groupby('trade_date')['vix_close'].rank(pct=True)
        df['vix_pct_chg_rank'] = df.groupby('trade_date')['vix_pct_chg'].rank(pct=True)
    
    # 新增融资融券相关特征
    if 'margin_rzye' in df.columns:
        df['margin_rzye_rank'] = df.groupby('trade_date')['margin_rzye'].rank(pct=True)
        df['margin_rzrqye_rank'] = df.groupby('trade_date')['margin_rzrqye'].rank(pct=True)
    
    for col in ['mom_5', 'mom_20', 'ep', 'bp', 'chip_score', 'chip_bottom_heavy']:
        df[f'{col}_rank'] = df.groupby('trade_date')[col].rank(pct=True)
    
    return df

FEATURE_COLS = ['mom_5', 'mom_20', 'bias_5', 'ep', 'bp', 'log_mv', 'chip_score', 'chip_bottom_heavy',
                'mom_5_rank', 'mom_20_rank', 'ep_rank', 'bp_rank']

# 添加新的feature列
FEATURE_COLS.extend(['vix_rank', 'vix_pct_chg_rank', 'margin_rzye_rank', 'margin_rzrqye_rank'])

def train_model(train_df):
    df = train_df.sort_values(['ts_code', 'trade_date'])
    df['label'] = (df.groupby('ts_code')['close'].shift(-5) / df.groupby('ts_code')['close'].shift(-1) - 1 > 0.03).astype(int)
    
    # 过滤掉包含NaN的行
    valid_cols = [col for col in FEATURE_COLS if col in df.columns]
    sub = df.dropna(subset=valid_cols + ['label']).copy()
    
    X = sub[valid_cols].replace([np.inf, -np.inf], 0).fillna(0)
    y = sub['label']
    
    scaler = RobustScaler()
    X_s = scaler.fit_transform(X)
    
    model = xgb.XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.05)
    model.fit(X_s, y)
    
    return model, scaler, valid_cols

def run_backtest_v14(df):
    print("\n" + "!"*50 + "\n  Super-Weekly V14: New Features Integration\n" + "!"*50)
    test_dates = sorted(df[df['trade_date'] >= '2023-01-01']['trade_date'].unique())
    prices = df.set_index(['trade_date', 'ts_code'])[['open', 'close', 'high', 'low', 'pre_close']].to_dict('index')
    capital = INITIAL_CAP
    holdings = [] # {'ts_code', 'shares', 'buy_px', 'days_held'}
    equity = []
    
    cur_model, cur_scaler, valid_cols = None, None, []
    last_month = None
    last_year = None
    
    for i, date in enumerate(tqdm(test_dates[:-1])):
        d_signal = date
        d_trade  = test_dates[i+1]
        
        # 1. 每年更新模型
        year = date.year
        if year != last_year: # Note: last_year should be initialized
            pass # Simplified for speed in this demo, usually monthly re-train.
        
        # 借用每月重训逻辑
        month = date.month
        if month != last_month:
            train_data = df[(df['trade_date'] < date) & (df['trade_date'] >= date - pd.Timedelta(days=365*3))]
            cur_model, cur_scaler, valid_cols = train_model(train_data)
            last_month = month
            last_year = year

        # 2. 每日风控检查 (止盈止损/持仓时间)
        stocks_to_sell = []
        for pos in holdings:
            px_current = prices.get((d_signal, pos['ts_code']))
            if px_current:
                ret = px_current['close'] / pos['buy_px'] - 1
                pos['days_held'] += 1
                if ret < STOP_LOSS:
                    stocks_to_sell.append((pos['ts_code'], 'StopLoss'))
                elif ret > TAKE_PROFIT:
                    stocks_to_sell.append((pos['ts_code'], 'TakeProfit'))
                elif pos['days_held'] >= MAX_HOLD_DAYS:
                    stocks_to_sell.append((pos['ts_code'], 'TimeExit'))
        
        # 3. 换仓检查 (每5天强制换仓)
        is_rebal_day = (i % REBAL_FREQ == 0)
        
        # 执行卖出 (T+1 开盘)
        for pos in list(holdings):
            should_exit = False
            # a) 是否触发了风控
            if pos['ts_code'] in [s[0] for s in stocks_to_sell]:
                should_exit = True
            # b) 是否到了全局换仓日
            elif is_rebal_day:
                should_exit = True
            
            if should_exit:
                px_sell = prices.get((d_trade, pos['ts_code']))
                if px_sell:
                    down_limit = get_limit_price(pos['ts_code'], px_sell['pre_close'], 'down')
                    if px_sell['open'] > down_limit:
                        exit_px = px_sell['open'] * (1 - SLIPPAGE)
                        revenue = pos['shares'] * exit_px
                        capital += (revenue - max(5, revenue*COMMISSION) - revenue*STAMP_DUTY)
                        holdings.remove(pos)

        # 4. 买入 (仅在换仓日，且有空位时)
        if is_rebal_day and len(holdings) < TOP_N and cur_model:
            day_data = df[df['trade_date'] == d_signal].dropna(subset=valid_cols)
            if not day_data.empty:
                X = cur_scaler.transform(day_data[valid_cols].fillna(0))
                day_data['prob'] = cur_model.predict_proba(X)[:, 1]
                # 排除当前已有的
                current_codes = [p['ts_code'] for p in holdings]
                picks = day_data[~day_data['ts_code'].isin(current_codes)].sort_values('prob', ascending=False).head(TOP_N - len(holdings))
                
                if not picks.empty:
                    cash_per = capital / (TOP_N - len(holdings)) if (TOP_N - len(holdings)) > 0 else 0
                    for _, row in picks.iterrows():
                        px_buy = prices.get((d_trade, row['ts_code']))
                        if px_buy:
                            up_limit = get_limit_price(row['ts_code'], px_buy['pre_close'], 'up')
                            if px_buy['open'] < up_limit:
                                buy_px = px_buy['open'] * (1 + SLIPPAGE)
                                # 确保剩余现金足够
                                shares = int(min(cash_per, capital) / buy_px / 100) * 100
                                if shares >= 100:
                                    capital -= (shares * buy_px + max(5, shares*buy_px*COMMISSION))
                                    holdings.append({'ts_code': row['ts_code'], 'shares': shares, 'buy_px': buy_px, 'days_held': 0})

        mv = sum(p['shares'] * prices.get((d_trade, p['ts_code']), {'close': p['buy_px']})['close'] for p in holdings)
        equity.append({'date': d_trade, 'nav': capital + mv})
        
    return pd.DataFrame(equity)

if __name__ == "__main__":
    df = load_data('20200101', '20260101')
    df = build_features(df)
    # Initialize variables avoided in logic but needed
    last_year = None
    eq_df = run_backtest_v14(df)
    
    total_ret = (eq_df['nav'].iloc[-1]/INITIAL_CAP - 1)*100
    print(f"\nSuper-Weekly V14 (New Features) 最终收益: {total_ret:+.2f}%")
    eq_df.to_csv(os.path.join(OUT_DIR, 'super_weekly_v14_equity.csv'), index=False)
    
    plt.figure(figsize=(10,6))
    plt.plot(eq_df['date'], eq_df['nav'], label='V14 New Features')
    plt.title('V14 New Features Weekly Strategy')
    plt.savefig(os.path.join(OUT_DIR, 'weekly_v14_curve.png'))

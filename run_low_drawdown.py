"""
Low Drawdown Strategy - Low Risk Version
Features:
1. Dynamic Position Sizing based on market conditions
2. Stop Loss mechanism
3. Market timing (avoid bear markets)
4. Volatility filter
5. Diversified holdings
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
TOP_N         = 5
COMMISSION    = 0.0003
STAMP_DUTY    = 0.0005
SLIPPAGE      = 0.001
REBAL_FREQ    = 5

MAX_POSITION_PCT = 0.2
STOP_LOSS_PCT    = 0.08
TRAILING_STOP_PCT = 0.05
MAX_DRAWDOWN_LIMIT = 0.15

def get_limit_price(code, pre_close, direction='up'):
    ratio = 0.2 if code.startswith(('30', '68')) else 0.1
    if direction == 'up':
        return round(pre_close * (1 + ratio), 2)
    return round(pre_close * (1 - ratio), 2)

def load_data(start, end):
    files = []
    date_strs = sorted([f.replace('.parquet', '') for f in os.listdir(DATA_DIR) if f.endswith('.parquet')])
    for ds in tqdm(date_strs, desc="Loading data"):
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

def get_index_data(start, end):
    pro = ts.pro_api(TUSHARE_TOKEN)
    idx = pro.index_daily(ts_code='000001.SH', start_date=start.replace('-', ''), end_date=end.replace('-', ''))
    idx['trade_date'] = pd.to_datetime(idx['trade_date'])
    idx = idx.sort_values('trade_date')
    idx['ma20'] = idx['close'].rolling(20).mean()
    idx['ma60'] = idx['close'].rolling(60).mean()
    idx['vol_ma20'] = idx['vol'].rolling(20).mean()
    return idx.set_index('trade_date')

def calc_market_regime(idx_df, date):
    if date not in idx_df.index:
        return 0.5, 0.5
    row = idx_df.loc[date]
    trend_score = 0
    if row['close'] > row['ma20']:
        trend_score += 0.5
    if row['close'] > row['ma60']:
        trend_score += 0.5
    if row['ma20'] > row['ma60']:
        trend_score += 0.5
    vol_ratio = row['vol'] / row['vol_ma20'] if row['vol_ma20'] > 0 else 1
    return min(trend_score / 1.5, 1.0), vol_ratio

def calc_vix_adjustment(idx_df, date, lookback=20):
    if date not in idx_df.index:
        return 1.0
    loc = idx_df.index.get_loc(date)
    if loc < lookback:
        return 1.0
    recent = idx_df.iloc[loc-lookback+1:loc+1]['close']
    returns = recent.pct_change().dropna()
    volatility = returns.std() * np.sqrt(252)
    if volatility > 0.30:
        return 0.3
    elif volatility > 0.25:
        return 0.5
    elif volatility > 0.20:
        return 0.7
    else:
        return 1.0

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
    for col in ['mom_5', 'mom_20', 'ep', 'bp', 'chip_score', 'chip_bottom_heavy']:
        df[f'{col}_rank'] = df.groupby('trade_date')[col].rank(pct=True)
    return df

def add_labels(df, horizon=5):
    df = df.sort_values(['ts_code', 'trade_date'])
    entry = df.groupby('ts_code')['open'].shift(-1)
    exit_ = df.groupby('ts_code')['open'].shift(-1-horizon)
    df['ret'] = (exit_ - entry) / (entry + 1e-8)
    df['label'] = (df['ret'] > 0.02).astype(int)
    return df

FEATURE_COLS = ['mom_5', 'mom_20', 'bias_5', 'ep', 'bp', 'log_mv', 'chip_score', 'chip_bottom_heavy',
                'mom_5_rank', 'mom_20_rank', 'ep_rank', 'bp_rank']

def train_model(train_df):
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

def run_backtest_with_risk_control(df, idx_df):
    print("\n" + "="*60)
    print("  LOW DRAWDOWN STRATEGY - RISK CONTROLLED")
    print("="*60)
    
    test_dates = sorted(df[df['trade_date'] >= '2023-01-01']['trade_date'].unique())
    rebal_dates = test_dates[::REBAL_FREQ]
    
    prices = df.set_index(['trade_date', 'ts_code'])[['open', 'close', 'high', 'low', 'pre_close']].to_dict('index')
    capital = INITIAL_CAP
    holdings = []
    equity = []
    max_nav = INITIAL_CAP
    daily_stop_triggered = False
    
    cur_model, cur_scaler = None, None
    last_month = None
    
    for i, date in enumerate(tqdm(rebal_dates[:-1])):
        d_signal = date
        d_trade  = test_dates[test_dates.index(date) + 1]
        
        month = date.month
        if month != last_month:
            train_data = df[(df['trade_date'] < date) & (df['trade_date'] >= date - pd.Timedelta(days=365*3))]
            cur_model, cur_scaler = train_model(train_data)
            last_month = month
        
        trend_score, vol_ratio = calc_market_regime(idx_df, d_signal)
        vix_adj = calc_vix_adjustment(idx_df, d_signal)
        position_pct = min(trend_score * vix_adj, 1.0)
        
        if position_pct < 0.3:
            print(f"\n{d_signal.strftime('%Y-%m-%d')}: Market risk high, position reduced to {position_pct:.0%}")
        
        for pos in list(holdings):
            key_sell = (d_trade, pos['ts_code'])
            if key_sell in prices:
                px_sell = prices[key_sell]
                down_limit = get_limit_price(pos['ts_code'], px_sell['pre_close'], 'down')
                if px_sell['open'] <= down_limit:
                    continue
                
                current_price = px_sell['open']
                pnl_pct = (current_price - pos['buy_px']) / pos['buy_px']
                
                if pnl_pct < -STOP_LOSS_PCT:
                    print(f"  STOP LOSS: {pos['ts_code']} at {pnl_pct:.1%}")
                elif pos.get('high_px') and (pos['high_px'] - current_price) / pos['high_px'] > TRAILING_STOP_PCT:
                    print(f"  TRAILING STOP: {pos['ts_code']}")
                else:
                    if pnl_pct < -STOP_LOSS_PCT or (pos.get('high_px') and (pos['high_px'] - current_price) / pos['high_px'] > TRAILING_STOP_PCT):
                        pass
                
                exit_px = px_sell['open'] * (1 - SLIPPAGE)
                revenue = pos['shares'] * exit_px
                capital += (revenue - max(5, revenue*COMMISSION) - revenue*STAMP_DUTY)
                holdings.remove(pos)
        
        current_nav = capital
        for p in holdings:
            key = (d_trade, p['ts_code'])
            if key in prices:
                current_nav += p['shares'] * prices[key]['close']
        
        if (max_nav - current_nav) / max_nav > MAX_DRAWDOWN_LIMIT and not daily_stop_triggered:
            print(f"\n  MAX DRAWDOWN HIT: {(max_nav - current_nav) / max_nav:.1%} > {MAX_DRAWDOWN_LIMIT:.0%}")
            print("  Clearing all positions...")
            for pos in list(holdings):
                key_sell = (d_trade, pos['ts_code'])
                if key_sell in prices:
                    px_sell = prices[key_sell]
                    exit_px = px_sell['open'] * (1 - SLIPPAGE)
                    revenue = pos['shares'] * exit_px
                    capital += (revenue - max(5, revenue*COMMISSION) - revenue*STAMP_DUTY)
            holdings = []
            daily_stop_triggered = True
        
        if current_nav > max_nav:
            max_nav = current_nav
            daily_stop_triggered = False
        
        if position_pct >= 0.3 and cur_model:
            day_data = df[df['trade_date'] == d_signal].dropna(subset=FEATURE_COLS)
            X = cur_scaler.transform(day_data[FEATURE_COLS].fillna(0))
            day_data['prob'] = cur_model.predict_proba(X)[:, 1]
            
            picks = day_data.sort_values('prob', ascending=False).head(TOP_N * 2)
            
            available_cash = capital * position_pct
            cash_per_stock = available_cash / TOP_N
            
            bought = 0
            for _, row in picks.iterrows():
                if bought >= TOP_N:
                    break
                key_buy = (d_trade, row['ts_code'])
                if key_buy in prices:
                    px_buy = prices[key_buy]
                    up_limit = get_limit_price(row['ts_code'], px_buy['pre_close'], 'up')
                    if px_buy['open'] >= up_limit:
                        continue
                    buy_px = px_buy['open'] * (1 + SLIPPAGE)
                    shares = int(cash_per_stock / buy_px / 100) * 100
                    if shares >= 100:
                        cost = shares * buy_px + max(5, shares*buy_px*COMMISSION)
                        if cost <= capital:
                            capital -= cost
                            holdings.append({
                                'ts_code': row['ts_code'], 
                                'shares': shares, 
                                'buy_px': buy_px,
                                'high_px': buy_px
                            })
                            bought += 1
        
        mv = 0
        for p in holdings:
            key = (d_trade, p['ts_code'])
            if key in prices:
                current_px = prices[key]['close']
                mv += p['shares'] * current_px
                if current_px > p.get('high_px', p['buy_px']):
                    p['high_px'] = current_px
            else:
                mv += p['shares'] * p['buy_px']
        
        nav = capital + mv
        equity.append({'date': d_trade, 'nav': nav, 'position_pct': position_pct})
        max_nav = max(max_nav, nav)
    
    return pd.DataFrame(equity)

if __name__ == "__main__":
    print("Loading stock data...")
    df = load_data('20200101', '20260101')
    print("Loading index data...")
    idx_df = get_index_data('2020-01-01', '2026-01-01')
    
    print("Building features...")
    df = build_features(df)
    df = add_labels(df, horizon=5)
    
    print("Running backtest with risk control...")
    eq_df = run_backtest_with_risk_control(df, idx_df)
    
    eq_df['cummax'] = eq_df['nav'].cummax()
    eq_df['drawdown'] = (eq_df['nav'] - eq_df['cummax']) / eq_df['cummax'] * 100
    
    final_ret = (eq_df['nav'].iloc[-1] / INITIAL_CAP - 1) * 100
    max_dd = eq_df['drawdown'].min()
    
    print(f"\n{'='*60}")
    print(f"  LOW DRAWDOWN STRATEGY RESULTS")
    print(f"{'='*60}")
    print(f"  Final NAV:     {eq_df['nav'].iloc[-1]:,.0f}")
    print(f"  Total Return:  {final_ret:+.1f}%")
    print(f"  Max Drawdown:  {max_dd:.1f}%")
    print(f"{'='*60}")
    
    eq_df.to_csv(os.path.join(OUT_DIR, 'low_drawdown_equity.csv'), index=False)
    
    plt.figure(figsize=(14, 6))
    plt.subplot(1, 2, 1)
    plt.plot(eq_df['date'], eq_df['nav'], label='NAV', color='#2E86AB')
    plt.axhline(y=INITIAL_CAP, color='gray', linestyle='--', alpha=0.5)
    plt.title('Low Drawdown Strategy - NAV')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.fill_between(eq_df['date'], eq_df['drawdown'], 0, alpha=0.5, color='#E94F37')
    plt.title('Drawdown (%)')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'low_drawdown_curve.png'), dpi=150)
    print(f"\nChart saved: low_drawdown_curve.png")

import os
import sys
import time
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
VIX_DIR       = r'D:\iquant_data\data_v2\vix1'
MARGIN_DIR    = r'D:\iquant_data\data_v2\margin1'
OUT_DIR       = r'C:\Users\liuqi\quant_system_v2'

INITIAL_CAP   = 100_000.0
TOP_N         = 5
MAX_HOLDINGS  = 5
COMMISSION    = 0.0003
STAMP_DUTY    = 0.0005
SLIPPAGE      = 0.001
REBAL_FREQ    = 5

TRAILING_ACTIVATE = 0.15
TRAILING_STOP     = 0.08
HARD_STOP_LOSS    = -0.15
MAX_HOLD_DAYS     = 10

def get_limit_price(code, pre_close, direction='up'):
    ratio = 0.2 if code.startswith(('30', '68')) else 0.1
    if direction == 'up':
        return round(pre_close * (1 + ratio), 2)
    return round(pre_close * (1 - ratio), 2)

def load_vix_data(start, end):
    files = []
    date_strs = sorted([f.replace('.parquet', '') for f in os.listdir(VIX_DIR) if f.endswith('.parquet')])
    for ds in date_strs:
        if not (start <= ds <= end): continue
        try:
            df = pd.read_parquet(os.path.join(VIX_DIR, f"{ds}.parquet"))
            df['trade_date'] = ds
            files.append(df)
        except: continue
    if files:
        vix_df = pd.concat(files, ignore_index=True)
        vix_df['trade_date'] = pd.to_datetime(vix_df['trade_date'])
        vix_df = vix_df.rename(columns={'close': 'vix_close', 'pct_chg': 'vix_pct_chg'})
        return vix_df[['trade_date', 'vix_close', 'vix_pct_chg']]
    return pd.DataFrame(columns=['trade_date', 'vix_close', 'vix_pct_chg'])

def load_margin_data(start, end):
    files = []
    date_strs = sorted([f.replace('.parquet', '') for f in os.listdir(MARGIN_DIR) if f.endswith('.parquet')])
    for ds in date_strs:
        if not (start <= ds <= end): continue
        try:
            df = pd.read_parquet(os.path.join(MARGIN_DIR, f"{ds}.parquet"))
            df['trade_date'] = ds
            files.append(df)
        except: continue
    if files:
        margin_df = pd.concat(files, ignore_index=True)
        margin_df['trade_date'] = pd.to_datetime(margin_df['trade_date'])
        margin_df = margin_df.rename(columns={'rzye': 'margin_rzye', 'rzmre': 'margin_rzmre'})
        return margin_df[['trade_date', 'margin_rzye', 'margin_rzmre']]
    return pd.DataFrame(columns=['trade_date', 'margin_rzye', 'margin_rzmre'])

def load_data(start, end):
    print("V18: loading data...")
    files = []
    date_strs = sorted([f.replace('.parquet', '') for f in os.listdir(DATA_DIR) if f.endswith('.parquet')])
    for ds in tqdm(date_strs, desc="Loading Parquet"):
        if not (start <= ds <= end): continue
        try:
            p = pd.read_parquet(os.path.join(DATA_DIR, f"{ds}.parquet"),
                                columns=['ts_code','trade_date','open','close','high','low','pre_close','vol'])
            b = pd.read_parquet(os.path.join(BASIC_DIR, f"{ds}.parquet"),
                                columns=['ts_code','pe','pb','circ_mv'])
            chip_path = os.path.join(CHIP_DIR, f"{ds}.parquet")
            if os.path.exists(chip_path):
                c = pd.read_parquet(chip_path,
                                    columns=['ts_code','winner_rate','cost_15pct','cost_50pct','cost_85pct'])
            else:
                c = pd.DataFrame(columns=['ts_code','winner_rate','cost_15pct','cost_50pct','cost_85pct'])
            files.append(pd.merge(pd.merge(p, b, on='ts_code'), c, on='ts_code', how='left'))
        except: continue
    df = pd.concat(files, ignore_index=True)
    df['trade_date'] = pd.to_datetime(df['trade_date'].astype(str))
    vix_df = load_vix_data(start, end)
    if not vix_df.empty:
        df = pd.merge(df, vix_df, on='trade_date', how='left')
    margin_df = load_margin_data(start, end)
    if not margin_df.empty:
        df = pd.merge(df, margin_df, on='trade_date', how='left')
    df = df.drop_duplicates(subset=['ts_code', 'trade_date'])
    return df.sort_values(['ts_code', 'trade_date']).reset_index(drop=True)

def build_features(df):
    df = df.copy().sort_values(['ts_code', 'trade_date'])
    g = df.groupby('ts_code')['close']
    # V7 original features
    for w in [5, 20]:
        df[f'mom_{w}'] = g.transform(lambda x: x / x.shift(w) - 1)
        df[f'bias_{w}'] = (df['close'] - g.transform(lambda x: x.rolling(w).mean())) / (g.transform(lambda x: x.rolling(w).mean()) + 1e-8)
    df['ep'] = 1.0 / (df['pe'] + 1e-8)
    df['bp'] = 1.0 / (df['pb'] + 1e-8)
    df['log_mv'] = np.log(df['circ_mv'] + 1)
    df['chip_score'] = df['winner_rate'] * (df['close'] > df['cost_50pct']).astype(int)
    df['chip_bottom_heavy'] = (df['cost_85pct'] - df['cost_50pct']) / (df['cost_50pct'] - df['cost_15pct'] + 1e-8)
    # V18 new features
    df['mom_10'] = g.transform(lambda x: x / x.shift(10) - 1)
    df['mom_60'] = g.transform(lambda x: x / x.shift(60) - 1)
    df['vol_20'] = g.pct_change().transform(lambda x: x.rolling(20).std())
    g_vol = df.groupby('ts_code')['vol']
    df['vol_ratio_5'] = df['vol'] / (g_vol.transform(lambda x: x.rolling(5).mean()) + 1e-8)
    df['high_20d'] = df.groupby('ts_code')['high'].transform(lambda x: x.rolling(20).max())
    df['breakout_20d'] = (df['close'] >= df['high_20d']).astype(int)
    delta = df.groupby('ts_code')['close'].transform(lambda x: x.diff())
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta).clip(lower=0).rolling(14).mean()
    rs = gain / (loss + 1e-8)
    df['rsi_14'] = 100 - (100 / (1 + rs))
    df['price_pos_20d'] = (df['close'] - df.groupby('ts_code')['low'].transform(lambda x: x.rolling(20).min())) / \
                          (df.groupby('ts_code')['high'].transform(lambda x: x.rolling(20).max()) - \
                           df.groupby('ts_code')['low'].transform(lambda x: x.rolling(20).min()) + 1e-8)
    if 'vix_close' in df.columns:
        df['vix_rank'] = df.groupby('trade_date')['vix_close'].rank(pct=True)
    if 'margin_rzye' in df.columns:
        df['margin_rank'] = df.groupby('trade_date')['margin_rzye'].rank(pct=True)
    # Cross-sectional ranking
    rank_cols = ['mom_5', 'mom_10', 'mom_20', 'mom_60', 'bias_5', 'bias_20',
                 'ep', 'bp', 'chip_score', 'chip_bottom_heavy',
                 'vol_20', 'vol_ratio_5', 'breakout_20d', 'rsi_14', 'price_pos_20d']
    for col in rank_cols:
        if col in df.columns:
            df[f'{col}_rank'] = df.groupby('trade_date')[col].rank(pct=True)
    if 'vix_rank' in df.columns:
        rank_cols.append('vix_rank')
    if 'margin_rank' in df.columns:
        rank_cols.append('margin_rank')
    print(f"V18: feature count = {len(rank_cols)} raw + ranking features")
    return df

def add_labels(df, horizon=5):
    df = df.sort_values(['ts_code', 'trade_date'])
    entry = df.groupby('ts_code')['open'].shift(-1)
    exit_ = df.groupby('ts_code')['open'].shift(-1-horizon)
    df['ret'] = (exit_ - entry) / (entry + 1e-8)
    df['label'] = (df['ret'] > 0.02).astype(int)
    label_counts = df['label'].value_counts()
    print(f"  Label dist: bull(>2%)={label_counts.get(1,0)}, bear/flat={label_counts.get(0,0)}")
    return df

FEATURE_COLS = [
    'mom_5', 'mom_20', 'bias_5', 'ep', 'bp', 'log_mv', 'chip_score', 'chip_bottom_heavy',
    'mom_5_rank', 'mom_20_rank', 'ep_rank', 'bp_rank',
    'mom_10', 'mom_60', 'bias_20',
    'vol_20', 'vol_ratio_5', 'breakout_20d', 'rsi_14', 'price_pos_20d',
    'mom_10_rank', 'mom_60_rank', 'bias_20_rank',
    'vol_20_rank', 'vol_ratio_5_rank', 'breakout_20d_rank', 'rsi_14_rank', 'price_pos_20d_rank',
]
OPTIONAL_FEATURE_COLS = ['vix_rank', 'margin_rank']

def is_bull_market(date, idx_daily):
    if idx_daily is None:
        return True
    past = idx_daily[idx_daily['trade_date'] <= date].tail(60)
    if len(past) < 60:
        return True
    ma20 = past['close'].iloc[-20:].mean()
    ma60 = past['close'].mean()
    return ma20 > ma60

def train_model(train_df, valid_cols):
    sub = train_df.dropna(subset=valid_cols + ['label']).copy()
    X = sub[valid_cols].replace([np.inf, -np.inf], 0).fillna(0)
    y = sub['label']
    pos = sub[y == 1].sample(min(len(sub[y==1]), 10000), random_state=42)
    neg = sub[y == 0].sample(min(len(sub[y==0]), 10000), random_state=42)
    bal = pd.concat([pos, neg])
    scaler = RobustScaler()
    X_bal = scaler.fit_transform(bal[valid_cols])
    model = xgb.XGBClassifier(n_estimators=150, max_depth=5, learning_rate=0.05,
                               subsample=0.8, eval_metric='logloss', random_state=42,
                               n_jobs=1)
    model.fit(X_bal, bal['label'], verbose=False)
    del bal, X_bal, sub; 
    import gc; gc.collect()
    return model, scaler

def run_backtest(df, idx_daily=None):
    print("\n" + "!"*60)
    print("  Super-Weekly V18: V7-Based Smart Enhancement")
    print("!"*60)
    test_dates = sorted(df[df['trade_date'] >= '2023-01-01']['trade_date'].unique())
    rebal_dates = test_dates[::REBAL_FREQ]
    prices = df.set_index(['trade_date', 'ts_code'])[['open', 'close', 'pre_close']].to_dict('index')
    valid_cols = [c for c in FEATURE_COLS + OPTIONAL_FEATURE_COLS if c in df.columns]
    print(f"  Valid features: {len(valid_cols)}")
    import time as _time
    start_time = _time.time()
    progress_log = os.path.join(OUT_DIR, 'v18_progress.log')
    with open(progress_log, 'w', encoding='utf-8') as pf:
        pf.write(f"V18 Backtest Started: {_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        pf.write(f"Total rebalance dates: {len(rebal_dates)}\n\n")
    capital = INITIAL_CAP
    holdings = []
    equity = []
    cur_model, cur_scaler = None, None
    last_month = None
    stats = {'total_trades': 0, 'trailing': 0, 'time_exit': 0, 'hard_stop': 0, 'normal': 0}
    total_rebals = len(rebal_dates[:-1])
    progress_log = os.path.join(OUT_DIR, 'v18_progress.log')
    for i, date in enumerate(tqdm(rebal_dates[:-1])):
        d_signal = date
        idx_in_test = test_dates.index(date)
        if idx_in_test + 1 >= len(test_dates):
            break
        d_trade = test_dates[idx_in_test + 1]
        month = date.month
        # === real-time progress to file ===
        mv_now = sum(p['shares'] * prices.get((d_signal, p['ts_code']), {'close': p['buy_px']})['close'] for p in holdings)
        nav_now = capital + mv_now
        pct = (i + 1) / total_rebals * 100
        elapsed = time.time() - start_time
        with open(progress_log, 'a', encoding='utf-8') as pf:
            pf.write(f"[{time.strftime('%H:%M:%S')}] {i+1}/{total_rebals} ({pct:.1f}%) | {d_signal.strftime('%Y-%m-%d')} | NAV={nav_now:.0f} | holdings={len(holdings)} | elapsed={elapsed:.0f}s\n")
        if month != last_month:
            train_data = df[(df['trade_date'] < date) & (df['trade_date'] >= date - pd.Timedelta(days=365*2))]
            if len(train_data) > 500:
                cur_model, cur_scaler = train_model(train_data, valid_cols)
                print(f"  {date.strftime('%Y-%m')}: model trained (2yr window, {len(train_data)} rows)")
                del train_data; import gc; gc.collect()
            last_month = month
        bull = is_bull_market(d_signal, idx_daily)
        max_hold = MAX_HOLD_DAYS if not bull else 12
        for pos in list(holdings):
            should_exit = True
            exit_reason = 'normal'
            px_now = prices.get((d_signal, pos['ts_code']))
            if px_now:
                ret = px_now['close'] / pos['buy_px'] - 1
                pos['days_held'] = pos.get('days_held', 0) + 1
                pos['max_ret'] = max(pos.get('max_ret', 0), ret)
                if pos['max_ret'] > TRAILING_ACTIVATE and (pos['max_ret'] - ret) > TRAILING_STOP:
                    exit_reason = 'trailing'; stats['trailing'] += 1
                elif ret < HARD_STOP_LOSS:
                    exit_reason = 'hard_stop'; stats['hard_stop'] += 1
                elif pos['days_held'] >= max_hold:
                    exit_reason = 'time_exit'; stats['time_exit'] += 1
                elif bull and ret > 0.05:
                    should_exit = False
                    exit_reason = 'hold_bull'
                else:
                    stats['normal'] += 1
            if should_exit:
                px_sell = prices.get((d_trade, pos['ts_code']))
                if px_sell:
                    down_limit = get_limit_price(pos['ts_code'], px_sell['pre_close'], 'down')
                    if px_sell['open'] <= down_limit:
                        continue
                    exit_px = px_sell['open'] * (1 - SLIPPAGE)
                    revenue = pos['shares'] * exit_px
                    capital += (revenue - max(5, revenue*COMMISSION) - revenue*STAMP_DUTY)
                    holdings.remove(pos)
        day_data = df[df['trade_date'] == d_signal].dropna(subset=valid_cols)
        if cur_model and not day_data.empty:
            X = cur_scaler.transform(day_data[valid_cols].fillna(0))
            day_data['prob'] = cur_model.predict_proba(X)[:, 1]
            current_codes = [p['ts_code'] for p in holdings]
            candidates = day_data[~day_data['ts_code'].isin(current_codes)]
            target_holdings = MAX_HOLDINGS if bull else 3
            n_buy = target_holdings - len(holdings)
            if n_buy > 0:
                picks = candidates.sort_values('prob', ascending=False).head(n_buy)
                if not picks.empty:
                    total_prob = picks['prob'].sum()
                    if total_prob < 1e-8: total_prob = 1.0
                    weights = picks['prob'] / total_prob
                    for (_, row), weight in zip(picks.iterrows(), weights):
                        px_buy = prices.get((d_trade, row['ts_code']))
                        if px_buy:
                            up_limit = get_limit_price(row['ts_code'], px_buy['pre_close'], 'up')
                            if px_buy['open'] >= up_limit:
                                continue
                            buy_px = px_buy['open'] * (1 + SLIPPAGE)
                            w = max(weight, 0.2)
                            w = w / sum(max(p, 0.2) for p in weights)
                            cash_for_this = capital * w
                            shares = int(cash_for_this / buy_px / 100) * 100
                            if shares >= 100:
                                capital -= (shares * buy_px + max(5, shares*buy_px*COMMISSION))
                                holdings.append({
                                    'ts_code': row['ts_code'], 'shares': shares, 'buy_px': buy_px,
                                    'days_held': 0, 'prob': row['prob'], 'max_ret': 0,
                                })
                                stats['total_trades'] += 1
        mv = sum(p['shares'] * prices.get((d_trade, p['ts_code']), {'close': p['buy_px']})['close'] for p in holdings)
        equity.append({'date': d_trade, 'nav': capital + mv})
    print(f"\n  Trade stats: buys={stats['total_trades']}, trailing={stats['trailing']}, hard_stop={stats['hard_stop']}, time_exit={stats['time_exit']}, normal={stats['normal']}")
    with open(progress_log, 'a', encoding='utf-8') as pf:
        pf.write(f"\n=== V18 DONE in {time.time() - start_time:.0f}s ===\n")
        pf.write(f"Total trades: {stats['total_trades']}, trailing={stats['trailing']}, hard_stop={stats['hard_stop']}, time_exit={stats['time_exit']}, normal={stats['normal']}\n")
    return pd.DataFrame(equity)

if __name__ == "__main__":
    import tushare as ts
    pro = ts.pro_api(TUSHARE_TOKEN)
    idx_daily = pro.index_daily(ts_code='000852.SH', start_date='20200101', end_date='20260101')
    idx_daily['trade_date'] = pd.to_datetime(idx_daily['trade_date'])
    idx_daily = idx_daily.sort_values('trade_date').reset_index(drop=True)
    print(f"  Index data: {len(idx_daily)} rows")
    df = load_data('20180101', '20260101')
    df = build_features(df)
    df = add_labels(df, horizon=5)
    eq_df = run_backtest(df, idx_daily)
    total_ret = (eq_df['nav'].iloc[-1] / INITIAL_CAP - 1) * 100
    days = (eq_df['date'].iloc[-1] - eq_df['date'].iloc[0]).days
    annual_ret = ((eq_df['nav'].iloc[-1] / INITIAL_CAP) ** (365/days) - 1) * 100
    peak = eq_df['nav'].cummax()
    drawdown = (eq_df['nav'] - peak) / peak
    max_dd = drawdown.min() * 100
    daily_ret = eq_df['nav'].pct_change().dropna()
    sharpe = (daily_ret.mean() / daily_ret.std() * np.sqrt(252)) if daily_ret.std() > 0 else 0
    calmar = annual_ret / abs(max_dd) if max_dd != 0 else 0
    win_rate = (daily_ret > 0).sum() / len(daily_ret) * 100 if len(daily_ret) > 0 else 0
    print(f"\n{'='*60}")
    print(f"  V18 Performance")
    print(f"{'='*60}")
    print(f"  Total Return: {total_ret:+.2f}%")
    print(f"  Annual Return: {annual_ret:+.2f}%")
    print(f"  Max Drawdown: {max_dd:.2f}%")
    print(f"  Sharpe: {sharpe:.2f}")
    print(f"  Calmar: {calmar:.2f}")
    print(f"  Win Rate: {win_rate:.1f}%")
    print(f"  Days: {days}")
    print(f"{'='*60}")
    eq_df['year'] = eq_df['date'].dt.year
    yearly = eq_df.groupby('year').apply(lambda x: (x['nav'].iloc[-1] / x['nav'].iloc[0] - 1) * 100)
    print(f"\n  Yearly Returns:")
    for y, ret in yearly.items():
        print(f"    {y}: {ret:+.1f}%")
    eq_df[['date', 'nav']].to_csv(os.path.join(OUT_DIR, 'super_weekly_v18_equity.csv'), index=False)
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    ax1 = axes[0, 0]
    ax1.plot(eq_df['date'], eq_df['nav'], linewidth=1.5, color='#FF6B35', label='V18 (V7-Based)')
    ax1.set_title('V18 Equity Curve', fontsize=13, fontweight='bold')
    ax1.set_ylabel('NAV'); ax1.legend(fontsize=11); ax1.grid(True, alpha=0.3)
    ax2 = axes[0, 1]
    ax2.fill_between(eq_df['date'], drawdown * 100, 0, alpha=0.4, color='red')
    ax2.set_title('Drawdown (%)', fontsize=12); ax2.set_ylabel('Drawdown %'); ax2.grid(True, alpha=0.3)
    ax3 = axes[1, 0]
    bull_dates = idx_daily[(idx_daily['trade_date'] >= '2023-01-01') & (idx_daily['trade_date'] <= '2025-12-31')]
    bull_ma20 = bull_dates['close'].rolling(20).mean()
    bull_ma60 = bull_dates['close'].rolling(60).mean()
    is_bull = bull_ma20 > bull_ma60
    for j in range(len(bull_dates)-1):
        color = '#4CAF50' if is_bull.iloc[j] else '#F44336'
        ax3.axvspan(bull_dates.iloc[j]['trade_date'], bull_dates.iloc[j+1]['trade_date'], alpha=0.15, color=color)
    ax3.plot(eq_df['date'], eq_df['nav'], linewidth=1.5, color='black')
    ax3.set_title('NAV (Green=Bull MA20>MA60, Red=Bear)', fontsize=10); ax3.set_ylabel('NAV'); ax3.grid(True, alpha=0.3)
    ax4 = axes[1, 1]
    colors = ['#4CAF50' if v > 0 else '#F44336' for v in yearly]
    ax4.bar(yearly.index, yearly.values, color=colors, alpha=0.8)
    ax4.set_title('Yearly Returns (%)', fontsize=12); ax4.set_ylabel('Return %')
    ax4.axhline(y=0, color='black', linewidth=0.5)
    for y, v in yearly.items():
        ax4.text(y, v + (2 if v > 0 else -5), f'{v:.1f}%', ha='center', fontsize=11, fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'weekly_v18_curve.png'), dpi=150)
    print(f"\n  Saved: weekly_v18_curve.png, super_weekly_v18_equity.csv")

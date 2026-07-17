"""
Super-Weekly Active Equity Strategy - Point-in-Time Generation Pipeline (V2.0)
=============================================================================
This script provides a production-grade, strictly look-ahead-free (Point-in-Time)
signal generation pipeline and historical backtester for the Super-Weekly strategy.

Features:
- Generates buy/sell signals for any specific date (default: latest trading date).
- Executes historical walk-forward backtests with monthly rolling XGBoost training.
- Fully supports periods before 2020 by using a robust set of essential features
  and filling missing chip data (cyq1) with 0s.
- Optimized for high performance (XGBoost tree_method='hist', n_jobs=-1,
  and training downsampling to weekly frequency).

Usage:
  1. Generate signals for the latest date:
     python generate_super_weekly_signals_pit.py --mode generate

  2. Generate signals for a specific historical date:
     python generate_super_weekly_signals_pit.py --mode generate --date 2026-03-11

  3. Run historical walk-forward backtest (2018 to 2026):
     python generate_super_weekly_signals_pit.py --mode backtest --start 2018-01-01
"""

import os
import sys
import argparse
import warnings
import json
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import RobustScaler
from tqdm import tqdm

warnings.filterwarnings('ignore')

# Configuration Paths
DATA_DIR      = r'D:\iquant_data\data_v2\data_day1'
BASIC_DIR     = r'D:\iquant_data\data_v2\other_day1'
CHIP_DIR      = r'D:\iquant_data\data_v2\cyq1'
OUT_DIR       = r'c:\Users\liuqi\quant_system_v2'
SIGNALS_DIR   = os.path.join(OUT_DIR, 'signals')
os.makedirs(SIGNALS_DIR, exist_ok=True)

# Strategy Parameters
TOP_N         = 3        
COMMISSION    = 0.0003
STAMP_DUTY    = 0.0005
SLIPPAGE      = 0.001
REBAL_FREQ    = 5

# Feature definitions
FEATURE_COLS = [
    'mom_5', 'mom_20', 'bias_5', 'ep', 'bp', 'log_mv',
    'chip_score', 'chip_bottom_heavy',
    'mom_5_rank', 'mom_20_rank', 'ep_rank', 'bp_rank'
]
ESSENTIAL_COLS = ['mom_5', 'mom_20', 'bias_5', 'ep', 'bp', 'log_mv']

def get_limit_price(code, pre_close, direction='up'):
    """Calculate the 10% (main board) or 20% (ChiNext/Star) limit price."""
    ratio = 0.2 if code.startswith(('30', '68')) else 0.1
    if direction == 'up':
        return round(pre_close * (1 + ratio), 2)
    return round(pre_close * (1 - ratio), 2)

def load_pit_data(start_date, end_date):
    """Load price, fundamental, and chip data strictly within the point-in-time window."""
    print(f"Loading raw data from {start_date} to {end_date}...")
    files = []
    
    # List and filter available database parquet files
    all_files = sorted([f.replace('.parquet', '') for f in os.listdir(DATA_DIR) if f.endswith('.parquet')])
    filtered_dates = [d for d in all_files if start_date <= d <= end_date]
    
    for ds in tqdm(filtered_dates, desc="Loading parquet data"):
        try:
            # Price data
            p = pd.read_parquet(os.path.join(DATA_DIR, f"{ds}.parquet"), 
                                columns=['ts_code', 'trade_date', 'open', 'close', 'high', 'low', 'pre_close', 'vol'])
            # Fundamentals data
            b = pd.read_parquet(os.path.join(BASIC_DIR, f"{ds}.parquet"), 
                                columns=['ts_code', 'pe', 'pb', 'circ_mv'])
            # Chip data (starts in 2020)
            chip_path = os.path.join(CHIP_DIR, f"{ds}.parquet")
            if os.path.exists(chip_path):
                c = pd.read_parquet(chip_path, columns=['ts_code', 'winner_rate', 'cost_15pct', 'cost_50pct', 'cost_85pct'])
            else:
                c = pd.DataFrame(columns=['ts_code', 'winner_rate', 'cost_15pct', 'cost_50pct', 'cost_85pct'])
            
            # Merge datasets
            m1 = pd.merge(p, b, on='ts_code')
            m2 = pd.merge(m1, c, on='ts_code', how='left')
            files.append(m2)
        except Exception:
            continue
            
    if not files:
        raise ValueError("No data found in the specified date range.")
        
    df = pd.concat(files, ignore_index=True)
    df['trade_date'] = pd.to_datetime(df['trade_date'].astype(str))
    # Deduplicate in case of data overlap
    df = df.drop_duplicates(['trade_date', 'ts_code'], keep='last')
    return df.sort_values(['ts_code', 'trade_date']).reset_index(drop=True)

def build_features(df):
    """Build momentum, valuation, size, chip indicators, and cross-sectional ranks."""
    df = df.copy().sort_values(['ts_code', 'trade_date'])
    g = df.groupby('ts_code')['close']
    
    # Technical momentum & bias indicators
    for w in [5, 20]:
        df[f'mom_{w}'] = g.transform(lambda x: x / x.shift(w) - 1)
        ma = g.transform(lambda x: x.rolling(w).mean())
        df[f'bias_{w}'] = (df['close'] - ma) / (ma + 1e-8)
    
    # Valuation and Size indicators
    df['ep'] = 1.0 / (df['pe'] + 1e-8)
    df['bp'] = 1.0 / (df['pb'] + 1e-8)
    df['log_mv'] = np.log(df['circ_mv'] + 1)
    
    # Chip indicators (fill NaNs with 0 before scaling to support pre-2020)
    df['chip_score'] = df['winner_rate'] * (df['close'] > df['cost_50pct']).astype(int)
    df['chip_bottom_heavy'] = (df['cost_85pct'] - df['cost_50pct']) / (df['cost_50pct'] - df['cost_15pct'] + 1e-8)
    
    # Cross-sectional ranks by trade date
    for col in ['mom_5', 'mom_20', 'ep', 'bp', 'chip_score', 'chip_bottom_heavy']:
        df[f'{col}_rank'] = df.groupby('trade_date')[col].rank(pct=True)
        
    return df

def add_labels(df, horizon=5):
    """Label samples based on future weekly return (T+1 Open to T+6 Open)."""
    df = df.sort_values(['ts_code', 'trade_date'])
    entry = df.groupby('ts_code')['open'].shift(-1)
    exit_ = df.groupby('ts_code')['open'].shift(-1-horizon)
    df['ret'] = (exit_ - entry) / (entry + 1e-8)
    df['label'] = (df['ret'] > 0.02).astype(int)
    return df

def train_super_model(train_df):
    """Train XGBoost model using RobustScaler, optimized for speed & pre-2020 compatibility."""
    sub = train_df.dropna(subset=ESSENTIAL_COLS + ['label']).copy()
    
    # Replace infinite values and fill NaNs (like pre-2020 missing chip features) with 0
    sub[FEATURE_COLS] = sub[FEATURE_COLS].replace([np.inf, -np.inf], np.nan).fillna(0)
    
    y = sub['label']
    pos = sub[y == 1]
    neg = sub[y == 0].sample(min(len(pos)*2, len(sub)-len(pos)), random_state=42)
    bal = pd.concat([pos, neg])
    
    scaler = RobustScaler()
    X_bal = scaler.fit_transform(bal[FEATURE_COLS])
    
    # Optimized XGBoost Classifier parameters
    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        eval_metric='logloss',
        tree_method='hist',
        n_jobs=-1,
        random_state=42
    )
    model.fit(X_bal, bal['label'])
    return model, scaler

def run_daily_super_backtest(df, start_date='2018-01-01'):
    """Run historical daily backtest with point-in-time rolling walk-forward training."""
    print(f"\nRunning walk-forward backtest from {start_date}...")
    test_dates = sorted(df[df['trade_date'] >= pd.Timestamp(start_date)]['trade_date'].unique())
    all_dates = sorted(df['trade_date'].unique())
    
    rebal_dates = set(test_dates[::REBAL_FREQ])
    prices = df.set_index(['trade_date', 'ts_code'])[['open', 'close', 'pre_close']].to_dict('index')
    
    capital = 100_000.0
    holdings = []
    equity = []
    
    cur_model, cur_scaler = None, None
    last_month = None
    
    for idx, date in enumerate(tqdm(test_dates, desc="Backtesting days")):
        # 1. Rebalance Execution (at Open)
        if date in rebal_dates:
            # A. Sell all active holdings at Open (unless limit-down)
            for pos in list(holdings):
                key_sell = (date, pos['ts_code'])
                if key_sell in prices:
                    px_sell = prices[key_sell]
                    down_limit = get_limit_price(pos['ts_code'], px_sell['pre_close'], 'down')
                    if px_sell['open'] <= down_limit:
                        continue  # Cannot sell, carry over
                    exit_px = px_sell['open'] * (1 - SLIPPAGE)
                    revenue = pos['shares'] * exit_px
                    cost = max(5.0, revenue * COMMISSION) + revenue * STAMP_DUTY
                    capital += (revenue - cost)
                    holdings.remove(pos)
            # B. Rolling train model if month changes
            month = date.month
            if month != last_month:
                # Ensure training end date is strictly look-ahead-free (at least 7 trading days before rebalance date)
                cutoff_idx = all_dates.index(date) - 7
                if cutoff_idx >= 0:
                    cutoff_date = all_dates[cutoff_idx]
                    train_data = df[(df['trade_date'] <= cutoff_date) & (df['trade_date'] >= cutoff_date - pd.Timedelta(days=365*2))]
                else:
                    train_data = pd.DataFrame()
                
                # Downsample training to weekly frequency to increase speed
                unique_train_dates = sorted(train_data['trade_date'].unique())
                train_dates_downsampled = unique_train_dates[::5]
                train_data_downsampled = train_data[train_data['trade_date'].isin(train_dates_downsampled)]
                
                cur_model, cur_scaler = train_super_model(train_data_downsampled)
                last_month = month
            
            # C. Generate Signals on the previous trading day T-1
            d_signal_idx = all_dates.index(date) - 1
            if d_signal_idx >= 0:
                d_signal = all_dates[d_signal_idx]
                day_data = df[df['trade_date'] == d_signal].dropna(subset=ESSENTIAL_COLS).copy()
                if cur_model and not day_data.empty:
                    X_test = day_data[FEATURE_COLS].replace([np.inf, -np.inf], np.nan).fillna(0)
                    X = cur_scaler.transform(X_test)
                    day_data['prob'] = cur_model.predict_proba(X)[:, 1]
                    picks = day_data.sort_values('prob', ascending=False).head(TOP_N)
                    
                    # D. Buy picks at Open of date T
                    if not picks.empty:
                        cash_per = capital / TOP_N
                        for _, row in picks.iterrows():
                            key_buy = (date, row['ts_code'])
                            if key_buy in prices:
                                px_buy = prices[key_buy]
                                up_limit = get_limit_price(row['ts_code'], px_buy['pre_close'], 'up')
                                if px_buy['open'] >= up_limit:
                                    continue  # Cannot buy, skip
                                buy_px = px_buy['open'] * (1 + SLIPPAGE)
                                shares = int(cash_per / buy_px / 100) * 100
                                if shares >= 100:
                                    cost = shares * buy_px + max(5.0, shares * buy_px * COMMISSION)
                                    capital -= cost
                                    holdings.append({'ts_code': row['ts_code'], 'shares': shares, 'buy_px': buy_px})
        
        # 2. Daily NAV Calculation (at Close)
        mv = 0.0
        for p in holdings:
            key_day = (date, p['ts_code'])
            close_px = prices[key_day]['close'] if key_day in prices else p['buy_px']
            mv += p['shares'] * close_px
            
        nav = capital + mv
        equity.append({'trade_date': date, 'nav': nav})
        
    eq_df = pd.DataFrame(equity)
    eq_df['pnl'] = eq_df['nav'].pct_change().fillna(0)
    return eq_df

def generate_pit_signals(df, target_date):
    """Generate trade picks for a specific point-in-time date T (executed at T+1 Open)."""
    target_dt = pd.Timestamp(target_date)
    all_dates = sorted(df['trade_date'].unique())
    
    if target_dt not in all_dates:
        # Fallback to nearest date prior to target_date
        past_dates = [d for d in all_dates if d <= target_dt]
        if not past_dates:
            raise ValueError(f"No trading dates available on or before {target_date}.")
        target_dt = past_dates[-1]
        print(f"Warning: {target_date} is not a trading day. Using nearest trading day: {target_dt.strftime('%Y-%m-%d')}")
        
    print(f"\nGenerating signals for Close of: {target_dt.strftime('%Y-%m-%d')} (Execution: Next Open)")
    
    # 1. Training Set: 2 years lookback ending on or before T-7 (to ensure all labels fully realized by T-1)
    target_idx = all_dates.index(target_dt)
    cutoff_idx = target_idx - 7
    if cutoff_idx >= 0:
        cutoff_date = all_dates[cutoff_idx]
        train_data = df[(df['trade_date'] <= cutoff_date) & (df['trade_date'] >= cutoff_date - pd.Timedelta(days=365*2))]
    else:
        train_data = pd.DataFrame()
        
    unique_train_dates = sorted(train_data['trade_date'].unique())
    train_dates_downsampled = unique_train_dates[::5]
    train_data_downsampled = train_data[train_data['trade_date'].isin(train_dates_downsampled)]
    
    # 2. Train point-in-time model
    print("Training point-in-time XGBoost model...")
    model, scaler = train_super_model(train_data_downsampled)
    
    # 3. Predict on target date T
    day_data = df[df['trade_date'] == target_dt].dropna(subset=ESSENTIAL_COLS).copy()
    if day_data.empty:
        print("Error: No prediction data available for target date.")
        return
        
    X_test = day_data[FEATURE_COLS].replace([np.inf, -np.inf], np.nan).fillna(0)
    X = scaler.transform(X_test)
    day_data['prob'] = model.predict_proba(X)[:, 1]
    
    picks = day_data.sort_values('prob', ascending=False).head(TOP_N)
    
    # Print trading signal report
    print("\n" + "="*80)
    print(f"  SUPER-WEEKLY PORTFOLIO SIGNAL REPORT | SIGNAL DATE: {target_dt.strftime('%Y-%m-%d')}")
    print(f"  RECOMMENDED TRADES EXECUTION: NEXT TRADING DAY OPEN")
    print("="*80)
    for i, (_, row) in enumerate(picks.iterrows()):
        print(f"  [{i+1}] Stock Code: {row['ts_code']} | Prob: {row['prob']:.2%} | Close Price: {row['close']:.2f}")
    print("="*80)
    print("  Execution Instructions:\n"
          "  1. Buy the selected stocks at the Open of the next trading day.\n"
          "  2. Hold for a period of 5 trading days.\n"
          "  3. Risk management: Set standard -15% hard stop-loss per position.")
    print("="*80)
    
    # Save signal output
    sig_output = {
        'signal_date': target_dt.strftime('%Y-%m-%d'),
        'picks': picks[['ts_code', 'close', 'prob']].to_dict('records')
    }
    filename = f"super_weekly_signals_{target_dt.strftime('%Y%m%d')}.json"
    filepath = os.path.join(SIGNALS_DIR, filename)
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(sig_output, f, indent=4, ensure_ascii=False)
    print(f"Signal file saved successfully to: [signals/{filename}](file:///{filepath})")

def main():
    parser = argparse.ArgumentParser(description="Super-Weekly PIT Signal Pipeline")
    parser.add_argument('--mode', type=str, choices=['generate', 'backtest'], default='generate',
                        help="Operating mode: 'generate' for trading signals, 'backtest' for walk-forward backtest.")
    parser.add_argument('--date', type=str, default=None,
                        help="Point-in-time date for signal generation (YYYY-MM-DD). If omitted, uses the latest data date.")
    parser.add_argument('--start', type=str, default='2018-01-01',
                        help="Start date for backtesting (YYYY-MM-DD).")
    args = parser.parse_args()
    
    # Determine the date range to load
    # To run a backtest starting at start_date, we need data from (start_date - 2 years) for training
    if args.mode == 'backtest':
        start_load = (pd.Timestamp(args.start) - pd.Timedelta(days=365*2)).strftime('%Y%m%d')
        end_load = '20260620'
    else:
        # For signal generation, we need up to the target date (or latest)
        end_dt = pd.Timestamp(args.date) if args.date else pd.Timestamp.now()
        start_load = (end_dt - pd.Timedelta(days=365*2.5)).strftime('%Y%m%d')
        end_load = end_dt.strftime('%Y%m%d')
        
    # Load and build features
    raw_df = load_pit_data(start_load, end_load)
    df = build_features(raw_df)
    df = add_labels(df, horizon=5)
    
    if args.mode == 'backtest':
        eq_df = run_daily_super_backtest(df, start_date=args.start)
        
        # Output summary and save PNL
        pnl_path = os.path.join(OUT_DIR, 'etf-valuation-strategy', 'data', 'daily_super_weekly_pnl.csv')
        os.makedirs(os.path.dirname(pnl_path), exist_ok=True)
        eq_df.to_csv(pnl_path, index=False)
        print(f"\n[SUCCESS] Walk-forward backtest PNL saved to {pnl_path}")
        
        years = (eq_df['trade_date'].iloc[-1] - eq_df['trade_date'].iloc[0]).days / 365.25
        cagr = (eq_df['nav'].iloc[-1] / 100_000.0) ** (1.0 / years) - 1.0
        vol = eq_df['pnl'].std() * np.sqrt(252)
        sharpe = cagr / vol if vol > 0 else 0
        mdd = ((eq_df['nav'] - eq_df['nav'].cummax()) / eq_df['nav'].cummax()).min()
        print("\n" + "="*80)
        print(f"  HISTORICAL WALK-FORWARD BACKTEST SUMMARY ({args.start} to {eq_df['trade_date'].iloc[-1].strftime('%Y-%m-%d')})")
        print("="*80)
        print(f"  CAGR:          {cagr:.2%}")
        print(f"  Volatility:    {vol:.2%}")
        print(f"  Sharpe Ratio:  {sharpe:.2f}")
        print(f"  Max Drawdown:  {mdd:.2%}")
        print("="*80)
        
    else:
        # Determine target date
        target_date = args.date if args.date else raw_df['trade_date'].max().strftime('%Y-%m-%d')
        generate_pit_signals(df, target_date)

if __name__ == "__main__":
    main()

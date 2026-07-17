import os
import pandas as pd
import numpy as np
import joblib
import pickle
from tqdm import tqdm
import sys
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from dotenv import load_dotenv

# Ensure environment variables are loaded
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(ROOT_DIR)
load_dotenv(os.path.join(ROOT_DIR, ".env"))

from infra_data.storage import DataStorage

DATA_DIR = r'D:\iquant_data\data_v2'
PRICE_DIR = os.path.join(DATA_DIR, 'data_day1')
RANK_DIR  = os.path.join(DATA_DIR, 'ths_rank1')
CHIP_DIR  = os.path.join(DATA_DIR, 'cyq1')
OTHER_DIR = os.path.join(DATA_DIR, 'other_day1')
MODEL_DIR = os.path.join(ROOT_DIR, 'results', 'models')
os.makedirs(MODEL_DIR, exist_ok=True)

def get_limit_price(code, pre_close, direction='up'):
    ratio = 0.2 if code.startswith(('30', '68')) else 0.1
    if direction == 'up':
        return round(pre_close * (1 + ratio), 2)
    return round(pre_close * (1 - ratio), 2)

def load_data_for_date_range(start_date, end_date, dates):
    storage = DataStorage()
    valid_dates_series = pd.Series([pd.to_datetime(d) for d in dates 
                                     if start_date <= d <= end_date]).sort_values()
    news_market_df, news_stock_sector_df = storage.load_news_data(start_date, end_date, valid_dates_series)
    
    if not news_market_df.empty:
        news_market_df['trade_date'] = news_market_df['trade_date'].dt.strftime('%Y%m%d')
    if not news_stock_sector_df.empty:
        news_stock_sector_df['trade_date'] = news_stock_sector_df['trade_date'].dt.strftime('%Y%m%d')
    
    return news_market_df, news_stock_sector_df

def load_options_features():
    print("[INFO] Loading Options PCR & QVIX features...")
    pcr_csv = r"D:\iquant_data\data_v2\qiquan\historical_pcr.csv"
    if not os.path.exists(pcr_csv):
        print(f"[WARNING] Options PCR CSV not found at: {pcr_csv}")
        return pd.DataFrame()
    try:
        df_pcr = pd.read_csv(pcr_csv)
        df_pcr['date'] = pd.to_datetime(df_pcr['date'])
        df_pcr['trade_date'] = df_pcr['date'].dt.strftime('%Y%m%d')
        df_pcr_clean = df_pcr[['trade_date', 'pcr_50', 'oi_pcr_50']].rename(columns={
            'pcr_50': 'opt_pcr_vol_50',
            'oi_pcr_50': 'opt_pcr_oi_50'
        })
    except Exception as e:
        print(f"[ERROR] Failed to load PCR: {e}")
        return pd.DataFrame()

    import akshare as ak
    try:
        df_qvix = ak.index_option_50etf_qvix()
        df_qvix['date'] = pd.to_datetime(df_qvix['date'])
        df_qvix['trade_date'] = df_qvix['date'].dt.strftime('%Y%m%d')
        df_qvix['opt_qvix_close'] = df_qvix['close']
        df_qvix['opt_qvix_ma'] = df_qvix['close'].rolling(20).mean()
        df_qvix['opt_qvix_std'] = df_qvix['close'].rolling(20).std()
        df_qvix['opt_qvix_zscore'] = (df_qvix['close'] - df_qvix['opt_qvix_ma']) / df_qvix['opt_qvix_std']
        df_qvix_clean = df_qvix[['trade_date', 'opt_qvix_close', 'opt_qvix_zscore']].fillna(0)
    except Exception as e:
        print(f"[ERROR] Failed to load QVIX: {e}")
        df_qvix_clean = pd.DataFrame()

    if df_qvix_clean.empty:
        return df_pcr_clean
    merged = pd.merge(df_pcr_clean, df_qvix_clean, on='trade_date', how='outer').sort_values('trade_date').reset_index(drop=True)
    merged[['opt_pcr_vol_50', 'opt_pcr_oi_50', 'opt_qvix_close', 'opt_qvix_zscore']] = \
        merged[['opt_pcr_vol_50', 'opt_pcr_oi_50', 'opt_qvix_close', 'opt_qvix_zscore']].ffill().bfill().fillna(0)
    return merged

def prepare_features_for_date(d_curr, d_next, news_market_df, news_stock_sector_df, options_df=None):
    p_rank = os.path.join(RANK_DIR, f"{d_curr}.parquet")
    p_chip = os.path.join(CHIP_DIR, f"{d_curr}.parquet")
    p_price = os.path.join(PRICE_DIR, f"{d_curr}.parquet")
    
    if not all(os.path.exists(p) for p in [p_rank, p_chip, p_price]):
        return None
    
    rank_df = pd.read_parquet(p_rank)
    rank_df['hot_rank_pct'] = rank_df['hot'].rank(pct=True)
    
    chip_df = pd.read_parquet(p_chip)
    chip_df['chip_concentration'] = (chip_df['cost_85pct'] - chip_df['cost_15pct']) / (chip_df['cost_50pct'] + 1e-8)
    
    price_df = pd.read_parquet(p_price, columns=['ts_code', 'close', 'pct_chg', 'amount', 'vol', 'pre_close'])
    
    df = pd.merge(rank_df[['ts_code', 'hot_rank_pct']], price_df, on='ts_code')
    df = pd.merge(df, chip_df[['ts_code', 'chip_concentration', 'winner_rate']], on='ts_code')
    df['trade_date'] = d_next
    
    if not news_market_df.empty:
        df = pd.merge(df, news_market_df, on='trade_date', how='left')
    else:
        df['news_market_impact'] = 0.0
        
    if not news_stock_sector_df.empty:
        df = pd.merge(df, news_stock_sector_df, on=['trade_date', 'ts_code'], how='left')
    else:
        df['news_stock_impact'] = 0.0
        df['news_sector_impact'] = 0.0
        
    df[['news_market_impact', 'news_stock_impact', 'news_sector_impact']] = \
        df[['news_market_impact', 'news_stock_impact', 'news_sector_impact']].fillna(0.0)
        
    if options_df is not None and not options_df.empty:
        opt_row = options_df[options_df['trade_date'] == d_curr]
        if not opt_row.empty:
            for col in ['opt_pcr_vol_50', 'opt_pcr_oi_50', 'opt_qvix_close', 'opt_qvix_zscore']:
                df[col] = float(opt_row[col].values[0])
        else:
            for col in ['opt_pcr_vol_50', 'opt_pcr_oi_50', 'opt_qvix_close', 'opt_qvix_zscore']:
                df[col] = 0.0
    else:
        for col in ['opt_pcr_vol_50', 'opt_pcr_oi_50', 'opt_qvix_close', 'opt_qvix_zscore']:
            df[col] = 0.0
            
    return df

def prepare_training_data(start_date, end_date, dates, options_df=None):
    cache_path = r'D:\iquant_data\data_v2\qiquan\dragon_features_cache.parquet'
    if not os.path.exists(cache_path):
        print(f"[WARNING] Cache parquet not found at: {cache_path}")
        return None, None, None
        
    df_all = pd.read_parquet(cache_path)
    
    try:
        start_idx = dates.index(start_date)
        min_trade_date = dates[start_idx+1]
    except Exception:
        min_trade_date = start_date
        
    try:
        end_idx = dates.index(end_date)
        max_trade_date = dates[end_idx+1] if end_idx+1 < len(dates) else dates[-1]
    except Exception:
        max_trade_date = end_date
        
    df_train = df_all[(df_all['trade_date'] >= min_trade_date) & (df_all['trade_date'] <= max_trade_date)]
    
    feature_cols = [
        'hot_rank_pct', 'pct_chg', 'amount', 'vol', 'chip_concentration', 'winner_rate', 
        'news_market_impact', 'news_stock_impact', 'news_sector_impact',
        'opt_pcr_vol_50', 'opt_pcr_oi_50', 'opt_qvix_close', 'opt_qvix_zscore'
    ]
    
    X = df_train[feature_cols].fillna(0)
    y = df_train['label']
    
    return X, y, feature_cols

def train_model(X, y, model_name):
    print(f"[INFO] Training model {model_name}...")
    model = XGBClassifier(
        n_estimators=150, 
        max_depth=5, 
        learning_rate=0.08, 
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric='logloss',
        n_jobs=-1,
        tree_method='hist'
    )
    model.fit(X, y)
    return model

def is_new_stock(ts_code, date_int, stock_dates):
    if not stock_dates or ts_code not in stock_dates:
        return False
    dates = stock_dates[ts_code]
    count = sum(1 for d in dates if d < date_int)
    return count < 10

def run_monthly_retraining_backtest():
    dates = sorted([f.replace('.parquet', '') for f in os.listdir(PRICE_DIR) if f.endswith('.parquet')])
    
    cache_path = r'D:\iquant_data\data_v2\qiquan\dragon_features_cache.parquet'
    if not os.path.exists(cache_path):
        print(f"[ERROR] Parquet cache not found at {cache_path}! Please run build_dragon_features_cache.py first.")
        return
    print(f"[INFO] Loading dragon features cache from {cache_path}...")
    df_cache = pd.read_parquet(cache_path)
    
    # Load stock dates cache
    stock_cache_path = os.path.join(ROOT_DIR, 'trade_stock_dates_cache.pkl')
    stock_dates = {}
    if os.path.exists(stock_cache_path):
        try:
            with open(stock_cache_path, 'rb') as f:
                stock_dates = pickle.load(f)
            print(f"[INFO] Loaded stock dates cache for {len(stock_dates)} stocks.")
        except Exception as e:
            print(f"[WARNING] Failed to load stock cache: {e}")
            
    train_start = '20200101'
    test_start = '20240101'
    test_end = '20261231'
    
    test_dates = [d for d in dates if test_start <= d <= test_end]
    
    initial_cap = 100000.0
    capital = initial_cap
    equity = []
    
    # Strict holdings tracker
    holdings = [] # list of dicts: {'ts_code': str, 'shares': int, 'buy_px': float, 'sell_date': str}
    trade_log = []
    
    current_month = None
    current_model = None
    current_feats = None
    
    # Loop over dates
    for i in tqdm(range(len(test_dates)-2), desc="Strict T+1 Walk-Forward"):
        d_curr = test_dates[i]       # T (feature calculation and scoring)
        d_next = test_dates[i+1]     # T+1 (buying open)
        d_sell = test_dates[i+2]     # T+2 (selling)
        
        curr_dt = pd.to_datetime(d_curr)
        
        # A. Sell holdings from previous cycle (T-1 buy)
        # We sell today if sell_date == d_curr
        if holdings:
            to_sell = [pos for pos in holdings if pos.get('sell_date') == d_curr]
            if to_sell:
                p_sell_file = os.path.join(PRICE_DIR, f"{d_curr}.parquet")
                if os.path.exists(p_sell_file):
                    sell_data = pd.read_parquet(p_sell_file, columns=['ts_code', 'open', 'high', 'low', 'close', 'pre_close'])
                    for pos in to_sell:
                        stock_perf = sell_data[sell_data['ts_code'] == pos['ts_code']]
                        if stock_perf.empty:
                            sell_px = pos['buy_px'] * 0.9985
                        else:
                            row = stock_perf.iloc[0]
                            down_limit = get_limit_price(row['ts_code'], row['pre_close'], 'down')
                            
                            # If open is down limit and high is down limit, cannot sell
                            if row['open'] <= down_limit and row['high'] <= down_limit:
                                # Roll sale to the next trading day
                                try:
                                    curr_idx = test_dates.index(d_curr)
                                    next_tr_date = test_dates[curr_idx+1]
                                    pos['sell_date'] = next_tr_date
                                    print(f"[LIMIT DOWN] {pos['ts_code']} is limit down on {d_curr}. Rolling sale to {next_tr_date}")
                                except Exception:
                                    sell_px = row['close'] * 0.9985
                                    capital += pos['shares'] * sell_px
                                    trade_log.append({
                                        'buy_date': pos['buy_date'],
                                        'sell_date': d_curr,
                                        'ts_code': pos['ts_code'],
                                        'shares': pos['shares'],
                                        'buy_px': pos['buy_px'],
                                        'sell_px': sell_px,
                                        'ret': (sell_px / pos['buy_px'] - 1),
                                        'prob': pos['prob']
                                    })
                                    holdings.remove(pos)
                                continue
                                
                            # Check T+2 intraday +4% trigger or sell at close
                            if row['high'] >= pos['buy_px'] * 1.04:
                                sell_px = pos['buy_px'] * 1.04 * 0.9985 # less fees
                            else:
                                sell_px = row['close'] * 0.9985
                                
                        revenue = pos['shares'] * sell_px
                        capital += revenue
                        trade_log.append({
                            'buy_date': pos['buy_date'],
                            'sell_date': d_curr,
                            'ts_code': pos['ts_code'],
                            'shares': pos['shares'],
                            'buy_px': pos['buy_px'],
                            'sell_px': sell_px,
                            'ret': (sell_px / pos['buy_px'] - 1),
                            'prob': pos['prob']
                        })
                        holdings.remove(pos)
                    
        # B. Monthly Retraining
        if curr_dt.month != current_month or current_model is None:
            current_month = curr_dt.month
            train_end = d_curr
            X_train, y_train, feats = prepare_training_data(train_start, train_end, dates)
            
            if X_train is not None and len(X_train) > 0:
                # Downsample negative class to balance training ALWAYS (1:2 ratio)
                pos_mask = (y_train == 1)
                neg_mask = (y_train == 0)
                n_pos = pos_mask.sum()
                n_neg = neg_mask.sum()
                if n_pos > 0 and n_neg > 0:
                    neg_idx = y_train[neg_mask].sample(min(n_pos * 2, n_neg), random_state=42).index
                    pos_idx = y_train[pos_mask].index
                    bal_idx = pos_idx.union(neg_idx)
                    X_train_bal = X_train.loc[bal_idx]
                    y_train_bal = y_train.loc[bal_idx]
                    print(f"  [BALANCE] Balanced training set: {len(pos_idx)} positive, {len(neg_idx)} negative samples.")
                else:
                    X_train_bal = X_train
                    y_train_bal = y_train
                    
                model_name = f'model_{d_curr}.joblib'
                model_path = os.path.join(MODEL_DIR, model_name)
                
                current_model = train_model(X_train_bal, y_train_bal, model_name)
                current_feats = feats
                joblib.dump((current_model, current_feats), model_path)
                print(f"[SUCCESS] Saved model to {model_path}")
                
        if current_model is None:
            equity.append({'date': pd.to_datetime(d_curr), 'nav': capital})
            continue
            
        # C. Predict candidates at T-day (d_curr)
        df = df_cache[df_cache['trade_date'] == d_next].copy()
        if df.empty:
            equity.append({'date': pd.to_datetime(d_curr), 'nav': capital + sum(p['shares'] * p['buy_px'] for p in holdings)})
            continue
            
        # Apply filters
        # 1. No 688
        df = df[~df['ts_code'].str.startswith('688')]
        
        # 2. Market cap <= 500,000 (50 billion)
        p_other = os.path.join(OTHER_DIR, f"{d_curr}.parquet")
        if os.path.exists(p_other):
            try:
                other_df = pd.read_parquet(p_other, columns=['ts_code', 'circ_mv'])
                df = pd.merge(df, other_df, on='ts_code', how='inner')
                df = df[df['circ_mv'] <= 500000]
            except Exception:
                pass
                
        # 3. New stock filter
        if stock_dates:
            df = df[~df['ts_code'].apply(lambda x: is_new_stock(x, int(d_curr), stock_dates))]
            
        if df.empty:
            equity.append({'date': pd.to_datetime(d_curr), 'nav': capital + sum(p['shares'] * p['buy_px'] for p in holdings)})
            continue
            
        X = df[current_feats].fillna(0)
        try:
            df['prob'] = current_model.predict_proba(X)[:, 1]
        except Exception:
            df['prob'] = 0
            
        picks = df[df['prob'] > 0.8].sort_values('prob', ascending=False).head(3)
        if picks.empty:
            picks = df.sort_values('prob', ascending=False).head(1)
            
        # D. Buy at T+1 open (d_next)
        p_next_file = os.path.join(PRICE_DIR, f"{d_next}.parquet")
        if os.path.exists(p_next_file):
            next_prices = pd.read_parquet(p_next_file, columns=['ts_code', 'open', 'pre_close'])
            buy_candidates = pd.merge(picks[['ts_code', 'prob']], next_prices, on='ts_code')
            if not buy_candidates.empty:
                cash_per_stock = capital / 3
                for _, row in buy_candidates.iterrows():
                    up_limit = get_limit_price(row['ts_code'], row['pre_close'], 'up')
                    if row['open'] >= up_limit:
                        continue # Limit up cannot buy
                        
                    buy_px = row['open'] * 1.001
                    shares = int(cash_per_stock / buy_px / 100) * 100
                    if shares >= 100:
                        cost = shares * buy_px
                        capital -= cost
                        holdings.append({
                            'ts_code': row['ts_code'],
                            'shares': shares,
                            'buy_px': buy_px,
                            'sell_date': d_sell,
                            'buy_date': d_next,
                            'prob': float(row['prob'])
                        })
                        
        equity.append({'date': pd.to_datetime(d_curr), 'nav': capital + sum(p['shares'] * p['buy_px'] for p in holdings)})

    # Finalize remaining holdings at the end of backtest
    if holdings:
        last_sell_day = test_dates[-1]
        p_sell_file = os.path.join(PRICE_DIR, f"{last_sell_day}.parquet")
        if os.path.exists(p_sell_file):
            sell_data = pd.read_parquet(p_sell_file, columns=['ts_code', 'close'])
            for pos in list(holdings):
                stock_perf = sell_data[sell_data['ts_code'] == pos['ts_code']]
                sell_px = stock_perf.iloc[0]['close'] * 0.9985 if not stock_perf.empty else pos['buy_px']
                capital += pos['shares'] * sell_px
                holdings.remove(pos)

    eq_df = pd.DataFrame(equity)
    
    if len(eq_df) == 0:
        print("[ERROR] No backtest records generated!")
        return
        
    # Calculate indicators
    total_ret = capital / initial_cap - 1
    years = len(eq_df) / 252.0 if len(eq_df) > 0 else 1
    ann_ret = (1+total_ret)**(1/years)-1 if years > 0 else 0
    df_ret = eq_df['nav'].pct_change()
    mdd = ((eq_df['nav'] - eq_df['nav'].cummax()) / eq_df['nav'].cummax()).min()
    vol = df_ret.std() * np.sqrt(252)
    sharpe = ann_ret / vol if vol > 0 else 0
    
    print("\n" + "="*80)
    print(" STRICT T+1 MONTHLY WALK-FORWARD WITH OPTIONS RESULTS")
    print("="*80)
    print(f"Final Capital: RMB {capital:,.2f}")
    print(f"Total Return: {total_ret:.2%}")
    print(f"Annual Return: {ann_ret:.2%}")
    print(f"Max Drawdown: {mdd:.2%}")
    print(f"Sharpe Ratio: {sharpe:.2f}")
    print("="*80)
    
    # Save results
    eq_df.to_csv('dragon_daily_real_t1_monthly_retrain_equity.csv', index=False)
    if trade_log:
        pd.DataFrame(trade_log).to_csv('dragon_daily_real_t1_monthly_retrain_trades.csv', index=False)
        print(f"[SUCCESS] Saved trade logs to dragon_daily_real_t1_monthly_retrain_trades.csv!")
    
    # Save the latest model (from last fold) to the root for live predictions
    if current_model is not None:
        joblib.dump((current_model, current_feats), 'daily_dragon_news_model.joblib')
        print(f"[SUCCESS] Exported latest model to daily_dragon_news_model.joblib!")
        
    plt.figure(figsize=(12,6))
    plt.plot(eq_df['date'], eq_df['nav'], label='Strict T+1 Retrain with Options', linewidth=2, color='#E65100')
    plt.title('Daily Dragon Strategy - Strict A-Share T+1 with Options')
    plt.xlabel('Date')
    plt.ylabel('Capital')
    plt.legend()
    plt.grid(True)
    plt.savefig('dragon_daily_real_t1_monthly_retrain.png', dpi=150)
    print("[SUCCESS] Saved strict backtest plot to dragon_daily_real_t1_monthly_retrain.png")

if __name__ == "__main__":
    run_monthly_retraining_backtest()

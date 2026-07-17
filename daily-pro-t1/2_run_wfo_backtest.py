import os
import sys
import pandas as pd
import numpy as np
import xgboost as xgb
from tqdm import tqdm
from sklearn.preprocessing import RobustScaler
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Configuration
INITIAL_CAP = 1000000.0
TOP_N = 3
SLIPPAGE = 0.001
COMMISSION = 0.0003
STAMP_DUTY = 0.0005
MARKET_CAP_LIMIT = 5000000 # 50 Billion (Unit: 10k)

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(OUT_DIR, 'data', 'super_dataset.parquet')

def train_model(train_df, features):
    sub = train_df.dropna(subset=['label']).copy()
    if len(sub) < 100: return None, None
    X = sub[features].copy()
    if 'hot_rank_pct' in X.columns:
        X['hot_rank_pct'] = X['hot_rank_pct'].fillna(0.5)
    X = X.fillna(0)
    y = sub['label']
    
    scaler = RobustScaler()
    X_s = scaler.fit_transform(X)
    
    # Simple hyperparameters for stability
    model = xgb.XGBClassifier(
        n_estimators=100, 
        max_depth=5, 
        learning_rate=0.05, 
        random_state=42,
        tree_method='hist',
        n_jobs=4, 
        eval_metric='logloss'
    )
    model.fit(X_s, y)
    return model, scaler

def run_wfo_backtest(df, news_col_name, label_name):
    print(f"\n>>> Running WFO Backtest for: {label_name} using feature: {news_col_name}")
    
    start_date = '20240101'
    end_date = '20260331'
    
    # Filter test scope using buy_date
    test_scope = df[(df['buy_date'] >= start_date) & (df['buy_date'] <= end_date)].copy()
    test_dates = sorted(test_scope['buy_date'].unique())
    
    features = ['hot_rank_pct', 'chip_concentration', 'winner_rate', news_col_name]
    
    capital = INITIAL_CAP
    equity = []
    
    current_model = None
    cur_scaler = None
    last_train_month = -1
    
    for d_buy in tqdm(test_dates, desc=f"Simulating {label_name}"):
        curr_dt = pd.to_datetime(d_buy)
        
        # 1. Walk-Forward Retraining (Monthly)
        if curr_dt.month != last_train_month:
            # We use ALL available history strictly strictly BEFORE the 1st of the current month
            first_day_of_month = curr_dt.replace(day=1).strftime('%Y%m%d')
            train_data = df[df['buy_date'] < first_day_of_month].copy()
            
            if not train_data.empty:
                print(f"\n[{curr_dt.strftime('%Y-%m-%d')}] Retraining model on {len(train_data)} past samples (up to {first_day_of_month})...", flush=True)
                new_model, new_scaler = train_model(train_data, features)
                if new_model is not None:
                    current_model = new_model
                    cur_scaler = new_scaler
            
            last_train_month = curr_dt.month
            
        # 2. Daily Prediction
        day_data = df[df['buy_date'] == d_buy].copy()
        if day_data.empty:
            equity.append({'date': curr_dt, 'nav': capital})
            continue
            
        # Filter rules
        # < 50B Market Cap, Non-688
        day_data = day_data[(day_data['circ_mv'] <= MARKET_CAP_LIMIT) & (~day_data['ts_code'].str.startswith('688'))]
        
        if current_model is not None and not day_data.empty:
            feature_data = day_data[features].copy()
            if 'hot_rank_pct' in feature_data.columns:
                feature_data['hot_rank_pct'] = feature_data['hot_rank_pct'].fillna(0.5)
            X_test = cur_scaler.transform(feature_data.fillna(0))
            
            day_data['prob'] = current_model.predict_proba(X_test)[:, 1]
            # Must meet minimum confidence to invest
            day_data = day_data[day_data['prob'] > 0.55] 
            # Rank by model probability
            picks = day_data.sort_values('prob', ascending=False).head(TOP_N)
        else:
            picks = pd.DataFrame()
            
        # 3. Execution Execution (Buy d_buy Open, Sell d_sell Open -> strict T+1)
        day_pnl = 0
        if not picks.empty:
            alloc = capital / TOP_N
            for _, row in picks.iterrows():
                code = row['ts_code']
                
                open_buy = row['entry_open']
                pre_close = row['entry_pre_close']
                open_sell = row['exit_open']
                
                # Check NaNs
                if pd.isna(open_buy) or pd.isna(pre_close) or pd.isna(open_sell):
                    continue
                    
                # Limit up filter on entry
                limit_ratio = 1.195 if code.startswith('300') else 1.095
                if open_buy >= round(pre_close * limit_ratio, 2):
                    continue
                    
                # Use calculated true_t1_ret which accounts for dividends/splits safely
                ret = row['true_t1_ret']
                ret -= (SLIPPAGE * 2 + COMMISSION * 2 + STAMP_DUTY)
                day_pnl += alloc * ret
                
        capital += day_pnl
        equity.append({'date': curr_dt, 'nav': capital})
        
    eq_df = pd.DataFrame(equity)
    return eq_df

if __name__ == "__main__":
    if not os.path.exists(DATA_PATH):
        print(f"Error: Dataset not found at {DATA_PATH}. Please run 1_build_dataset.py first.")
        sys.exit(1)
        
    print("Loading Mega Dataset...")
    df = pd.read_parquet(DATA_PATH)
    print(f"Loaded {len(df)} rows.")
    
    # Run backtest for Major (Post-market, day T)
    eq_major = run_wfo_backtest(df, "news_major_impact", "News Major")
    
    # Run backtest for Major1 (Pre-market, day T+1)
    eq_major1 = run_wfo_backtest(df, "news_major1_impact", "News Major1")
    
    # Plot results
    plt.figure(figsize=(12, 7))
    plt.plot(eq_major['date'], eq_major['nav'], label='News Major (Post-market)', color='royalblue')
    plt.plot(eq_major1['date'], eq_major1['nav'], label='News Major1 (Pre-market)', color='tomato')
    plt.title('Daily Pro T+1 WFO Backtest: Major vs Major1')
    plt.xlabel('Date')
    plt.ylabel('NAV')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    output_png = os.path.join(OUT_DIR, 'wfo_comparison.png')
    plt.savefig(output_png, dpi=150)
    print(f"\nFinal major NAV: {eq_major.iloc[-1]['nav']:.2f}")
    print(f"Final major1 NAV: {eq_major1.iloc[-1]['nav']:.2f}")
    print(f"Plot saved to {output_png}")

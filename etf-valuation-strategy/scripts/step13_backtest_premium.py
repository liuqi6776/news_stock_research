import os
import pandas as pd
import numpy as np

# Define paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(PROJECT_DIR, 'data')
RESULTS_DIR = os.path.join(PROJECT_DIR, 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

def load_premium_data(ma_window=250, val_window=1200):
    print(f"Loading CSI 1000 data with MA={ma_window}, ValWindow={val_window}...")
    df_1000_price = pd.read_csv(os.path.join(DATA_DIR, 'zz1000_daily.csv'))
    df_1000_val = pd.read_csv(os.path.join(DATA_DIR, 'zz1000_valuation.csv'))
    
    # Load other assets
    df_nasdaq_price = pd.read_csv(os.path.join(DATA_DIR, 'nasdaq_etf_daily.csv'))
    df_gold_price = pd.read_csv(os.path.join(DATA_DIR, 'gold_etf_daily.csv'))
    df_bond_price = pd.read_csv(os.path.join(DATA_DIR, 'bond_etf_daily.csv'))
    
    # Format dates
    for df in [df_1000_price, df_1000_val, df_nasdaq_price, df_gold_price, df_bond_price]:
        df['trade_date'] = pd.to_datetime(df['trade_date'].astype(str))
        df.sort_values('trade_date', inplace=True)
        df.reset_index(drop=True, inplace=True)
        
    # Nasdaq split adjustment on 2022-01-14 (if Nasdaq is used)
    # Check if nasdaq exists in index and adjust
    split_date = pd.to_datetime('2022-01-14')
    if df_nasdaq_price['trade_date'].min() < split_date:
        # Check if Nasdaq ETF split adjustment factor is needed
        adj_factor = 1.038 / 5.192
        mask = df_nasdaq_price['trade_date'] < split_date
        for col in ['close', 'open', 'high', 'low', 'pre_close']:
            if col in df_nasdaq_price.columns:
                df_nasdaq_price.loc[mask, col] *= adj_factor

    # CSI 1000 price MA and valuation quantiles
    df_1000_price['ma'] = df_1000_price['close'].rolling(ma_window).mean()
    df_1000_val['pe_q'] = df_1000_val['pe_ttm'].rolling(window=val_window, min_periods=250).rank(pct=True)
    df_1000_val['pb_q'] = df_1000_val['pb'].rolling(window=val_window, min_periods=250).rank(pct=True)
    df_1000_val['val_q'] = (df_1000_val['pe_q'] + df_1000_val['pb_q']) / 2.0
    
    # Merge CSI 1000 price and valuation
    m1000 = pd.merge(df_1000_price[['trade_date', 'close', 'pct_chg', 'ma']], 
                    df_1000_val[['trade_date', 'val_q']], on='trade_date', how='inner')
    m1000.rename(columns={'close': 'close_1000', 'pct_chg': 'pct_chg_1000', 'ma': 'ma_1000', 'val_q': 'val_q_1000'}, inplace=True)
    
    # Map other assets returns
    df_nasdaq_price['nasdaq_ret'] = df_nasdaq_price['pct_chg'] / 100.0
    df_gold_price['gold_ret'] = df_gold_price['pct_chg'] / 100.0
    df_bond_price['bond_ret'] = df_bond_price['pct_chg'] / 100.0
    
    nasdaq_map = df_nasdaq_price.set_index('trade_date')['nasdaq_ret'].to_dict()
    gold_map = df_gold_price.set_index('trade_date')['gold_ret'].to_dict()
    bond_map = df_bond_price.set_index('trade_date')['bond_ret'].to_dict()
    
    # Use index daily price to get aligned trading dates
    trading_dates = m1000['trade_date'].tolist()
    m1000_dict = m1000.set_index('trade_date').to_dict(orient='index')
    
    rows = []
    for dt in trading_dates:
        row1000 = m1000_dict.get(dt)
        if row1000 is None:
            continue
            
        nasdaq_ret = nasdaq_map.get(dt, 0.0)
        gold_ret = gold_map.get(dt, 0.0)
        bond_ret = bond_map.get(dt, 0.03 / 242.0)  # Default 3% annualized yield for bonds before listing
        
        if pd.isna(bond_ret):
            bond_ret = 0.03 / 242.0
            
        rows.append({
            'trade_date': dt,
            'close_1000': row1000['close_1000'],
            'ret_1000': row1000['pct_chg_1000'] / 100.0,
            'ma_1000': row1000['ma_1000'],
            'val_q_1000': row1000['val_q_1000'],
            'ret_nasdaq': nasdaq_ret,
            'ret_gold': gold_ret,
            'ret_bond': bond_ret
        })
        
    df_unified = pd.DataFrame(rows).sort_values('trade_date').reset_index(drop=True)
    print(f"Aligned premium dataset built. Shape: {df_unified.shape}, Dates: {df_unified['trade_date'].min().strftime('%Y-%m-%d')} to {df_unified['trade_date'].max().strftime('%Y-%m-%d')}")
    return df_unified

def run_premium_backtest(
    df_period,
    val_coeff=0.6,
    W_nasdaq=0.15,
    W_gold=0.10,
    alpha_annual=0.11,
    dev_threshold=0.10,
    initial_capital=1000000.0
):
    if len(df_period) == 0:
        return None
        
    df_period = df_period.copy().reset_index(drop=True)
    df_period['year_week'] = df_period['trade_date'].dt.strftime('%Y-%U')
    rebalance_check_dates = set(df_period.groupby('year_week')['trade_date'].first())
    
    # Model ZZ1000 Enhanced Return (CSI 1000 daily return + annualized alpha)
    # alpha_daily = (1 + alpha_annual)^(1/242) - 1
    alpha_daily = (1.0 + alpha_annual) ** (1.0 / 242.0) - 1.0
    df_period['ret_zz1000_enhanced'] = (1.0 + df_period['ret_1000']) * (1.0 + alpha_daily) - 1.0
    
    # Portfolio holdings value
    val_zz1000 = 0.0
    val_nasdaq = 0.0
    val_gold = 0.0
    val_bond = initial_capital  # Start all in Bond ETF
    
    nav_history = []
    weight_history = []
    
    for idx, row in df_period.iterrows():
        dt = row['trade_date']
        
        # 1. Update holdings values with daily returns
        if idx > 0:
            val_zz1000 *= (1.0 + row['ret_zz1000_enhanced'])
            val_nasdaq *= (1.0 + row['ret_nasdaq'])
            val_gold *= (1.0 + row['ret_gold'])
            val_bond *= (1.0 + row['ret_bond'])
            
        nav = val_zz1000 + val_nasdaq + val_gold + val_bond
        
        # 2. Rebalancing logic
        if dt in rebalance_check_dates or idx == 0:
            Q = row['val_q_1000']
            if pd.isna(Q):
                Q = 0.5  # Fallback to fair valuation if missing
                
            # Valuation-based sizing for A-shares
            W_val = val_coeff * (1.0 - Q)
            
            # Distance from MA250 for 5-tier trend timing
            close_1000 = row['close_1000']
            ma_1000 = row['ma_1000']
            
            if pd.isna(ma_1000):
                M_trend = 1.0  # Fallback if MA not available yet
            else:
                D = (close_1000 - ma_1000) / ma_1000
                
                # Extreme undervaluation exception
                if Q <= 0.15:
                    M_trend = 1.0
                else:
                    # 5-tier trend scaling
                    if D >= 0.05:
                        M_trend = 1.0
                    elif 0.0 <= D < 0.05:
                        M_trend = 0.8
                    elif -0.05 <= D < 0.0:
                        M_trend = 0.6
                    elif -0.10 <= D < -0.05:
                        M_trend = 0.4
                    else:
                        M_trend = 0.3  # 保底配置不清仓
                        
            W_zz1000_timed = W_val * M_trend
            
            # Current weights
            W_curr_zz1000 = val_zz1000 / nav if nav > 0 else 0.0
            W_curr_nasdaq = val_nasdaq / nav if nav > 0 else 0.0
            W_curr_gold = val_gold / nav if nav > 0 else 0.0
            W_curr_bond = val_bond / nav if nav > 0 else 0.0
            
            # 3-tier valuation entry rules
            if Q <= 0.20:
                # Low Valuation -> Lump-Sum Buy
                W_target_zz1000 = W_zz1000_timed
            elif 0.20 < Q <= 0.80:
                # Fair Valuation -> DCA Build-up
                if W_curr_zz1000 < W_zz1000_timed:
                    # Gradually increase position by maximum 5%
                    W_target_zz1000 = min(W_zz1000_timed, W_curr_zz1000 + 0.05)
                else:
                    W_target_zz1000 = W_zz1000_timed  # Direct reduce
            else:
                # High Valuation -> Freeze buying
                if W_curr_zz1000 < W_zz1000_timed:
                    W_target_zz1000 = W_curr_zz1000  # Do not buy
                else:
                    W_target_zz1000 = W_zz1000_timed  # Direct reduce
                    
            # Static weights for US stocks and Gold
            W_target_nasdaq = W_nasdaq
            W_target_gold = W_gold
            
            # Adjust if sum of targets exceeds 100%
            total_equity = W_target_zz1000 + W_target_nasdaq + W_target_gold
            if total_equity > 1.0:
                W_target_zz1000 /= total_equity
                W_target_nasdaq /= total_equity
                W_target_gold /= total_equity
                W_target_bond = 0.0
            else:
                W_target_bond = 1.0 - total_equity
                
            # Check deviation threshold for rebalancing execution
            devs = [
                abs(W_curr_zz1000 - W_target_zz1000),
                abs(W_curr_nasdaq - W_target_nasdaq),
                abs(W_curr_gold - W_target_gold),
                abs(W_curr_bond - W_target_bond)
            ]
            
            if any(d > dev_threshold for d in devs) or idx == 0:
                # Target asset values
                val_target_zz1000 = nav * W_target_zz1000
                val_target_nasdaq = nav * W_target_nasdaq
                val_target_gold = nav * W_target_gold
                val_target_bond = nav * W_target_bond
                
                # Transaction costs: 0.05% for A-shares/bonds, 0.10% for Nasdaq/Gold
                trade_zz1000 = abs(val_target_zz1000 - val_zz1000)
                trade_nasdaq = abs(val_target_nasdaq - val_nasdaq)
                trade_gold = abs(val_target_gold - val_gold)
                trade_bond = abs(val_target_bond - val_bond)
                
                cost = (trade_zz1000 * 0.0005 + 
                        trade_bond * 0.0005 + 
                        trade_nasdaq * 0.0010 + 
                        trade_gold * 0.0010)
                
                nav -= cost
                val_zz1000 = nav * W_target_zz1000
                val_nasdaq = nav * W_target_nasdaq
                val_gold = nav * W_target_gold
                val_bond = nav * W_target_bond
                
        # Record history
        nav_history.append({'trade_date': dt, 'nav': nav})
        weight_history.append({
            'trade_date': dt,
            'w_zz1000': val_zz1000 / nav if nav > 0 else 0.0,
            'w_nasdaq': val_nasdaq / nav if nav > 0 else 0.0,
            'w_gold': val_gold / nav if nav > 0 else 0.0,
            'w_bond': val_bond / nav if nav > 0 else 0.0
        })
        
    df_nav = pd.DataFrame(nav_history).set_index('trade_date')
    df_weights = pd.DataFrame(weight_history).set_index('trade_date')
    return df_nav, df_weights

def compute_metrics(nav_series, initial_capital=1000000.0):
    total_ret = nav_series.iloc[-1] / initial_capital - 1
    years = (nav_series.index[-1] - nav_series.index[0]).days / 365.25
    cagr = (nav_series.iloc[-1] / initial_capital) ** (1.0 / years) - 1 if years > 0 else 0.0
    
    daily_rets = nav_series.pct_change().dropna()
    ann_vol = daily_rets.std() * np.sqrt(252)
    sharpe = (daily_rets.mean() * 252) / ann_vol if ann_vol > 0 else 0.0
    
    cum_max = nav_series.cummax()
    drawdowns = (nav_series - cum_max) / cum_max
    max_dd = drawdowns.min()
    calmar = cagr / abs(max_dd) if max_dd != 0 else 0.0
    
    return {
        'Total Return': total_ret,
        'CAGR': cagr,
        'Volatility': ann_vol,
        'Sharpe': sharpe,
        'Max Drawdown': max_dd,
        'Calmar': calmar
    }

def main():
    # Load unified dataset
    df_unified = load_premium_data(ma_window=250, val_window=1200)
    
    # Split dates
    is_start, is_end = "2015-01-01", "2024-02-05"
    oos_start, oos_end = "2024-02-06", "2026-03-13"
    
    df_is = df_unified[(df_unified['trade_date'] >= pd.to_datetime(is_start)) & (df_unified['trade_date'] <= pd.to_datetime(is_end))].copy()
    df_oos = df_unified[(df_unified['trade_date'] >= pd.to_datetime(oos_start)) & (df_unified['trade_date'] <= pd.to_datetime(oos_end))].copy()
    
    # Run backtests
    print("\nRunning In-Sample Premium Backtest...")
    df_nav_is, df_weights_is = run_premium_backtest(df_is)
    metrics_is = compute_metrics(df_nav_is['nav'])
    
    print("Running Out-of-Sample Premium Backtest...")
    df_nav_oos, df_weights_oos = run_premium_backtest(df_oos)
    metrics_oos = compute_metrics(df_nav_oos['nav'])
    
    # Save results
    df_nav_is.to_csv(os.path.join(RESULTS_DIR, 'nav_premium_is.csv'))
    df_nav_oos.to_csv(os.path.join(RESULTS_DIR, 'nav_premium_oos.csv'))
    df_weights_is.to_csv(os.path.join(RESULTS_DIR, 'weights_premium_is.csv'))
    df_weights_oos.to_csv(os.path.join(RESULTS_DIR, 'weights_premium_oos.csv'))
    
    print("\n" + "="*50)
    print("  PREMIUM STRATEGY PERFORMANCE")
    print("="*50)
    print("In-Sample (2015 - 2024):")
    for k, v in metrics_is.items():
        print(f"  {k:15}: {v:.2%}" if k != 'Sharpe' and k != 'Calmar' else f"  {k:15}: {v:.2f}")
    print("\nOut-of-Sample (2024 - 2026):")
    for k, v in metrics_oos.items():
        print(f"  {k:15}: {v:.2%}" if k != 'Sharpe' and k != 'Calmar' else f"  {k:15}: {v:.2f}")
    print("="*50)

if __name__ == "__main__":
    main()

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
STUDY_DIR = SCRIPT_DIR
RESEARCH_DIR = os.path.dirname(os.path.dirname(STUDY_DIR))
CACHE_DIR = os.path.join(RESEARCH_DIR, 'cache')
RESULTS_DIR = os.path.join(RESEARCH_DIR, 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

def load_data():
    parquet_path = os.path.join(CACHE_DIR, 'cb_pit_daily.parquet')
    if not os.path.exists(parquet_path):
        raise FileNotFoundError(f"PIT data file not found at {parquet_path}. Please run downloader first.")
    df = pd.read_parquet(parquet_path)
    df['trade_date'] = pd.to_datetime(df['trade_date'])
    
    # Sort by code and date to compute rolling factors correctly
    df = df.sort_values(['ts_code', 'trade_date']).reset_index(drop=True)
    
    # Calculate Double-Low score: close price + conversion premium rate
    df['double_low'] = df['close'] + df['premium']
    
    # 20-day momentum of underlying stock (approximated by convert_value return)
    df['stock_mom_20'] = df.groupby('ts_code')['convert_value'].pct_change(20)
    
    # 20-day volatility of underlying stock
    df['stock_ret'] = df.groupby('ts_code')['convert_value'].pct_change()
    df['stock_vol_20'] = df.groupby('ts_code')['stock_ret'].transform(
        lambda x: x.rolling(20).std() * np.sqrt(252)
    )
    
    # Estimated YTM: (118.0 - close) / (close * years_to_maturity)
    # Maturity is assumed 6 years (2191.5 days) from list date
    df['years_to_maturity'] = (df['list_date'] + pd.to_timedelta(2191.5, unit='D') - df['trade_date']).dt.days / 365.25
    df['years_to_maturity'] = df['years_to_maturity'].clip(lower=0.1) # avoid division by zero or negative
    df['ytm'] = (118.0 - df['close']) / (df['close'] * df['years_to_maturity'])
    
    # Distance to strong redemption: 130 - convert_value
    df['dist_redempt'] = 130.0 - df['convert_value']
    
    # Clean up temporary stock return column
    df = df.drop(columns=['stock_ret'])
    
    return df

def run_backtest(
    df_pit,
    N=20,
    rebalance_freq='2W',
    min_rating='A',            # Exclude rating < A
    min_size=1.0,              # Exclude size < 1亿
    min_maturity=0.5,          # Exclude remaining maturity < 6 months
    single_side_cost=0.0005,   # 0.05% commission + slippage
    initial_capital=1000000.0,
    use_multi_factor=True,     # Toggle Multi-Factor strategy vs Baseline Double-Low
    use_risk_control=True,     # Stop-loss and 130 CNY warning line
    use_position_mgmt=True     # Market double-low mean position sizing
):
    df_pit = df_pit.copy()
    
    # Rating rank mapping
    rating_ranks = {'AAA': 6, 'AA+': 5, 'AA': 4, 'AA-': 3, 'A+': 2, 'A': 1}
    min_rating_rank = rating_ranks.get(min_rating, 1)
    
    def get_rating_rank(r):
        if pd.isna(r):
            return 0
        r_str = str(r).upper().strip()
        for key in rating_ranks:
            if r_str.startswith(key):
                return rating_ranks[key]
        return 0
        
    df_pit['rating_rank'] = df_pit['rating'].apply(get_rating_rank)
    
    # Get unique sorted trading dates
    trading_dates = sorted(df_pit['trade_date'].unique())
    df_dates = pd.DataFrame({'trade_date': trading_dates})
    
    # Determine rebalancing dates
    if rebalance_freq == 'W':
        df_dates['group'] = df_dates['trade_date'].dt.strftime('%Y-%U')
    elif rebalance_freq == '2W':
        weeks = df_dates['trade_date'].dt.isocalendar().week
        years = df_dates['trade_date'].dt.isocalendar().year
        df_dates['group'] = years.astype(str) + '_' + (weeks // 2).astype(str)
    elif rebalance_freq == 'M':
        df_dates['group'] = df_dates['trade_date'].dt.strftime('%Y-%m')
    else:
        raise ValueError(f"Invalid rebalance_freq: {rebalance_freq}")
        
    # Rebalance signals are generated at the close of the first day of each group
    rebalance_signal_dates = set(df_dates.groupby('group')['trade_date'].first())
    
    # Portfolio state
    cash = initial_capital
    shares = {} # {code: shares}
    purchase_prices = {} # {code: price}
    last_known_close = {} # {code: last_close}
    
    nav_history = []
    holdings_history = []
    
    # Signal queues
    target_weights = None
    target_position_coef = 1.0
    
    for idx, dt in enumerate(trading_dates):
        df_today = df_pit[df_pit['trade_date'] == dt].set_index('ts_code')
        
        # 1. Update valuation and execute intraday/close Risk Controls (Stop-loss & 130 CNY Warning Line)
        val_holdings_close = 0.0
        for code, sh in list(shares.items()):
            if code in df_today.index:
                close_price = df_today.loc[code, 'close']
                
                # Check stop-loss (5% drop from purchase price) and warning line (price >= 130)
                purch_price = purchase_prices.get(code, close_price)
                is_stop_loss = use_risk_control and (close_price <= 0.95 * purch_price)
                is_warning_line = use_risk_control and (close_price >= 130.0)
                
                if is_stop_loss or is_warning_line:
                    # Liquidate immediately at today's close price
                    cash += sh * close_price * (1.0 - single_side_cost)
                    del shares[code]
                    if code in purchase_prices:
                        del purchase_prices[code]
                    reason = "Stop-Loss" if is_stop_loss else "Redemption Warning (>=130)"
                    # Optional log statement (silent to avoid output cluttering)
                else:
                    val_holdings_close += sh * close_price
                    last_known_close[code] = close_price
            else:
                # Delisting handling
                close_price = last_known_close.get(code, 100.0)
                cash += sh * close_price * (1.0 - single_side_cost)
                del shares[code]
                if code in purchase_prices:
                    del purchase_prices[code]
                
        nav_close = val_holdings_close + cash
        nav_history.append({'trade_date': dt, 'nav': nav_close, 'cash': cash})
        
        # Keep track of close prices
        for code in df_today.index:
            last_known_close[code] = df_today.loc[code, 'close']
            
        # Record holdings
        holdings_history.append({
            'trade_date': dt,
            'holdings': {code: sh * df_today.loc[code, 'close'] / nav_close if code in df_today.index else 0.0 for code, sh in shares.items()}
        })
        
        # 2. Rebalance Execution (at today's close using yesterday's signal target_weights)
        if target_weights is not None:
            # Liquidate current holdings at close
            cash_temp = cash
            for code, sh in list(shares.items()):
                if code in df_today.index:
                    price = df_today.loc[code, 'close']
                else:
                    price = last_known_close.get(code, 100.0)
                cash_temp += sh * price * (1.0 - single_side_cost)
                
            # Apply position management sizing
            cash_allocated = cash_temp * target_position_coef
            
            # Buy target bonds
            shares = {}
            purchase_prices = {}
            for code, weight in target_weights.items():
                if code in df_today.index:
                    price = df_today.loc[code, 'close']
                    target_value = cash_allocated * weight
                    shares[code] = target_value / (price * (1.0 + single_side_cost))
                    purchase_prices[code] = price
                    
            # Remaining cash is the unallocated portion plus change
            cash = cash_temp - sum(sh * df_today.loc[code, 'close'] * (1.0 + single_side_cost) for code, sh in shares.items())
            target_weights = None
            
        # 3. Rebalance Signal Generation (at T close for execution at T+1 close)
        if dt in rebalance_signal_dates:
            # Calculate market double-low mean
            market_active = df_today.dropna(subset=['close', 'premium']).copy()
            market_dl_mean = market_active['double_low'].mean() if not market_active.empty else 100.0
            
            # Step 4: Position Sizing based on market mean
            if use_position_mgmt:
                if market_dl_mean < 120.0:
                    target_position_coef = 1.0
                elif market_dl_mean <= 140.0:
                    target_position_coef = 0.5
                else:
                    target_position_coef = 0.0
            else:
                target_position_coef = 1.0
                
            # Step 1: Apply filtering pool
            df_active = market_active.copy()
            # Exclude ST正股
            df_active = df_active[~df_active['stock_name'].str.contains('ST', na=False)]
            # Exclude size < 1亿
            if min_size > 0:
                df_active = df_active[df_active['issue_size'] >= min_size]
            # Exclude maturity < 6 months
            if min_maturity > 0:
                df_active = df_active[df_active['years_to_maturity'] >= min_maturity]
            # Exclude rating < A
            df_active = df_active[df_active['rating_rank'] >= min_rating_rank]
            
            # Select target assets
            if not df_active.empty and target_position_coef > 0:
                if use_multi_factor:
                    # Step 2: Multi-Factor scoring
                    r_dl = df_active['double_low'].rank(pct=True, ascending=True)
                    r_prem = df_active['premium'].rank(pct=True, ascending=True)
                    
                    mom_filled = df_active['stock_mom_20'].fillna(df_active['stock_mom_20'].median())
                    vol_filled = df_active['stock_vol_20'].fillna(df_active['stock_vol_20'].median())
                    
                    r_mom = mom_filled.rank(pct=True, ascending=False)
                    r_vol = vol_filled.rank(pct=True, ascending=True)
                    r_scale = df_active['issue_size'].rank(pct=True, ascending=True)
                    
                    r_ytm = df_active['ytm'].rank(pct=True, ascending=False)
                    r_dist = df_active['dist_redempt'].rank(pct=True, ascending=False)
                    
                    df_active['score'] = (
                        0.30 * r_dl + 
                        0.30 * r_prem + 
                        0.10 * r_mom + 
                        0.10 * r_vol + 
                        0.10 * r_scale + 
                        0.05 * r_ytm + 
                        0.05 * r_dist
                    )
                    df_selected = df_active.sort_values('score').head(N)
                else:
                    # Simple Double-Low sort
                    df_selected = df_active.sort_values('double_low').head(N)
                    
                target_codes = df_selected.index.tolist()
                target_weights = {code: 1.0 / len(target_codes) for code in target_codes}
            else:
                target_weights = {}
                
    df_nav = pd.DataFrame(nav_history).set_index('trade_date')
    return df_nav, holdings_history

def compute_metrics(nav_series, initial_capital=1000000.0):
    total_ret = nav_series.iloc[-1] / initial_capital - 1
    days = (nav_series.index[-1] - nav_series.index[0]).days
    years = days / 365.25
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
    try:
        df_pit = load_data()
        print(f"Loaded PIT data with rolling factors. Total rows: {len(df_pit)}")
        
        # Test backtest on new strategy
        print("Running Multi-Factor Strategy test backtest...")
        df_nav, _ = run_backtest(df_pit, N=20, rebalance_freq='2W')
        metrics = compute_metrics(df_nav['nav'])
        
        print("\n" + "="*50)
        print("  MULTI-FACTOR STRATEGY RESULTS (N=20, 2W)")
        print("="*50)
        for k, v in metrics.items():
            print(f"  {k:15}: {v:.2%}" if k != 'Sharpe' and k != 'Calmar' else f"  {k:15}: {v:.2f}")
        print("="*50)
        
    except Exception as e:
        print(f"Error running backtest: {e}")

if __name__ == '__main__':
    main()

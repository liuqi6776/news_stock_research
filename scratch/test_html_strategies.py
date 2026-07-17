import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import warnings

warnings.filterwarnings('ignore')

# Define paths
DATA_DIR = r"c:\Users\liuqi\quant_system_v2\etf-valuation-strategy\data"
RESULTS_DIR = r"c:\Users\liuqi\quant_system_v2\etf-valuation-strategy\results"
os.makedirs(RESULTS_DIR, exist_ok=True)

def bs_put_price(S, K, T, r, sigma):
    if T <= 0:
        return max(K - S, 0.0)
    if S <= 0 or K <= 0 or sigma <= 0:
        return 0.0
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    return max(price, 0.0)

def load_data():
    files = {
        'hs300': 'hs300_daily.csv',
        'zz500': 'zz500_daily.csv',
        'gold': 'gold_etf_daily.csv',
        'nasdaq': 'nasdaq_etf_daily.csv',
        'bond': 'bond_etf_daily.csv'
    }
    
    price_dfs = {}
    for name, fname in files.items():
        path = os.path.join(DATA_DIR, fname)
        df = pd.read_csv(path)
        df['trade_date'] = pd.to_datetime(df['trade_date'].astype(str))
        df.sort_values('trade_date', inplace=True)
        df.reset_index(drop=True, inplace=True)
        
        # Adjust Nasdaq ETF split
        if name == 'nasdaq':
            split_date = pd.to_datetime('2022-01-14')
            adj_factor = 1.038 / 5.192
            mask = df['trade_date'] < split_date
            for col in ['close', 'open', 'high', 'low', 'pre_close']:
                df.loc[mask, col] *= adj_factor
                
        df['ret'] = df['pct_chg'] / 100.0
        df['vol'] = df['ret'].rolling(window=60, min_periods=20).std()
        df['ma'] = df['close'].rolling(window=200, min_periods=20).mean()
        price_dfs[name] = df
        
    val_files = {
        'hs300': 'hs300_valuation.csv',
        'zz500': 'zz500_valuation.csv'
    }
    
    df_bond_yield = pd.read_csv(os.path.join(DATA_DIR, 'bond_yield_china.csv'))
    df_bond_yield['trade_date'] = pd.to_datetime(df_bond_yield['trade_date'].astype(str))
    df_bond_yield.sort_values('trade_date', inplace=True)
    
    val_dfs = {}
    for name, fname in val_files.items():
        path = os.path.join(DATA_DIR, fname)
        df = pd.read_csv(path)
        df['trade_date'] = pd.to_datetime(df['trade_date'].astype(str))
        df.sort_values('trade_date', inplace=True)
        df = pd.merge(df, df_bond_yield, on='trade_date', how='left').ffill()
        df['erp'] = 100.0 / df['pe_ttm'] - df['yield_10y']
        df['erp_rank_q'] = df['erp'].rolling(window=1400, min_periods=250).rank(pct=True)
        df['val_q_erp'] = 1.0 - df['erp_rank_q']
        val_dfs[name] = df
        
    qvix_path = os.path.join(DATA_DIR, 'qvix_daily.csv')
    if os.path.exists(qvix_path):
        df_qvix = pd.read_csv(qvix_path)
        df_qvix['trade_date'] = pd.to_datetime(df_qvix['date'].astype(str))
        df_qvix = df_qvix[['trade_date', 'close']].rename(columns={'close': 'qvix'})
    else:
        df_qvix = pd.DataFrame(columns=['trade_date', 'qvix'])
        
    df_unified = price_dfs['hs300'][['trade_date', 'close', 'ret', 'vol', 'ma']].rename(
        columns={'close': 'close_hs300', 'ret': 'ret_hs300', 'vol': 'vol_hs300', 'ma': 'ma_hs300'}
    )
    
    for name in ['zz500', 'gold', 'nasdaq', 'bond']:
        df_asset = price_dfs[name][['trade_date', 'close', 'ret', 'vol', 'ma']].rename(
            columns={'close': f'close_{name}', 'ret': f'ret_{name}', 'vol': f'vol_{name}', 'ma': f'ma_{name}'}
        )
        df_unified = pd.merge(df_unified, df_asset, on='trade_date', how='inner')
        
    for name in ['hs300', 'zz500']:
        df_val = val_dfs[name][['trade_date', 'val_q_erp', 'yield_10y']].rename(
            columns={'val_q_erp': f'val_q_{name}', 'yield_10y': f'yield_10y_{name}'}
        )
        df_unified = pd.merge(df_unified, df_val, on='trade_date', how='inner')
        
    df_unified = pd.merge(df_unified, df_qvix, on='trade_date', how='left')
    df_unified['qvix'] = df_unified['qvix'].ffill().bfill().fillna(20.0)
    
    df_unified.sort_values('trade_date', inplace=True)
    df_unified.reset_index(drop=True, inplace=True)
    return df_unified

def run_backtest(
    df,
    vol_target=None,
    val_tilt=0.0,
    buy_put=False,
    trend_filter=False,
    equity_boost=None,
    initial_capital=1000000.0,
    execution_cost=0.0005,
    rf_rate=0.02
):
    df = df.copy().reset_index(drop=True)
    assets = ['hs300', 'zz500', 'gold', 'nasdaq', 'bond']
    equity_assets = ['hs300', 'zz500']
    
    df['year_week'] = df['trade_date'].dt.strftime('%Y-%U')
    rebalance_dates = set(df.groupby('year_week')['trade_date'].first())
    
    ret_cols = [f'ret_{a}' for a in assets]
    df_returns = df[ret_cols].copy()
    df_returns.columns = assets
    
    val = {a: 0.0 for a in assets}
    val_cash = initial_capital
    options_held = []
    nav_history = []
    
    for idx, row in df.iterrows():
        dt = row['trade_date']
        
        # 1. Update holdings values
        if idx > 0:
            for a in assets:
                val[a] *= (1.0 + row[f'ret_{a}'])
            val_cash *= (1.0 + rf_rate / 242.0)
            
        # 2. Check option payoff
        payoff_today = 0.0
        active_options = []
        for opt in options_held:
            if idx >= opt['expiry_idx']:
                asset = opt['asset']
                close_price = row[f'close_{asset}']
                purchase_price = opt['purchase_price']
                strike_price = opt['strike_price']
                payoff = opt['purchase_val'] * max(strike_price / purchase_price - close_price / purchase_price, 0.0)
                payoff_today += payoff
            else:
                active_options.append(opt)
        options_held = active_options
        val_cash += payoff_today
        
        nav = sum(val.values()) + val_cash
        
        # 3. Target weights
        if equity_boost is not None:
            # Config 4: boost A-shares to 90%, bond/gold/nasdaq to 10% total
            w_target = {'hs300': 0.45, 'zz500': 0.45, 'gold': 0.033, 'nasdaq': 0.033, 'bond': 0.034}
        else:
            vols = np.array([row[f'vol_{a}'] for a in assets])
            vols = np.where(vols <= 0, 1e-4, vols)
            inv_vols = 1.0 / vols
            w_rp = inv_vols / inv_vols.sum()
            w_target = {a: w_rp[i] for i, a in enumerate(assets)}
            
        # Valuation overlay
        if val_tilt > 0.0:
            for a in equity_assets:
                val_q = row[f'val_q_{a}']
                if not pd.isna(val_q):
                    w_target[a] *= (1.0 - val_tilt * (val_q - 0.5))
                    
        # Trend filter
        if trend_filter:
            for a in assets:
                close_px = row[f'close_{a}']
                ma_px = row[f'ma_{a}']
                trend_up = close_px >= ma_px if not pd.isna(ma_px) else True
                if not trend_up:
                    if a in equity_assets:
                        val_q = row[f'val_q_{a}']
                        if pd.isna(val_q) or val_q > 0.20:
                            w_target[a] *= 0.5
                    elif a in ['gold', 'nasdaq']:
                        w_target[a] *= 0.5
                        
        # Normalize weights
        w_sum = sum(w_target.values())
        for a in assets:
            w_target[a] /= w_sum
            
        # Vol target layer
        if vol_target is not None:
            if idx >= 60:
                cov_matrix = df_returns.iloc[idx - 60 + 1:idx + 1].cov().values
            else:
                cov_matrix = df_returns.iloc[0:idx + 1].cov().values if idx > 5 else np.eye(len(assets)) * (0.01 / 252.0)
            w_vector = np.array([w_target[a] for a in assets])
            port_variance = np.dot(w_vector, np.dot(cov_matrix, w_vector))
            port_vol = np.sqrt(port_variance * 252.0)
            
            sf = min(1.0, vol_target / max(port_vol, 1e-6))
            w_target_final = {a: w_target[a] * sf for a in assets}
            w_target_final['cash'] = 1.0 - sf
        else:
            w_target_final = {a: w_target[a] for a in assets}
            w_target_final['cash'] = 0.0
            
        # 4. Rebalance
        w_curr = {a: val[a] / nav if nav > 0 else 0.0 for a in assets}
        w_curr['cash'] = val_cash / nav if nav > 0 else 0.0
        devs = [abs(w_curr[a] - w_target_final[a]) for a in assets] + [abs(w_curr['cash'] - w_target_final['cash'])]
        max_dev = max(devs)
        
        is_rebal_day = (dt in rebalance_dates) or (max_dev > 0.10) or (idx == 0)
        if is_rebal_day:
            val_target = {a: nav * w_target_final[a] for a in assets}
            val_target_cash = nav * w_target_final['cash']
            trade_vol = sum(abs(val_target[a] - val[a]) for a in assets) + abs(val_target_cash - val_cash)
            cost = trade_vol * execution_cost
            nav -= cost
            val_cash = nav * w_target_final['cash']
            for a in assets:
                val[a] = nav * w_target_final[a]
                
        # 5. Buy Put Options
        if buy_put and (idx % 20 == 0):
            T_years = 20.0 / 252.0
            for a in equity_assets:
                val_q = row[f'val_q_{a}']
                val_holding = val[a]
                if (not pd.isna(val_q)) and (val_q > 0.70) and (val_holding > 0.0):
                    S0 = row[f'close_{a}']
                    K = S0 * 0.95
                    r = (row[f'yield_10y_{a}'] / 100.0) if not pd.isna(row[f'yield_10y_{a}']) else 0.025
                    current_iv = row[f'vol_{a}'] * np.sqrt(252.0)
                    if pd.isna(current_iv) or current_iv <= 0:
                        current_iv = 0.20
                        
                    put_price_per_share = bs_put_price(S0, K, T_years, r, current_iv)
                    pct_premium = put_price_per_share / S0
                    opt_premium_cost = val_holding * pct_premium
                    
                    val_cash -= opt_premium_cost
                    nav -= opt_premium_cost
                    options_held.append({
                        'expiry_idx': idx + 20,
                        'asset': a,
                        'purchase_val': val_holding,
                        'purchase_price': S0,
                        'strike_price': K
                    })
                    
        nav_history.append({'trade_date': dt, 'nav': nav})
        
    return pd.DataFrame(nav_history).set_index('trade_date')

def run_equal_weight_backtest(df, initial_capital=1000000.0, execution_cost=0.0005):
    df = df.copy().reset_index(drop=True)
    assets = ['hs300', 'zz500', 'gold', 'nasdaq', 'bond']
    df['year_week'] = df['trade_date'].dt.strftime('%Y-%U')
    rebalance_dates = set(df.groupby('year_week')['trade_date'].first())
    
    val = {a: initial_capital / len(assets) for a in assets}
    nav_history = []
    
    for idx, row in df.iterrows():
        dt = row['trade_date']
        if idx > 0:
            for a in assets:
                val[a] *= (1.0 + row[f'ret_{a}'])
        nav = sum(val.values())
        
        if dt in rebalance_dates:
            trade_vol = sum(abs(nav / len(assets) - val[a]) for a in assets)
            cost = trade_vol * execution_cost
            nav -= cost
            val = {a: nav / len(assets) for a in assets}
            
        nav_history.append({'trade_date': dt, 'nav': nav})
    return pd.DataFrame(nav_history).set_index('trade_date')

def run_leverage_simulation(nav_df, leverage_factor=2.5, financing_rate=0.045):
    daily_rets = nav_df['nav'].pct_change().fillna(0.0)
    daily_financing = financing_rate / 252.0
    leveraged_rets = leverage_factor * daily_rets - (leverage_factor - 1.0) * daily_financing
    leveraged_nav = (1.0 + leveraged_rets).cumprod() * 1000000.0
    leveraged_nav.index = nav_df.index
    return pd.DataFrame({'nav': leveraged_nav})

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
    return cagr, ann_vol, sharpe, max_dd

def main():
    print("Loading data...")
    df = load_data()
    
    # Filter full period (2018-01-03 to 2026-03-13)
    df_backtest = df[(df['trade_date'] >= '2018-01-03') & (df['trade_date'] <= '2026-03-13')].copy()
    print(f"Backtest period: {df_backtest['trade_date'].min().strftime('%Y-%m-%d')} to {df_backtest['trade_date'].max().strftime('%Y-%m-%d')}")
    
    # Run backtests
    print("\n[Running Config 1] Risk Parity (No Vol Target, No Timing, No Puts, No Trend)")
    c1_nav = run_backtest(df_backtest, vol_target=None, val_tilt=0.0, buy_put=False, trend_filter=False)
    
    print("[Running Config 2] Risk Parity + Vol Target (10%)")
    c2_nav = run_backtest(df_backtest, vol_target=0.10, val_tilt=0.0, buy_put=False, trend_filter=False)
    
    print("[Running Config 3] Risk Parity + Vol Target + Timing + Puts + Trend")
    c3_nav = run_backtest(df_backtest, vol_target=0.10, val_tilt=0.4, buy_put=True, trend_filter=True)
    
    print("[Running Config 4] Equity-Heavy (90% Equities)")
    c4_nav = run_backtest(df_backtest, vol_target=None, val_tilt=0.0, buy_put=False, trend_filter=False, equity_boost=True)
    
    print("[Running Config 5] Leveraged 2.5x Portfolio")
    c5_nav = run_leverage_simulation(c3_nav, leverage_factor=2.5, financing_rate=0.045)
    
    print("[Running Config 6] 5-Asset Equal Weight")
    c6_nav = run_equal_weight_backtest(df_backtest)
    
    print("[Running Config 7] Pure HS300 ETF")
    c7_nav = pd.DataFrame(index=df_backtest['trade_date'])
    c7_nav['nav'] = ((df_backtest['ret_hs300'] + 1.0).cumprod() * 1000000.0).values

    
    # Calculate metrics
    configs = {
        '风险平价(无杠杆)': c1_nav,
        '+ 波动目标提到10%': c2_nav,
        '+ 估值择时 + Put': c3_nav,
        '减国债 / 增A股权益(90%)': c4_nav,
        '整体加杠杆 2.5x': c5_nav,
        '5资产等权买入持有': c6_nav,
        '纯沪深300 ETF': c7_nav
    }
    
    results = []
    for name, nav_df in configs.items():
        cagr, vol, sharpe, mdd = compute_metrics(nav_df['nav'])
        results.append({
            'Strategy': name,
            'CAGR': f"{cagr:.1%}",
            'Volatility': f"{vol:.1%}",
            'Sharpe': f"{sharpe:.2f}",
            'Max Drawdown': f"{mdd:.1%}"
        })
        
    df_results = pd.DataFrame(results)
    print("\n" + "="*80)
    print("  VERIFIED HTML TABLE BACKTEST RESULTS (FULL PERIOD: 2018-2026)")
    print("="*80)
    print(df_results.to_string(index=False))
    print("="*80)
    
    # Save CSV
    out_path = os.path.join(RESULTS_DIR, 'html_strategies_test_results.csv')
    df_results.to_csv(out_path, index=False)
    print(f"\nSaved verified metrics to: {out_path}")
    
    # Generate Chart
    plt.figure(figsize=(14, 8))
    for name, nav_df in configs.items():
        plt.plot(nav_df.index, nav_df['nav'] / 1e6, label=name, linewidth=1.5)
    plt.title("Quant Engine verified: 7 HTML Strategy Configurations NAV Comparison (2018-2026)", fontsize=12, fontweight='bold')
    plt.ylabel("Normalized NAV")
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    chart_path = os.path.join(RESULTS_DIR, 'html_strategies_comparison.png')
    plt.savefig(chart_path, dpi=300)
    plt.close()
    print(f"Saved comparison chart to: {chart_path}")

if __name__ == "__main__":
    main()

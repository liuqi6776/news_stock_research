import os
import pandas as pd
import numpy as np
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Define paths
DATA_DIR = r"c:\Users\liuqi\quant_system_v2\etf-valuation-strategy\data"
SCRIPTS_DIR = r"c:\Users\liuqi\quant_system_v2\etf-valuation-strategy\scripts"
sys.path.append(SCRIPTS_DIR)

import step11_risk_parity_strategy as rp

# Load daily stock PNL
pnl_file = os.path.join(DATA_DIR, 'daily_super_weekly_pnl.csv')
if not os.path.exists(pnl_file):
    print(f"Error: {pnl_file} not found. Please run generate_daily_super_weekly.py first.")
    sys.exit(1)

df_stock = pd.read_csv(pnl_file)
df_stock['trade_date'] = pd.to_datetime(df_stock['trade_date'].astype(str))

df_unified = rp.load_data_8assets(ma_window=200, val_window=1400, vol_lookback=60)

# Merge daily stock strategy PNL
df_all = pd.merge(df_unified, df_stock[['trade_date', 'pnl']], on='trade_date', how='inner')
print(f"Merged dataset shape: {df_all.shape}, Dates: {df_all['trade_date'].min().strftime('%Y-%m-%d')} to {df_all['trade_date'].max().strftime('%Y-%m-%d')}")

# 1. Compute correlations
assets = ['hs300', 'zz500', 'chinext', 'div', 'gold', 'nasdaq', 'bond', 'cbond']
data_rets = {a: df_all[f'ret_{a}'] for a in assets}
data_rets['active_stock'] = df_all['pnl']

df_rets = pd.DataFrame(data_rets)
corr = df_rets.corr()
print("\nCorrelation matrix including Super-Weekly Strategy:")
print(corr[['active_stock']].round(3))

# 2. Define Replacement Active Equity Portfolio Backtest
def run_backtest_replaced_equity(df_period, vol_target=0.10):
    df_period = df_period.copy().reset_index(drop=True)
    active_assets = ['active_stock', 'gold', 'nasdaq', 'bond', 'cbond']
    
    # Scale active stock picker return (1.0x unscaled)
    df_period['ret_active_stock'] = df_period['pnl']
    df_period['vol_active_stock'] = df_period['ret_active_stock'].rolling(window=60, min_periods=20).std().bfill()
    
    df_period['year_week'] = df_period['trade_date'].dt.strftime('%Y-%U')
    rebalance_check_dates = set(df_period.groupby('year_week')['trade_date'].first())
    
    ret_cols = ['ret_active_stock', 'ret_gold', 'ret_nasdaq', 'ret_bond', 'ret_cbond']
    df_returns = df_period[ret_cols].copy()
    df_returns.columns = active_assets
    
    val = {a: 0.0 for a in active_assets}
    val_cash = 1000000.0
    nav_history = []
    
    for idx, row in df_period.iterrows():
        dt = row['trade_date']
        
        if idx > 0:
            for a in active_assets:
                val[a] *= (1.0 + row[f'ret_{a}'])
            val_cash *= (1.0 + 0.02 / 242.0)
            
        nav = sum(val.values()) + val_cash
        
        # Risk Parity weights
        vols = np.array([row[f'vol_{a}'] for a in active_assets])
        vols = np.where(vols <= 0, 1e-4, vols)
        inv_vols = 1.0 / vols
        w_rp = inv_vols / inv_vols.sum()
        w_target = {a: w_rp[i] for i, a in enumerate(active_assets)}
        
        # Vol target layer
        if idx >= 60:
            cov_matrix = df_returns.iloc[idx - 60 + 1:idx + 1].cov().values
        else:
            cov_matrix = df_returns.iloc[0:idx + 1].cov().values if idx > 5 else np.eye(len(active_assets)) * (0.01 / 252.0)
            
        w_vector = np.array([w_target[a] for a in active_assets])
        port_variance = np.dot(w_vector, np.dot(cov_matrix, w_vector))
        port_vol = np.sqrt(port_variance * 252.0)
        
        sf = min(1.0, vol_target / max(port_vol, 1e-6))
        w_target_final = {a: w_target[a] * sf for a in active_assets}
        w_target_final['cash'] = 1.0 - sf
        
        w_curr = {a: val[a] / nav if nav > 0 else 0.0 for a in active_assets}
        w_curr['cash'] = val_cash / nav if nav > 0 else 0.0
        devs = [abs(w_curr[a] - w_target_final[a]) for a in active_assets] + [abs(w_curr['cash'] - w_target_final['cash'])]
        max_dev = max(devs)
        
        is_rebal_day = (dt in rebalance_check_dates) or (max_dev > 0.05) or (idx == 0)
        
        if is_rebal_day:
            val_target = {a: nav * w_target_final[a] for a in active_assets}
            val_target_cash = nav * w_target_final['cash']
            trade_vol = sum(abs(val_target[a] - val[a]) for a in active_assets) + abs(val_target_cash - val_cash)
            cost = trade_vol * 0.0005
            nav -= cost
            val_cash = nav * w_target_final['cash']
            for a in active_assets:
                val[a] = nav * w_target_final[a]
                
        nav_history.append({'trade_date': dt, 'nav': nav})
        
    df_nav = pd.DataFrame(nav_history).set_index('trade_date')
    return df_nav

# 3. Define Active Satellite 9-Asset Risk Parity Backtest
def run_backtest_9assets(
    df_period,
    vol_target=0.10,
    val_tilt=0.0,
    q_threshold=0.20,
    strike_ratio=0.95,
    trend_reduce_factor=0.5,
    vol_lookback=60,
    dev_threshold=0.05,
    initial_capital=1000000.0,
    rf_rate=0.02,
    buy_put=True
):
    if len(df_period) == 0:
        return None
        
    df_period = df_period.copy().reset_index(drop=True)
    
    # 9 Assets list
    assets = ['hs300', 'zz500', 'chinext', 'div', 'gold', 'nasdaq', 'bond', 'cbond', 'active_stock']
    equity_assets = ['hs300', 'zz500', 'chinext', 'div']
    
    df_period['ret_active_stock'] = df_period['pnl']
    df_period['vol_active_stock'] = df_period['ret_active_stock'].rolling(window=vol_lookback, min_periods=20).std().bfill()
    
    df_period['year_week'] = df_period['trade_date'].dt.strftime('%Y-%U')
    rebalance_check_dates = set(df_period.groupby('year_week')['trade_date'].first())
    
    ret_cols = [f'ret_{a}' for a in assets]
    df_returns = df_period[ret_cols].copy()
    df_returns.columns = assets
    
    val = {a: 0.0 for a in assets}
    val_cash = initial_capital
    options_held = []
    nav_history = []
    
    for idx, row in df_period.iterrows():
        dt = row['trade_date']
        
        if idx > 0:
            for a in assets:
                val[a] *= (1.0 + row[f'ret_{a}'])
            val_cash *= (1.0 + rf_rate / 242.0)
            
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
        
        # Risk Parity weights
        vols = np.array([row[f'vol_{a}'] for a in assets])
        vols = np.where(vols <= 0, 1e-4, vols)
        inv_vols = 1.0 / vols
        w_rp = inv_vols / inv_vols.sum()
        w_target = {a: w_rp[i] for i, a in enumerate(assets)}
        
        # Valuation Timing (only A-shares)
        for a in equity_assets:
            val_q = row[f'val_q_{a}']
            if not pd.isna(val_q):
                w_target[a] *= (1.0 - val_tilt * (val_q - 0.5))
                
        # Trend filter (skip for active_stock)
        for a in assets:
            if a == 'active_stock':
                continue
            close_px = row[f'close_{a}']
            ma_px = row[f'ma_{a}']
            trend_up = close_px >= ma_px if not pd.isna(ma_px) else True
            
            if not trend_up:
                if a in equity_assets:
                    val_q = row[f'val_q_{a}']
                    if pd.isna(val_q) or val_q > q_threshold:
                        w_target[a] *= trend_reduce_factor
                else:
                    if a in ['gold', 'nasdaq']:
                        w_target[a] *= trend_reduce_factor
                        
        # Normalize weights
        w_sum = sum(w_target.values())
        for a in assets:
            w_target[a] /= w_sum
            
        # Vol target layer
        if idx >= vol_lookback:
            cov_matrix = df_returns.iloc[idx - vol_lookback + 1:idx + 1].cov().values
        else:
            cov_matrix = df_returns.iloc[0:idx + 1].cov().values if idx > 5 else np.eye(len(assets)) * (0.01 / 252.0)
            
        w_vector = np.array([w_target[a] for a in assets])
        port_variance = np.dot(w_vector, np.dot(cov_matrix, w_vector))
        port_vol = np.sqrt(port_variance * 252.0)
        
        sf = min(1.0, vol_target / max(port_vol, 1e-6))
        w_target_final = {a: w_target[a] * sf for a in assets}
        w_target_final['cash'] = 1.0 - sf
        
        w_curr = {a: val[a] / nav if nav > 0 else 0.0 for a in assets}
        w_curr['cash'] = val_cash / nav if nav > 0 else 0.0
        devs = [abs(w_curr[a] - w_target_final[a]) for a in assets] + [abs(w_curr['cash'] - w_target_final['cash'])]
        max_dev = max(devs)
        
        is_rebal_day = (dt in rebalance_check_dates) or (max_dev > dev_threshold) or (idx == 0)
        
        if is_rebal_day:
            val_target = {a: nav * w_target_final[a] for a in assets}
            val_target_cash = nav * w_target_final['cash']
            trade_vol = sum(abs(val_target[a] - val[a]) for a in assets) + abs(val_target_cash - val_cash)
            cost = trade_vol * 0.0005
            nav -= cost
            val_cash = nav * w_target_final['cash']
            for a in assets:
                val[a] = nav * w_target_final[a]
                
        # Put options buying
        if buy_put and (idx % 20 == 0):
            T_years = 20.0 / 252.0
            for a in equity_assets:
                val_q = row[f'val_q_{a}']
                val_holding = val[a]
                if (not pd.isna(val_q)) and (val_q > 0.70) and (val_holding > 0.0):
                    S0 = row[f'close_{a}']
                    K = S0 * strike_ratio
                    r = (row[f'yield_10y_{a}'] / 100.0) if not pd.isna(row[f'yield_10y_{a}']) else 0.025
                    current_iv = (row['qvix'] / 100.0) if 'qvix' in row else row[f'vol_{a}'] * np.sqrt(252.0)
                    if pd.isna(current_iv) or current_iv <= 0:
                        current_iv = 0.20
                    put_price_per_share = rp.bs_put_price(S0, K, T_years, r, current_iv)
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
        
    df_nav = pd.DataFrame(nav_history).set_index('trade_date')
    return df_nav

def evaluate_and_plot():
    # 1. Run Baseline (Option A)
    print("1. Running Baseline 8-asset Risk Parity...")
    df_nav_base, _, _ = rp.run_backtest_risk_parity(
        df_all, vol_target=0.10, val_tilt=0.0, strike_ratio=0.95, buy_put=True
    )
    
    # 2. Run Active Replacement (Option B)
    print("2. Running Active Replacement (5-Asset)...")
    df_nav_replace = run_backtest_replaced_equity(df_all, vol_target=0.10)
    
    # 3. Run Active Satellite (Option C)
    print("3. Running Active Satellite (9-Asset)...")
    df_nav_satellite = run_backtest_9assets(
        df_all, vol_target=0.10, val_tilt=0.0, strike_ratio=0.95, buy_put=True
    )
    
    # 4. Run Core-Satellite (Option D)
    print("4. Running Core-Satellite Portfolios...")
    base_rets = df_nav_base['nav'].pct_change().fillna(0)
    stock_rets = df_all.set_index('trade_date')['pnl']
    
    df_nav_cs10 = pd.DataFrame(index=df_nav_base.index)
    df_nav_cs10['nav'] = (0.90 * base_rets + 0.10 * stock_rets).add(1.0).cumprod() * 1000000.0
    
    df_nav_cs20 = pd.DataFrame(index=df_nav_base.index)
    df_nav_cs20['nav'] = (0.80 * base_rets + 0.20 * stock_rets).add(1.0).cumprod() * 1000000.0

    # Segments
    is_start, is_end = "2018-01-05", "2024-02-05"
    oos_start, oos_end = "2024-02-06", "2026-03-11"
    
    strategies = {
        'Baseline 8-Asset RP': df_nav_base,
        'Active Replacement': df_nav_replace,
        'Active Satellite': df_nav_satellite,
        'Core-Satellite 10%': df_nav_cs10,
        'Core-Satellite 20%': df_nav_cs20
    }
    
    # Print function for segment
    def print_metrics_table(start_dt, end_dt, name):
        print("\n" + "="*95)
        print(f"  PERFORMANCE SUMMARY: {name} ({start_dt} to {end_dt})")
        print("="*95)
        print(f"{'Strategy Name':<28s} | {'Total Ret':<9s} {'CAGR':<8s} {'Vol':<7s} {'Sharpe':<7s} {'MDD':<8s} {'Calmar':<6s}")
        print("-" * 95)
        for s_name, df_nav in strategies.items():
            mask = (df_nav.index >= pd.Timestamp(start_dt)) & (df_nav.index <= pd.Timestamp(end_dt))
            nav_s = df_nav['nav'][mask]
            if len(nav_s) < 2: continue
            
            # Re-scale segment start to 1,000,000 to compute segment returns
            seg_nav = nav_s / nav_s.iloc[0] * 1000000.0
            
            # Compute segment metrics
            years = (nav_s.index[-1] - nav_s.index[0]).days / 365.25
            cagr = (seg_nav.iloc[-1] / 1000000.0) ** (1.0 / years) - 1.0 if years > 0 else 0.0
            total_ret = seg_nav.iloc[-1] / 1000000.0 - 1.0
            daily_rets = seg_nav.pct_change().dropna()
            ann_vol = daily_rets.std() * np.sqrt(252)
            sharpe = cagr / ann_vol if ann_vol > 0 else 0.0
            cum_max = seg_nav.cummax()
            mdd = ((seg_nav - cum_max) / cum_max).min()
            calmar = cagr / abs(mdd) if mdd != 0 else 0.0
            
            print(f"{s_name:<28s} | {total_ret:<9.2%} {cagr:<8.2%} {ann_vol:<7.2%} {sharpe:<7.2f} {mdd:<8.2%} {calmar:<6.2f}")
        print("="*95)
        
    print_metrics_table(df_all['trade_date'].min(), df_all['trade_date'].max(), "FULL PERIOD")
    print_metrics_table(is_start, is_end, "IN-SAMPLE")
    print_metrics_table(oos_start, oos_end, "OUT-OF-SAMPLE (BLIND TEST)")
    
    # Save CSVs
    df_nav_base.rename(columns={'nav': 'nav_baseline'}).to_csv(os.path.join(DATA_DIR, 'nav_baseline_full.csv'))
    df_nav_replace.rename(columns={'nav': 'nav_replace_full'}).to_csv(os.path.join(DATA_DIR, 'nav_replace_full.csv'))
    df_nav_satellite.rename(columns={'nav': 'nav_satellite_full'}).to_csv(os.path.join(DATA_DIR, 'nav_satellite_full.csv'))
    df_nav_cs10.rename(columns={'nav': 'nav_cs10_full'}).to_csv(os.path.join(DATA_DIR, 'nav_cs10_full.csv'))
    df_nav_cs20.rename(columns={'nav': 'nav_cs20_full'}).to_csv(os.path.join(DATA_DIR, 'nav_cs20_full.csv'))
    print("Daily NAV curves saved to ETF data folder.")
    
    # Plotting helper
    def generate_plot(start_dt, end_dt, filename, title):
        plt.figure(figsize=(14, 10))
        plt.subplot(2, 1, 1)
        for s_name, color, df_nav in [
            ('Baseline 8-Asset RP', '#1e88e5', df_nav_base),
            ('Active Replacement', '#8e24aa', df_nav_replace),
            ('Active Satellite', '#00897b', df_nav_satellite),
            ('Core-Satellite 10%', '#fb8c00', df_nav_cs10),
            ('Core-Satellite 20%', '#f4511e', df_nav_cs20)
        ]:
            mask = (df_nav.index >= pd.Timestamp(start_dt)) & (df_nav.index <= pd.Timestamp(end_dt))
            nav_s = df_nav['nav'][mask]
            # scale segment start to 1.0
            plt.plot(nav_s.index, nav_s / nav_s.iloc[0], label=s_name, color=color, linewidth=1.5)
            
        plt.title(title, fontsize=12, fontweight='bold')
        plt.ylabel("Normalized NAV")
        plt.legend()
        plt.grid(True, linestyle=':', alpha=0.6)
        
        plt.subplot(2, 1, 2)
        for s_name, color, df_nav in [
            ('Baseline 8-Asset RP', '#1e88e5', df_nav_base),
            ('Active Replacement', '#8e24aa', df_nav_replace),
            ('Active Satellite', '#00897b', df_nav_satellite),
            ('Core-Satellite 10%', '#fb8c00', df_nav_cs10),
            ('Core-Satellite 20%', '#f4511e', df_nav_cs20)
        ]:
            mask = (df_nav.index >= pd.Timestamp(start_dt)) & (df_nav.index <= pd.Timestamp(end_dt))
            nav_s = df_nav['nav'][mask]
            seg_nav = nav_s / nav_s.iloc[0]
            cum_max = seg_nav.cummax()
            dd = (seg_nav - cum_max) / cum_max * 100.0
            plt.plot(dd.index, dd, label=s_name, color=color, linewidth=1.0, alpha=0.7)
            
        plt.ylabel("Drawdown (%)")
        plt.legend()
        plt.grid(True, linestyle=':', alpha=0.6)
        plt.tight_layout()
        
        # Save to both folders
        plot_path_results = os.path.join(rp.RESULTS_DIR, filename)
        plot_path_artifacts = os.path.join(r"C:\Users\liuqi\.gemini\antigravity\brain\aedf743b-815e-4a43-a730-13a66c11d107", filename)
        plt.savefig(plot_path_results, dpi=300)
        plt.savefig(plot_path_artifacts, dpi=300)
        plt.close()
        print(f"Saved plot: {filename}")
        
    generate_plot(df_all['trade_date'].min(), df_all['trade_date'].max(), 'nav_integrated_comparison_full.png', "Integrated Strategy NAV Comparison (Full Period: 2018-2026)")
    generate_plot(is_start, is_end, 'nav_integrated_comparison_is.png', "Integrated Strategy NAV Comparison (In-Sample: 2018-2024)")
    generate_plot(oos_start, oos_end, 'nav_integrated_comparison_oos.png', "Integrated Strategy NAV Comparison (Out-of-Sample: 2024-2026)")

def main():
    evaluate_and_plot()

if __name__ == "__main__":
    main()

import os
import pandas as pd
import numpy as np
from scipy.stats import norm
import sys

# Add script folder to path
SCRIPT_DIR = r"c:\Users\liuqi\quant_system_v2\etf-valuation-strategy\scripts"
sys.path.append(SCRIPT_DIR)
import step11_risk_parity_strategy as rp

# Load data
df_unified = rp.load_data_8assets(ma_window=200, val_window=1400, vol_lookback=60)
is_start, is_end = "2015-01-01", "2024-02-05"
oos_start, oos_end = "2024-02-06", "2026-03-13"

df_is = df_unified[(df_unified['trade_date'] >= pd.to_datetime(is_start)) & (df_unified['trade_date'] <= pd.to_datetime(is_end))].copy()
df_oos = df_unified[(df_unified['trade_date'] >= pd.to_datetime(oos_start)) & (df_unified['trade_date'] <= pd.to_datetime(oos_end))].copy()

# Modify the backtest function to support a max bond weight constraint
def run_backtest_with_bond_cap(df_period, max_bond_weight=0.20, vol_target=0.06, val_tilt=0.4, q_threshold=0.20, strike_ratio=0.97, buy_put=True):
    if len(df_period) == 0:
        return None
    df_period = df_period.copy().reset_index(drop=True)
    
    assets = ['hs300', 'zz500', 'chinext', 'div', 'gold', 'nasdaq', 'bond', 'cbond']
    equity_assets = ['hs300', 'zz500', 'chinext', 'div']
    
    df_period['year_week'] = df_period['trade_date'].dt.strftime('%Y-%U')
    rebalance_check_dates = set(df_period.groupby('year_week')['trade_date'].first())
    
    ret_cols = [f'ret_{a}' for a in assets]
    df_returns = df_period[ret_cols].copy()
    df_returns.columns = assets
    
    val = {a: 0.0 for a in assets}
    val_cash = 1000000.0
    options_held = []
    nav_history = []
    
    for idx, row in df_period.iterrows():
        dt = row['trade_date']
        
        if idx > 0:
            for a in assets:
                val[a] *= (1.0 + row[f'ret_{a}'])
            val_cash *= (1.0 + 0.02 / 242.0)
            
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
        
        holdings_value = sum(val.values())
        nav = holdings_value + val_cash
        
        # Calculate weights
        vols = np.array([row[f'vol_{a}'] for a in assets])
        vols = np.where(vols <= 0, 1e-4, vols)
        inv_vols = 1.0 / vols
        
        # We apply the bond cap constraint in Risk Parity weights
        # Base weight calculation
        w_rp = inv_vols / inv_vols.sum()
        w_target = {a: w_rp[i] for i, a in enumerate(assets)}
        
        # Capping bond weight and re-distributing
        if w_target['bond'] > max_bond_weight:
            excess = w_target['bond'] - max_bond_weight
            w_target['bond'] = max_bond_weight
            # Re-distribute excess to the other 7 assets proportional to their raw weights
            other_sum = sum(w_target[a] for a in assets if a != 'bond')
            for a in assets:
                if a != 'bond':
                    w_target[a] += excess * (w_target[a] / other_sum)
                    
        # Valuation timing
        for a in equity_assets:
            val_q = row[f'val_q_{a}']
            if not pd.isna(val_q):
                w_target[a] *= (1.0 - val_tilt * (val_q - 0.5))
                
        # Trend filter
        for a in assets:
            close_px = row[f'close_{a}']
            ma_px = row[f'ma_{a}']
            trend_up = close_px >= ma_px if not pd.isna(ma_px) else True
            if not trend_up:
                if a in equity_assets:
                    val_q = row[f'val_q_{a}']
                    if pd.isna(val_q) or val_q > q_threshold:
                        w_target[a] *= 0.5
                elif a in ['gold', 'nasdaq']:
                    w_target[a] *= 0.5
                    
        w_sum = sum(w_target.values())
        for a in assets:
            w_target[a] /= w_sum
            
        # Vol target layer
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
        
        w_curr = {a: val[a] / nav if nav > 0 else 0.0 for a in assets}
        w_curr['cash'] = val_cash / nav if nav > 0 else 0.0
        devs = [abs(w_curr[a] - w_target_final[a]) for a in assets] + [abs(w_curr['cash'] - w_target_final['cash'])]
        max_dev = max(devs)
        
        is_rebal_day = (dt in rebalance_check_dates) or (max_dev > 0.05) or (idx == 0)
        
        if is_rebal_day:
            val_target = {a: nav * w_target_final[a] for a in assets}
            val_target_cash = nav * w_target_final['cash']
            trade_vol = sum(abs(val_target[a] - val[a]) for a in assets) + abs(val_target_cash - val_cash)
            cost = trade_vol * 0.0005
            nav -= cost
            val_cash = nav * w_target_final['cash']
            for a in assets:
                val[a] = nav * w_target_final[a]
                
        # Options purchase
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

print("Testing different Treasury Bond ETF caps:")
caps = [1.0, 0.50, 0.40, 0.30, 0.20, 0.10, 0.0]
for cap in caps:
    # Use best parameters from grid search (vol_target = 0.07 or 0.08, val_tilt = 0.4, strike_ratio = 0.97)
    df_nav_is = run_backtest_with_bond_cap(df_is, max_bond_weight=cap, vol_target=0.08, val_tilt=0.4, strike_ratio=0.97)
    metrics_is = rp.compute_metrics(df_nav_is['nav'])
    
    df_nav_oos = run_backtest_with_bond_cap(df_oos, max_bond_weight=cap, vol_target=0.08, val_tilt=0.4, strike_ratio=0.97)
    metrics_oos = rp.compute_metrics(df_nav_oos['nav'])
    
    print(f"Bond Cap: {cap:<5} | IS CAGR: {metrics_is['CAGR']:.2%} MDD: {metrics_is['Max Drawdown']:.2%} | OOS CAGR: {metrics_oos['CAGR']:.2%} MDD: {metrics_oos['Max Drawdown']:.2%}")

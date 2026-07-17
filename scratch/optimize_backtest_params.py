import os, sys
sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), 'research'))
sys.path.append(os.path.join(os.getcwd(), 'research', '期权'))

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from backtest_options_model import load_market_data, run_backtest, calc_metrics, P

PRED_FILE_OPT = 'research/study_005_1d_advanced/predictions/predictions_005_options_wf.parquet'

ohlc, pctchg, regime_map = load_market_data()
pred_opt = pd.read_parquet(PRED_FILE_OPT)
pred_opt['ds'] = pred_opt['trade_date'].astype(str)

print("Grid search for Option-Enhanced Model:")
print(f"{'th_up':<6} | {'th_crash':<8} | {'Trades':<6} | {'CAGR':<7} | {'Sharpe':<6} | {'MaxDD':<7}")
print("-" * 55)

best_sharpe = -1
best_params = None

for th_up in [0.45, 0.50, 0.55]:
    for th_crash in [0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]:
        P_temp = P.copy()
        P_temp['th_up'] = th_up
        P_temp['th_crash'] = th_crash
        
        pnl_s, stats = run_backtest(pred_opt, ohlc, pctchg, regime_map, P_temp)
        m, _, _ = calc_metrics(pnl_s)
        
        print(f"{th_up:<6.2f} | {th_crash:<8.2f} | {stats['trades']:<6} | {m['CAGR']:>6.1%} | {m['Sharpe']:>5.2f} | {m['MaxDD']:>6.1%}")
        
        if stats['trades'] >= 100 and m['Sharpe'] > best_sharpe:
            best_sharpe = m['Sharpe']
            best_params = (th_up, th_crash, stats['trades'], m['CAGR'], m['Sharpe'], m['MaxDD'])

if best_params:
    print("\nBest Option-Enhanced Parameters (with >= 100 trades):")
    print(f"  th_up: {best_params[0]}")
    print(f"  th_crash: {best_params[1]}")
    print(f"  Trades: {best_params[2]}")
    print(f"  CAGR: {best_params[3]:.1%}")
    print(f"  Sharpe: {best_params[4]:.2f}")
    print(f"  MaxDD: {best_params[5]:.1%}")
else:
    print("\nNo configuration met the trades >= 100 threshold.")

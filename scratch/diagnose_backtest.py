import os, sys
sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), 'research'))
sys.path.append(os.path.join(os.getcwd(), 'research', '期权'))

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from backtest_options_model import load_market_data, run_backtest, P

PRED_FILE_BASE = 'research/study_005_1d_advanced/predictions/predictions_005_wf.parquet'
PRED_FILE_OPT = 'research/study_005_1d_advanced/predictions/predictions_005_options_wf.parquet'

ohlc, pctchg, regime_map = load_market_data()

print("\n--- BASELINE BACKTEST STATS ---")
pred_base = pd.read_parquet(PRED_FILE_BASE)
pred_base['ds'] = pred_base['trade_date'].astype(str)
pnl_base, stats_base = run_backtest(pred_base, ohlc, pctchg, regime_map, P)
for k, v in stats_base.items():
    print(f"  {k}: {v}")

print("\n--- OPTION-ENHANCED BACKTEST STATS ---")
pred_opt = pd.read_parquet(PRED_FILE_OPT)
pred_opt['ds'] = pred_opt['trade_date'].astype(str)
pnl_opt, stats_opt = run_backtest(pred_opt, ohlc, pctchg, regime_map, P)
for k, v in stats_opt.items():
    print(f"  {k}: {v}")

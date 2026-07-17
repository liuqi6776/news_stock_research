import sys
import os
import pandas as pd
import numpy as np

# Add the paths to python path
SCRIPT_DIR = r"c:\Users\liuqi\quant_system_v2\research\期权"
sys.path.append(SCRIPT_DIR)

import backtest_options_model as bom

def main():
    # Load market data
    ohlc, pctchg, regime_map = bom.load_market_data()
    
    # 1. Backtest Baseline
    print("Running Baseline backtest...")
    pred_base = pd.read_parquet(bom.PRED_FILE_BASE)
    pred_base['ds'] = pred_base['trade_date'].astype(str)
    P_base = bom.P.copy()
    P_base['th_up'] = 0.50
    P_base['th_crash'] = 0.45
    pnl_base, _ = bom.run_backtest(pred_base, ohlc, pctchg, regime_map, P_base)
    
    # 2. Backtest Option-Enhanced
    print("Running Option-Enhanced backtest...")
    pred_opt = pd.read_parquet(bom.PRED_FILE_OPT)
    pred_opt['ds'] = pred_opt['trade_date'].astype(str)
    P_opt = bom.P.copy()
    P_opt['th_up'] = 0.50
    P_opt['th_crash'] = 0.45
    pnl_opt, _ = bom.run_backtest(pred_opt, ohlc, pctchg, regime_map, P_opt)
    
    # Save daily PNL to CSV
    df_pnl = pd.DataFrame({
        'pnl_base': pnl_base,
        'pnl_opt': pnl_opt
    })
    df_pnl.index.name = 'trade_date'
    
    out_path = r"c:\Users\liuqi\quant_system_v2\etf-valuation-strategy\data\daily_stock_pnl.csv"
    df_pnl.to_csv(out_path)
    print(f"Daily PNL series saved to {out_path}")
    print(f"Covered dates: {df_pnl.index.min()} to {df_pnl.index.max()}")
    print(df_pnl.head())
    print(df_pnl.tail())

if __name__ == '__main__':
    main()

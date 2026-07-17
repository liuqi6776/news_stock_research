"""
step4_grid_search_005.py
对 Study 005 进行参数网格搜索。
主要搜索：
- th_up: [0.40, 0.45, 0.50]
- th_crash: [0.15, 0.18, 0.20, 0.23, 0.25, 0.30]
使用快速回测逻辑（只算 Full Period CAGR 和 MaxDD）。
"""
import os, sys, warnings
import pandas as pd
import numpy as np

warnings.filterwarnings('ignore')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
STUDY_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.append(SCRIPT_DIR)
import step3_backtest_advanced as bt

def run_grid_search():
    print("Loading Data...")
    ohlc, pctchg, regime_map, pred = bt.load_data()
    
    # 固定的基础参数
    base_P = {
        'max_pos':     3,
        'gap_low':     0.02,
        'gap_high':    0.06,
        'stop_loss':  -0.05,
        'tp_trigger':  0.06,
        'tp_pullback': 0.015,
        'tp_floor':    0.05,
        'max_per_ind': 2
    }
    
    th_ups = [0.40, 0.45, 0.50]
    th_crashs = [0.15, 0.18, 0.20, 0.23, 0.25, 0.30]
    
    results = []
    
    print("\nStarting Grid Search...")
    print(f"{'th_up':<6} | {'th_crash':<8} | {'Trades':<6} | {'CAGR':<8} | {'MaxDD':<8} | {'Sharpe':<6}")
    print("-" * 55)
    
    for tu in th_ups:
        for tc in th_crashs:
            P = base_P.copy()
            P['th_up'] = tu
            P['th_crash'] = tc
            
            pnl_s, stats = bt.run_backtest(pred, ohlc, pctchg, regime_map, P)
            
            if len(pnl_s) == 0 or stats.get('trades', 0) == 0:
                continue
                
            m, eq, dd = bt.calc_metrics(pnl_s)
            
            print(f"{tu:<6.2f} | {tc:<8.2f} | {stats['trades']:<6d} | {m['CAGR']:>6.1%} | {m['MaxDD']:>6.1%} | {m['Sharpe']:>4.2f}")
            
            results.append({
                'th_up': tu,
                'th_crash': tc,
                'trades': stats['trades'],
                'CAGR': m['CAGR'],
                'MaxDD': m['MaxDD'],
                'Sharpe': m['Sharpe']
            })
            
    df_res = pd.DataFrame(results)
    df_res = df_res.sort_values('CAGR', ascending=False)
    
    print("\nTop 5 by CAGR:")
    print(df_res.head(5).to_string(index=False, formatters={
        'CAGR': lambda x: f"{x:.1%}",
        'MaxDD': lambda x: f"{x:.1%}",
        'Sharpe': lambda x: f"{x:.2f}"
    }))
    
    out_path = os.path.join(STUDY_DIR, 'results', 'grid_search_005.csv')
    df_res.to_csv(out_path, index=False)
    print(f"\nSaved grid search results to {out_path}")

if __name__ == '__main__':
    run_grid_search()

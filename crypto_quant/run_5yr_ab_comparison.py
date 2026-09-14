"""
5-Year Full-Cycle Backtest Comparison: Strategy A vs Strategy B (2021 - 2026)
方案 A (BTC 脉冲引领跟随) vs 方案 B (做市商高频自适应网格) 过去5年真实表现对比
"""

import os
import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from crypto_quant.backtest_5yr_ab import backtest_strategy_a_impulse, backtest_strategy_b_maker_grid

def run_5yr_ab_test():
    print("==========================================================================================")
    print("[*] 5-Year High-Frequency Empirical Backtest (2021 - 2026): Strategy A vs Strategy B")
    print("==========================================================================================")
    
    # 1. Load 5-year 1h data
    btc = pd.read_parquet("data/crypto_cache/BTCUSDT_1h_2021_2026.parquet")
    eth = pd.read_parquet("data/crypto_cache/ETHUSDT_1h_2021_2026.parquet")
    bnb = pd.read_parquet("data/crypto_cache/BNBUSDT_1h_2021_2026.parquet")
    sol = pd.read_parquet("data/crypto_cache/SOLUSDT_1h_2021_2026.parquet")
    
    print(f"[+] Data loaded: ~{len(eth)} hourly bars ({eth.index[0].date()} to {eth.index[-1].date()})")
    print("------------------------------------------------------------------------------------------")
    
    # 2. Backtest Strategy A (BTC Impulse Lead-Lag Momentum Follower)
    print("\n>>> [STRATEGY A] BTC Impulse Leading Follower Strategy / 方案 A: BTC 脉冲引领跟随策略")
    print("    Rule: BTC 1h impulse >= +1.5% & Taker Buy >= 58% -> Long Altcoin on next open")
    print("    Risk: Stop Loss -2.5%, Take Profit +5.0%, Max Hold 12h, Taker Fee 0.04%")
    
    strat_a_results = []
    for sym, df in [("ETHUSDT", eth), ("SOLUSDT", sol), ("BNBUSDT", bnb)]:
        res = backtest_strategy_a_impulse(
            btc_df=btc,
            target_df=df,
            symbol_name=sym,
            impulse_ret_threshold=0.015,
            taker_buy_threshold=0.58,
            take_profit=0.05,
            stop_loss=-0.025,
            max_hold_bars=12,
            fee_rate=0.0004
        )
        strat_a_results.append({
            "Target Asset": sym,
            "Total Return": f"{res['total_return']:+,.2f}%",
            "CAGR": f"{res['cagr']:+.2f}%",
            "Sharpe": f"{res['sharpe']:.2f}",
            "Max Drawdown": f"{res['max_drawdown']:.2f}%",
            "Win Rate": f"{res['win_rate']:.1f}%",
            "Profit Factor": f"{res['profit_factor']:.2f}",
            "Trades": res["trades"],
            "Benchmark (Buy&Hold)": f"{res['benchmark_return']:+,.2f}%"
        })
    print(pd.DataFrame(strat_a_results).to_string(index=False))
    
    # 3. Backtest Strategy B (Adaptive High-Frequency Maker Grid)
    print("\n>>> [STRATEGY B] High-Frequency Adaptive Maker Grid / 方案 B: 高频自适应做市挂单网格")
    print("    Rule: 10 Grid Levels dynamically spaced by 0.5 ATR, Maker Fee 0.02%, Macro Crash Filter (MA240)")
    print("    Capital: 50% Cash, 50% Asset. Captures intraday oscillation while earning Maker spread")
    
    strat_b_results = []
    for sym, df in [("ETHUSDT", eth), ("SOLUSDT", sol), ("BNBUSDT", bnb)]:
        res = backtest_strategy_b_maker_grid(
            df=df,
            symbol_name=sym,
            grid_levels=10,
            grid_spacing_atr_pct=0.5,
            trend_filter_ma=240,
            maker_fee=0.0002,
            initial_capital=10000.0
        )
        strat_b_results.append({
            "Target Asset": sym,
            "Total Return": f"{res['total_return']:+,.2f}%",
            "CAGR": f"{res['cagr']:+.2f}%",
            "Sharpe": f"{res['sharpe']:.2f}",
            "Max Drawdown": f"{res['max_drawdown']:.2f}%",
            "Grid Fills (Trades)": res["grid_trades"],
            "Final Capital": f"${res['final_capital']:,.2f}",
            "Benchmark (Buy&Hold)": f"{res['benchmark_return']:+,.2f}%"
        })
    print(pd.DataFrame(strat_b_results).to_string(index=False))
    print("\n==========================================================================================\n")

if __name__ == "__main__":
    run_5yr_ab_test()

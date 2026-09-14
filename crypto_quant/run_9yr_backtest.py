"""
9-Year Multi-Cycle Quantitative Backtesting Runner
9年跨越完整多轮牛熊周期的加密货币量化回测与周期归因分析
"""

import sys
import os
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from crypto_quant.backtester import CryptoBacktester
from crypto_quant.strategies import (
    dual_ema_trend_strategy,
    bollinger_breakout_strategy,
    rsi_momentum_strategy,
    supertrend_strategy
)

# Defined Crypto Market Cycles (2017 - 2026)
CYCLES = {
    "1. 2017-2018 (ICO Mania & Great Bear -84%)": ("2017-08-17", "2018-12-31"),
    "2. 2019-2020 (Echo Bubble & 312 Crash)": ("2019-01-01", "2020-03-31"),
    "3. 2020-2021 (DeFi/Halving Super Bull $69k)": ("2020-04-01", "2021-11-10"),
    "4. 2021-2022 (Luna/FTX Deleveraging Bear -77%)": ("2021-11-11", "2022-12-31"),
    "5. 2023-2026 (Spot ETF & Institutional Era)": ("2023-01-01", "2026-09-13"),
}

def load_data(symbol="BTCUSDT"):
    cache_path = f"data/crypto_cache/{symbol}_1d_2017_2026.parquet"
    if not os.path.exists(cache_path):
        raise FileNotFoundError(f"Cache not found at {cache_path}")
    df = pd.read_parquet(cache_path)
    return df

def run_9year_analysis(symbol="BTCUSDT"):
    print(f"\n==========================================================================================")
    print(f"[*] Comprehensive 9-Year Full-Cycle Quantitative Backtest: {symbol} (Daily Bars: 2017 - 2026)")
    print(f"==========================================================================================")
    
    df = load_data(symbol)
    start_date = df.index[0].strftime('%Y-%m-%d')
    end_date = df.index[-1].strftime('%Y-%m-%d')
    start_price = df['close'].iloc[0]
    end_price = df['close'].iloc[-1]
    bh_return = (end_price / start_price - 1.0) * 100
    
    print(f"[+] Total Sample Duration: {start_date} to {end_date} ({(df.index[-1] - df.index[0]).days} days, {len(df)} daily candles)")
    print(f"[+] Asset Price Evolution: ${start_price:,.2f} -> ${end_price:,.2f} (Buy & Hold Return: {bh_return:+,.2f}%)")
    print(f"------------------------------------------------------------------------------------------")

    # Daily data: 365 periods/year for 24/7 crypto
    backtester = CryptoBacktester(
        initial_capital=10000.0,
        commission_rate=0.0005,
        slippage=0.0002,
        allow_short=False,
        annual_periods=365
    )

    strategies = {
        "SuperTrend (10, 3.0)": supertrend_strategy(df, period=10, multiplier=3.0),
        "Dual EMA Trend (12, 26)": dual_ema_trend_strategy(df, fast_period=12, slow_period=26),
        "Bollinger Breakout (20, 2.0)": bollinger_breakout_strategy(df, period=20, std_dev=2.0),
        "RSI Momentum (14, 70)": rsi_momentum_strategy(df, rsi_period=14, oversold_entry=35.0, overbought_exit=70.0)
    }

    # 1. Full 9-Year Overall Performance
    print("\n[PART 1] 9-Year Full Horizon Backtest Results (2017-2026)")
    overall_rows = []
    results_map = {}
    for name, signals in strategies.items():
        res = backtester.run(df, signals)
        results_map[name] = res
        m = res["metrics"]
        overall_rows.append({
            "Strategy": name,
            "Total Return": f"{m['Total Return'] * 100:+,.2f}%",
            "CAGR": f"{m['Annualized Return (CAGR)'] * 100:.2f}%",
            "Sharpe": f"{m['Sharpe Ratio']:.2f}",
            "Max Drawdown": f"{m['Max Drawdown'] * 100:.2f}%",
            "Calmar": f"{m['Calmar Ratio']:.2f}",
            "Win Rate": f"{m['Win Rate'] * 100:.1f}%",
            "Trades": m["Total Trades"]
        })
    
    # Add Buy & Hold baseline
    bm_res = backtester.run(df, pd.Series(1, index=df.index))
    bm_m = bm_res["metrics"]
    overall_rows.append({
        "Strategy": "Buy & Hold (Baseline)",
        "Total Return": f"{bm_m['Total Return'] * 100:+,.2f}%",
        "CAGR": f"{bm_m['Annualized Return (CAGR)'] * 100:.2f}%",
        "Sharpe": f"{bm_m['Sharpe Ratio']:.2f}",
        "Max Drawdown": f"{bm_m['Max Drawdown'] * 100:.2f}%",
        "Calmar": f"{bm_m['Calmar Ratio']:.2f}",
        "Win Rate": "N/A",
        "Trades": 1
    })

    summary_df = pd.DataFrame(overall_rows)
    print(summary_df.to_string(index=False))
    print("------------------------------------------------------------------------------------------")

    # 2. Cycle-by-Cycle Stress Testing & Breakdown
    print("\n[PART 2] Multi-Cycle Regime Stress Test Breakdown / 分周期牛熊压力测试")
    for cycle_name, (c_start, c_end) in CYCLES.items():
        sub_df = df.loc[c_start:c_end]
        if sub_df.empty:
            continue
        c_bh_return = (sub_df['close'].iloc[-1] / sub_df['close'].iloc[0] - 1.0) * 100
        print(f"\n>>> Cycle: {cycle_name}")
        print(f"    Duration: {c_start} to {c_end} ({len(sub_df)} bars) | Benchmark Return: {c_bh_return:+.2f}%")
        
        cycle_rows = []
        for name, full_signals in strategies.items():
            sub_signals = full_signals.loc[sub_df.index]
            sub_res = backtester.run(sub_df, sub_signals)
            sm = sub_res["metrics"]
            cycle_rows.append({
                "Strategy": name,
                "Cycle Return": f"{sm['Total Return'] * 100:+.2f}%",
                "Max DD": f"{sm['Max Drawdown'] * 100:.2f}%",
                "Sharpe": f"{sm['Sharpe Ratio']:.2f}",
                "Win Rate": f"{sm['Win Rate'] * 100:.1f}%",
                "Trades": sm["Total Trades"]
            })
        print(pd.DataFrame(cycle_rows).to_string(index=False))

    print("\n==========================================================================================\n")

if __name__ == "__main__":
    symbol = sys.argv[1] if len(sys.argv) > 1 else "BTCUSDT"
    run_9year_analysis(symbol)

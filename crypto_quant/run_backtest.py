"""
Crypto Quantitative Backtesting Demonstration Runner
加密货币量化策略回测运行与对比测试
"""

import sys
import os
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd
from crypto_quant.data_fetcher import fetch_history_klines_paginated, fetch_klines, fetch_24h_ticker
from crypto_quant.backtester import CryptoBacktester
from crypto_quant.strategies import (
    dual_ema_trend_strategy,
    bollinger_breakout_strategy,
    rsi_momentum_strategy,
    supertrend_strategy
)

def run_crypto_analysis(symbol: str = "BTCUSDT", interval: str = "1h", candles: int = 1200):
    print(f"\n=================================================================")
    print(f"[*] Initializing Crypto Quant Research for {symbol} ({interval})")
    print(f"=================================================================")

    # 1. Fetch 24h market snapshot
    ticker = fetch_24h_ticker(symbol, market_type="spot")
    print(f"[+] Current Market: Last Price ${float(ticker['lastPrice']):,.2f} | 24h Change: {ticker['priceChangePercent']}% | Volume: {float(ticker['volume']):,.2f}")

    # 2. Fetch historical K-lines
    print(f"[+] Fetching {candles} bars of historical K-lines from Binance Public API...")
    df = fetch_history_klines_paginated(symbol=symbol, interval=interval, total_candles=candles, market_type="spot")
    print(f"[+] Loaded {len(df)} candles from {df.index[0]} to {df.index[-1]}")

    # 3. Setup backtester (1h intervals: 8760 periods/year, 0.05% taker fee, 0.02% slippage)
    backtester = CryptoBacktester(
        initial_capital=10000.0,
        commission_rate=0.0005,
        slippage=0.0002,
        allow_short=False,
        annual_periods=8760
    )

    # 4. Run Strategies
    strategies = {
        "Dual EMA Trend (12, 26)": dual_ema_trend_strategy(df, fast_period=12, slow_period=26),
        "Bollinger Breakout (20, 2.0)": bollinger_breakout_strategy(df, period=20, std_dev=2.0),
        "RSI Momentum (14, Overbought 70)": rsi_momentum_strategy(df, rsi_period=14, oversold_entry=35.0, overbought_exit=70.0),
        "SuperTrend (10, 3.0)": supertrend_strategy(df, period=10, multiplier=3.0)
    }

    comparison_rows = []
    benchmark_m = None

    for name, signals in strategies.items():
        res = backtester.run(df, signals)
        m = res["metrics"]
        if benchmark_m is None:
            benchmark_m = {
                "Total Return": f"{m['Benchmark Return'] * 100:+.2f}%",
                "Annualized CAGR": f"{m['Benchmark CAGR'] * 100:+.2f}%",
                "Max Drawdown": f"{m['Benchmark Max Drawdown'] * 100:.2f}%",
                "Sharpe": "N/A"
            }

        comparison_rows.append({
            "Strategy": name,
            "Total Return": f"{m['Total Return'] * 100:+.2f}%",
            "CAGR": f"{m['Annualized Return (CAGR)'] * 100:+.2f}%",
            "Alpha": f"{m['Alpha (Excess Return)'] * 100:+.2f}%",
            "Sharpe": f"{m['Sharpe Ratio']:.2f}",
            "Max DD": f"{m['Max Drawdown'] * 100:.2f}%",
            "Win Rate": f"{m['Win Rate'] * 100:.1f}%",
            "Trades": m["Total Trades"]
        })

    summary_df = pd.DataFrame(comparison_rows)
    print("\n======================= STRATEGY COMPARISON =======================")
    print(summary_df.to_string(index=False))
    print(f"\n[*] Benchmark (Buy & Hold): Return = {benchmark_m['Total Return']}, CAGR = {benchmark_m['Annualized CAGR']}, Max DD = {benchmark_m['Max Drawdown']}")
    print("===================================================================\n")

if __name__ == "__main__":
    symbol = sys.argv[1] if len(sys.argv) > 1 else "BTCUSDT"
    run_crypto_analysis(symbol=symbol)

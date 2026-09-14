"""
Crypto Quantitative Backtesting Engine / 加密货币量化回测引擎
Designed for 24/7 continuous crypto markets with maker/taker fee structures and slippage.
专为 24/7 不间断加密市场设计，支持做市商/吃单费率模型、滑点模拟与多空双向交易。
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List, Tuple

class CryptoBacktester:
    """
    Backtesting engine for cryptocurrency strategies.
    
    Parameters:
    -----------
    initial_capital: float
        Starting balance in quote currency (e.g. 10,000 USDT)
    commission_rate: float
        Fee per trade (default: 0.0005 = 0.05% taker fee)
    slippage: float
        Estimated price slippage per trade (default: 0.0002 = 0.02%)
    allow_short: bool
        Whether to allow short selling (True for Futures, False for Spot)
    annual_periods: int
        Periods per year for annualizing metrics (default: 365 * 24 = 8760 for 1h data)
    risk_free_rate: float
        Annual risk-free interest rate (default: 0.03 = 3%)
    """
    def __init__(
        self,
        initial_capital: float = 10000.0,
        commission_rate: float = 0.0005,
        slippage: float = 0.0002,
        allow_short: bool = False,
        annual_periods: int = 8760,
        risk_free_rate: float = 0.03
    ):
        self.initial_capital = initial_capital
        self.commission_rate = commission_rate
        self.slippage = slippage
        self.allow_short = allow_short
        self.annual_periods = annual_periods
        self.risk_free_rate = risk_free_rate

    def run(
        self,
        df: pd.DataFrame,
        signal_series: pd.Series,
        price_col: str = "close",
        open_col: str = "open"
    ) -> Dict[str, Any]:
        """
        Run backtest on given price dataframe and target position signals.

        Parameters:
        -----------
        df: pd.DataFrame
            OHLCV dataframe with datetime index.
        signal_series: pd.Series
            Target position: 1 (Long), 0 (Flat), -1 (Short).
        price_col: str
            Column used for mark-to-market valuation.
        open_col: str
            Column used for trade execution price.

        Returns:
        --------
        Dict containing performance summary, equity curve, and trade records.
        """
        data = df.copy()
        raw_signals = signal_series.reindex(data.index).fillna(0)
        
        if not self.allow_short:
            raw_signals = raw_signals.clip(lower=0)

        # Position is executed at next bar open to avoid look-ahead bias
        # 信号在下一根 K 线的 Open 执行，避免未来函数前视偏差
        positions = raw_signals.shift(1).fillna(0)
        
        prices = data[price_col]
        open_prices = data[open_col] if open_col in data.columns else prices
        returns = prices.pct_change().fillna(0)

        # Detect trades
        position_changes = positions.diff().fillna(positions.iloc[0])
        trade_occurred = position_changes != 0

        # Transaction costs: commission + slippage applied whenever position changes
        costs = position_changes.abs() * (self.commission_rate + self.slippage)

        # Strategy returns
        strategy_returns = (positions * returns) - costs

        # Cumulative returns and equity curves
        cum_strategy_returns = (1.0 + strategy_returns).cumprod()
        cum_benchmark_returns = (1.0 + returns).cumprod()

        equity_curve = self.initial_capital * cum_strategy_returns
        benchmark_curve = self.initial_capital * cum_benchmark_returns

        # Drawdown computation
        peak = equity_curve.cummax()
        drawdown = (equity_curve - peak) / peak
        max_drawdown = drawdown.min()

        # Benchmark drawdown
        bm_peak = benchmark_curve.cummax()
        bm_drawdown = (benchmark_curve - bm_peak) / bm_peak
        bm_max_drawdown = bm_drawdown.min()

        # Performance metrics
        total_bars = len(strategy_returns)
        years = total_bars / self.annual_periods if self.annual_periods > 0 else 1.0

        total_return = (equity_curve.iloc[-1] / self.initial_capital) - 1.0
        benchmark_return = (benchmark_curve.iloc[-1] / self.initial_capital) - 1.0

        cagr = (1.0 + total_return) ** (1.0 / max(years, 1e-4)) - 1.0 if (1.0 + total_return) > 0 else -1.0
        bm_cagr = (1.0 + benchmark_return) ** (1.0 / max(years, 1e-4)) - 1.0 if (1.0 + benchmark_return) > 0 else -1.0

        annualized_vol = strategy_returns.std() * np.sqrt(self.annual_periods)
        bm_vol = returns.std() * np.sqrt(self.annual_periods)

        excess_return = cagr - self.risk_free_rate
        sharpe_ratio = excess_return / (annualized_vol + 1e-10)

        downside_returns = strategy_returns[strategy_returns < 0]
        downside_std = downside_returns.std() * np.sqrt(self.annual_periods)
        sortino_ratio = excess_return / (downside_std + 1e-10)

        calmar_ratio = cagr / abs(max_drawdown) if abs(max_drawdown) > 0 else np.nan

        # Trade analytics
        trade_indices = data.index[trade_occurred].tolist()
        num_trades = int(trade_occurred.sum())
        
        # Analyze individual trade profits
        trade_logs = []
        in_trade = False
        entry_price = 0.0
        entry_time = None
        entry_dir = 0

        for i in range(len(positions)):
            pos = positions.iloc[i]
            prev_pos = positions.iloc[i - 1] if i > 0 else 0
            cur_time = data.index[i]
            cur_price = open_prices.iloc[i]

            if pos != prev_pos:
                # Close previous trade
                if in_trade:
                    exit_price = cur_price * (1.0 - self.slippage if entry_dir == 1 else 1.0 + self.slippage)
                    pnl_pct = (exit_price - entry_price) / entry_price if entry_dir == 1 else (entry_price - exit_price) / entry_price
                    pnl_pct -= (self.commission_rate * 2) # round trip fee
                    trade_logs.append({
                        "entry_time": entry_time,
                        "exit_time": cur_time,
                        "direction": "LONG" if entry_dir == 1 else "SHORT",
                        "entry_price": entry_price,
                        "exit_price": exit_price,
                        "pnl_pct": pnl_pct,
                        "holding_bars": i - entry_bar
                    })
                    in_trade = False

                # Open new trade
                if pos != 0:
                    in_trade = True
                    entry_dir = int(pos)
                    entry_price = cur_price * (1.0 + self.slippage if entry_dir == 1 else 1.0 - self.slippage)
                    entry_time = cur_time
                    entry_bar = i

        trades_df = pd.DataFrame(trade_logs)
        win_rate = 0.0
        profit_factor = 0.0
        avg_trade_pnl = 0.0

        if not trades_df.empty:
            wins = trades_df[trades_df["pnl_pct"] > 0]
            losses = trades_df[trades_df["pnl_pct"] <= 0]
            win_rate = len(wins) / len(trades_df)
            total_gain = wins["pnl_pct"].sum()
            total_loss = abs(losses["pnl_pct"].sum())
            profit_factor = total_gain / total_loss if total_loss > 0 else np.nan
            avg_trade_pnl = trades_df["pnl_pct"].mean()

        metrics = {
            "Total Return": total_return,
            "Annualized Return (CAGR)": cagr,
            "Benchmark Return": benchmark_return,
            "Benchmark CAGR": bm_cagr,
            "Alpha (Excess Return)": total_return - benchmark_return,
            "Annualized Volatility": annualized_vol,
            "Sharpe Ratio": sharpe_ratio,
            "Sortino Ratio": sortino_ratio,
            "Max Drawdown": max_drawdown,
            "Benchmark Max Drawdown": bm_max_drawdown,
            "Calmar Ratio": calmar_ratio,
            "Total Trades": num_trades,
            "Win Rate": win_rate,
            "Profit Factor": profit_factor,
            "Average Trade Return": avg_trade_pnl,
            "Total Bars": total_bars,
            "Years": years
        }

        result_df = pd.DataFrame({
            "price": prices,
            "position": positions,
            "strategy_return": strategy_returns,
            "equity": equity_curve,
            "drawdown": drawdown,
            "benchmark_equity": benchmark_curve
        })

        return {
            "metrics": metrics,
            "result_df": result_df,
            "trades_df": trades_df
        }

    def print_performance_report(self, results: Dict[str, Any], title: str = "Strategy Performance Report"):
        """Format and print a clean markdown performance table."""
        m = results["metrics"]
        print(f"\n=======================================================")
        print(f"[*] {title}")
        print(f"=======================================================")
        print(f"  Initial Capital:        ${self.initial_capital:,.2f}")
        print(f"  Final Strategy Capital: ${results['result_df']['equity'].iloc[-1]:,.2f}")
        print(f"  Total Return:           {m['Total Return'] * 100:+.2f}%")
        print(f"  Annualized CAGR:        {m['Annualized Return (CAGR)'] * 100:+.2f}%")
        print(f"  Benchmark (Buy & Hold): {m['Benchmark Return'] * 100:+.2f}%")
        print(f"  Alpha (Excess Return):  {m['Alpha (Excess Return)'] * 100:+.2f}%")
        print(f"-------------------------------------------------------")
        print(f"  Sharpe Ratio:           {m['Sharpe Ratio']:.2f}")
        print(f"  Sortino Ratio:          {m['Sortino Ratio']:.2f}")
        print(f"  Calmar Ratio:           {m['Calmar Ratio']:.2f}")
        print(f"  Max Drawdown:           {m['Max Drawdown'] * 100:.2f}%")
        print(f"  Benchmark Max Drawdown: {m['Benchmark Max Drawdown'] * 100:.2f}%")
        print(f"  Annual Volatility:      {m['Annualized Volatility'] * 100:.2f}%")
        print(f"-------------------------------------------------------")
        print(f"  Total Trades:           {m['Total Trades']}")
        print(f"  Win Rate:               {m['Win Rate'] * 100:.2f}%")
        print(f"  Profit Factor:          {m['Profit Factor']:.2f}")
        print(f"  Avg Trade Return:       {m['Average Trade Return'] * 100:+.2f}%")
        print(f"=======================================================\n")

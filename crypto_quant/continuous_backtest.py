# -*- coding: utf-8 -*-
"""
Continuous State Backtest Engine (Phase 16)
Eliminates segmented slice state-reset artifacts:
1. Runs full history continuously (e.g. 2024-01-01 to 2026-09-13) in a single pass
2. Retains complete StrategyState across every bar (loss streaks, cooldown flags, peak equity)
3. Supports state persistence & recovery (to_dict / from_dict / to_json / from_json)
4. Provides non-resetting slice_report(start, end) that reports carryover positions and continuous MTM
"""

import json
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any, Tuple
import numpy as np
import pandas as pd

from crypto_quant.execution_model import ExecutionModel, StopCheckResult


@dataclass
class StrategyState:
    timestamp: str  # ISO format string
    position: float = 0.0  # Signed position: +size (long), -size (short), 0.0 (cash)
    position_size: float = 0.0  # Absolute magnitude of active position
    entry_time: Optional[str] = None
    entry_price: float = 0.0
    loss_streak: int = 0
    in_waterfall: bool = False  # Long stop-loss cooldown (requires green candle to reset)
    in_short_squeeze: bool = False  # Short stop-loss cooldown (requires red candle to reset)
    realized_equity: float = 1.0
    unrealized_pnl: float = 0.0
    peak_equity: float = 1.0
    latest_drawdown: float = 0.0
    rolling_preds_buffer: List[float] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "StrategyState":
        return cls(**d)

    def to_json(self) -> str:
        return json.dumps(self.to_dict())

    @classmethod
    def from_json(cls, s: str) -> "StrategyState":
        return cls.from_dict(json.loads(s))


@dataclass
class TradeRecord:
    symbol: str
    direction: int  # +1 Long, -1 Short
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    entry_price: float
    exit_price: float
    position_size: float
    gross_return: float
    net_return: float
    fee_and_slippage: float
    funding_fees: float
    duration_bars: int
    exit_reason: str  # 'signal_exit', 'intrabar_stop', 'gap_stop'


@dataclass
class SliceReport:
    start_date: str
    end_date: str
    total_return: float
    max_drawdown: float
    daily_sharpe: float
    calmar_ratio: float
    win_rate: float
    profit_factor: float
    total_trades: int
    carryover_position: Optional[Dict[str, Any]]
    equity_series: pd.Series
    trades: List[TradeRecord]

    def to_summary_dict(self) -> Dict[str, Any]:
        return {
            "start_date": self.start_date,
            "end_date": self.end_date,
            "total_return_pct": round(self.total_return * 100.0, 2),
            "max_drawdown_pct": round(self.max_drawdown * 100.0, 2),
            "daily_sharpe": round(self.daily_sharpe, 2),
            "calmar_ratio": round(self.calmar_ratio, 2),
            "win_rate_pct": round(self.win_rate * 100.0, 2),
            "profit_factor": round(self.profit_factor, 2),
            "total_trades": self.total_trades,
            "carryover_position": self.carryover_position,
        }


class ContinuousBacktestEngine:
    """
    Executes a single continuous multi-year simulation with institutional risk controls.
    """

    def __init__(
        self,
        symbol: str,
        stop_loss: float = 0.035,
        deadband: float = 0.20,
        use_short: bool = True,
        trial_mode: bool = True,
        execution_model: Optional[ExecutionModel] = None,
        initial_state: Optional[StrategyState] = None,
    ):
        self.symbol = symbol
        self.stop_loss = stop_loss
        self.deadband = deadband
        self.use_short = use_short
        self.trial_mode = trial_mode
        self.exec_model = execution_model or ExecutionModel()
        self.initial_state = initial_state

    def run(
        self,
        df_bars: pd.DataFrame,
        series_pred: pd.Series,
        series_funding: Optional[pd.Series] = None,
        series_fng: Optional[pd.Series] = None,
    ) -> "BacktestResult":
        """
        Runs the full history continuously.
        df_bars must contain ['open', 'high', 'low', 'close'] with DatetimeIndex.
        """
        # Ensure alignment
        common_idx = df_bars.index.intersection(series_pred.index)
        bars = df_bars.loc[common_idx]
        preds = series_pred.loc[common_idx]
        funding = series_funding.reindex(common_idx).fillna(0.0) if series_funding is not None else pd.Series(0.0, index=common_idx)
        fng = series_fng.reindex(common_idx).fillna(50.0) if series_fng is not None else pd.Series(50.0, index=common_idx)

        # Precompute rolling signals causally: shift(1) before rolling
        prior_mean = preds.shift(1).rolling(72).mean()
        prior_std = preds.shift(1).rolling(72).std() + 1e-8
        z_vals = ((preds - prior_mean) / prior_std).fillna(0.0)

        # Top/bottom exhaustion indicators
        ema72 = bars["close"].shift(1).ewm(span=72).mean()
        stretch = ((bars["close"].shift(1) - ema72) / (ema72 + 1e-8)).fillna(0.0)
        stretch_risk = np.clip(stretch / 0.05, -1.0, 1.0)
        fund_risk = np.clip((funding * 100.0) / 0.02, -1.0, 1.0)
        fng_risk = np.clip((fng - 50.0) / 30.0, -1.0, 1.0)

        top_risk = 0.45 * np.maximum(0.0, stretch_risk) + 0.35 * np.maximum(0.0, fund_risk) + 0.20 * np.maximum(0.0, fng_risk)
        bot_risk = 0.45 * np.maximum(0.0, -stretch_risk) + 0.35 * np.maximum(0.0, -fund_risk) + 0.20 * np.maximum(0.0, -fng_risk)

        base_size_long = np.clip(1.0 - 0.65 * np.maximum(0.0, top_risk - 0.25) / 0.75, 0.35, 1.0)
        base_size_short = np.clip(1.0 - 0.65 * np.maximum(0.0, bot_risk - 0.25) / 0.75, 0.35, 1.0)

        # Trial mode metrics: ATR and macro trend
        tr = pd.concat([
            bars["high"] - bars["low"],
            (bars["high"] - bars["close"].shift(1)).abs(),
            (bars["low"] - bars["close"].shift(1)).abs()
        ], axis=1).max(axis=1)
        atr14 = tr.rolling(14).mean().bfill()
        atr_ratio = atr14 / (bars["close"] + 1e-8)
        m_vol = np.clip(0.025 / (atr_ratio + 1e-8), 0.40, 1.10)

        ema144 = bars["close"].shift(1).ewm(span=144).mean()
        trend_bias = (bars["close"].shift(1) - ema144).fillna(0.0)

        # Simulation loop state
        n = len(common_idx)
        if self.initial_state is not None:
            state = self.initial_state
        else:
            state = StrategyState(timestamp=str(common_idx[0]))

        state_history: List[StrategyState] = []
        equity_records = []
        position_records = []
        trades: List[TradeRecord] = []

        active_trade: Optional[Dict[str, Any]] = None
        if state.position != 0:
            active_trade = {
                "direction": 1 if state.position > 0 else -1,
                "entry_time": pd.to_datetime(state.entry_time) if state.entry_time else common_idx[0],
                "entry_price": state.entry_price,
                "position_size": state.position_size,
                "entry_idx": 0,
                "cum_funding": 0.0,
            }

        opens = bars["open"].values
        highs = bars["high"].values
        lows = bars["low"].values
        closes = bars["close"].values
        z_arr = z_vals.values
        fng_arr = fng.values
        fund_arr = funding.values
        sz_l_arr = base_size_long.values
        sz_s_arr = base_size_short.values
        m_vol_arr = m_vol.values
        trend_arr = trend_bias.values

        for i in range(n):
            ts = common_idx[i]
            cur_open = opens[i]
            cur_high = highs[i]
            cur_low = lows[i]
            cur_close = closes[i]
            cur_z = z_arr[i]
            cur_fund = fund_arr[i]

            # 1. State-driven cooldown release check at start of bar
            if state.in_waterfall and cur_close >= cur_open:
                state.in_waterfall = False
            if state.in_short_squeeze and cur_close <= cur_open:
                state.in_short_squeeze = False

            # 2. If in position: check intrabar stop-loss first (absolute priority)
            is_stopped = False
            if state.position != 0:
                pos_dir = 1 if state.position > 0 else -1
                stop_res = self.exec_model.check_intrabar_stop(
                    pos_direction=pos_dir,
                    entry_price=state.entry_price,
                    stop_loss_pct=self.stop_loss,
                    open_p=cur_open,
                    high_p=cur_high,
                    low_p=cur_low,
                    close_p=cur_close,
                )
                if stop_res.is_stopped:
                    is_stopped = True
                    fill_p = stop_res.fill_price
                    # Close trade immediately at fill_p
                    gross_ret = (fill_p / state.entry_price - 1.0) * pos_dir
                    fee = self.exec_model.taker_fee * 2.0  # entry + exit fee
                    net_ret = gross_ret - fee

                    # Close active trade record
                    if active_trade:
                        trades.append(TradeRecord(
                            symbol=self.symbol,
                            direction=pos_dir,
                            entry_time=active_trade["entry_time"],
                            exit_time=ts,
                            entry_price=state.entry_price,
                            exit_price=fill_p,
                            position_size=state.position_size,
                            gross_return=gross_ret,
                            net_return=net_ret + active_trade["cum_funding"],
                            fee_and_slippage=fee,
                            funding_fees=active_trade["cum_funding"],
                            duration_bars=i - active_trade["entry_idx"] + 1,
                            exit_reason=stop_res.reason,
                        ))
                        active_trade = None

                    state.realized_equity *= (1.0 + net_ret * state.position_size)
                    state.unrealized_pnl = 0.0
                    state.position = 0.0
                    state.position_size = 0.0
                    state.entry_price = 0.0
                    state.entry_time = None
                    state.loss_streak += 1

                    if pos_dir == 1:
                        state.in_waterfall = True
                    else:
                        state.in_short_squeeze = True

            # 3. Discrete 8h Funding Settlement (if still in position)
            if state.position != 0 and self.exec_model.is_funding_settlement_bar(ts):
                funding_flow = self.exec_model.compute_funding_cashflow(
                    timestamp=ts,
                    signed_position=state.position,
                    funding_rate=cur_fund,
                    mark_price=cur_close,
                    enforce_settlement_hours=False,
                )
                # Funding cash flow impacts equity
                f_pct = funding_flow / (cur_close + 1e-8)
                state.realized_equity += state.realized_equity * f_pct
                if active_trade:
                    active_trade["cum_funding"] += f_pct

            # 4. Check regular exit signal (if not stopped out)
            if state.position != 0 and not is_stopped:
                pos_dir = 1 if state.position > 0 else -1
                should_exit = False
                if pos_dir == 1 and cur_z < self.deadband:
                    should_exit = True
                elif pos_dir == -1 and cur_z > -self.deadband:
                    should_exit = True

                if should_exit:
                    fill_p = self.exec_model.get_fill_price(cur_close, side=-pos_dir)
                    gross_ret = (fill_p / state.entry_price - 1.0) * pos_dir
                    fee = self.exec_model.taker_fee * 2.0
                    net_ret = gross_ret - fee

                    if active_trade:
                        trades.append(TradeRecord(
                            symbol=self.symbol,
                            direction=pos_dir,
                            entry_time=active_trade["entry_time"],
                            exit_time=ts,
                            entry_price=state.entry_price,
                            exit_price=fill_p,
                            position_size=state.position_size,
                            gross_return=gross_ret,
                            net_return=net_ret + active_trade["cum_funding"],
                            fee_and_slippage=fee,
                            funding_fees=active_trade["cum_funding"],
                            duration_bars=i - active_trade["entry_idx"] + 1,
                            exit_reason="signal_exit",
                        ))
                        active_trade = None

                    state.realized_equity *= (1.0 + net_ret * state.position_size)
                    state.unrealized_pnl = 0.0
                    state.position = 0.0
                    state.position_size = 0.0
                    state.entry_price = 0.0
                    state.entry_time = None

                    if net_ret < 0:
                        state.loss_streak += 1
                    else:
                        state.loss_streak = 0

            # 5. Check Entry Signal (if cash and not in active cooldown)
            if state.position == 0:
                cur_peak = max(state.peak_equity, state.realized_equity)
                cur_dd = max(0.0, (cur_peak - state.realized_equity) / cur_peak)

                if self.trial_mode:
                    # Multi-downsizing factors
                    m_str = 1.0 if state.loss_streak == 0 else (0.70 if state.loss_streak == 1 else (0.50 if state.loss_streak == 2 else 0.25))
                    m_dd = 1.0 if cur_dd <= 0.04 else (0.75 if cur_dd <= 0.08 else (0.50 if cur_dd <= 0.12 else 0.25))
                    m_v = m_vol_arr[i]
                    m_conf = 0.65 if abs(cur_z) < 1.4 else 1.0
                    m_tr_l = 0.40 if trend_arr[i] < 0 else 1.0
                    m_tr_s = 0.50 if trend_arr[i] > 0 else 1.0

                    s_long = float(np.clip(sz_l_arr[i] * m_str * m_dd * m_v * m_tr_l * m_conf, 0.15, 1.0))
                    s_short = float(np.clip(sz_s_arr[i] * m_str * m_dd * m_v * m_tr_s * m_conf, 0.15, 1.0))
                else:
                    s_long = float(sz_l_arr[i])
                    s_short = float(sz_s_arr[i])

                if cur_z > 1.0 and fng_arr[i] < 85 and not state.in_waterfall:
                    fill_p = self.exec_model.get_fill_price(cur_close, side=1)
                    state.position = s_long
                    state.position_size = s_long
                    state.entry_price = fill_p
                    state.entry_time = str(ts)
                    active_trade = {
                        "direction": 1,
                        "entry_time": ts,
                        "entry_price": fill_p,
                        "position_size": s_long,
                        "entry_idx": i,
                        "cum_funding": 0.0,
                    }
                elif self.use_short and cur_z < -1.0 and fng_arr[i] > 15 and not state.in_short_squeeze:
                    fill_p = self.exec_model.get_fill_price(cur_close, side=-1)
                    state.position = -s_short
                    state.position_size = s_short
                    state.entry_price = fill_p
                    state.entry_time = str(ts)
                    active_trade = {
                        "direction": -1,
                        "entry_time": ts,
                        "entry_price": fill_p,
                        "position_size": s_short,
                        "entry_idx": i,
                        "cum_funding": 0.0,
                    }

            # 6. Update Mark-to-Market PnL & Drawdown at close
            if state.position != 0:
                pos_dir = 1 if state.position > 0 else -1
                cur_unrealized_pct = (cur_close / state.entry_price - 1.0) * pos_dir
                state.unrealized_pnl = state.realized_equity * state.position_size * cur_unrealized_pct
            else:
                state.unrealized_pnl = 0.0

            total_equity = state.realized_equity + state.unrealized_pnl
            state.peak_equity = max(state.peak_equity, total_equity)
            state.latest_drawdown = max(0.0, (state.peak_equity - total_equity) / (state.peak_equity + 1e-8))
            state.timestamp = str(ts)

            # Record snapshot
            equity_records.append(total_equity)
            position_records.append(state.position)
            state_history.append(StrategyState(
                timestamp=state.timestamp,
                position=state.position,
                position_size=state.position_size,
                entry_time=state.entry_time,
                entry_price=state.entry_price,
                loss_streak=state.loss_streak,
                in_waterfall=state.in_waterfall,
                in_short_squeeze=state.in_short_squeeze,
                realized_equity=state.realized_equity,
                unrealized_pnl=state.unrealized_pnl,
                peak_equity=state.peak_equity,
                latest_drawdown=state.latest_drawdown,
            ))

        eq_series = pd.Series(equity_records, index=common_idx, name=f"{self.symbol}_equity")
        pos_series = pd.Series(position_records, index=common_idx, name=f"{self.symbol}_position")

        return BacktestResult(
            symbol=self.symbol,
            equity_series=eq_series,
            position_series=pos_series,
            state_history=state_history,
            trades=trades,
            final_state=state,
            common_index=common_idx,
        )


@dataclass
class BacktestResult:
    symbol: str
    equity_series: pd.Series
    position_series: pd.Series
    state_history: List[StrategyState]
    trades: List[TradeRecord]
    final_state: StrategyState
    common_index: pd.DatetimeIndex

    def slice_report(self, start_date: str, end_date: str) -> SliceReport:
        """
        Slices the continuous MTM result without re-zeroing any indicators, z-scores,
        loss streaks, or carryover positions.
        """
        sub_eq = self.equity_series.loc[start_date:end_date]
        if len(sub_eq) == 0:
            raise ValueError(f"No equity data available between {start_date} and {end_date}")

        # Normalized equity curve inside the slice
        norm_eq = sub_eq / sub_eq.iloc[0]
        total_ret = float(norm_eq.iloc[-1] - 1.0)

        # Max drawdown inside slice
        cum_max = norm_eq.cummax()
        dd_series = (norm_eq - cum_max) / cum_max
        max_dd = float(dd_series.min())

        # GIPS daily resampled Sharpe
        daily_eq = sub_eq.resample("1D").last().dropna()
        daily_rets = daily_eq.pct_change().dropna()
        if len(daily_rets) > 1 and daily_rets.std() > 1e-8:
            daily_sharpe = float((daily_rets.mean() / daily_rets.std()) * np.sqrt(365))
        else:
            daily_sharpe = 0.0

        # Calmar
        days = (pd.to_datetime(end_date) - pd.to_datetime(start_date)).days
        ann_factor = 365.0 / max(1.0, days)
        ann_ret = (1.0 + total_ret) ** ann_factor - 1.0 if total_ret > -1.0 else -1.0
        calmar = float(ann_ret / abs(max_dd)) if abs(max_dd) > 1e-4 else 0.0

        # Trades occurring within slice
        start_ts = pd.to_datetime(start_date)
        end_ts = pd.to_datetime(end_date)

        sub_trades = [
            t for t in self.trades
            if start_ts <= t.exit_time <= end_ts or (t.entry_time <= end_ts and t.exit_time >= start_ts)
        ]

        win_trades = [t for t in sub_trades if t.net_return > 0]
        loss_trades = [t for t in sub_trades if t.net_return <= 0]
        win_rate = len(win_trades) / len(sub_trades) if sub_trades else 0.0

        gross_gains = sum(t.net_return * t.position_size for t in win_trades)
        gross_losses = abs(sum(t.net_return * t.position_size for t in loss_trades))
        profit_factor = (gross_gains / gross_losses) if gross_losses > 1e-8 else (99.0 if gross_gains > 0 else 1.0)

        # Carryover position at start_date
        carryover_pos = None
        start_idx = self.equity_series.index.get_indexer([start_ts], method="nearest")[0]
        state_at_start = self.state_history[start_idx]
        if state_at_start.position != 0:
            carryover_pos = {
                "symbol": self.symbol,
                "position": state_at_start.position,
                "entry_time": state_at_start.entry_time,
                "entry_price": state_at_start.entry_price,
                "unrealized_pnl_pct": round(state_at_start.unrealized_pnl / (state_at_start.realized_equity + 1e-8) * 100.0, 2),
            }

        return SliceReport(
            start_date=start_date,
            end_date=end_date,
            total_return=total_ret,
            max_drawdown=max_dd,
            daily_sharpe=daily_sharpe,
            calmar_ratio=calmar,
            win_rate=win_rate,
            profit_factor=profit_factor,
            total_trades=len(sub_trades),
            carryover_position=carryover_pos,
            equity_series=sub_eq,
            trades=sub_trades,
        )

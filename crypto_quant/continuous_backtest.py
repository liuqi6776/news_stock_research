# -*- coding: utf-8 -*-
"""
Continuous State Backtest Engine (Phase 17)
Eliminates segmented slice state-reset artifacts and look-ahead timing biases:
1. Runs full history continuously (e.g. 2024-01-01 to 2026-09-13) in a single pass.
2. Causal execution timing: signals confirmed at bar t close -> orders filled at bar t+1 open.
3. Intrabar high/low stop-loss piercing with conservative gap-open slippage model.
4. Complete StrategyState persistence & true restart equivalence (0 position/equity mismatches).
5. Provides non-resetting slice_report(start, end) reporting carryover positions and continuous MTM.
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
    in_waterfall: bool = False  # Long stop-loss cooldown
    in_short_squeeze: bool = False  # Short stop-loss cooldown
    realized_equity: float = 1.0
    unrealized_pnl: float = 0.0
    peak_equity: float = 1.0
    latest_drawdown: float = 0.0
    active_trade: Optional[Dict[str, Any]] = None
    pending_order: Optional[Dict[str, Any]] = None
    rolling_preds_buffer: List[float] = field(default_factory=list)
    rolling_closes_buffer: List[float] = field(default_factory=list)
    rolling_highs_buffer: List[float] = field(default_factory=list)
    rolling_lows_buffer: List[float] = field(default_factory=list)
    ewm72_sum_w: float = 0.0
    ewm72_sum_wx: float = 0.0
    ewm144_sum_w: float = 0.0
    ewm144_sum_wx: float = 0.0
    cum_funding_recorded: float = 0.0

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
    Executes a single continuous multi-year simulation with causal timing and institutional risk controls.
    Signals confirmed at bar t close -> orders executed at bar t+1 open.
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
        cost_filter_mult: float = 0.0,
        latency_penalty: float = 0.0,
    ):
        self.symbol = symbol
        self.stop_loss = stop_loss
        self.deadband = deadband
        self.use_short = use_short
        self.trial_mode = trial_mode
        self.exec_model = execution_model or ExecutionModel(latency_penalty=latency_penalty)
        if latency_penalty > 0.0 and execution_model is not None:
            self.exec_model.latency_penalty = latency_penalty
        self.initial_state = initial_state
        self.cost_filter_mult = cost_filter_mult
        self.latency_penalty = latency_penalty

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

        n = len(common_idx)
        if self.initial_state is not None:
            state = StrategyState.from_dict(self.initial_state.to_dict())
        else:
            state = StrategyState(timestamp=str(common_idx[0]))

        state_history: List[StrategyState] = []
        equity_records = []
        position_records = []
        trades: List[TradeRecord] = []
        filter_stats = {
            "total_signals_generated": 0,
            "signals_cost_filtered": 0,
            "signals_executed": 0,
        }

        active_trade: Optional[Dict[str, Any]] = None
        if state.active_trade is not None:
            active_trade = dict(state.active_trade)
            active_trade["entry_time"] = pd.to_datetime(active_trade["entry_time"])
        elif state.position != 0:
            active_trade = {
                "direction": 1 if state.position > 0 else -1,
                "entry_time": pd.to_datetime(state.entry_time) if state.entry_time else common_idx[0],
                "entry_price": state.entry_price,
                "position_size": state.position_size,
                "entry_idx": 0,
                "cum_funding": state.cum_funding_recorded,
            }

        pending_order: Optional[Dict[str, Any]] = None
        if state.pending_order is not None:
            pending_order = dict(state.pending_order)

        opens = bars["open"].values
        highs = bars["high"].values
        lows = bars["low"].values
        closes = bars["close"].values
        preds_arr = preds.values
        fund_arr = funding.values
        fng_arr = fng.values

        # Recursive EWM weights
        alpha72 = 2.0 / (72.0 + 1.0)
        alpha144 = 2.0 / (144.0 + 1.0)

        w72 = float(state.ewm72_sum_w)
        wx72 = float(state.ewm72_sum_wx)
        w144 = float(state.ewm144_sum_w)
        wx144 = float(state.ewm144_sum_wx)

        roll_preds: List[float] = list(state.rolling_preds_buffer)
        roll_closes: List[float] = list(state.rolling_closes_buffer)
        roll_highs: List[float] = list(state.rolling_highs_buffer)
        roll_lows: List[float] = list(state.rolling_lows_buffer)

        for i in range(n):
            ts = common_idx[i]
            cur_open = float(opens[i])
            cur_high = float(highs[i])
            cur_low = float(lows[i])
            cur_close = float(closes[i])
            cur_pred = float(preds_arr[i])
            cur_fund = float(fund_arr[i])
            cur_fng = float(fng_arr[i])

            # ----------------------------------------------------
            # Causal Indicators for bar i (Shift-1 from history)
            # ----------------------------------------------------
            if len(roll_preds) >= 2:
                pred_win = roll_preds[-72:]
                prior_mean = float(np.mean(pred_win))
                prior_std = float(np.std(pred_win, ddof=1)) + 1e-8
                cur_z = (cur_pred - prior_mean) / prior_std
            else:
                cur_z = 0.0

            if w72 > 0:
                cur_ema72 = wx72 / w72
                last_c = roll_closes[-1]
                stretch = (last_c - cur_ema72) / (cur_ema72 + 1e-8)
            else:
                cur_ema72 = cur_close
                stretch = 0.0

            if w144 > 0:
                cur_ema144 = wx144 / w144
                last_c = roll_closes[-1]
                trend_bias = last_c - cur_ema144
            else:
                trend_bias = 0.0

            stretch_risk = np.clip(stretch / 0.05, -1.0, 1.0)
            fund_risk = np.clip((cur_fund * 100.0) / 0.02, -1.0, 1.0)
            fng_risk = np.clip((cur_fng - 50.0) / 30.0, -1.0, 1.0)

            top_risk = 0.45 * max(0.0, stretch_risk) + 0.35 * max(0.0, fund_risk) + 0.20 * max(0.0, fng_risk)
            bot_risk = 0.45 * max(0.0, -stretch_risk) + 0.35 * max(0.0, -fund_risk) + 0.20 * max(0.0, -fng_risk)

            base_size_long = float(np.clip(1.0 - 0.65 * max(0.0, top_risk - 0.25) / 0.75, 0.35, 1.0))
            base_size_short = float(np.clip(1.0 - 0.65 * max(0.0, bot_risk - 0.25) / 0.75, 0.35, 1.0))

            # ATR 14
            if len(roll_closes) >= 14:
                # Compute TR for last 14 bars
                h_arr = np.array(roll_highs[-14:])
                l_arr = np.array(roll_lows[-14:])
                c_prev = np.array(roll_closes[-15:-1]) if len(roll_closes) > 14 else np.array(roll_closes[-14:])
                tr1 = h_arr - l_arr
                tr2 = np.abs(h_arr - c_prev)
                tr3 = np.abs(l_arr - c_prev)
                tr = np.maximum(tr1, np.maximum(tr2, tr3))
                atr14 = float(np.mean(tr))
            else:
                atr14 = cur_high - cur_low

            atr_ratio = atr14 / (cur_close + 1e-8)
            m_vol = float(np.clip(0.025 / (atr_ratio + 1e-8), 0.40, 1.10))

            # ----------------------------------------------------
            # 1. Check Gap Stop on Open (if in position)
            # ----------------------------------------------------
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
                if stop_res.is_stopped and stop_res.is_gap:
                    is_stopped = True
                    fill_p = stop_res.fill_price
                    gross_ret = (fill_p / state.entry_price - 1.0) * pos_dir
                    fee = self.exec_model.taker_fee * 2.0
                    net_ret = gross_ret - fee

                    cum_f = active_trade["cum_funding"] if active_trade else 0.0
                    entry_t = active_trade["entry_time"] if active_trade else ts
                    dur = i - (active_trade["entry_idx"] if active_trade else i) + 1

                    trades.append(TradeRecord(
                        symbol=self.symbol,
                        direction=pos_dir,
                        entry_time=entry_t,
                        exit_time=ts,
                        entry_price=state.entry_price,
                        exit_price=fill_p,
                        position_size=state.position_size,
                        gross_return=gross_ret,
                        net_return=net_ret + cum_f,
                        fee_and_slippage=fee,
                        funding_fees=cum_f,
                        duration_bars=dur,
                        exit_reason="gap_stop",
                    ))
                    active_trade = None
                    pending_order = None

                    state.realized_equity *= (1.0 + (net_ret + cum_f) * state.position_size)
                    state.unrealized_pnl = 0.0
                    state.position = 0.0
                    state.position_size = 0.0
                    state.entry_price = 0.0
                    state.entry_time = None
                    state.loss_streak += 1
                    state.cum_funding_recorded = 0.0

                    if pos_dir == 1:
                        state.in_waterfall = True
                    else:
                        state.in_short_squeeze = True

            # ----------------------------------------------------
            # 2. Execute Pending Orders at bar Open (Causal Timing)
            # ----------------------------------------------------
            if pending_order is not None and not is_stopped:
                order_type = pending_order["type"]
                side = pending_order["side"]

                if order_type == "exit":
                    fill_p = self.exec_model.get_fill_price(cur_open, side=side)
                    pos_dir = 1 if state.position > 0 else -1
                    gross_ret = (fill_p / state.entry_price - 1.0) * pos_dir
                    fee = self.exec_model.taker_fee * 2.0
                    net_ret = gross_ret - fee

                    cum_f = active_trade["cum_funding"] if active_trade else 0.0
                    entry_t = active_trade["entry_time"] if active_trade else ts
                    dur = i - (active_trade["entry_idx"] if active_trade else i) + 1

                    trades.append(TradeRecord(
                        symbol=self.symbol,
                        direction=pos_dir,
                        entry_time=entry_t,
                        exit_time=ts,
                        entry_price=state.entry_price,
                        exit_price=fill_p,
                        position_size=state.position_size,
                        gross_return=gross_ret,
                        net_return=net_ret + cum_f,
                        fee_and_slippage=fee,
                        funding_fees=cum_f,
                        duration_bars=dur,
                        exit_reason=pending_order.get("reason", "signal_exit"),
                    ))
                    active_trade = None

                    state.realized_equity *= (1.0 + (net_ret + cum_f) * state.position_size)
                    state.unrealized_pnl = 0.0
                    state.position = 0.0
                    state.position_size = 0.0
                    state.entry_price = 0.0
                    state.entry_time = None
                    state.cum_funding_recorded = 0.0

                    if net_ret < 0:
                        state.loss_streak += 1
                    else:
                        state.loss_streak = 0

                    pending_order = None

                elif order_type == "entry":
                    fill_p = self.exec_model.get_fill_price(cur_open, side=side)
                    size = pending_order["size"]

                    state.position = side * size
                    state.position_size = size
                    state.entry_price = fill_p
                    state.entry_time = str(ts)
                    state.cum_funding_recorded = 0.0

                    active_trade = {
                        "direction": side,
                        "entry_time": ts,
                        "entry_price": fill_p,
                        "position_size": size,
                        "entry_idx": i,
                        "cum_funding": 0.0,
                    }
                    pending_order = None

            # ----------------------------------------------------
            # 3. Check Intrabar Stop on High/Low (if in position)
            # ----------------------------------------------------
            if state.position != 0 and not is_stopped:
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
                    gross_ret = (fill_p / state.entry_price - 1.0) * pos_dir
                    fee = self.exec_model.taker_fee * 2.0
                    net_ret = gross_ret - fee

                    cum_f = active_trade["cum_funding"] if active_trade else 0.0
                    entry_t = active_trade["entry_time"] if active_trade else ts
                    dur = i - (active_trade["entry_idx"] if active_trade else i) + 1

                    trades.append(TradeRecord(
                        symbol=self.symbol,
                        direction=pos_dir,
                        entry_time=entry_t,
                        exit_time=ts,
                        entry_price=state.entry_price,
                        exit_price=fill_p,
                        position_size=state.position_size,
                        gross_return=gross_ret,
                        net_return=net_ret + cum_f,
                        fee_and_slippage=fee,
                        funding_fees=cum_f,
                        duration_bars=dur,
                        exit_reason="intrabar_stop",
                    ))
                    active_trade = None

                    state.realized_equity *= (1.0 + (net_ret + cum_f) * state.position_size)
                    state.unrealized_pnl = 0.0
                    state.position = 0.0
                    state.position_size = 0.0
                    state.entry_price = 0.0
                    state.entry_time = None
                    state.loss_streak += 1
                    state.cum_funding_recorded = 0.0

                    if pos_dir == 1:
                        state.in_waterfall = True
                    else:
                        state.in_short_squeeze = True

            # ----------------------------------------------------
            # 4. Discrete 8h Funding Settlement (if in position)
            # ----------------------------------------------------
            if state.position != 0 and self.exec_model.is_funding_settlement_bar(ts):
                funding_flow = self.exec_model.compute_funding_cashflow(
                    timestamp=ts,
                    signed_position=state.position,
                    funding_rate=cur_fund,
                    mark_price=cur_close,
                    enforce_settlement_hours=False,
                )
                f_pct = funding_flow / (cur_close + 1e-8)
                state.realized_equity += state.realized_equity * f_pct
                state.cum_funding_recorded += f_pct
                if active_trade:
                    active_trade["cum_funding"] += f_pct

            # ----------------------------------------------------
            # 5. Cooldown Release Check at Close of Bar
            # ----------------------------------------------------
            if state.in_waterfall and cur_close >= cur_open:
                state.in_waterfall = False
            if state.in_short_squeeze and cur_close <= cur_open:
                state.in_short_squeeze = False

            # ----------------------------------------------------
            # 6. Signal Generation at Bar Close -> Queues Pending Order for t+1 Open
            # ----------------------------------------------------
            if state.position != 0:
                pos_dir = 1 if state.position > 0 else -1
                should_exit = False
                if pos_dir == 1 and cur_z < self.deadband:
                    should_exit = True
                elif pos_dir == -1 and cur_z > -self.deadband:
                    should_exit = True

                if should_exit:
                    pending_order = {
                        "type": "exit",
                        "side": -pos_dir,
                        "signal_time": str(ts),
                        "reason": "signal_exit",
                    }
            elif state.position == 0 and pending_order is None:
                cur_peak = max(state.peak_equity, state.realized_equity)
                cur_dd = max(0.0, (cur_peak - state.realized_equity) / cur_peak)

                if self.trial_mode:
                    m_str = 1.0 if state.loss_streak == 0 else (0.70 if state.loss_streak == 1 else (0.50 if state.loss_streak == 2 else 0.25))
                    m_dd = 1.0 if cur_dd <= 0.04 else (0.75 if cur_dd <= 0.08 else (0.50 if cur_dd <= 0.12 else 0.25))
                    m_v = m_vol
                    m_conf = 0.65 if abs(cur_z) < 1.4 else 1.0
                    m_tr_l = 0.40 if trend_bias < 0 else 1.0
                    m_tr_s = 0.50 if trend_bias > 0 else 1.0

                    s_long = float(np.clip(base_size_long * m_str * m_dd * m_v * m_tr_l * m_conf, 0.15, 1.0))
                    s_short = float(np.clip(base_size_short * m_str * m_dd * m_v * m_tr_s * m_conf, 0.15, 1.0))
                else:
                    s_long = base_size_long
                    s_short = base_size_short

                # Cost-Aware Execution Filter (Phase 18 P0 Enhancement)
                # Evaluates whether predicted return magnitude overcomes transaction friction + funding
                roundtrip_friction = (self.exec_model.taker_fee + self.exec_model.normal_slippage + self.exec_model.latency_penalty) * 2.0
                est_funding_cost = abs(cur_fund)
                cost_hurdle = self.cost_filter_mult * (roundtrip_friction + est_funding_cost)
                signal_magnitude = abs(cur_pred)

                passes_cost_filter = True
                if self.cost_filter_mult > 0.0:
                    passes_cost_filter = (signal_magnitude >= cost_hurdle)

                if cur_z > 1.0 and cur_fng < 85 and not state.in_waterfall:
                    filter_stats["total_signals_generated"] += 1
                    if passes_cost_filter:
                        filter_stats["signals_executed"] += 1
                        pending_order = {
                            "type": "entry",
                            "side": 1,
                            "size": s_long,
                            "signal_time": str(ts),
                        }
                    else:
                        filter_stats["signals_cost_filtered"] += 1
                elif self.use_short and cur_z < -1.0 and cur_fng > 15 and not state.in_short_squeeze:
                    filter_stats["total_signals_generated"] += 1
                    if passes_cost_filter:
                        filter_stats["signals_executed"] += 1
                        pending_order = {
                            "type": "entry",
                            "side": -1,
                            "size": s_short,
                            "signal_time": str(ts),
                        }
                    else:
                        filter_stats["signals_cost_filtered"] += 1

            # ----------------------------------------------------
            # 7. Update Mark-to-Market PnL & Drawdown at Bar Close
            # ----------------------------------------------------
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
            state.pending_order = dict(pending_order) if pending_order is not None else None
            state.active_trade = dict(active_trade) if active_trade is not None else None
            if state.active_trade is not None:
                state.active_trade["entry_time"] = str(state.active_trade["entry_time"])

            # ----------------------------------------------------
            # 8. Update Rolling Buffers and Recursive EWM State for bar i+1
            # ----------------------------------------------------
            roll_preds.append(cur_pred)
            roll_closes.append(cur_close)
            roll_highs.append(cur_high)
            roll_lows.append(cur_low)
            if len(roll_preds) > 72:
                roll_preds.pop(0)
            if len(roll_closes) > 72:
                roll_closes.pop(0)
                roll_highs.pop(0)
                roll_lows.pop(0)

            w72 = (1.0 - alpha72) * w72 + 1.0
            wx72 = (1.0 - alpha72) * wx72 + cur_close
            w144 = (1.0 - alpha144) * w144 + 1.0
            wx144 = (1.0 - alpha144) * wx144 + cur_close

            state.ewm72_sum_w = w72
            state.ewm72_sum_wx = wx72
            state.ewm144_sum_w = w144
            state.ewm144_sum_wx = wx144
            state.rolling_preds_buffer = list(roll_preds)
            state.rolling_closes_buffer = list(roll_closes)
            state.rolling_highs_buffer = list(roll_highs)
            state.rolling_lows_buffer = list(roll_lows)

            # Record snapshot
            equity_records.append(total_equity)
            position_records.append(state.position)
            state_history.append(StrategyState.from_dict(state.to_dict()))

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
            filter_stats=filter_stats,
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
    filter_stats: Dict[str, int] = field(default_factory=dict)

    def slice_report(self, start_date: str, end_date: str) -> SliceReport:
        """
        Slices the continuous MTM result without re-zeroing any indicators, z-scores,
        loss streaks, or carryover positions.
        """
        sub_eq = self.equity_series.loc[start_date:end_date]
        if len(sub_eq) == 0:
            raise ValueError(f"No equity data available between {start_date} and {end_date}")

        norm_eq = sub_eq / sub_eq.iloc[0]
        total_ret = float(norm_eq.iloc[-1] - 1.0)

        cum_max = norm_eq.cummax()
        dd_series = (norm_eq - cum_max) / cum_max
        max_dd = float(dd_series.min())

        daily_eq = sub_eq.resample("1D").last().dropna()
        daily_rets = daily_eq.pct_change().dropna()
        if len(daily_rets) > 1 and daily_rets.std() > 1e-8:
            daily_sharpe = float((daily_rets.mean() / daily_rets.std()) * np.sqrt(365))
        else:
            daily_sharpe = 0.0

        days = (pd.to_datetime(end_date) - pd.to_datetime(start_date)).days
        ann_factor = 365.0 / max(1.0, days)
        ann_ret = (1.0 + total_ret) ** ann_factor - 1.0 if total_ret > -1.0 else -1.0
        calmar = float(ann_ret / abs(max_dd)) if abs(max_dd) > 1e-6 else 0.0

        dt_start = pd.to_datetime(start_date)
        dt_end = pd.to_datetime(end_date)
        slice_trades = [t for t in self.trades if dt_start <= t.exit_time <= dt_end]

        if len(slice_trades) > 0:
            wins = [t for t in slice_trades if t.net_return > 0]
            losses = [t for t in slice_trades if t.net_return <= 0]
            win_rate = float(len(wins) / len(slice_trades))
            sum_gains = sum(t.net_return * t.position_size for t in wins)
            sum_losses = abs(sum(t.net_return * t.position_size for t in losses))
            profit_factor = float(sum_gains / (sum_losses + 1e-8)) if sum_losses > 0 else (99.0 if sum_gains > 0 else 1.0)
        else:
            win_rate = 0.0
            profit_factor = 0.0

        start_idx = self.equity_series.index.get_loc(sub_eq.index[0])
        st_at_start = self.state_history[start_idx]
        carryover = None
        if st_at_start.position != 0:
            carryover = {
                "position": st_at_start.position,
                "entry_time": st_at_start.entry_time,
                "entry_price": st_at_start.entry_price,
                "unrealized_pnl": st_at_start.unrealized_pnl,
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
            total_trades=len(slice_trades),
            carryover_position=carryover,
            equity_series=sub_eq,
            trades=slice_trades,
        )

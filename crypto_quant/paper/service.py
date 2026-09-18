# -*- coding: utf-8 -*-
"""
Service Module: Paper Signal Service Orchestration & Idempotency (Phase 22)
==========================================================================
Main service orchestrator handling historical gap replay, causal order execution,
strategy indicator calculation, atomic state persistence, and audit journaling.
"""

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

from crypto_quant.paper.config import (
    ENABLE_REAL_ORDERS,
    EXPERIMENT_ID,
    JOURNAL_PATH,
    MINIMUM_WARMUP_BARS,
    ONE_WAY_COST,
    PAPER_MODE,
    PORTFOLIO_WEIGHTS,
    STATE_PATH,
    SYMBOLS,
    TIMEFRAME,
    print_startup_banner,
)
from crypto_quant.paper.execution import PaperExecutionEngine, PaperFill, PaperOrder
from crypto_quant.paper.journal import EventTypes, PaperJournal
from crypto_quant.paper.market_data import DataValidationError, MarketDataFetcher
from crypto_quant.paper.monitoring import PaperMonitor
from crypto_quant.paper.portfolio import PaperPortfolioManager
from crypto_quant.paper.state import PaperStrategyState, PortfolioPaperState, StateCorruptionError
from crypto_quant.paper.strategy import StructuralTrendPaperStrategy


import crypto_quant.paper.config as paper_config


class UnrecoverableDataGapError(Exception):
    """Raised when a gap in 4h klines exceeds the fetch window or cannot be reconciled causally."""
    pass


class PaperService:
    """
    Continuous, idempotent, restart-safe Paper Signal Service.
    """

    def __init__(
        self,
        state_path: Union[str, Path] = STATE_PATH,
        journal_path: Union[str, Path] = JOURNAL_PATH,
        snapshot_path: Optional[Union[str, Path]] = None,
        symbols: Optional[List[str]] = None,
        one_way_cost: float = ONE_WAY_COST,
        dry_run: bool = False,
    ):
        # Strict compile-time and runtime safety enforcement
        if paper_config.ENABLE_REAL_ORDERS is not False:
            raise RuntimeError("CRITICAL: Real orders are strictly forbidden in PaperService!")
        if paper_config.PAPER_MODE is not True:
            raise RuntimeError("CRITICAL: PAPER_MODE must be True!")

        self.state_path = Path(state_path)
        self.journal_path = Path(journal_path)
        self.snapshot_path = Path(snapshot_path) if snapshot_path else Path(paper_config.STATUS_SNAPSHOT_PATH)
        self.symbols = symbols or SYMBOLS
        self.one_way_cost = one_way_cost
        self.dry_run = dry_run

        self.fetcher = MarketDataFetcher()
        self.journal = PaperJournal(self.journal_path)
        self.monitor = PaperMonitor(snapshot_path=self.snapshot_path)
        self.execution = PaperExecutionEngine(one_way_cost=self.one_way_cost)
        self.strategy = StructuralTrendPaperStrategy()
        self.portfolio_manager = PaperPortfolioManager(one_way_cost=self.one_way_cost)

        self.state: Optional[PortfolioPaperState] = None

    def load_or_initialize_state(self) -> PortfolioPaperState:
        """Loads existing state atomically or initializes a fresh state."""
        if not self.dry_run:
            paper_config.init_forward_experiment_file()
        if self.state_path.exists():
            self.state = PortfolioPaperState.load(self.state_path)
        else:
            self.state = PortfolioPaperState.create_initial()
        return self.state

    def run_once(self, external_candles: Optional[Dict[str, pd.DataFrame]] = None) -> Dict[str, Any]:
        """
        Executes a single evaluation pass:
        1. Validates configuration and safety flags.
        2. Fetches / validates closed 4h candles.
        3. Identifies unprocessed closed bars.
        4. Replays unprocessed gap bars in strict chronological order.
        5. Atomically saves state and records audit events.
        """
        from datetime import datetime, timezone

        self.journal.log_event(EventTypes.SERVICE_STARTED, {"dry_run": self.dry_run})
        self.load_or_initialize_state()

        if self.state.service_first_start_time is None:
            self.state.service_first_start_time = datetime.now(timezone.utc).isoformat()

        candles_data: Dict[str, pd.DataFrame] = {}
        candles_info: Dict[str, Any] = {"latest_closed_bars": {}, "data_age_seconds": {}}

        # Fetch market data for all symbols
        for sym in self.symbols:
            if external_candles and sym in external_candles:
                df = external_candles[sym]
            else:
                df = self.fetcher.fetch_closed_klines(sym, interval=TIMEFRAME, limit=250)
            candles_data[sym] = df

            latest_bar_time = str(df.index[-1])
            candles_info["latest_closed_bars"][sym] = latest_bar_time
            # Record data fetch event
            self.journal.log_event(
                EventTypes.DATA_FETCHED,
                {"rows": len(df), "latest_bar": latest_bar_time},
                symbol=sym,
                bar_time=latest_bar_time,
            )

        # Establish forward boundary at initial run if not set
        if self.state.forward_start_bar_time is None:
            common_latest = min([str(candles_data[s].index[-1]) for s in self.symbols])
            self.state.forward_start_bar_time = common_latest

        # Process each symbol through incremental state machine
        processed_counts = {}
        for sym in self.symbols:
            sym_state = self.state.eth_state if "ETH" in sym else self.state.sol_state
            n_proc = self._process_symbol_gap(sym, sym_state, candles_data[sym])
            processed_counts[sym] = n_proc

        # Update aggregate portfolio equity
        last_common_time = str(min([candles_data[s].index[-1] for s in self.symbols]))
        self.portfolio_manager.update_portfolio_aggregate(self.state, last_common_time)

        # Atomically save state unless dry_run
        if not self.dry_run:
            self.state.save_atomic(self.state_path)
            self.journal.log_event(
                EventTypes.STATE_SAVED,
                {"state_path": str(self.state_path), "portfolio_equity": self.state.portfolio_equity},
            )

        # Generate snapshot and format summary
        snapshot = self.monitor.generate_snapshot(self.state, candles_info)
        self.journal.log_event(EventTypes.SERVICE_STOPPED, {"processed_bars": processed_counts})

        return snapshot

    def _process_symbol_gap(
        self,
        symbol: str,
        state: PaperStrategyState,
        full_df: pd.DataFrame,
    ) -> int:
        """
        Finds all closed bars after state.last_processed_bar_time and processes
        them in strict chronological order.
        """
        last_proc_time = state.last_processed_bar_time

        if last_proc_time is None:
            # First run: warmup is required. We take all bars from minimum_warmup_bars - 1 to end
            unprocessed_idx = full_df.index[self.strategy.minimum_warmup_bars - 1 :]
        else:
            last_ts = pd.to_datetime(last_proc_time)
            unprocessed_idx = full_df.index[full_df.index > last_ts]

            if len(unprocessed_idx) > 0:
                expected_next = last_ts + pd.Timedelta(hours=4)
                first_unprocessed = unprocessed_idx[0]
                if first_unprocessed != expected_next:
                    raise UnrecoverableDataGapError(
                        f"Unrecoverable downtime gap detected for {symbol}: "
                        f"last processed bar was {last_ts}, expected next bar is {expected_next}, "
                        f"but received {first_unprocessed}. State advance blocked."
                    )

        if len(unprocessed_idx) == 0:
            # Idempotent skip
            return 0

        # Process each bar in the gap sequentially
        count = 0
        for current_bar_time in unprocessed_idx:
            self._step_bar(symbol, state, full_df, current_bar_time)
            count += 1

        return count

    def _step_bar(
        self,
        symbol: str,
        state: PaperStrategyState,
        full_df: pd.DataFrame,
        current_bar_time: pd.Timestamp,
    ) -> None:
        """
        Executes a single closed bar step causally:
        1. Evaluates open-to-open return for position held from previous bar.
        2. Fills any existing pending order at this bar's OPEN.
        3. Updates single asset equity, fees, and realized PnL.
        4. Evaluates strategy at this bar's CLOSE.
        5. Generates any new pending order to be filled at next bar's OPEN.
        6. Updates state.
        """
        bar_loc = full_df.index.get_loc(current_bar_time)
        bar_series = full_df.iloc[bar_loc]
        curr_open = float(bar_series["open"])
        curr_close = float(bar_series["close"])
        bar_time_str = str(current_bar_time)

        # Classify as RECOVERY_REPLAY vs TRUE_FORWARD
        is_recovery = False
        if self.state.forward_start_bar_time is not None:
            is_recovery = (bar_time_str < self.state.forward_start_bar_time)
        if not is_recovery:
            self.state.recovery_replay_completed = True

        # -------------------------------------------------------------
        # STEP 1: Update equity from previous completed bar interval
        # -------------------------------------------------------------
        self.portfolio_manager.update_single_asset_bar_return(
            state=state,
            curr_open=curr_open,
            curr_close=curr_close,
        )

        # -------------------------------------------------------------
        # STEP 2: Execute Pending Order at curr_open
        # -------------------------------------------------------------
        turnover = 0.0
        if state.pending_order is not None:
            p_order = PaperOrder.from_dict(state.pending_order)
            fill = self.execution.execute_fill(p_order, reference_open=curr_open, fill_time=bar_time_str)

            self.journal.log_event(
                EventTypes.ORDER_FILLED,
                {**fill.to_dict(), "is_recovery_replay": is_recovery},
                symbol=symbol,
                bar_time=bar_time_str,
            )

            if p_order.side == "BUY":
                turnover = p_order.quantity_fraction
                state.position = 1
                state.position_size = p_order.quantity_fraction
                state.entry_price = curr_open
                state.entry_time = bar_time_str
                state.highest_price_since_entry = curr_close
                state.trailing_stop_price = float(p_order.metadata.get("initial_stop", curr_close * 0.95))
                state.bars_in_position = 0
                self.journal.log_event(
                    EventTypes.POSITION_OPENED,
                    {"entry_price": state.entry_price, "size": state.position_size, "is_recovery_replay": is_recovery},
                    symbol=symbol,
                    bar_time=bar_time_str,
                )
            elif p_order.side == "SELL":
                turnover = state.position_size
                gross_ret = (curr_open / (state.entry_price + 1e-8)) - 1.0
                net_ret = gross_ret - 2.0 * self.one_way_cost

                self.journal.log_event(
                    EventTypes.POSITION_CLOSED,
                    {
                        "exit_price": curr_open,
                        "entry_price": state.entry_price,
                        "gross_ret": gross_ret,
                        "net_ret": net_ret,
                        "reason": p_order.reason,
                        "is_recovery_replay": is_recovery,
                    },
                    symbol=symbol,
                    bar_time=bar_time_str,
                )
                state.position = 0
                state.position_size = 0.0
                state.entry_price = 0.0
                state.entry_time = None
                state.highest_price_since_entry = 0.0
                state.trailing_stop_price = 0.0
                state.bars_in_position = 0

            # Pending order fulfilled
            state.pending_order = None

        # -------------------------------------------------------------
        # STEP 3: Record interval state for the upcoming bar
        # -------------------------------------------------------------
        state.last_bar_open = curr_open
        state.last_bar_pos = float(state.position) * float(state.position_size)
        state.last_bar_turnover = turnover

        # -------------------------------------------------------------
        # STEP 4: Evaluate Strategy at bar CLOSE
        # -------------------------------------------------------------
        history_df = full_df.iloc[: bar_loc + 1]
        signal, order_details, ind = self.strategy.evaluate_bar(history_df, state)

        # -------------------------------------------------------------
        # STEP 5: Emit New Pending Order for Next Bar OPEN
        # -------------------------------------------------------------
        if signal == "BUY" and order_details is not None:
            order = self.execution.create_order(
                symbol=symbol,
                side="BUY",
                quantity_fraction=order_details["quantity_fraction"],
                signal_time=bar_time_str,
                intended_fill_time="NEXT_BAR_OPEN",
                reason=order_details["reason"],
                metadata={"initial_stop": order_details["initial_stop"]},
            )
            state.pending_order = order.to_dict()
            state.last_signal = "BUY"
            self.journal.log_event(
                EventTypes.SIGNAL_CREATED,
                {"signal": "BUY", "indicators": ind, "is_recovery_replay": is_recovery},
                symbol=symbol,
                bar_time=bar_time_str,
            )
            self.journal.log_event(
                EventTypes.ORDER_CREATED,
                {**order.to_dict(), "is_recovery_replay": is_recovery},
                symbol=symbol,
                bar_time=bar_time_str,
            )

        elif signal == "EXIT" and order_details is not None:
            order = self.execution.create_order(
                symbol=symbol,
                side="SELL",
                quantity_fraction=order_details["quantity_fraction"],
                signal_time=bar_time_str,
                intended_fill_time="NEXT_BAR_OPEN",
                reason=order_details["reason"],
                metadata={"ratcheted_stop": order_details["ratcheted_stop"]},
            )
            state.pending_order = order.to_dict()
            state.last_signal = "EXIT"
            self.journal.log_event(
                EventTypes.SIGNAL_CREATED,
                {"signal": "EXIT", "reason": order_details["reason"], "is_recovery_replay": is_recovery},
                symbol=symbol,
                bar_time=bar_time_str,
            )
            self.journal.log_event(
                EventTypes.ORDER_CREATED,
                {**order.to_dict(), "is_recovery_replay": is_recovery},
                symbol=symbol,
                bar_time=bar_time_str,
            )

        if state.position == 1:
            state.bars_in_position += 1

        state.last_processed_bar_time = bar_time_str
        self.journal.log_event(
            EventTypes.BAR_ACCEPTED,
            {
                "close": curr_close,
                "position": state.position,
                "stop": state.trailing_stop_price,
                "total_equity": state.total_equity,
                "realized_equity": state.realized_equity,
                "is_recovery_replay": is_recovery,
            },
            symbol=symbol,
            bar_time=bar_time_str,
        )

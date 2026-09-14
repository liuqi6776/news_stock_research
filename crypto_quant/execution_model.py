# -*- coding: utf-8 -*-
"""
Execution Model for Crypto Quantitative System (Phase 16)
Provides institutional-grade intrabar execution simulation:
1. High/Low intrabar stop-loss piercing check
2. Conservative gap-open slippage model
3. Conservative signal priority: stop-loss overrides exit/reversal signals on same candle
4. Event-based 8h perpetual funding settlement (00:00, 08:00, 16:00 UTC)
"""

from dataclasses import dataclass
from typing import Tuple, Optional
import numpy as np
import pandas as pd


@dataclass
class StopCheckResult:
    is_stopped: bool
    fill_price: float
    reason: str  # 'none', 'gap_stop', 'intrabar_stop'
    is_gap: bool


class ExecutionModel:
    """
    Simulates realistic perpetual futures order fills, intrabar stop execution,
    slippage under gap conditions, and discrete 8h funding cash flows.
    """

    def __init__(
        self,
        taker_fee: float = 0.0004,
        stop_slippage: float = 0.0010,
        gap_slippage: float = 0.0015,
        normal_slippage: float = 0.0004,
        latency_penalty: float = 0.0,
    ):
        self.taker_fee = taker_fee
        self.stop_slippage = stop_slippage
        self.gap_slippage = gap_slippage
        self.normal_slippage = normal_slippage
        self.latency_penalty = latency_penalty

    def check_intrabar_stop(
        self,
        pos_direction: int,
        entry_price: float,
        stop_loss_pct: float,
        open_p: float,
        high_p: float,
        low_p: float,
        close_p: float,
    ) -> StopCheckResult:
        """
        Evaluates whether a stop-loss order was triggered during the bar.
        
        Rules:
        - Long:
            stop_price = entry_price * (1.0 - stop_loss_pct)
            If open_p <= stop_price:
                Gapped down below stop price at candle open.
                fill_price = open_p * (1.0 - self.gap_slippage)
                return StopCheckResult(True, fill_price, 'gap_stop', True)
            Else if low_p <= stop_price:
                Intrabar low breached stop price.
                fill_price = stop_price * (1.0 - self.stop_slippage)
                return StopCheckResult(True, fill_price, 'intrabar_stop', False)
        - Short:
            stop_price = entry_price * (1.0 + stop_loss_pct)
            If open_p >= stop_price:
                Gapped up above stop price at candle open.
                fill_price = open_p * (1.0 + self.gap_slippage)
                return StopCheckResult(True, fill_price, 'gap_stop', True)
            Else if high_p >= stop_price:
                Intrabar high breached stop price.
                fill_price = stop_price * (1.0 + self.stop_slippage)
                return StopCheckResult(True, fill_price, 'intrabar_stop', False)
        """
        if pos_direction == 1:  # Long
            stop_price = entry_price * (1.0 - stop_loss_pct)
            if open_p <= stop_price:
                fill_price = open_p * (1.0 - self.gap_slippage)
                return StopCheckResult(True, fill_price, "gap_stop", True)
            elif low_p <= stop_price:
                fill_price = stop_price * (1.0 - self.stop_slippage)
                return StopCheckResult(True, fill_price, "intrabar_stop", False)

        elif pos_direction == -1:  # Short
            stop_price = entry_price * (1.0 + stop_loss_pct)
            if open_p >= stop_price:
                fill_price = open_p * (1.0 + self.gap_slippage)
                return StopCheckResult(True, fill_price, "gap_stop", True)
            elif high_p >= stop_price:
                fill_price = stop_price * (1.0 + self.stop_slippage)
                return StopCheckResult(True, fill_price, "intrabar_stop", False)

        return StopCheckResult(False, 0.0, "none", False)

    def is_funding_settlement_bar(self, timestamp: pd.Timestamp) -> bool:
        """
        Binance Perpetual Funding settles at 00:00:00, 08:00:00, 16:00:00 UTC.
        In a 4-hour bar framework, candles starting at 00:00, 08:00, 16:00 UTC
        correspond to the settlement timestamp.
        """
        ts = pd.to_datetime(timestamp, utc=True)
        return ts.hour in (0, 8, 16) and ts.minute == 0

    def compute_funding_cashflow(
        self,
        timestamp: pd.Timestamp,
        signed_position: float,  # e.g. +0.75 for long, -0.50 for short
        funding_rate: float,
        mark_price: float,
        enforce_settlement_hours: bool = True,
    ) -> float:
        """
        Calculates the funding fee cash flow for the position:
        - Longs pay positive funding, receive negative funding
        - Shorts receive positive funding, pay negative funding
        Cashflow = - signed_position * funding_rate * mark_price
        """
        if enforce_settlement_hours and not self.is_funding_settlement_bar(timestamp):
            return 0.0

        return -float(signed_position * funding_rate * mark_price)

    def get_fill_price(
        self,
        intended_price: float,
        side: int,  # +1 buy, -1 sell
        is_aggressive: bool = True,
        latency_penalty: Optional[float] = None,
    ) -> float:
        """
        Calculates execution fill price with normal slippage and optional latency penalty.
        side = +1 (buy): fills higher
        side = -1 (sell): fills lower
        latency_penalty: models 50ms-300ms adverse selection / queue delay on taker orders.
        """
        pen = self.latency_penalty if latency_penalty is None else latency_penalty
        slip = (self.normal_slippage + pen) if is_aggressive else 0.0
        if side == 1:
            return intended_price * (1.0 + slip)
        else:
            return intended_price * (1.0 - slip)

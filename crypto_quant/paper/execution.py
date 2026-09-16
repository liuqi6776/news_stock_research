# -*- coding: utf-8 -*-
"""
Execution Module: Paper Orders, Simulated Fills & Friction Accounting (Phase 22)
==============================================================================
Simulates causal execution filled strictly at the open of bar t+1 following
a confirmed signal at the close of bar t. Tracks transparent cost decomposition.
"""

from dataclasses import asdict, dataclass, field
import hashlib
from typing import Any, Dict, Optional

from crypto_quant.paper.config import EXPERIMENT_ID, ONE_WAY_COST


@dataclass
class PaperOrder:
    """
    Paper order record emitted upon bar t close confirmation.
    """
    order_id: str
    symbol: str
    side: str  # 'BUY' or 'SELL'
    quantity_fraction: float
    signal_time: str
    intended_fill_time: str
    reason: str
    status: str = "PENDING"  # 'PENDING', 'FILLED', 'CANCELLED'
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "PaperOrder":
        return cls(**d)

    @staticmethod
    def generate_order_id(experiment_id: str, symbol: str, signal_time: str, side: str, reason: str) -> str:
        """Deterministically generates idempotent order ID."""
        content = f"{experiment_id}:{symbol}:{signal_time}:{side}:{reason}"
        return hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]


@dataclass
class PaperFill:
    """
    Simulated fill record executed at bar t+1 open.
    """
    order_id: str
    symbol: str
    side: str
    fill_time: str
    reference_open: float
    simulated_fill_price: float
    assumed_fee: float
    assumed_slippage: float
    total_assumed_cost: float
    quantity_fraction: float
    status: str = "FILLED"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "PaperFill":
        return cls(**d)


class PaperExecutionEngine:
    """
    Handles causal simulated fills with realistic cost decomposition.
    """

    def __init__(self, one_way_cost: float = ONE_WAY_COST):
        self.one_way_cost = one_way_cost
        # Transparent institutional split of assumed cost (4 bps fee + 4 bps spread/slippage)
        self.fee_share = self.one_way_cost * 0.5
        self.slippage_share = self.one_way_cost * 0.5

    def create_order(
        self,
        symbol: str,
        side: str,
        quantity_fraction: float,
        signal_time: str,
        intended_fill_time: str,
        reason: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> PaperOrder:
        """Creates deterministic, idempotent pending paper order."""
        order_id = PaperOrder.generate_order_id(
            experiment_id=EXPERIMENT_ID,
            symbol=symbol,
            signal_time=signal_time,
            side=side,
            reason=reason,
        )
        return PaperOrder(
            order_id=order_id,
            symbol=symbol,
            side=side,
            quantity_fraction=quantity_fraction,
            signal_time=signal_time,
            intended_fill_time=intended_fill_time,
            reason=reason,
            status="PENDING",
            metadata=metadata or {},
        )

    def execute_fill(self, order: PaperOrder, reference_open: float, fill_time: str) -> PaperFill:
        """
        Executes simulated fill at open price with directional friction penalty:
        - BUY: pays open * (1 + cost)
        - SELL: receives open * (1 - cost)
        """
        if reference_open <= 0:
            raise ValueError(f"Invalid reference open price: {reference_open}")

        if order.side.upper() == "BUY":
            fill_price = reference_open * (1.0 + self.one_way_cost)
        elif order.side.upper() == "SELL":
            fill_price = reference_open * (1.0 - self.one_way_cost)
        else:
            raise ValueError(f"Unknown order side: {order.side}")

        order.status = "FILLED"

        return PaperFill(
            order_id=order.order_id,
            symbol=order.symbol,
            side=order.side,
            fill_time=fill_time,
            reference_open=reference_open,
            simulated_fill_price=fill_price,
            assumed_fee=self.fee_share,
            assumed_slippage=self.slippage_share,
            total_assumed_cost=self.one_way_cost,
            quantity_fraction=order.quantity_fraction,
            status="FILLED",
        )

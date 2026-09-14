# -*- coding: utf-8 -*-
"""
Portfolio Mark-to-Market State and Risk Management Engine (Phase 16)
Upgrades single-asset closed-trade accounting into institutional mark-to-market portfolio risk management:
1. Continuous bar-by-bar MTM equity tracking including live unrealized PnL
2. Portfolio-level peak equity and joint drawdown calculation
3. Shared cross-asset drawdown throttling (jointly throttles ETH and SOL upon collective portfolio stress)
4. Cross-asset streak and exposure limits
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any
import numpy as np
import pandas as pd


@dataclass
class PortfolioState:
    timestamp: str
    cash: float = 1.0
    positions: Dict[str, float] = field(default_factory=dict)       # signed exposure per asset
    position_sizes: Dict[str, float] = field(default_factory=dict)  # absolute size per asset
    entry_prices: Dict[str, float] = field(default_factory=dict)
    mark_prices: Dict[str, float] = field(default_factory=dict)
    unrealized_pnls: Dict[str, float] = field(default_factory=dict)
    total_mtm_equity: float = 1.0
    peak_mtm_equity: float = 1.0
    portfolio_drawdown: float = 0.0
    joint_loss_streak: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "PortfolioState":
        return cls(**d)


class PortfolioRiskManager:
    """
    Coordinates risk across multiple asset sleeves (e.g. ETHUSDT + SOLUSDT).
    Prevents correlated blowout by throttling all assets when portfolio-level
    mark-to-market drawdown crosses risk thresholds.
    """

    def __init__(
        self,
        dd_warn_threshold: float = 0.04,   # 4% DD -> 0.75x
        dd_severe_threshold: float = 0.08, # 8% DD -> 0.50x
        dd_max_threshold: float = 0.12,    # 12% DD -> 0.25x
        max_gross_leverage: float = 1.50,
        max_net_leverage: float = 1.00,
    ):
        self.dd_warn_threshold = dd_warn_threshold
        self.dd_severe_threshold = dd_severe_threshold
        self.dd_max_threshold = dd_max_threshold
        self.max_gross_leverage = max_gross_leverage
        self.max_net_leverage = max_net_leverage

    def compute_portfolio_drawdown_multiplier(self, portfolio_dd: float) -> float:
        """
        Computes joint portfolio drawdown sizing throttle:
        - DD <= 4%: 1.0x (full allocation)
        - 4% < DD <= 8%: 0.75x (prudent reduction)
        - 8% < DD <= 12%: 0.50x (defensive halving)
        - DD > 12%: 0.25x (emergency throttle)
        """
        if portfolio_dd <= self.dd_warn_threshold:
            return 1.0
        elif portfolio_dd <= self.dd_severe_threshold:
            return 0.75
        elif portfolio_dd <= self.dd_max_threshold:
            return 0.50
        else:
            return 0.25

    def compute_cross_asset_streak_multiplier(self, joint_loss_streak: int) -> float:
        """
        Throttles new entry sizing if recent portfolio trades across assets
        have experienced consecutive stop-losses.
        """
        if joint_loss_streak == 0:
            return 1.0
        elif joint_loss_streak == 1:
            return 0.80
        elif joint_loss_streak == 2:
            return 0.60
        else:
            return 0.35

    def update_portfolio_state(
        self,
        current_state: PortfolioState,
        timestamp: pd.Timestamp,
        mark_prices: Dict[str, float],
    ) -> PortfolioState:
        """
        Re-evaluates portfolio mark-to-market equity and drawdown.
        """
        ts_str = str(timestamp)
        unrealized = {}
        total_pos_value = 0.0

        for sym, pos in current_state.positions.items():
            if pos != 0 and sym in mark_prices and sym in current_state.entry_prices:
                mp = mark_prices[sym]
                ep = current_state.entry_prices[sym]
                pos_dir = 1 if pos > 0 else -1
                pos_sz = current_state.position_sizes.get(sym, abs(pos))
                unrealized_pct = (mp / ep - 1.0) * pos_dir
                u_pnl = current_state.cash * pos_sz * unrealized_pct
                unrealized[sym] = u_pnl
                total_pos_value += u_pnl
            else:
                unrealized[sym] = 0.0

        total_mtm = current_state.cash + total_pos_value
        peak = max(current_state.peak_mtm_equity, total_mtm)
        dd = max(0.0, (peak - total_mtm) / (peak + 1e-8))

        return PortfolioState(
            timestamp=ts_str,
            cash=current_state.cash,
            positions=dict(current_state.positions),
            position_sizes=dict(current_state.position_sizes),
            entry_prices=dict(current_state.entry_prices),
            mark_prices=dict(mark_prices),
            unrealized_pnls=unrealized,
            total_mtm_equity=total_mtm,
            peak_mtm_equity=peak,
            portfolio_drawdown=dd,
            joint_loss_streak=current_state.joint_loss_streak,
        )

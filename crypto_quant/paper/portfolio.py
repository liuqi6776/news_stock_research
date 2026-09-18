# -*- coding: utf-8 -*-
"""
Portfolio Accounting Module: Sleeves, Shared Cash & Drawdowns (Phase 22)
=======================================================================
Tracks normalized equity for ETH sleeve, SOL sleeve, and 50/50 portfolio.
Maintains both independent sleeve accounting and shared cash accounting.
"""

from typing import Any, Dict, Optional, Tuple
from crypto_quant.paper.config import INITIAL_EQUITY, ONE_WAY_COST, PORTFOLIO_WEIGHTS
from crypto_quant.paper.state import PaperStrategyState, PortfolioPaperState


class PaperPortfolioManager:
    """
    Manages dual-asset paper accounts and aggregate portfolio equity.
    """

    def __init__(
        self,
        eth_weight: float = PORTFOLIO_WEIGHTS.get("ETHUSDT", 0.5),
        sol_weight: float = PORTFOLIO_WEIGHTS.get("SOLUSDT", 0.5),
        one_way_cost: float = ONE_WAY_COST,
    ):
        self.eth_weight = eth_weight
        self.sol_weight = sol_weight
        self.one_way_cost = one_way_cost

    def update_single_asset_bar_return(
        self,
        state: PaperStrategyState,
        curr_open: float = 0.0,
        next_open: Optional[float] = None,
        turnover: float = 0.0,
        funding_pnl: float = 0.0,
        holding_pos: Optional[float] = None,
        curr_close: Optional[float] = None,
    ) -> float:
        """
        Updates single-asset equity causally from open_k to open_{k+1}:
        bar_ret = active_pos * (open_{k+1} / open_k - 1) - turnover * cost + funding_pnl
        Supports both (curr_open, next_open) call style and incremental step style.
        """
        if next_open is not None:
            # Called with (curr_open, next_open)
            open_to_open_ret = (next_open / (curr_open + 1e-8)) - 1.0 if curr_open > 0 else 0.0
            active_pos = float(state.position) * float(state.position_size) if holding_pos is None else holding_pos
            fric_turnover = turnover
        else:
            # Called incrementally using state.last_bar_open
            if state.last_bar_open is not None and state.last_bar_open > 0:
                open_to_open_ret = (curr_open / (state.last_bar_open + 1e-8)) - 1.0
            else:
                open_to_open_ret = 0.0
            active_pos = state.last_bar_pos if holding_pos is None else holding_pos
            fric_turnover = state.last_bar_turnover if turnover == 0.0 else turnover

        friction = fric_turnover * self.one_way_cost
        bar_ret = active_pos * open_to_open_ret - friction + funding_pnl

        # Compound total equity
        state.total_equity *= (1.0 + bar_ret)
        state.total_fees += friction * 0.5
        state.total_slippage += friction * 0.5

        # Update peak and drawdown
        state.peak_equity = max(state.peak_equity, state.total_equity)
        if state.peak_equity > 1e-8:
            state.drawdown = (state.total_equity - state.peak_equity) / state.peak_equity
        else:
            state.drawdown = 0.0

        # Update realized vs unrealized equity
        if state.position == 0:
            state.realized_equity = state.total_equity
            state.unrealized_pnl = 0.0
        else:
            ref_c = curr_close if curr_close is not None else curr_open
            if state.entry_price > 0:
                state.unrealized_pnl = state.position_size * ((ref_c / state.entry_price) - 1.0)

        return bar_ret

    def update_portfolio_aggregate(self, portfolio_state: PortfolioPaperState, last_time: str) -> None:
        """
        Recomputes aggregate 50/50 portfolio equity and drawdown:
        portfolio_equity = w_eth * eth_equity + w_sol * sol_equity
        """
        eth_eq = portfolio_state.eth_state.total_equity
        sol_eq = portfolio_state.sol_state.total_equity

        port_eq = self.eth_weight * eth_eq + self.sol_weight * sol_eq
        portfolio_state.portfolio_equity = port_eq
        portfolio_state.portfolio_peak = max(portfolio_state.portfolio_peak, port_eq)

        if portfolio_state.portfolio_peak > 1e-8:
            portfolio_state.portfolio_drawdown = (
                (port_eq - portfolio_state.portfolio_peak) / portfolio_state.portfolio_peak
            )
        else:
            portfolio_state.portfolio_drawdown = 0.0

        portfolio_state.last_update_time = str(last_time)

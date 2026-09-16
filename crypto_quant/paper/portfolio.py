# -*- coding: utf-8 -*-
"""
Portfolio Accounting Module: Sleeves, Shared Cash & Drawdowns (Phase 22)
=======================================================================
Tracks normalized equity for ETH sleeve, SOL sleeve, and 50/50 portfolio.
Maintains both independent sleeve accounting and shared cash accounting.
"""

from typing import Any, Dict, Tuple
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
        curr_open: float,
        next_open: float,
        turnover: float,
        funding_pnl: float = 0.0,
    ) -> float:
        """
        Updates single-asset equity causally from open_k to open_k+1:
        bar_ret = active_pos * (open_{k+1} / open_k - 1) - turnover * cost + funding
        """
        if curr_open <= 0 or next_open <= 0:
            return 0.0

        open_to_open_ret = (next_open / curr_open) - 1.0
        active_pos = float(state.position) * float(state.position_size)
        friction = turnover * self.one_way_cost

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

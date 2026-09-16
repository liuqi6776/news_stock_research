# -*- coding: utf-8 -*-
"""
Unit Tests: Sleeve & 50/50 Portfolio Accounting (Phase 22)
==========================================================
Verifies:
1. test_eth_sleeve_accounting
2. test_sol_sleeve_accounting
3. test_50_50_portfolio_accounting
"""

import pytest
from crypto_quant.paper.portfolio import PaperPortfolioManager
from crypto_quant.paper.state import PaperStrategyState, PortfolioPaperState


def test_eth_sleeve_accounting():
    """Verifies single-asset ETH sleeve return compounding and friction deduction."""
    mgr = PaperPortfolioManager(one_way_cost=0.0008)
    state = PaperStrategyState(symbol="ETHUSDT", position=1, position_size=1.0)
    state.total_equity = 1.0

    # Holding 1.0 size from 2000 to 2100 (+5%), with 0 turnover
    bar_ret = mgr.update_single_asset_bar_return(state, curr_open=2000.0, next_open=2100.0, turnover=0.0)

    assert bar_ret == pytest.approx(0.05, rel=1e-7)
    assert state.total_equity == pytest.approx(1.05, rel=1e-7)
    assert state.peak_equity == pytest.approx(1.05, rel=1e-7)
    assert state.drawdown == 0.0


def test_sol_sleeve_accounting():
    """Verifies single-asset SOL sleeve return compounding and drawdown tracking."""
    mgr = PaperPortfolioManager(one_way_cost=0.0008)
    state = PaperStrategyState(symbol="SOLUSDT", position=1, position_size=1.0)
    state.total_equity = 1.0

    # Holding 1.0 size from 100 to 90 (-10%), with turnover = 1.0 (entered at this bar)
    bar_ret = mgr.update_single_asset_bar_return(state, curr_open=100.0, next_open=90.0, turnover=1.0)

    # -10% return - 8 bps friction = -0.1008
    assert bar_ret == pytest.approx(-0.1008, rel=1e-6)
    assert state.total_equity == pytest.approx(0.8992, rel=1e-6)
    assert state.drawdown == pytest.approx(-0.1008, rel=1e-6)


def test_50_50_portfolio_accounting():
    """Verifies aggregate 50/50 dual-asset portfolio equity calculation."""
    mgr = PaperPortfolioManager(eth_weight=0.5, sol_weight=0.5)
    port_state = PortfolioPaperState.create_initial()

    # ETH gained 10% (1.10), SOL lost 4% (0.96)
    port_state.eth_state.total_equity = 1.10
    port_state.sol_state.total_equity = 0.96

    mgr.update_portfolio_aggregate(port_state, last_time="2024-01-01 08:00:00")

    # 0.5 * 1.10 + 0.5 * 0.96 = 1.03
    assert port_state.portfolio_equity == pytest.approx(1.03, rel=1e-7)
    assert port_state.portfolio_peak == pytest.approx(1.03, rel=1e-7)
    assert port_state.portfolio_drawdown == 0.0

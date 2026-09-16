# -*- coding: utf-8 -*-
"""
Unit Tests: Paper Execution Timing & Cost Breakdown (Phase 22)
=============================================================
Verifies:
1. test_signal_at_close_fills_next_open
2. test_cost_breakdown_transparency
"""

import pytest
from crypto_quant.paper.execution import PaperExecutionEngine, PaperOrder


def test_signal_at_close_fills_next_open():
    """A signal confirmed at bar t close must be filled at bar t+1 open with cost penalty."""
    exec_engine = PaperExecutionEngine(one_way_cost=0.0008)

    order = exec_engine.create_order(
        symbol="ETHUSDT",
        side="BUY",
        quantity_fraction=1.0,
        signal_time="2024-01-01 04:00:00",
        intended_fill_time="2024-01-01 08:00:00",
        reason="BOLLINGER_BREAKOUT",
    )

    assert order.status == "PENDING"

    # Bar t+1 arrives with open price = 2500.0
    ref_open = 2500.0
    fill = exec_engine.execute_fill(order, reference_open=ref_open, fill_time="2024-01-01 08:00:00")

    assert fill.status == "FILLED"
    assert fill.reference_open == 2500.0
    # Buy fill price = 2500 * (1 + 0.0008) = 2502.0
    assert fill.simulated_fill_price == pytest.approx(2502.0, rel=1e-7)
    assert fill.total_assumed_cost == 0.0008
    assert fill.assumed_fee == 0.0004
    assert fill.assumed_slippage == 0.0004

    # Sell fill test
    sell_order = exec_engine.create_order(
        symbol="ETHUSDT",
        side="SELL",
        quantity_fraction=1.0,
        signal_time="2024-01-05 04:00:00",
        intended_fill_time="2024-01-05 08:00:00",
        reason="TRAILING_STOP",
    )
    sell_ref_open = 2600.0
    sell_fill = exec_engine.execute_fill(sell_order, reference_open=sell_ref_open, fill_time="2024-01-05 08:00:00")

    # Sell fill price = 2600 * (1 - 0.0008) = 2597.92
    assert sell_fill.simulated_fill_price == pytest.approx(2597.92, rel=1e-7)

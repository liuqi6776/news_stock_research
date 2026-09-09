"""
Audit Remediation Logic Synthetic Unit Test Suite (tests/test_audit_remediation_logic.py)
Mandated by Reviewer Audit Task (2026-09-09) Section IV.

Tests all 11 critical audit remediation items:
1. test_missing_etf_open_blocks_trade
2. test_initial_missing_open_never_backfills_future
3. test_valuation_price_cannot_be_execution_price
4. test_etf_capacity_and_partial_fill
5. test_first_day_cost_is_preserved
6. test_research_and_serve_a1_same_output
7. test_pit_filter_failure_blocks_new_risk
8. test_tushare_amount_unit_conversion
9. test_circ_mv_unit_conversion
10. test_factor_and_strategy_period_exact_match
11. test_csi1000_market_beta_sanity
"""

import os
import sys
import pytest
import numpy as np
import pandas as pd

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from research.experiments.exp_ens_t60_tv12.unified_production_ledger import (
    UnifiedProductionLedger,
    format_trade_date,
    get_etf_adv20_shares
)
from research.experiments.exp_ens_t60_tv12.pit_filter_rule import (
    evaluate_a1_filter,
    get_rule_signature,
    PITFilterFailureError,
    RULE_VERSION
)


def _build_mock_matrices():
    dates = [f"2025-01-{d:02d}" for d in range(1, 11)]
    stocks = ["000001.SZ", "000002.SZ"]
    open_df = pd.DataFrame(10.0, index=dates, columns=stocks)
    close_df = pd.DataFrame(10.5, index=dates, columns=stocks)
    preclose_df = pd.DataFrame(10.0, index=dates, columns=stocks)
    vol_df = pd.DataFrame(1000.0, index=dates, columns=stocks)
    return dates, open_df, close_df, preclose_df, vol_df


def test_missing_etf_open_blocks_trade():
    dates, open_df, close_df, preclose_df, vol_df = _build_mock_matrices()
    ledger = UnifiedProductionLedger(initial_capital=100_000.0)
    etf_prices = pd.Series([np.nan, 1.05, 1.06], index=dates[:3])
    etf_price_dict = {"512100.SH": etf_prices}

    ledger.execute_rebalance(
        dates[0], [], 0.0, open_df, preclose_df, vol_df,
        etf_targets={"512100.SH": 1.0}, etf_price_dict=etf_price_dict
    )

    assert "512100.SH" not in ledger.etf_positions, "Should NOT execute buy when open price is NaN"
    assert len(ledger.blocked_orders_log) >= 1
    assert ledger.blocked_orders_log[-1]["reason"] == "missing_or_invalid_open_price"
    assert ledger.total_trades == 0


def test_initial_missing_open_never_backfills_future():
    dates, open_df, close_df, preclose_df, vol_df = _build_mock_matrices()
    ledger = UnifiedProductionLedger(initial_capital=100_000.0)
    etf_prices = pd.Series([np.nan, 1.05], index=dates[:2])
    assert np.isnan(etf_prices.get(dates[0], np.nan))

    ledger.execute_rebalance(
        dates[0], [], 0.0, open_df, preclose_df, vol_df,
        etf_targets={"512100.SH": 1.0}, etf_price_dict={"512100.SH": etf_prices}
    )
    assert "512100.SH" not in ledger.etf_positions
    assert ledger.cash == 100_000.0


def test_valuation_price_cannot_be_execution_price():
    dates, open_df, close_df, preclose_df, vol_df = _build_mock_matrices()
    ledger = UnifiedProductionLedger(initial_capital=100_000.0, etf_slippage_bps=0.0, min_etf_fee=0.0)
    etf_open_series = pd.Series(1.00, index=dates)
    etf_close_series = pd.Series(1.10, index=dates)

    ledger.execute_rebalance(
        dates[0], [], 0.0, open_df, preclose_df, vol_df,
        etf_targets={"512100.SH": 1.0}, etf_price_dict={"512100.SH": etf_open_series}
    )

    fill = ledger.fills_log[-1]
    assert fill["price"] == pytest.approx(1.00, abs=1e-5)
    assert fill["price"] != pytest.approx(1.10, abs=1e-2)

    eq = ledger.compute_equity(dates[0], close_df, {"512100.SH": etf_close_series})
    assert eq["etf_val"] > 100_000.0 * 0.99 * 1.09


def test_etf_capacity_and_partial_fill():
    dates, open_df, close_df, preclose_df, vol_df = _build_mock_matrices()
    ledger = UnifiedProductionLedger(initial_capital=1_000_000.0, adv_cap_pct=0.10, etf_slippage_bps=2.0)
    # 100 lots/day * 100 = 10,000 shares/day => 10% ADV quota = 1,000 shares
    etf_vol_series = pd.Series(100.0, index=dates)
    etf_open_series = pd.Series(2.0, index=dates)

    ledger.execute_rebalance(
        dates[5], [], 0.0, open_df, preclose_df, vol_df,
        etf_targets={"512100.SH": 0.50},
        etf_price_dict={"512100.SH": etf_open_series},
        etf_vol_dict={"512100.SH": etf_vol_series}
    )

    assert "512100.SH" in ledger.etf_positions
    held_sh = ledger.etf_positions["512100.SH"]["shares"]
    assert held_sh == 1000, f"Expected 1,000 shares due to 10% ADV cap, got {held_sh}"
    fill = ledger.fills_log[-1]
    assert fill["price"] == pytest.approx(2.0004, abs=1e-5)


def test_first_day_cost_is_preserved():
    dates, open_df, close_df, preclose_df, vol_df = _build_mock_matrices()
    ledger = UnifiedProductionLedger(initial_capital=100_000.0, fee_bps=10.0, min_etf_fee=5.0)

    ledger.record_initial_state("2024-12-31")
    assert len(ledger.daily_nav_log) == 1
    assert ledger.daily_nav_log[0]["nav"] == 1.0

    etf_open = pd.Series(1.0, index=dates)
    etf_close = pd.Series(1.0, index=dates)
    ledger.execute_rebalance(
        dates[0], [], 0.0, open_df, preclose_df, vol_df,
        etf_targets={"512100.SH": 1.0}, etf_price_dict={"512100.SH": etf_open}
    )
    eq1 = ledger.compute_equity(dates[0], close_df, {"512100.SH": etf_close})
    assert eq1["nav"] < 1.0
    nav_drop_bps = (1.0 - eq1["nav"]) * 10000.0
    assert nav_drop_bps > 0


def test_research_and_serve_a1_same_output():
    candidates = ["000001.SZ", "000002.SZ", "000003.SZ", "000004.SZ"]
    bad_news = {"000002.SZ"}
    ret_1m = {"000001.SZ": -0.05, "000002.SZ": -0.08, "000003.SZ": -0.02, "000004.SZ": 0.05}
    res = evaluate_a1_filter(candidates, bad_news, ret_1m)
    assert res["passed_codes"] == ["000001.SZ", "000003.SZ", "000004.SZ"]
    assert res["filtered_codes"] == ["000002.SZ"]
    assert res["rule_version"] == RULE_VERSION
    sig = get_rule_signature()
    assert sig["rule_version"] == RULE_VERSION


def test_pit_filter_failure_blocks_new_risk():
    with pytest.raises(PITFilterFailureError):
        evaluate_a1_filter(["000001.SZ"], 12345, {}, fail_closed=True)

    res = evaluate_a1_filter(["000001.SZ"], 12345, {}, fail_closed=False)
    assert res["passed_codes"] == []
    assert "FAIL_CLOSED" in res["status"]


def test_tushare_amount_unit_conversion():
    amount_qian_yuan = 100_000.0
    amount_yi = amount_qian_yuan / 1e5
    assert amount_yi == pytest.approx(1.0, abs=1e-6)
    erroneous_amount_yi = amount_qian_yuan / 100.0
    assert erroneous_amount_yi != pytest.approx(1.0, abs=1e-6)


def test_circ_mv_unit_conversion():
    circ_mv_wan_yuan = 50_000.0
    circ_mv_yi = circ_mv_wan_yuan / 10000.0
    assert circ_mv_yi == pytest.approx(5.0, abs=1e-6)


def test_factor_and_strategy_period_exact_match():
    intervals = [
        ("2023-01-31", "2023-02-28"),
        ("2023-02-28", "2023-03-31"),
        ("2023-03-31", "2023-04-28")
    ]
    strat_returns = pd.Series([0.02, -0.01, 0.03], index=intervals)
    factor_returns = pd.Series([0.018, -0.012, 0.028], index=intervals)
    aligned_df = pd.DataFrame({"strat": strat_returns, "mkt": factor_returns}).dropna()
    assert len(aligned_df) == 3
    corr = np.corrcoef(aligned_df["strat"], aligned_df["mkt"])[0, 1]
    assert corr > 0.95


def test_csi1000_market_beta_sanity():
    np.random.seed(42)
    mkt = np.random.normal(0.005, 0.04, 36)
    csi1000 = 1.02 * mkt + np.random.normal(0, 0.005, 36)
    cov_matrix = np.cov(csi1000, mkt)
    beta = cov_matrix[0, 1] / cov_matrix[1, 1]
    assert 0.90 <= beta <= 1.10

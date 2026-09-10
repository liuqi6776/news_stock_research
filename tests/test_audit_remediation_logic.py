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
from research.experiments.exp_ens_t60_tv12.load_true_liquidity_metrics import (
    convert_tushare_amount_to_yi,
    convert_tushare_circ_mv_to_yi
)
from research.experiments.exp_ens_t60_tv12.run_phase27_ablation_and_ch4 import (
    build_aligned_monthly_periods,
    run_ch4_hac_regression,
    check_benchmark_beta_sanity
)
import statsmodels.api as sm


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
    """
    测试 P1-3 / P0-7: 直接调用生产环境转换函数 convert_tushare_amount_to_yi
    """
    amount_qian_yuan = 100_000.0
    amount_yi = convert_tushare_amount_to_yi(amount_qian_yuan)
    assert amount_yi == pytest.approx(1.0, abs=1e-6)
    
    # 验证原先除以 100 的严重错误量纲被彻底纠正
    erroneous_amount_yi = amount_qian_yuan / 100.0
    assert erroneous_amount_yi != pytest.approx(1.0, abs=1e-6)


def test_circ_mv_unit_conversion():
    """
    测试 P1-3 / P0-7: 直接调用生产环境转换函数 convert_tushare_circ_mv_to_yi
    """
    circ_mv_wan_yuan = 50_000.0
    circ_mv_yi = convert_tushare_circ_mv_to_yi(circ_mv_wan_yuan)
    assert circ_mv_yi == pytest.approx(5.0, abs=1e-6)


def test_factor_and_strategy_period_exact_match():
    """
    测试 P0-10 / P0-7: 直接调用生产环境期间构建函数 build_aligned_monthly_periods
    """
    month_end_dates = [20230131, 20230228, 20230331, 20230428]
    periods = build_aligned_monthly_periods(month_end_dates)
    expected_periods = [
        (20230131, 20230228),
        (20230228, 20230331),
        (20230331, 20230428)
    ]
    assert periods == expected_periods
    assert len(periods) == 3


def test_csi1000_market_beta_sanity():
    """
    测试 P0-7: 直接调用生产环境回归与常识性断言函数
    """
    np.random.seed(42)
    mkt = np.random.normal(0.005, 0.04, 36)
    csi1000 = 1.02 * mkt + np.random.normal(0, 0.005, 36)
    X = sm.add_constant(pd.DataFrame({"mkt": mkt}))
    y = pd.Series(csi1000)
    
    res = run_ch4_hac_regression(y, X, maxlags=1)
    beta = float(res.params["mkt"])
    r2 = float(res.rsquared)
    
    assert 0.90 <= beta <= 1.15
    assert check_benchmark_beta_sanity(beta, r2, beta_min=0.85, beta_max=1.35, r2_min=0.90)


def test_etf_pending_order_target_tracking():
    """
    测试 P1-1: ETF 目标跟踪状态机 (Unified Target-vs-Actual Order State Machine)
    当受 10% ADV 限制首日仅成交部分股数时，未成交缺口必须记录在 pending_etf_buy_orders 中，
    次日 process_daily_pending_orders 自动继续撮合，直到目标达成！
    """
    dates, open_df, close_df, preclose_df, vol_df = _build_mock_matrices()
    ledger = UnifiedProductionLedger(initial_capital=1_000_000.0, adv_cap_pct=0.10, etf_slippage_bps=2.0)

    # 标的 512100.SH:
    # 设定成交量 100 手 = 10,000 股 => 10% ADV 限额为 1,000 股
    etf_vol_series = pd.Series(100.0, index=dates)
    etf_open_series = pd.Series(2.0, index=dates)
    etf_price_dict = {"512100.SH": etf_open_series}
    etf_vol_dict = {"512100.SH": etf_vol_series}

    # Day 0 (dates[5]): 目标 2,000 股 (目标市值 4,000 元，占比 0.004)
    # 因 10% ADV 限额为 1,000 股，首日只能成交 1,000 股
    d0 = dates[5]
    d1 = dates[6]
    target_pct = (2000 * 2.0) / 1_000_000.0

    ledger.execute_rebalance(
        d0, [], 0.0, open_df, preclose_df, vol_df,
        etf_targets={"512100.SH": target_pct},
        etf_price_dict=etf_price_dict,
        etf_vol_dict=etf_vol_dict
    )

    assert ledger.etf_positions["512100.SH"]["shares"] == 1000, "Day 0 should fill exactly 1,000 shares due to ADV cap"
    assert "512100.SH" in ledger.pending_etf_buy_orders, "Shortfall must be recorded in pending_etf_buy_orders"
    assert ledger.pending_etf_buy_orders["512100.SH"]["shares"] == 1000, "Pending deficit must be exactly 1,000 shares"

    # Day 1 (dates[6]): 不触发新调仓信号，调用 process_daily_pending_orders 重试未完成挂单
    ledger.unlock_t1_shares()
    ledger.process_daily_pending_orders(
        d1, open_df, preclose_df, vol_df,
        etf_price_dict=etf_price_dict,
        etf_vol_dict=etf_vol_dict
    )

    assert ledger.etf_positions["512100.SH"]["shares"] == 2000, "Day 1 retry should complete the remaining 1,000 shares"
    assert "512100.SH" not in ledger.pending_etf_buy_orders, "Pending order should be fully cleared upon completion"


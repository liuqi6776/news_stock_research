# -*- coding: utf-8 -*-
"""alignment 测试: walk-forward 持有期拼接（rb 当日不含 / rb_next 当日含）、基准与组合月收益。"""
import numpy as np
import pandas as pd
import pytest

from repro_core import alignment


def test_hold_slices_excludes_rebal_day():
    trade = ["20200102", "20200103", "20200106", "20200107", "20200108"]
    rebal = ["20200103", "20200107"]
    slices = list(alignment.hold_slices(rebal, trade))
    assert slices[0] == ("20200103", "20200107", ["20200106", "20200107"])
    # rb 当日(20200103)不在持有期; rb_next 当日(20200107)在持有期


def test_hold_slices_skips_missing_dates():
    trade = ["20200102", "20200103", "20200106"]
    rebal = ["20200102", "20200110"]  # 20200110 不在交易日
    assert list(alignment.hold_slices(rebal, trade)) == []


def test_benchmark_monthly():
    daily = pd.Series([0.01, 0.02], index=["20200106", "20200107"])
    hold = ["20200106", "20200107"]
    assert alignment.benchmark_monthly(hold, daily) == pytest.approx(1.01 * 1.02 - 1)


def test_benchmark_monthly_missing_filled_zero():
    daily = pd.Series([0.01], index=["20200106"])
    assert alignment.benchmark_monthly(["20200106", "20200107"], daily) == pytest.approx(0.01)


def test_portfolio_monthly_equal_weight():
    # 2 只股票 2 天, 等权日收益均值连乘, 再减成本
    pct = pd.DataFrame({"A": [1.0, 2.0], "B": [3.0, -1.0]}, index=["d1", "d2"])
    hold = ["d1", "d2"]
    r_d1 = (0.01 + 0.03) / 2
    r_d2 = (0.02 - 0.01) / 2
    gross = (1 + r_d1) * (1 + r_d2) - 1
    assert alignment.portfolio_monthly(hold, ["A", "B"], pct, cost_bps=20.0) == pytest.approx(
        gross - 20 / 10000.0)


def test_portfolio_monthly_weighted():
    pct = pd.DataFrame({"A": [1.0], "B": [3.0]}, index=["d1"])
    w = pd.Series({"A": 0.25, "B": 0.75})
    # 日收益 = 0.25*0.01 + 0.75*0.03 = 0.025
    assert alignment.portfolio_monthly(["d1"], ["A", "B"], pct, cost_bps=0.0, weights=w) == pytest.approx(0.025)


def test_walk_forward_series_matches_manual():
    trade = ["d1", "d2", "d3", "d4", "d5"]
    rebal = ["d1", "d3", "d5"]
    pct = pd.DataFrame({"A": [1.0, 2.0, 3.0, 4.0, 5.0], "B": [0.5, 1.0, 1.5, 2.0, 2.5]},
                       index=trade)
    picks = {"d1": ["A", "B"], "d3": ["A"]}
    port = alignment.walk_forward_series(picks, rebal, trade, pct, cost_bps=10.0)
    assert list(port.index) == ["d3", "d5"]
    # 手动: 第一段 hold=[d2,d3], 等权
    r1 = ((1 + 0.01) * (1 + 0.03)) ** 0.5 * ((1 + 0.02) * (1 + 0.015)) ** 0.5 - 1  # 等权需逐日
    # 逐日等权: d2: (0.02+0.01)/2=0.015, d3: (0.03+0.015)/2=0.0225
    m1 = (1.015 * 1.0225 - 1) - 10 / 10000.0
    # 第二段 hold=[d4,d5], 仅 A: (1.04*1.05-1) - 0.001
    m2 = (1.04 * 1.05 - 1) - 10 / 10000.0
    assert port["d3"] == pytest.approx(m1)
    assert port["d5"] == pytest.approx(m2)


def test_benchmark_series_alignment():
    hold_map = {"20200331": ["20200401", "20200402"]}
    daily = pd.Series([0.01, 0.02], index=["20200401", "20200402"])
    s = alignment.benchmark_series(hold_map, daily)
    assert s["20200331"] == pytest.approx(1.01 * 1.02 - 1)

# -*- coding: utf-8 -*-
"""metrics 纯函数测试: MaxDD 相对口径 / CAGR / Sharpe / 汇总。"""
import numpy as np
import pandas as pd
import pytest

from repro_core import metrics


def test_max_drawdown_monotonic_zero():
    nav = pd.Series([1.0, 1.1, 1.2, 1.3])
    assert metrics.max_drawdown(nav) == pytest.approx(0.0)


def test_max_drawdown_relative():
    # 高水位 1.2 -> 谷 0.9: 相对回撤 = (1.2-0.9)/1.2 = 0.25
    nav = pd.Series([1.0, 1.1, 1.2, 0.9, 1.2])
    assert metrics.max_drawdown(nav) == pytest.approx(0.25)


def test_max_drawdown_rejects_nonpositive_nav():
    with pytest.raises(AssertionError):
        metrics.max_drawdown(pd.Series([1.0, 0.0, 1.0]))


def test_max_drawdown_bounds():
    rng = np.random.default_rng(1)
    nav = pd.Series((1 + rng.normal(0.01, 0.05, 200)).cumprod())
    mdd = metrics.max_drawdown(nav)
    assert 0.0 <= mdd <= 1.0


def test_cagr_single_period():
    # 2 个观测点, ppy=2 -> 1 年; final=1.2 -> cagr=0.2
    assert metrics.cagr(pd.Series([1.0, 1.2]), periods_per_year=2.0) == pytest.approx(0.2)


def test_cagr_two_years():
    # 24 个观测点, ppy=12 -> 2 年; final=1.21 -> cagr = sqrt(1.21)-1 = 0.1
    nav = pd.Series([1.0] + [1.21] * 23)
    assert metrics.cagr(nav, periods_per_year=12.0) == pytest.approx(0.1)


def test_sharpe_constant_returns_nan():
    assert np.isnan(metrics.sharpe(pd.Series([0.01, 0.01, 0.01])))


def test_sharpe_value():
    rets = pd.Series([0.01, -0.01, 0.02, -0.005])
    assert metrics.sharpe(rets, periods_per_year=12.0) == pytest.approx(
        rets.mean() / rets.std(ddof=1) * np.sqrt(12.0))


def test_win_rate():
    assert metrics.win_rate(pd.Series([0.1, -0.1, 0.05])) == pytest.approx(2 / 3)


def test_excess_return():
    pr = pd.Series([0.1, 0.1, 0.1])
    br = pd.Series([0.05, 0.05, 0.05])
    # (1.1^3 / 1.05^3) - 1
    assert metrics.excess_return(pr, br) == pytest.approx((1.1 / 1.05) ** 3 - 1)


def test_calmar():
    assert metrics.calmar(0.2, 0.1) == pytest.approx(2.0)
    assert np.isnan(metrics.calmar(0.2, 0.0))


def test_summary_from_returns_fields():
    rets = pd.Series(np.linspace(0.005, 0.02, 24))
    bm = pd.Series(0.008, index=rets.index)
    s = metrics.summary_from_returns(rets, bm)
    for k in ("n", "nav", "cagr", "sharpe", "mdd", "win", "calmar", "excess"):
        assert k in s
    assert s["n"] == 24
    assert 0.0 <= s["mdd"] <= 1.0

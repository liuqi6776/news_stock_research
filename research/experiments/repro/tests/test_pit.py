# -*- coding: utf-8 -*-
"""pit 测试: 未来收益不含当日(PIT) / 成分股权重 PIT / Rank IC / NW t。"""
import numpy as np
import pandas as pd
import pytest

from repro_core import pit


def test_forward_returns_excludes_current_day():
    pct = pd.Series([1.0] * 6, index=[f"d{i}" for i in range(6)])
    fwd = pit.forward_returns(pct, days=2)
    # 第 0 日: cum2/cum0 - 1 = 1.01^2 - 1（只含 d1,d2 两天, 不含 d0）
    assert fwd["d0"] == pytest.approx(1.01 ** 2 - 1)
    # 最后 2 个交易日无未来收益
    assert np.isnan(fwd["d4"]) and np.isnan(fwd["d5"])


def test_latest_index_weight_pit():
    ws = {"20200131": {"A", "B"}, "20200228": {"B", "C"}}
    assert pit.latest_index_weight(ws, "20200210") == {"A", "B"}   # 取 <= 的最近一期
    assert pit.latest_index_weight(ws, "20200305") == {"B", "C"}
    assert pit.latest_index_weight(ws, "20191201") is None          # 无可用 -> None


def test_rank_ic_perfect():
    f = pd.Series({"a": 1.0, "b": 2.0, "c": 3.0, "d": 4.0})
    r = pd.Series({"a": 0.01, "b": 0.02, "c": 0.03, "d": 0.04})
    assert pit.rank_ic(f, r, min_n=4) == pytest.approx(1.0)


def test_rank_ic_reverse():
    f = pd.Series({"a": 1.0, "b": 2.0, "c": 3.0, "d": 4.0})
    r = pd.Series({"a": 0.04, "b": 0.03, "c": 0.02, "d": 0.01})
    assert pit.rank_ic(f, r, min_n=4) == pytest.approx(-1.0)


def test_rank_ic_too_few():
    f = pd.Series({"a": 1.0, "b": 2.0})
    r = pd.Series({"a": 0.01, "b": 0.02})
    assert np.isnan(pit.rank_ic(f, r, min_n=50))


def test_newey_west_t_few_obs():
    t, m = pit.newey_west_t([0.1])
    assert t == 0.0


def test_newey_west_t_constant_series():
    t, m = pit.newey_west_t([0.05] * 30)
    assert abs(t) > 1e3       # 零方差 -> 极大 t
    assert m == pytest.approx(0.05)


def test_monthly_cross_section_ic_basic():
    factor_map = {c: pd.Series({rb: v}, index=[rb]) for c, rb, v in
                  [("A", "20200131", 1.0), ("B", "20200131", 2.0), ("C", "20200131", 3.0)]}
    fwd_map = {c: pd.Series({rb: r}, index=[rb]) for c, rb, r in
               [("A", "20200131", 0.01), ("B", "20200131", 0.02), ("C", "20200131", 0.03)]}
    ic = pit.monthly_cross_section_ic(factor_map, fwd_map, ["20200131"],
                                      {"20200131": {"A", "B", "C"}}, min_n=3)
    assert ic["20200131"] == pytest.approx(1.0)


def test_picks_top_n_uses_pit_membership():
    factor_map = {
        "A": pd.Series({"20200131": 9.0}), "B": pd.Series({"20200131": 8.0}),
        "C": pd.Series({"20200131": 7.0}), "D": pd.Series({"20200131": 6.0}),
    }
    picks = pit.picks_top_n(factor_map, ["20200131"], {"20200131": {"A", "B", "D"}}, top_n=2)
    assert picks["20200131"] == ["A", "B"]   # C 不在成分内, 排除

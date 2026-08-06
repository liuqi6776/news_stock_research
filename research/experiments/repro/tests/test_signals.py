# -*- coding: utf-8 -*-
"""signals 测试: MA20 三档/五档/N档、VolTarget、DD 触发、CPPI/TIPP、T-1 位移。"""
import numpy as np
import pandas as pd
import pytest

from repro_core import signals


def test_ma20_3_tier_full_half_zero():
    assert signals.ma20_3_tier(100.0, 100.0) == pytest.approx(1.0)   # c >= m
    assert signals.ma20_3_tier(99.0, 100.0, deep=0.98) == pytest.approx(0.5)  # 0.98m <= c < m
    assert signals.ma20_3_tier(97.0, 100.0, deep=0.98) == pytest.approx(0.0)  # c < 0.98m
    assert np.isnan(signals.ma20_3_tier(np.nan, 100.0))


def test_ma20_n_tier():
    b = [1.0, 0.99, 0.98, 0.97]
    w = [1.0, 0.75, 0.5, 0.25]
    assert signals.ma20_n_tier(100.0, 100.0, b, w) == pytest.approx(1.0)
    assert signals.ma20_n_tier(99.5, 100.0, b, w) == pytest.approx(0.75)
    assert signals.ma20_n_tier(98.5, 100.0, b, w) == pytest.approx(0.5)
    assert signals.ma20_n_tier(97.2, 100.0, b, w) == pytest.approx(0.25)
    assert signals.ma20_n_tier(96.0, 100.0, b, w) == pytest.approx(0.0)


def test_t1_shift_no_lookahead():
    s = pd.Series([1.0, 2.0, 3.0], index=["a", "b", "c"])
    s1 = signals.t1_shift(s)
    assert np.isnan(s1["a"])      # 首日无 T-1 信息
    assert s1["b"] == 1.0         # T 日信号只用 T-1 日信息
    assert s1["c"] == 2.0


def test_vol_target():
    assert signals.vol_target(0.10, tgt=0.20, floor_w=0.20) == pytest.approx(1.0)   # 2.0 -> clip 1.0
    assert signals.vol_target(0.40, tgt=0.20, floor_w=0.20) == pytest.approx(0.5)
    assert signals.vol_target(0.80, tgt=0.20, floor_w=0.20) == pytest.approx(0.25)  # 0.25 (≥ floor 0.2)
    assert signals.vol_target(1.50, tgt=0.20, floor_w=0.20) == pytest.approx(0.2)   # 0.133 -> floor 0.2
    assert np.isnan(signals.vol_target(np.nan))


def test_vol_penalty():
    assert signals.vol_penalty(0.20, v_hi=0.30, v_lo=0.50) == pytest.approx(1.0)
    # v=0.40 -> (0.50-0.40)/(0.50-0.30)=0.5
    assert signals.vol_penalty(0.40, v_hi=0.30, v_lo=0.50, v_min=0.5) == pytest.approx(0.5)
    assert signals.vol_penalty(np.nan) == pytest.approx(1.0)


def test_dd_weight_half_and_zero():
    par = dict(half=-0.15, zero=-0.25, fix=-0.05)
    # 回撤 -0.20 -> 半仓
    w, wh = signals.dd_weight(-0.20, **par, w_half=False)
    assert w == pytest.approx(0.5) and wh is True
    # 回撤 -0.30 -> 空仓
    w, wh = signals.dd_weight(-0.30, **par, w_half=True)
    assert w == pytest.approx(0.0) and wh is True
    # 修复到 -0.03 -> 解除半仓, 满仓
    w, wh = signals.dd_weight(-0.03, **par, w_half=True)
    assert w == pytest.approx(1.0) and wh is False


def test_cppi_tipp_floor_ratchet():
    w, floor = signals.cppi_weight(nav=1.0, floor=0.80, hwm=1.0, m=3.0, alpha=0.90)
    assert floor == pytest.approx(0.90)          # floor = max(0.8, 0.9*1.0)
    assert w == pytest.approx(0.3)               # 3*(1.0-0.9)/1.0
    # hwm 上移 -> floor 只升不降（TIPP 棘轮）
    _, floor2 = signals.cppi_weight(nav=1.2, floor=0.90, hwm=1.2, m=3.0, alpha=0.90)
    assert floor2 == pytest.approx(1.08)


def test_ma20_trend():
    assert signals.ma20_trend(100.0, 100.0) == pytest.approx(1.0)
    assert signals.ma20_trend(98.0, 100.0) == pytest.approx(0.5)   # (0.98-0.96)/0.04
    assert signals.ma20_trend(95.0, 100.0) == pytest.approx(0.0)


def test_position_weight_ma20_uses_t1_values():
    # 传入的是 T-1 序列的值; 语义验证: 与 ma20_3_tier 一致
    w, wh, fl = signals.position_weight("ma20", {"deep": 0.98}, c=99.0, m=100.0,
                                        v=np.nan, mo=np.nan, dd_prev=0.0,
                                        w_half=False, nav=1.0, floor=0.9, hwm=1.0)
    assert w == pytest.approx(0.5) and wh is False and fl == pytest.approx(0.9)


def test_position_weight_unknown_type_raises():
    with pytest.raises(ValueError):
        signals.position_weight("unknown", {}, np.nan, np.nan, np.nan, np.nan,
                                0.0, False, 1.0, 0.9, 1.0)

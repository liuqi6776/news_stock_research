# -*- coding: utf-8 -*-
"""hrp 测试: 权重和=1 / 非负 / 降级路径（单列、少样本 -> 等权）。"""
import numpy as np
import pandas as pd
import pytest

from repro_core import hrp


def test_hrp_weights_sum_to_one_and_nonneg():
    rng = np.random.default_rng(0)
    rets = pd.DataFrame(rng.normal(0.001, 0.02, size=(240, 8)),
                        columns=[f"s{i}" for i in range(8)])
    w = hrp.hrp_weights(rets)
    assert w.sum() == pytest.approx(1.0)
    assert (w >= -1e-9).all()
    assert len(w) == 8


def test_hrp_weights_deterministic():
    rng = np.random.default_rng(1)
    rets = pd.DataFrame(rng.normal(0.0, 0.02, size=(200, 5)))
    w1 = hrp.hrp_weights(rets)
    w2 = hrp.hrp_weights(rets)
    assert w1.equals(w2)


def test_hrp_single_column_equal_weight():
    rets = pd.DataFrame({"A": np.random.default_rng(0).normal(0, 0.02, 100)})
    w = hrp.hrp_weights(rets)
    assert w["A"] == pytest.approx(1.0)


def test_hrp_too_few_obs_equal_weight():
    rng = np.random.default_rng(0)
    rets = pd.DataFrame(rng.normal(0, 0.02, size=(10, 4)), columns=list("ABCD"))  # < min_obs=20
    w = hrp.hrp_weights(rets)
    assert w.sum() == pytest.approx(1.0)
    assert w.sub(0.25).abs().max() < 1e-9


def test_hrp_weight_map_uses_prior_window():
    trade = [f"d{i}" for i in range(30)]
    pct = pd.DataFrame(np.random.default_rng(2).normal(0, 1, size=(30, 3)),
                       index=trade, columns=list("ABC"))
    picks = {"d20": ["A", "B", "C"]}
    wmap = hrp.hrp_weight_map(picks, ["d20"], trade, pct, window=10)
    # 窗口取 [d10, d19]（不含调仓日 d20）—— T-1 已知
    assert "d20" in wmap
    assert wmap["d20"].sum() == pytest.approx(1.0)

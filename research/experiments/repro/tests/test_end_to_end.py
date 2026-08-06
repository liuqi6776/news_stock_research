# -*- coding: utf-8 -*-
"""端到端回归: 固定种子复现实验 == 冻结的 expected_metrics（容差 1%）。

冻结流程: python research/experiments/repro/scripts/generate_expected.py
          -> 人工核对指标 -> 冻结 expected_metrics/metrics.json
本测试重跑同一代码路径, 验证"核心逻辑 + 固定种子"的复现稳定性（外部可独立执行）。
"""
import json
import os

import numpy as np
import pytest

from repro_core import experiment

REPRO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXPECTED = os.path.join(REPRO_DIR, "expected_metrics", "metrics.json")


def _load_expected():
    if not os.path.isfile(EXPECTED):
        pytest.skip("expected_metrics/metrics.json 未冻结: 先运行 scripts/generate_expected.py")
    with open(EXPECTED, encoding="utf-8") as fh:
        return json.load(fh)


def test_end_to_end_matches_frozen_metrics():
    exp = _load_expected()
    params = exp["params"]
    actual = experiment.run_repro_experiment(
        seed=params["seed"], top_n=params["top_n"], cost_bps=params["cost_bps"],
        forward_days=params["forward_days"], start=params["start"],
        end=params["end"], n_stocks=params["n_stocks"])

    def close(a, e):
        return abs(a - e) <= 0.01 * abs(e) + 1e-9

    assert actual["params"] == exp["params"]
    assert actual["ic"]["n"] == exp["ic"]["n"]
    for k in ("mean", "ir", "nw_t", "pos_ratio"):
        assert close(actual["ic"][k], exp["ic"][k]), f"ic.{k}: {actual['ic'][k]} vs {exp['ic'][k]}"
    for k in ("n_month", "nav", "cagr", "sharpe", "mdd", "win", "calmar",
              "excess_v_idx", "bench_nav", "bench_cagr"):
        assert close(actual["backtest"][k], exp["backtest"][k]), \
            f"backtest.{k}: {actual['backtest'][k]} vs {exp['backtest'][k]}"


def test_end_to_end_mathematical_properties():
    """不依赖冻结值的基本数学性质（任何环境都应成立）。"""
    res = experiment.run_repro_experiment(seed=20260806)
    ic = res["ic"]
    bt = res["backtest"]
    assert ic["n"] >= 1
    assert -1.0 <= ic["mean"] <= 1.0
    assert ic["pos_ratio"] >= 0.0
    assert bt["n_month"] >= 1
    assert bt["nav"] > 0
    assert 0.0 <= bt["mdd"] <= 1.0
    assert np.isfinite(bt["cagr"])


def test_different_seed_different_result():
    r1 = experiment.run_repro_experiment(seed=1)
    r2 = experiment.run_repro_experiment(seed=2)
    assert abs(r1["ic"]["mean"] - r2["ic"]["mean"]) > 1e-6

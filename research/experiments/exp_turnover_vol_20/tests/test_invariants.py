# -*- coding: utf-8 -*-
"""exp_turnover_vol_20 硬性 invariant 检查（审查报告 P0-4 验收标准）。"""


def run_invariants(metrics):
    errs = []
    if metrics.get("nav") is not None and not (metrics["nav"] > 0):
        errs.append(f"nav 未保持 > 0: {metrics['nav']}")
    if metrics.get("mdd") is not None:
        if not (0.0 <= metrics["mdd"] <= 1.0):
            errs.append(f"MaxDD 超出 [0,1]: {metrics['mdd']}")
    for k in ("ic_mean", "icir", "nw_t", "cagr", "sharpe", "excess"):
        v = metrics.get(k)
        if v is None or (isinstance(v, float) and v != v):
            errs.append(f"{k} 缺失或为 NaN")
    return errs

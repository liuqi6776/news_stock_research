# -*- coding: utf-8 -*-
"""exp_ma20_risk_control 硬性 invariant 检查。"""


def run_invariants(rows):
    errs = []
    for lb, m in rows.items():
        if not isinstance(m, dict):
            continue  # 顶层元数据（data_snapshot 等字符串）不参与 invariant
        if m.get("mdd") is None or not (0.0 <= m["mdd"] <= 1.0):
            errs.append(f"[{lb}] MaxDD 超出 [0,1]: {m.get('mdd')}")
        if m.get("cagr") is None or not (m["cagr"] > -1.0):
            errs.append(f"[{lb}] CAGR 非法: {m.get('cagr')}")
        for k in ("sharpe", "excess_v_etf", "calmar", "avg_weight"):
            v = m.get(k)
            if v is None or (isinstance(v, float) and v != v):
                errs.append(f"[{lb}] {k} 缺失或为 NaN")
    return errs

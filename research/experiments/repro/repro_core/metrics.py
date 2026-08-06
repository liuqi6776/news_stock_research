# -*- coding: utf-8 -*-
"""绩效指标纯函数（从 run_validation.py / risk_control_bt.py 提取, 口径完全一致）。

口径约定（上游同款）:
  - MaxDD 为相对回撤: ((cummax - nav) / cummax).max() ∈ [0,1]
  - CAGR: nav.iloc[-1] ** (1 / years) - 1
  - Sharpe: mean / std(ddof=1) * sqrt(periods_per_year)
  - 月胜率: (ret > 0).mean()
  - 超额(累计): (1+pr).prod() / (1+br).prod() - 1
  - 卡玛: cagr / mdd if mdd > 0 else nan
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def max_drawdown(nav: pd.Series) -> float:
    """相对最大回撤, ∈[0,1]。上游口径: ((nav.cummax()-nav)/nav.cummax()).max()"""
    nav = nav.astype(float)
    assert (nav > 0).all(), f"nav 存在非正数 (min={nav.min():.4f}), 无法计算回撤"
    mdd = float(((nav.cummax() - nav) / nav.cummax()).max())
    assert 0.0 <= mdd <= 1.0, f"MaxDD 超出 [0,1]: {mdd:.4f}"
    return mdd


def cagr(nav: pd.Series, periods_per_year: float = 12.0) -> float:
    """年化复合收益: nav ** (1/years) - 1。nav 需为累计净值序列(以 1 起)。"""
    nav = nav.astype(float)
    years = max(len(nav) / periods_per_year, 1e-12)
    final = nav.iloc[-1]
    if final <= 0:
        return np.nan
    return float(final ** (1.0 / years) - 1.0)


def sharpe(rets: pd.Series, periods_per_year: float = 12.0) -> float:
    """年化 Sharpe: mean / std(ddof=1) * sqrt(ppy); 无波动返回 nan"""
    rets = rets.astype(float).dropna()
    if len(rets) < 2:
        return np.nan
    sd = rets.std(ddof=1)
    if sd <= 0:
        return np.nan
    return float(rets.mean() / sd * np.sqrt(periods_per_year))


def win_rate(rets: pd.Series) -> float:
    rets = rets.astype(float).dropna()
    return float((rets > 0).mean()) if len(rets) else np.nan


def excess_return(port_rets: pd.Series, bench_rets: pd.Series) -> float:
    """累计超额: (1+pr).prod() / (1+br).prod() - 1"""
    pr = port_rets.astype(float)
    br = bench_rets.astype(float).reindex(pr.index).dropna()
    if len(br) == 0:
        return np.nan
    return float((1 + pr.reindex(br.index)).prod() / (1 + br).prod() - 1.0)


def calmar(cagr_val: float, mdd_val: float) -> float:
    return float(cagr_val / mdd_val) if mdd_val and mdd_val > 0 else np.nan


def summary_from_returns(port_rets: pd.Series, bench_rets: pd.Series | None = None,
                         periods_per_year: float = 12.0) -> dict:
    """从月频收益序列一次性汇总绩效指标（等价 risk_control_bt 汇总段）。

    port_rets: 策略收益序列（月频, 已扣成本）; bench_rets: 同频基准（可选）。
    返回: {n, nav, cagr, sharpe, mdd, win, calmar, excess(如有基准)}.
    """
    pr = port_rets.astype(float).dropna()
    nav = (1 + pr).cumprod()
    out = {
        "n": int(len(pr)),
        "nav": float(nav.iloc[-1]),
        "cagr": cagr(nav, periods_per_year),
        "sharpe": sharpe(pr, periods_per_year),
        "mdd": max_drawdown(nav),
        "win": win_rate(pr),
        "calmar": calmar(cagr(nav, periods_per_year), max_drawdown(nav)),
    }
    if bench_rets is not None:
        out["excess"] = excess_return(pr, bench_rets)
    return out

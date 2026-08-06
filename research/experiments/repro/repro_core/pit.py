# -*- coding: utf-8 -*-
"""PIT（Point-In-Time）对齐纯函数（从 run_validation.py 提取, 口径一致）。

上游 PIT 纪律:
  - 指数成分股: 取发布日期 <= 调仓日的最近一期权重（不提前知道未来成分变化）;
  - 未来收益: 因子 T 日可得, 收益取 T+1 ~ T+FORWARD_DAYS,
    实现为 cum.shift(-N)/cum - 1（不含当日, 用复权 pct_chg 累乘）;
  - 月度截面 Rank IC: winsorize(factor).rank().corr(forward_ret.rank())。
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def latest_index_weight(weight_sets: dict[str, set], date_str: str) -> set | None:
    """PIT: 取发布日期 <= date_str 的最近一期成分股集合; 无可用 -> None。"""
    avail = [d for d in sorted(weight_sets) if d <= date_str]
    return weight_sets.get(avail[-1]) if avail else None


def forward_returns(pct_chg: pd.Series, days: int = 20) -> pd.Series:
    """未来 days 日收益（不含当日, PIT 干净）: cum.shift(-days)/cum - 1。

    pct_chg: 日 pct 序列(单位 %); 缺失填充 0 后累乘（与上游一致）。"""
    pct = pct_chg.fillna(0.0)
    cum = (1 + pct / 100.0).cumprod()
    return cum.shift(-days) / cum - 1.0


def winsorize(s: pd.Series, lo: float = 0.01, hi: float = 0.99) -> pd.Series:
    return s.clip(s.quantile(lo), s.quantile(hi))


def rank_ic(factor: pd.Series, fwd_ret: pd.Series, min_n: int = 50,
            winsor: tuple[float, float] = (0.01, 0.99)) -> float:
    """月度截面 Rank IC: winsorize(factor).rank().corr(fwd_ret.rank())。样本不足 -> nan。"""
    f = factor.dropna()
    r = fwd_ret.reindex(f.index).dropna()
    if len(r) < min_n:
        return np.nan
    fw = f.clip(f.quantile(winsor[0]), f.quantile(winsor[1]))
    return float(fw.rank().corr(r.rank()))


def newey_west_t(ics, lag: int = 4) -> tuple[float, float]:
    """Newey-West t 统计（lag 阶 NW 校正, 上游口径）: 返回 (t, mean)。"""
    ics = np.asarray(ics, dtype=float)
    n = len(ics)
    if n < 2:
        return 0.0, 0.0
    mean = ics.mean()
    var = ics.var(ddof=1)
    for l in range(1, min(lag, n - 1) + 1):
        cov = np.cov(ics[:-l], ics[l:], ddof=1)[0, 1]
        var += 2 * (1 - l / (lag + 1)) * cov
    se = np.sqrt(max(var, 1e-12) / n)
    return mean / se, mean


def monthly_cross_section_ic(factor_map: dict, fwd_map: dict, rebal_dates: list[str],
                             weight_sets: dict[str, set], min_n: int = 50) -> pd.Series:
    """月度截面 Rank IC 序列（索引=调仓日）。

    factor_map: {code: 因子日频序列}; fwd_map: {code: 未来收益日频序列};
    weight_sets: {发布日期: 成分股集合}。与上游 run_fast 的 IC 循环等价。"""
    out = {}
    for rb in rebal_dates:
        members = latest_index_weight(weight_sets, rb)
        if members is None:
            continue
        fvals, rvals = {}, {}
        for code in members:
            fs, fr = factor_map.get(code), fwd_map.get(code)
            if fs is None or fr is None:
                continue
            if rb not in fs.index or rb not in fr.index:
                continue
            fv, rv = fs.loc[rb], fr.loc[rb]
            if pd.notna(fv) and pd.notna(rv):
                fvals[code], rvals[code] = fv, rv
        if len(fvals) < min_n:
            continue
        ic = rank_ic(pd.Series(fvals), pd.Series(rvals), min_n=min_n)
        if np.isfinite(ic):
            out[rb] = ic
    return pd.Series(out)


def picks_top_n(factor_map: dict, rebal_dates: list[str], weight_sets: dict[str, set],
                top_n: int = 60) -> dict:
    """每调仓日 Top-N 选股（PIT 成分内, 因子高值=好, 缺失剔除）。返回 {rb: [codes]}。"""
    out = {}
    for rb in rebal_dates:
        members = latest_index_weight(weight_sets, rb)
        if members is None:
            continue
        fvals = {}
        for code in members:
            fs = factor_map.get(code)
            if fs is None:
                continue
            if rb in fs.index and pd.notna(fs.loc[rb]):
                fvals[code] = fs.loc[rb]
        if len(fvals) < top_n:
            continue
        out[rb] = pd.Series(fvals).nlargest(top_n).index.tolist()
    return out

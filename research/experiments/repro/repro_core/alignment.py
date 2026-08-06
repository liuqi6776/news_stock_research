# -*- coding: utf-8 -*-
"""基准对齐 + walk-forward 持有期拼接（从 run_validation.py / risk_control_bt.py 提取）。

上游 walk-forward 拼接口径:
  - 调仓日 rebal_dates 为每月最后交易日（升序）;
  - 对相邻调仓日 (rb, rb_next): 持有期 hold = trade_dates[hi+1 : hn+1]
    （rb 当日收盘后调仓, 收益从 rb 次日开始, 含 rb_next 当日, 无前视）;
  - 基准月收益: (1 + 日收益.reindex(hold).fillna(0)).prod() - 1;
  - 组合月收益: 等权/权重加权日收益连乘, 再扣成本 net = gross - cost_bps/10000。
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def hold_slices(rebal_dates: list[str], trade_dates: list[str]):
    """walk-forward 持有期切分, 逐个产出 (rb, rb_next, hold)。

    hold = trade_dates[index(rb)+1 : index(rb_next)+1]
    """
    idx = {d: i for i, d in enumerate(trade_dates)}
    for i in range(len(rebal_dates) - 1):
        rb, rb_next = rebal_dates[i], rebal_dates[i + 1]
        if rb not in idx or rb_next not in idx:
            continue
        hi, hn = idx[rb], idx[rb_next]
        yield rb, rb_next, trade_dates[hi + 1:hn + 1]


def benchmark_monthly(hold: list[str], daily_pct: pd.Series) -> float:
    """持有期基准月收益: (1 + 日收益).prod() - 1（日收益单位为小数）。"""
    r = daily_pct.reindex(hold).fillna(0.0)
    return float((1 + r).prod() - 1.0)


def portfolio_monthly(hold: list[str], picks: list[str], pct_df: pd.DataFrame,
                      cost_bps: float = 20.0, weights: pd.Series | None = None) -> float:
    """持有期组合月收益（净）。

    pct_df: 日 pct_chg 宽表(单位 %, 行=日期字符串, 列=股票代码)
    等权: 日收益 = 成分日收益均值;  加权: weights 加权和(权重和为 1)
    net = gross - cost_bps/10000  （上游: 直接减, 非乘）
    """
    sub = pct_df.reindex(columns=picks).reindex(hold).fillna(0.0) / 100.0
    if weights is not None:
        w = weights.reindex(sub.columns).fillna(0.0)
        cr = (sub * w).sum(axis=1, min_count=1)
    else:
        cr = sub.mean(axis=1)
    gross = float((1 + cr).prod() - 1.0)
    return gross - cost_bps / 10000.0


def walk_forward_series(picks_map: dict, rebal_dates: list[str], trade_dates: list[str],
                        pct_df: pd.DataFrame, cost_bps: float = 20.0,
                        weight_map: dict | None = None) -> pd.Series:
    """完整 walk-forward 组合月收益序列（索引=rb_next, 与上游 port_rets 语义一致）。"""
    out = {}
    for rb, rb_next, hold in hold_slices(rebal_dates, trade_dates):
        picks = picks_map.get(rb)
        if not picks:
            continue
        w = weight_map.get(rb) if weight_map else None
        out[rb_next] = portfolio_monthly(hold, picks, pct_df, cost_bps, w)
    return pd.Series(out)


def benchmark_series(hold_map: dict, daily_pct: pd.Series) -> pd.Series:
    """基准月收益序列（hold_map: rb_next -> hold, 对齐组合收益索引）。"""
    return pd.Series({k: benchmark_monthly(h, daily_pct) for k, h in hold_map.items()})


def benchmark_hold_map(rebal_dates: list[str], trade_dates: list[str]) -> dict:
    """全部 rb_next -> hold 的映射（供基准对齐使用）。"""
    return {rb_next: hold for _rb, rb_next, hold in hold_slices(rebal_dates, trade_dates)}

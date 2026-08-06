# -*- coding: utf-8 -*-
"""HRP（层次风险平价）权重纯函数（从 risk_control_bt.py 的 _hrp_weights 提取, 口径一致）。

上游用法: 120 日收益窗口（T-1 已知, 不含调仓当日）, LedoitWolf 收缩协方差,
距离 = sqrt(0.5*(1-corr)), single linkage 分层, 递归二分按逆波动率分配。
降级路径与上游一致（样本不足 -> 等权; 协方差失败 -> 样本协方差 + 对角脊）。
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def inverse_vol(cov: pd.DataFrame, codes: list[str]) -> float:
    """子簇逆波动率之和: sum(1 / sqrt(diag(cov[codes,codes])))。"""
    sub = cov.loc[codes, codes]
    vols = np.sqrt(np.diag(sub))
    iv = 1.0 / (vols + 1e-9)
    return float(iv.sum())


def _recursive_bisect(codes: list, cov: pd.DataFrame) -> dict:
    """递归二分: 簇权重按子簇逆波动率占比分配, 叶子权重 = 1.0。"""
    if len(codes) == 1:
        return {codes[0]: 1.0}
    mid = len(codes) // 2
    left, right = codes[:mid], codes[mid:]
    wl = inverse_vol(cov, left)
    wr = inverse_vol(cov, right)
    total = wl + wr
    if total <= 0:
        return {c: 1.0 / len(codes) for c in codes}
    lw = wl / total
    out = {}
    for c, w in _recursive_bisect(left, cov).items():
        out[c] = lw * w
    for c, w in _recursive_bisect(right, cov).items():
        out[c] = (1 - lw) * w
    return out


def hrp_weights(returns: pd.DataFrame, min_obs: int = 20) -> pd.Series:
    """HRP 权重（和=1, 非负）。

    returns: DataFrame(日收益, 行=交易日, 列=个股)。与上游 _hrp_weights 一致:
      样本 < min_obs 或仅 1 列 -> 等权;
      LedoitWolf 失败 -> 样本协方差 + 1e-6 脊;
      linkage 失败 -> 上三角向量降级。
    """
    r = returns.dropna(how="all")
    if len(r) < min_obs or len(r.columns) < 2:
        return pd.Series(1.0 / len(r.columns), index=r.columns)
    try:
        from sklearn.covariance import LedoitWolf
        cov = LedoitWolf().fit(r).covariance_
    except Exception:
        cov = r.cov().values + np.eye(len(r.columns)) * 1e-6
    from scipy.cluster.hierarchy import leaves_list, linkage
    from scipy.spatial.distance import squareform

    corr = pd.DataFrame(r.corr().values, index=r.columns, columns=r.columns).fillna(0.0)
    dist = pd.DataFrame(np.sqrt(0.5 * (1.0 - corr)), index=corr.index, columns=corr.columns)
    cov_pd = pd.DataFrame(cov, index=r.columns, columns=r.columns)
    try:
        links = linkage(squareform(dist.values, checks=False), method="single")
    except Exception:
        links = linkage(dist.values[np.triu_indices_from(dist.values, k=1)], method="single")
    sorted_codes = list(r.columns[leaves_list(links)])

    w = pd.Series(_recursive_bisect(sorted_codes, cov_pd)).reindex(r.columns).fillna(0.0)
    s = w.sum()
    if s > 0:
        w = w / s
    return w


def hrp_weight_map(picks_map: dict, rebal_dates: list[str], trade_dates: list[str],
                   pct_df: pd.DataFrame, window: int = 120) -> dict:
    """批量 HRP 权重: 每个调仓日取调仓前 window 个交易日收益（T-1 已知, 不含调仓当日）。

    与上游一致: win = trade_dates[max(0, hi-window):hi]
    返回 {rb: 权重 Series}。
    """
    idx = {d: i for i, d in enumerate(trade_dates)}
    out = {}
    for rb in picks_map:
        if rb not in idx:
            continue
        hi = idx[rb]
        win = trade_dates[max(0, hi - window):hi]
        rets = pct_df.reindex(columns=picks_map[rb]).reindex(win)
        out[rb] = hrp_weights(rets)
    return out

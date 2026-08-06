# -*- coding: utf-8 -*-
"""仓位信号纯函数（从 risk_control_bt.py 提取, 口径一致）。

关键纪律:
  - MA20/Vol/动量等仓位信号一律取 T-1 日收盘已知信息、T 日生效（2026-08-03 同日前视修复）,
    即调用方应传 idx_close.shift(1) / ma20.shift(1) / vol20.shift(1) 序列（见 t1_shift）;
  - RS12 弱时段持 512100 ETF 不动（由上层循环决定是否应用风控）;
  - 降仓部分按现金(0 收益)缓冲: r_t = w * 组合日收益;
  - DD 变体: T 日仓位由 T-1 末 shadow 回撤决定, shadow NAV 在当日收益应用后更新。
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def t1_shift(x: pd.Series) -> pd.Series:
    """T-1 位移: 返回 shift(1) 后的序列（信号 T 日生效, 只用 T-1 日及以前信息）。"""
    return x.shift(1)


def ma20_3_tier(c: float, m: float, deep: float = 0.98) -> float:
    """MA20 三档: c>=m -> 1.0; deep*m<=c<m -> 0.5; 否则 0.0。输入缺失 -> nan。"""
    if not (np.isfinite(c) and np.isfinite(m)):
        return np.nan
    if c >= m:
        return 1.0
    if c >= deep * m:
        return 0.5
    return 0.0


def ma20_n_tier(c: float, m: float, boundaries: list[float], weights: list[float]) -> float:
    """MA20 N 档: 按 r=c/m 自高到低取首个 r>=bnd 的权重, 否则 0.0（十档/廿档同构）。"""
    if not (np.isfinite(c) and np.isfinite(m)):
        return np.nan
    r = c / m
    for wgt, bnd in zip(weights, boundaries):
        if r >= bnd:
            return float(wgt)
    return 0.0


def vol_target(v: float, tgt: float = 0.20, floor_w: float = 0.20) -> float:
    """VolTarget: w = clip(tgt / σ20, floor_w, 1.0); σ 缺失/非正 -> nan。"""
    if not np.isfinite(v) or v <= 0:
        return np.nan
    return float(np.clip(tgt / v, floor_w, 1.0))


def vol_penalty(v: float, v_hi: float = 0.30, v_lo: float = 0.50, v_min: float = 0.5) -> float:
    """波动惩罚系数: v<=v_hi -> 1.0; v>v_hi -> clip((v_lo-v)/(v_lo-v_hi), v_min, 1.0)。"""
    if not np.isfinite(v):
        return 1.0
    if v > v_hi:
        return float(np.clip((v_lo - v) / (v_lo - v_hi), v_min, 1.0))
    return 1.0


def dd_weight(dd_prev: float, half: float, zero: float, fix: float, w_half: bool) -> tuple[float, bool]:
    """DD 触发: 返回 (w, w_half_new)。

    高水位回撤 dd_prev = shadow/shadow_hwm - 1:
      - 回撤修复 >= fix  -> 解除半仓
      - 回撤 <= half      -> 半仓
      - 回撤 <= zero      -> 空仓(0)
      - 否则               -> 1.0 / 0.5(半仓状态)
    """
    if w_half and dd_prev >= fix:
        w_half = False
    if dd_prev <= half:
        w_half = True
    w = 0.0 if dd_prev <= zero else (0.5 if w_half else 1.0)
    return w, w_half


def cppi_weight(nav: float, floor: float, hwm: float, m: float = 3.0, alpha: float = 0.90) -> tuple[float, float]:
    """CPPI/TIPP: floor = max(floor, alpha*hwm); w = clip(m*(nav-floor)/nav, 0, 1)。
    返回 (w, new_floor)。"""
    floor = max(floor, alpha * hwm)
    w = float(np.clip(m * (nav - floor) / nav, 0.0, 1.0)) if nav > 0 else 0.0
    return w, floor


def ma20_trend(c: float, m: float, lo_ratio: float = 0.96, hi_ratio: float = 1.00) -> float:
    """MA20 连续趋势分档（multi 变体）: trend = clip((c/m - 0.96)/0.04, 0, 1)。"""
    if not (np.isfinite(c) and np.isfinite(m)):
        return 1.0
    return float(np.clip((c / m - lo_ratio) / (hi_ratio - lo_ratio), 0.0, 1.0))


def position_weight(rtype: str, par: dict, c: float, m: float, v: float,
                    mo: float, dd_prev: float, w_half: bool, nav: float,
                    floor: float, hwm: float) -> tuple[float, bool, float]:
    """统一仓位信号入口（等价 risk_control_bt 内联逻辑）。

    参数均应为 T-1 日已知信息（c=idx_close_1, m=ma20_1, v=vol20_1, mo=mom20_1,
    dd_prev=shadow/shadow_hwm-1 于 T-1 末）。返回 (w, w_half_new, floor_new)。
    rtype: ma20 / ma20_5 / ma20_10 / ma20_20 / ma20_5_vol / vol / dd / cppi / multi
    """
    w_half_new, floor_new = w_half, floor
    if rtype == "ma20":
        w = ma20_3_tier(c, m, par.get("deep", 0.98))
    elif rtype in ("ma20_5", "ma20_10", "ma20_20"):
        w = ma20_n_tier(c, m, par["boundaries"], par["weights"])
    elif rtype == "ma20_5_vol":
        w5 = ma20_n_tier(c, m, par["boundaries"], par["weights"])
        w = w5 * vol_penalty(v, par.get("v_hi", 0.30), par.get("v_lo", 0.50), par.get("v_min", 0.5))
    elif rtype == "vol":
        w = vol_target(v, par.get("tgt", 0.20), par.get("floor_w", 0.20))
    elif rtype == "dd":
        w, w_half_new = dd_weight(dd_prev, par["half"], par["zero"], par["fix"], w_half)
    elif rtype == "cppi":
        w, floor_new = cppi_weight(nav, floor, hwm, par.get("m", 3.0), par.get("alpha", 0.90))
    elif rtype == "multi":
        trend = ma20_trend(c, m)
        vp = vol_penalty(v, par.get("v_hi", 0.30), par.get("v_lo", 0.50), par.get("v_min", 0.2))
        mom_pen = par.get("mom_min", 0.6) if np.isfinite(mo) and mo < 0 else 1.0
        w = trend * vp * mom_pen
    else:
        raise ValueError(f"未知风控类型: {rtype}")
    if not np.isfinite(w):
        w = 1.0  # 信号缺失时按满仓处理（上游: 无有限值则维持 1.0）
    return float(w), w_half_new, floor_new

# -*- coding: utf-8 -*-
"""固定种子合成数据生成器 —— 外部无需私有数据即可独立复现核心逻辑。

数据结构与上游一致:
  - 日期: '%Y%m%d' 字符串
  - pct: 单位 %（日 pct_chg）; close: 复权累计价
  - 指数成分股: {发布日期: set(代码)}（PIT 对齐用）
  - 指数日线: close + pct_chg

所有生成器均以 rng = np.random.default_rng(seed) 起步, 固定种子 -> 完全可复现。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

DEFAULT_START = "20200102"
DEFAULT_END = "20241231"


def make_trade_dates(start: str = DEFAULT_START, end: str = DEFAULT_END) -> list[str]:
    """模拟 A 股交易日序列（周一~周五, 不做节假日剔除; 上游为真实交易日, 仅结构等价）。"""
    idx = pd.bdate_range(start, end)
    return [d.strftime("%Y%m%d") for d in idx]


def make_codes(n: int) -> list[str]:
    return [f"{600000 + i:06d}" for i in range(n)]


def make_price_panel(trade_dates: list[str], n_stocks: int = 120, seed: int = 7,
                     mu: float = 0.0004, sigma: float = 0.020,
                     alpha: pd.Series | None = None, alpha_scale: float = 0.002) -> tuple[pd.DataFrame, pd.DataFrame]:
    """几何布朗运动日收益 + 价格, 含 0.3 倍共性因子（相关矩阵非单位阵, HRP 有意义）。

    alpha: 若提供（每只股票的 latent alpha 序列）, 则个股日漂移率 = mu + alpha_scale*alpha[c],
           使"高 alpha = 高预期收益"成立（端到端链路可捕捉到信号）。
    返回 (pct_df 单位 %, close_df)。列=股票代码, 行=日期字符串。"""
    rng = np.random.default_rng(seed)
    n = len(trade_dates)
    codes = make_codes(n_stocks)
    rets = rng.normal(mu, sigma, size=(n, n_stocks))
    common = rng.normal(0.0, 0.008, size=(n, 1))
    rets = rets + 0.3 * common
    if alpha is not None:
        rets = rets + alpha_scale * np.asarray(alpha.reindex(codes).fillna(0.0), dtype=float)[None, :]
    pct = pd.DataFrame(rets * 100.0, index=trade_dates, columns=codes)
    close = pd.DataFrame((1 + rets).cumprod(axis=0), index=trade_dates, columns=codes)
    return pct, close


def make_index_weights(trade_dates: list[str], n_stocks: int = 120, k: int = 80,
                       seed: int = 11) -> tuple[list[str], dict[str, set]]:
    """每月最后交易日 -> k 只成分股（随机子集, 模拟中证1000成分变化的 PIT 结构）。

    返回 (rebal_dates 升序, {rb: set(代码)})。"""
    months = {}
    for d in trade_dates:
        months[d[:6]] = d
    rebal = sorted(months.values())[:-1]
    rng = np.random.default_rng(seed)
    codes = make_codes(n_stocks)
    out = {}
    for rb in rebal:
        out[rb] = set(rng.choice(codes, size=k, replace=False).tolist())
    return rebal, out


def make_index_daily(trade_dates: list[str], seed: int = 3) -> pd.DataFrame:
    """指数日线（模拟 000852.SH）: close + pct_chg(单位 %)。"""
    rng = np.random.default_rng(seed)
    rets = rng.normal(0.0002, 0.012, size=len(trade_dates))
    close = (1 + rets).cumprod()
    return pd.DataFrame({"close": close, "pct_chg": rets * 100.0}, index=trade_dates)


def make_latent_alpha(n_stocks: int, seed: int = 5) -> pd.Series:
    """每只股票固定 latent alpha（因子"高值=好"的生成来源）。"""
    rng = np.random.default_rng(seed)
    return pd.Series(rng.normal(0.0, 1.0, n_stocks), index=make_codes(n_stocks))


def make_factor_panel(pct_df: pd.DataFrame, alpha: pd.Series, seed: int = 17) -> pd.DataFrame:
    """日频因子面板: alpha + 白噪声, 形状同 pct_df（列=股票, 行=日期）。

    噪声与收益独立, 使 Rank IC 为正但非 1（端到端测试有意义）。"""
    rng = np.random.default_rng(seed)
    noise = rng.normal(0.0, 0.5, size=pct_df.shape)
    return pd.DataFrame(alpha.values[None, :] + noise, index=pct_df.index, columns=pct_df.columns)


def make_etf_daily(trade_dates: list[str], seed: int = 21) -> pd.Series:
    """512100 ETF 日收益（小数, 单位 1; 上游 etf_ret）。"""
    rng = np.random.default_rng(seed)
    return pd.Series(rng.normal(0.0001, 0.010, size=len(trade_dates)), index=trade_dates)

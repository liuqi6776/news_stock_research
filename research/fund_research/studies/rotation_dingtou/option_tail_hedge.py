# -*- coding: utf-8 -*-
"""
P3: 保护性认沽期权尾部对冲 (VolTarget7% 组合之上)
==========================================================
核心问题: DD触发/CPPI/止损/IM空头对冲已全部证伪, 认沽期权是唯一未试过的防线。
认沽期权 ≠ 期货空头: 只付权利金(1-1.5%年化), 不承担9.3%基差成本, 尾部事件才赔付。

双轨验证:
  Track 2 (主, 概念验证): 组合自身尾部保险
    - 每日扣保费 (年化1.0%/1.5%按日摊销)
    - 月末: 组合月收益 < -thr 时, 赔付 (月跌幅-thr)×月初市值
    - 回答: "买尾部保险值不值"

  Track 1 (辅, 真实落地): A股认沽对冲 (中证500指数, Black-Scholes定价)
    - 覆盖组合A股暴露 (量化10%+红利10% = 20%)
    - 月度滚动: 每月初买10%虚值认沽, BS定价 (IV=指数60日已实现波动率)
    - 月末: 指数月跌超10%时赔付 (K-ST)/S0×20%
    - 回答: "1-1.5%预算能买到什么A股保护, 对组合实际帮助多大"

基线: 9资产组合 + VolTarget7% (floor=0.5) [P1冻结参数]
费用: 基金申购0.15% / 赎回阶梯 / FIFO; 期权保费额外计入
区间: 全样本2018-2026 + OOS 2023-2026 (与P1一致)

用法:
  C:/Users/liuqi/anaconda3/python.exe studies/rotation_dingtou/option_tail_hedge.py
"""
import os, sys
import numpy as np
import pandas as pd
from scipy.stats import norm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vol_target as vt  # 复用9资产定义/净值/费用/VolTarget回测

IDX_DIR = os.path.join(os.path.dirname(os.path.dirname(vt.ROOT)), "research", "chip_momentum", "data", "index_daily")
SQRT_252 = np.sqrt(252.0)

# ---- Track 2: 组合自身尾部保险 ----
def apply_portfolio_insurance(eq, cost_ann, thr):
    """组合尾部保险 (月度滚动认沽):
    - 每日扣保费: cost_ann/252 × 当日市值
    - 月末: 当月组合收益(扣保费后) < -thr 时, 赔付 = (月跌幅-thr)×月初市值
    返回 (保险后净值, 统计)
    """
    idx = eq.index
    vals = eq.values.astype(float)
    n = len(vals)
    base = np.ones(n)
    base[1:] = vals[1:] / vals[:-1]   # 原组合日收益倍数
    out = np.empty(n)
    out[0] = vals[0]
    m_start = vals[0]      # 月初市值 (保险后口径)
    m_ret = 1.0            # 当月累计收益倍数 (扣保费后)
    n_payout = 0
    total_prem = 0.0
    total_payout = 0.0
    for i in range(1, n):
        r = base[i] - 1.0
        prem = out[i-1] * (cost_ann / 252.0)
        total_prem += prem
        out[i] = out[i-1] * (1 + r) - prem
        m_ret *= (1 + r) * (1 - cost_ann / 252.0)
        if idx[i].month != idx[i-1].month:
            m_ret_net = m_ret - 1.0
            if m_ret_net < -thr:
                payout = (-m_ret_net - thr) * m_start
                out[i] += payout
                total_payout += payout
                n_payout += 1
            m_start = out[i]
            m_ret = 1.0
    return pd.Series(out, index=idx), {"保费": total_prem, "赔付": total_payout, "赔付次数": n_payout}


# ---- Track 1: A股认沽 (BS定价) ----
def bs_put(S, K, T, r, sigma):
    """Black-Scholes 欧式认沽价格"""
    if sigma <= 0 or S <= 0:
        return 0.0
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def load_idx_close(code):
    df = pd.read_parquet(os.path.join(IDX_DIR, f"{code}.parquet"))
    s = pd.Series(df["close"].values.astype(float),
                  index=pd.to_datetime(df["trade_date"].astype(str)))
    return s.sort_index()


def apply_index_put(eq, idx_close, coverage, otm, r=0.02, iv_premium=1.0):
    """中证500认沽对冲:
    - 月度滚动: 每月初买 (1-otm)×现价 认沽, BS定价 (IV=指数60日已实现波动率×iv_premium)
    - 月末: 指数跌超 otm 时赔付 (K-ST)/S0 × coverage × 市值
    coverage: 覆盖组合A股暴露比例 (0.20)
    iv_premium: IV溢价系数 (真实期权IV通常高于已实现波动率, 敏感性用1.0/1.3)
    返回 (保险后净值, 统计)
    """
    idx = eq.index
    vals = eq.values.astype(float)
    n = len(vals)
    base = np.ones(n)
    base[1:] = vals[1:] / vals[:-1]
    out = np.empty(n)
    out[0] = vals[0]

    S = idx_close.reindex(idx).ffill()
    ret = idx_close.pct_change()
    sigma_ann = (ret.rolling(60).std() * SQRT_252 * iv_premium).reindex(idx).ffill()   # 60日已实现年化波动率×IV溢价, 对齐eq索引

    total_prem = 0.0
    total_payout = 0.0
    n_payout = 0
    prem_pct_list = []
    cur_K = None
    cur_S0 = None

    for i in range(1, n):
        r_eq = base[i] - 1.0
        out[i] = out[i-1] * (1 + r_eq)
        is_first = idx[i].month != idx[i-1].month          # 每月第一天: 买入 (T-1收盘定价, 无前视)
        is_last = (i + 1 >= n) or (idx[i+1].month != idx[i].month)  # 每月最后一天: 到期结算
        if is_first:
            S0 = float(S.iloc[i-1]) if np.isfinite(S.iloc[i-1]) else np.nan
            sig = float(sigma_ann.iloc[i-1]) if np.isfinite(sigma_ann.iloc[i-1]) else np.nan
            if np.isfinite(S0) and S0 > 0:
                sig = sig if np.isfinite(sig) and sig > 0 else 0.22
                K = S0 * (1 - otm)
                put_px = bs_put(S0, K, 1/12, r, sig) / S0   # 占名义比例
                prem = out[i-1] * coverage * put_px         # 基于上月末市值
                out[i] -= prem                              # 月初当天扣保费 (真正影响净值)
                total_prem += prem
                prem_pct_list.append(put_px)
                cur_K, cur_S0 = K, S0
        if is_last:
            ST = float(S.iloc[i]) if np.isfinite(S.iloc[i]) else np.nan
            if cur_K is not None and cur_S0 and np.isfinite(ST) and ST < cur_K:
                payout = (cur_K - ST) / cur_S0 * coverage * out[i]
                out[i] += payout
                total_payout += payout
                n_payout += 1
            cur_K, cur_S0 = None, None

    prem_ann = np.mean(prem_pct_list) * 12 * coverage if prem_pct_list else 0
    return pd.Series(out, index=idx), {"保费": total_prem, "赔付": total_payout,
                                       "赔付次数": n_payout, "月度权利金均值": np.mean(prem_pct_list) if prem_pct_list else 0,
                                       "年化保费率": prem_ann}


# ---- 指标 ----
def metrics(eq, lump=1_000_000):
    years = (eq.index[-1] - eq.index[0]).days / 365.0
    ann = (eq.iloc[-1] / lump) ** (1 / years) - 1
    mdd = float((eq / eq.cummax() - 1).min())
    r = eq.pct_change().fillna(0)
    vol = r.std() * SQRT_252
    sh = (r.mean() * 252) / vol if vol > 0 else 0
    return {"期末": eq.iloc[-1], "年化": ann, "回撤": mdd, "波动": vol, "夏普": sh}


def print_table(rows, title):
    print(f"\n{'='*115}")
    print(title)
    print(f"{'='*115}")
    hdr = f"{'方案':38s} | {'期末':>10s} {'年化':>7s} {'回撤':>8s} {'波动':>7s} {'夏普':>6s} | {'保费':>8s} {'赔付':>8s} {'赔付次数':>6s}"
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(r)
    print()


def main():
    t0 = pd.Timestamp.now()
    print("=" * 115)
    print("P3: 保护性认沽期权尾部对冲 (VolTarget7% 组合之上, 双轨验证)")
    print("=" * 115)

    navs = vt.load_navs()
    weights = {n: w for n, (_, w) in vt.ASSETS.items()}

    # 基线: VolTarget7% (floor=0.5) [P1冻结参数]
    BASE = (0.07, 0.5)
    idx500 = load_idx_close("000905.SH")

    for label, start, end in [
        ("全样本 2018-2026", vt.START, vt.END),
        ("OOS 2023-2026", vt.OOS_START, vt.END),
    ]:
        print(f"\n{'#'*115}")
        print(f"# {label}")
        print(f"{'#'*115}")

        eq, _ = vt.run_backtest(navs, weights, tgt_vol=BASE[0], floor_w=BASE[1],
                                lump=1_000_000, dca=0, start=start, end=end)
        rows = []
        m0 = metrics(eq)
        rows.append(f"{'基线 VolTarget7% (无对冲)':38s} | {m0['期末']:>9,.0f} {m0['年化']:>6.1%} {m0['回撤']:>7.1%} {m0['波动']:>6.1%} {m0['夏普']:>5.2f} | {'-':>8s} {'-':>8s} {'-':>6s}")

        # Track 2: 组合保险
        for cost, thr in [(0.01, 0.04), (0.015, 0.04), (0.015, 0.05)]:
            eq_i, st = apply_portfolio_insurance(eq, cost, thr)
            m = metrics(eq_i)
            rows.append(
                f"{f'组合保险 成本{cost:.1%} 阈值-{thr:.0%}':38s} | {m['期末']:>9,.0f} {m['年化']:>6.1%} {m['回撤']:>7.1%} {m['波动']:>6.1%} {m['夏普']:>5.2f} | "
                f"{st['保费']/1e6:>7.1%} {st['赔付']/1e6:>7.1%} {st['赔付次数']:>6d}")

        # Track 1: A股认沽 (BS定价, 敏感性: IV溢价 + 虚值档位)
        for otm, cover, prem in [(0.10, 0.20, 1.0), (0.10, 0.20, 1.3), (0.10, 0.55, 1.0), (0.05, 0.20, 1.3)]:
            tag = f"A股认沽 500 {otm:.0%}虚值 覆盖{cover:.0%}"
            tag += "" if prem == 1.0 else f" IV×{prem:.1f}"
            eq_i, st = apply_index_put(eq, idx500, coverage=cover, otm=otm, iv_premium=prem)
            m = metrics(eq_i)
            rows.append(
                f"{tag:38s} | {m['期末']:>9,.0f} {m['年化']:>6.1%} {m['回撤']:>7.1%} {m['波动']:>6.1%} {m['夏普']:>5.2f} | "
                f"{st['保费']/1e6:>7.1%} {st['赔付']/1e6:>7.1%} {st['赔付次数']:>6d}")

        print_table(rows, f"{label}: VolTarget7% + 尾部对冲对比 (一次性100万)")

        # Track 1 保险成本明细
        if label.startswith("全样本"):
            prem_pct = st["月度权利金均值"]
            print(f"  [A股认沽明细] 月度权利金均值={prem_pct:.3%} (占覆盖名义), 年化保费≈{st['年化保费率']:.2%} (覆盖20%后), 赔付{st['赔付次数']}次")

    print(f"\n总耗时 {(pd.Timestamp.now()-t0).total_seconds():.0f}s")


if __name__ == "__main__":
    main()

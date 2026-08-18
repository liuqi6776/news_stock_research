# -*- coding: utf-8 -*-
"""
定投场景: 2020-01 ~ 2023-12 每月投1万, 2024-01~2026-08 只持有不投
权重方案: 等权 / 进攻20债80股 / 保守60债40股
检查: 2023年底市值 & 2026年8月市值
"""
import os, sys, time
import numpy as np
import pandas as pd
from scipy.optimize import brentq

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
TRAIN_START = "2018-01-01"
NAV_DIR = r"D:\iquant_data\data_v2\fund2\nav"

ASSETS = {
    "纯债":    "000015",
    "黄金":    "000216",
    "纳指":    "000834",
    "沪深300": "050002",
    "QDII债":  "004998",
    "原油":    "501018",
}

_AC = {}
def acc_nav(code):
    if code not in _AC:
        p = os.path.join(NAV_DIR, f"{code}.parquet")
        df = pd.read_parquet(p, columns=["date", "acc_nav"])
        s = pd.Series(df["acc_nav"].to_numpy(float), index=pd.to_datetime(df["date"]))
        s = s[~s.index.duplicated(keep="last")].sort_index().dropna()
        s = s[(s.index >= "2019-06-01") & (s.index <= "2026-08-06")]
        _AC[code] = s
    return _AC[code]


def fee_rate(days):
    if days < 7:   return 0.015
    if days < 365: return 0.005
    if days < 730: return 0.0025
    return 0.0


def load():
    navs, rets = {}, {}
    for c, code in ASSETS.items():
        s = acc_nav(code)
        navs[c] = s
        r = s.pct_change().dropna().clip(-0.2, 0.2)
        rets[c] = r
    return navs, rets


def plan_weights(rets):
    cats = list(ASSETS.keys())
    df = pd.DataFrame({c: rets[c] for c in cats})
    df = df[(df.index >= TRAIN_START) & (df.index <= "2020-12-31")].fillna(0)
    cov = df.cov().values * 252
    n = len(cats)
    plans = {}
    plans["等权"] = {c: 1/n for c in cats}
    vol = np.sqrt(np.diag(cov))
    w = 1.0 / vol; w /= w.sum()
    plans["逆波动"] = {cats[i]: w[i] for i in range(n)}
    bonds = ["纯债", "QDII债"]
    risky = [c for c in cats if c not in bonds]
    plans["保守60债40股"] = {c: 0.6/len(bonds) if c in bonds else 0.4/len(risky) for c in cats}
    plans["进攻20债80股"] = {c: 0.2/len(bonds) if c in bonds else 0.8/len(risky) for c in cats}
    return plans


def simulate(navs, weights, invest_start, invest_end, hold_end, monthly=10_000):
    """
    invest_start ~ invest_end: 每月投 monthly 元, 按权重买
    invest_end ~ hold_end: 停止投资, 只持有
    返回: 每日市值, 投入明细, 关键时点市值
    """
    cats = list(weights.keys())
    all_nav = pd.DataFrame({c: navs[c] for c in cats})
    all_nav = all_nav[(all_nav.index >= invest_start) & (all_nav.index <= hold_end)].ffill().bfill()

    # 定投日: 每月第一个交易日
    inv_dates_all = pd.date_range(invest_start, invest_end, freq="MS")  # 月初
    td = all_nav.index.to_numpy()
    invest_days = []
    for d in inv_dates_all:
        d64 = pd.Timestamp(d).to_datetime64()
        k = int(np.searchsorted(td, d64))
        k = min(k, len(td) - 1)
        invest_days.append(pd.Timestamp(td[k]))

    # 每资产累计份额
    shares = {c: 0.0 for c in cats}
    cum_shares = {c: pd.Series(0.0, index=all_nav.index) for c in cats}
    total_invested = 0.0
    cashflows = []

    for d in invest_days:
        total_invested += monthly
        cashflows.append((d, -monthly))
        for c in cats:
            nv = float(all_nav.loc[d, c])
            new_sh = monthly * weights[c] / nv
            shares[c] += new_sh
        # 更新累计份额曲线
        for c in cats:
            m = cum_shares[c].index >= d
            cum_shares[c].loc[m] = shares[c]

    # 每日市值
    mv = pd.Series(0.0, index=all_nav.index)
    for c in cats:
        mv = mv + cum_shares[c] * all_nav[c]

    # 关键时点
    def val_at(date_str):
        d = pd.Timestamp(date_str)
        if d > mv.index[-1]:
            d = mv.index[-1]
        return float(mv.asof(d))

    # 期末赎回费估算 (简化: 平均持有期)
    avg_days = 365 * 2  # 大部分持有 > 1年
    fee_rate_avg = fee_rate(int(avg_days))
    # 只有持有 < 2年的部分有0.25%费, > 2年免费
    # 简化: 用 0.1% 作为平均费率
    avg_fee = 0.001

    v_2023 = val_at("2023-12-31")
    v_2026 = val_at("2026-08-06")
    v_2023_net = v_2023 * (1 - avg_fee)
    v_2026_net = v_2026 * (1 - avg_fee)

    # XIRR 到2023年底
    cf_2023 = [(d, v) for d, v in cashflows] + [(pd.Timestamp("2023-12-31"), v_2023_net)]
    # XIRR 到2026年8月
    cf_2026 = [(d, v) for d, v in cashflows] + [(pd.Timestamp("2026-08-06"), v_2026_net)]

    def xirr(cfs):
        if len(cfs) < 2: return np.nan
        try:
            d0 = cfs[0][0]
            days = np.array([(d - d0).days for d, _ in cfs], dtype=float)
            flows = np.array([v for _, v in cfs], dtype=float)
            return brentq(lambda r: float(np.sum(flows / (1+r)**(days/365))), -0.5, 5.0)
        except:
            return np.nan

    return {
        "每日市值": mv,
        "总投入": total_invested,
        "投入笔数": len(invest_days),
        "2023年底市值": v_2023,
        "2023年底净": v_2023_net,
        "2026年8月市值": v_2026,
        "2026年8月净": v_2026_net,
        "XIRR_2023": xirr(cf_2023),
        "XIRR_2026": xirr(cf_2026),
        "现金流": cashflows,
    }


def yearly_snapshot(mv, cashflows, weights):
    """按年末算: 累计投入 / 市值 / 收益率"""
    cum_inv = pd.Series(0.0, index=mv.index)
    for d, v in cashflows:
        if v < 0:
            cum_inv.loc[cum_inv.index >= d] += -v
    rows = []
    for y in range(2020, 2027):
        ey = pd.Timestamp(f"{y}-12-31")
        if ey > mv.index[-1]:
            ey = mv.index[-1]
        val = float(mv.asof(ey))
        inv = float(cum_inv.asof(ey))
        ret = val / inv - 1 if inv > 0 else 0
        rows.append((y, inv, val, ret))
    return rows


def main():
    t0 = time.time()
    navs, rets = load()
    plans = plan_weights(rets)

    INVEST_START = "2020-01-01"
    INVEST_END   = "2023-12-31"
    HOLD_END     = "2026-08-06"
    MONTHLY      = 10_000

    total_months = 48  # 2020-01 ~ 2023-12
    total_invest = total_months * MONTHLY

    print("=" * 100)
    print(f"定投回测: {INVEST_START} ~ {INVEST_END} 每月投 {MONTHLY:,}元, 然后持有至 {HOLD_END}")
    print(f"总投入: {total_months} 个月 × {MONTHLY:,} = {total_invest:,} 元")
    print(f"训练期(定权重): 2018-01 ~ 2020-12 (只用2020前数据定权重的话也行)")
    print("=" * 100)

    # 打印权重
    print("\n权重方案:")
    for pn, ws in plans.items():
        print(f"  {pn:12s}: " + "  ".join(f"{c}={w:.0%}" for c, w in ws.items()))

    print("\n" + "=" * 100)
    print("结果汇总")
    print("=" * 100)
    hdr = (f"{'方案':12s} | {'总投入':>8s} | {'2023年底市值':>12s} {'2023收益':>8s} {'2023 XIRR':>9s} | "
           f"{'2026年8月市值':>13s} {'2026收益':>8s} {'2026 XIRR':>9s} | {'期间回撤':>8s}")
    print(hdr)
    print("-" * len(hdr))

    all_sims = {}
    for pn, ws in plans.items():
        sim = simulate(navs, ws, INVEST_START, INVEST_END, HOLD_END, MONTHLY)
        all_sims[pn] = sim
        mv = sim["每日市值"]
        # 回撤 (整个区间)
        mdd = float((mv / mv.cummax() - 1).min())
        r2023 = sim["2023年底净"] / sim["总投入"] - 1
        r2026 = sim["2026年8月净"] / sim["总投入"] - 1
        print(f"{pn:12s} | {sim['总投入']:>8,.0f} | "
              f"{sim['2023年底净']:>11,.0f}元 {r2023:>7.1%} {sim['XIRR_2023']:>8.2%} | "
              f"{sim['2026年8月净']:>12,.0f}元 {r2026:>7.1%} {sim['XIRR_2026']:>8.2%} | "
              f"{mdd:>7.1%}")

    # 逐年明细 (等权 & 进攻)
    print("\n" + "=" * 100)
    print("逐年明细 (累计投入 vs 市值)")
    print("=" * 100)
    for pn in ["等权", "进攻20债80股", "保守60债40股"]:
        sim = all_sims[pn]
        mv = sim["每日市值"]
        rows = yearly_snapshot(mv, sim["现金流"], plans[pn])
        print(f"\n  [{pn}]")
        print(f"  {'年份':>6s}  {'累计投入':>10s}  {'市值':>10s}  {'收益率':>8s}  {'当年新增投入':>10s}")
        prev_inv = 0
        for y, inv, val, ret in rows:
            new_inv = inv - prev_inv
            marker = " ◄ 定投期" if y <= 2023 else " ◄ 持有期"
            print(f"  {y:>6d}  {inv:>9,.0f}元  {val:>9,.0f}元  {ret:>7.1%}  {new_inv:>9,.0f}元{marker}")
            prev_inv = inv

    # 持有期收益 (2023底→2026年8月)
    print("\n" + "=" * 100)
    print("持有期收益 (2023年底 → 2026年8月, 不再投入)")
    print("=" * 100)
    for pn in ["等权", "进攻20债80股", "保守60债40股"]:
        sim = all_sims[pn]
        v23 = sim["2023年底市值"]
        v26 = sim["2026年8月市值"]
        hold_return = v26 / v23 - 1
        hold_days = (pd.Timestamp("2026-08-06") - pd.Timestamp("2023-12-31")).days
        hold_ann = (v26 / v23) ** (365.0 / hold_days) - 1
        print(f"  {pn:12s}: {v23:>10,.0f} → {v26:>10,.0f}  收益 {hold_return:>6.1%}  年化 {hold_ann:>6.1%}")

    print(f"\n总耗时 {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()

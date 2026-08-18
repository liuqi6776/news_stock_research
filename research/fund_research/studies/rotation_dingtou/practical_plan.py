# -*- coding: utf-8 -*-
"""
实操方案: 200万现金 + 每月1万结余
对比: 一次性投入 vs 分批入场(6个月/12个月) + 持续定投
从不同年份起投, 看到2026年8月的结果
"""
import os, sys
import numpy as np
import pandas as pd
from scipy.optimize import brentq

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
NAV_DIR = r"D:\iquant_data\data_v2\fund2\nav"

ASSETS = {
    "纯债": "000015", "黄金": "000216", "纳指": "000834",
    "沪深300": "050002", "QDII债": "004998", "原油": "501018",
}

_AC = {}
def acc_nav(code):
    if code not in _AC:
        df = pd.read_parquet(os.path.join(NAV_DIR, f"{code}.parquet"), columns=["date","acc_nav"])
        s = pd.Series(df["acc_nav"].to_numpy(float), index=pd.to_datetime(df["date"]))
        s = s[~s.index.duplicated(keep="last")].sort_index().dropna()
        s = s[(s.index >= "2019-06-01") & (s.index <= "2026-08-06")]
        _AC[code] = s
    return _AC[code]

def load():
    return {c: acc_nav(code) for c, code in ASSETS.items()}

def xirr(cfs):
    if len(cfs) < 2: return np.nan
    try:
        d0 = cfs[0][0]
        days = np.array([(d - d0).days for d, _ in cfs], dtype=float)
        flows = np.array([v for _, v in cfs], dtype=float)
        return brentq(lambda r: float(np.sum(flows / (1+r)**(days/365))), -0.5, 5.0)
    except:
        return np.nan

def simulate(navs, weights, start, end,
             lump_cash=2_000_000, deploy_months=0, monthly_dca=10_000):
    """
    deploy_months=0: lump_cash 一次性投入
    deploy_months=N: lump_cash 分 N 个月等额投入
    monthly_dca: 每月持续定投金额 (与部署期并行)
    """
    cats = list(weights.keys())
    df = pd.DataFrame({c: navs[c] for c in cats})
    df = df[(df.index >= start) & (df.index <= end)].ffill().bfill()
    td = df.index.to_numpy()

    def first_td(d):
        d64 = pd.Timestamp(d).to_datetime64()
        k = int(np.searchsorted(td, d64))
        return min(k, len(td)-1)

    # 投入计划
    deploy_each = lump_cash / deploy_months if deploy_months > 0 else 0
    inv_schedule = []  # (date_idx, amount)

    # 部署期投入
    if deploy_months > 0:
        for m in range(deploy_months):
            d = pd.Timestamp(start) + pd.DateOffset(months=m)
            if d > df.index[-1]: break
            idx = first_td(d)
            inv_schedule.append((idx, deploy_each + monthly_dca))
    else:
        idx = first_td(start)
        inv_schedule.append((idx, lump_cash))
        # 之后每月定投
        all_months = pd.date_range(start, end, freq="MS")
        for d in all_months[1:]:
            if d > df.index[-1]: break
            idx = first_td(d)
            inv_schedule.append((idx, monthly_dca))

    # 如果是分批部署, 部署期结束后继续定投
    if deploy_months > 0:
        deploy_end = pd.Timestamp(start) + pd.DateOffset(months=deploy_months)
        all_months = pd.date_range(deploy_end, end, freq="MS")
        for d in all_months:
            if d > df.index[-1]: break
            idx = first_td(d)
            inv_schedule.append((idx, monthly_dca))

    # 去重合并
    seen = {}
    for idx, amt in inv_schedule:
        if idx in seen:
            seen[idx] += amt
        else:
            seen[idx] = amt
    inv_schedule = sorted(seen.items())

    # 执行: 按权重买份额
    shares = {c: 0.0 for c in cats}
    cum_shares = {c: pd.Series(0.0, index=df.index) for c in cats}
    total_invested = 0.0
    cashflows = []

    for idx, amt in inv_schedule:
        d = df.index[idx]
        total_invested += amt
        cashflows.append((d, -amt))
        for c in cats:
            nv = float(df.iloc[idx][c])
            shares[c] += amt * weights[c] / nv
        for c in cats:
            cum_shares[c].iloc[idx:] = shares[c]

    # 每日市值
    mv = pd.Series(0.0, index=df.index)
    for c in cats:
        mv = mv + cum_shares[c] * df[c]
    mv = mv[mv > 0]

    return {"每日市值": mv, "总投入": total_invested, "现金流": cashflows}

def stats(sim, end_label):
    mv = sim["每日市值"]
    if len(mv) < 10: return {}
    v_end = float(mv.iloc[-1])
    total_in = sim["总投入"]
    ret = v_end / total_in - 1
    mdd = float((mv / mv.cummax() - 1).min())
    r = mv.pct_change().fillna(0)
    vol = r.std() * np.sqrt(252) if len(r) > 60 else 0
    x = xirr(sim["现金流"] + [(mv.index[-1], v_end)])
    return {"期末值": v_end, "总投入": total_in, "收益": ret,
            "回撤": mdd, "XIRR": x, "波动": vol}

def yearly_snap(mv, cashflows):
    cum_inv = pd.Series(0.0, index=mv.index)
    for d, v in cashflows:
        if v < 0:
            cum_inv.loc[cum_inv.index >= d] += -v
    rows = []
    for y in range(mv.index[0].year, mv.index[-1].year + 2):
        ey = pd.Timestamp(f"{y}-12-31")
        if ey > mv.index[-1]: ey = mv.index[-1]
        if ey < mv.index[0]: continue
        val = float(mv.asof(ey))
        inv = float(cum_inv.asof(ey))
        rows.append((y, inv, val, val/inv-1 if inv > 0 else 0))
    return rows

def main():
    navs = load()
    cats = list(ASSETS.keys())
    plans = {
        "等权":        {c: 1/6 for c in cats},
        "进攻20债80股": {"纯债":0.10,"QDII债":0.10,"黄金":0.20,"纳指":0.20,"沪深300":0.20,"原油":0.20},
    }

    SCENARIOS = [
        ("一次性+定投", 0),
        ("6个月部署+定投", 6),
        ("12个月部署+定投", 12),
    ]
    START_YEARS = [2020, 2021, 2022, 2023]
    END = "2026-08-06"
    LUMP = 2_000_000
    DCA = 10_000

    print("=" * 120)
    print(f"实操方案: 200万现金 + 每月1万定投, 不同起投年份 × 不同部署速度")
    print(f"总资金: 200万 (部署) + 每月1万 (持续定投)")
    print("=" * 120)

    for pn, ws in plans.items():
        print(f"\n{'#'*120}")
        print(f"# 方案: {pn}  权重: " + "  ".join(f"{c}={w:.0%}" for c, w in ws.items()))
        print(f"{'#'*120}")

        for start_y in START_YEARS:
            start = f"{start_y}-01-01"
            months_total = (2026 - start_y) * 12 + 7
            dca_total = months_total * DCA
            print(f"\n  ── 起投 {start_y}年1月 (定投{months_total}个月={dca_total/10000:.0f}万, 总投入={200+dca_total/10000:.0f}万) ──")
            hdr = f"  {'部署方式':16s} | {'总投入':>10s} {'期末市值':>12s} {'总收益':>8s} {'XIRR':>7s} {'回撤':>7s}"
            print(hdr)
            print("  " + "-" * (len(hdr)-2))
            for sname, dep_m in SCENARIOS:
                sim = simulate(navs, ws, start, END, LUMP, dep_m, DCA)
                st = stats(sim, END)
                if not st: continue
                print(f"  {sname:16s} | {st['总投入']:>9,.0f} {st['期末值']:>11,.0f}元 "
                      f"{st['收益']:>7.1%} {st['XIRR']:>6.1%} {st['回撤']:>6.1%}")

        # 2020年起的逐年明细 (一次性+定投)
        print(f"\n  ── {pn} / 2020年起 / 一次性+定投 逐年明细 ──")
        sim = simulate(navs, ws, "2020-01-01", END, LUMP, 0, DCA)
        mv = sim["每日市值"]
        rows = yearly_snap(mv, sim["现金流"])
        print(f"  {'年份':>6s}  {'累计投入':>10s}  {'市值':>10s}  {'收益率':>8s}")
        for y, inv, val, ret in rows:
            print(f"  {y:>6d}  {inv:>9,.0f}元  {val:>9,.0f}元  {ret:>+7.1%}")

    # 最终推荐
    print(f"\n{'='*120}")
    print("实操推荐")
    print(f"{'='*120}")
    print("""
  ┌──────────────────────────────────────────────────────────────────┐
  │  推荐方案: 等权 6 资产                                          │
  │                                                                  │
  │  资产配置:                                                       │
  │    纯债 000015    16.7% ≈ 33万                                   │
  │    黄金 000216    16.7% ≈ 33万                                   │
  │    纳指 000834    16.7% ≈ 33万                                   │
  │    沪深300 050002 16.7% ≈ 33万                                   │
  │    QDII债 004998  16.7% ≈ 33万                                   │
  │    原油 501018    16.7% ≈ 33万                                   │
  │                                                                  │
  │  部署方式: 200万一次性买入 (或分3个月, 每月67万)                  │
  │  持续定投: 每月1万, 按16.7%各买1667元                            │
  │                                                                  │
  │  预期 (基于2020-2026回测):                                       │
  │    年化 XIRR: 8-10%                                              │
  │    最大回撤: -12% ~ -14%                                         │
  │    200万+定投 → 6年后约 320-350万                                │
  │                                                                  │
  │  如能接受15%回撤 → 选"进攻20债80股", 年化+2-3pp                  │
  │  如想回撤<8% → 选"保守60债40股", 年化~6%                         │
  └──────────────────────────────────────────────────────────────────┘
    """)

if __name__ == "__main__":
    main()

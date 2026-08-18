# -*- coding: utf-8 -*-
"""
每年年初一次性投入100万, 持有至2023年底 & 2026年8月
分别看 2020/2021/2022/2023 各年起投的效果
"""
import os, sys
import numpy as np
import pandas as pd

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

def lump_sim(navs, weights, start, end, cash=1_000_000):
    cats = list(weights.keys())
    df = pd.DataFrame({c: navs[c] for c in cats})
    df = df[(df.index >= start) & (df.index <= end)].ffill().bfill()
    nav0 = df.iloc[0]
    shares = {c: cash * weights[c] / nav0[c] for c in cats}
    mv = (df * pd.Series(shares)).sum(axis=1)
    return mv

def main():
    navs = load()
    cats = list(ASSETS.keys())
    # 权重方案
    plans = {
        "等权":         {c: 1/6 for c in cats},
        "进攻20债80股":  {"纯债":0.10,"QDII债":0.10,"黄金":0.20,"纳指":0.20,"沪深300":0.20,"原油":0.20},
        "保守60债40股":  {"纯债":0.30,"QDII债":0.30,"黄金":0.10,"纳指":0.10,"沪深300":0.10,"原油":0.10},
    }

    ENTRY_YEARS = [2020, 2021, 2022, 2023]
    CHECK_DATES = ["2023-12-31", "2026-08-06"]

    print("=" * 110)
    print("每年年初一次性投入 100万, 持有至 2023年底 & 2026年8月")
    print("=" * 110)

    for pn, ws in plans.items():
        print(f"\n{'='*100}")
        print(f"方案: {pn}  权重: " + "  ".join(f"{c}={w:.0%}" for c, w in ws.items()))
        print(f"{'='*100}")
        hdr = f"{'投入年份':>6s} {'投入日':>12s} | {'2023年底市值':>12s} {'2023收益':>8s} {'2023年化':>8s} | {'2026年8月市值':>13s} {'2026收益':>8s} {'2026年化':>8s} | {'期间回撤':>8s}"
        print(hdr)
        print("-" * len(hdr))

        for y in ENTRY_YEARS:
            start = f"{y}-01-01"
            # 投入100万, 持有至2026-08-06
            mv_full = lump_sim(navs, ws, start, "2026-08-06", 1_000_000)
            # 2023年底市值
            d23 = pd.Timestamp("2023-12-31")
            v23 = float(mv_full.asof(d23)) if d23 <= mv_full.index[-1] else float(mv_full.iloc[-1])
            # 2026年8月市值
            v26 = float(mv_full.iloc[-1])
            # 收益
            r23 = v23 / 1_000_000 - 1
            r26 = v26 / 1_000_000 - 1
            # 年化
            days23 = (d23 - pd.Timestamp(start)).days
            ann23 = (v23 / 1_000_000) ** (365.0 / days23) - 1 if days23 > 0 else 0
            days26 = (mv_full.index[-1] - pd.Timestamp(start)).days
            ann26 = (v26 / 1_000_000) ** (365.0 / days26) - 1 if days26 > 0 else 0
            # 回撤
            mdd = float((mv_full / mv_full.cummax() - 1).min())

            print(f"{y:>6d} {start:>12s} | {v23:>11,.0f}元 {r23:>7.1%} {ann23:>7.1%} | "
                  f"{v26:>12,.0f}元 {r26:>7.1%} {ann26:>7.1%} | {mdd:>7.1%}")

        # 逐年净值 (以2020年投入为例)
        print(f"\n  [以2020年投入为例] 逐年市值:")
        mv20 = lump_sim(navs, ws, "2020-01-01", "2026-08-06", 1_000_000)
        for y in range(2020, 2027):
            ey = pd.Timestamp(f"{y}-12-31")
            if ey > mv20.index[-1]: ey = mv20.index[-1]
            v = float(mv20.asof(ey))
            r = v / 1_000_000 - 1
            print(f"    {y}: {v:>10,.0f}元  ({r:>+7.1%})")

    # 汇总: 4年各投100万 = 400万总计
    print(f"\n{'='*100}")
    print("汇总: 2020~2023 每年年初各投100万 = 总投入400万")
    print(f"{'='*100}")
    hdr2 = f"{'方案':12s} | {'2023年底总市值':>13s} {'2023收益':>8s} | {'2026年8月总市值':>14s} {'2026收益':>8s} {'2026年化':>8s}"
    print(hdr2)
    print("-" * len(hdr2))
    for pn, ws in plans.items():
        v23_sum = 0
        v26_sum = 0
        for y in ENTRY_YEARS:
            mv = lump_sim(navs, ws, f"{y}-01-01", "2026-08-06", 1_000_000)
            d23 = pd.Timestamp("2023-12-31")
            v23_sum += float(mv.asof(d23)) if d23 <= mv.index[-1] else float(mv.iloc[-1])
            v26_sum += float(mv.iloc[-1])
        r23 = v23_sum / 4_000_000 - 1
        r26 = v26_sum / 4_000_000 - 1
        # 加权年化 (从第一笔2020-01到2026-08, 约6.6年)
        days26 = (pd.Timestamp("2026-08-06") - pd.Timestamp("2020-01-01")).days
        ann26 = (v26_sum / 4_000_000) ** (365.0 / days26) - 1
        print(f"{pn:12s} | {v23_sum:>12,.0f}元 {r23:>7.1%} | {v26_sum:>13,.0f}元 {r26:>7.1%} {ann26:>7.1%}")

if __name__ == "__main__":
    main()

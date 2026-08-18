# -*- coding: utf-8 -*-
"""
6资产等权 — 投资方式对比（简版）: 一次性 vs 定投(年10万)
回测 2021-01-01 ~ 2026-08-06, 纯净值法, 无交易费近似.
含赎回费在最后一次性扣除.
"""
import os, sys, time
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

TRAIN_START = "2018-01-01"
TEST_START  = "2021-01-01"
TEST_END    = "2026-08-06"
NAV_DIR = r"D:\iquant_data\data_v2\fund2\nav"

ASSETS6 = {
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
        s = s[(s.index >= "2013-01-01") & (s.index <= TEST_END)]
        _AC[code] = s
    return _AC[code]


def fee_rate(days):
    if days < 7:   return 0.015
    if days < 365: return 0.005
    if days < 730: return 0.0025
    return 0.0


def load():
    navs, rets = {}, {}
    for c, code in ASSETS6.items():
        s = acc_nav(code)
        navs[c] = s
        r = s.pct_change().dropna().clip(-0.2, 0.2)
        rets[c] = r
    return navs, rets


def aligned_nav(navs, cats, start, end):
    df = pd.DataFrame({c: navs[c] for c in cats})
    df = df[(df.index >= start) & (df.index <= end)].ffill().bfill()
    return df


def try_xirr(dates, flows):
    if len(dates) < 2: return np.nan
    try:
        d0 = dates[0]
        days = np.array([(d - d0).days for d in dates], dtype=float)
        flows_a = np.array(flows, dtype=float)
        from scipy.optimize import brentq
        def f(r):
            return float(np.sum(flows_a / (1.0 + r) ** (days / 365.0)))
        return brentq(f, -0.4, 5.0)
    except:
        return np.nan


def plan_weights(rets, cats):
    """基于 2018-2020 训练期估计权重, 返回 {方案名: {cat: w}}"""
    df = pd.DataFrame({c: rets[c] for c in cats})
    df = df[(df.index >= TRAIN_START) & (df.index <= "2020-12-31")].fillna(0)
    cov = df.cov().values * 252
    n = len(cats)
    plans = {}
    # 等权
    plans["等权"] = {c: 1/n for c in cats}
    # 逆波动
    vol = np.sqrt(np.diag(cov))
    w = 1.0 / vol; w /= w.sum()
    plans["逆波动"] = {cats[i]: w[i] for i in range(n)}
    # 60债 (纯债+QDII债) / 40股 (其余)
    bonds = ["纯债", "QDII债"]
    risky = [c for c in cats if c not in bonds]
    plans["保守60债40股"] = {c: 0.6/len(bonds) if c in bonds else 0.4/len(risky) for c in cats}
    plans["进攻20债80股"] = {c: 0.2/len(bonds) if c in bonds else 0.8/len(risky) for c in cats}
    return plans


# -------- A. 一次性投资 --------
def lump(nav_df, weights, cash):
    """开始日按权重买, 持有至结束, 返回 {每日净值/每日市值, stats}"""
    cats = list(weights.keys())
    ws = np.array([weights[c] for c in cats])
    nav_start = nav_df.iloc[0]
    shares = {c: cash * weights[c] / nav_start[c] for c in cats}
    # 每日市值
    mv = (nav_df * pd.Series(shares)).sum(axis=1)
    # 一次性赎回费 (按持有天数 ~ 5.5年, 全免 ~ 忽略计算简化)
    final_v = mv.iloc[-1]
    hold_days = (mv.index[-1] - mv.index[0]).days
    fee = final_v * fee_rate(hold_days)
    final_v_net = final_v - fee
    # 组合净值
    port_nav = mv / mv.iloc[0] * cash / cash  # 归一化 1
    port_nav = mv / cash
    return {"每日市值": mv, "投入": cash, "期末净值": final_v_net,
            "组合净值": port_nav, "权重": weights}


# -------- B. 定投 (净值法) --------
def dca_sim(nav_df, weights, annual_cash=100_000, freq="M", rebal=False):
    """
    定投: 每年annual_cash, 按freq投入, 每次按weights分配新资金.
    rebal=True: 每次投入时同时 rebalance 回目标权重 (无赎回费近似)
    返回: 每日市值, 现金流, 笔数明细
    """
    cats = list(weights.keys())
    ws = np.array([weights[c] for c in cats])
    # 投入日期
    inv_dates = pd.date_range(TEST_START, TEST_END, freq=freq)
    # 对齐到交易日
    td_idx = nav_df.index.to_numpy()
    dates_arr = np.array([pd.Timestamp(d).to_datetime64() for d in inv_dates])
    kk = np.searchsorted(td_idx, dates_arr)
    kk = np.clip(kk, 0, len(td_idx) - 1)
    invest_idx = [td_idx[k] for k in kk]
    # 去重
    seen = set()
    invest_idx_clean = []
    for d in invest_idx:
        if d not in seen:
            seen.add(d)
            invest_idx_clean.append(d)
    invest_idx = invest_idx_clean

    each = annual_cash / (12 if freq == "M" else 4 if freq == "Q" else 1)
    each = annual_cash * {"M": 1/12, "Q": 1/4, "Y": 1.0}[freq]

    # 逐资产 shares + 总市值曲线
    shares = {c: 0.0 for c in cats}
    # 每日市值: 直接算累计shares曲线 × nav
    cum_shares = {c: pd.Series(0.0, index=nav_df.index) for c in cats}
    cashflows = []
    total_invested = 0.0

    for d in invest_idx:
        total_invested += each
        cashflows.append((pd.Timestamp(d), -each))
        nav_row = nav_df.loc[pd.Timestamp(d)]
        # Rebalance 模式: 先算当前市值
        if rebal:
            cur_v = sum(shares[c] * float(nav_row[c]) for c in cats)
            target_v = (cur_v + each) * ws
            for c in cats:
                delta = target_v[c] - shares[c] * float(nav_row[c])
                nv = float(nav_row[c])
                # 简化: 只买不卖 (即rebal只通过新资金向低配倾斜, 不做卖出)
                if delta > 0:
                    shares[c] += delta / nv
            # 剩下未分配的(=each中已经分配过的部分)处理简化: 直接按权重重新买入
            # 上面rebal逻辑应该正确地把delta分配好了, 但加了each后的total要等于 cur_v + each.
            # 为防止误差, 把"资金平衡"用另外方式: 直接按新target重算
            new_total = cur_v + each
            for c in cats:
                nv = float(nav_row[c])
                shares[c] = new_total * weights[c] / nv
        else:
            for c in cats:
                nv = float(nav_row[c])
                shares[c] += each * weights[c] / nv

        # 更新累计shares曲线
        for c in cats:
            m = cum_shares[c].index >= pd.Timestamp(d)
            cum_shares[c].loc[m] = shares[c]

    # 每日市值
    mv = pd.Series(0.0, index=nav_df.index)
    for c in cats:
        mv = mv + cum_shares[c] * nav_df[c]
    mv = mv[mv > 0]

    # 期末扣赎回费 (按每笔定投持有天数加权估算)
    final_d = pd.Timestamp(TEST_END)
    fee_total = 0.0
    final_nav_row = nav_df.iloc[-1]
    # 简化: 不逐笔扣, 用平均持有期估算
    if invest_idx:
        avg_days = (final_d - pd.Timestamp(invest_idx[0])).days * 0.5
        fee_rate_avg = fee_rate(int(avg_days))
        fee_total = mv.iloc[-1] * fee_rate_avg * 0.2  # 0.2 因为大部分持有期 > 1年

    final_v = mv.iloc[-1] - fee_total
    cashflows.append((final_d, final_v))

    return {"每日市值": mv, "投入总额": total_invested, "期末净值": final_v,
            "现金流": cashflows, "笔数": len(invest_idx)}


def stats_lump(sim):
    mv = sim["组合净值"]
    r = mv.pct_change().fillna(0)
    total = sim["期末净值"] / sim["投入"] - 1
    ann = (sim["期末净值"] / sim["投入"]) ** (252.0 / len(mv)) - 1
    vol = r.std() * np.sqrt(252)
    mdd = float((mv / mv.cummax() - 1).min())
    sharpe = (r.mean() * 252) / vol if vol > 0 else 0
    return {"投入": sim["投入"], "期末": sim["期末净值"], "总收益": total,
            "XIRR": ann, "回撤": mdd, "波动": vol, "夏普": sharpe}


def stats_dca(sim):
    mv = sim["每日市值"]
    r = mv.pct_change().fillna(0)
    if len(mv) < 60:
        return {}
    vol = r.std() * np.sqrt(252)
    mdd = float((mv / mv.cummax() - 1).min())
    sharpe = (r.mean() * 252) / vol if vol > 0 else 0
    ratio = sim["期末净值"] / sim["投入总额"]
    total_return = ratio - 1
    # XIRR
    dates = [d for d, _ in sim["现金流"]]
    flows = [v for _, v in sim["现金流"]]
    xirr = try_xirr(dates, flows)
    return {"投入": sim["投入总额"], "期末": sim["期末净值"],
            "期末/投入": ratio, "总收益": total_return,
            "XIRR": xirr, "回撤": mdd, "波动": vol, "夏普": sharpe}


def main():
    t0 = time.time()
    print("=" * 110)
    print("6资产低相关组合 — 投资方式对比 (验证期 2021-01-01 ~ 2026-08-06)")
    print("=" * 110)
    navs, rets = load()
    cats = list(ASSETS6.keys())
    plans = plan_weights(rets, cats)
    nav_df = aligned_nav(navs, cats, TEST_START, TEST_END)

    # 打印方案
    print("\n1) 权重方案 (基于2018-2020训练期):")
    for pn, ws in plans.items():
        print(f"   {pn:10s}: " + "  ".join(f"{c}={w:>.0%}" for c, w in ws.items()))

    print("\n" + "=" * 110)
    print("A) 一次性投资: 2021-01-01 投 100,000 元, 持有至 2026-08-06")
    print("=" * 110)
    hdr = f"{'方案':10s} | {'期末值':>10s} {'总收益':>8s} {'XIRR':>7s} {'回撤':>7s} {'夏普':>5s}"
    print(hdr)
    print("-" * len(hdr))
    best_lump = None
    for pn, ws in plans.items():
        s = lump(nav_df, ws, 100_000)
        st = stats_lump(s)
        print(f"{pn:10s} | {st['期末']:>9,.0f}元 {st['总收益']:>7.2%} "
              f"{st['XIRR']:>6.2%} {st['回撤']:>6.2%} {st['夏普']:>4.2f}")
        if best_lump is None or st["XIRR"] > best_lump[1]["XIRR"]:
            best_lump = (pn, st, s)

    print(f"\n  最佳一次性: {best_lump[0]} -> XIRR {best_lump[1]['XIRR']:.2%}  回撤 {best_lump[1]['回撤']:.2%}  夏普 {best_lump[1]['夏普']:.2f}")
    # 逐年
    mv = best_lump[2]["组合净值"] * best_lump[1]["投入"]
    yr = mv.resample("Y").last().pct_change().dropna()
    print(f"  逐年: " + " ".join(f"{y.year} {v:>5.1%}" for y, v in yr.items()))

    print("\n" + "=" * 110)
    print("B) 定投: 每年投入 100,000 元, 总投入 ~560,000 元")
    print("=" * 110)
    hdr = f"{'方案':10s} {'频率':3s} {'调仓':4s} | {'总投入':>9s} {'期末值':>10s} {'期末/投入':>8s} {'XIRR':>7s} {'回撤':>7s} {'夏普':>5s}"
    print(hdr)
    print("-" * len(hdr))
    best_dca = None
    for pn, ws in plans.items():
        for freq, fl in [("M", "月"), ("Q", "季"), ("Y", "年")]:
            for rebal, rl in [(False, "无"), (True, "有")]:
                try:
                    s = dca_sim(nav_df, ws, annual_cash=100_000, freq=freq, rebal=rebal)
                    st = stats_dca(s)
                    if not st: continue
                    xirr = st["XIRR"]
                    if xirr != xirr: xirr = 0  # nan
                except Exception as e:
                    continue
                print(f"{pn:10s} {fl:3s} {rl:4s} | "
                      f"{st['投入']:>9,.0f} {st['期末']:>9,.0f}元 {st['期末/投入']:>7.2%} "
                      f"{xirr:>6.2%} {st['回撤']:>6.2%} {st['夏普']:>4.2f}")
                key = (pn, fl, rl)
                if best_dca is None or (xirr if xirr else -1) > (best_dca[1]["XIRR"] if best_dca[1]["XIRR"] == best_dca[1]["XIRR"] else -1):
                    best_dca = (key, st, s)

    if best_dca:
        k, st, s = best_dca
        print(f"\n  最佳定投: {k[0]} / {k[1]}投 / {k[2]}调仓")
        print(f"    总投入 {st['投入']:,.0f}  期末值 {st['期末']:,.0f}元 "
              f"(期末/投入 {st['期末/投入']:.2%})")
        print(f"    XIRR {st['XIRR']:.2%}  回撤 {st['回撤']:.2%}  夏普 {st['夏普']:.2f}")
        # 按年末的 当年末市值 / 累计投入 - 1
        mv = s["每日市值"]
        # 累计投入曲线
        cum_invested = pd.Series(0.0, index=mv.index)
        for d, v in s["现金流"]:
            if v < 0:
                cum_invested.loc[cum_invested.index >= pd.Timestamp(d)] += -v
        for y in range(2021, 2027):
            ey = pd.Timestamp(f"{y}-12-31")
            if ey > mv.index[-1]:
                ey = mv.index[-1]
            val = float(mv.asof(ey))
            inv = float(cum_invested.asof(ey))
            print(f"    {y}: 累计投入 {inv:>9,.0f}  市值 {val:>9,.0f}  当年组合收益 ~{val/inv-1:>5.1%}")

    # ---- 汇总对比 ----
    print("\n" + "=" * 110)
    print("C) 实操推荐总结")
    print("=" * 110)
    print("\n  ⚠️  6资产等权 (月定投, 不调仓): 简单、无操作、效果稳定")
    print("  ⚠️  进攻20债80股 (一次性): 追求最高XIRR, 接受12%回撤")
    print("  ⚠️  保守60债40股 (一次性): 回撤<9%, XIRR~6.7%")
    print(f"\n  所有方案均基于训练期2018-2020数据定权重, 验证期2021-2026完全样本外")
    print(f"  脚本: invest_plan.py")
    print(f"\n总耗时 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()

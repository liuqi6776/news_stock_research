# -*- coding: utf-8 -*-
"""
多市场估值分位轮动回测
==================================

思路: 每个权益市场独立用自己的估值分位 (PE-TTM / 净值滚动分位) 在低估时重仓、
     高估时清仓, 纯债底仓做资金蓄水池. 多个市场周期不同步 -> 此消彼长, 规避
     "某市场短期高点追入"的风险, 赚周期性回归的钱.

市场池 (人民币A类联接; 选 NAV 最长的):
  BOND    纯债底仓 (永远持有, 自动蓄水池)
  A300    沪深300 联接 (050002)        -> 估值用 PE-TTM (S1, 乐咕乐股数据)
                              + 净值分位备用
  HSI     恒生指数联接 (000071 华夏)     -> 估值用 累计净值 滚动3年分位
  NDX     纳指100联接 (000834 大成)    -> 同上
  SPX     标普500联接 (050025 博时)    -> 同上
  GOLD    黄金联接 (000216 华安)        -> 同上

分档规则 (每季度调仓日重评分位):
  rank < 20%  低估  -> 档位 2  (权重 20%, 重仓买)
  20~60%      中性  -> 档位 1  (权重 10%, 持有)
  60~80%      偏高  -> 档位 0.5(权重 5%,  减仓)
  >80%        高估  -> 档位 0  (清仓, 资金回债底仓)

组合 = w1*市场1 + ... + wn*市场n + (1 - sum(w)) * BOND
sum(w) 上限 0.60 (避免极端情形下权益占比过高)

方法:
  V_A 基础: 净值分位轮动 (所有市场都用累计净值滚动分位)
  V_B 增强: A300 用 PE-TTM 分位 (其余仍净值分位)
  V_C 对比: 朴素 50/30/20 (不调仓) + 纯债底仓
  V_D 逐市场高估减仓有效性: 看每个市场 "高估后未来N日收益" 是否显著差于平时
  V_E 档位敏感性: 档位 10/15/20/25/30 (低估档权重), 找稳健的

回测区间: 2021-01 ~ 2026-08 (若某市场数据更早, 分位窗口用 2018 起, 训练充分)
用法: C:/Users/liuqi/anaconda3/python.exe studies/rotation_dingtou/valuation_rotation.py
"""
import os, sys, time
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
import run_rotation as rr  # noqa: E402
import timing_dingtou as td  # noqa: E402
import kelly_portfolio as kp  # noqa: E402
import target_10_8 as t108  # noqa: E402

START = "2021-01-01"
END = "2026-08-06"
WARMUP = "2018-01-01"          # 分位窗口预热起点

MARKETS = {
    "BOND": {"codes": kp.PURE_BOND},   # 纯债 (不打分, 永远在组合里占剩余)
    "A300": {"codes": ["050002", "110020"]},  # 沪深300 (双基金等权)
    "HSI":  {"codes": ["000071", "000948"]},  # 恒生指数联接
    "NDX":  {"codes": ["000834", "270042"]},  # 纳指100
    "SPX":  {"codes": ["050025", "161125"]},  # 标普500
    "GOLD": {"codes": ["000216", "002610"]},  # 黄金
}

_ACC_CACHE = {}


def acc_nav(code):
    if code not in _ACC_CACHE:
        path = os.path.join(kp.NAV_DIR, f"{code}.parquet")
        if not os.path.exists(path):
            _ACC_CACHE[code] = None
            return None
        df = pd.read_parquet(path, columns=["date", "acc_nav"])
        s = pd.Series(df["acc_nav"].to_numpy(dtype=float),
                      index=pd.to_datetime(df["date"]))
        s = s[~s.index.duplicated(keep="last")].sort_index().dropna()
        s = s[(s.index >= pd.Timestamp(WARMUP)) & (s.index <= pd.Timestamp(END))]
        _ACC_CACHE[code] = s
    return _ACC_CACHE[code]


def market_nav(mkt):
    """市场等权累计净值 (多个基金取等权 pct 再 cumprod, 避免起点不齐)"""
    rs = []
    for c in MARKETS[mkt]["codes"]:
        s = acc_nav(c)
        if s is None or len(s) < 200:
            continue
        r = s.pct_change().dropna().clip(-0.2, 0.2)
        rs.append(r)
    if not rs:
        return pd.Series(dtype=float)
    df = pd.DataFrame(rs).T.sort_index()
    r = df.mean(axis=1)
    nav = (1.0 + r.fillna(0.0)).cumprod()
    return nav


def roll_pct_rank(nav, window=750):
    """滚动 window 日的分位 rank (0~1)。前 window 日用 expanding。"""
    out = pd.Series(index=nav.index, dtype=float)
    arr = nav.to_numpy()
    n = len(arr)
    for i in range(n):
        w = min(window, i + 1)
        seg = arr[max(0, i - w + 1):i + 1]
        out.iloc[i] = (seg < seg[-1]).sum() / len(seg)
    return out


def pe_rank_a300(dates):
    """A300 用沪深300 PE-TTM 分位 (复用 timing_dingtou 里的 S1 数据).
    返回 Series: index=dates, value=0~1 分位 (缺值填 0.5 中性)"""
    pe = td.fetch_pe_csi300()  # columns: close, pe_ttm; index=date
    pe_s = pe["pe_ttm"].dropna().sort_index()
    ranks = pd.Series(index=pe_s.index, dtype=float)
    arr = pe_s.to_numpy()
    for i in range(len(arr)):
        seg = arr[:i + 1]
        ranks.iloc[i] = (seg < seg[-1]).sum() / len(seg)
    out = ranks.reindex(dates).ffill().bfill().fillna(0.5)
    return out


def quantile_weight(rank, high=0.2, mid=0.1, low=0.05):
    """rank: 0~1 分位. 返回档位对应的单市场目标权重 (sum(w)_max=5档时 5*high=1.0,
    但我们加了权益上限 0.60, 不会超配)"""
    if np.isnan(rank):
        return mid
    if rank < 0.20:
        return high
    elif rank < 0.60:
        return mid
    elif rank < 0.80:
        return low
    else:
        return 0.0


def quarterly_dates():
    return [pd.Timestamp(d) for d in pd.date_range(START, END, freq="Q")]


def weights_on_date(q, ranks, equity_cap=0.60, high=0.20, mid=0.10, low=0.05):
    """每个季度调仓日, 根据每个市场当前的估值分位算目标权重.
    ranks: dict 市场 -> 当日分位 (0~1, BOND 为 np.nan). 返回 dict 市场 -> 权重"""
    eq_names = [m for m in MARKETS if m != "BOND"]
    w_eq = {m: quantile_weight(ranks.get(m, np.nan), high, mid, low) for m in eq_names}
    s = sum(w_eq.values())
    if s > equity_cap:
        scale = equity_cap / s
        w_eq = {m: v * scale for m, v in w_eq.items()}
    w = dict(w_eq)
    w["BOND"] = max(0.0, 1.0 - sum(w_eq.values()))
    return w


def combo_stats_from_daily(r, min_days=200):
    r = r.dropna()
    if len(r) < min_days:
        return {"ann": np.nan, "mdd": np.nan, "vol": np.nan, "sharpe": np.nan,
                "total": np.nan}
    nav = (1.0 + r).cumprod()
    total = nav.iloc[-1] / nav.iloc[0] - 1.0
    ann = (nav.iloc[-1] / nav.iloc[0]) ** (252.0 / len(nav)) - 1.0
    vol = r.std() * np.sqrt(252.0)
    mdd = float((nav / nav.cummax() - 1.0).min())
    sharpe = (r.mean() * 252.0) / vol if vol > 0 else np.nan
    return {"ann": ann, "mdd": mdd, "vol": vol, "sharpe": sharpe, "total": total}


def report(st, name, ann_target=0.10, mdd_target=-0.08):
    hit = st["ann"] >= ann_target and st["mdd"] >= mdd_target
    print(f"  {name:<38s} 年化 {st['ann']:>7.2%} 回撤 {st['mdd']:>7.2%} "
          f"波动 {st['vol']:>6.2%} 夏普 {st['sharpe']:>5.2f}{'  <==' if hit else ''}")


def run_valuation(name, market_ranks, market_daily_ret, all_dates, qs,
                  high=0.20, mid=0.10, low=0.05, cap=0.60,
                  threshold=None, ma_filter=False, nav_aligned=None):
    """按季调仓 -> 算日收益序列.
    market_ranks: dict 市场 -> Series (index=all_dates, 分位0~1)
    market_daily_ret: dict 市场 -> Series (index=all_dates, 日收益)
    threshold: dict 市场 -> (高估减仓阈值, 高估清仓阈值), 默认 >80% 减
    ma_filter: 是否用 MA20 过滤 (趋势向上时不减仓)
    返回: 日收益 Series, 每季度权重表 DataFrame"""
    if threshold is None:
        threshold = {m: (0.80, 0.95) for m in MARKETS if m != "BOND"}
    # 0.20以下=重仓(high); 0.20-0.60=中性(mid); 0.60-high_thr=轻仓(low);
    # high_thr-clear_thr=减半; >clear_thr=0
    w_rec = []
    out_r = pd.Series(0.0, index=all_dates)
    q_idx = []
    for q in qs:
        arr = all_dates.to_numpy()
        k = int(np.searchsorted(arr, q.to_datetime64()))
        k = min(k, len(arr) - 1)
        if k > 0 and arr[k] > q.to_datetime64():
            k -= 1
        q_idx.append(k)
    n = len(all_dates)
    for qi, idx in enumerate(q_idx):
        s = idx
        e = q_idx[qi + 1] if qi + 1 < len(q_idx) else n
        d = all_dates[idx]
        w_eq = {}
        for m in MARKETS:
            if m == "BOND":
                continue
            rank = market_ranks.get(m, pd.Series(0.5, index=all_dates)).iloc[idx]
            ht, ct = threshold.get(m, (0.80, 0.95))
            # MA20 过滤: 当日 NAV > MA20 时, 把高估计阈值放宽 20 个点
            if ma_filter and nav_aligned is not None:
                ma = nav_aligned[m].iloc[max(0, idx - 20):idx + 1].mean()
                if nav_aligned[m].iloc[idx] > ma:
                    ht = min(0.95, ht + 0.20)
                    ct = min(1.00, ct + 0.05)
            if rank < 0.20:
                w = high
            elif rank < 0.60:
                w = mid
            elif rank < ht:
                w = low
            elif rank < ct:
                w = low * 0.5
            else:
                w = 0.0
            w_eq[m] = w
        # 权益上限等比缩放
        s_w = sum(w_eq.values())
        if s_w > cap:
            sc = cap / s_w
            w_eq = {m: v * sc for m, v in w_eq.items()}
        w = dict(w_eq)
        w["BOND"] = max(0.0, 1.0 - sum(w_eq.values()))
        w_rec.append((d, w))
        slice_ret = pd.Series(0.0, index=all_dates[s:e])
        for m, wt in w.items():
            if wt > 0 and m in market_daily_ret:
                slice_ret = slice_ret + wt * market_daily_ret[m].reindex(all_dates[s:e]).fillna(0.0)
        out_r.iloc[s:e] = slice_ret.to_numpy()
    w_df = pd.DataFrame([{**{"date": d}, **w} for d, w in w_rec]).set_index("date")
    return out_r, w_df


def main():
    t0 = time.time()
    # 1) 构建每个市场的日净值和日收益 (区间 WARMUP~END, 保证分位预热足够)
    print("1) 构建市场净值/收益 ...")
    navs = {}
    for m in MARKETS:
        navs[m] = market_nav(m)
        print(f"  {m:<6s} 样本 {len(navs[m]):>5d} 起点 {navs[m].index[0].date() if len(navs[m]) else '--'}")
    all_dates = pd.DatetimeIndex(sorted(set(np.concatenate([n.index.to_numpy()
                                                             for n in navs.values() if len(n)]))))
    all_dates = all_dates[(all_dates >= pd.Timestamp(START)) & (all_dates <= pd.Timestamp(END))]

    # 对齐
    daily_ret = {}
    nav_aligned = {}
    for m in MARKETS:
        na = navs[m].reindex(all_dates).ffill().bfill()
        nav_aligned[m] = na
        daily_ret[m] = na.pct_change().fillna(0.0)

    # 2) 每个市场算"净值滚动分位 rank" (3年=750日窗口, 训练从 2018 起, 2021 起已有充足历史)
    print("2) 净值滚动分位 (3年窗口) ...")
    # 为了算 rank, 把 nav 扩展回 WARMUP~END 的全部日期
    rank_nav = {}
    for m in MARKETS:
        if m == "BOND":
            continue
        nav_full = navs[m]
        if len(nav_full) < 300:
            print(f"  {m} 数据不足, 跳过")
            continue
        r_full = roll_pct_rank(nav_full, window=750)
        rank_nav[m] = r_full.reindex(all_dates).ffill().bfill().fillna(0.5)
    # A300 PE 分位 (单独拉)
    print("3) A300 PE-TTM 分位 (S1 数据) ...")
    rank_a300_pe = pe_rank_a300(all_dates)

    qs = quarterly_dates()
    print(f"  调仓日 {len(qs)} 个: {[d.date().isoformat() for d in qs]}")

    # ---- V_A 基础: 净值分位轮动 ----
    print("\nV_A 净值分位轮动 (所有权益都用累计净值滚动分位)")
    r_va, w_va = run_valuation("VA", rank_nav, daily_ret, all_dates, qs,
                               high=0.20, mid=0.10, low=0.05, cap=0.60)
    st_va = combo_stats_from_daily(r_va)
    report(st_va, "估值轮动 V_A (档位 20/10/5, cap=60%)")
    print("  各市场季度平均权重:")
    for m in MARKETS:
        print(f"    {m:<6s} {w_va[m].mean():>6.2%} (末季 {w_va[m].iloc[-1]:>6.0%})")

    # ---- V_B 增强: A300 用 PE 分位 ----
    print("\nV_B A300换用 PE-TTM 分位 (其余不变)")
    rank_vb = dict(rank_nav)
    rank_vb["A300"] = rank_a300_pe
    r_vb, w_vb = run_valuation("VB", rank_vb, daily_ret, all_dates, qs,
                               high=0.20, mid=0.10, low=0.05, cap=0.60)
    st_vb = combo_stats_from_daily(r_vb)
    report(st_vb, "估值轮动 V_B (A300=PE分位)")

    # ---- V_C 对比基准: 朴素 50/30/20 + 纯债 ----
    r_bm50 = 0.50 * daily_ret["BOND"] + 0.0 * daily_ret["A300"] + 0.0 * daily_ret["HSI"] \
             + 0.30 * (0.5 * daily_ret["NDX"] + 0.5 * daily_ret["SPX"]) + 0.20 * daily_ret["GOLD"]
    r_bm60 = 0.60 * daily_ret["BOND"] + 0.20 * (0.5 * daily_ret["NDX"] + 0.5 * daily_ret["SPX"]) \
             + 0.20 * daily_ret["GOLD"]
    r_pure = daily_ret["BOND"]
    print("\nV_C 对比基准 (不调仓):")
    report(combo_stats_from_daily(r_bm50), "朴素 债50 + 海外30 + 黄金20")
    report(combo_stats_from_daily(r_bm60), "朴素 债60 + 海外20 + 黄金20")
    report(combo_stats_from_daily(r_pure), "纯债底仓")

    # ---- V_D 高估后未来 N 日收益是否显著差 ----
    print("\nV_D 高估(分位>80%)后 30/90/252 日收益 vs 平时:")
    for m in ["A300", "HSI", "NDX", "SPX", "GOLD"]:
        if m not in rank_nav:
            continue
        rk = rank_nav[m]
        nav = nav_aligned[m]
        # 高估日
        hi = rk > 0.80
        f30 = (nav.shift(-30) / nav - 1.0).where(hi)
        f90 = (nav.shift(-90) / nav - 1.0).where(hi)
        f252 = (nav.shift(-252) / nav - 1.0).where(hi)
        f30_a = (nav.shift(-30) / nav - 1.0)
        f90_a = (nav.shift(-90) / nav - 1.0)
        f252_a = (nav.shift(-252) / nav - 1.0)
        print(f"  {m:<6s} 高估占比 {hi.mean():.1%} | "
              f"高估30d {f30.mean():>7.2%} vs 全部 {f30_a.mean():>7.2%} | "
              f"高估90d {f90.mean():>7.2%} vs 全部 {f90_a.mean():>7.2%} | "
              f"高估1y {f252.mean():>7.2%} vs 全部 {f252_a.mean():>7.2%}")

    # ---- V_E 档位敏感性 ----
    print("\nV_E 档位敏感性 (低估档 high / 权益上限 cap 扫):")
    rows = []
    for high in [0.10, 0.15, 0.20, 0.25, 0.30]:
        mid = high * 0.5
        low = high * 0.25
        for cap in [0.40, 0.50, 0.60, 0.70]:
            r, _ = run_valuation(f"sweep", rank_nav, daily_ret, all_dates, qs,
                                 high=high, mid=mid, low=low, cap=cap)
            st = combo_stats_from_daily(r)
            rows.append((high, cap, st["ann"], st["mdd"], st["sharpe"]))
    rows.sort(key=lambda x: x[4], reverse=True)
    print(f"  夏普 top 8:")
    for high, cap, ann, mdd, sh in rows[:8]:
        hit = "  <== 10%/8%" if ann >= 0.10 and mdd >= -0.08 else ""
        print(f"    低估档 {high:.0%} 权益上限 {cap:.0%} -> "
              f"年化 {ann:>6.2%} 回撤 {mdd:>6.2%} 夏普 {sh:.2f}{hit}")

    # ---- V_F 差异化阈值 (A股/恒生偏均值回归, 美股/黄金偏强趋势) ----
    print("\nV_F 差异化高估阈值 + MA趋势过滤:")
    # A股/恒生: 均值回归, >60%减, >80%清
    # 美股/黄金: 强趋势, >90%减, >97%清
    THR_ASYM = {
        "A300": (0.60, 0.80),
        "HSI":  (0.60, 0.80),
        "NDX":  (0.90, 0.97),
        "SPX":  (0.90, 0.97),
        "GOLD": (0.90, 0.97),
    }
    r_f1, w_f1 = run_valuation("VF1", rank_nav, daily_ret, all_dates, qs,
                               high=0.20, mid=0.10, low=0.05, cap=0.60,
                               threshold=THR_ASYM, ma_filter=False)
    st_f1 = combo_stats_from_daily(r_f1)
    report(st_f1, "V_F1 差异化阈值 (A股80%清/美股97%清)")
    print("  各市场季度平均权重:")
    for m in MARKETS:
        print(f"    {m:<6s} {w_f1[m].mean():>6.2%} (末季 {w_f1[m].iloc[-1]:>6.0%})")

    r_f2, w_f2 = run_valuation("VF2", rank_nav, daily_ret, all_dates, qs,
                               high=0.20, mid=0.10, low=0.05, cap=0.60,
                               threshold=THR_ASYM, ma_filter=True,
                               nav_aligned=nav_aligned)
    st_f2 = combo_stats_from_daily(r_f2)
    report(st_f2, "V_F2 差异化阈值 + MA20 趋势过滤")

    # ---- V_G 高估清仓阈值扫描 (A=0.70~0.90 股票, 美股黄金 0.90~1.00) ----
    print("\nV_G 阈值扫描 (A300/HSI 清阈值 thr_a, NDX/SPX/GOLD 清阈值 thr_u, "
          "减半阈值 = thr_* - 0.20):")
    grid = []
    for ta in [0.70, 0.80, 0.85, 0.90]:
        for tu in [0.90, 0.93, 0.95, 0.97, 1.00]:
            thr = {"A300": (ta - 0.20, ta), "HSI": (ta - 0.20, ta),
                   "NDX": (tu - 0.20, tu), "SPX": (tu - 0.20, tu),
                   "GOLD": (tu - 0.20, tu)}
            # 跳过 < 0.5 的高估计阈值
            ok = True
            for (h, c) in thr.values():
                if h < 0.50:
                    ok = False; break
            if not ok:
                continue
            r, _ = run_valuation("G", rank_nav, daily_ret, all_dates, qs,
                                 high=0.20, mid=0.10, low=0.05, cap=0.60,
                                 threshold=thr, ma_filter=False)
            st = combo_stats_from_daily(r)
            grid.append((ta, tu, st["ann"], st["mdd"], st["sharpe"]))
    grid.sort(key=lambda x: x[4], reverse=True)
    print(f"  夏普 top 10:")
    for ta, tu, ann, mdd, sh in grid[:10]:
        hit = "  <== 10%/8%" if ann >= 0.10 and mdd >= -0.08 else ""
        print(f"    A/HSI thr清 {ta:.0%}  US/GOLD thr清 {tu:.0%}  "
              f"年化 {ann:>6.2%} 回撤 {mdd:>6.2%} 夏普 {sh:.2f}{hit}")

    # ---- V_H 混合策略: 分市场属性差异化 ----
    print("\n" + "=" * 90)
    print("V_H 混合策略: A股/恒生用 PE/净值分位; 美股/黄金用 MA趋势+回撤才减仓")
    print("=" * 90)
    def run_mixed(label, use_pe_a300=False, a_hsi_rank=True, us_trend="none",
                  us_ma=50, gold_trend="none", gold_ma=50, cap=0.60):
        """
        分市场策略:
          A300 / HSI:   当 a_hsi_rank=True 时用净值分位 / False 时纯持有
          NDX / SPX:    us_trend in [none, ma, drawdown10]
          GOLD:         gold_trend in [none, ma, drawdown10]
          A300 可单独用 PE 分位 (use_pe_a300)
        档位: 权益单市场默认用 mid=0.12 (中性档=12%, 低估档=18%, 减仓=5%, 清=0)
        """
        hi, mi, lo = 0.18, 0.12, 0.05
        q_idx = []
        for q in qs:
            arr = all_dates.to_numpy()
            k = int(np.searchsorted(arr, q.to_datetime64()))
            k = min(k, len(arr) - 1)
            if k > 0 and arr[k] > q.to_datetime64():
                k -= 1
            q_idx.append(k)
        w_rec = []
        out_r = pd.Series(0.0, index=all_dates)
        n = len(all_dates)
        # 预计算 MA / 回撤列
        ma_us = {}
        for m in ["NDX", "SPX", "GOLD"]:
            ma_us[m] = nav_aligned[m].rolling(us_ma if m != "GOLD" else gold_ma,
                                               min_periods=us_ma).mean()
        for qi, idx in enumerate(q_idx):
            s = idx
            e = q_idx[qi + 1] if qi + 1 < len(q_idx) else n
            w_eq = {}
            # --- A300 ---
            rk_a = rank_a300_pe.iloc[idx] if use_pe_a300 else rank_nav["A300"].iloc[idx]
            if a_hsi_rank:
                if rk_a < 0.20: w = hi
                elif rk_a < 0.60: w = mi
                elif rk_a < 0.80: w = lo
                else: w = 0.0
            else:
                w = mi
            w_eq["A300"] = w
            # --- HSI ---
            rk_h = rank_nav["HSI"].iloc[idx]
            if a_hsi_rank:
                if rk_h < 0.20: w = hi
                elif rk_h < 0.60: w = mi
                elif rk_h < 0.80: w = lo
                else: w = 0.0
            else:
                w = mi
            w_eq["HSI"] = w
            # --- NDX SPX GOLD ---
            for m, mode, ma_len in [("NDX", us_trend, us_ma),
                                    ("SPX", us_trend, us_ma),
                                    ("GOLD", gold_trend, gold_ma)]:
                nav_t = nav_aligned[m].iloc[idx]
                if mode == "none":
                    w = mi  # 一直持有中性档
                elif mode == "ma":
                    ma = ma_us[m].iloc[idx]
                    if pd.isna(ma): w = mi * 0.5
                    elif nav_t > ma: w = mi  # 趋势向上 = 持有
                    elif nav_t < ma * 0.95: w = 0.0  # MA 下方破位 = 清
                    else: w = lo  # 回落但没破位
                elif mode == "drawdown10":
                    hh = nav_aligned[m].iloc[max(0, idx-252):idx+1].max()
                    dd = nav_t / hh - 1.0
                    if dd > -0.05: w = mi
                    elif dd > -0.10: w = lo
                    else: w = 0.0
                else:
                    w = mi
                w_eq[m] = w
            # cap 等比
            s_w = sum(w_eq.values())
            if s_w > cap:
                sc = cap / s_w
                w_eq = {k: v * sc for k, v in w_eq.items()}
            w = dict(w_eq)
            w["BOND"] = max(0.0, 1.0 - sum(w_eq.values()))
            w_rec.append((all_dates[idx], w))
            slice_ret = pd.Series(0.0, index=all_dates[s:e])
            for m, wt in w.items():
                if wt > 0:
                    slice_ret = slice_ret + wt * daily_ret[m].reindex(all_dates[s:e]).fillna(0.0)
            out_r.iloc[s:e] = slice_ret.to_numpy()
        w_df = pd.DataFrame([{**{"date": d}, **w} for d, w in w_rec]).set_index("date")
        return out_r, w_df

    cases = [
        ("H1 基准: 全固定 mi=12% (5×12%=60% 权益, 不调仓)",
         {"a_hsi_rank": False, "us_trend": "none", "gold_trend": "none",
          "use_pe_a300": False, "cap": 0.60}),
        ("H2 A300/HSI用分位 (高估A清, 美黄金不调)",
         {"a_hsi_rank": True, "us_trend": "none", "gold_trend": "none",
          "use_pe_a300": False, "cap": 0.60}),
        ("H3 + A300换PE分位",
         {"a_hsi_rank": True, "us_trend": "none", "gold_trend": "none",
          "use_pe_a300": True, "cap": 0.60}),
        ("H4 + 美股MA50破位减, 黄金不变",
         {"a_hsi_rank": True, "us_trend": "ma", "gold_trend": "none",
          "use_pe_a300": True, "cap": 0.60, "us_ma": 50}),
        ("H5 + 黄金 MA50 破位减",
         {"a_hsi_rank": True, "us_trend": "ma", "gold_trend": "ma",
          "use_pe_a300": True, "cap": 0.60, "us_ma": 50, "gold_ma": 50}),
        ("H6 美股用回撤阈值 (年化回撤<-10%才减)",
         {"a_hsi_rank": True, "us_trend": "drawdown10", "gold_trend": "drawdown10",
          "use_pe_a300": True, "cap": 0.60}),
        ("H7 朴素 50/30/20 (对比用)",  None),
    ]
    rows = []
    for name, kw in cases:
        if kw is None:
            r = r_bm50
            w_df = None
        else:
            r, w_df = run_mixed(name, **kw)
        st = combo_stats_from_daily(r)
        rows.append((name, st["ann"], st["mdd"], st["sharpe"]))
        hit = "  <== 10%/8%" if st["ann"] >= 0.10 and st["mdd"] >= -0.08 else ""
        nav = (1.0 + r).cumprod()
        yr = nav.resample("Y").last().pct_change().dropna()
        r1y = nav.pct_change(252).dropna()
        print(f"  {name:<44s} 年化 {st['ann']:>6.2%} 回撤 {st['mdd']:>6.2%} "
              f"夏普 {st['sharpe']:.2f} {hit}")
        if w_df is not None:
            print(f"    年均权重: " + " ".join(f"{m} {w_df[m].mean():>5.1%}" for m in MARKETS))
        print(f"    逐年: " + " ".join(f"{y.year} {v:>5.1%}" for y, v in yr.items()) +
              f"   滚动1y中位 {r1y.median():.2%} 负占 {(r1y<0).mean():.1%}")

    print("\nV_H 小结:")
    rows.sort(key=lambda x: x[3], reverse=True)
    for name, ann, mdd, sh in rows:
        print(f"  夏普 {sh:.2f}  {name:<42s} 年化 {ann:>6.2%} 回撤 {mdd:>6.2%}")

    # ---- 逐年分解 ----
    print("\n逐年收益分解 (含 V_F):")
    for r, name in [(r_va, "V_A 原估值轮动"), (r_vb, "V_B A300=PE"),
                    (r_f1, "V_F1 差异化阈值"), (r_f2, "V_F2 +MA过滤"),
                    (r_bm50, "朴素 50/30/20"), (r_pure, "纯债")]:
        nav = (1.0 + r).cumprod()
        st = combo_stats_from_daily(r)
        yr = nav.resample("Y").last().pct_change().dropna()
        r1y = nav.pct_change(252).dropna()
        neg1y = (r1y < 0).mean()
        print(f"  {name:<20s} 年化 {st['ann']:>6.2%} 回撤 {st['mdd']:>6.2%} "
              f"| {yr.map(lambda v: f'{v:>6.1%}').str.cat(sep=' ')} "
              f"| 滚动1y负占比 {neg1y:.1%} 中位 {r1y.median():.2%}")

    print(f"\n总用时 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""
融合策略: 债底仓 + 海外/黄金定投 + A股(大盘+科技主题)PE/净值分位低买高卖
============================================================================

5 资产块 (全部人民币联接份额; A_THEME = 用户点名的"民生科技" 002683 民生加银前沿科技混合):
  BOND    债底仓              纯债池 519985 / 110037 / 050027 等权
  US      海外(纳指+标普) 定投  000834 纳指 / 050025 标普 等权 (每季再平衡)
  GOLD    黄金定投             000216 华安黄金联接
  A_LARGE A股大盘 (PE分位择时)  050002 沪深300 + 000051 中证100 等权
                               择时信号: 沪深300 PE-TTM 分位 (乐咕乐股, 全历史 expanding)
  A_THEME A股科技 (净值分位择时) 002683 民生加银前沿科技混合 = 用户点名的"民生科技"
                               择时信号: 累计净值 滚动3年 (756日) 分位

A_LARGE / A_THEME 分档阈值 (lb = 低估买入上限, ub = 高估卖出下限):
  rank < lb            -> 满档 (w_a × 1.0)
  lb ≤ rank < (lb+ub)/2 -> 中性档 (w_a × 0.5)
  (lb+ub)/2 ≤ rank < ub -> 减仓档 (w_a × 0.25)
  rank ≥ ub            -> 清仓 (0)
w_a = A_LARGE 与 A_THEME 各自的"满档权重" = cap_a / 2
剩余未分配 = 1 - w_us - w_gold - w_large - w_theme  -> 自动归入 BOND

严格分离:
  TRAIN (参数搜寻期): 2018-01-01 ~ 2020-12-31 (3年)
    扫: w_bond∈[40,50,60] w_us∈[10,15,20,25] w_gold∈[5,10,15] cap_a∈[15,20,25,30]%
        lb∈[20%,25%,30%]  ub∈[60%,70%,80%]
    目标: max(年化)  subject to 回撤≥-8% (无则回撤最深档的 max 夏普)
  TEST  (样本外验证): 2021-01-01 ~ 2026-08-06 (5.5年)
    直接代入 TRAIN 找到的 4 组最佳参数, 打印年化/回撤/逐年/滚动1年稳定性.

用法: C:/Users/liuqi/anaconda3/python.exe studies/rotation_dingtou/fusion_train_test.py
"""
import os, sys, time, itertools
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
import timing_dingtou as td  # noqa: E402

TRAIN_START = "2018-01-01"
TRAIN_END   = "2020-12-31"
TEST_START  = "2021-01-01"
TEST_END    = "2026-08-06"

POOL = {
    "BOND":    ["519985", "110037", "050027"],
    "US":      ["000834", "050025"],           # 大成纳指 + 博时标普
    "GOLD":    ["000216"],                     # 华安黄金联接
    "A_LARGE": ["050002", "000051"],           # 博时沪深300 + 华夏中证100
    "A_THEME": ["002683"],                     # 民生加银前沿科技 = 民生科技
}
NAV_DIR = r"D:\iquant_data\data_v2\fund2\nav"
_AC = {}

def acc_nav(code):
    if code not in _AC:
        p = os.path.join(NAV_DIR, f"{code}.parquet")
        if not os.path.exists(p):
            _AC[code] = None; return None
        df = pd.read_parquet(p, columns=["date", "acc_nav"])
        s = pd.Series(df["acc_nav"].to_numpy(dtype=float),
                      index=pd.to_datetime(df["date"]))
        s = s[~s.index.duplicated(keep="last")].sort_index().dropna()
        s = s[(s.index >= pd.Timestamp(TRAIN_START) - pd.Timedelta(days=400)) &
              (s.index <= pd.Timestamp(TEST_END))]
        _AC[code] = s
    return _AC[code]


def pool_daily_ret(mkt, clip=0.2):
    rs = {}
    for c in POOL[mkt]:
        s = acc_nav(c)
        if s is None or len(s) < 300: continue
        r = s.pct_change().dropna().clip(-clip, clip)
        rs[c] = r
    if not rs: return pd.Series(dtype=float)
    return pd.DataFrame(rs).sort_index().mean(axis=1)


def pool_nav(mkt):
    r = pool_daily_ret(mkt)
    return (1.0 + r.fillna(0)).cumprod()


def roll_pct_rank(nav, w=756):
    """窗口 w 日滚动分位; Python loop 慢, 用 numpy 向量近似 (逐段)"""
    out = pd.Series(index=nav.index, dtype=float)
    arr = nav.to_numpy()
    n = len(arr)
    for i in range(n):
        seg = arr[max(0, i - w + 1):i + 1]
        out.iloc[i] = float((seg < seg[-1]).sum()) / len(seg)
    return out


def pe_rank_csi300(dates):
    """沪深300 PE-TTM expanding 分位, 复用 timing_dingtou 缓存"""
    df = td.fetch_pe_csi300()  # index=date, cols=close, pe_ttm
    s = df["pe_ttm"].dropna().sort_index()
    ranks = pd.Series(index=s.index, dtype=float)
    arr = s.to_numpy()
    for i in range(len(arr)):
        seg = arr[:i + 1]
        ranks.iloc[i] = float((seg < seg[-1]).sum()) / len(seg)
    return ranks.reindex(dates).ffill().bfill().fillna(0.5)


def a_weight(rank, lb, ub):
    if pd.isna(rank): return 0.5
    if rank < lb:        return 1.00
    if rank < (lb+ub)/2: return 0.50
    if rank < ub:        return 0.25
    return 0.0


def combo_stats(r):
    r = r.dropna()
    if len(r) < 60: return {"ann": np.nan, "mdd": np.nan, "vol": np.nan, "sharpe": np.nan, "total": np.nan}
    nav = (1.0 + r).cumprod()
    total = nav.iloc[-1] / nav.iloc[0] - 1
    ann = (nav.iloc[-1] / nav.iloc[0]) ** (252.0 / len(nav)) - 1
    vol = r.std() * np.sqrt(252)
    mdd = float((nav / nav.cummax() - 1).min())
    sharpe = (r.mean() * 252.0) / vol if vol > 0 else np.nan
    return {"ann": ann, "mdd": mdd, "vol": vol, "sharpe": sharpe, "total": total}


def run_backtest(mask, all_dates, ret_dict, rank_large, rank_theme, qs, w_us, w_gold, cap_a, lb, ub):
    """按季度调仓, 季末重算 rank, 重分配权重; mask: boolean numpy array aligned with all_dates."""
    dates = all_dates[mask]
    if len(dates) < 60: return None
    # 取该段内的季度调仓日
    q_in = [q for q in qs if q >= dates[0] and q <= dates[-1]]
    if not q_in: return None
    q_idx = []
    arr_d = dates.to_numpy()
    for q in q_in:
        k = int(np.searchsorted(arr_d, q.to_datetime64()))
        k = min(k, len(arr_d) - 1)
        if k > 0 and arr_d[k] > q.to_datetime64(): k -= 1
        q_idx.append(k)
    out = pd.Series(0.0, index=dates)
    n = len(dates)
    w_rec = []
    for qi, idx in enumerate(q_idx):
        s = idx
        e = q_idx[qi + 1] if qi + 1 < len(q_idx) else n
        d = dates[idx]
        rL = rank_large.reindex(dates).ffill().iloc[idx]
        rT = rank_theme.reindex(dates).ffill().iloc[idx]
        wL_ = cap_a / 2 * a_weight(rL, lb, ub) / 100.0
        wT_ = cap_a / 2 * a_weight(rT, lb, ub) / 100.0
        w_us_ = w_us / 100.0
        w_gold_ = w_gold / 100.0
        w_bond_ = max(0.0, 1.0 - wL_ - wT_ - w_us_ - w_gold_)
        w_rec.append((d, w_bond_, w_us_, w_gold_, wL_, wT_))
        sr = pd.Series(0.0, index=dates[s:e])
        for w, m in [(w_bond_, "BOND"), (w_us_, "US"), (w_gold_, "GOLD"),
                     (wL_, "A_LARGE"), (wT_, "A_THEME")]:
            if w > 0:
                sr = sr + w * ret_dict[m].reindex(dates[s:e]).fillna(0.0)
        out.iloc[s:e] = sr.to_numpy()
    return out, w_rec


def main():
    t0 = time.time()
    print("1) 构建 5 资产日收益 ...")
    rets = {m: pool_daily_ret(m) for m in POOL}
    navs = {m: pool_nav(m) for m in POOL}
    all_dates = pd.DatetimeIndex(sorted(set().union(*[r.index for r in rets.values()])))
    rets_a = {m: rets[m].reindex(all_dates).fillna(0.0) for m in POOL}
    train_mask = (all_dates >= pd.Timestamp(TRAIN_START)) & (all_dates <= pd.Timestamp(TRAIN_END))
    test_mask  = (all_dates >= pd.Timestamp(TEST_START))  & (all_dates <= pd.Timestamp(TEST_END))
    qs_all = [pd.Timestamp(d) for d in pd.date_range(TRAIN_START, TEST_END, freq="Q")]
    print(f"   train: {train_mask.sum()} 天, test: {test_mask.sum()} 天")

    print("2) A_LARGE (沪深300 PE 分位) + A_THEME (净值3年滚动分位) 信号 ...")
    rank_large = pe_rank_csi300(all_dates)
    rank_theme = roll_pct_rank(navs["A_THEME"].reindex(all_dates).ffill().bfill(), w=756)

    # 基准表现 (训练期/测试期各自)
    for name, w in [("100%BOND", {"BOND":1.0}), ("朴素 50/30/20",
            {"BOND":0.5, "US":0.3, "GOLD":0.2, "A_LARGE":0.0, "A_THEME":0.0})]:
        r = sum(w.get(m,0) * rets_a[m] for m in POOL)
        print(f"\n   基准 {name}:")
        st1 = combo_stats(r[train_mask]); st2 = combo_stats(r[test_mask])
        print(f"     训练期(18-20): 年化 {st1['ann']:.2%} 回撤 {st1['mdd']:.2%} 夏普 {st1['sharpe']:.2f}")
        print(f"     验证期(21-26): 年化 {st2['ann']:.2%} 回撤 {st2['mdd']:.2%} 夏普 {st2['sharpe']:.2f}")

    # ---- TRAIN 期网格搜索 ----
    print("\n3) 训练期 (2018-2020) 网格搜索最佳配比 & 阈值 ...")
    w_bonds = [40, 50, 60]
    w_uss   = [10, 15, 20, 25]
    w_golds = [5, 10, 15]
    cap_as  = [15, 20, 25, 30]
    lbs     = [0.20, 0.25, 0.30]
    ubs     = [0.60, 0.70, 0.80]
    grid = list(itertools.product(w_bonds, w_uss, w_golds, cap_as, lbs, ubs))
    print(f"   共 {len(grid)} 组参数")
    results = []
    t1 = time.time()
    for (wb, wu, wg, ca, lb, ub) in grid:
        # wb 实际会被 A 股档位自动调节 (仅作参考约束)
        # 先检查 wu+wg+ca <= 100 (否则跳过)
        if wu + wg + ca > 100: continue
        res = run_backtest(train_mask.to_numpy() if hasattr(train_mask, 'to_numpy') else train_mask,
                           all_dates, rets_a, rank_large, rank_theme, qs_all,
                           wu, wg, ca, lb, ub)
        if res is None: continue
        r, wrec = res
        st = combo_stats(r)
        if st["ann"] is np.nan: continue
        # 记录
        results.append((wb, wu, wg, ca, lb, ub,
                        st["ann"], st["mdd"], st["sharpe"], st["total"]))
    print(f"   搜索耗时 {time.time()-t1:.0f}s, 有效组合 {len(results)}")
    df_r = pd.DataFrame(results, columns=[
        "w_bond", "w_us", "w_gold", "cap_a", "lb", "ub",
        "t_ann", "t_mdd", "t_sharpe", "t_total"])
    # 筛选: 回撤 ≥ -8% (满足 8% 内), 取年化前 8
    feasible = df_r[df_r["t_mdd"] >= -0.08].sort_values("t_ann", ascending=False)
    print(f"\n   训练期 回撤≤8% 的组合 {len(feasible)} 个, 年化 top 12:")
    for _, r in feasible.head(12).iterrows():
        print(f"     B{r.w_bond:.0f}/U{r.w_us:.0f}/G{r.w_gold:.0f} "
              f"A_max={r.cap_a:.0f}%  阈值 lb={r.lb:.0%}/ub={r.ub:.0%}  "
              f"年化 {r.t_ann:.2%} 回撤 {r.t_mdd:.2%} 夏普 {r.t_sharpe:.2f}")
    # 无回撤达标, 就取回撤最浅 top 12
    if len(feasible) == 0:
        loosest = df_r.sort_values("t_mdd", ascending=False).head(12)
        print(f"   无回撤≤8% 组合. 回撤最浅 top 12:")
        for _, r in loosest.iterrows():
            print(f"     B{r.w_bond:.0f}/U{r.w_us:.0f}/G{r.w_gold:.0f} "
                  f"A_max={r.cap_a:.0f}%  lb={r.lb:.0%}/ub={r.ub:.0%}  "
                  f"年化 {r.t_ann:.2%} 回撤 {r.t_mdd:.2%} 夏普 {r.t_sharpe:.2f}")

    # 候选 4 组带入验证期:
    cand = []
    if len(feasible) > 0:
        for i in range(min(4, len(feasible))):
            r = feasible.iloc[i]
            cand.append((f"训练达标#{i+1} 年化最高",
                        int(r.w_bond), int(r.w_us), int(r.w_gold), int(r.cap_a),
                        float(r.lb), float(r.ub)))
    # 再追加夏普最优 & 回撤最浅 (不一定满足 8%)
    sh_best = df_r.sort_values("t_sharpe", ascending=False).iloc[0]
    cand.append(("训练夏普最优",
                int(sh_best.w_bond), int(sh_best.w_us), int(sh_best.w_gold), int(sh_best.cap_a),
                float(sh_best.lb), float(sh_best.ub)))
    mdd_best = df_r.sort_values("t_mdd", ascending=False).iloc[0]
    cand.append(("训练回撤最浅",
                int(mdd_best.w_bond), int(mdd_best.w_us), int(mdd_best.w_gold), int(mdd_best.cap_a),
                float(mdd_best.lb), float(mdd_best.ub)))

    # ---- TEST 期验证 ----
    print("\n" + "=" * 110)
    print("4) 验证期 (2021-2026) — 直接代入训练期参数, 不调整")
    print("=" * 110)
    rows = []
    for name, wb, wu, wg, ca, lb, ub in cand:
        res_t = run_backtest(test_mask.to_numpy() if hasattr(test_mask, 'to_numpy') else test_mask,
                             all_dates, rets_a, rank_large, rank_theme, qs_all,
                             wu, wg, ca, lb, ub)
        if res_t is None: continue
        r_t, wrec_t = res_t
        st = combo_stats(r_t)
        nav_t = (1.0 + r_t).cumprod()
        yr = nav_t.resample("Y").last().pct_change().dropna()
        r1y = nav_t.pct_change(252).dropna()
        rows.append((name, wb, wu, wg, ca, lb, ub, st))
        hit = "  <== 10%/8%" if st["ann"] >= 0.10 and st["mdd"] >= -0.08 else ""
        print(f"\n  [{name}]  B{wb}/U{wu}/G{wg}/A_max={ca}%  阈值 lb={lb:.0%}/ub={ub:.0%}")
        print(f"     年化 {st['ann']:.2%} 回撤 {st['mdd']:.2%} 波动 {st['vol']:.2%} 夏普 {st['sharpe']:.2f} {hit}")
        print(f"     年均权重: BOND {np.mean([x[1] for x in wrec_t]):>5.1%}  "
              f"US {np.mean([x[2] for x in wrec_t]):>5.1%}  "
              f"GOLD {np.mean([x[3] for x in wrec_t]):>5.1%}  "
              f"A_LARGE {np.mean([x[4] for x in wrec_t]):>5.1%}  "
              f"A_THEME {np.mean([x[5] for x in wrec_t]):>5.1%}")
        print(f"     逐年: " + " ".join(f"{y.year} {v:>5.1%}" for y, v in yr.items()))
        if len(r1y):
            print(f"     滚动1年: 中位 {r1y.median():.2%} 负占比 {(r1y<0).mean():.1%} "
                  f"min {r1y.min():.2%} max {r1y.max():.2%}")
    # 对比: 朴素 50/30/20
    bm = sum((0.5 if m=="BOND" else 0.3 if m=="US" else 0.2 if m=="GOLD" else 0.0)
             * rets_a[m] for m in POOL)
    st_bm = combo_stats(bm[test_mask])
    nav_bm = (1.0 + bm[test_mask]).cumprod()
    yr_bm = nav_bm.resample("Y").last().pct_change().dropna()
    r1y_bm = nav_bm.pct_change(252).dropna()
    print(f"\n  [基准 朴素 50/30/20]")
    print(f"     年化 {st_bm['ann']:.2%} 回撤 {st_bm['mdd']:.2%} 波动 {st_bm['vol']:.2%} 夏普 {st_bm['sharpe']:.2f}")
    print(f"     逐年: " + " ".join(f"{y.year} {v:>5.1%}" for y, v in yr_bm.items()))
    print(f"     滚动1年: 中位 {r1y_bm.median():.2%} 负占比 {(r1y_bm<0).mean():.1%}")

    print(f"\n总耗时 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()

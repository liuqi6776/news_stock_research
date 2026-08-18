# -*- coding: utf-8 -*-
"""
目标可行性回测: 年化 10% + 最大回撤 8% 以内
====================================================

问题: 用"基金配比 + 现有选基法则(4433动态池) + 低相关资产(海外/黄金)"能否做到
      年化 >=10% 且 最大回撤 <= 8% (2021-01 ~ 2026-08)?

资产块 (日收益, acc_nav 累计净值口径, 剔除异常, 裁剪到回测区间):
  BASE  底仓 = 纯债池 + 红利池 (一直持有)
  DYN   进攻仓 = 4433 动态选基池 (每季度等权再平衡), 叠加 4信号择时
        (低估区 n_sig>=2 持有 / 其余持币, 收益=0)
  LARGE 大盘宽基池 (择时同 DYN, 对照)
  US    纳指 + 标普500 联接 (低相关海外)
  GOLD  黄金 ETF 联接 (低相关另类)

方法:
  V_A 静态配比扫描: BASE + DYN(择时) 的权重 0~100%
  V_B 跨资产网格: BASE/DYN/US/GOLD 四块权重网格 (步长5%), 找满足
      ann>=10% & mdd<=8% 的组合, 并报告帕累托前沿
  V_C 动态配比: 按 4 信号强度(n_sig) 分档调整 DYN 权重 (低估重仓/高估轻仓)

注: 未计申赎费; QDII 节假日差异用 fillna(0); 汇率影响含在净值中。
用法:
  C:/Users/liuqi/anaconda3/python.exe studies/rotation_dingtou/target_10_8.py
"""
import os
import sys
import time

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # fund_research
sys.path.insert(0, ROOT)
import run_rotation as rr  # noqa: E402
import timing_dingtou as td  # noqa: E402
import kelly_portfolio as kp  # noqa: E402

START = "2021-01-01"
END = "2026-08-06"

# 低相关资产候选 (A类人民币份额; 脚本内做缺文件容错)
US_NDX = ["000834", "270042"]          # 大成/广发 纳斯达克100 联接
US_SPX = ["161125", "050025"]          # 易方达/博时 标普500 联接
GOLD = ["000216", "002610"]            # 华安/博时 黄金ETF联接

_acc_cache = {}


def acc_nav(code):
    if code not in _acc_cache:
        path = os.path.join(kp.NAV_DIR, f"{code}.parquet")
        if not os.path.exists(path):
            _acc_cache[code] = None
            return None
        df = pd.read_parquet(path, columns=["date", "acc_nav"])
        s = pd.Series(df["acc_nav"].to_numpy(dtype=float),
                      index=pd.to_datetime(df["date"]))
        s = s[~s.index.duplicated(keep="last")].sort_index().dropna()
        _acc_cache[code] = s
    return _acc_cache[code]


def pool_ret(codes, clip=0.2):
    """等权日收益 (缺文件/无数据基金自动跳过)"""
    out = {}
    for c in codes:
        s = acc_nav(c)
        if s is None or len(s) < 200:
            continue
        r = s.pct_change().dropna()
        r = r[(r >= -clip) & (r <= clip)]
        r = r[(r.index >= pd.Timestamp(START)) & (r.index <= pd.Timestamp(END))]
        out[c] = r
    if not out:
        return pd.Series(dtype=float)
    return pd.DataFrame(out).sort_index().mean(axis=1)


def build_dyn_ret(qdates, sel_dyn):
    """4433 动态池日收益: 季度 [q_i, q_{i+1}) 内取当季通过者等权日收益"""
    parts = []
    for i, d in enumerate(qdates):
        d_end = qdates[i + 1] if i + 1 < len(qdates) else pd.Timestamp(END)
        sel = [c for c in sel_dyn[i] if acc_nav(c) is not None]
        if not sel:
            continue
        df = pd.DataFrame({c: acc_nav(c).pct_change().clip(-0.2, 0.2) for c in sel})
        seg = df[(df.index >= d) & (df.index < d_end)].mean(axis=1)
        if not seg.empty:
            parts.append(seg)
    return pd.concat(parts).sort_index()


def hold_daily_from_nsig(sig, dates, thr=2):
    """n_sig >= thr (低估区) -> 1, 否则 0; 季度内恒定"""
    hold = pd.Series(0.0, index=dates)
    qs = sig.index
    for i, q in enumerate(qs):
        mask = (dates >= q) & (dates < (qs[i + 1] if i + 1 < len(qs)
                                        else dates.max() + pd.Timedelta(days=1)))
        hold[mask] = 1.0 if sig["n_sig"].iloc[i] >= thr else 0.0
    return hold


def stats_of(nav):
    r = nav.pct_change().dropna()
    total = nav.iloc[-1] / nav.iloc[0] - 1.0
    ann = (nav.iloc[-1] / nav.iloc[0]) ** (252.0 / len(nav)) - 1.0
    vol = r.std() * np.sqrt(252.0)
    mdd = float((nav / nav.cummax() - 1.0).min())
    sharpe = (r.mean() * 252.0) / vol if vol > 0 else np.nan
    return {"total": total, "ann": ann, "vol": vol, "mdd": mdd, "sharpe": sharpe}


def combo_stats(r):
    return stats_of((1.0 + r).cumprod())


def report(name, r):
    st = combo_stats(r)
    hit = st["ann"] >= 0.10 and st["mdd"] >= -0.08
    print(f"  {name:<46s} 年化 {st['ann']:>7.2%} 回撤 {st['mdd']:>7.2%} "
          f"波动 {st['vol']:>6.2%} 夏普 {st['sharpe']:>5.2f}{'  <== 达标!' if hit else ''}")
    return st, hit


def main():
    t0 = time.time()
    print("1) 构建资产块 ...")
    base_ret = 0.5 * pool_ret(kp.PURE_BOND) + 0.5 * pool_ret(kp.DIVIDEND)
    large_ret = pool_ret(kp.LARGE_CAP)
    us_ret = 0.5 * pool_ret(US_NDX) + 0.5 * pool_ret(US_SPX)
    gold_ret = pool_ret(GOLD)

    print("2) 计算 4433 动态池 + 4 信号 ...")
    basic = rr.load_basic()
    active = basic[basic["fund_type"].isin(rr.ACTIVE_TYPES)]
    codes = active["code"].astype(str).tolist()
    basic_ft = basic.set_index("code")["fund_type"]
    qdates = [pd.Timestamp(d) for d in pd.date_range(START, END, freq="Q")]
    rr.END = pd.Timestamp(END)
    out, union = rr.compute_window_sums(codes, qdates)
    sel_dyn = [rr.select_4433(out, codes, basic_ft, i) for i in range(len(qdates))]
    sig = td.compute_signals(qdates, sel_dyn, out, codes)
    sig = sig.set_index("date")
    print(f"  低估季度(n_sig>=2): {int((sig['n_sig'] >= 2).sum())} / {len(sig)}")

    print("3) 构建动态池日收益 (季度内等权, 不计费) ...")
    dyn_ret = build_dyn_ret(qdates, sel_dyn)

    all_dates = pd.DatetimeIndex(sorted(set(base_ret.index) | set(dyn_ret.index)
                                        | set(us_ret.index) | set(gold_ret.index)))
    hold = hold_daily_from_nsig(sig, all_dates)
    base_a = base_ret.reindex(all_dates).fillna(0.0)
    dyn_raw = dyn_ret.reindex(all_dates).fillna(0.0)
    large_raw = large_ret.reindex(all_dates).fillna(0.0)
    us_a = us_ret.reindex(all_dates).fillna(0.0)
    gold_a = gold_ret.reindex(all_dates).fillna(0.0)
    dyn_t = hold * dyn_raw
    large_t = hold * large_raw

    print("\n资产块单独表现 (2021-01 ~ 2026-08):")
    for name, s in [("BASE 债+红利底仓", base_a), ("DYN 4433择时仓", dyn_t),
                    ("LARGE 大盘择时", large_t), ("US 纳指+标普", us_a),
                    ("GOLD 黄金", gold_a)]:
        st = combo_stats(s)
        print(f"  {name:<16s} 年化 {st['ann']:>7.2%} 回撤 {st['mdd']:>7.2%} "
              f"波动 {st['vol']:>6.2%} 夏普 {st['sharpe']:>5.2f}")

    # ---- V_A 静态配比: BASE + DYN ----
    print("\nV_A BASE + DYN(择时) 静态配比:")
    for w in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0]:
        r = (1 - w) * base_a + w * dyn_t
        report(f"w_dyn={w:.0%}", r)

    # ---- V_B 跨资产网格 ----
    print("\nV_B 四资产权重网格 (BASE/DYN/US/GOLD, 步长5%) ...")
    vals = np.round(np.arange(0, 1.0 + 1e-9, 0.05), 2)
    n_comb, best, all_rows = 0, [], []
    for wb in vals:
        for wd in vals:
            if wb + wd > 1.0 + 1e-9:
                continue
            for wu in vals:
                rem = 1.0 - wb - wd - wu
                if rem < -1e-9 or round(rem, 2) not in vals:
                    continue
                n_comb += 1
                r = wb * base_a + wd * dyn_t + wu * us_a + rem * gold_a
                st = combo_stats(r)
                w = (wb, wd, wu, round(rem, 2))
                all_rows.append((st["ann"], st["mdd"], st["vol"], st["sharpe"], w))
                if st["ann"] >= 0.10 and st["mdd"] >= -0.08:
                    best.append((st["ann"], st["mdd"], w))
    print(f"  网格组合数 {n_comb}")
    if best:
        best.sort(key=lambda x: x[1], reverse=True)
        print(f"  达标组合 {len(best)} 个, 回撤最浅前 10:")
        for ann, mdd, w in best[:10]:
            print(f"    年化 {ann:.2%} 回撤 {mdd:.2%}  "
                  f"BASE {w[0]:.0%} DYN {w[1]:.0%} US {w[2]:.0%} GOLD {w[3]:.0%}")
    else:
        print("  无组合满足 年化>=10% 且 回撤<=8%")
    pf = {}
    for ann, mdd, vol, sh, w in all_rows:
        bucket = -int(round(mdd * 100, 0))
        if bucket not in pf or ann > pf[bucket][0]:
            pf[bucket] = (ann, mdd, sh, w)
    print("  帕累托前沿 (回撤档 -> 最高年化):")
    for bucket in sorted(pf):
        ann, mdd, sh, w = pf[bucket]
        print(f"    回撤~{bucket:>2d}%: 最高年化 {ann:>6.2%} (夏普 {sh:.2f})  "
              f"BASE {w[0]:.0%} DYN {w[1]:.0%} US {w[2]:.0%} GOLD {w[3]:.0%}")
    best_sh = sorted(all_rows, key=lambda x: x[3], reverse=True)
    print("  夏普 top5:")
    for ann, mdd, vol, sh, w in best_sh[:5]:
        print(f"    夏普 {sh:.2f} 年化 {ann:.2%} 回撤 {mdd:.2%}  "
              f"BASE {w[0]:.0%} DYN {w[1]:.0%} US {w[2]:.0%} GOLD {w[3]:.0%}")

    # ---- V_C 动态配比 (信号分级) ----
    print("\nV_C 按 n_sig 分档动态配比 (低估重仓/高估轻仓):")
    n0, n1, n2 = 0.10, 0.25, 0.50
    dyn_w = pd.Series(0.0, index=all_dates)
    qs = sig.index
    for i, q in enumerate(qs):
        ns = int(sig["n_sig"].iloc[i])
        w = n0 if ns == 0 else (n1 if ns == 1 else n2)
        mask = (all_dates >= q) & (all_dates < (qs[i + 1] if i + 1 < len(qs)
                                                else all_dates.max() + pd.Timedelta(days=1)))
        dyn_w[mask] = w
    report("BASE + 动态DYN (10/25/50%)", (1 - dyn_w) * base_a + dyn_w * dyn_raw)
    report("BASE + 动态DYN + US10% + GOLD10%",
           (1 - dyn_w - 0.10 - 0.10) * base_a + dyn_w * dyn_raw + 0.10 * us_a + 0.10 * gold_a)

    # ---- V_D 推荐组合稳定性: 逐年分解 + 权重扰动 ----
    print("\nV_D 推荐组合稳定性检验 (全区间优化权重 -> 分段/扰动验证):")
    recos = {
        "A 底仓50+纳指标普30+黄金20": {"BASE": 0.50, "DYN": 0.0, "US": 0.30, "GOLD": 0.20},
        "B 底仓40+4433择时10+海外30+黄金20": {"BASE": 0.40, "DYN": 0.10, "US": 0.30, "GOLD": 0.20},
        "C 底仓60+海外20+黄金20": {"BASE": 0.60, "DYN": 0.0, "US": 0.20, "GOLD": 0.20},
        "D 底仓40+海外40+黄金20": {"BASE": 0.40, "DYN": 0.0, "US": 0.40, "GOLD": 0.20},
    }
    for name, w in recos.items():
        r = w["BASE"] * base_a + w["DYN"] * dyn_t + w["US"] * us_a + w["GOLD"] * gold_a
        nav = (1.0 + r).cumprod()
        st = combo_stats(r)
        yr = nav.resample("Y").last()
        yr = yr / yr.shift(1) - 1.0
        print(f"\n  {name}: 年化 {st['ann']:.2%} 回撤 {st['mdd']:.2%} 夏普 {st['sharpe']:.2f}")
        print("    逐年收益: " + "  ".join(f"{y.year} {v:>6.1%}" for y, v in yr.iloc[1:].items()))
        r1y = nav.pct_change(252).dropna()
        print(f"    滚动1年: 中位 {r1y.median():.2%} 最小 {r1y.min():.2%} 最大 {r1y.max():.2%} "
              f"负占比 {(r1y < 0).mean():.1%}")
    # 权重扰动鲁棒性 (推荐组合 A 附近 ±5%)
    print("\n  组合 A 权重扰动 (±5%) 的年化/回撤范围:")
    w0 = recos["A 底仓50+纳指标普30+黄金20"]
    anns, mdds = [], []
    for db in [-0.05, 0.0, 0.05]:
        for du in [-0.05, 0.0, 0.05]:
            dg = -(db + du)
            if abs(dg) > 0.05 + 1e-9:
                continue
            w = {"BASE": w0["BASE"] + db, "DYN": 0.0, "US": w0["US"] + du, "GOLD": w0["GOLD"] + dg}
            st = combo_stats(w["BASE"] * base_a + w["US"] * us_a + w["GOLD"] * gold_a)
            anns.append(st["ann"]); mdds.append(st["mdd"])
    print(f"    年化 {min(anns):.2%} ~ {max(anns):.2%}  回撤 {min(mdds):.2%} ~ {max(mdds):.2%}")

    # ---- V_E 样本外验证: 训练期定权重 -> 测试期检验 ----
    print("\n" + "=" * 100)
    print("V_E 样本外验证 (in-sample 定权重 -> out-of-sample 检验)")
    print("=" * 100)
    split = pd.Timestamp("2024-01-01")
    is_mask = all_dates < split
    oos_mask = all_dates >= split
    is_dates = all_dates[is_mask]
    oos_dates = all_dates[oos_mask]

    def seg_stats(r, dates):
        """r: 日收益 Series (已对齐 all_dates); dates: 该段 DatetimeIndex"""
        rs = r.reindex(dates).fillna(0.0)
        if len(rs) < 10:
            return {"ann": np.nan, "mdd": np.nan, "sharpe": np.nan, "vol": np.nan}
        return stats_of((1.0 + rs).cumprod())

    def grid_search(dates):
        """在 dates 区间上网格搜索: 返回 (达标最优, 夏普最优) 两个权重"""
        vals = np.round(np.arange(0, 1.0 + 1e-9, 0.05), 2)
        meet, all_r = [], []
        for wb in vals:
            for wu in vals:
                wg = round(1.0 - wb - wu, 2)
                if wg < -1e-9 or wg not in vals:
                    continue
                r = wb * base_a + wu * us_a + wg * gold_a  # 不含 DYN (V_B 显示 DYN 无贡献)
                st = seg_stats(r, dates)
                w = (wb, wu, wg)
                all_r.append((st["ann"], st["mdd"], st["sharpe"], w))
                if st["ann"] >= 0.10 and st["mdd"] >= -0.08:
                    meet.append((st["ann"], st["mdd"], st["sharpe"], w))
        # 达标中夏普最高; 无达标则全局夏普最高
        best_meet = max(meet, key=lambda x: x[2]) if meet else None
        best_sharpe = max(all_r, key=lambda x: x[2]) if all_r else None
        return best_meet, best_sharpe, len(meet), len(all_r)

    print(f"\n  训练期: {is_dates[0].date()} ~ {is_dates[-1].date()} ({len(is_dates)} 交易日)")
    print(f"  测试期: {oos_dates[0].date()} ~ {oos_dates[-1].date()} ({len(oos_dates)} 交易日)")

    # 1) 训练期资产单独表现
    print("\n  [训练期 2021-2023] 资产单独表现:")
    for name, s in [("BASE", base_a), ("US", us_a), ("GOLD", gold_a)]:
        st = seg_stats(s, is_dates)
        print(f"    {name:<6s} 年化 {st['ann']:>7.2%} 回撤 {st['mdd']:>7.2%} 夏普 {st['sharpe']:>5.2f}")
    print("\n  [测试期 2024-2026] 资产单独表现:")
    for name, s in [("BASE", base_a), ("US", us_a), ("GOLD", gold_a)]:
        st = seg_stats(s, oos_dates)
        print(f"    {name:<6s} 年化 {st['ann']:>7.2%} 回撤 {st['mdd']:>7.2%} 夏普 {st['sharpe']:>5.2f}")

    # 2) 训练期网格搜索
    best_meet, best_sh, n_meet, n_total = grid_search(is_dates)
    print(f"\n  [训练期] 网格搜索 {n_total} 组合, 达标(年化>=10%&回撤<=8%) {n_meet} 个")
    if best_meet:
        print(f"    训练期达标最优: 年化 {best_meet[0]:.2%} 回撤 {best_meet[1]:.2%} 夏普 {best_meet[2]:.2f}  "
              f"BASE {best_meet[3][0]:.0%} US {best_meet[3][1]:.0%} GOLD {best_meet[3][2]:.0%}")
    if best_sh:
        print(f"    训练期夏普最优: 年化 {best_sh[0]:.2%} 回撤 {best_sh[1]:.2%} 夏普 {best_sh[2]:.2f}  "
              f"BASE {best_sh[3][0]:.0%} US {best_sh[3][1]:.0%} GOLD {best_sh[3][2]:.0%}")

    # 3) 将训练期权重代入测试期
    print("\n  [测试期] 用训练期权重检验:")
    test_cases = []
    if best_meet:
        test_cases.append(("训练期达标最优", best_meet[3]))
    if best_sh:
        test_cases.append(("训练期夏普最优", best_sh[3]))
    # 加入"朴素固定配比"对照 (不依赖训练期优化)
    test_cases.append(("朴素固定 50/30/20", (0.50, 0.30, 0.20)))
    test_cases.append(("朴素固定 60/20/20", (0.60, 0.20, 0.20)))
    for name, w in test_cases:
        r = w[0] * base_a + w[1] * us_a + w[2] * gold_a
        st_is = seg_stats(r, is_dates)
        st_oos = seg_stats(r, oos_dates)
        hit = st_oos["ann"] >= 0.10 and st_oos["mdd"] >= -0.08
        print(f"    {name:<20s} 权重 B{w[0]:.0%}/U{w[1]:.0%}/G{w[2]:.0%}  "
              f"训练 年化{st_is['ann']:>6.2%}/回撤{st_is['mdd']:>6.2%}  "
              f"测试 年化{st_oos['ann']:>6.2%}/回撤{st_oos['mdd']:>6.2%}  "
              f"{'达标' if hit else '未达标'}")

    # 4) Walk-forward: 2年训练 -> 1年测试, 滚动
    print("\n  [Walk-forward] 2年训练 -> 1年测试:")
    wf_windows = [
        (pd.Timestamp("2021-01-01"), pd.Timestamp("2022-12-31"), pd.Timestamp("2023-01-01"), pd.Timestamp("2023-12-31")),
        (pd.Timestamp("2022-01-01"), pd.Timestamp("2023-12-31"), pd.Timestamp("2024-01-01"), pd.Timestamp("2024-12-31")),
        (pd.Timestamp("2023-01-01"), pd.Timestamp("2024-12-31"), pd.Timestamp("2025-01-01"), pd.Timestamp("2026-08-06")),
    ]
    for is_s, is_e, oos_s, oos_e in wf_windows:
        is_d = all_dates[(all_dates >= is_s) & (all_dates <= is_e)]
        oos_d = all_dates[(all_dates >= oos_s) & (all_dates <= oos_e)]
        bm, bs, nm, nt = grid_search(is_d)
        w = bm[3] if bm else (bs[3] if bs else (0.5, 0.3, 0.2))
        r = w[0] * base_a + w[1] * us_a + w[2] * gold_a
        st_is = seg_stats(r, is_d)
        st_oos = seg_stats(r, oos_d)
        tag = "达标最优" if bm else ("夏普最优" if bs else "默认")
        hit = st_oos["ann"] >= 0.10 and st_oos["mdd"] >= -0.08
        print(f"    训练{is_s.year}-{is_e.year} 测试{oos_s.year}  "
              f"权重B{w[0]:.0%}/U{w[1]:.0%}/G{w[2]:.0%}({tag})  "
              f"训练{st_is['ann']:.2%}/{st_is['mdd']:.2%}  "
              f"测试{st_oos['ann']:.2%}/{st_oos['mdd']:.2%}  {'达标' if hit else '未达标'}")

    print(f"\n总用时 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()

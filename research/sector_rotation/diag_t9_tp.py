# -*- coding: utf-8 -*-
"""止盈卖出对比: 进场不变 (s123>=3 月末生效), 卖出改为: 每只 ETF 独立 10%/20%/50% 止盈
              vs 基线 T7 / MA5+20.
              输出: 1) 组合绩效对比; 2) 每档 TP 的 ETF 逐笔交易(含"永远没涨到"标记);
                    3) 持有期分布; 4) 踏空检查."""
import os
import sys
from collections import Counter

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "fund_research", "studies", "rotation_dingtou"))

from etf_optimize_backtest2 import (  # noqa: E402
    load_industry_daily, load_hv_daily, build_series, calc_stats,
    COST, HV_WEIGHTS,
)
from sector_rotation_traditional import TRADITIONAL_ETFS, build_signals4  # noqa: E402
from diag_t9_exit import run_backtest, monthly_view  # noqa: E402

# ---- 数据 ----
panel = load_industry_daily()
hv = load_hv_daily()
trad_codes = [c for _, c in TRADITIONAL_ETFS]
trad_panel = {c: s for c, s in panel.items() if c in set(trad_codes)}
ew_trad = build_series(trad_panel)

all_dates = sorted(set().union(*[set(s.index) for s in hv.values()]))
hdf = pd.DataFrame(index=all_dates)
for code, s in hv.items():
    hdf[code] = s.reindex(all_dates)
v8_daily = (hdf * pd.Series(HV_WEIGHTS)).sum(axis=1).fillna(0)

# 每只 ETF 每日净值 (用于单只追踪)
etf_nav = {code: (1 + s).cumprod() for code, s in trad_panel.items()}

# 日频日期索引
dates = ew_trad.index

# 月频信号 -> 进场日期 (月末 s123>=3 -> T+1 生效)
monthly_nav = {}
for code, s in panel.items():
    monthly_nav[code] = (1 + s).cumprod().groupby(s.index.str[:6]).last()
nav_panel = pd.DataFrame(monthly_nav).sort_index()
sig = build_signals4(list(nav_panel.index), nav_panel, trad_codes)

sig_daily = sig["s123"].reindex(pd.Index(dates.str[:6])).fillna(0).astype(int)
sig_daily.index = dates
month_end_dates = set(dates.to_series().groupby(dates.str[:6]).max().values)

entry_dates = []
for d in month_end_dates:
    if d in sig_daily.index and sig_daily.loc[d] >= 3:
        # 月末信号 -> 次日进场; 若月末是最后一天(超出索引), 则用最后一天
        idx = list(dates).index(d)
        in_d = dates[idx + 1] if idx + 1 < len(dates) else d
        entry_dates.append(in_d)
entry_dates = sorted(set(entry_dates))

N = len(trad_codes)
W_EQ = 1.0 / N  # 单只 ETF 等权权重

# ============ 工具: ETF 净值按日期取值 (缺则前后最近) ============
ETF_INDEX = {}
ETF_VALUES = {}
for code in trad_codes:
    s = etf_nav[code]
    ETF_INDEX[code] = list(s.index)
    ETF_VALUES[code] = s.values


def nav_on(code, d):
    """返回 code 在 d 日或最近的净值 (d 不存在时向后补)"""
    idx = pd.Index(ETF_INDEX[code]).get_indexer([d], method="pad")[0]
    if idx < 0:
        idx = pd.Index(ETF_INDEX[code]).get_indexer([d], method="backfill")[0]
    return float(ETF_VALUES[code][idx]), ETF_INDEX[code][idx]


def day_index_in_dates(d):
    return list(dates).index(d)


# ============ 核心: TP 止盈回测 ============
def run_tp_backtest(tp):
    positions = {}  # code -> {'w': W_EQ, 'cost': float, 'in_date': str (实际有净值的日期)}
    nav = 1.0
    prev_cash_w = 1.0
    records = []
    etf_trades = []

    for i, d in enumerate(dates):
        # ---- 进场 ----
        if d in entry_dates:
            for code in trad_codes:
                if code not in positions:
                    cost, real_in = nav_on(code, d)
                    positions[code] = {"w": W_EQ, "cost": cost, "in_date": real_in}
                    etf_trades.append({"code": code, "in_date": real_in, "cost": cost,
                                       "signal_date": d, "in_port_nav": nav})

        # ---- 止盈检查 (基于有净值的最新价格) ----
        to_sell = []
        for code, pos in positions.items():
            cur, real_d = nav_on(code, d)
            ret = cur / pos["cost"] - 1
            if ret >= tp:
                to_sell.append((code, cur, ret))
        for code, cur, ret in to_sell:
            pos = positions.pop(code)
            for t in reversed(etf_trades):
                if t["code"] == code and "out_date" not in t:
                    t["out_date"] = d
                    t["out_nav"] = cur
                    t["hold_days"] = day_index_in_dates(d) - day_index_in_dates(pos["in_date"]) + 1
                    t["ret"] = ret
                    t["reached_tp"] = True
                    t["out_port_nav"] = nav
                    break

        # ---- 当日收益 ----
        in_w = sum(p["w"] for p in positions.values())
        cash_w = 1.0 - in_w
        r_etfs = 0.0
        pre_i = max(i - 1, 0)
        for code, pos in positions.items():
            cur, _ = nav_on(code, d)
            pre, _ = nav_on(code, dates[pre_i])
            r_etfs += pos["w"] * (cur / pre - 1)
        r_cash = cash_w * float(v8_daily.get(d, 0.0))
        r = r_etfs + r_cash
        c = abs(cash_w - prev_cash_w) * COST
        nav *= (1 + r - c)
        records.append({"d": d, "nav": nav, "w": in_w, "n_held": len(positions)})
        prev_cash_w = cash_w

    # 未平仓
    last_d = dates[-1]
    for code, pos in positions.items():
        cur, _ = nav_on(code, last_d)
        ret = cur / pos["cost"] - 1
        for t in reversed(etf_trades):
            if t["code"] == code and "out_date" not in t:
                t["out_date"] = last_d
                t["out_nav"] = cur
                t["hold_days"] = day_index_in_dates(last_d) - day_index_in_dates(pos["in_date"]) + 1
                t["ret"] = ret
                t["reached_tp"] = False
                t["out_port_nav"] = nav
                t["note"] = "未达止盈(至末端)"
                break

    out = pd.DataFrame(records).set_index("d")
    out["ym"] = out.index.str[:6]
    return out, pd.DataFrame(etf_trades)


# ============ 基线 ============
base_variants = [
    ("T7 月频s123<=1出",       dict(exit_kind="s123le1")),
    ("MA5 +20日缓冲",          dict(exit_kind="ma5", min_hold=20)),
]
baseline_monthly = {}
for name, kw in base_variants:
    d = run_backtest(**kw)
    baseline_monthly[name] = monthly_view(d)

# ============ 止盈变体 ============
TP_VARIANTS = [("止盈10%", 0.10), ("止盈20%", 0.20), ("止盈50%", 0.50)]
tp_results = {}
tp_monthly = {}
for name, tp in TP_VARIANTS:
    df_daily, df_trades = run_tp_backtest(tp)
    tp_results[name] = (df_daily, df_trades)
    m = df_daily.groupby("ym").agg(nav=("nav", "last"), w=("w", "last"), n=("n_held", "last"))
    m["ret"] = m["nav"].pct_change().fillna(0)
    tp_monthly[name] = m

# ============ 对比总表 ============
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", None)

print("=" * 135)
print(f"{'方案':<18} {'FinalNAV':>8} {'CAGR':>8} {'MaxDD':>8} {'Sharpe':>7} {'Calmar':>7} {'avg_w':>6}")
print("-" * 135)
all_stats = {}
for name, m in baseline_monthly.items():
    st = calc_stats(m)
    all_stats[name] = st
    print(f"{name:<18} {st['FinalNAV']:>8.2f} {st['CAGR']:>7.2%} {st['MaxDD']:>7.2%} "
          f"{st['Sharpe']:>6.2f} {st['Calmar']:>6.2f} {st['avg_w']:>5.0%}")
for name, m in tp_monthly.items():
    st = calc_stats(m)
    all_stats[name] = st
    print(f"{name:<18} {st['FinalNAV']:>8.2f} {st['CAGR']:>7.2%} {st['MaxDD']:>7.2%} "
          f"{st['Sharpe']:>6.2f} {st['Calmar']:>6.2f} {st['avg_w']:>5.0%}")

for start_label, start_ym in [("2021-01起", "2021-01"), ("2024-01起", "2024-01")]:
    print(f"\n{start_label}:")
    for name, m in list(baseline_monthly.items()) + list(tp_monthly.items()):
        sub = m[m.index >= start_ym]
        if len(sub) < 2:
            continue
        st = calc_stats(sub)
        print(f"  {name:<16} CAGR={st['CAGR']:>7.2%} MaxDD={st['MaxDD']:>7.2%} "
              f"Calmar={st['Calmar']:>5.2f} 仓位={st['avg_w']:.0%} FinalNAV={st['FinalNAV']:.3f}")

# ============ ETF 级逐笔交易 ============
for tp_name, tp in TP_VARIANTS:
    _, trades = tp_results[tp_name]
    reached = trades[trades["reached_tp"] == True]
    missed = trades[trades["reached_tp"] == False]
    n_total = len(trades)
    n_reach = len(reached)
    n_miss = len(missed)
    print(f"\n{'=' * 135}")
    print(f"[{tp_name}] 总 {n_total} 笔 ETF 单: 成功止盈 {n_reach} 笔 ({n_reach/n_total:.1%})"
          f" | 未达止盈(至末端) {n_miss} 笔 ({n_miss/n_total:.1%})")
    if len(reached) > 0:
        hds = reached["hold_days"]
        rs = reached["ret"] * 100
        print(f"  成功止盈: 平均持仓 {hds.mean():.0f} 天 (min {hds.min()} / max {hds.max()})  "
              f"平均收益 {rs.mean():.2f}% (min {rs.min():.2f}% / max {rs.max():.2f}%)")
    if len(missed) > 0:
        hds = missed["hold_days"]
        rs = missed["ret"] * 100
        print(f"  未达止盈: 平均持仓 {hds.mean():.0f} 天 (至今未触 {tp_name})  "
              f"当前平均收益 {rs.mean():.2f}% (min {rs.min():.2f}% / max {rs.max():.2f}%)")
        # "长不回去的" = 进场至今收益为负 OR 距目标仍很远 (比如当前 < 进场价)
        bad = missed[missed["ret"] < 0]
        if len(bad) > 0:
            print(f"  至今仍浮亏: {len(bad)} 笔 (最深 {bad['ret'].min()*100:.2f}%):")
            for _, row in bad.sort_values("ret").head(12).iterrows():
                print(f"    {row['code']} {row['in_date']}进 成本{row['cost']:.4f} "
                      f"→今{row['out_nav']:.4f} ({row['ret']*100:+.2f}%) 持{row['hold_days']}天")

    # 持有期分布 (成功止盈的)
    if len(reached) > 0:
        bins = [0, 15, 30, 60, 90, 120, 180, 365, 720, 99999]
        labels = ["0-15", "16-30", "31-60", "61-90", "91-120", "121-180", "181-365", "366-720", "720+"]
        cats = pd.cut(reached["hold_days"], bins=bins, labels=labels, right=True)
        dist = cats.value_counts().sort_index()
        print(f"\n  持有期分布 (成功止盈 {len(reached)} 笔):")
        for lab, cnt in dist.items():
            if cnt > 0:
                bar = "█" * int(cnt / len(reached) * 40)
                print(f"    {lab:<10}天 {cnt:>4}笔 {cnt/len(reached):>5.0%}  {bar}")
    # 未达止盈的持有期
    if len(missed) > 0:
        bins = [0, 365, 720, 1080, 1440, 2000, 99999]
        labels = ["<1年", "1-2年", "2-3年", "3-4年", "4-5年", "5+年"]
        cats = pd.cut(missed["hold_days"], bins=bins, labels=labels, right=True)
        dist = cats.value_counts().sort_index()
        print(f"\n  持有期分布 (未达止盈 {len(missed)} 笔):")
        for lab, cnt in dist.items():
            if cnt > 0:
                bar = "█" * int(cnt / len(missed) * 40)
                print(f"    {lab:<8} {cnt:>4}笔 {cnt/len(missed):>5.0%}  {bar}")

# ============ 踏空检查 ============
print(f"\n{'=' * 135}")
print("踏空检查 (出场后6个月内传统行业等权涨幅>10% 记踏空)")
for name, m in list(baseline_monthly.items()) + list(tp_monthly.items()):
    exits = m.index[(m["w"].diff() < 0)]
    misses = 0
    for ym in exits:
        idx = list(m.index).index(ym)
        if idx + 6 < len(m):
            # 传统行业6个月收益: 用 ew_trad 月频算
            pass
    # 另一种: 直接用 ew_trad 组合月频 做 6个月 fwd
    ew_m = ew_trad.groupby(ew_trad.index.str[:6]).apply(lambda s: (1+s).prod() - 1)
    ew_m_6 = ew_m.rolling(6).apply(lambda x: (1+x).prod() - 1, raw=True).shift(-1)
    for ym in exits:
        if ym in ew_m_6.index and pd.notna(ew_m_6.loc[ym]) and ew_m_6.loc[ym] > 0.10:
            misses += 1
    print(f"  {name:<18} 出场 {len(exits)} 次, 踏空 {misses} 次")

# 保存
from etf_optimize_backtest2 import OUT_DIR  # noqa: E402
summary = {}
for name, st in all_stats.items():
    summary[name] = st
pd.DataFrame(summary).T.to_csv(
    os.path.join(OUT_DIR, "traditional_t9_tp_stats.csv"), encoding="utf-8-sig")
for tp_name, _ in TP_VARIANTS:
    _, tr = tp_results[tp_name]
    tr.to_csv(os.path.join(OUT_DIR, f"traditional_t9_{tp_name}_trades.csv"),
              index=False, encoding="utf-8-sig")
print(f"\n[saved] traditional_t9_tp_stats.csv + traditional_t9_止盈*_trades.csv")

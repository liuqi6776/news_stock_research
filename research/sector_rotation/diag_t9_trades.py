# -*- coding: utf-8 -*-
"""MA5+20日缓冲 逐笔交易明细 (与 状态机36次切换=18笔 完全一致)"""
import os
import sys
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "fund_research", "studies", "rotation_dingtou"))

from diag_t9_exit import build_entry_raw, build_exit_raw, dates, ew_trad, v8_daily, COST  # noqa: E402

entry = build_entry_raw()
exit_raw = build_exit_raw("ma5").shift(1).fillna(False)
min_hold = 20

# 手动算 nav_trad
nav_trad_ = (1 + ew_trad).cumprod()

nav = 1.0
state = "out"
prev_w = 0.0
hold_days = 0
in_port = 1.0
in_trad = 1.0
trades = []

for i, d in enumerate(dates):
    if i > 0:
        p = dates[i - 1]
        if state == "out":
            if entry.loc[p]:
                state = "in"
                hold_days = 0
                in_port = nav
                in_trad = float(nav_trad_.loc[d])
                trades.append({"进场日": d, "进场组合NAV": round(in_port, 6),
                               "进场传统NAV": round(in_trad, 6)})
        else:
            hold_days += 1
            if hold_days >= min_hold and exit_raw.loc[p]:
                state = "out"
                t = trades[-1]
                t["出场日"] = d
                t["出场组合NAV"] = round(nav, 6)
                t["出场传统NAV"] = round(float(nav_trad_.loc[d]), 6)
                t["持仓天数"] = hold_days
                t["组合收益"] = f"{nav / in_port - 1:.2%}"
                t["传统收益"] = f"{nav_trad_.loc[d] / in_trad - 1:.2%}"
                t["进出成本"] = "60bps"
    w = 1.0 if state == "in" else 0.0
    r = w * float(ew_trad.loc[d]) + (1 - w) * float(v8_daily.get(d, 0.0))
    c = abs(w - prev_w) * COST
    nav *= (1 + r - c)
    prev_w = w

if trades and trades[-1].get("出场日") is None:
    t = trades[-1]
    t["出场日"] = "持仓中"
    t["出场组合NAV"] = round(nav, 6)
    t["出场传统NAV"] = round(float(nav_trad_.iloc[-1]), 6)
    t["持仓天数"] = hold_days
    t["组合收益"] = f"{nav / t['进场组合NAV'] - 1:.2%}"
    t["传统收益"] = f"{nav_trad_.iloc[-1] / t['进场传统NAV'] - 1:.2%}"
    t["进出成本"] = "30bps进场"

df = pd.DataFrame(trades)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 220)

print("=" * 220)
print(f"MA5+20日缓冲 逐笔交易明细 (初始资金=1.00元, 单次换手成本={COST*10000:.0f}bps={COST*100:.2f}%)")
print("=" * 220)
print(df.to_string(index=False))

print("\n" + "=" * 220)
closed = [t for t in trades if t.get("出场日") != "持仓中"]
n_closed = len(closed)
wins = [t for t in closed if t["出场组合NAV"] > t["进场组合NAV"]]
losses = [t for t in closed if t["出场组合NAV"] < t["进场组合NAV"]]
print(f"总交易笔数: {len(trades)} 笔 (已闭合 {n_closed} / 持仓中 {len(trades)-n_closed})")
print(f"进出场切换: {len(trades) * 2} 次 (进 {len(trades)} + 出 {len(trades) if state == 'out' else len(trades)-1})")
print(f"盈利 {len(wins)} 笔 / 亏损 {len(losses)} 笔 → 胜率 {len(wins)/n_closed:.1%}")

def _rt(t): return (t["出场组合NAV"] / t["进场组合NAV"] - 1) * 100
if closed:
    rs = [_rt(t) for t in closed]
    hds = [t["持仓天数"] for t in closed]
    print(f"单笔收益: 均值 {sum(rs)/len(rs):.2f}% / 最大 {max(rs):.2f}% / 最小 {min(rs):.2f}%")
    print(f"持仓天数: 均值 {sum(hds)/len(hds):.0f} 天 / 最长 {max(hds)} 天 / 最短 {min(hds)} 天")

amount = 10000
print(f"\n每次进场投入 {amount} 元 (闭合笔 进出各扣 {amount*COST:.0f}元手续费):")
sum_profit = 0.0
sum_cost = 0.0
for i, t in enumerate(trades, 1):
    if t.get("出场日") != "持仓中":
        pnl = (t["出场组合NAV"] / t["进场组合NAV"] - 1) * amount
        cost = amount * 2 * COST
        sum_profit += pnl
        sum_cost += cost
        print(f"  第{i:>2}笔 {t['进场日']}->{t['出场日']}  赚/亏 {pnl:>+8.0f}元  手续费 -{cost:.0f}元  净 {pnl-cost:>+8.0f}元  持仓{t['持仓天数']:>2}天  ({t['组合收益']})")
print(f"  ─────────────────────────────────────────")
print(f"  合计: 毛收益 {sum_profit:+.0f}元  总手续费 -{sum_cost:.0f}元  净收益 {sum_profit-sum_cost:+.0f}元")
print(f"\n全期组合 NAV = {nav:.6f}  (初始1.00 → 累计 {nav-1:.2%})")

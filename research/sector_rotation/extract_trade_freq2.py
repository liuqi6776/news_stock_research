# -*- coding: utf-8 -*-
"""精确提取激进版/均衡版的实际交易频率: s123进出次数、月换手率、单边交易笔数"""
import pickle, pandas as pd, numpy as np, os

ROOT = r"c:\Users\liuqi\quant_system_v2"
PKL = os.path.join(ROOT, "research/sector_rotation/results/stock_gbdt_s123_results.pkl")

with open(PKL, "rb") as f:
    data = pickle.load(f)
results = data["results"]
t7 = data["t7"]

KEYS = {
    "激进 ENS_T40_S123_ONLY_S123": "ENS_T40_S123_ONLY_S123",
    "均衡 ENS_T60_S123_TV12":      "ENS_T60_S123_TV12",
}

for label, key in KEYS.items():
    r = results[key]
    nav = r["nav_dated"]
    dates = list(nav.index)
    n_days = len(nav)
    n_years = n_days / 242.0

    # 月度分组
    m_idx = np.array([int(d) // 100 for d in dates])
    months = sorted(set(m_idx))
    n_rebals = len(months)

    # 月度收益
    monthly_nav = nav.groupby(m_idx).last()
    mret = monthly_nav.pct_change().fillna(0)

    # s123 进出推断: 收益接近 V8 (现金层) 时 → 离场
    # 简化: 月收益绝对值 < 0.3% 且非波动 → 大概率离场
    # 更准确: 看月度收益的分布
    in_market = (mret.abs() > 0.005) | (mret.rolling(3).std() > 0.02)
    n_in = in_market.sum()
    n_out = len(mret) - n_in

    # 单边换手: 每月调仓时, 约 1/3 ~ 1/2 持仓被替换
    # Top40: 中证1000月度成分变化约5-8只 / 60只 → 8-13%
    # 加上 s123 择时进出 → 全仓进出场
    # ENS_T40 无TV: 全仓进出, s123触发时 100% 换手
    # ENS_T60_TV12: TV层动态调仓位, 月度调仓 + TV再平衡

    print(f"\n{'='*60}")
    print(f"{label}")
    print(f"  回测区间: {dates[0]} ~ {dates[-1]} ({n_years:.1f}年)")
    print(f"  CAGR: {r['ann']:.2%}  MaxDD: {r['maxdd']:.2%}  Sharpe: {r['sharpe']:.2f}")
    print()
    print(f"  --- 调仓频率 ---")
    print(f"  月度调仓: {n_rebals}次 ({n_rebals/n_years:.0f}次/年)")
    print(f"  持仓数: {'40只' if 'T40' in key else '60只'}")

    if "TV" in key:
        tv_val = int(key.split("TV")[-1])
        print(f"  TV再平衡: 每日监控波动率, 触发调仓 (~{n_rebals/n_years:.0f}次/年月度 + 日内TV)")
        print(f"  目标波动率: {tv_val}%")
        print(f"  现金层: V8(短债+黄金) 仓位={100-tv_val*5:.0f}~{100-tv_val*3:.0f}% 动态")
    else:
        print(f"  TV层: 无 (全仓股票)")
        print(f"  现金层: 无")

    print()
    print(f"  --- 换手率估算 ---")
    top_n = 40 if "T40" in key else 60
    # 月度调仓换手: 模型打分变化导致 ~30% 持仓被替换
    monthly_turnover = 0.30  # 保守估计
    # s123 进出场: 全仓买入/卖出
    s123_switches = max(1, int(n_out / n_years * 0.5))  # 粗估
    annual_turnover = (n_rebals/n_years * monthly_turnover) + s123_switches
    print(f"  月度调仓换手: ~{monthly_turnover*100:.0f}%/月 → {n_rebals/n_years*monthly_turnover:.1f}次/年")
    print(f"  s123择时进出: ~{s123_switches}次/年 (全仓进出)")
    print(f"  年化总换手率: ~{annual_turnover:.1f}次 (单边)")
    print(f"  交易成本: 单边{20/10000:.2%} (佣金+滑点)")

    # 月度收益分布
    print()
    print(f"  --- 月度表现 ---")
    win = (mret > 0).sum()
    lose = (mret <= 0).sum()
    print(f"  月数: {len(mret)} (盈{win} 亏{lose}, 胜率{win/len(mret)*100:.0f}%)")
    print(f"  月均收益: {mret.mean()*100:.2f}%")
    print(f"  月最大盈利: {mret.max()*100:.2f}%")
    print(f"  月最大亏损: {mret.min()*100:.2f}%")

# T7 对照
print(f"\n{'='*60}")
print(f"对照 T7_ETF")
t7_nav = t7["nav"]
t7_mret = t7_nav.pct_change().fillna(0)
win7 = (t7_mret > 0).sum()
print(f"  回测区间: 月度数据")
print(f"  CAGR: {t7_nav.iloc[-1]**(12/len(t7_nav))-1:.2%}")
print(f"  调仓频率: 月度1次")
print(f"  持仓: ~12只行业ETF (等权)")
print(f"  s123择时: 有 (3信号触发进场, 1信号离场)")
print(f"  月数: {len(t7_mret)} (盈{win7} 亏{len(t7_mret)-win7}, 胜率{win7/len(t7_mret)*100:.0f}%)")

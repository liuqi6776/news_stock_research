# -*- coding: utf-8 -*-
"""提取激进版/均衡版的交易频率统计"""
import pickle, pandas as pd, numpy as np, os

ROOT = r"c:\Users\liuqi\quant_system_v2"
PKL = os.path.join(ROOT, "research/sector_rotation/results/stock_gbdt_s123_results.pkl")

with open(PKL, "rb") as f:
    data = pickle.load(f)
results = data["results"]

KEYS = {
    "激进 ENS_T40_S123_ONLY_S123": "ENS_T40_S123_ONLY_S123",
    "均衡 ENS_T60_S123_TV12":      "ENS_T60_S123_TV12",
    "对照 T7_ETF":                 None,  # T7 单独处理
}

for label, key in KEYS.items():
    if key is None:
        t7 = data["t7"]
        print(f"\n{'='*60}")
        print(f"{label}")
        print(f"  T7 (ETF版), 月度调仓, 数据在 t7 dict")
        continue
    if key not in results:
        print(f"\n{label}: 未找到")
        continue
    r = results[key]
    nav = r["nav_dated"]
    dates = nav.index
    n_days = len(nav)
    n_years = n_days / 242.0

    # 从 nav 序列推断调仓: 月初 nav 跳变
    # 更精确: 看 portfolio_log (被置 None), 改用 nav 变化推断
    # 调仓频率: 每月第一个交易日
    months = sorted(set(int(d) // 100 for d in dates))
    n_rebals = len(months)

    # 月度 nav
    m_idx = [int(d) // 100 for d in nav.index]
    monthly_nav = nav.groupby(m_idx).last()

    # s123 进出统计: state_in 切换次数
    # 从 nav 趋势推断: 当月收益接近0 且持续 → 可能在场外
    # 但更直接的是看原始 log
    print(f"\n{'='*60}")
    print(f"{label}")
    print(f"  回测天数: {n_days} ({n_years:.1f}年)")
    print(f"  月度调仓次数: {n_rebals} ({n_rebals/n_years:.0f}次/年)")
    print(f"  CAGR: {r['ann']:.2%}")
    print(f"  MaxDD: {r['maxdd']:.2%}")
    print(f"  Sharpe: {r['sharpe']:.2f}")

    # s123 择时: 统计在场/离场比例
    # T40: 40只股票, T60: 60只股票
    top_n = 40 if "T40" in key else 60
    print(f"  持仓数量: {top_n}只")
    if "TV12" in key:
        print(f"  目标波动率: 12% (含V8债券+黄金现金层)")
    elif "TV" in key:
        print(f"  目标波动率: {key.split('TV')[-1][:2]}%")
    else:
        print(f"  目标波动率: 无 (全仓股票)")

    # 月度调仓换手率估算
    # 每月调仓时, 大约换掉 30-50% 的持仓 (中证1000成分月度变化)
    print(f"  调仓频率: 每月1次 (月初第一个交易日)")
    print(f"  预计月换手率: ~30-50% (Top{top_n}成分变化 + s123进出)")
    print(f"  预计年换手率: ~{n_rebals/n_years * 0.4:.0f}次全换 / 年交易天数: {n_days}天")

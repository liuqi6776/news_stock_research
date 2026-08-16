# -*- coding: utf-8 -*-
"""生成最佳策略（均衡版 ENS_T60_S123_TV12）收益曲线 + 详细回测指标。

数据源: research/sector_rotation/results/stock_gbdt_s123_results.pkl（修复后口径, 2019-06~2026-08）
对照: 进取版 ENS_T40_S123_ONLY_S123 + ETF 原版 T7
产出: results/best_strategy_curve.png + best_strategy_compare.png + best_strategy_metrics.json
"""
import json
import os
import pickle

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = r"c:\Users\liuqi\quant_system_v2"
HERE = os.path.join(ROOT, "research", "sector_rotation", "results")
PKL = os.path.join(HERE, "stock_gbdt_s123_results.pkl")
SQRT_242 = np.sqrt(242.0)

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

BEST = "ENS_T60_S123_TV12"            # 均衡版（风险调整后最优）
AGGR = "ENS_T40_S123_ONLY_S123"       # 进取版（年化最高）


def to_datetime(idx):
    return pd.to_datetime(idx.astype(str), format="%Y%m%d")


def daily_metrics(nav_dated):
    """nav_dated: 日频 NAV（int yyyymmdd index）。返回详细指标 dict。"""
    nav = nav_dated.astype(float)
    nav = nav / nav.iloc[0]
    n = len(nav)
    yrs = n / 242.0
    ret = nav.pct_change().fillna(0.0)
    cum = nav.iloc[-1] - 1.0
    ann = (1 + cum) ** (1 / yrs) - 1 if yrs > 0 else np.nan
    vol = ret.std(ddof=1) * SQRT_242
    sharpe = ret.mean() / (ret.std(ddof=1) + 1e-12) * SQRT_242
    dd = (nav - nav.cummax()) / nav.cummax()
    mdd = dd.min()
    calmar = ann / (-mdd + 1e-9)
    # 下行波动 / Sortino（标准 downside deviation = sqrt(mean(min(r,0)^2))）
    downside_daily = np.sqrt(np.mean(np.minimum(ret, 0.0) ** 2))
    sortino = ret.mean() / (downside_daily + 1e-12) * SQRT_242
    # 月频回撤
    nav_m = nav.groupby((nav_dated.index // 100).astype(str)).last()
    mdd_monthly = ((nav_m - nav_m.cummax()) / nav_m.cummax()).min()
    # 月度收益 & 胜率
    monthly_ret = nav_m.pct_change().dropna()
    win_rate = (monthly_ret > 0).mean()
    # 年度收益
    yearly = {}
    for y, g in nav.groupby((nav_dated.index // 10000).astype(str)):
        yearly[y] = float(g.iloc[-1] / g.iloc[0] - 1.0)
    return {
        "start": str(nav_dated.index[0]), "end": str(nav_dated.index[-1]),
        "n_days": int(n), "years": float(yrs),
        "final_nav": float(nav.iloc[-1]),
        "cum_return": float(cum),
        "cagr": float(ann),
        "annual_vol": float(vol),
        "sharpe": float(sharpe),
        "sortino": float(sortino),
        "maxdd_daily": float(mdd),
        "maxdd_monthly": float(mdd_monthly),
        "calmar": float(calmar),
        "monthly_win_rate": float(win_rate),
        "yearly_return": yearly,
    }


def main():
    d = pickle.load(open(PKL, "rb"))
    rs = d["results"]
    best_nav = rs[BEST]["nav_dated"].sort_index()
    aggr_nav = rs[AGGR]["nav_dated"].sort_index()
    t7 = d["t7"]["nav"]
    t7.index = [str(i) for i in t7.index]

    m_best = daily_metrics(best_nav)
    m_aggr = daily_metrics(aggr_nav)

    # T7 为月频: 简化指标（月频口径）
    t7_n = t7 / t7.iloc[0]
    t7_ret = t7_n.pct_change().fillna(0.0)
    t7_ann = (1 + (t7_n.iloc[-1] - 1)) ** (12 / (len(t7_n) - 1)) - 1
    t7_dd = ((t7_n - t7_n.cummax()) / t7_n.cummax()).min()
    t7_sharpe = t7_ret.mean() / (t7_ret.std(ddof=1) + 1e-12) * np.sqrt(12)

    # ---- 控制台详细指标 ----
    print("=" * 78)
    print("最佳策略（均衡版 ENS_T60_S123_TV12）详细回测指标")
    print("=" * 78)
    print(f"回测区间: {m_best['start']} ~ {m_best['end']}  ({m_best['n_days']} 交易日 / {m_best['years']:.1f} 年)")
    print(f"初始资金: 100.0 万 → 期末 {m_best['final_nav']*100:.1f} 万")
    print(f"累计收益: {m_best['cum_return']:+.2%}")
    print(f"年化收益 CAGR: {m_best['cagr']:+.2%}")
    print(f"年化波动率:    {m_best['annual_vol']:.2%}")
    print(f"Sharpe:        {m_best['sharpe']:.2f}")
    print(f"Sortino:       {m_best['sortino']:.2f}")
    print(f"最大回撤(日频): {m_best['maxdd_daily']:.2%}")
    print(f"最大回撤(月频): {m_best['maxdd_monthly']:.2%}")
    print(f"Calmar:        {m_best['calmar']:.2f}")
    print(f"月度胜率:      {m_best['monthly_win_rate']:.1%}")
    print("\n年度收益:")
    for y, v in m_best["yearly_return"].items():
        print(f"  {y}: {v:+8.2%}")
    print("\n--- 对照 ---")
    print(f"进取版 ENS_T40_S123: CAGR {m_aggr['cagr']:+.2%} | 日频MaxDD {m_aggr['maxdd_daily']:.2%} "
          f"| 月频MaxDD {m_aggr['maxdd_monthly']:.2%} | Sharpe {m_aggr['sharpe']:.2f}")
    print(f"ETF 原版 T7:        CAGR {t7_ann:+.2%} | MaxDD(月频) {t7_dd:.2%} | Sharpe {t7_sharpe:.2f}")

    # ---- 图 1: 最佳策略三面板（净值 + 回撤 + 年度收益） ----
    best_dt = to_datetime(best_nav.index)
    fig, axes = plt.subplots(3, 1, figsize=(13, 15), gridspec_kw={"height_ratios": [2.2, 1.2, 1.2]})
    fig.suptitle("最佳策略（均衡版 ENS_T60_S123_TV12）收益曲线 | 2019-06 ~ 2026-08",
                 fontsize=14, fontweight="bold")

    ax = axes[0]
    ax.plot(best_dt, best_nav / best_nav.iloc[0], color="crimson", lw=1.6, label="ENS_T60_S123_TV12")
    ax.plot(to_datetime(aggr_nav.index), aggr_nav / aggr_nav.iloc[0], color="steelblue", lw=1.2,
            alpha=0.8, label="进取版 ENS_T40_S123（对照）")
    ax.set_ylabel("净值（初始=1.0）")
    ax.set_title(f"净值曲线（累计 {m_best['cum_return']:+.1%}）")
    ax.legend(fontsize=10); ax.grid(alpha=0.3)

    ax = axes[1]
    dd_b = (best_nav / best_nav.cummax() - 1)
    ax.fill_between(best_dt, dd_b, 0, color="crimson", alpha=0.35)
    ax.plot(best_dt, dd_b, color="crimson", lw=0.8)
    ax.set_ylabel("回撤")
    ax.set_title(f"回撤（日频 MaxDD {m_best['maxdd_daily']:.2%} / 月频 {m_best['maxdd_monthly']:.2%}）")
    ax.grid(alpha=0.3)

    ax = axes[2]
    years = list(m_best["yearly_return"].keys())
    vals = [m_best["yearly_return"][y] for y in years]
    colors = ["#d62728" if v < 0 else "#2ca02c" for v in vals]
    ax.bar(years, vals, color=colors)
    for i, v in enumerate(vals):
        ax.text(i, v + (0.01 if v >= 0 else -0.02), f"{v:+.0%}", ha="center", fontsize=8)
    ax.axhline(0, color="black", lw=0.7)
    ax.set_title("年度收益")
    ax.grid(alpha=0.3, axis="y")

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    out1 = os.path.join(HERE, "best_strategy_curve.png")
    plt.savefig(out1, dpi=120)
    plt.close(fig)
    print(f"\n[图1] {out1}")

    # ---- 图 2: 三线对照（均衡 vs 进取 vs T7） ----
    t7_dt = pd.to_datetime(t7.index, format="%Y%m")
    fig, axes = plt.subplots(2, 1, figsize=(13, 10))
    fig.suptitle("最佳策略 vs 进取版 vs ETF 原版 T7（净值 / 回撤对照）", fontsize=14, fontweight="bold")
    ax = axes[0]
    ax.plot(best_dt, best_nav / best_nav.iloc[0], color="crimson", lw=1.8, label=f"均衡版（最佳）Sharpe {m_best['sharpe']:.2f}")
    ax.plot(to_datetime(aggr_nav.index), aggr_nav / aggr_nav.iloc[0], color="steelblue", lw=1.3,
            label=f"进取版 Sharpe {m_aggr['sharpe']:.2f}")
    ax.plot(t7_dt, t7_n, color="darkgreen", lw=1.3, label=f"T7 ETF Sharpe {t7_sharpe:.2f}")
    ax.set_ylabel("净值（初始=1.0）"); ax.legend(fontsize=10); ax.grid(alpha=0.3)

    ax = axes[1]
    ax.plot(best_dt, best_nav / best_nav.cummax() - 1, color="crimson", lw=1.2, label="均衡版")
    ax.plot(to_datetime(aggr_nav.index), aggr_nav / aggr_nav.cummax() - 1, color="steelblue", lw=1.2, label="进取版")
    ax.plot(t7_dt, t7_n / t7_n.cummax() - 1, color="darkgreen", lw=1.2, label="T7")
    ax.set_ylabel("回撤"); ax.legend(fontsize=10); ax.grid(alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    out2 = os.path.join(HERE, "best_strategy_compare.png")
    plt.savefig(out2, dpi=120)
    plt.close(fig)
    print(f"[图2] {out2}")

    # ---- 指标 JSON ----
    report = {
        "best_strategy": BEST,
        "best_metrics": m_best,
        "compare": {
            "aggr_ens_t40_s123": m_aggr,
            "t7_etf": {"cagr": float(t7_ann), "maxdd_monthly": float(t7_dd), "sharpe": float(t7_sharpe)},
        },
    }
    outj = os.path.join(HERE, "best_strategy_metrics.json")
    with open(outj, "w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2)
    print(f"[json] {outj}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

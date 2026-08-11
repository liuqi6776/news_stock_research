# -*- coding: utf-8 -*-
"""无脑全仓（行业等权）策略：买卖机制 + 逐年收益明细

规则:
  - 买什么: 每月末, 持有全部有收益数据的行业（~100个）, 等权
  - 什么时候买: 每月最后一个交易日（月度调仓）
  - 怎么买: 全部资金买入, 下月初按上月收益自然再平衡（等权重置）
  - 卖出: 不主动卖出, 每月重置等权（隐含卖出涨幅过大的+买入落后的）
  - 成本: 30bps 双边换手

输出:
  results/equal_weight_diary.csv   逐月 NAV/收益/持有行业数
  results/equal_weight_yearly.csv  逐年收益
"""
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

COST = 30 / 10000.0


def load_industry_ret():
    ret = pd.read_csv(os.path.join(OUT_DIR, "industry_ret.csv"), index_col=0).sort_index()
    ret.index = ret.index.astype(str)
    return ret


def main():
    ret_df = load_industry_ret()
    dates = list(ret_df.index)

    nav = 1.0
    records = []
    n_ind = len(ret_df.columns)

    # 上一期的实际权重（考虑月内漂移），用于计算再平衡换手成本
    prev_weights = None

    for i in range(len(dates) - 1):
        d_sig = dates[i]   # 调仓日（月末）
        d_ret = dates[i + 1]  # 持有收益日（下月末）
        ret_row = ret_df.loc[d_ret]

        # 等权目标: 所有有效行业权重 = 1/n
        valid = ret_row.dropna()
        n = len(valid)
        target = pd.Series(1.0 / n, index=valid.index)

        if prev_weights is None:
            # 首期建仓: 按目标权重买入, 成本 = 30bps 双边
            cost = COST
        else:
            # 再平衡: 上期权重经月内漂移后 → 调整回等权
            # 漂移后权重 w_i' = w_i * (1+r_i) / Σw_j(1+r_j)
            drift = prev_weights * (1 + ret_row.reindex(prev_weights.index).fillna(0))
            drift = drift / drift.sum()
            # 换手 = Σ|目标 - 漂移| / 2
            all_i = set(drift.index) | set(target.index)
            turn = sum(abs(target.get(c, 0) - drift.get(c, 0)) for c in all_i) / 2.0
            cost = turn * COST

        port_ret = (ret_row * target).sum()
        nav *= (1 + port_ret - cost)
        records.append({"date": d_ret, "nav": nav, "ret": port_ret, "cost": cost,
                        "turnover": cost / COST if cost > 0 else 0,
                        "n_hold": n, "sig_date": d_sig})
        prev_weights = target.copy()

    nav_df = pd.DataFrame(records).set_index("date")
    nav_df.to_csv(os.path.join(OUT_DIR, "equal_weight_diary.csv"))

    # 逐年收益
    nav_df["year"] = nav_df.index.str[:4]
    nav_df["month"] = nav_df.index.str[4:6]
    yearly = []
    for y, g in nav_df.groupby("year"):
        y_start = g["nav"].iloc[0] / (1 + g["ret"].iloc[0])  # 年初 NAV
        y_end = g["nav"].iloc[-1]
        y_ret = y_end / y_start - 1
        n_months = len(g)
        y_cagr = (y_end / y_start) ** (12 / n_months) - 1 if n_months >= 3 else np.nan
        yearly.append({"year": y, "年初NAV": y_start, "年末NAV": y_end,
                       "年收益": y_ret, "月数": n_months,
                       "月均收益": g["ret"].mean(), "年化(月复利)": y_cagr,
                       "年末累计NAV": y_end})
    ydf = pd.DataFrame(yearly)
    ydf.to_csv(os.path.join(OUT_DIR, "equal_weight_yearly.csv"), index=False, encoding="utf-8-sig")

    print("=" * 110)
    print("无脑全仓（行业等权）逐年收益明细")
    print("=" * 110)
    print(f"{'年份':<8}{'年初NAV':>10}{'年末NAV':>10}{'年收益':>10}{'月数':>6}{'月均收益':>10}{'年化(月复利)':>12}{'年末累计NAV':>12}")
    print("-" * 110)
    for _, r in ydf.iterrows():
        print(f"{r['year']:<8}{r['年初NAV']:>10.4f}{r['年末NAV']:>10.4f}"
              f"{r['年收益']:>9.2%}{r['月数']:>6d}{r['月均收益']:>9.2%}"
              f"{r['年化(月复利)']:>11.2%}{r['年末累计NAV']:>12.4f}")

    # 全期统计
    tot_nav = nav_df["nav"].iloc[-1]
    n_months = len(nav_df)
    years = n_months / 12
    cagr = tot_nav ** (1 / years) - 1
    maxdd = ((nav_df["nav"].cummax() - nav_df["nav"]) / nav_df["nav"].cummax()).max()
    print("-" * 110)
    print(f"全期: {dates[0]} ~ {dates[-1]} ({n_months} 个月, {years:.1f} 年)")
    print(f"最终NAV: {tot_nav:.4f}  CAGR: {cagr:.2%}  MaxDD: {maxdd:.2%}")

    # 逐年月度明细（最近2年）
    print("\n" + "=" * 110)
    print("最近 24 个月逐月明细（调仓日 → 持有月收益）")
    print("=" * 110)
    tail = nav_df.tail(24)
    print(f"{'调仓日(月末)':<14}{'收益月':<12}{'持有行业数':>10}{'当月收益':>10}{'累计NAV':>10}")
    print("-" * 60)
    for d, r in tail.iterrows():
        print(f"{r['sig_date']:<16}{d:<12}{r['n_hold']:>10d}{r['ret']:>9.2%}{r['nav']:>10.4f}")

    # 每年最后调仓日 & 领涨行业
    print("\n" + "=" * 110)
    print("每年买卖时点：每月最后一个交易日调仓（示例：每年1月调仓日）")
    print("=" * 110)
    for y in sorted(set(nav_df.index.str[:4])):
        jan = [d for d in nav_df.index if d.startswith(y + "01")]
        dec = [d for d in nav_df.index if d.startswith(y + "12")]
        print(f"  {y}: 1月调仓日={jan[0] if jan else '-'}, "
              f"12月调仓日={dec[-1] if dec else '-'}, "
              f"每年12次调仓")

    # 图: 逐年收益
    fig, axes = plt.subplots(1, 2, figsize=(16, 5.5))
    ax = axes[0]
    ax.bar(ydf["year"], ydf["年收益"] * 100, color=["#2ca02c" if v > 0 else "#d62728" for v in ydf["年收益"]])
    ax.axhline(0, color="gray", lw=0.8)
    ax.set_title("逐年收益（%）", fontsize=12)
    ax.set_ylabel("年收益 %")
    for i, r in enumerate(ydf["年收益"]):
        ax.text(i, r * 100 + 1.5, f"{r*100:.1f}%", ha="center", fontsize=9)

    ax = axes[1]
    ax.plot(range(len(nav_df)), nav_df["nav"], lw=1.8, color="#1f77b4")
    ax.fill_between(range(len(nav_df)), nav_df["nav"], 1, alpha=0.15, color="#1f77b4")
    ax.set_title("无脑全仓 NAV 曲线", fontsize=12)
    ax.set_ylabel("NAV")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    png = os.path.join(OUT_DIR, "equal_weight_detail.png")
    fig.savefig(png, dpi=150, bbox_inches="tight")
    print(f"\n[saved] {png}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

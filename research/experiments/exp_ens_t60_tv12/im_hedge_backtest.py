# -*- coding: utf-8 -*-
"""杠杆三·IM 对冲: 在当前 ENS 全市场 T40 + tiered + dd_degrade 上叠加中证1000股指期货空头对冲。

背景:
  - 历史结论"IM 空头对冲证伪"建立在常数 9.3%/年贴水成本假设上 (risk_control_bt2.py / option_tail_hedge.py)。
  - 但 measure_q1_im_basis.py 实测 2023-2026 IM 主力连续滚仓捕获仅 -0.29%/年 (2026 甚至升水 +2.14%/年)。
  - 即: 真实贴水成本远小于 9.3%, 对冲几乎免费 → 值得在当前 ENS 全市场 T40 上重新验证。

方法 (日频, T-1 敞口、T 日对冲腿, 无前视):
  - 基线组合日收益 r_p = nav_s.pct_change()
  - 做空 beta × w_{t-1} 份 IM 期货: 对冲腿 = -beta × w_prev × fut_ret_t
  - 对冲后 r_h = r_p - beta × w_prev × fut_ret_t
  - 对照口径: 用 spot_ret 代替 fut_ret (理论现货对冲, 无贴水) 以隔离贴水拖累。

数据:
  - nav_s / w_s 来自 engine.run_backtest_tiered(..., return_exposure=True), w_s = 当日收盘股票市值/总权益。
  - fut_ret / spot_ret 来自 im_basis_analysis.csv (IM 主力连续 vs 中证1000现货, 日频对齐)。
  - 对冲评估区间 = fut_ret 覆盖期 (2023-01 起), 即 GBDT 真正 OOS 段。
"""
import os
import sys

import numpy as np
import pandas as pd

ROOT = r"c:\Users\liuqi\quant_system_v2"
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "research", "experiments", "exp_ens_t60_tv12"))
sys.path.insert(0, os.path.join(ROOT, "research", "sector_rotation"))

from engine import init_shared, run_backtest_tiered  # noqa: E402

BETAS = [0.0, 0.3, 0.5, 0.7, 1.0]
SQRT_242 = np.sqrt(242.0)


def metrics(nav_s):
    nav_s = nav_s.sort_index().astype(float)
    tot = nav_s.iloc[-1] / nav_s.iloc[0] - 1.0
    yrs = len(nav_s) / 242.0
    ann = (1 + tot) ** (1 / yrs) - 1 if yrs > 0 else 0.0
    dd_s = nav_s / nav_s.cummax() - 1.0
    ret = nav_s.pct_change().fillna(0.0)
    sharpe = ret.mean() / (ret.std() + 1e-8) * SQRT_242
    nav_m = nav_s.groupby((nav_s.index // 100).astype(str)).last()
    dd_m = nav_m / nav_m.cummax() - 1.0
    return {"ann": ann, "maxdd": dd_s.min(), "maxdd_m": dd_m.min(),
            "calmar": ann / (-dd_s.min() + 1e-9), "sharpe": sharpe}


def main():
    print("[1] 加载全市场 shared + 跑基线 (tiered + dd_degrade -10%x0.5)...", flush=True)
    sh = init_shared("fullmarket")
    nav_s, monthly, w_s = run_backtest_tiered(
        sh, "ENS", "T40", tgt_vol=None, timing_mode="tiered",
        dd_degrade=-0.10, dd_degrade_scale=0.5, return_exposure=True)
    print(f"    cal_dates {nav_s.index.min()}~{nav_s.index.max()}, {len(nav_s)} 日", flush=True)

    # ---- 中证1000现货 / IM 主力连续日收益 (im_basis_analysis.csv, 升序对齐) ----
    basis = pd.read_csv(os.path.join(ROOT, "im_basis_analysis.csv"))
    basis["date"] = basis["date"].str.replace("-", "").astype(int)
    fut_ret = basis.set_index("date")["fut_ret"].dropna()
    spot_ret = basis.set_index("date")["spot_ret"].dropna()

    # ---- 对冲评估区间 = fut_ret 覆盖期 ----
    lo = fut_ret.index.min()
    nav_c = nav_s[nav_s.index >= lo]
    w_c = w_s[w_s.index >= lo]
    r_p = nav_c.pct_change().fillna(0.0)
    w_prev = w_c.shift(1).fillna(0.0)
    fut_a = fut_ret.reindex(nav_c.index).fillna(0.0)
    spot_a = spot_ret.reindex(nav_c.index).fillna(0.0)

    # 滚仓捕获复核 (应≈-0.29%/年, 验证数据方向)
    fc = (1 + fut_a).prod() - 1
    sc = (1 + spot_a).prod() - 1
    roll = (1 + fc) / (1 + sc) - 1
    yrs = len(nav_c) / 242.0
    roll_ann = (1 + roll) ** (1 / yrs) - 1
    print(f"\n[复核] 对冲区间 {lo}~{nav_c.index.max()} ({len(nav_c)}日)")
    print(f"    IM期货累计 {fc:+.2%} / 现货累计 {sc:+.2%} / 做多滚仓捕获 {roll:+.2%} (年化 {roll_ann:+.2%})", flush=True)

    hdr = f"{'口径':<20} {'beta':>5} {'CAGR':>8} {'Sharpe':>7} {'日MaxDD':>8} {'月MaxDD':>8} {'Calmar':>6} {'敞口':>6}"
    print("\n" + hdr)
    print("-" * 72)

    def show(label, beta, r, wavg):
        nh = (1 + r).cumprod()
        m = metrics(nh)
        print(f"{label:<20} {beta:>5.1f} {m['ann']:7.2%} {m['sharpe']:7.2f} "
              f"{m['maxdd']:7.2%} {m['maxdd_m']:7.2%} {m['calmar']:6.2f} {wavg:5.2f}", flush=True)
        return m, nh

    # 基线 (无对冲)
    base_m, _ = show("基线(无对冲)", 0.0, r_p, w_c.mean())

    print("-" * 72)
    # 真实期货对冲 (含贴水) + 理论现货对冲 (无贴水) 对照
    nav_fut = {}
    for beta in BETAS[1:]:
        r_fut = r_p - beta * w_prev * fut_a
        r_sp = r_p - beta * w_prev * spot_a
        show("IM期货对冲(真实)", beta, r_fut, w_c.mean())
        show("现货对冲(无贴水)", beta, r_sp, w_c.mean())
        nav_fut[beta] = (1 + r_fut).cumprod()
        print("-" * 72)

    # ---- 分年度稳健性 (真实期货对冲, 每年度累计收益 / 年度内最大回撤) ----
    def yearly(nav):
        out = {}
        for y, g in nav.groupby(nav.index // 10000):
            out[y] = (g.iloc[-1] / g.iloc[0] - 1.0, (g / g.cummax() - 1.0).min())
        return out

    yrs = sorted(set(nav_c.index // 10000))
    nav_base = (1 + r_p).cumprod()
    print(f"\n# 分年度 (年度收益 / 年度内最大回撤)")
    print(f"{'year':<6} {'基线':>18} {'β0.3':>18} {'β0.5':>18} {'β0.7':>18}")
    print("-" * 78)
    for y in yrs:
        b = yearly(nav_base).get(y, (np.nan, np.nan))
        row = f"{y:<6} {b[0]:>8.1%}/{b[1]:>8.1%}"
        for beta in [0.3, 0.5, 0.7]:
            v = yearly(nav_fut[beta]).get(y, (np.nan, np.nan))
            row += f" {v[0]:>8.1%}/{v[1]:>8.1%}"
        print(row)

    print(f"\n[完成] 基线月频MaxDD={base_m['maxdd_m']:.2%}, Sharpe={base_m['sharpe']:.2f}")


if __name__ == "__main__":
    main()

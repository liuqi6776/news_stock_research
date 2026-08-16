# -*- coding: utf-8 -*-
"""P2-r7: 冻结版 ENS_T60_TV12 的 DSR 敏感性分析（日频/周频/月频 × N_trials × 子区间 × 自相关修正）

背景: 月频 DSR(N=222)=0.30 < 0.95。本脚本回答: 换成日频收益（n 从 71 → 1746）是否显著?
并做 4 维敏感性:
  1. 频率: 日频 / 周频 / 月频（同一年化口径下观察期 sr 不同）
  2. N_trials: 222 / 600 / 1000（族规模不确定性）
  3. 子区间: 全期 / 2020+ / 2023+（真GBDT时段）/ 2024+ / 2025+
  4. 自相关修正: 日频一阶自相关 Lo(2002) 修正（var_sr × (1+2ρ1), 保守）

输出: dsr_sensitivity_report.json + dsr_sensitivity.png + 控制台
"""
import json
import os
import pickle
import sys

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pbo_cscv import deflated_sharpe  # noqa: E402

ROOT = r"c:\Users\liuqi\quant_system_v2"
HERE = os.path.dirname(os.path.abspath(__file__))
PKL = os.path.join(ROOT, "research", "sector_rotation", "results", "stock_gbdt_s123_results.pkl")
BEST = "ENS_T60_S123_TV12"
N_TRIALS = (222, 600, 1000)

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False


def resample_returns(nav, freq):
    """nav: pd.Series(index=int yyyymmdd). 返回观察期收益序列。"""
    if freq == "daily":
        return nav.pct_change().dropna()
    dt = pd.to_datetime(nav.index.astype(str), format="%Y%m%d")
    if freq == "weekly":
        w = nav.groupby(dt.to_period("W")).last()
    elif freq == "monthly":
        w = nav.groupby(dt.to_period("M")).last()
    else:
        raise ValueError(freq)
    return w.pct_change().dropna()


def sr_stats(r):
    sr = float(r.mean() / (r.std(ddof=1) + 1e-12))
    skew = float(stats.skew(r))
    kurt = float(stats.kurtosis(r, fisher=False))
    rho1 = float(r.autocorr(lag=1)) if len(r) > 5 else np.nan
    return sr, skew, kurt, rho1


def dsr_row(r, n_trials, rho_adj=False):
    n = len(r)
    sr, skew, kurt, rho1 = sr_stats(r)
    sr0 = sr
    n0 = n
    if rho_adj and np.isfinite(rho1):
        # Lo(2002) 一阶: var_sr × (1+2ρ1), 等效降低有效样本
        sr0 = sr
        n0 = n  # var 修正进 deflated_sharpe? 直接在 var_sr 乘子
    out = {f"N{nt}": deflated_sharpe(sr0, n0, nt, skew, kurt) for nt in n_trials}
    # 自相关修正版（在方差层面 ×(1+2ρ1), 手写）
    if rho_adj and np.isfinite(rho1) and np.isfinite(sr):
        var_sr = (1.0 + (kurt - 1.0) / 4.0 * sr ** 2 - skew * sr) / (n - 1.0)
        var_sr *= (1.0 + 2.0 * rho1)
        sr0 = np.sqrt(var_sr) * (1.0 - 0.5772156649) * stats.norm.ppf(1 - 1.0 / 222) \
            + np.sqrt(var_sr) * 0.5772156649 * stats.norm.ppf(1 - 1.0 / (222 * np.e))
        denom = np.sqrt(1.0 - skew * sr + (kurt - 1.0) / 4.0 * sr ** 2)
        z = (sr - sr0) * np.sqrt(n - 1.0) / denom
        out["N222_rho_adj"] = float(stats.norm.cdf(z))
    return out


def main():
    d = pickle.load(open(PKL, "rb"))
    nav = d["results"][BEST]["nav_dated"].sort_index().astype(float)
    nav = nav / nav.iloc[0]

    # ---- 1) 频率 × N_trials ----
    freq_rows = {}
    for freq in ("daily", "weekly", "monthly"):
        r = resample_returns(nav, freq)
        sr, skew, kurt, rho1 = sr_stats(r)
        dsr = dsr_row(r, N_TRIALS)
        freq_rows[freq] = {
            "n": int(len(r)), "sr": sr, "skew": skew, "kurt": kurt, "rho1": rho1,
            "dsr": {k: float(v) if np.isfinite(v) else None for k, v in dsr.items()},
        }

    # ---- 2) 子区间（日频, 从对应日期截断） ----
    sub_rows = {}
    for label, cutoff in (("full", None), ("2020+", 20200101), ("2023+", 20230101),
                          ("2024+", 20240101), ("2025+", 20250101)):
        nv = nav if cutoff is None else nav[nav.index >= cutoff]
        r = resample_returns(nv, "daily")
        sr, skew, kurt, rho1 = sr_stats(r)
        dsr = dsr_row(r, N_TRIALS)
        sub_rows[label] = {
            "n": int(len(r)), "sr": sr, "skew": skew, "kurt": kurt, "rho1": rho1,
            "dsr": {k: float(v) if np.isfinite(v) else None for k, v in dsr.items()},
        }

    # ---- 3) 日频自相关修正（全期/2023+） ----
    r_full = resample_returns(nav, "daily")
    r_2023 = resample_returns(nav[nav.index >= 20230101], "daily")
    adj_full = dsr_row(r_full, N_TRIALS, rho_adj=True)
    adj_2023 = dsr_row(r_2023, N_TRIALS, rho_adj=True)

    report = {
        "scope": "P2-r7 ENS_T60_TV12 DSR 日频敏感性 (2026-08-16)",
        "frozen": BEST,
        "method": "Bailey-López de Prado (2014), 观察期(非年化) Sharpe; 自相关修正=Lo(2002) 一阶 var_sr×(1+2ρ1)",
        "freq_n_trials": freq_rows,
        "subperiod_daily": sub_rows,
        "autocorr_adj_daily": {"full": adj_full, "2023+": adj_2023},
        "readout": (
            "日频 n=1746 但 kurt≈7-10 肥尾惩罚放大方差; 自相关修正后更保守。"
            "若所有频率/N 下 DSR 仍 <0.95 → 确认收益显著性不足是统计现实(样本期短), 支持独立 OOS 纪律。"),
    }
    with open(os.path.join(HERE, "dsr_sensitivity_report.json"), "w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2)

    # ---- 控制台 ----
    print("=" * 80)
    print(f"冻结版 {BEST}  DSR 敏感性（月频基准: N222=0.30）")
    print("=" * 80)
    print(f"{'频率':<8}{'n':>6}{'sr':>9}{'skew':>8}{'kurt':>8}{'ρ1':>8} | {'N222':>7} {'N600':>7} {'N1000':>7}")
    for freq, v in freq_rows.items():
        d = v["dsr"]
        print(f"{freq:<8}{v['n']:>6}{v['sr']:>9.3f}{v['skew']:>8.2f}{v['kurt']:>8.1f}"
              f"{v['rho1']:>8.2f} | {d['N222']:>7.3f} {d['N600']:>7.3f} {d['N1000']:>7.3f}")
    print("\n子区间（日频）:")
    for label, v in sub_rows.items():
        d = v["dsr"]
        print(f"  {label:<8} n={v['n']:>4} sr={v['sr']:.3f} skew={v['skew']:.2f} kurt={v['kurt']:.1f} "
              f"| N222={d['N222']:.3f} N600={d['N600']:.3f} N1000={d['N1000']:.3f}")
    print("\n日频自相关修正 Lo(2002):")
    for label, (rv, adj) in {"full": (r_full, adj_full), "2023+": (r_2023, adj_2023)}.items():
        print(f"  {label:<8} ρ1={sr_stats(rv)[3]:.2f} → N222_adj={adj['N222_rho_adj']:.3f}")

    # ---- 图: 频率×N_trials 线图 + 子区间条形 ----
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2))
    fig.suptitle(f"DSR 敏感性 — {BEST}（月频基准 N222=0.30）", fontsize=13, fontweight="bold")
    ax = axes[0]
    for freq, c in (("daily", "crimson"), ("weekly", "steelblue"), ("monthly", "darkgreen")):
        dsrs = [freq_rows[freq]["dsr"][f"N{nt}"] for nt in N_TRIALS]
        ax.plot(N_TRIALS, dsrs, marker="o", label=f"{freq} (n={freq_rows[freq]['n']})", color=c)
    ax.axhline(0.95, ls="--", color="gray", lw=1)
    ax.text(N_TRIALS[-1], 0.96, "0.95 阈值", fontsize=9, ha="right")
    ax.axhline(0.30, ls=":", color="crimson", lw=1)
    ax.set_xlabel("N_trials（族规模）"); ax.set_ylabel("DSR"); ax.legend(); ax.grid(alpha=0.3)

    ax = axes[1]
    labels = list(sub_rows.keys())
    vals = [sub_rows[l]["dsr"]["N222"] for l in labels]
    colors = ["#2ca02c" if v >= 0.95 else "#d62728" for v in vals]
    ax.bar(labels, vals, color=colors)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize=10)
    ax.axhline(0.95, ls="--", color="gray", lw=1)
    ax.set_ylim(0, 1.05); ax.set_ylabel("DSR (N=222)"); ax.set_title("子区间（日频收益）")
    ax.grid(alpha=0.3, axis="y")

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    out_png = os.path.join(HERE, "dsr_sensitivity.png")
    plt.savefig(out_png, dpi=120)
    plt.close(fig)
    print(f"\n[保存] dsr_sensitivity_report.json / dsr_sensitivity.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

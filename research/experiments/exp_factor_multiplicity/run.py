# -*- coding: utf-8 -*-
"""
exp_factor_multiplicity: 21 因子多重检验控制（审查 P0-3）

一条命令:
    C:\\Users\\liuqi\\anaconda3\\python.exe research/experiments/exp_factor_multiplicity/run.py

内容:
    1. 收集 21 因子的 Newey-West t（来自上游 summary_*.txt）
    2. BH-FDR / Bonferroni 多重检验校正 → 校正后仍显著的因子
    3. NW lag 敏感性: 对 IC 序列重算 lag=0/4/19 的 HAC t 值
    4. IC 序列自相关诊断 (Ljung-Box): 判断 lag=4 是否充分
    5. Deflated Sharpe Ratio (Bailey-Lopez de Prado 2014) 近似: 对 turnover_vol_20
       IC 信号在 N=21 次因子尝试后的 deflate 概率（IC 序列近似，非收益序列）

输出: multiplicity_report.json + 控制台表格
"""
import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, ROOT)

import statsmodels.api as sm  # noqa: E402
from scipy import stats  # noqa: E402
from statsmodels.stats.multitest import multipletests  # noqa: E402
from statsmodels.stats.diagnostic import acorr_ljungbox  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(ROOT, "research", "factor_dic", "results")

# 21 因子（与 factor_dic_validation.md 六区一致，按文件 glob 自动收集）
# summary_*.txt 全为因子验证输出（results/ 目录下）

EULER = 0.5772156649  # Euler-Mascheroni
N_TRIALS = 21         # 因子筛选尝试次数（本轮只统计因子族；组合路径/风控规则另计）


def load_summaries():
    """读取全部 summary_*.txt → {factor: {ic_n, ic_mean, icir, nw_t, ...}}"""
    out = {}
    for f in sorted(os.listdir(RES)):
        if not f.startswith("summary_") or not f.endswith(".txt"):
            continue
        factor = f[len("summary_"):-len(".txt")]
        d = {}
        with open(os.path.join(RES, f), encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if "=" in line:
                    k, v = line.split("=", 1)
                    try:
                        d[k] = float(v)
                    except ValueError:
                        d[k] = v
        if "nw_t" in d:
            out[factor] = d
    return out


def load_ic(factor):
    """读 ic_<factor>.csv → 月度 IC 序列 (pd.Series)"""
    path = os.path.join(RES, f"ic_{factor}.csv")
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path, index_col=0)
    return df.iloc[:, 0].dropna().astype(float)


def hac_t(y, lag):
    """IC 序列对截距回归, HAC 标准误 → t 值"""
    n = len(y)
    if n < 30:
        return np.nan
    X = np.ones((n, 1))
    res = sm.OLS(np.asarray(y, float), X).fit(
        cov_type="HAC", cov_kwds={"maxlags": lag})
    return float(res.tvalues[0])


def p_from_t(t, n):
    """t → 双边 p（正态近似）"""
    return 2.0 * stats.norm.sf(abs(t))


def deflated_sharpe(sr_hat, n, n_trials, skew, kurt):
    """Bailey-Lopez de Prado (2014) DSR。
    sr_hat 用 IC 序列的 mean/std 近似（非收益 Sharpe，结果仅作量级参考）。
    """
    if not np.isfinite(sr_hat) or n < 30:
        return np.nan
    g3 = skew
    g4 = kurt
    var_sr = (1.0 + (g4 - 1.0) / 4.0 * sr_hat ** 2 - g3 * sr_hat) / (n - 1.0)
    sr0 = np.sqrt(var_sr) * (
        (1.0 - EULER) * stats.norm.ppf(1.0 - 1.0 / n_trials)
        + EULER * stats.norm.ppf(1.0 - 1.0 / (n_trials * np.e)))
    denom = np.sqrt(1.0 - g3 * sr_hat + (g4 - 1.0) / 4.0 * sr_hat ** 2)
    z = (sr_hat - sr0) * np.sqrt(n - 1.0) / denom
    return float(stats.norm.cdf(z))


def main():
    summaries = load_summaries()
    assert len(summaries) == 21, f"预期 21 因子, 实际 {len(summaries)}: {sorted(summaries)}"

    # 1) NW t → p → 多重检验
    factors = sorted(summaries)
    t_vals = {f: summaries[f]["nw_t"] for f in factors}
    p_vals = np.array([p_from_t(t_vals[f], summaries[f].get("ic_n", 60)) for f in factors])
    reject_bh, q_bh, _, _ = multipletests(p_vals, alpha=0.05, method="fdr_bh")
    reject_bf, p_bf, _, _ = multipletests(p_vals, alpha=0.05, method="bonferroni")

    # 2) NW lag 敏感性（对 IC 序列重算 HAC t）
    lag_sens = {}
    for f in factors:
        ic = load_ic(f)
        if ic is None:
            continue
        lag_sens[f] = {
            "n": int(len(ic)),
            "t_lag0": hac_t(ic, 0),
            "t_lag4": hac_t(ic, 4),
            "t_lag19": hac_t(ic, 19),
        }

    # 3) IC 自相关诊断（lag=4 NW 是否充分）
    lb_diag = {}
    for f in factors:
        ic = load_ic(f)
        if ic is None or len(ic) < 20:
            continue
        lb = acorr_ljungbox(ic, lags=[4, 8, 12], return_df=True)
        lb_diag[f] = {
            "lb_p4": float(lb["lb_pvalue"].iloc[0]),
            "lb_p8": float(lb["lb_pvalue"].iloc[1]),
            "lb_p12": float(lb["lb_pvalue"].iloc[2]),
        }

    # 4) DSR（对 turnover_vol_20 IC 序列近似）
    dsr = {}
    for f in ("turnover_vol_20",):
        ic = load_ic(f)
        if ic is None:
            continue
        sr_hat = float(ic.mean() / ic.std(ddof=1))
        dsr[f] = {
            "ic_sr_approx": sr_hat,
            "skew": float(stats.skew(ic)),
            "kurt": float(stats.kurtosis(ic, fisher=False)),
            "dsr_n21": deflated_sharpe(sr_hat, len(ic), N_TRIALS,
                                       float(stats.skew(ic)),
                                       float(stats.kurtosis(ic, fisher=False))),
        }

    report = {
        "n_factors": len(factors),
        "n_trials_assumed": N_TRIALS,
        "factors": {},
    }
    for f in factors:
        report["factors"][f] = {
            "nw_t": t_vals[f],
            "p_raw": float(p_vals[factors.index(f)]),
            "q_bh": float(q_bh[factors.index(f)]),
            "reject_bh_005": bool(reject_bh[factors.index(f)]),
            "p_bonf": float(p_bf[factors.index(f)]),
            "reject_bonf_005": bool(reject_bf[factors.index(f)]),
            "lag_sens": lag_sens.get(f, {}),
            "lb_diag": lb_diag.get(f, {}),
        }
    report["dsr_turnover_vol_20"] = dsr.get("turnover_vol_20", {})

    with open(os.path.join(HERE, "multiplicity_report.json"), "w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2)

    # ---- 控制台输出 ----
    print(f"== 21 因子多重检验 (N trials = {N_TRIALS}, alpha=0.05) ==")
    print(f"{'因子':<22}{'NWt':>7}{'p_raw':>9}{'q_BH':>9}{'BH✗':>5}{'p_bonf':>9}{'BF✗':>5}"
          f"{'t0':>7}{'t4':>7}{'t19':>7}")
    n_bh = n_bf = 0
    for f in factors:
        r = report["factors"][f]
        ls = r["lag_sens"]
        t0 = f"{ls['t_lag0']:.2f}" if ls else "-"
        t4 = f"{ls['t_lag4']:.2f}" if ls else "-"
        t19 = f"{ls['t_lag19']:.2f}" if ls else "-"
        print(f"{f:<22}{r['nw_t']:>7.2f}{r['p_raw']:>9.2e}{r['q_bh']:>9.2e}"
              f"{'*' if r['reject_bh_005'] else '':>5}{r['p_bonf']:>9.2e}"
              f"{'*' if r['reject_bonf_005'] else '':>5}{t0:>7}{t4:>7}{t19:>7}")
        n_bh += int(r["reject_bh_005"])
        n_bf += int(r["reject_bonf_005"])
    print(f"\nBH-FDR 显著: {n_bh}/{len(factors)} | Bonferroni 显著: {n_bf}/{len(factors)}")

    print("\n== NW lag 敏感性 (t_lag0/t4/t19) 与 IC 自相关 (LB p 值) ==")
    print(f"{'因子':<22}{'t0→t4→t19':>20}{'LB4':>8}{'LB8':>8}{'LB12':>8}")
    for f in factors:
        r = report["factors"][f]
        ls = r["lag_sens"]
        if not ls:
            continue
        lb = r["lb_diag"]
        print(f"{f:<22}{ls['t_lag0']:>7.2f}→{ls['t_lag4']:>6.2f}→{ls['t_lag19']:>6.2f}"
              f"{lb['lb_p4']:>8.3f}{lb['lb_p8']:>8.3f}{lb['lb_p12']:>8.3f}")

    d = report["dsr_turnover_vol_20"]
    if d:
        print(f"\n== DSR 近似 (Bailey-Lopez de Prado 2014, N=21 trials) ==")
        print(f"turnover_vol_20: IC-SR≈{d['ic_sr_approx']:.3f} skew={d['skew']:.2f} "
              f"kurt={d['kurt']:.2f} → DSR={d['dsr_n21']:.3f} "
              f"({'通过' if d['dsr_n21'] >= 0.95 else '不通过'})")

    print(f"\n[保存] {os.path.join(HERE, 'multiplicity_report.json')}")


if __name__ == "__main__":
    main()

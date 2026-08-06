# -*- coding: utf-8 -*-
"""
P2-r4: turnover_vol_20 策略收益序列正式 DSR / PBO（审查 P2「继续研究」第一项）

对「实际策略月度净收益序列」而非 IC 序列做多重检验与过拟合概率量化:

  1. 收益矩阵: 读取 research/factor_dic/results/returns_*.csv
     （run_validation.run_fast 新增持久化的 Top50 月度净收益, 21 因子对齐成 T×21 矩阵）
  2. DSR (Bailey-López de Prado 2014): 对 turnover_vol_20 真实策略月收益
     的 Sharpe 做 deflate（含偏度/峰度 + 尝试次数 N）, 并给出基准相对（超额）版本;
     N 做 21/100/600 三档敏感性（600≈原始字典规模, 反映完整选择过程自由度未知）
  3. PBO (Bailey et al. 2017 CSCV): 21 因子作为一次因子族试错的策略空间,
     组合对称交叉验证 (随机 1000 组半样本切分, seed=42), 得 P(λ≤0) 过拟合概率

输出: dsr_pbo_report.json + 控制台
退出码: 0=完成（本步为研究输出, 不设通过/失败门槛; 显著性判断以 pbo<0.50 / DSR>0.95 为参考）
"""
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(ROOT, "research", "factor_dic", "results")

EULER = 0.5772156649
N_TRIALS_BASE = 21      # 因子族试错次数（与 exp_factor_multiplicity 一致）
N_TRIALS_SENS = (21, 100, 600)


def deflated_sharpe(sr_hat, n, n_trials, skew, kurt):
    """Bailey-López de Prado (2014) DSR（按观察期 Sharpe, 不年化）。"""
    if not np.isfinite(sr_hat) or n < 30:
        return np.nan
    var_sr = (1.0 + (kurt - 1.0) / 4.0 * sr_hat ** 2 - skew * sr_hat) / (n - 1.0)
    sr0 = np.sqrt(var_sr) * (
        (1.0 - EULER) * stats.norm.ppf(1.0 - 1.0 / n_trials)
        + EULER * stats.norm.ppf(1.0 - 1.0 / (n_trials * np.e)))
    denom = np.sqrt(1.0 - skew * sr_hat + (kurt - 1.0) / 4.0 * sr_hat ** 2)
    z = (sr_hat - sr0) * np.sqrt(n - 1.0) / denom
    return float(stats.norm.cdf(z))


def load_returns_matrix():
    """returns_*.csv → DataFrame(T×M); 排除 returns_bench.csv。"""
    out = {}
    for f in sorted(os.listdir(RES)):
        if not f.startswith("returns_") or f.endswith(".csv") is False:
            continue
        if f == "returns_bench.csv":
            continue
        factor = f[len("returns_"):-len(".csv")]
        df = pd.read_csv(os.path.join(RES, f), index_col=0)
        s = df.iloc[:, 0].dropna().astype(float)
        if len(s) >= 30:
            out[factor] = s
    mat = pd.DataFrame(out).sort_index()
    return mat, sorted(out)


def load_bench():
    fp = os.path.join(RES, "returns_bench.csv")
    if not os.path.exists(fp):
        return None
    df = pd.read_csv(fp, index_col=0)
    return df.iloc[:, 0].dropna().astype(float)


def cscv_pbo(returns_matrix, n_splits=1000, seed=42):
    """组合对称交叉验证: 过拟合概率 PBO = P(λ ≤ 0)。

    λ_s = log( r̄_OOS / (M - r̄_OOS) ), r̄_OOS 为 IS 上半区策略在 OOS 的平均排名。
    """
    T, M = returns_matrix.shape
    rng = np.random.default_rng(seed)
    vals = returns_matrix.values.astype(float)
    half = T // 2
    all_idx = np.arange(T)

    def sr_sub(idx):
        sub = vals[idx]
        mu = np.nanmean(sub, axis=0)
        sd = np.nanstd(sub, axis=0, ddof=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(sd > 0, mu / sd, np.nan)

    lambdas = []
    for _ in range(n_splits):
        idx = rng.choice(all_idx, size=half, replace=False)
        is_sr = sr_sub(idx)
        oos_sr = sr_sub(np.setdiff1d(all_idx, idx))
        valid = np.isfinite(is_sr) & np.isfinite(oos_sr)
        if valid.sum() < 6:
            continue
        m_valid = int(valid.sum())
        is_rank = stats.rankdata(is_sr[valid])     # 1..m_valid
        oos_rank = stats.rankdata(oos_sr[valid])   # 1..m_valid
        top_half = is_rank > m_valid / 2.0         # IS 上半区策略
        r_bar = oos_rank[top_half].mean()
        if not (0 < r_bar < m_valid):
            continue
        lambdas.append(np.log(r_bar / (m_valid - r_bar)))
    lambdas = np.asarray(lambdas)
    pbo = float((lambdas <= 0).mean()) if len(lambdas) else np.nan
    return {"n_splits_used": int(len(lambdas)), "lambda_mean": float(lambdas.mean()) if len(lambdas) else np.nan,
            "pbo": pbo}


def main():
    mat, factors = load_returns_matrix()
    bench = load_bench()
    print(f"[load] 收益矩阵: {mat.shape[0]} 个月 × {mat.shape[1]} 个因子; 因子: {factors}")

    tv20 = mat["turnover_vol_20"]
    n = len(tv20)
    sr_hat = float(tv20.mean() / tv20.std(ddof=1))
    skew = float(stats.skew(tv20))
    kurt = float(stats.kurtosis(tv20, fisher=False))

    dsr_out = {"n_month": int(n), "sr_monthly": sr_hat, "skew": skew, "kurt": kurt,
               "dsr_by_n_trials": {}}
    for nt in N_TRIALS_SENS:
        dsr_out["dsr_by_n_trials"][str(nt)] = deflated_sharpe(sr_hat, n, nt, skew, kurt)

    # 基准相对（超额）版本: 策略 - 基准（对齐月份）
    dsr_excess = None
    if bench is not None:
        exc = (tv20 - bench.reindex(tv20.index)).dropna()
        if len(exc) >= 30:
            sr_e = float(exc.mean() / exc.std(ddof=1))
            dsr_excess = {
                "n_month": int(len(exc)), "excess_sr_monthly": sr_e,
                "skew": float(stats.skew(exc)), "kurt": float(stats.kurtosis(exc, fisher=False)),
                "dsr_by_n_trials": {str(nt): deflated_sharpe(sr_e, len(exc), nt,
                                    float(stats.skew(exc)), float(stats.kurtosis(exc, fisher=False)))
                                    for nt in N_TRIALS_SENS},
            }

    # 21 因子 IS Sharpe 排序（上下文: turnover_vol_20 在族内的排名）
    sr_all = (mat.mean() / mat.std(ddof=1)).sort_values(ascending=False)
    rank_tv20 = int(sr_all.rank(ascending=False)["turnover_vol_20"])

    pbo_out = cscv_pbo(mat)

    bench_sr = None
    if bench is not None:
        b = bench.reindex(mat.index).dropna()
        bench_sr = float(b.mean() / b.std(ddof=1))

    report = {
        "scope": "P2-r4 strategy-return DSR/PBO (2026-08-06)",
        "data_source": "research/factor_dic/results/returns_*.csv (run_validation.run_fast Top50 月度净收益)",
        "n_factors": int(mat.shape[1]),
        "n_month": int(mat.shape[0]),
        "turnover_vol_20": dsr_out,
        "turnover_vol_20_excess_vs_bench": dsr_excess,
        "family_rank_by_monthly_sr": {f: float(v) for f, v in sr_all.items()},
        "turnover_vol_20_rank_in_family": rank_tv20,
        "bench_monthly_sr": bench_sr,
        "pbo_cscv": pbo_out,
        "interpretation": ("PBO<0.50: 族内最佳选择无明显过拟合; DSR≥0.95: 收益序列显著性经 deflate 后仍稳健。"
                           "两者均为策略层证据, 仍不替代冻结后独立 OOS。"),
    }
    with open(os.path.join(HERE, "dsr_pbo_report.json"), "w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2)

    # ---- 控制台 ----
    print(f"\n== turnover_vol_20 策略收益序列 (Top60 月频, 20bps, {n} 个月) ==")
    print(f"月 Sharpe={sr_hat:.3f}  skew={skew:.2f}  kurt={kurt:.2f}")
    for nt, d in dsr_out["dsr_by_n_trials"].items():
        print(f"  DSR(N={nt:>3}): {d:.3f} {'✓≥0.95' if d >= 0.95 else '✗<0.95'}")
    if dsr_excess:
        print(f"超额(相对基准) 月 Sharpe={dsr_excess['excess_sr_monthly']:.3f}")
        for nt, d in dsr_excess["dsr_by_n_trials"].items():
            print(f"  超额 DSR(N={nt:>3}): {d:.3f} {'✓≥0.95' if d >= 0.95 else '✗<0.95'}")
    if bench_sr is not None:
        print(f"基准月 Sharpe={bench_sr:.3f}")

    print(f"\n== 21 因子族内月度 Sharpe 排序 ==")
    for f, v in sr_all.items():
        print(f"  {f:<24}{v:>8.3f}")
    print(f"turnover_vol_20 族内排名: {rank_tv20}/{len(sr_all)}")

    print(f"\n== PBO (CSCV, 21 策略 × {mat.shape[0]} 月, 1000 随机半样本切分) ==")
    p = pbo_out["pbo"]
    print(f"P(λ≤0) = {p:.3f}  使用切分数: {pbo_out['n_splits_used']}  λ均值={pbo_out['lambda_mean']:.3f}")
    if p < 0.50:
        print("  → 族内最优选择无明显过拟合证据（PBO<0.50）")
    else:
        print("  → 存在过拟合信号（PBO≥0.50）: 族内选优结果在 OOS 上不优于中位水平")

    print(f"\n[保存] {os.path.join(HERE, 'dsr_pbo_report.json')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

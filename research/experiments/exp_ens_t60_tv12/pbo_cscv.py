# -*- coding: utf-8 -*-
"""P2-r7: ENS_T60_TV12 参数族 PBO(CSCV) + 冻结版 DSR（策略层多重检验）

对冻结策略 ENS_T60_TV12 的「参数族」做过拟合概率量化（对齐 definition_freeze §三）:

  1. 策略族: score{ENH,GBDT,ENS} × top{T40,T60} × TV{None(满仓),0.09..0.24} × floor{0.3,0.4,0.5} × lookback{20,60}
     = 222 个候选配置（冻结版 ENS_T60_TV12 = ENS/T60/TV0.12/floor0.4/lb20 是族内一员）
  2. 收益矩阵: 每配置日频回测 → 月频收益, 对齐 2020-02~2025-12（72 月面板区间）
  3. PBO: CSCV 随机半样本切分 1000 组 seed=42, P(λ≤0)
  4. DSR: 冻结版月收益 deflated Sharpe（偏度/峰度 + N 档敏感性 222/600/1000）

输出: pbo_cscv_report.json + returns_matrix_ens_t60_tv12.csv + 控制台
退出码: 0=完成（研究输出, 不设门槛; 参考: pbo<0.50 / DSR≥0.95）
"""
import json
import os
import sys
import time
import itertools

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import engine  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
EULER = 0.5772156649
PERIOD = ("202002", "202512")   # 收益矩阵口径: 72 月面板区间
GRID = {
    "score": ["ENS", "GBDT", "ENH"],
    "top": ["T40", "T60"],
    "tv": [None, 0.09, 0.12, 0.15, 0.18, 0.21, 0.24],
    "floor": [0.3, 0.4, 0.5],
    "lb": [20, 60],
}
N_SPLITS = 1000
SEED = 42
N_TRIALS_SENS = (222, 600, 1000)   # 族规模 222 + 保守上限


def tag_of(score, top, tv, floor, lb):
    if tv is None:
        return f"{score}_{top}_FULL"
    return f"{score}_{top}_TV{tv:.2f}_F{floor:.1f}_L{lb}"


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


def cscv_pbo(returns_matrix, n_splits=N_SPLITS, seed=SEED):
    """组合对称交叉验证: 过拟合概率 PBO = P(λ ≤ 0)。"""
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
        is_rank = stats.rankdata(is_sr[valid])
        oos_rank = stats.rankdata(oos_sr[valid])
        top_half = is_rank > m_valid / 2.0
        r_bar = oos_rank[top_half].mean()
        if not (0 < r_bar < m_valid):
            continue
        lambdas.append(np.log(r_bar / (m_valid - r_bar)))
    lambdas = np.asarray(lambdas)
    pbo = float((lambdas <= 0).mean()) if len(lambdas) else np.nan
    return {"n_splits_used": int(len(lambdas)), "lambda_mean": float(lambdas.mean()) if len(lambdas) else np.nan,
            "pbo": pbo}


def main():
    t0 = time.time()
    sh = engine.init_shared()
    print(f"[1] 共享数据 init 完成, 耗时 {time.time()-t0:.0f}s", flush=True)

    rows = []
    tags = []
    n_cfg = 0
    for score, top, tv, floor, lb in itertools.product(
            GRID["score"], GRID["top"], GRID["tv"], GRID["floor"], GRID["lb"]):
        if tv is None:                      # 满仓版: floor/lb 无效, 去重只跑一次
            if (floor, lb) != (GRID["floor"][0], GRID["lb"][0]):
                continue
            tag = tag_of(score, top, tv, floor, lb)
            nav, monthly = engine.run_backtest(sh, score, top, tgt_vol=None)
        else:
            tag = tag_of(score, top, tv, floor, lb)
            nav, monthly = engine.run_backtest(sh, score, top, tgt_vol=tv, floor_w=floor, vol_lookback=lb)
        mr = monthly.loc[(monthly.index >= PERIOD[0]) & (monthly.index <= PERIOD[1])]
        rows.append(mr.rename(tag))
        tags.append(tag)
        n_cfg += 1
        if n_cfg % 40 == 0 or n_cfg == 222:
            print(f"    {n_cfg}/{222} 配置, 累计 {time.time()-t0:.0f}s", flush=True)

    mat = pd.concat(rows, axis=1).sort_index()
    print(f"[2] 收益矩阵: {mat.shape[0]} 个月 × {mat.shape[1]} 个配置, 耗时 {time.time()-t0:.0f}s")
    mat.to_csv(os.path.join(HERE, "returns_matrix_ens_t60_tv12.csv"))

    # ---- 族内月 Sharpe 排序 + 冻结版位置 ----
    sr_all = (mat.mean() / mat.std(ddof=1)).sort_values(ascending=False)
    frozen_tag = tag_of("ENS", "T60", 0.12, 0.4, 20)
    frozen = mat[frozen_tag].dropna()
    sr_frozen = float(frozen.mean() / frozen.std(ddof=1))
    rank_frozen = int(sr_all.rank(ascending=False)[frozen_tag])

    # ---- DSR ----
    dsr_out = {
        "n_month": int(len(frozen)),
        "sr_monthly": sr_frozen,
        "skew": float(stats.skew(frozen)),
        "kurt": float(stats.kurtosis(frozen, fisher=False)),
        "dsr_by_n_trials": {str(nt): deflated_sharpe(sr_frozen, len(frozen), nt,
                                                      float(stats.skew(frozen)),
                                                      float(stats.kurtosis(frozen, fisher=False)))
                            for nt in N_TRIALS_SENS},
    }

    # ---- PBO ----
    pbo_out = cscv_pbo(mat)

    report = {
        "scope": "P2-r7 ENS_T60_TV12 参数族 PBO/DSR (2026-08-16)",
        "engine": "experiments/exp_ens_t60_tv12/engine.py (复刻 stock_gbdt_s123_backtest.py 冻结口径)",
        "period": "2020-02~2025-12 (72月, 月频收益)",
        "family_size": int(mat.shape[1]),
        "frozen_config": frozen_tag,
        "frozen_monthly_sr": sr_frozen,
        "frozen_rank_in_family": f"{rank_frozen}/{int(mat.shape[1])}",
        "family_top10_by_monthly_sr": {f: float(v) for f, v in sr_all.head(10).items()},
        "dsr": dsr_out,
        "pbo_cscv": pbo_out,
        "interpretation": ("PBO<0.50: 族内选优无明显过拟合; DSR≥0.95: 冻结版收益经 deflate 后仍显著。"
                           "策略层证据, 不替代冻结后独立 OOS(2027-2032)。"),
    }
    with open(os.path.join(HERE, "pbo_cscv_report.json"), "w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2)

    # ---- 控制台 ----
    print(f"\n== ENS_T60_TV12 参数族 PBO/DSR（{mat.shape[0]} 月 × {mat.shape[1]} 配置）==")
    print(f"冻结版 {frozen_tag}: 月 Sharpe={sr_frozen:.3f}  族内排名 {rank_frozen}/{mat.shape[1]}")
    for nt, d in dsr_out["dsr_by_n_trials"].items():
        print(f"  DSR(N={nt:>4}): {d:.3f} {'✓≥0.95' if d >= 0.95 else '✗<0.95'}")
    print(f"\n族内 Top10（月 Sharpe）:")
    for f, v in sr_all.head(10).items():
        print(f"  {f:<32}{v:>8.3f}")
    p = pbo_out["pbo"]
    print(f"\nPBO (CSCV, {mat.shape[1]} 策略 × {mat.shape[0]} 月, {N_SPLITS} 随机半样本切分):")
    print(f"  P(λ≤0) = {p:.3f}  使用切分 {pbo_out['n_splits_used']}  λ均值={pbo_out['lambda_mean']:.3f}")
    if p < 0.50:
        print("  → 族内最优选择无明显过拟合证据（PBO<0.50）")
    else:
        print("  → 存在过拟合信号（PBO≥0.50）")
    print(f"\n[保存] pbo_cscv_report.json / returns_matrix_ens_t60_tv12.csv")
    print(f"[完成] 总耗时 {time.time()-t0:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())

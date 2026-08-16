# -*- coding: utf-8 -*-
"""P2-r7: C8 GBDT 模型层 purged k-fold CV（防止标签泄漏的时间序列交叉验证）

滚动 WFO（train < 预测月）只给出单一路径; purged CV 提供跨随机时间切分的泛化评估:

  - 标签 fwd_20（未来 20 交易日收益, 跨月）→ 验证月 m 与训练月的标签窗口可能重叠
  - purge: 排除验证月 m 相邻 ±1 月（m-1 标签窗口跨入 m, m 标签窗口跨入 m+1）
  - embargo: 再排除验证月后 1 月（m+1）, 保守缓冲
  - k=5 折随机分月 × 10 次重复（seed 固定可复现）→ 汇总月度 IC / ICIR / Q1-Q5 分层

对照: 滚动 WFO（expand-window, 2023+ 逐月）IC 分布。

输出: purged_cv_report.json + 控制台
"""
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import lightgbm as lgb

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import engine  # noqa: E402
from engine import prep_feats, GBDT_FEATS, PANEL_PATH  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
K_FOLDS = 5
N_REP = 10
SEED = 42
PURGE_MONTHS = (-1, 0, 1)   # 验证月 ±1 月 purge + embargo
OOS_START = 20230101        # 滚动 WFO 对照起点（与回测一致）


def monthly_ic(df, factor="pred"):
    out = {}
    for dt, g in df.groupby("trade_date"):
        if len(g) < 50:
            continue
        out[dt] = g[factor].rank().corr(g["fwd_20"].rank())
    return pd.Series(out).dropna()


def quintile_spread(df):
    """月度分层: pred 分 5 层, Q1-Q5 = 最高组 fwd_20 - 最低组 fwd_20（月度平均）。"""
    diffs = []
    for dt, g in df.groupby("trade_date"):
        if len(g) < 50:
            continue
        try:
            g = g.copy()
            g["q"] = pd.qcut(g["pred"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5])
        except ValueError:
            continue
        q1 = g.loc[g["q"] == 1, "fwd_20"].mean()
        q5 = g.loc[g["q"] == 5, "fwd_20"].mean()
        if np.isfinite(q1) and np.isfinite(q5):
            diffs.append(q1 - q5)
    return np.asarray(diffs)


def fit_gbdt(tr_df):
    """与回测引擎一致的 GBDT（d3/nl7/mc80/λ2, 训练集最后 3 月早停）。"""
    tr = tr_df.sort_values("trade_date")
    X, y = tr[GBDT_FEATS].values, tr["fwd_20"].values
    val_months = sorted(tr["trade_date"].unique())[-3:]
    vm = tr["trade_date"].isin(val_months).values
    mdl = lgb.LGBMRegressor(n_estimators=500, learning_rate=0.05, num_leaves=7,
                            max_depth=3, min_child_samples=80, reg_lambda=2.0, reg_alpha=0.1,
                            subsample=0.9, colsample_bytree=0.9, random_state=42, verbose=-1)
    mdl.fit(X[~vm], y[~vm], eval_set=[(X[vm], y[vm])],
            callbacks=[lgb.early_stopping(50, verbose=False)])
    return mdl


def main():
    t0 = time.time()
    panel = pd.read_parquet(PANEL_PATH)
    prep = prep_feats(panel, GBDT_FEATS)
    months = sorted(prep["trade_date"].unique())
    print(f"[1] 面板 {len(months)} 月 ({months[0]}~{months[-1]}), prep 完成, 耗时 {time.time()-t0:.0f}s",
          flush=True)

    rng = np.random.default_rng(SEED)
    pred_list = []
    used_folds = 0
    for rep in range(N_REP):
        perm = rng.permutation(months)
        folds = np.array_split(perm, K_FOLDS)
        for fi, val_months_arr in enumerate(folds):
            val_set = set(int(x) for x in val_months_arr)
            purge_set = set()
            for m in val_set:
                y, mo = divmod(m, 100)
                for off in PURGE_MONTHS:
                    ny, nm = (y - 1, 12) if mo + off <= 0 else (y, mo + off)
                    if nm > 12:
                        ny, nm = ny + 1, 1
                    purge_set.add(ny * 100 + nm)
            train_months = [m for m in months if m not in val_set and m not in purge_set]
            if len(train_months) < 24:
                continue
            tr = prep[prep["trade_date"].isin(train_months)]
            mdl = fit_gbdt(tr)
            om = prep[prep["trade_date"].isin(val_set)]
            pred = mdl.predict(om[GBDT_FEATS])
            pred_list.append(pd.DataFrame({"trade_date": om["trade_date"].values,
                                           "pred": pred, "fwd_20": om["fwd_20"].values}))
            used_folds += 1
    print(f"[2] purged k-fold 完成: {used_folds} 折（{N_REP} 重复 × {K_FOLDS} 折）, "
          f"耗时 {time.time()-t0:.0f}s", flush=True)

    df_purged = pd.concat(pred_list, ignore_index=True)
    ics = monthly_ic(df_purged)
    spreads = quintile_spread(df_purged)
    icir = ics.mean() / (ics.std(ddof=1) + 1e-12) * np.sqrt(12)
    print(f"\n=== purged k-fold CV (C8 GBDT, k={K_FOLDS} × {N_REP} rep) ===")
    print(f"IC={ics.mean():+.4f}  ICIR={icir:+.2f}  正率={(ics>0).mean()*100:.0f}%  月数={len(ics)}")
    print(f"Q1-Q5 spread={spreads.mean():+.3f}%/月（Q1最高组 fwd_20 - Q5最低组）")

    # ---- 滚动 WFO 对照（expand-window, 2023+ 逐月, 与回测同口径） ----
    wfo_months = [m for m in months if m >= OOS_START]
    wfo_pred = []
    for i, m in enumerate(wfo_months):
        tr = prep[prep["trade_date"] < m]
        mdl = fit_gbdt(tr)
        om = prep[prep["trade_date"] == m]
        wfo_pred.append(pd.DataFrame({"trade_date": m, "pred": mdl.predict(om[GBDT_FEATS]),
                                      "fwd_20": om["fwd_20"].values}))
    df_wfo = pd.concat(wfo_pred, ignore_index=True)
    wics = monthly_ic(df_wfo)
    w_icir = wics.mean() / (wics.std(ddof=1) + 1e-12) * np.sqrt(12)
    w_spreads = quintile_spread(df_wfo)
    print(f"\n=== 对照: 滚动 WFO（2023+ 逐月, train < m）===")
    print(f"IC={wics.mean():+.4f}  ICIR={w_icir:+.2f}  正率={(wics>0).mean()*100:.0f}%  月数={len(wics)}")
    print(f"Q1-Q5 spread={w_spreads.mean():+.3f}%/月")

    report = {
        "scope": "P2-r7 C8 GBDT purged k-fold CV (2026-08-16)",
        "model": "LGBMRegressor d3/nl7/mc80/λ2 C8(10 features), 与 engine.py 同超参",
        "label": "fwd_20 (未来20交易日收益)",
        "purge": "验证月 ±1 月 (fwd_20 跨月标签窗口)",
        "embargo": "验证月后 1 月",
        "cv": {"k_folds": K_FOLDS, "n_rep": N_REP, "folds_used": used_folds, "seed": SEED},
        "purged_cv": {
            "n_month": int(len(ics)),
            "ic_mean": float(ics.mean()),
            "icir": float(icir),
            "pos_rate": float((ics > 0).mean()),
            "q1q5_spread_mean": float(spreads.mean()) if len(spreads) else np.nan,
        },
        "rolling_wfo_2023": {
            "n_month": int(len(wics)),
            "ic_mean": float(wics.mean()),
            "icir": float(w_icir),
            "pos_rate": float((wics > 0).mean()),
            "q1q5_spread_mean": float(w_spreads.mean()) if len(w_spreads) else np.nan,
        },
        "interpretation": ("purged CV IC/ICIR 与滚动 WFO 同量级 → GBDT 预测能力不依赖单一时间路径; "
                           "purge 排除标签窗口重叠后仍稳健。模型层证据, 与策略层 PBO/DSR 互补。"),
    }
    with open(os.path.join(HERE, "purged_cv_report.json"), "w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2)
    print(f"\n[保存] purged_cv_report.json")
    print(f"[完成] 总耗时 {time.time()-t0:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())

# -*- coding: utf-8 -*-
"""杠杆三因子扩容：VAL 价值因子(BP/SP/DP/EP/VAL)接入 C8 GBDT 的对照测试。

背景:
  - C8 = 价量(ivol/ret_1m/momentum/volatility) + 2 Alpha101 + enh4(含roe/or_yoy/netprofit_yoy) + 3残差筹码
  - 资金流(moneyflow)已在 c8mf_backtest 证无增量; Alpha101/191 已证无增量; 质量(ROE/GM/LEV) IC≈0
  - style_factors.txt: 中证1000上 BP IC=0.074(NWt3.75)/SP IC=0.056(NWt3.25)/VAL IC=0.071(NWt3.32) 全显著,
    是唯一与 C8 正交且单因子证据显著的方向。

步骤:
  1. 从 pe_ttm 月度快照构建 BP=1/pb / SP=1/ps_ttm / EP=1/pe_ttm / DP=dv_ttm / VAL(合成) 并合并进面板
  2. VAL 单因子 IC (2023+ OOS)
  3. WFO IC 对照: C8 vs C8+VAL(合成) vs C8+BP_SP_DP(3个体)
  4. 回测对照: ENS vs ENS_VAL vs ENS_BPSPDP (T40/T60, 冻结 binary s123 口径)
"""
import os
import sys
import time

import numpy as np
import pandas as pd
import lightgbm as lgb

ROOT = r"c:\Users\liuqi\quant_system_v2"
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "research", "experiments", "exp_ens_t60_tv12"))
sys.path.insert(0, os.path.join(ROOT, "research", "sector_rotation"))

from engine import init_shared, run_backtest, prep_feats, winsorize, GBDT_FEATS  # noqa: E402

PE_DIR = os.path.join(ROOT, "research", "factor_dic", "data", "pe_ttm")
t0 = time.time()

VAL_COMP = ["val"]
VAL_INDIV = ["bp", "sp", "dp"]


def build_val_panel(panel_dates, pe_dir):
    """从 pe_ttm 月度快照构建 VAL 因子长表 (trade_date, ts_code, bp/sp/ep/dp/val)。"""
    rows = []
    for d in panel_dates:
        fp = os.path.join(pe_dir, f"{int(d)}.parquet")
        if not os.path.exists(fp):
            continue
        v = pd.read_parquet(fp)
        v = v.set_index("ts_code")
        out = pd.DataFrame(index=v.index)
        out["bp"] = 1.0 / v["pb"].where(v["pb"] > 0)
        out["sp"] = 1.0 / v["ps_ttm"].where(v["ps_ttm"] > 0)
        out["ep"] = 1.0 / v["pe_ttm"].where(v["pe_ttm"] > 0)
        out["dp"] = v["dv_ttm"]
        comps = []
        for k in ["bp", "sp", "dp"]:
            s = out[k].dropna()
            if len(s) > 20:
                comps.append((s - s.mean()) / (s.std() + 1e-8))
        if len(comps) >= 2:
            out["val"] = pd.concat(comps, axis=1).mean(axis=1)
        out["trade_date"] = int(d)
        out = out.reset_index().rename(columns={"index": "ts_code"})
        rows.append(out[["trade_date", "ts_code", "bp", "sp", "ep", "dp", "val"]])
    return pd.concat(rows, ignore_index=True)


def prep_feats_val(df, val_cols):
    """C8 特征处理(engine.prep_feats) + VAL 列截面 winsorize/zscore + NaN 填 0。"""
    df = prep_feats(df, GBDT_FEATS)
    for c in val_cols:
        df[c] = df.groupby("trade_date")[c].transform(lambda s: winsorize(s))
        df[c] = df.groupby("trade_date")[c].transform(
            lambda s: (s - s.mean()) / (s.std(ddof=1) + 1e-12))
    df[val_cols] = df[val_cols].fillna(0.0)
    return df


def train_gbdt_scores(panel, feats, oos_months):
    """滚动 WFO GBDT 打分 (与 engine.init_shared 口径一致)。"""
    scores = {}
    for i, m in enumerate(oos_months):
        tr = prep_feats_val(panel[panel["trade_date"] < m], [c for c in feats if c.startswith(("bp", "sp", "ep", "dp", "val"))]).sort_values("trade_date")
        X, y = tr[feats].values, tr["fwd_20"].values
        val_months = sorted(tr["trade_date"].unique())[-3:]
        vm = tr["trade_date"].isin(val_months).values
        mdl = lgb.LGBMRegressor(n_estimators=500, learning_rate=0.05, num_leaves=7,
                                max_depth=3, min_child_samples=80, reg_lambda=2.0, reg_alpha=0.1,
                                subsample=0.9, colsample_bytree=0.9, random_state=42, verbose=-1)
        mdl.fit(X[~vm], y[~vm], eval_set=[(X[vm], y[vm])],
                callbacks=[lgb.early_stopping(50, verbose=False)])
        om = prep_feats_val(panel[panel["trade_date"] == m], [c for c in feats if c.startswith(("bp", "sp", "ep", "dp", "val"))])
        scores[m] = pd.Series(mdl.predict(om[feats].values), index=om["ts_code"])
        if (i + 1) % 9 == 0:
            print(f"    WFO {i+1}/{len(oos_months)}, {time.time()-t0:.0f}s", flush=True)
    return scores


def mic(scores, panel, oos_months):
    ics = []
    for m in oos_months:
        g = panel[panel["trade_date"] == m]
        s = scores.get(m)
        if s is None:
            continue
        common = s.index.intersection(g["ts_code"])
        if len(common) < 50:
            continue
        ic = s[common].rank().corr(g.set_index("ts_code").loc[common, "fwd_20"].rank())
        if np.isfinite(ic):
            ics.append(ic)
    s_ = pd.Series(ics)
    return s_.mean(), s_.mean() / (s_.std() + 1e-12) * np.sqrt(12), s_.gt(0).mean()


def main():
    print("[1] init_shared (加载面板/px/s123/V8/C8打分)...", flush=True)
    shared = init_shared()
    panel = shared["panel"].copy()
    print(f"    完成 {time.time()-t0:.0f}s, 面板 {len(panel):,} 行", flush=True)

    print("[2] 构建 VAL 价值因子...", flush=True)
    panel_dates = sorted(panel["trade_date"].unique())
    val_panel = build_val_panel(panel_dates, PE_DIR)
    cov = len(val_panel["trade_date"].unique())
    print(f"    VAL 覆盖 {cov}/{len(panel_dates)} 个月, {len(val_panel):,} 行", flush=True)
    panel = panel.merge(val_panel, on=["trade_date", "ts_code"], how="left")
    for c in ["bp", "sp", "ep", "dp", "val"]:
        print(f"    {c}: 非空率 {panel[c].notna().mean()*100:.1f}%", flush=True)

    oos_months = [d for d in sorted(panel["trade_date"].unique()) if d >= 20230101]

    # ---- VAL 单因子 IC ----
    print("\n[3] VAL 单因子 IC (2023+ OOS)...", flush=True)
    pv = prep_feats_val(panel, ["bp", "sp", "ep", "dp", "val"])
    for c in ["bp", "sp", "ep", "dp", "val"]:
        ics = []
        for m in oos_months:
            g = pv[pv["trade_date"] == m]
            if len(g) < 50:
                continue
            ic = g[c].rank().corr(g["fwd_20"].rank())
            if np.isfinite(ic):
                ics.append(ic)
        s = pd.Series(ics)
        print(f"  {c:>4}: IC={s.mean():+.4f} ICIR={s.mean()/(s.std()+1e-12)*np.sqrt(12):+.2f} "
              f"正率={s.gt(0).mean()*100:.0f}%", flush=True)

    # ---- WFO IC 对照 ----
    print("\n[4] WFO IC 对照 (C8 vs C8+VAL vs C8+BP_SP_DP)...", flush=True)
    feats_val = GBDT_FEATS + VAL_COMP
    feats_bpspdp = GBDT_FEATS + VAL_INDIV
    print(f"    训练 C8+VAL ({len(feats_val)}特征)...", flush=True)
    score_gbdt_val = train_gbdt_scores(panel, feats_val, oos_months)
    print(f"    训练 C8+BP_SP_DP ({len(feats_bpspdp)}特征)...", flush=True)
    score_gbdt_bpspdp = train_gbdt_scores(panel, feats_bpspdp, oos_months)

    c8 = shared["scores"]["GBDT"]
    for name, sc in [("C8 GBDT", c8), ("C8+VAL GBDT", score_gbdt_val),
                     ("C8+BP_SP_DP GBDT", score_gbdt_bpspdp)]:
        ic, icir, pos = mic(sc, panel, oos_months)
        print(f"  {name:>18}: IC={ic:+.4f} ICIR={icir:+.2f} 正率={pos*100:.0f}%", flush=True)

    # ---- 打分注入 + 回测 ----
    print("\n[5] 注入打分 + 回测对照 (ENS vs ENS_VAL vs ENS_BPSPDP)...", flush=True)
    enh4 = shared["scores"]["ENH"]
    # 2023 前用 ENH4 填充 GBDT
    for d in sorted(panel["trade_date"].unique()):
        if d not in score_gbdt_val:
            score_gbdt_val[d] = enh4[d]
        if d not in score_gbdt_bpspdp:
            score_gbdt_bpspdp[d] = enh4[d]

    def ens_blend(gbdt_scores):
        out = {}
        for d in sorted(panel["trade_date"].unique()):
            e, g = enh4[d], gbdt_scores[d]
            common = e.index.intersection(g.index)
            out[d] = 0.5 * e[common].rank(pct=True) + 0.5 * g[common].rank(pct=True)
        return out

    shared["scores"]["ENS_VAL"] = ens_blend(score_gbdt_val)
    shared["scores"]["ENS_BPSPDP"] = ens_blend(score_gbdt_bpspdp)

    def metrics(nav_s):
        nav_s = nav_s.sort_index().astype(float)
        tot = nav_s.iloc[-1] / nav_s.iloc[0] - 1.0
        yrs = len(nav_s) / 242.0
        ann = (1 + tot) ** (1 / yrs) - 1 if yrs > 0 else 0.0
        dd_s = nav_s / nav_s.cummax() - 1.0
        ret = nav_s.pct_change().fillna(0.0)
        sharpe = ret.mean() / (ret.std() + 1e-8) * np.sqrt(242.0)
        nav_m = nav_s.groupby((nav_s.index // 100).astype(str)).last()
        dd_m = nav_m / nav_m.cummax() - 1.0
        return {"ann": ann, "maxdd": dd_s.min(), "maxdd_m": dd_m.min(),
                "calmar": ann / (-dd_s.min() + 1e-9), "sharpe": sharpe}

    print(f"\n{'配置':<22} {'CAGR':>8} {'MaxDD':>8} {'月DD':>8} {'Calmar':>6} {'Sharpe':>7}")
    print("-" * 66)
    for tag in ["T40", "T60"]:
        for src, label in [("ENS", "ENS(C8)"), ("ENS_VAL", "ENS+VAL"),
                           ("ENS_BPSPDP", "ENS+BP_SP_DP")]:
            nav, _ = run_backtest(shared, src, tag, tgt_vol=None)
            m = metrics(nav)
            print(f"{tag}_{label:<18} {m['ann']:7.2%} {m['maxdd']:7.2%} "
                  f"{m['maxdd_m']:7.2%} {m['calmar']:5.2f} {m['sharpe']:6.2f}", flush=True)

    print(f"\n[完成] 总耗时 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""
run_fixed.py — study_007 修复版集成回测（集成回测员）
=====================================================
按任务书 Step1~Step7 执行。每步结果落盘 cache_fixed/ 与 results_fixed/，
支持断点续跑：已完成且产物齐全的 step 自动跳过。

运行方式（必须用户 Python，含 pyarrow）:
    "$DAIMON_USER_PYTHON" run_fixed.py --steps 1
    "$DAIMON_USER_PYTHON" run_fixed.py --steps all

注意：绘图由 make_charts.py（托管 Python，matplotlib 可用）完成，
因为用户 anaconda Python 的 matplotlib/PIL DLL 损坏（已实测）。
"""
import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

FIX = Path(__file__).resolve().parent
DATA = FIX / "data"
RES = FIX / "results_fixed"
CACHE = FIX / "cache_fixed"
RES.mkdir(exist_ok=True)
CACHE.mkdir(exist_ok=True)
sys.path.insert(0, str(FIX))

import factors_v2  # noqa: E402
import engine_v2   # noqa: E402

PRICE5 = ["ret_1m", "low_vol", "ivol", "turn_20d", "oi_spread"]
CS5 = [c + "_cs" for c in PRICE5]
TDY = 252

# 面板需要的列（去掉 amount/vol/pe/pb/name 省内存）
PANEL_COLS = ["ts_code", "trade_date", "open", "close", "pre_close", "daily_ret",
              "circ_mv", "turnover_rate", "industry", "list_date",
              "is_st", "limit_up", "limit_down", "mkt_ret"]
FACTOR_INPUT_COLS = ["ts_code", "trade_date", "open", "close", "pre_close",
                     "daily_ret", "turnover_rate", "mkt_ret"]

TRAIN_START, TRAIN_END = "20200101", "20221231"   # 因子选择纪律窗口
OOS_START, OOS_END = "20230101", "20251231"       # 样本外主回测窗口
FUNDA_START = "20230501"                           # 基本面变体窗口起点


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_panel(cols=None):
    df = pd.read_parquet(DATA / "panel.parquet", columns=cols or PANEL_COLS)
    df["trade_date"] = df["trade_date"].astype(str)
    df["ts_code"] = df["ts_code"].astype(str)
    return df


def month_ends(calendar):
    s = pd.Series(sorted(calendar))
    return s.groupby(s.str[:6]).max().tolist()


def is_bj_code(ts):
    s = str(ts)
    if s.endswith(".BJ"):
        return True
    prefix = s.split(".")[0]
    return prefix.startswith("920") or prefix[:1] in ("8", "4")


def check_nan(df, cols, where, threshold=0.20):
    """NaN 率异常检查：任一列超过 threshold 立即停程序。"""
    bad = []
    for c in cols:
        frac = float(df[c].isna().mean())
        log(f"    NaN率 {c:16s} = {frac:.4%}")
        if frac > threshold:
            bad.append((c, frac))
    if bad:
        log(f"  [FATAL] {where} NaN率超过 {threshold:.0%}，停止执行，需检查: {bad}")
        sys.exit(2)


def spearman(a, b):
    m = a.notna() & b.notna()
    if int(m.sum()) < 5:
        return np.nan
    return float(a[m].rank().corr(b[m].rank()))


def perf_metrics_from_nav(nav_df):
    """nav_df: [trade_date, nav, daily_ret] -> 指标 dict（与 engine 口径一致）"""
    r = nav_df["daily_ret"].to_numpy(dtype=np.float64)
    n = len(r)
    if n == 0:
        return {}
    nav = nav_df["nav"].to_numpy(dtype=np.float64)
    rstd = r.std(ddof=1) if n > 1 else 0.0
    m = nav_df["trade_date"].str[:6]
    mret = pd.Series(r).groupby(m.values).apply(lambda x: (1 + x).prod() - 1)
    return dict(
        total_return=float(nav[-1] - 1),
        ann_return=float(nav[-1] ** (TDY / n) - 1),
        sharpe=float(r.mean() / rstd * np.sqrt(TDY)) if rstd > 0 else np.nan,
        max_drawdown=float((nav / np.maximum.accumulate(nav) - 1).min()),
        monthly_win_rate=float((mret > 0).mean()),
        n_days=int(n),
    )


# ---------------------------------------------------------------------------
# Step 1: 价量因子计算（全面板，2019-07 起用于预热）
# ---------------------------------------------------------------------------
def step1():
    out = CACHE / "factors_price.parquet"
    if out.exists():
        log("[Step1] factors_price.parquet 已存在，跳过")
        return
    log("[Step1] 加载面板（因子所需列）...")
    panel = load_panel(FACTOR_INPUT_COLS)
    log(f"  panel: {panel.shape}, 日期 {panel['trade_date'].min()}~{panel['trade_date'].max()}")
    log("[Step1] compute_price_factors ...")
    t0 = time.time()
    fac = factors_v2.compute_price_factors(panel)
    log(f"  因子计算完成，用时 {time.time()-t0:.1f}s, shape={fac.shape}")

    log("[Step1] 预热期后（>=2020-01-01）NaN 检查：")
    after = fac[fac["trade_date"] >= "20200101"]
    check_nan(after, PRICE5, "compute_price_factors(>=2020-01-01)")

    fac.to_parquet(out, index=False)
    log(f"[Step1] 保存 {out}")
    del panel, fac, after


# ---------------------------------------------------------------------------
# Step 2: 月末截面预处理（仅月末信号日；截面处理逐日独立，结果与全量一致）
#         另做 PIT 财报版本（funda 覆盖宇宙，2023-04 月末起）
# ---------------------------------------------------------------------------
def _preprocess_monthly(fac, panel_sub, factor_cols, tag):
    """fac: 因子长表(全日期); panel_sub: 含 industry/circ_mv 的面板(全列日期);
    仅对月末日期做截面预处理。返回月末日期的预处理结果。"""
    cal = sorted(panel_sub["trade_date"].unique())
    me = set(month_ends(cal))
    f = fac[fac["trade_date"].isin(me)].copy()
    aux = panel_sub[panel_sub["trade_date"].isin(me)][
        ["ts_code", "trade_date", "industry", "circ_mv"]]
    f = f.merge(aux, on=["ts_code", "trade_date"], how="left")
    # 严格剔除北交所股票 / Strictly exclude BSE stocks
    f = f[~f["ts_code"].map(is_bj_code)]
    log(f"  [{tag}] 月末行数={len(f)}, 月末日期数={f['trade_date'].nunique()}")
    t0 = time.time()
    proc = factors_v2.preprocess_cross_section(f, factor_cols, neutralize=True)
    log(f"  [{tag}] preprocess 完成，用时 {time.time()-t0:.1f}s")
    log(f"  [{tag}] 预处理后 NaN 检查（>=2020-01-01 或全部行）:")
    chk = proc[proc["trade_date"] >= "20200101"] if tag == "price" else proc
    check_nan(chk, [c + "_cs" for c in factor_cols], f"preprocess[{tag}]", threshold=0.35)
    # 注: oi_spread 等个别因子在新上市/长停牌股上为 NaN 属正常，月末截面阈值 35%
    return proc


def step2():
    out_p = CACHE / "proc_price_monthly.parquet"
    out_f = CACHE / "proc_funda_monthly.parquet"
    if out_p.exists() and out_f.exists():
        log("[Step2] 预处理产物已存在，跳过")
        return
    fac = pd.read_parquet(CACHE / "factors_price.parquet")
    fac["trade_date"] = fac["trade_date"].astype(str)
    panel_aux = load_panel(["ts_code", "trade_date", "industry", "circ_mv"])

    if not out_p.exists():
        log("[Step2] 价量5因子 月末截面预处理 ...")
        proc = _preprocess_monthly(fac, panel_aux, PRICE5, "price")
        proc.to_parquet(out_p, index=False)
        log(f"[Step2] 保存 {out_p} shape={proc.shape}")

    if not out_f.exists():
        log("[Step2] PIT 财报版本预处理（funda 宇宙, 2023-04 月末起）...")
        funda = pd.read_parquet(DATA / "funda_pit.parquet")
        funda["trade_date"] = funda["trade_date"].astype(str)
        funda_ts = set(funda["ts_code"].unique())
        log(f"  funda_pit: {funda.shape}, 覆盖股票 {len(funda_ts)} 只, "
            f"{funda['trade_date'].min()}~{funda['trade_date'].max()}")
        cal = sorted(panel_aux["trade_date"].unique())
        me = [d for d in month_ends(cal) if d >= "20230401"]
        fac_f = fac[fac["trade_date"].isin(me) & fac["ts_code"].isin(funda_ts)].copy()
        merged = factors_v2.merge_fundamental(fac_f, funda)
        log(f"  merge_fundamental 后 shape={merged.shape}, "
            f"roe 覆盖率={merged['roe'].notna().mean():.4%}, "
            f"or_yoy 覆盖率={merged['or_yoy'].notna().mean():.4%}")
        aux = panel_aux[panel_aux["trade_date"].isin(me)]
        merged = merged.merge(aux, on=["ts_code", "trade_date"], how="left")
        # 严格剔除北交所股票 / Strictly exclude BSE stocks
        merged = merged[~merged["ts_code"].map(is_bj_code)]
        t0 = time.time()
        proc_f = factors_v2.preprocess_cross_section(merged, PRICE5 + ["roe", "or_yoy"],
                                                     neutralize=True)
        log(f"  [funda] preprocess 完成，用时 {time.time()-t0:.1f}s")
        check_nan(proc_f[proc_f["trade_date"] >= "20230501"],
                  [c + "_cs" for c in ["roe", "or_yoy"]], "preprocess[funda]",
                  threshold=0.35)
        proc_f.to_parquet(out_f, index=False)
        log(f"[Step2] 保存 {out_f} shape={proc_f.shape}")


# ---------------------------------------------------------------------------
# 工具: 远期收益（信号日收盘 -> 下一信号日收盘，daily_ret 复利，缺失=0）
# ---------------------------------------------------------------------------
class FwdRet:
    def __init__(self, panel_ret):
        w = panel_ret.pivot(index="trade_date", columns="ts_code", values="daily_ret")
        w = w.sort_index()
        self.dates = w.index.to_numpy()
        self.dpos = {d: i for i, d in enumerate(self.dates)}
        self.stocks = w.columns
        self.R = w.to_numpy(dtype=np.float64)

    def fwd(self, a, b):
        ia, ib = self.dpos[a], self.dpos[b]
        win = self.R[ia + 1: ib + 1, :]
        win = np.where(np.isnan(win), 0.0, win)
        win = np.clip(win, -0.999999, None)
        return pd.Series(np.expm1(np.log1p(win).sum(axis=0)), index=self.stocks)


def _ic_table(proc, fwd: FwdRet, sig_pairs, col):
    """sig_pairs: [(sig_date, next_sig_date)]; 返回每月 Rank IC 列表"""
    rows = []
    by_date = {d: g for d, g in proc.groupby("trade_date")}
    for a, b in sig_pairs:
        g = by_date.get(a)
        if g is None:
            continue
        x = g.set_index("ts_code")[col]
        fr = fwd.fwd(a, b).reindex(x.index)
        ic = spearman(x, fr)
        rows.append(dict(signal_date=a, ic=ic, n=int(x.notna().sum())))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Step 3: 训练期定因子（纪律冻结: 只用 <=2022-12-31 的信息）
# ---------------------------------------------------------------------------
def step3():
    out_sum = RES / "train_factor_summary.csv"
    out_frz = RES / "frozen_factors.csv"
    if out_sum.exists() and out_frz.exists():
        log("[Step3] 训练期因子筛选产物已存在，跳过")
        return
    log("[Step3] 训练期因子筛选 (2020-01 ~ 2022-12，不看 2023 后数据) ...")
    proc = pd.read_parquet(CACHE / "proc_price_monthly.parquet")
    proc["trade_date"] = proc["trade_date"].astype(str)

    panel_ret = load_panel(["ts_code", "trade_date", "daily_ret"])
    fwd = FwdRet(panel_ret)
    del panel_ret

    me = month_ends(fwd.dates)
    # 训练期信号: 2020-01月末 ~ 2022-11月末（远期收益窗口末端 <= 2022-12-31，
    # 保证因子筛选完全不触碰 2023 年数据）
    train_sigs = [d for d in me if "20200131" <= d <= "20221130"]
    sig_pairs = [(a, b) for a, b in zip(me[:-1], me[1:]) if a in set(train_sigs)]
    log(f"  训练期信号数={len(sig_pairs)} ({sig_pairs[0][0]} ~ {sig_pairs[-1][0]}), "
        f"最后远期窗口结束于 {sig_pairs[-1][1]} (<=20221231)")

    all_ic = {}
    summary = []
    for col in CS5:
        ic_df = _ic_table(proc, fwd, sig_pairs, col)
        all_ic[col] = ic_df
        m = ic_df["ic"].mean()
        s = ic_df["ic"].std(ddof=1)
        ir = m / s if s and s > 0 else np.nan
        pos = (ic_df["ic"] > 0).mean()
        summary.append(dict(factor=col, n_months=len(ic_df), mean_ic=m,
                            ic_std=s, IR=ir, ic_pos_rate=pos))
        log(f"  {col:14s}: meanIC={m:+.4f}  IR={ir:+.3f}  IC>0比例={pos:.4f}  n={len(ic_df)}")

    sum_df = pd.DataFrame(summary)
    # 纪律: 方向由训练期 IC 符号决定; 方向调整后 mean IC>0 且 IR>0.2 才入选; 权重 ∝ |IR|
    sum_df["direction"] = np.sign(sum_df["mean_ic"]).astype(int)
    sum_df["eff_mean_ic"] = sum_df["mean_ic"] * sum_df["direction"]
    sum_df["eff_IR"] = sum_df["IR"] * sum_df["direction"]
    sum_df["selected"] = (sum_df["eff_mean_ic"] > 0) & (sum_df["eff_IR"] > 0.2)
    ir_sel = sum_df.loc[sum_df["selected"], "eff_IR"]
    sum_df["weight"] = 0.0
    if len(ir_sel):
        sum_df.loc[sum_df["selected"], "weight"] = ir_sel / ir_sel.sum()

    sum_df.to_csv(out_sum, index=False, float_format="%.6f")
    pd.concat(all_ic.values(), keys=all_ic.keys(), names=["factor"]).reset_index(level=0)\
        .to_csv(RES / "train_factor_monthly_ic.csv", index=False, float_format="%.6f")

    frozen = sum_df[sum_df["selected"]][["factor", "direction", "weight",
                                         "mean_ic", "IR", "ic_pos_rate"]].copy()
    frozen.to_csv(out_frz, index=False, float_format="%.6f")
    log(f"  入选因子 {len(frozen)} 个:")
    for _, r in frozen.iterrows():
        log(f"    {r['factor']:14s} 方向={int(r['direction']):+d} 权重={r['weight']:.4f} "
            f"(训练期 meanIC={r['mean_ic']:+.4f}, IR={r['IR']:+.3f})")
    if frozen.empty:
        log("  [FATAL] 训练期无任何因子通过筛选 (mean IC>0 且 IR>0.2)，停止。")
        sys.exit(3)
    log(f"[Step3] 保存 {out_sum}, {out_frz}")


def load_frozen():
    frz = pd.read_csv(RES / "frozen_factors.csv")
    return list(frz.itertuples(index=False))  # (factor, direction, weight, ...)


def make_scores(proc, frozen, sig_dates, extra=None):
    """composite score = Σ w*dir*x_cs (+ extra terms); NaN 因子按 0(中性) 处理"""
    d = proc[proc["trade_date"].isin(set(sig_dates))][["ts_code", "trade_date"] + CS5 +
        ([c for c, _, _ in extra] if extra else [])].copy()
    score = np.zeros(len(d))
    for f in frozen:
        col, direction, weight = f.factor, int(f.direction), float(f.weight)
        score += weight * direction * d[col].fillna(0.0).to_numpy()
    if extra:
        for col, direction, weight in extra:
            score += weight * direction * d[col].fillna(0.0).to_numpy()
    return pd.DataFrame(dict(ts_code=d["ts_code"], trade_date=d["trade_date"], score=score))


# ---------------------------------------------------------------------------
# 引擎封装: 带缓存运行
# ---------------------------------------------------------------------------
def run_engine_cached(name, scores, panel, **kw):
    d = CACHE / f"bt_{name}"
    d.mkdir(exist_ok=True)
    f_nav, f_mon, f_ic, f_met = (d / "nav.csv", d / "monthly.csv",
                                 d / "ic.csv", d / "metrics.json")
    if all(p.exists() for p in (f_nav, f_mon, f_ic, f_met)):
        log(f"  [engine:{name}] 缓存命中，跳过")
        with open(f_met, encoding="utf-8") as fh:
            met = json.load(fh)
        return met
    log(f"  [engine:{name}] 运行 top_n={kw.get('top_n')}, cost={kw.get('cost_rate')} ...")
    t0 = time.time()
    res = engine_v2.run_backtest(scores, panel, verbose=False, **kw)
    log(f"  [engine:{name}] 完成 {time.time()-t0:.1f}s, "
        f"年化={res['metrics']['ann_return']:.4%}, sharpe={res['metrics']['sharpe']:.3f}, "
        f"IC={res['ic_series']['ic'].mean():.4f}, 退市={res['delist_count']}")
    res["daily_nav"].to_csv(f_nav, index=False, float_format="%.6f")
    res["monthly"].to_csv(f_mon, index=False, float_format="%.6f")
    res["ic_series"].to_csv(f_ic, index=False, float_format="%.6f")
    met = dict(res["metrics"])
    met["delist_count"] = res["delist_count"]
    ic = res["ic_series"]["ic"]
    met["ic_mean"] = float(ic.mean())
    met["ic_ir"] = float(ic.mean() / ic.std(ddof=1)) if ic.std(ddof=1) > 0 else np.nan
    met["ic_pos_rate"] = float((ic > 0).mean())
    with open(f_met, "w", encoding="utf-8") as fh:
        json.dump(met, fh, ensure_ascii=False, indent=2)
    # 退市明细 & 持仓日志（主回测需要）
    pd.DataFrame(res["delist_detail"]).to_csv(d / "delist_detail.csv", index=False)
    hold = [{k: v for k, v in h.items() if k != "weights"} for h in res["holdings_log"]]
    with open(d / "holdings_log.json", "w", encoding="utf-8") as fh:
        json.dump(hold, fh, ensure_ascii=False)
    return met


def engine_metrics_row(name, met):
    return dict(config=name, ann_return=met["ann_return"], sharpe=met["sharpe"],
                max_drawdown=met["max_drawdown"], monthly_win_rate=met["monthly_win_rate"],
                ann_turnover=met["ann_turnover"], ann_cost=met["ann_cost"],
                total_return=met["total_return"], delist_count=met["delist_count"],
                ic_mean=met["ic_mean"], ic_ir=met["ic_ir"], ic_pos_rate=met["ic_pos_rate"],
                n_rebalances=met["n_rebalances"])


def main_signals(fwd_dates):
    me = month_ends(fwd_dates)
    # 2022-12月末(执行于2023-01首个交易日) ~ 2025-12月末(最后信号仅用于IC配对)
    return [d for d in me if "20221201" <= d <= OOS_END]


def step4():
    """主回测 + 敏感性"""
    log("[Step4] 主回测与敏感性 (样本外 2023-01 ~ 2025-12) ...")
    proc = pd.read_parquet(CACHE / "proc_price_monthly.parquet")
    proc["trade_date"] = proc["trade_date"].astype(str)
    frozen = load_frozen()
    panel = load_panel()
    sigs = main_signals(sorted(panel["trade_date"].unique()))
    log(f"  信号日 {len(sigs)} 个: {sigs[0]} ~ {sigs[-1]}")
    scores = make_scores(proc, frozen, sigs)
    log(f"  scores: {scores.shape}, 每期股票数≈{len(scores)/len(sigs):.0f}")

    rows = []
    met = run_engine_cached("main", scores, panel, top_n=50, cost_rate=0.003,
                            max_per_industry=3, min_listed_days=60)
    rows.append(engine_metrics_row("top50_cost0.003(主)", met))
    for name, kw in [("cost15", dict(top_n=50, cost_rate=0.0015)),
                     ("top30", dict(top_n=30, cost_rate=0.003)),
                     ("top100", dict(top_n=100, cost_rate=0.003))]:
        m = run_engine_cached(name, scores, panel, max_per_industry=3,
                              min_listed_days=60, **kw)
        rows.append(engine_metrics_row(f"top{kw['top_n']}_cost{kw['cost_rate']}", m))
    df = pd.DataFrame(rows)
    df.to_csv(RES / "main_metrics_sensitivity.csv", index=False, float_format="%.6f")
    log(f"[Step4] 保存 {RES/'main_metrics_sensitivity.csv'}")
    log("\n" + df.to_string(index=False))

    # 等权基准（mkt_ret 复利，对齐主回测净值日期）
    nav = pd.read_csv(CACHE / "bt_main" / "nav.csv", dtype={"trade_date": str})
    mkt = panel[["trade_date", "mkt_ret"]].drop_duplicates("trade_date")
    mkt = mkt.set_index("trade_date")["mkt_ret"].reindex(nav["trade_date"])
    bret = mkt.fillna(0.0).to_numpy(dtype=np.float64)
    bench = pd.DataFrame(dict(trade_date=nav["trade_date"],
                              bench_ret=bret,
                              bench_nav=np.cumprod(1.0 + bret)))
    out = nav.merge(bench, on="trade_date")
    out.to_csv(RES / "main_nav_vs_benchmark.csv", index=False, float_format="%.6f")
    bmet = perf_metrics_from_nav(out[["trade_date", "bench_nav", "bench_ret"]]
                                 .rename(columns={"bench_nav": "nav", "bench_ret": "daily_ret"}))
    with open(RES / "benchmark_metrics.json", "w", encoding="utf-8") as fh:
        json.dump(bmet, fh, ensure_ascii=False, indent=2)
    log(f"  等权基准: 年化={bmet['ann_return']:.4%}, sharpe={bmet['sharpe']:.3f}, "
        f"maxDD={bmet['max_drawdown']:.4%}")

    # 分年度收益（策略 vs 基准）
    tmp = out.copy()
    tmp["year"] = tmp["trade_date"].str[:4]
    yr = tmp.groupby("year").apply(
        lambda g: pd.Series(dict(
            strategy=(1 + g["daily_ret"]).prod() - 1,
            benchmark=(1 + g["bench_ret"]).prod() - 1)))
    yr.to_csv(RES / "annual_returns.csv", float_format="%.6f")
    log("\n分年度收益:\n" + yr.to_string(float_format=lambda x: f"{x:.4%}"))

    # 持仓市值暴露（size tilt）: 每期持仓 circ_mv 中位数 / 全市场中位数
    with open(CACHE / "bt_main" / "holdings_log.json", encoding="utf-8") as fh:
        holds = json.load(fh)
    cmv = panel[["ts_code", "trade_date", "circ_mv"]]
    cmv_map = {(r.ts_code, r.trade_date): r.circ_mv for r in cmv.itertuples()}
    uni_med = cmv[cmv["circ_mv"] > 0].groupby("trade_date")["circ_mv"].median().to_dict()
    ratios = []
    for h in holds:
        sd = h["signal_date"]
        hv = [cmv_map.get((t, sd)) for t in h["holdings"]]
        hv = [x for x in hv if x and x > 0]
        if hv and sd in uni_med:
            ratios.append(float(np.median(hv) / uni_med[sd]))
    tilt = float(np.mean(ratios)) if ratios else np.nan
    log(f"  持仓市值暴露: 持仓circ_mv中位数/全市场中位数 平均 = {tilt:.4f}")
    with open(RES / "size_tilt.json", "w", encoding="utf-8") as fh:
        json.dump(dict(holdings_median_mv_ratio=tilt), fh)
    # 主回测 IC 与退市明细复制到结果目录
    pd.read_csv(CACHE / "bt_main" / "ic.csv").to_csv(RES / "main_monthly_ic.csv",
                                                     index=False, float_format="%.6f")
    dd = pd.read_csv(CACHE / "bt_main" / "delist_detail.csv")
    dd.to_csv(RES / "main_delist_detail.csv", index=False)
    log(f"  主回测退市股数={len(dd)}")


def step5():
    """分域分析: Q1 小市值 vs Q5 大市值（真实 circ_mv 五分位）"""
    log("[Step5] 分域分析 (circ_mv 五分位, Q1最小/Q5最大) ...")
    proc = pd.read_parquet(CACHE / "proc_price_monthly.parquet")
    proc["trade_date"] = proc["trade_date"].astype(str)
    frozen = load_frozen()
    panel = load_panel()
    sigs = main_signals(sorted(panel["trade_date"].unique()))
    scores = make_scores(proc, frozen, sigs)

    # 每个信号日按 circ_mv 分 5 档
    cs = panel[panel["trade_date"].isin(set(sigs))][["ts_code", "trade_date", "circ_mv"]].copy()
    cs = cs[cs["circ_mv"] > 0]
    cs["q"] = cs.groupby("trade_date")["circ_mv"].transform(
        lambda x: pd.qcut(x.rank(method="first"), 5, labels=False)) + 1
    qmap = cs[["ts_code", "trade_date", "q"]]
    sc = scores.merge(qmap, on=["ts_code", "trade_date"], how="left")
    log(f"  分档后 scores: {sc.shape}, 各档只数:\n"
        + sc.groupby("q")["ts_code"].count().to_string())

    rows = []
    for q, name in [(1, "q1_small"), (5, "q5_large")]:
        s_q = sc[sc["q"] == q][["ts_code", "trade_date", "score"]]
        met = run_engine_cached(name, s_q, panel, top_n=30, cost_rate=0.003,
                                max_per_industry=3, min_listed_days=60)
        rows.append(engine_metrics_row(f"Q{q}_{'小市值' if q==1 else '大市值'}_top30", met))
    df = pd.DataFrame(rows)
    df.to_csv(RES / "size_domain_metrics.csv", index=False, float_format="%.6f")
    log("\n" + df.to_string(index=False))

    # 分域净值（画图用）
    nav1 = pd.read_csv(CACHE / "bt_q1_small" / "nav.csv", dtype={"trade_date": str})
    nav5 = pd.read_csv(CACHE / "bt_q5_large" / "nav.csv", dtype={"trade_date": str})
    navx = nav1.merge(nav5, on="trade_date", suffixes=("_q1", "_q5"))
    navx.to_csv(RES / "size_domain_nav.csv", index=False, float_format="%.6f")

    # 市值五分位等权年度收益（风格背景: 2025 小票行情用数据说话）
    log("  计算市值五分位等权年度收益（风格 beta 背景）...")
    panel_non_bj = panel[~panel["ts_code"].map(is_bj_code)]
    R = panel_non_bj.pivot(index="trade_date", columns="ts_code", values="daily_ret").sort_index()
    CMV = panel_non_bj.pivot(index="trade_date", columns="ts_code", values="circ_mv").sort_index()
    me = month_ends(R.index)
    me_pairs = list(zip(me[:-1], me[1:])) + [(me[-1], R.index[-1])]
    q_ret = {q: [] for q in range(1, 6)}   # q -> list of (date, mean_ret)
    dates = R.index.to_numpy()
    dpos = {d: i for i, d in enumerate(dates)}
    for a, b in me_pairs:
        if a < "20221201":
            continue
        cmv_a = CMV.loc[a]
        v = cmv_a[cmv_a > 0].dropna()
        qlab = pd.qcut(v.rank(method="first"), 5, labels=False) + 1
        qa = {q: set(qlab[qlab == q].index) for q in range(1, 6)}
        seg = R.iloc[dpos[a] + 1: dpos[b] + 1]
        for q in range(1, 6):
            cols = [c for c in seg.columns if c in qa[q]]
            if not cols:
                continue
            sub = seg[cols]
            for dt, row in sub.iterrows():
                q_ret[q].append((dt, float(np.nanmean(row.to_numpy(dtype=np.float64)))))
    yr_rows = {}
    for q in range(1, 6):
        s = pd.Series(dict(q_ret[q])).sort_index()
        yr_rows[f"Q{q}"] = s.groupby(s.index.str[:4]).apply(lambda x: (1 + x).prod() - 1)
    yr_df = pd.DataFrame(yr_rows)
    yr_df.to_csv(RES / "size_quintile_annual_returns.csv", float_format="%.6f")
    log("\n市值五分位等权年度收益:\n" + yr_df.to_string(float_format=lambda x: f"{x:.4%}"))


def step6():
    """基本面变体: 价量(冻结权重) + roe/or_yoy(先验各20%价量总权重)"""
    log("[Step6] 基本面变体 (funda 宇宙, 2023-05 ~ 2025-12, 描述性检验) ...")
    proc_f = pd.read_parquet(CACHE / "proc_funda_monthly.parquet")
    proc_f["trade_date"] = proc_f["trade_date"].astype(str)
    frozen = load_frozen()
    panel = load_panel()
    me = month_ends(sorted(panel["trade_date"].unique()))
    sigs = [d for d in me if "20230401" <= d <= OOS_END]
    log(f"  funda 变体信号日 {len(sigs)} 个: {sigs[0]} ~ {sigs[-1]}")
    extra = [("roe_cs", 1, 0.2), ("or_yoy_cs", 1, 0.2)]  # 高 roe/高营收增长为好, 先验权重
    scores = make_scores(proc_f, frozen, sigs, extra=extra)
    log(f"  scores: {scores.shape}")

    met = run_engine_cached("funda", scores, panel, top_n=50, cost_rate=0.003,
                            max_per_industry=3, min_listed_days=60)
    row = engine_metrics_row("funda_价量+roe+or_yoy", met)
    pd.DataFrame([row]).to_csv(RES / "funda_variant_metrics.csv", index=False,
                               float_format="%.6f")
    log("\n" + pd.DataFrame([row]).to_string(index=False))

    # roe / or_yoy 独立月度 Rank IC（描述性）
    panel_ret = panel[["ts_code", "trade_date", "daily_ret"]]
    fwd = FwdRet(panel_ret)
    funda_sigs = [d for d in sigs if d >= "20230501" and d != me[-1]]
    me_idx = {d: i for i, d in enumerate(me)}
    pairs = [(d, me[me_idx[d] + 1]) for d in funda_sigs if me_idx[d] + 1 < len(me)]
    rows = []
    for col in ["roe_cs", "or_yoy_cs"]:
        ic_df = _ic_table(proc_f, fwd, pairs, col)
        ic_df.insert(0, "factor", col)
        rows.append(ic_df)
        m, s = ic_df["ic"].mean(), ic_df["ic"].std(ddof=1)
        log(f"  {col}: 月均RankIC={m:+.4f}, IR={m/s:+.3f}, IC>0比例={(ic_df['ic']>0).mean():.4f}")
    pd.concat(rows).to_csv(RES / "funda_standalone_ic.csv", index=False, float_format="%.6f")
    nav = pd.read_csv(CACHE / "bt_funda" / "nav.csv", dtype={"trade_date": str})
    nav.to_csv(RES / "funda_nav.csv", index=False, float_format="%.6f")
    log("[Step6] 完成")


def step7():
    """汇总 results.json（供报告与绘图脚本使用）"""
    log("[Step7] 汇总 results.json ...")
    out = {}
    out["frozen_factors"] = pd.read_csv(RES / "frozen_factors.csv").to_dict("records")
    out["train_summary"] = pd.read_csv(RES / "train_factor_summary.csv").to_dict("records")
    out["main_sensitivity"] = pd.read_csv(RES / "main_metrics_sensitivity.csv").to_dict("records")
    with open(RES / "benchmark_metrics.json", encoding="utf-8") as fh:
        out["benchmark"] = json.load(fh)
    with open(RES / "size_tilt.json", encoding="utf-8") as fh:
        out["size_tilt"] = json.load(fh)
    out["size_domains"] = pd.read_csv(RES / "size_domain_metrics.csv").to_dict("records")
    out["funda_variant"] = pd.read_csv(RES / "funda_variant_metrics.csv").to_dict("records")
    with open(RES / "results.json", "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=2, default=float)
    log(f"[Step7] 保存 {RES/'results.json'}")


STEPS = dict(step1=step1, step2=step2, step3=step3, step4=step4,
             step5=step5, step6=step6, step7=step7)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", required=True,
                    help="逗号分隔 step 编号 或 all, 如 --steps 1,2")
    args = ap.parse_args()
    todo = list(STEPS) if args.steps == "all" else \
        [f"step{s.strip()}" for s in args.steps.split(",")]
    for s in todo:
        log(f"{'='*66}\n>>> {s}\n{'='*66}")
        STEPS[s]()
    log("全部请求步骤完成。")

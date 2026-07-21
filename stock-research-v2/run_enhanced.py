# -*- coding: utf-8 -*-
"""
run_enhanced.py — study_007 组合构建增强检验（组合优化检验员）
================================================================
在 run_fixed.py 冻结因子（2020-2022 训练期）基础上，样本外 2023-01~2025-12 测试：
  A组: 过滤与加权变体（engine_v3 多头 top50, cost=0.003）
       A0 基线等权无过滤（复算核对 results_fixed）
       A1 剔北交所
       A2 剔北交所 + 剔最小市值20%
       A3 A2 + inv_vol 加权
       A4 A2 + score 加权
  B组: 多空纯 alpha（ls_engine 50/50）
       B1 全宇宙 LS；B2 scores 预剔北交所+最小市值20% 后 LS
  C组: 风格归因与净 alpha 核算（风格匹配基准 + LS 与 Q1-Q5 风格利差回归）

运行（计算必须用户 Python，含 pyarrow）:
    "$DAIMON_USER_PYTHON" run_enhanced.py
绘图由 make_charts_enhanced.py（托管 Python, matplotlib 可用）完成。
"""
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
RES_F = FIX / "results_fixed"
CACHE = FIX / "cache_fixed"
OUT = FIX / "results_enhanced"
OUT.mkdir(exist_ok=True)
sys.path.insert(0, str(FIX))

import engine_v3   # noqa: E402
import ls_engine   # noqa: E402

TDY = 252
PANEL_COLS = ["ts_code", "trade_date", "open", "close", "pre_close", "daily_ret",
              "circ_mv", "turnover_rate", "industry", "list_date",
              "is_st", "limit_up", "limit_down", "mkt_ret"]
OOS_START, OOS_END = "20230101", "20251231"
BASELINE_ANN = 0.175339   # results_fixed top50_cost0.003 年化（核对锚点）


def log(msg):
    try:
        msg_str = str(msg).encode(sys.stdout.encoding or 'gbk', errors='replace').decode(sys.stdout.encoding or 'gbk')
    except Exception:
        msg_str = str(msg)
    print(f"[{time.strftime('%H:%M:%S')}] {msg_str}", flush=True)


def month_ends(calendar):
    s = pd.Series(sorted(calendar))
    return s.groupby(s.str[:6]).max().tolist()


def is_bj_code(ts):
    s = str(ts)
    if s.endswith(".BJ"):
        return True
    prefix = s.split(".")[0]
    return prefix.startswith("920") or prefix[:1] in ("8", "4")


# ---------------------------------------------------------------------------
# 数据加载与 score 合成（复刻 run_fixed.py Step3/Step4 逻辑）
# ---------------------------------------------------------------------------
def load_all():
    log("加载冻结因子 frozen_factors.csv ...")
    frozen = pd.read_csv(RES_F / "frozen_factors.csv")
    log(f"  冻结因子 {len(frozen)} 个: "
        + ", ".join(f"{r.factor}(w={r.weight:.4f},d={int(r.direction)})"
                    for r in frozen.itertuples()))

    log("加载月末预处理缓存 proc_price_monthly.parquet ...")
    proc = pd.read_parquet(CACHE / "proc_price_monthly.parquet")
    proc["trade_date"] = proc["trade_date"].astype(str)
    proc["ts_code"] = proc["ts_code"].astype(str)

    log("加载面板 panel.parquet（>=2022-10，留 inv_vol 20日预热窗口）...")
    panel = pd.read_parquet(DATA / "panel.parquet", columns=PANEL_COLS)
    panel["trade_date"] = panel["trade_date"].astype(str)
    panel["ts_code"] = panel["ts_code"].astype(str)
    panel = panel[panel["trade_date"] >= "20221001"].reset_index(drop=True)
    log(f"  panel: {panel.shape}, {panel['trade_date'].min()}~{panel['trade_date'].max()}, "
        f"{panel['ts_code'].nunique()} 股")

    cal = sorted(panel["trade_date"].unique())
    sigs = [d for d in month_ends(cal) if "20221201" <= d <= OOS_END]
    log(f"  信号日 {len(sigs)} 个: {sigs[0]} ~ {sigs[-1]}")

    # composite score = Σ w*dir*x_cs，NaN 按 0（与 run_fixed.make_scores 一致）
    d = proc[proc["trade_date"].isin(set(sigs))].copy()
    score = np.zeros(len(d))
    for r in frozen.itertuples():
        score += float(r.weight) * float(r.direction) * d[r.factor].fillna(0.0).to_numpy()
    scores = pd.DataFrame(dict(ts_code=d["ts_code"], trade_date=d["trade_date"],
                               score=score))
    log(f"  scores: {scores.shape}, 每期≈{len(scores)/len(sigs):.0f} 股")
    return frozen, proc, panel, sigs, scores


# ---------------------------------------------------------------------------
# A 组：engine_v3 多头变体
# ---------------------------------------------------------------------------
A_VARIANTS = [
    ("A0_基线等权无过滤",      dict(weighting="equal")),
    ("A1_剔北交所",            dict(weighting="equal", exclude_bj=True)),
    ("A2_剔BJ剔最小市值20%",   dict(weighting="equal", exclude_bj=True,
                                    min_mcap_quantile=0.2)),
    ("A3_A2+inv_vol加权",      dict(weighting="inv_vol", exclude_bj=True,
                                    min_mcap_quantile=0.2)),
    ("A4_A2+score加权",        dict(weighting="score", exclude_bj=True,
                                    min_mcap_quantile=0.2)),
]


def run_group_a(scores, panel):
    log("=" * 72)
    log("A组: 过滤与加权变体 (engine_v3, top_n=50, cost=0.003)")
    log("=" * 72)
    res = {}
    for name, kw in A_VARIANTS:
        t0 = time.time()
        log(f"  [{name}] 运行中 ...")
        r = engine_v3.run_backtest(scores, panel, top_n=50, cost_rate=0.003,
                                   max_per_industry=3, min_listed_days=60,
                                   verbose=False, **kw)
        m = r["metrics"]
        log(f"  [{name}] 完成 {time.time()-t0:.0f}s | 年化={m['ann_return']:.2%} "
            f"夏普={m['sharpe']:.3f} 回撤={m['max_drawdown']:.2%} "
            f"月均单边换手={m['avg_monthly_one_side_turnover']:.2%} "
            f"市值比={m['holdings_mcap_ratio']:.3f}")
        res[name] = r
    # A0 核对
    a0 = res["A0_基线等权无过滤"]["metrics"]["ann_return"]
    log(f"  [核对] A0 年化={a0:.6f} vs results_fixed {BASELINE_ANN:.6f}, "
        f"差={a0-BASELINE_ANN:+.6f}")
    if abs(a0 - BASELINE_ANN) > 0.005:
        log("  [FATAL] A0 与 results_fixed 偏差>0.5pp，停止，需查明原因")
        sys.exit(3)
    log("  [核对通过] A0 与 results_fixed 一致")
    return res


# ---------------------------------------------------------------------------
# B 组：多空纯 alpha
# ---------------------------------------------------------------------------
def run_group_b(scores, panel):
    log("=" * 72)
    log("B组: 多空纯 alpha (ls_engine 50/50, cost=0.003)")
    log("=" * 72)
    res = {}
    t0 = time.time()
    log("  [B1_全宇宙LS] 运行中 ...")
    res["B1_全宇宙LS"] = ls_engine.run_longshort(scores, panel, n_long=50, n_short=50,
                                               cost_rate=0.003, min_listed_days=60,
                                               hedge_ratio=1.0, verbose=False)
    m = res["B1_全宇宙LS"]["metrics"]
    log(f"  [B1_全宇宙LS] 完成 {time.time()-t0:.0f}s | 年化={m['ann_return']:.2%} "
        f"夏普={m['sharpe']:.3f} 回撤={m['max_drawdown']:.2%} 月胜率={m['monthly_win_rate']:.2%}")

    # B2: scores 预剔北交所 + 各信号日最小市值20%
    t0 = time.time()
    log("  [B2_剔BJ剔小市值LS] scores 预过滤 ...")
    cs = panel[panel["trade_date"].isin(set(scores["trade_date"].unique()))][
        ["ts_code", "trade_date", "circ_mv"]].copy()
    cs["circ_mv"] = pd.to_numeric(cs["circ_mv"], errors="coerce")
    thr = cs.groupby("trade_date")["circ_mv"].quantile(0.2).rename("thr")
    s2 = scores[~scores["ts_code"].map(is_bj_code)].merge(
        cs, on=["ts_code", "trade_date"], how="left").merge(
        thr, on="trade_date", how="left")
    s2 = s2[s2["circ_mv"].isna() | (s2["circ_mv"] >= s2["thr"])]
    s2 = s2[["ts_code", "trade_date", "score"]]
    log(f"  [B2] 过滤后 scores: {s2.shape}（原 {scores.shape}）")
    res["B2_剔BJ剔小市值LS"] = ls_engine.run_longshort(s2, panel, n_long=50, n_short=50,
                                                     cost_rate=0.003, min_listed_days=60,
                                                     hedge_ratio=1.0, verbose=False)
    m = res["B2_剔BJ剔小市值LS"]["metrics"]
    log(f"  [B2_剔BJ剔小市值LS] 完成 {time.time()-t0:.0f}s | 年化={m['ann_return']:.2%} "
        f"夏普={m['sharpe']:.3f} 回撤={m['max_drawdown']:.2%} 月胜率={m['monthly_win_rate']:.2%}")
    log("  ⚠️ A股个股不可做空，B组为研究性 alpha 纯度检验，不代表可实盘收益")
    return res


# ---------------------------------------------------------------------------
# 市值五分位月度收益（等权，每月末再平衡，口径同 run_fixed.step5）
# ---------------------------------------------------------------------------
def quintile_monthly_returns(panel, sigs):
    log("计算市值五分位月度收益（Q1最小~Q5最大，等权，月末再平衡）...")
    R = panel.pivot(index="trade_date", columns="ts_code", values="daily_ret").sort_index()
    CMV = panel.pivot(index="trade_date", columns="ts_code", values="circ_mv").sort_index()
    me = [d for d in month_ends(R.index) if d >= "20221201"]
    me_pairs = list(zip(me[:-1], me[1:])) + [(me[-1], R.index[-1])]
    dates = R.index.to_numpy()
    dpos = {d: i for i, d in enumerate(dates)}
    q_daily = {q: [] for q in range(1, 6)}
    qlab_map = {}   # signal_date -> Series(ts_code -> q)
    for a, b in me_pairs:
        cmv_a = CMV.loc[a]
        v = cmv_a[cmv_a > 0].dropna()
        qlab = pd.qcut(v.rank(method="first"), 5, labels=False) + 1
        qlab_map[a] = qlab
        qa = {q: set(qlab[qlab == q].index) for q in range(1, 6)}
        seg = R.iloc[dpos[a] + 1: dpos[b] + 1]
        for q in range(1, 6):
            cols = [c for c in seg.columns if c in qa[q]]
            if not cols:
                continue
            sub = seg[cols].to_numpy(dtype=np.float64)
            for i, dt in enumerate(seg.index):
                q_daily[q].append((dt, float(np.nanmean(sub[i]))))
    qm = {}
    for q in range(1, 6):
        s = pd.Series(dict(q_daily[q])).sort_index()
        qm[q] = s.groupby(s.index.str[:6]).apply(lambda x: (1 + x).prod() - 1)
    qm = pd.DataFrame(qm)
    qm.columns = [f"Q{q}" for q in range(1, 6)]
    qm.index.name = "month"
    qm.to_csv(OUT / "size_quintile_monthly_returns.csv", float_format="%.6f")
    log(f"  五分位月收益 shape={qm.shape}, 已存 size_quintile_monthly_returns.csv")
    return qm, qlab_map


# ---------------------------------------------------------------------------
# C 组：风格归因与净 alpha 核算
# ---------------------------------------------------------------------------
def style_attribution(res_a, panel, sigs, qlab_map, qm, mkt_m):
    """对每个多头变体：
    - 月均超额 vs 全A等权基准, IR = mean/std*sqrt(12)
    - 持仓市值五分位权重（每期持仓在信号日的五分位分布）
    - 风格匹配基准（年度: Σ_q w_q × 五分位年度收益）→ 净 alpha（年化）
    - 交叉验证：月度口径风格匹配基准（Σ_q w_q × 五分位月收益）→ 净 alpha（年化）
    """
    log("=" * 72)
    log("C组: 风格归因与净 alpha 核算")
    log("=" * 72)
    q_annual = pd.read_csv(RES_F / "size_quintile_annual_returns.csv",
                           index_col=0)
    log("  五分位年度收益（results_fixed）:\n"
        + q_annual.to_string(float_format=lambda x: f"{x:.4%}"))

    # 每期持仓的五分位权重
    rows = []
    detail = {}
    for name, r in res_a.items():
        w_rows = []
        for h in r["holdings_log"]:
            sd = h["signal_date"]
            ql = qlab_map.get(sd)
            if ql is None:
                continue
            qs = ql.reindex(h["holdings"]).dropna()
            if len(qs) == 0:
                continue
            wq = qs.value_counts(normalize=True)
            w_rows.append(dict(signal_date=sd,
                               **{f"Q{q}": float(wq.get(q, 0.0)) for q in range(1, 6)}))
        wdf = pd.DataFrame(w_rows)
        detail[name] = wdf
        wdf.to_csv(OUT / f"holdings_quintile_weights_{name}.csv",
                   index=False, float_format="%.6f")

        # 策略月收益 / 年收益
        nav = r["daily_nav"].copy()
        nav["month"] = nav["trade_date"].str[:6]
        nav["year"] = nav["trade_date"].str[:4]
        mret = nav.groupby("month")["daily_ret"].apply(lambda x: (1 + x).prod() - 1)
        yret = nav.groupby("year")["daily_ret"].apply(lambda x: (1 + x).prod() - 1)

        # 超额 vs 全A等权基准 + IR
        ex = (mret - mkt_m).dropna()
        ir = float(ex.mean() / ex.std(ddof=1) * np.sqrt(12)) if ex.std(ddof=1) > 0 else np.nan
        ann_ex_geo = float((1 + ex).prod() ** (12 / len(ex)) - 1)

        # 年度口径风格匹配基准：每年信号日的平均五分位权重 × 五分位年收益
        wdf["year"] = wdf["signal_date"].str[:4]
        wq_y = wdf.groupby("year")[[f"Q{q}" for q in range(1, 6)]].mean()
        style_ann = {}
        for y in ["2023", "2024", "2025"]:
            if y not in wq_y.index or y not in q_annual.index.astype(str):
                continue
            w = wq_y.loc[y].to_numpy()
            qv = q_annual.loc[int(y) if q_annual.index.dtype != object else y].to_numpy()
            style_ann[y] = float(w @ qv)
        strat_ann = {y: float(yret.get(y, np.nan)) for y in ["2023", "2024", "2025"]}
        net_y = {y: strat_ann[y] - style_ann.get(y, np.nan) for y in strat_ann}
        # 年化（几何）
        years = [y for y in ["2023", "2024", "2025"] if y in style_ann]
        g_strat = float(np.prod([1 + strat_ann[y] for y in years]) ** (1 / len(years)) - 1)
        g_style = float(np.prod([1 + style_ann[y] for y in years]) ** (1 / len(years)) - 1)
        net_ann_geo = g_strat - g_style

        # 月度口径交叉验证：风格匹配月收益 = 当月信号日权重 × 五分位月收益
        w_m = wdf.set_index("signal_date")[[f"Q{q}" for q in range(1, 6)]]
        # 月收益归属月 = 信号日所在月的下一月（持仓期）；改用执行月对齐：
        # 持仓期收益主要落在信号月之后，近似用信号日月份+1 对齐五分位月收益
        sm = []
        for sd, wrow in w_m.iterrows():
            month = pd.Period(sd[:6], freq="M") + 1
            key = str(month).replace("-", "")
            if key in qm.index:
                sm.append((key, float(wrow.to_numpy() @ qm.loc[key].to_numpy())))
        sm = pd.Series(dict(sm)).sort_index()
        common = mret.index.intersection(sm.index)
        ex_style_m = (mret.loc[common] - sm.loc[common])
        net_m_geo = float((1 + ex_style_m).prod() ** (12 / len(common)) - 1)
        net_m_arith = float(ex_style_m.mean() * 12)

        rows.append(dict(
            variant=name,
            excess_ann_geo=ann_ex_geo, excess_IR=ir,
            w_Q1=float(w_m["Q1"].mean()), w_Q2=float(w_m["Q2"].mean()),
            w_Q3=float(w_m["Q3"].mean()), w_Q4=float(w_m["Q4"].mean()),
            w_Q5=float(w_m["Q5"].mean()),
            strat_2023=strat_ann["2023"], strat_2024=strat_ann["2024"],
            strat_2025=strat_ann["2025"],
            style_2023=style_ann.get("2023", np.nan),
            style_2024=style_ann.get("2024", np.nan),
            style_2025=style_ann.get("2025", np.nan),
            net_2023=net_y["2023"], net_2024=net_y["2024"], net_2025=net_y["2025"],
            strat_ann_geo=g_strat, style_ann_geo=g_style,
            net_alpha_ann=net_ann_geo,
            net_alpha_ann_monthly_geo=net_m_geo,
            net_alpha_ann_monthly_arith=net_m_arith,
        ))
        log(f"  [{name}] 超额IR={ir:.3f} 年度风格净alpha(年化)={net_ann_geo:.2%} "
            f"(2023:{net_y['2023']:+.2%} 2024:{net_y['2024']:+.2%} 2025:{net_y['2025']:+.2%}) "
            f"月度口径={net_m_geo:.2%}")
        log(f"      平均持仓五分位权重: "
            + " ".join(f"Q{q}={w_m[f'Q{q}'].mean():.1%}" for q in range(1, 6)))
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "style_attribution.csv", index=False, float_format="%.6f")
    return df


# ---------------------------------------------------------------------------
# B 组 LS 与风格利差（Q1-Q5）相关性/回归
# ---------------------------------------------------------------------------
def ls_style_regression(res_b, qm):
    log("-" * 72)
    log("B组 LS 与风格利差（Q1-Q5 月度）回归检验")
    log("-" * 72)
    rows = []
    x = (qm["Q1"] - qm["Q5"]).rename("Q1-Q5")
    for name, r in res_b.items():
        mon = r["monthly"].set_index("month")["spread"]
        common = mon.index.intersection(x.index)
        yv = mon.loc[common].to_numpy()
        xv = x.loc[common].to_numpy()
        corr = float(np.corrcoef(yv, xv)[0, 1])
        X = np.column_stack([np.ones(len(xv)), xv])
        beta, alpha_m = np.linalg.lstsq(X, yv, rcond=None)[0][1], \
            np.linalg.lstsq(X, yv, rcond=None)[0][0]
        yhat = X @ np.array([alpha_m, beta])
        ss_res = float(((yv - yhat) ** 2).sum())
        ss_tot = float(((yv - yv.mean()) ** 2).sum())
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
        rows.append(dict(variant=name, n_months=len(common),
                         corr_Q1minusQ5=corr, beta_Q1minusQ5=float(beta),
                         alpha_monthly=float(alpha_m),
                         alpha_ann_simple=float((1 + alpha_m) ** 12 - 1),
                         r2=r2))
        log(f"  [{name}] n={len(common)} corr={corr:+.3f} beta={beta:+.3f} "
            f"R²={r2:.3f} 回归截距月alpha={alpha_m:.2%} (≈年化{(1+alpha_m)**12-1:.2%})")
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "ls_style_regression.csv", index=False, float_format="%.6f")
    return df


# ---------------------------------------------------------------------------
# 汇总落盘 + 报告
# ---------------------------------------------------------------------------
def annual_table(nav, bench_m=None):
    nav = nav.copy()
    nav["year"] = nav["trade_date"].str[:4]
    return nav.groupby("year")["daily_ret"].apply(lambda x: (1 + x).prod() - 1)


def main():
    t_all = time.time()
    frozen, proc, panel, sigs, scores = load_all()

    # 全A等权基准月收益（对齐样本外月）
    mkt = panel[["trade_date", "mkt_ret"]].drop_duplicates("trade_date")
    mkt = mkt[(mkt["trade_date"] >= OOS_START) & (mkt["trade_date"] <= OOS_END)]
    mkt_m = mkt.assign(month=mkt["trade_date"].str[:6]).groupby("month")["mkt_ret"] \
        .apply(lambda x: (1 + x).prod() - 1)

    # ---- A 组 ----
    res_a = run_group_a(scores, panel)

    # ---- B 组 ----
    res_b = run_group_b(scores, panel)

    # ---- 五分位月度收益 & 五分位标签 ----
    qm, qlab_map = quintile_monthly_returns(panel, sigs)

    # ---- C 组 ----
    attr_df = style_attribution(res_a, panel, sigs, qlab_map, qm, mkt_m)
    lsreg_df = ls_style_regression(res_b, qm)

    # ---- 汇总对照表 ----
    rows = []
    for name, r in res_a.items():
        m = r["metrics"]
        yr = annual_table(r["daily_nav"])
        att = attr_df[attr_df["variant"] == name].iloc[0]
        rows.append(dict(
            group="A", variant=name,
            ann_return=m["ann_return"], sharpe=m["sharpe"],
            max_drawdown=m["max_drawdown"], monthly_win_rate=m["monthly_win_rate"],
            avg_monthly_one_side_turnover=m["avg_monthly_one_side_turnover"],
            ann_cost=m["ann_cost"], holdings_mcap_ratio=m["holdings_mcap_ratio"],
            ret_2023=float(yr.get("2023", np.nan)),
            ret_2024=float(yr.get("2024", np.nan)),
            ret_2025=float(yr.get("2025", np.nan)),
            excess_IR=att["excess_IR"], net_alpha_ann=att["net_alpha_ann"],
            net_alpha_ann_monthly=att["net_alpha_ann_monthly_geo"],
            delist_count=r["delist_count"],
        ))
    for name, r in res_b.items():
        m = r["metrics"]
        rows.append(dict(
            group="B", variant=name,
            ann_return=m["ann_return"], sharpe=m["sharpe"],
            max_drawdown=m["max_drawdown"], monthly_win_rate=m["monthly_win_rate"],
            avg_monthly_one_side_turnover=np.nan,
            ann_cost=m["ann_cost"], holdings_mcap_ratio=np.nan,
            ret_2023=np.nan, ret_2024=np.nan, ret_2025=np.nan,
            excess_IR=np.nan, net_alpha_ann=np.nan, net_alpha_ann_monthly=np.nan,
            long_ann_return=m["long_ann_return"],
            short_ann_return=m["short_ann_return"],
            delist_count=r["delist_count"],
        ))
    summary = pd.DataFrame(rows)
    summary.to_csv(OUT / "variants_summary.csv", index=False, float_format="%.6f")

    # ---- 净值/月度数据落盘（供绘图）----
    nav_out = None
    for name, r in {**res_a, **res_b}.items():
        tag = name.split("_")[0]
        df = r["daily_nav"][["trade_date", "nav"]].rename(columns={"nav": tag})
        nav_out = df if nav_out is None else nav_out.merge(df, on="trade_date", how="outer")
    nav_out = nav_out.sort_values("trade_date")
    # 基准
    bnav = mkt.assign(nav=(1 + mkt["mkt_ret"]).cumprod())[["trade_date", "nav"]] \
        .rename(columns={"nav": "BENCH"})
    nav_out = nav_out.merge(bnav, on="trade_date", how="left")
    nav_out.to_csv(OUT / "navs_all.csv", index=False, float_format="%.6f")

    # A组分年度超额（策略-基准）
    bench_y = mkt.assign(year=mkt["trade_date"].str[:4]).groupby("year")["mkt_ret"] \
        .apply(lambda x: (1 + x).prod() - 1)
    ex_rows = []
    for name, r in res_a.items():
        yr = annual_table(r["daily_nav"])
        for y in ["2023", "2024", "2025"]:
            ex_rows.append(dict(variant=name.split("_")[0], year=y,
                                excess=float(yr.get(y, np.nan) - bench_y.get(y, np.nan))))
    pd.DataFrame(ex_rows).to_csv(OUT / "annual_excess_A.csv", index=False,
                                 float_format="%.6f")

    # B组月度
    for name, r in res_b.items():
        tag = name.split("_")[0]
        r["monthly"].to_csv(OUT / f"monthly_{tag}.csv", index=False, float_format="%.6f")
        r["daily_nav"].to_csv(OUT / f"nav_{tag}.csv", index=False, float_format="%.6f")

    with open(OUT / "run_meta.json", "w", encoding="utf-8") as fh:
        json.dump(dict(oos_window=f"{OOS_START}~{OOS_END}", top_n=50, cost_rate=0.003,
                       n_signals=len(sigs),
                       baseline_check=dict(A0_ann=float(res_a['A0_基线等权无过滤']['metrics']['ann_return']),
                                           fixed_ann=BASELINE_ANN),
                       elapsed_sec=round(time.time() - t_all, 1)),
                  fh, ensure_ascii=False, indent=2)

    log("=" * 72)
    log("汇总对照表:")
    log(summary.to_string(index=False,
                          float_format=lambda x: f"{x:.4f}"))
    log(f"全部计算完成，耗时 {time.time()-t_all:.0f}s，输出目录: {OUT}")
    log("请运行 make_charts_enhanced.py（托管 python）生成图表，然后看 ENHANCED_RESULTS.md")


if __name__ == "__main__":
    main()

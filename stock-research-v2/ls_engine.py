# -*- coding: utf-8 -*-
"""
ls_engine.py — 多空分层回测引擎 / Long-short layered backtest engine
====================================================================

用途 / Purpose
--------------
检验因子的纯 alpha（剥离市场/风格 beta）：每月信号日按 score 排序，
多头买 top n_long 等权，空头卖 bottom n_short 等权，美元中性。
Test a factor's pure alpha (market/style beta stripped): on each monthly
signal day, go long the top n_long names equal-weight and short the bottom
n_short names equal-weight, dollar-neutral.

⚠️ 重要假设 / Important assumption
---------------------------------
A 股个股不可做空。本引擎为**研究性 alpha 检验**而设，假设可按名义价值
卖空个股（如通过股指期货/融券篮子近似）。结果不代表可直接实盘的收益。
Single-stock shorting is NOT feasible in China's A-share market. This engine
is a RESEARCH-ONLY alpha test assuming stocks can be shorted at notional
value (approximable via index futures / securities-lending baskets).

成交与收益拆分假设（与 engine_v2 完全一致）/ Execution & return-split assumptions
--------------------------------------------------------------------------------
- 信号日收盘出信号 → 次日开盘成交；买入当日收益 = close/open - 1（日内）；
  持有期每日收益 = daily_ret；调仓日隔夜收益 open/pre_close - 1 归旧持仓。
- 过滤：多头端剔除 ST、上市 < min_listed_days、信号日涨停（买不进）；
  空头端剔除 ST、上市 < min_listed_days、信号日跌停（不可卖开）。
- 停牌/缺数据日收益冻结为 0；退市股数据终止后收益为 0（空头端这对组合
  是有利的），多空两侧分别统计 delist 计数。
- 组合日收益 = 多头腿日收益 − hedge_ratio × 空头腿日收益 − 换手成本，
  多空两侧均按实际双边换手 × cost_rate 计费（空头腿换手按 hedge_ratio 缩放名义）。

接口 / API
----------
run_longshort(scores_df, panel, n_long=50, n_short=50, cost_rate=0.003,
              min_listed_days=60, hedge_ratio=1.0, verbose=True) -> dict

返回 dict 键 / returned keys:
    daily_nav, monthly, metrics, ic_series, rebalance_log,
    delist_count, delist_count_long, delist_count_short, delist_detail, assumption

依赖 / Dependencies: pandas, numpy, 同目录 engine_v2（复用其面板预处理与 IC）。
"""

import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from engine_v2 import _prepare_panel, _ic_series, TDY  # noqa: E402

ASSUMPTION_NOTE = (
    "A股个股不可做空；本引擎为研究性 alpha 检验，假设可按名义价值卖空个股"
    "（可用股指期货/融券篮子近似），结果不代表可直接实盘的收益。"
)


# ---------------------------------------------------------------------------
# 多空选股 / long-short selection on one signal day
# ---------------------------------------------------------------------------
def _select_ls(cs, scores_day, sig_date, n_long, n_short,
               min_listed_days, first_date):
    """
    单信号日多空选股 / one signal-day long-short selection.
    共同过滤：非 ST、上市满 min_listed_days 天；
    多头端另剔信号日涨停（买不进），空头端另剔信号日跌停（不可卖开）。
    返回 (long_list, short_list)，多头取 score 最高 n_long，空头取最低 n_short。
    """
    df = scores_day[["ts_code", "score"]].dropna(subset=["score"]).copy()
    if df.empty:
        return [], []
    if cs is not None and len(cs):
        keep = [c for c in ("ts_code", "is_st", "limit_up", "limit_down", "list_date")
                if c in cs.columns]
        df = df.merge(cs[keep], on="ts_code", how="left")

    # 共同过滤 / common filters
    if "is_st" in df.columns:
        df = df[~df["is_st"].fillna(False).astype(bool)]
    if min_listed_days and min_listed_days > 0:
        sig_ts = pd.to_datetime(sig_date, format="%Y%m%d", errors="coerce")
        fb = pd.to_datetime(df["ts_code"].map(first_date), format="%Y%m%d", errors="coerce")
        fb_ok = (sig_ts - fb).dt.days >= min_listed_days
        if "list_date" in df.columns and df["list_date"].notna().any():
            ld = pd.to_datetime(df["list_date"].astype(str), format="%Y%m%d", errors="coerce")
            ok = ((sig_ts - ld).dt.days >= min_listed_days).fillna(fb_ok)
        else:
            ok = fb_ok
        df = df[ok.fillna(True)]

    # 多头端：剔信号日涨停，取 score 最高 n_long / long leg
    dl = df
    if "limit_up" in dl.columns:
        dl = dl[~dl["limit_up"].fillna(False).astype(bool)]
    longs = dl.sort_values("score", ascending=False, kind="mergesort") \
              ["ts_code"].head(n_long).tolist()

    # 空头端：剔信号日跌停（不可卖开），取 score 最低 n_short / short leg
    ds = df
    if "limit_down" in ds.columns:
        ds = ds[~ds["limit_down"].fillna(False).astype(bool)]
    shorts = ds.sort_values("score", ascending=True, kind="mergesort") \
               ["ts_code"].head(n_short).tolist()
    return longs, shorts


# ---------------------------------------------------------------------------
# 主回测 / main long-short backtest
# ---------------------------------------------------------------------------
def run_longshort(scores_df, panel, n_long=50, n_short=50, cost_rate=0.003,
                  min_listed_days=60, hedge_ratio=1.0, verbose=True):
    """
    多空分层回测 / long-short layered backtest.

    Parameters
    ----------
    scores_df : DataFrame[ts_code, trade_date(信号日=每月最后交易日), score(越大越好)]
    panel     : 日频面板 DataFrame（schema 见 engine_v2 docstring）
    n_long / n_short : 多空两侧各持股数 / holdings per side
    cost_rate : 单边费率，双侧按实际换手计费 / one-side cost rate, both legs charged
    min_listed_days : 最小上市天数 / minimum listed days
    hedge_ratio : 空头名义 / 多头名义（多头名义恒为 1.0）/ short-to-long notional
    verbose   : 打印摘要 / print summary

    Returns
    -------
    dict: daily_nav, monthly, metrics, ic_series, rebalance_log,
          delist_count, delist_count_long, delist_count_short,
          delist_detail, assumption
    """
    t0 = time.time()
    env = _prepare_panel(panel)
    cal, dpos, cpos = env["calendar"], env["date_pos"], env["col_pos"]
    O, C, PC, R = env["O"], env["C"], env["PC"], env["R"]
    last_date, first_date, groups, p = (env["last_date"], env["first_date"],
                                        env["groups"], env["p"])
    n_days, n_stocks = len(cal), len(env["stocks"])

    # ---- 打分表预处理 / scores prep ----
    if not {"ts_code", "trade_date", "score"} <= set(scores_df.columns):
        raise ValueError("scores_df 需要 ts_code/trade_date/score 三列")
    s = scores_df.copy()
    s["ts_code"] = s["ts_code"].astype(str)
    s["trade_date"] = s["trade_date"].astype(str).str.strip()
    sig_days = sorted(d for d in s["trade_date"].unique() if d in dpos)
    score_groups = {d: g for d, g in s.groupby("trade_date")}
    if verbose and len(sig_days) < s["trade_date"].nunique():
        print(f"[ls_engine] 警告: {s['trade_date'].nunique() - len(sig_days)} 个信号日不在面板交易日历中，已跳过")

    # ---- 生成调仓事件 / build rebalance events ----
    events = []
    for d in sig_days:
        i = dpos[d] + 1
        if i >= n_days:
            continue  # 最后信号日无下一交易日 / no next trading day
        idx = groups.get(d)
        cs = p.iloc[idx] if idx is not None else None
        longs, shorts = _select_ls(cs, score_groups[d], d, n_long, n_short,
                                   min_listed_days, first_date)
        # 信号日涨跌停标记（列向量）/ signal-day limit flags as column arrays
        lu_flag = np.zeros(n_stocks, dtype=bool)
        ld_flag = np.zeros(n_stocks, dtype=bool)
        if cs is not None:
            if "limit_up" in cs.columns:
                for t, f in zip(cs["ts_code"].to_numpy(), cs["limit_up"].to_numpy()):
                    c = cpos.get(t)
                    if c is not None:
                        try:
                            lu_flag[c] = bool(f)
                        except Exception:
                            pass
            if "limit_down" in cs.columns:
                for t, f in zip(cs["ts_code"].to_numpy(), cs["limit_down"].to_numpy()):
                    c = cpos.get(t)
                    if c is not None:
                        try:
                            ld_flag[c] = bool(f)
                        except Exception:
                            pass
        events.append(dict(exec_i=i, exec_date=cal[i], sig_date=d,
                           longs=longs, shorts=shorts,
                           lu_flag=lu_flag, ld_flag=ld_flag))
    for k, e in enumerate(events):
        e["period_end"] = events[k + 1]["exec_date"] if k + 1 < len(events) else cal[-1]
    if not events:
        raise ValueError("没有可执行的调仓事件（信号日均无下一交易日）")

    # ---- 逐日模拟 / day-by-day simulation (vectorized per day) ----
    ev_by_i = {e["exec_i"]: k for k, e in enumerate(events)}
    wL = np.zeros(n_stocks)   # 多头权重，名义和 = 1.0 / long weights, notional 1.0
    wS = np.zeros(n_stocks)   # 空头权重，内部归一，×hedge_ratio / short weights (unit), x hedge_ratio
    nav = navL = navSh = 1.0
    recs = []                 # (date, nav, day_ret, nav_long, nav_short, long_leg, short_pnl)
    mevents = []              # (month, sellL, buyL, sellS, buyS, cost)
    rebalance_log = []
    delist_seen = set()       # (side, ts_code)
    delist_detail = []

    for i in range(events[0]["exec_i"], n_days):
        d = cal[i]
        k = ev_by_i.get(i)

        # ===================== 普通持有日 / normal holding day =====================
        if k is None:
            r = R[i]
            r = np.where(np.isnan(r), 0.0, r)   # 停牌/缺数据冻结为 0 / frozen at 0
            l_ret = float(wL @ r) if wL.any() else 0.0
            s_ret = float(wS @ r) if wS.any() else 0.0
            den = 1.0 + l_ret
            if wL.any() and den > 0:            # 权重随收益漂移 / drift
                wL = wL * (1.0 + r) / den
            den = 1.0 + s_ret
            if wS.any() and den > 0:
                wS = wS * (1.0 + r) / den
            day_r = l_ret - hedge_ratio * s_ret
            nav *= 1.0 + day_r
            navL *= 1.0 + l_ret
            navSh *= 1.0 - s_ret                # 空头腿 P&L（单位名义）/ short P&L per unit
            recs.append((d, nav, day_r, navL, navSh, l_ret, -s_ret))
            continue

        # ===================== 调仓执行日 / rebalance execution day =====================
        e = events[k]
        o, pc, cl = O[i], PC[i], C[i]
        pc_safe = np.where(np.isfinite(pc) & (pc > 0), pc, 1.0)
        o_safe = np.where(np.isfinite(o) & (o > 0), o, 1.0)
        on_mask = np.isfinite(o) & np.isfinite(pc) & (pc > 0)
        id_mask = np.isfinite(o) & np.isfinite(cl) & (o > 0)
        o_valid = np.isfinite(o) & (o > 0)

        # (a) 隔夜收益归旧持仓 / overnight return (open/pre_close-1) to OLD books
        r_on = np.where(on_mask, o / pc_safe - 1.0, 0.0)
        l_on = float(wL @ r_on) if wL.any() else 0.0
        s_on = float(wS @ r_on) if wS.any() else 0.0
        totL = float((wL * (1.0 + r_on)).sum())
        if wL.any() and totL > 0:
            wL = wL * (1.0 + r_on) / totL
        totS = float((wS * (1.0 + r_on)).sum())
        if wS.any() and totS > 0:
            wS = wS * (1.0 + r_on) / totS

        # (b) 不可交易集合 / untradable (stuck) sets
        # 多头不可卖：信号日跌停 或 执行日无开盘数据（停牌/退市）
        # long cannot-sell: signal-day limit-down or no exec-day open
        stuckL = (e["ld_flag"] | ~o_valid) & (wL > 0)
        # 空头不可平：信号日涨停（买不回） 或 执行日无开盘数据
        # short cannot-cover: signal-day limit-up or no exec-day open
        stuckS = (e["lu_flag"] | ~o_valid) & (wS > 0)

        # (c) 新账本：冻结 stuck + 其余名义等权分给可交易目标
        #     new books: frozen stuck weights + equal weight on tradable targets
        buyableL = [t for t in e["longs"]
                    if cpos.get(t) is not None and o_valid[cpos[t]] and not stuckL[cpos[t]]]
        selopenS = [t for t in e["shorts"]
                    if cpos.get(t) is not None and o_valid[cpos[t]] and not stuckS[cpos[t]]]
        newL = np.zeros(n_stocks)
        newL[stuckL] = wL[stuckL]
        w_stuckL = float(newL.sum())
        if buyableL:
            newL[[cpos[t] for t in buyableL]] = max(0.0, 1.0 - w_stuckL) / len(buyableL)
        newS = np.zeros(n_stocks)
        newS[stuckS] = wS[stuckS]
        w_stuckS = float(newS.sum())
        if selopenS:
            newS[[cpos[t] for t in selopenS]] = max(0.0, 1.0 - w_stuckS) / len(selopenS)

        # (d) 实际双边换手与成本（空头腿名义 ×hedge_ratio）
        #     actual two-sided turnover & cost (short leg scaled by hedge_ratio)
        dwL = newL - wL
        sellL = float(-dwL[dwL < 0].sum())
        buyL = float(dwL[dwL > 0].sum())
        dwS = newS - wS
        sellS = float(-dwS[dwS < 0].sum())
        buyS = float(dwS[dwS > 0].sum())
        cost = cost_rate * (sellL + buyL + hedge_ratio * (sellS + buyS))

        # (e) 日内收益归新账本（开盘成交 close/open-1）/ intraday to NEW books
        r_id = np.where(id_mask, cl / o_safe - 1.0, 0.0)
        l_id = float(newL @ r_id) if newL.any() else 0.0
        s_id = float(newS @ r_id) if newS.any() else 0.0

        # (f) 当日组合收益 = 多头腿 − hedge_ratio×空头腿 − 成本
        day_r = (l_on + l_id) - hedge_ratio * (s_on + s_id) - cost
        day_r = max(day_r, -0.999999)
        nav *= 1.0 + day_r
        navL *= 1.0 + l_on + l_id
        navSh *= 1.0 - (s_on + s_id)
        recs.append((d, nav, day_r, navL, navSh, l_on + l_id, -(s_on + s_id)))

        # 新账本权重漂移 / drift within the new books
        tot2 = float((newL * (1.0 + r_id)).sum())
        wL = newL * (1.0 + r_id) / tot2 if tot2 > 0 else newL
        tot2 = float((newS * (1.0 + r_id)).sum())
        wS = newS * (1.0 + r_id) / tot2 if tot2 > 0 else newS

        mevents.append((d[:6], sellL, buyL, sellS, buyS, cost))
        rebalance_log.append(dict(
            signal_date=e["sig_date"], exec_date=d,
            n_long=len(buyableL) + int(stuckL.sum()),
            n_short=len(selopenS) + int(stuckS.sum()),
            long_notional=float(newL.sum()),
            short_notional=float(newS.sum()) * hedge_ratio,
            sell_long=round(sellL, 6), buy_long=round(buyL, 6),
            sell_short=round(sellS, 6), buy_short=round(buyS, 6),
            cost=round(cost, 8)))

        # (g) 退市统计：持有且数据在本期结束前终止（每侧每股只计一次）
        #     delist count: held stock whose data ends before period end
        for side, w in (("long", wL), ("short", wS)):
            for ci in np.nonzero(w > 0)[0]:
                t = env["stocks"][ci]
                ld = last_date.get(t)
                if ld is not None and d <= ld < e["period_end"] \
                        and (side, t) not in delist_seen:
                    delist_seen.add((side, t))
                    delist_detail.append(dict(ts_code=t, side=side,
                                              signal_date=e["sig_date"],
                                              exec_date=d, last_data_date=ld))

    # ---- 输出组装 / assemble outputs ----
    daily_nav = pd.DataFrame(recs, columns=["trade_date", "nav", "daily_ret",
                                            "nav_long", "nav_short",
                                            "long_ret", "short_ret"])
    daily_nav["month"] = daily_nav["trade_date"].str[:6]

    g = daily_nav.groupby("month")
    monthly = pd.DataFrame(dict(
        long_ret=g["long_ret"].apply(lambda x: (1.0 + x).prod() - 1.0),
        short_ret=g["short_ret"].apply(lambda x: (1.0 + x).prod() - 1.0),
        spread=g["daily_ret"].apply(lambda x: (1.0 + x).prod() - 1.0),
    ))
    if mevents:
        mto = pd.DataFrame(mevents, columns=["month", "sell_long", "buy_long",
                                             "sell_short", "buy_short", "cost"])
        mto = mto.groupby("month").sum()
        monthly = monthly.join(mto)
    for c in ("sell_long", "buy_long", "sell_short", "buy_short", "cost"):
        monthly[c] = monthly[c].fillna(0.0)
    monthly["turnover_long"] = monthly["sell_long"] + monthly["buy_long"]
    monthly["turnover_short"] = hedge_ratio * (monthly["sell_short"] + monthly["buy_short"])
    monthly["turnover"] = monthly["turnover_long"] + monthly["turnover_short"]
    monthly = monthly.reset_index()[["month", "long_ret", "short_ret", "spread",
                                     "turnover_long", "turnover_short",
                                     "turnover", "cost"]]

    ic_series = _ic_series(score_groups, sig_days, env)

    r = daily_nav["daily_ret"].to_numpy()
    n = len(r)
    years = n / TDY if n else np.nan
    rstd = r.std(ddof=1) if n > 1 else 0.0
    cummax = daily_nav["nav"].cummax()
    delist_long = sum(1 for x in delist_detail if x["side"] == "long")
    delist_short = sum(1 for x in delist_detail if x["side"] == "short")
    metrics = dict(
        total_return=float(daily_nav["nav"].iloc[-1] - 1.0) if n else np.nan,
        ann_return=float(daily_nav["nav"].iloc[-1] ** (TDY / n) - 1.0) if n else np.nan,
        sharpe=float(r.mean() / rstd * np.sqrt(TDY)) if rstd > 0 else np.nan,
        max_drawdown=float((daily_nav["nav"] / cummax - 1.0).min()) if n else np.nan,
        monthly_win_rate=float((monthly["spread"] > 0).mean()) if len(monthly) else np.nan,
        ann_turnover=float(monthly["turnover"].sum() / years) if years else np.nan,
        ann_cost=float(monthly["cost"].sum() / years) if years else np.nan,
        long_ann_return=float(daily_nav["nav_long"].iloc[-1] ** (TDY / n) - 1.0) if n else np.nan,
        short_ann_return=float(daily_nav["nav_short"].iloc[-1] ** (TDY / n) - 1.0) if n else np.nan,
        hedge_ratio=float(hedge_ratio),
        n_days=n, n_rebalances=len(events),
        delist_count_long=delist_long, delist_count_short=delist_short,
        note=ASSUMPTION_NOTE,
    )

    if verbose:
        print(f"[ls_engine] 回测完成 / backtest done: {len(events)} 次调仓, "
              f"{n} 个交易日, 耗时 {time.time() - t0:.1f}s")
        print(f"  ⚠️ {ASSUMPTION_NOTE}")
        print(f"  年化收益 ann_return      = {metrics['ann_return']:.2%}")
        print(f"  夏普 sharpe             = {metrics['sharpe']:.3f}")
        print(f"  最大回撤 max_dd          = {metrics['max_drawdown']:.2%}")
        print(f"  月胜率 win_rate          = {metrics['monthly_win_rate']:.2%}")
        print(f"  多头腿年化 long_ann      = {metrics['long_ann_return']:.2%}")
        print(f"  空头腿年化 short_ann     = {metrics['short_ann_return']:.2%} (单位名义 P&L)")
        print(f"  年化换手 ann_turnover    = {metrics['ann_turnover']:.2f}x (双边/two-sided)")
        print(f"  年化成本 ann_cost        = {metrics['ann_cost']:.2%}")
        print(f"  退市计数 delist long/short = {delist_long}/{delist_short}")
        ic_mean = ic_series["ic"].mean() if len(ic_series) else np.nan
        print(f"  月均 Rank IC             = {ic_mean:.4f} (n={ic_series['ic'].notna().sum()})")

    return dict(daily_nav=daily_nav[["trade_date", "nav", "daily_ret",
                                     "nav_long", "nav_short", "long_ret", "short_ret"]],
                monthly=monthly, metrics=metrics, ic_series=ic_series,
                rebalance_log=pd.DataFrame(rebalance_log),
                delist_count=len(delist_detail),
                delist_count_long=delist_long, delist_count_short=delist_short,
                delist_detail=delist_detail, assumption=ASSUMPTION_NOTE)


# ---------------------------------------------------------------------------
# 自测 / self-test: synthetic panel (300 stocks x 500 days) + trending scores
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 70)
    print("ls_engine 自测 / self-test on synthetic panel (300 stocks x 500 days)")
    print("=" * 70)
    t_start = time.time()
    rng = np.random.default_rng(11)
    n_days, n_stocks = 500, 300
    dates = pd.bdate_range("2023-01-02", periods=n_days).strftime("%Y%m%d").tolist()
    stocks = [f"{i:06d}.SZ" for i in range(1, n_stocks + 1)]
    inds = [f"IND{k:02d}" for k in range(10)]

    # 退市股：一只进多头、一只进空头 / delisted: one for long leg, one for short leg
    LONG_DL, SHORT_DL = stocks[120], stocks[199]
    DELIST = {LONG_DL: dates[250], SHORT_DL: dates[380]}

    # 信号日 = 每月最后一个交易日 / signal day = last trading day of each month
    sig_days = [dates[i] for i in range(len(dates) - 1)
                if dates[i][:6] != dates[i + 1][:6]] + [dates[-1]]
    sig_set = set(sig_days)
    n_months = len(sig_days)
    print(f"合成日历 / synthetic calendar: {dates[0]}..{dates[-1]}, {n_months} 个信号日")

    # 每只股票的常驻漂移 mu（趋势源，使 score 有预测力）
    # persistent per-stock drift mu (the trend the score proxies)
    mu = rng.normal(0.0003, 0.0009, n_stocks)

    # ---- 合成面板 / build synthetic panel ----
    frames = []
    for si, ts in enumerate(stocks):
        last = DELIST.get(ts, dates[-1])
        nd = dates.index(last) + 1
        d_ts = dates[:nd]
        rets = np.clip(rng.normal(mu[si], 0.02, nd), -0.089, 0.089)
        close = 20.0 * np.exp(np.cumsum(rets))
        pre_close = np.concatenate([[20.0], close[:-1]])
        open_ = pre_close * (1.0 + rng.normal(0.0, 0.004, nd))
        daily_ret = close / pre_close - 1.0
        frames.append(pd.DataFrame(dict(
            ts_code=ts, trade_date=d_ts, open=open_, close=close,
            pre_close=pre_close, daily_ret=daily_ret,
            circ_mv=rng.uniform(2e9, 5e10), industry=inds[si % 10],
            is_st=(si % 50 == 0),
            limit_up=daily_ret >= 0.095, limit_down=daily_ret <= -0.095,
            list_date="20150101", mkt_ret=0.0003)))
    panel = pd.concat(frames, ignore_index=True)

    # 随机停牌 drop ~0.5% 行 / random suspensions
    drop_mask = (rng.random(len(panel)) < 0.005) & (~panel["ts_code"].isin(DELIST))
    panel = panel[~drop_mask].reset_index(drop=True)

    # ---- 趋势性打分 / trending scores: score = 常驻漂移 + 噪声 ----
    scores = pd.concat([pd.DataFrame(dict(
        ts_code=stocks, trade_date=d,
        score=mu + rng.normal(0.0, 0.0004, n_stocks)))
        for d in sig_days], ignore_index=True)
    # 保证退市股消失前分别进入多头/空头 / force delisted stocks into the books
    for ts, ld_, sc in ((LONG_DL, DELIST[LONG_DL], 1e6),
                        (SHORT_DL, DELIST[SHORT_DL], -1e6)):
        prev = [d for d in sig_days if d < ld_][-2:]
        m = (scores["ts_code"] == ts) & scores["trade_date"].isin(prev)
        scores.loc[m, "score"] = sc
        # 确保这些信号日不被涨跌停过滤挡住 / keep them tradable on those days
        pm = (panel["ts_code"] == ts) & panel["trade_date"].isin(prev)
        panel.loc[pm, "limit_up"] = False
        panel.loc[pm, "limit_down"] = False

    # ---- 跑回测 / run backtest ----
    res = run_longshort(scores, panel, n_long=30, n_short=30,
                        cost_rate=0.003, min_listed_days=60, verbose=True)

    nav_df, monthly, ics, met, rb = (res["daily_nav"], res["monthly"],
                                     res["ic_series"], res["metrics"],
                                     res["rebalance_log"])

    # 1) nav 无 NaN / no NaN in nav
    assert nav_df[["nav", "daily_ret", "nav_long", "nav_short"]].notna().all().all(), "nav 存在 NaN"
    assert np.isfinite(nav_df["nav"]).all(), "nav 存在 inf"
    assert nav_df["trade_date"].is_monotonic_increasing
    print(f"[PASS] nav 无 NaN/inf, 长度={len(nav_df)}, 日期单调递增")

    # 2) 美元中性：每次调仓后多头名义 == 空头名义（hedge_ratio=1）
    #    dollar-neutral: long notional == short notional at every rebalance
    assert np.allclose(rb["long_notional"], 1.0, atol=1e-9), "多头名义 != 1.0"
    assert np.allclose(rb["short_notional"], 1.0, atol=1e-9), "空头名义 != 1.0"
    assert np.allclose(rb["long_notional"], rb["short_notional"], atol=1e-9)
    res_h = run_longshort(scores, panel, n_long=30, n_short=30, cost_rate=0.003,
                          min_listed_days=60, hedge_ratio=0.5, verbose=False)
    rb_h = res_h["rebalance_log"]
    assert np.allclose(rb_h["short_notional"], 0.5 * rb_h["long_notional"], atol=1e-9)
    print(f"[PASS] 美元中性: hedge=1 时多空名义均为 1.0；hedge=0.5 时空头名义精确减半 "
          f"(max|Δ|={np.abs(rb_h['short_notional'] - 0.5 * rb_h['long_notional']).max():.2e})")

    # 3) 成本 > 0 且随 cost_rate 同比放大 / cost > 0 and scales with cost_rate
    assert (monthly["turnover"] > 0).any() and (monthly["cost"] > 0).any()
    res2 = run_longshort(scores, panel, n_long=30, n_short=30, cost_rate=0.006,
                         min_listed_days=60, verbose=False)
    assert np.allclose(res2["monthly"]["turnover"], monthly["turnover"]), "费率改变了换手？"
    assert np.allclose(res2["monthly"]["cost"], 2.0 * monthly["cost"]), "成本未随费率同比放大"
    print(f"[PASS] 成本随 cost_rate 加倍精确加倍, 平均月度双边换手={monthly['turnover'].mean():.3f}, "
          f"年化成本={met['ann_cost']:.2%}")

    # 4) IC 长度 = 月数 - 1 / IC length == months - 1
    assert len(ics) == n_months - 1, f"IC 长度 {len(ics)} != {n_months - 1}"
    assert ics["ic"].mean() > 0, "趋势性 score 应产生正 IC"
    print(f"[PASS] IC 序列长度={len(ics)} == 月数-1={n_months - 1}, "
          f"平均 IC={ics['ic'].mean():.4f} (趋势性 score, 应为正)")

    # 5) 多空两侧退市均被统计 / delist counted on both sides
    dd = {(x["side"], x["ts_code"]) for x in res["delist_detail"]}
    assert ("long", LONG_DL) in dd, f"多头退市股未统计: {dd}"
    assert ("short", SHORT_DL) in dd, f"空头退市股未统计: {dd}"
    print(f"[PASS] 退市统计: long={res['delist_count_long']}, short={res['delist_count_short']}, "
          f"明细={sorted(dd)}")

    # 6) 过滤生效：无 ST 入仓 / filters effective: no ST in either book
    st_set = {t for i, t in enumerate(stocks) if i % 50 == 0}
    held_codes = set()
    # 从面板截面复核：打分最高的 ST 不应出现在任何调仓名义里（抽查 rebalance 日志长度即可，
    # 权重未落盘，这里用 notional 和构造逻辑保证；另直接验证 ST 未入选）
    # re-derive selections to confirm no ST passed the filter
    from ls_engine import _select_ls  # noqa: E402  (self-import is a no-op at runtime)
    grp = {d: g for d, g in scores.groupby("trade_date")}
    pan_grp = panel.groupby("trade_date").indices
    for d in sig_days[:5]:
        cs = panel.iloc[pan_grp[d]]
        lg, sh = _select_ls(cs, grp[d], d, 30, 30, 60,
                            panel.groupby("ts_code")["trade_date"].min().to_dict())
        assert not (set(lg) & st_set) and not (set(sh) & st_set), "ST 股入选"
    print("[PASS] 过滤生效: 抽样 5 个信号日，多空两侧均无 ST 股入选")

    # 7) metrics 完备 / metrics complete
    for k in ("ann_return", "sharpe", "max_drawdown", "monthly_win_rate",
              "ann_cost", "note"):
        assert k in met, f"metrics 缺少 {k}"
    assert "不可做空" in met["note"]
    print(f"[PASS] metrics 完备: ann={met['ann_return']:.2%}, sharpe={met['sharpe']:.3f}, "
          f"max_dd={met['max_drawdown']:.2%}, 做空假设已注明")

    print("=" * 70)
    print(f"全部自测通过 / ALL SELF-TESTS PASSED  (总耗时 {time.time() - t_start:.1f}s)")
    print("=" * 70)

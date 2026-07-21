# -*- coding: utf-8 -*-
"""
engine_v2.py — 修复版 A 股截面选股回测引擎 / Fixed cross-sectional backtest engine (v2)
=======================================================================================

相对原引擎修复的四个致命错误 / Four fatal flaws of the original engine fixed here:

1. 日净值不再由月收益线性插值伪造 —— 逐日真实模拟：开盘成交、隔夜/日内收益拆分，
   夏普与最大回撤全部基于真实日收益/日净值。
   Daily NAV is truly simulated day by day (open execution, overnight/intraday split),
   never linearly interpolated from monthly returns.
2. 交易成本按每次调仓的实际双边换手扣除：cost = cost_rate × (卖出换手 + 买入换手)，
   与换手严格挂钩，而不是每月固定扣一次单边 0.3%。
   Cost is charged per rebalance on actual two-sided turnover.
3. 持有期内退市/停牌显式处理：缺数据日收益按 0（净值冻结）；整个持有期再无数据（退市）
   剩余按 0 并计入 delist_count，不再静默置零造成幸存者偏差。
   Suspension/delisting handled explicitly and counted, no silent survivorship bias.
4. 持有期 = 信号日次日开盘 → 下一次调仓开盘，与月度调仓严格对齐，
   不再使用固定 20 日收益造成错位重叠。
   Holding period aligned to the rebalance calendar (no fixed-20-day overlap).

接口 / API
----------
run_backtest(scores_df, panel, top_n=50, cost_rate=0.003,
             max_per_industry=3, min_listed_days=60, verbose=True) -> dict

panel schema: ts_code(str), trade_date(str 'YYYYMMDD'), open, close, pre_close,
              daily_ret(复权日收益), amount, circ_mv, turnover_rate, industry(str),
              is_st(bool), limit_up, limit_down, list_date, mkt_ret ...
scores schema: ts_code, trade_date(信号日=每月最后交易日), score(越大越好)

依赖 / Dependencies: pandas, numpy（读 parquet 请在调用方完成，本引擎只吃 DataFrame）。
"""

import numpy as np
import pandas as pd

TDY = 252  # 年化用的交易日数 / trading days per year for annualization


# ---------------------------------------------------------------------------
# 面板预处理 / Panel preparation: validate, clean, and pre-index to wide matrices
# ---------------------------------------------------------------------------
def _prepare_panel(panel):
    """
    校验面板并预索引为宽表 numpy 矩阵（行=交易日, 列=股票），供向量化取用。
    Validate the daily panel and pre-index it into wide numpy matrices
    (rows = trading days, cols = stocks) for fast vectorized access.
    """
    required = {"ts_code", "trade_date", "close"}
    missing = required - set(panel.columns)
    if missing:
        raise ValueError(f"panel 缺少必要列 / panel missing required columns: {missing}")

    p = panel.copy()
    p["ts_code"] = p["ts_code"].astype(str)
    p["trade_date"] = p["trade_date"].astype(str).str.strip()
    # 去重，保证 pivot 不炸 / de-duplicate so pivot is safe
    p = p.drop_duplicates(subset=["ts_code", "trade_date"], keep="last")
    p = p.reset_index(drop=True)

    # open / pre_close / daily_ret 互补推导，缺列容错 / tolerant derivation
    if "daily_ret" not in p.columns:
        if "pre_close" in p.columns:
            p["daily_ret"] = p["close"] / p["pre_close"] - 1.0
        else:
            p = p.sort_values(["ts_code", "trade_date"])
            p["daily_ret"] = p.groupby("ts_code")["close"].pct_change()
    if "pre_close" not in p.columns:
        p["pre_close"] = p["close"] / (1.0 + p["daily_ret"])
    if "open" not in p.columns:
        # 退化假设：无隔夜跳空 / degenerate fallback: no overnight gap
        p["open"] = p["pre_close"]

    calendar = np.sort(p["trade_date"].unique())
    stocks = np.sort(p["ts_code"].unique())
    date_pos = {d: i for i, d in enumerate(calendar)}
    col_pos = {t: i for i, t in enumerate(stocks)}

    def wide(col):
        w = p.pivot(index="trade_date", columns="ts_code", values=col)
        w = w.reindex(index=calendar, columns=stocks)
        return w.to_numpy(dtype=np.float64)

    O = wide("open")
    C = wide("close")
    PC = wide("pre_close")
    R = wide("daily_ret")

    # 每只股票首/末数据日：用于退市判定与上市天数回退 / per-stock first & last data date
    last_date = p.groupby("ts_code")["trade_date"].max().to_dict()
    first_date = p.groupby("ts_code")["trade_date"].min().to_dict()
    # trade_date -> 行位置，用于快速取信号日截面 / date -> row positions for cross-sections
    groups = p.groupby("trade_date").indices

    return dict(p=p, calendar=calendar, stocks=stocks, date_pos=date_pos,
                col_pos=col_pos, O=O, C=C, PC=PC, R=R,
                last_date=last_date, first_date=first_date, groups=groups)


# ---------------------------------------------------------------------------
# 选股 / Signal-day stock selection with all filters
# ---------------------------------------------------------------------------
def _select_targets(cs, scores_day, sig_date, top_n, max_per_industry,
                    min_listed_days, first_date):
    """
    单个信号日选股 / one signal-day selection:
    - 剔除 is_st、信号日 limit_up（买不进）、上市不足 min_listed_days 天
    - 每个行业最多 max_per_industry 只（按 score 从高到低贪心）
    - 取 top_n，等权（等权在调仓执行时分配）
    cs: 信号日面板截面 DataFrame（可为 None）；scores_day: 当日打分 DataFrame。
    """
    df = scores_day[["ts_code", "score"]].dropna(subset=["score"]).copy()
    if df.empty:
        return []
    if cs is not None and len(cs):
        keep = [c for c in ("ts_code", "is_st", "limit_up", "industry", "list_date")
                if c in cs.columns]
        df = df.merge(cs[keep], on="ts_code", how="left")

    # 1) 剔除 ST / drop ST stocks
    if "is_st" in df.columns:
        df = df[~df["is_st"].fillna(False).astype(bool)]
    # 2) 信号日涨停买不进 / limit-up on signal day -> cannot buy
    if "limit_up" in df.columns:
        df = df[~df["limit_up"].fillna(False).astype(bool)]
    # 3) 上市天数 / min listed days (list_date 优先，缺失时回退到面板内首次出现日)
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

    # 4) score 降序 + 行业数量上限（贪心）/ sort by score desc, greedy per-industry cap
    df = df.sort_values("score", ascending=False, kind="mergesort")
    if "industry" in df.columns:
        inds = df["industry"].fillna("NA").astype(str).to_numpy()
    else:
        inds = np.repeat("NA", len(df))
    picked, cnt = [], {}
    for t, g in zip(df["ts_code"].to_numpy(), inds):
        if cnt.get(g, 0) >= max_per_industry:
            continue
        picked.append(t)
        cnt[g] = cnt.get(g, 0) + 1
        if len(picked) >= top_n:
            break
    return picked


# ---------------------------------------------------------------------------
# IC 序列 / monthly Rank IC of score vs realized forward return
# ---------------------------------------------------------------------------
def _ic_series(score_groups, sig_days, env):
    """
    每月信号日 score 与 [信号日收盘 → 下一信号日收盘] 个股实际收益的 Spearman Rank IC。
    远期收益用 daily_ret 复利累计（缺失日按 0，与引擎的冻结约定一致）。
    Spearman Rank IC between signal-day score and the stock's realized return from
    signal-day close to next signal-day close (compounded daily_ret, missing days = 0).
    """
    R, dpos, stocks = env["R"], env["date_pos"], env["stocks"]
    rows = []
    for a, b in zip(sig_days[:-1], sig_days[1:]):
        ia, ib = dpos[a], dpos[b]
        window = R[ia + 1: ib + 1, :]                     # (sig, next_sig]
        w = np.where(np.isnan(window), 0.0, window)       # 缺失日冻结为 0
        w = np.clip(w, -0.999999, None)                   # 防 log1p 越界
        fwd = np.expm1(np.sum(np.log1p(w), axis=0))       # 复利远期收益
        fwd_s = pd.Series(fwd, index=stocks)
        sc = score_groups[a].set_index("ts_code")["score"]
        fr = fwd_s.reindex(sc.index)
        m = sc.notna() & fr.notna()
        # Spearman = Pearson on average ranks; 手写避免依赖 scipy（本机 scipy DLL 损坏）
        # manual rank-based Spearman to avoid the scipy dependency (broken DLLs here)
        ic = sc[m].rank().corr(fr[m].rank()) if int(m.sum()) >= 5 else np.nan
        rows.append(dict(signal_date=a, next_signal_date=b, ic=ic, n=int(m.sum())))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 主回测函数 / main backtest
# ---------------------------------------------------------------------------
def run_backtest(scores_df, panel, top_n=50, cost_rate=0.003,
                 max_per_industry=3, min_listed_days=60, verbose=True):
    """
    修复版截面回测 / fixed cross-sectional backtest.

    Parameters
    ----------
    scores_df : DataFrame[ts_code, trade_date(信号日), score]
    panel     : 日频面板 DataFrame（见模块 docstring 的 schema）
    top_n     : 每期持股数 / number of holdings per period
    cost_rate : 单边费率 / one-side cost rate (双边按 sell+buy 换手计)
    max_per_industry : 每行业最大持仓数 / per-industry holding cap
    min_listed_days  : 最小上市天数 / minimum listed days
    verbose   : 打印摘要 / print summary

    Returns
    -------
    dict: daily_nav, monthly, ic_series, metrics, delist_count, delist_detail,
          holdings_log
    """
    env = _prepare_panel(panel)
    cal, dpos, cpos = env["calendar"], env["date_pos"], env["col_pos"]
    O, C, PC, R = env["O"], env["C"], env["PC"], env["R"]
    last_date, first_date, groups, p = (env["last_date"], env["first_date"],
                                        env["groups"], env["p"])
    n_days = len(cal)

    # ---- 打分表预处理 / scores prep ----
    if not {"ts_code", "trade_date", "score"} <= set(scores_df.columns):
        raise ValueError("scores_df 需要 ts_code/trade_date/score 三列")
    s = scores_df.copy()
    s["ts_code"] = s["ts_code"].astype(str)
    s["trade_date"] = s["trade_date"].astype(str).str.strip()
    sig_days = sorted(d for d in s["trade_date"].unique() if d in dpos)
    score_groups = {d: g for d, g in s.groupby("trade_date")}
    if verbose and len(sig_days) < s["trade_date"].nunique():
        print(f"[engine_v2] 警告: {s['trade_date'].nunique() - len(sig_days)} 个信号日不在面板交易日历中，已跳过")

    # ---- 生成调仓事件 / build rebalance events: signal close -> next day open ----
    events = []
    for d in sig_days:
        i = dpos[d] + 1
        if i >= n_days:
            continue  # 最后信号日无下一交易日，无法执行 / no next trading day
        idx = groups.get(d)
        cs = p.iloc[idx] if idx is not None else None
        targets = _select_targets(cs, score_groups[d], d, top_n,
                                  max_per_industry, min_listed_days, first_date)
        ld_map = None
        if cs is not None and "limit_down" in cs.columns:
            ld_map = cs.set_index("ts_code")["limit_down"].to_dict()
        events.append(dict(exec_i=i, exec_date=cal[i], sig_date=d,
                           targets=targets, ld_map=ld_map))
    for k, e in enumerate(events):
        # 持有期结束日 = 下次调仓执行日（含），最后一期到数据末尾 / holding-period end
        e["period_end"] = events[k + 1]["exec_date"] if k + 1 < len(events) else cal[-1]

    if not events:
        raise ValueError("没有可执行的调仓事件（信号日均无下一交易日）")

    # ---- 逐日模拟 / day-by-day simulation ----
    ev_by_i = {e["exec_i"]: k for k, e in enumerate(events)}
    pos = {}                    # ts_code -> 组合权重 / portfolio weight
    nav = 1.0
    recs = []                   # (date, nav, daily_ret)
    mevents = []                # (month, sell_turnover, buy_turnover, cost)
    holdings_log = []
    delist_count = 0
    delist_seen = set()
    delist_detail = []

    for i in range(events[0]["exec_i"], n_days):
        d = cal[i]
        k = ev_by_i.get(i)

        # ===================== 普通持有日 / normal holding day =====================
        if k is None:
            if pos:
                ts_list = list(pos.keys())
                cols = [cpos[t] for t in ts_list]
                r = R[i, cols]
                # 停牌/缺数据日收益冻结为 0 / suspension or missing day -> frozen at 0
                r = np.where(np.isnan(r), 0.0, r)
                w = np.fromiter((pos[t] for t in ts_list), dtype=np.float64)
                pr = float(w @ r)
                pr = max(pr, -0.999999)                     # 数值保护 / numeric guard
                wn = w * (1.0 + r) / (1.0 + pr)             # 权重随收益漂移 / drift
                pos = dict(zip(ts_list, wn))
            else:
                pr = 0.0
            nav *= 1.0 + pr
            recs.append((d, nav, pr))
            continue

        # ===================== 调仓执行日 / rebalance execution day =====================
        e = events[k]
        # (a) 隔夜收益归旧持仓 / overnight return (open/pre_close-1) to OLD holdings
        old_drift, r_on_port = {}, 0.0
        if pos:
            for t, w in pos.items():
                o, pc = O[i, cpos[t]], PC[i, cpos[t]]
                r_on = 0.0 if (not np.isfinite(o) or not np.isfinite(pc) or pc <= 0) \
                    else o / pc - 1.0
                old_drift[t] = (w, r_on)
            tot = sum(w * (1.0 + r) for w, r in old_drift.values())
            tot = tot if tot > 0 else 1.0
            r_on_port = sum(w * r for w, r in old_drift.values())
            old_drift = {t: w * (1.0 + r) / tot for t, (w, r) in old_drift.items()}

        # (b) 不可卖集合：信号日跌停 或 执行日无数据（停牌/已退市）/ cannot-sell set
        stuck = set()
        for t in old_drift:
            flag = e["ld_map"].get(t, False) if e["ld_map"] else False
            try:
                if bool(flag):
                    stuck.add(t)
                    continue
            except Exception:
                pass
            o = O[i, cpos[t]]
            if not np.isfinite(o) or o <= 0:
                stuck.add(t)

        # (c) 新持仓：冻结 stuck（权重不变）+ 可买标的等权分配剩余权重
        #     new book: frozen stuck weights + equal weight on tradable buys
        buyable = []
        for t in e["targets"]:
            if t in stuck:
                continue
            c = cpos.get(t)
            if c is None:
                continue
            o = O[i, c]
            if not np.isfinite(o) or o <= 0:
                continue  # 执行日无数据/停牌，买不进 / not tradable at exec open
            buyable.append(t)
        w_stuck = sum(old_drift[t] for t in stuck)
        w_each = max(0.0, 1.0 - w_stuck) / len(buyable) if buyable else 0.0
        new_pos = {t: old_drift[t] for t in stuck}
        for t in buyable:
            new_pos[t] = w_each

        # (d) 实际双边换手与成本 / actual two-sided turnover and cost
        sell_to = buy_to = 0.0
        for t in set(old_drift) | set(new_pos):
            dw = new_pos.get(t, 0.0) - old_drift.get(t, 0.0)
            if dw > 0:
                buy_to += dw
            else:
                sell_to -= dw
        cost = cost_rate * (sell_to + buy_to)

        # (e) 日内收益归新持仓（开盘成交 close/open-1）/ intraday return to NEW book
        rid = {}
        for t in new_pos:
            c = cpos[t]
            o, cl = O[i, c], C[i, c]
            rid[t] = 0.0 if (not np.isfinite(o) or not np.isfinite(cl) or o <= 0) \
                else cl / o - 1.0
        r_id_port = sum(w * rid[t] for t, w in new_pos.items())

        # (f) 当日组合收益 = 隔夜(旧) + 日内(新) - 成本 / day return, cost deducted
        day_r = r_on_port + r_id_port - cost
        day_r = max(day_r, -0.999999)
        nav *= 1.0 + day_r
        recs.append((d, nav, day_r))

        # 权重漂移 / weight drift within the new book
        tot2 = sum(w * (1.0 + rid[t]) for t, w in new_pos.items())
        pos = ({t: w * (1.0 + rid[t]) / tot2 for t, w in new_pos.items()}
               if tot2 > 0 else dict(new_pos))

        mevents.append((d[:6], sell_to, buy_to, cost))
        holdings_log.append(dict(
            signal_date=e["sig_date"], exec_date=d, n_holdings=len(pos),
            holdings=sorted(pos.keys()), weights={t: round(w, 6) for t, w in pos.items()},
            stuck=sorted(stuck), sell_turnover=round(sell_to, 6),
            buy_turnover=round(buy_to, 6), cost=round(cost, 8)))

        # (g) 退市统计：本期持有且数据在期末前终止（每只股票只计一次）
        #     delist count: held stock whose data ends before period end (once per stock)
        for t in pos:
            ld = last_date.get(t)
            if ld is not None and d <= ld < e["period_end"] and t not in delist_seen:
                delist_seen.add(t)
                delist_count += 1
                delist_detail.append(dict(ts_code=t, signal_date=e["sig_date"],
                                          exec_date=d, last_data_date=ld))

    # ---- 输出组装 / assemble outputs ----
    daily_nav = pd.DataFrame(recs, columns=["trade_date", "nav", "daily_ret"])
    daily_nav["month"] = daily_nav["trade_date"].str[:6]

    mret = daily_nav.groupby("month")["daily_ret"].apply(lambda x: (1.0 + x).prod() - 1.0)
    monthly = mret.to_frame("ret")
    if mevents:
        mto = pd.DataFrame(mevents, columns=["month", "sell_turnover", "buy_turnover", "cost"])
        monthly = monthly.join(mto.groupby("month").sum())
    monthly[["sell_turnover", "buy_turnover", "cost"]] = \
        monthly[["sell_turnover", "buy_turnover", "cost"]].fillna(0.0)
    monthly["turnover"] = monthly["sell_turnover"] + monthly["buy_turnover"]
    monthly = monthly.reset_index()

    ic_series = _ic_series(score_groups, sig_days, env)

    r = daily_nav["daily_ret"].to_numpy()
    n = len(r)
    years = n / TDY if n else np.nan
    rstd = r.std(ddof=1) if n > 1 else 0.0
    cummax = daily_nav["nav"].cummax()
    metrics = dict(
        total_return=float(daily_nav["nav"].iloc[-1] - 1.0) if n else np.nan,
        ann_return=float(daily_nav["nav"].iloc[-1] ** (TDY / n) - 1.0) if n else np.nan,
        sharpe=float(r.mean() / rstd * np.sqrt(TDY)) if rstd > 0 else np.nan,
        max_drawdown=float((daily_nav["nav"] / cummax - 1.0).min()) if n else np.nan,
        monthly_win_rate=float((monthly["ret"] > 0).mean()) if len(monthly) else np.nan,
        ann_turnover=float(monthly["turnover"].sum() / years) if years else np.nan,
        ann_cost=float(monthly["cost"].sum() / years) if years else np.nan,
        n_days=n, n_rebalances=len(events),
    )

    if verbose:
        print("[engine_v2] 回测完成 / backtest done: "
              f"{metrics['n_rebalances']} 次调仓, {n} 个交易日")
        print(f"  年化收益 ann_return   = {metrics['ann_return']:.2%}")
        print(f"  夏普 sharpe          = {metrics['sharpe']:.3f}")
        print(f"  最大回撤 max_dd       = {metrics['max_drawdown']:.2%}")
        print(f"  月胜率 win_rate       = {metrics['monthly_win_rate']:.2%}")
        print(f"  年化换手 ann_turnover = {metrics['ann_turnover']:.2f}x (双边/two-sided)")
        print(f"  年化成本 ann_cost     = {metrics['ann_cost']:.2%}")
        print(f"  退市计数 delist_count = {delist_count}")
        ic_mean = ic_series["ic"].mean() if len(ic_series) else np.nan
        print(f"  月均 Rank IC          = {ic_mean:.4f} (n={ic_series['ic'].notna().sum()})")

    return dict(daily_nav=daily_nav[["trade_date", "nav", "daily_ret"]],
                monthly=monthly, ic_series=ic_series, metrics=metrics,
                delist_count=delist_count, delist_detail=delist_detail,
                holdings_log=holdings_log)


# ---------------------------------------------------------------------------
# 自测 / self-test with synthetic data (300 stocks x 500 days, 2 delisted mid-way)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 70)
    print("engine_v2 自测 / self-test on synthetic panel (300 stocks x 500 days)")
    print("=" * 70)
    rng = np.random.default_rng(7)
    n_days, n_stocks = 500, 300
    dates = pd.bdate_range("2023-01-02", periods=n_days).strftime("%Y%m%d").tolist()
    stocks = [f"{i:06d}.SZ" for i in range(1, n_stocks + 1)]
    inds = [f"IND{k:02d}" for k in range(10)]

    # 两只真实存在的股票中途消失（退市）/ two existing stocks vanish mid-way (delisted)
    DELIST = {stocks[120]: dates[250], stocks[199]: dates[380]}
    STUCK_TS = "000009.SZ"    # 强制跌停不可卖 / forced limit-down (cannot sell)
    LIMU_TS = "000005.SZ"     # 信号日常驻涨停 / limit-up on every signal day

    # 信号日 = 每月最后一个交易日 / signal day = last trading day of each month
    sig_days = [dates[i] for i in range(len(dates) - 1)
                if dates[i][:6] != dates[i + 1][:6]] + [dates[-1]]
    sig_set = set(sig_days)
    n_months = len(sig_days)
    print(f"合成日历 / synthetic calendar: {dates[0]}..{dates[-1]}, {n_months} 个信号日")

    # ---- 合成面板 / build synthetic panel ----
    frames = []
    for si, ts in enumerate(stocks):
        last = DELIST.get(ts, dates[-1])
        nd = dates.index(last) + 1
        d_ts = dates[:nd]
        rets = np.clip(rng.normal(0.0003, 0.022, nd), -0.089, 0.089)
        close = 20.0 * np.exp(np.cumsum(rets))
        pre_close = np.concatenate([[20.0], close[:-1]])
        open_ = pre_close * (1.0 + rng.normal(0.0, 0.004, nd))
        daily_ret = close / pre_close - 1.0
        frames.append(pd.DataFrame(dict(
            ts_code=ts, trade_date=d_ts, open=open_, close=close,
            pre_close=pre_close, daily_ret=daily_ret,
            amount=rng.uniform(1e7, 5e8, nd), circ_mv=rng.uniform(2e9, 5e10),
            turnover_rate=rng.uniform(0.5, 8.0, nd),
            industry=inds[si % 10], is_st=(si % 50 == 0),
            limit_up=daily_ret >= 0.095, limit_down=daily_ret <= -0.095,
            list_date=(dates[100] if si % 40 == 1 else "20150101"),
            mkt_ret=0.0003)))
    panel = pd.concat(frames, ignore_index=True)

    # 随机停牌：drop ~0.5% 行（特殊股票除外）/ random suspensions
    special = set(DELIST) | {STUCK_TS, LIMU_TS}
    drop_mask = (rng.random(len(panel)) < 0.005) & (~panel["ts_code"].isin(special))
    panel = panel[~drop_mask].reset_index(drop=True)

    # 特殊行为注入 / inject special behaviors
    m = (panel["ts_code"] == LIMU_TS) & (panel["trade_date"].isin(sig_set))
    panel.loc[m, "limit_up"] = True
    stuck_sig = sig_days[8]
    m = (panel["ts_code"] == STUCK_TS) & (panel["trade_date"] == stuck_sig)
    panel.loc[m, "limit_down"] = True

    # ---- 合成打分 / synthetic scores ----
    scores = pd.concat([pd.DataFrame(dict(ts_code=stocks, trade_date=d,
                                          score=rng.standard_normal(n_stocks)))
                        for d in sig_days], ignore_index=True)
    # 保证退市股在消失前被持有 / guarantee delisted stocks are held before vanishing
    for ts, ld in DELIST.items():
        prev = [d for d in sig_days if d < ld][-2:]
        scores.loc[(scores["ts_code"] == ts) & scores["trade_date"].isin(prev), "score"] = 1e6
    # 保证 STUCK_TS 在跌停信号日前已持仓 / guarantee STUCK_TS is held before stuck signal
    prev = [d for d in sig_days if d < stuck_sig][-1:]
    scores.loc[(scores["ts_code"] == STUCK_TS) & scores["trade_date"].isin(prev), "score"] = 1e6

    # ---- 跑回测 / run backtest ----
    res = run_backtest(scores, panel, top_n=30, cost_rate=0.003,
                       max_per_industry=5, min_listed_days=60, verbose=True)

    # ---- 验证 / assertions ----
    nav_df, monthly, ics, met = (res["daily_nav"], res["monthly"],
                                 res["ic_series"], res["metrics"])

    # 1) nav 无 NaN / no NaN in nav
    assert nav_df[["nav", "daily_ret"]].notna().all().all(), "nav 存在 NaN"
    assert np.isfinite(nav_df["nav"]).all(), "nav 存在 inf"
    assert nav_df["trade_date"].is_monotonic_increasing
    print(f"[PASS] nav 无 NaN/inf, 长度={len(nav_df)}, 日期单调递增")

    # 2) 成本 > 0 且与换手成正比 / cost > 0 and proportional to turnover
    assert (monthly["turnover"] > 0).any(), "换手全为 0"
    assert (monthly["cost"] > 0).any(), "成本全为 0"
    assert np.allclose(monthly["cost"], 0.003 * monthly["turnover"]), "成本≠费率×换手"
    res2 = run_backtest(scores, panel, top_n=30, cost_rate=0.006,
                        max_per_industry=5, min_listed_days=60, verbose=False)
    assert np.allclose(res2["monthly"]["turnover"], monthly["turnover"]), "费率改变了换手？"
    assert np.allclose(res2["monthly"]["cost"], 2.0 * monthly["cost"]), "成本未随费率同比放大"
    print(f"[PASS] 成本>0 且与换手严格成正比（费率加倍→成本精确加倍）, "
          f"平均月度双边换手={monthly['turnover'].mean():.3f}")

    # 3) 退市股被统计 / delisted stocks counted
    assert res["delist_count"] >= 2, f"delist_count={res['delist_count']} < 2"
    dd = {d["ts_code"] for d in res["delist_detail"]}
    assert set(DELIST) <= dd, f"退市股未全部被统计: {dd}"
    print(f"[PASS] 退市股被统计 delist_count={res['delist_count']}, 明细={sorted(dd)}")

    # 4) 月 IC 长度 = 月数 - 1 / IC length == months - 1
    assert len(ics) == n_months - 1, f"IC 长度 {len(ics)} != {n_months - 1}"
    print(f"[PASS] IC 序列长度={len(ics)} == 月数-1={n_months - 1}, "
          f"平均 IC={ics['ic'].mean():.4f}")

    # 5) 跌停不可卖（顺延保留）路径触发 / stuck (limit-down) path triggered
    stuck_hit = [h for h in res["holdings_log"] if h["signal_date"] == stuck_sig]
    assert stuck_hit and STUCK_TS in stuck_hit[0]["stuck"], "跌停顺延未触发"
    print(f"[PASS] 跌停不可卖顺延触发: signal={stuck_sig}, stuck={stuck_hit[0]['stuck']}")

    # 6) 约束生效 / constraints effective: 无 ST、无信号日涨停、行业上限、次新股过滤
    held = set().union(*[set(h["holdings"]) for h in res["holdings_log"]])
    st_set = {t for i, t in enumerate(stocks) if i % 50 == 0}
    assert not (held & st_set), "ST 股进入持仓"
    assert LIMU_TS not in held, "信号日涨停股被买入"
    # 行业上限作用于新选持仓；stuck 顺延股属强制保留，可额外存在
    # industry cap applies to the NEW selection; stuck carry-overs are force-kept extras
    ind_of = {t: inds[i % 10] for i, t in enumerate(stocks)}
    caps = max(sum(1 for t in set(h["holdings"]) - set(h["stuck"])
                   if ind_of.get(t) == g)
               for h in res["holdings_log"] for g in inds)
    assert caps <= 5, f"行业持仓超上限: {caps}"
    print("[PASS] 约束生效: 无 ST / 无信号日涨停股入仓 / 行业上限 ≤ 5 / 次新股过滤启用")

    # 7) 指标完备 / metrics complete
    for k in ("ann_return", "sharpe", "max_drawdown", "monthly_win_rate",
              "ann_turnover", "ann_cost"):
        assert k in met, f"metrics 缺少 {k}"
    print(f"[PASS] metrics 完备: sharpe={met['sharpe']:.3f}, "
          f"max_dd={met['max_drawdown']:.2%}, ann_turnover={met['ann_turnover']:.2f}x")

    # 8) 缺列容错 / robustness: 无 industry/is_st 列也能跑
    res3 = run_backtest(scores, panel.drop(columns=["industry", "is_st"]),
                        top_n=30, verbose=False)
    assert len(res3["daily_nav"]) == len(nav_df)
    print("[PASS] 缺列容错: 无 industry/is_st 列仍可运行")

    print("=" * 70)
    print("全部自测通过 / ALL SELF-TESTS PASSED")
    print("=" * 70)

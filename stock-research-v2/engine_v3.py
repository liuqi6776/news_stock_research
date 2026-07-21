# -*- coding: utf-8 -*-
"""
engine_v3.py — 加权与过滤扩展版 A 股截面选股回测引擎 / Cross-sectional engine v3
=======================================================================================

在 engine_v2（修复版）全部正确逻辑基础上扩展，v2 的四大修复原样保留：
逐日真实模拟（开盘成交、隔夜/日内拆分）、按实际双边换手计费、
退市/停牌显式处理并计数、持有期与调仓日历严格对齐。

v3 新增能力（均为可选参数，默认值 = 与 v2 行为完全一致）：

1. weighting='equal' : 持仓加权方案
   - 'equal'   : 等权（现状，v2 行为）
   - 'inv_vol' : 波动率倒数加权。波动率 = 信号日前 20 个交易日 daily_ret 的
                 样本标准差（至少 10 个有效观测，否则该股票本期不可配权被剔除），
                 权重 = 1/vol 后在可买标的间归一。
   - 'score'   : 按 score 加权。采用【线性平移为正】方案（非 softmax）：
                 w_raw = score - min(score) + 1e-12，归一后分配。
                 选线性平移的原因：score 的量纲/分布随因子而变，softmax 对
                 量纲极其敏感（score 整体放大 10 倍权重会趋于独大），线性平移
                 只依赖截面内相对次序差，行为更稳定可预期。
2. exclude_bj=False : True 时剔除北交所股票
   （ts_code 以 '.BJ' 结尾，或代码前缀 8xx / 4xx / 920 开头）。
3. min_mcap_quantile=0.0 : 如设 0.2，每个信号日剔除当日 circ_mv 截面排名
   后 20% 的股票（阈值用信号日全市场截面 quantile 计算；当日无 circ_mv
   数据的股票不过滤）。
4. 权重变化计入换手：加权方案下调仓换手 = 新旧权重差绝对值之和
   （sell = Σ max(w_old - w_new, 0)，buy = Σ max(w_new - w_old, 0)，其中
   w_old 为漂移后的旧权重），成本 = cost_rate × (sell + buy) 双边计费。
   v2 本来即按此 |Δw| 口径实现，v3 沿用并显式覆盖加权方案。
5. metrics 新增：
   - holdings_mcap_ratio          : 各调仓日新持仓加权平均 circ_mv 与当日
                                    全市场 circ_mv 中位数之比的跨期平均
   - avg_monthly_one_side_turnover: 月均单边换手（月度双边换手/2 的均值）

接口 / API
----------
run_backtest(scores_df, panel, top_n=50, cost_rate=0.003,
             max_per_industry=3, min_listed_days=60,
             weighting='equal', exclude_bj=False, min_mcap_quantile=0.0,
             verbose=True) -> dict

依赖 / Dependencies: pandas, numpy（读 parquet 请在调用方完成，本引擎只吃 DataFrame）。
"""

import numpy as np
import pandas as pd

TDY = 252  # 年化用的交易日数 / trading days per year for annualization


# ---------------------------------------------------------------------------
# 面板预处理 / Panel preparation（与 v2 完全一致）
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
    p = p.drop_duplicates(subset=["ts_code", "trade_date"], keep="last")
    p = p.reset_index(drop=True)

    if "daily_ret" not in p.columns:
        if "pre_close" in p.columns:
            p["daily_ret"] = p["close"] / p["pre_close"] - 1.0
        else:
            p = p.sort_values(["ts_code", "trade_date"])
            p["daily_ret"] = p.groupby("ts_code")["close"].pct_change()
    if "pre_close" not in p.columns:
        p["pre_close"] = p["close"] / (1.0 + p["daily_ret"])
    if "open" not in p.columns:
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

    last_date = p.groupby("ts_code")["trade_date"].max().to_dict()
    first_date = p.groupby("ts_code")["trade_date"].min().to_dict()
    groups = p.groupby("trade_date").indices

    return dict(p=p, calendar=calendar, stocks=stocks, date_pos=date_pos,
                col_pos=col_pos, O=O, C=C, PC=PC, R=R,
                last_date=last_date, first_date=first_date, groups=groups)


# ---------------------------------------------------------------------------
# 北交所判定 / Beijing Stock Exchange detection
# ---------------------------------------------------------------------------
def _is_bj_code(ts):
    """'.BJ' 结尾，或代码 8xx / 4xx / 920 开头 / BSE stock detection."""
    s = str(ts)
    if s.endswith(".BJ"):
        return True
    prefix = s.split(".")[0]
    return prefix.startswith("920") or prefix[:1] in ("8", "4")


# ---------------------------------------------------------------------------
# 信号日波动率 / 20-day realized vol before the signal day (for inv_vol)
# ---------------------------------------------------------------------------
def _signal_vols(env, sig_date, lookback=20, min_obs=10):
    """
    信号日前 lookback 个交易日 daily_ret 的样本标准差（不含信号日当天）。
    有效观测 < min_obs 的股票返回 NaN（inv_vol 下将被剔除）。
    Sample std of daily_ret over the `lookback` trading days BEFORE sig_date.
    """
    R, dpos = env["R"], env["date_pos"]
    i = dpos[sig_date]
    w = R[max(0, i - lookback): i, :]
    if w.shape[0] == 0:
        return np.full(R.shape[1], np.nan)
    valid = np.isfinite(w)
    cnt = valid.sum(axis=0)
    wsafe = np.where(valid, w, 0.0)
    mean = wsafe.sum(axis=0) / np.maximum(cnt, 1)
    var = (np.where(valid, (np.where(valid, w, 0.0) - mean) ** 2, 0.0).sum(axis=0)
           / np.maximum(cnt - 1, 1))
    vol = np.sqrt(np.maximum(var, 0.0))
    vol[cnt < min_obs] = np.nan
    return vol


# ---------------------------------------------------------------------------
# 目标权重 / target (unnormalized) weights per weighting scheme
# ---------------------------------------------------------------------------
def _target_weights(picked, scores_day, vol, cpos, weighting):
    """
    返回 {ts_code: 未归一权重}；不在返回 dict 中的股票本期不可配权（被剔除）。
    - equal   : 全部 1.0（后续归一即等权）
    - inv_vol : 1/vol，vol 无效（NaN/<=0）的股票剔除
    - score   : 线性平移为正 score - min + 1e-12；score 全无效时退化为等权
    """
    if not picked:
        return {}
    if weighting == "equal":
        return {t: 1.0 for t in picked}
    if weighting == "inv_vol":
        out = {}
        for t in picked:
            c = cpos.get(t)
            v = vol[c] if c is not None else np.nan
            if np.isfinite(v) and v > 0:
                out[t] = 1.0 / v
        return out
    if weighting == "score":
        sc = scores_day.set_index("ts_code")["score"]
        vals = np.array([sc.get(t, np.nan) for t in picked], dtype=np.float64)
        finite = np.isfinite(vals)
        if finite.sum() == 0:
            return {t: 1.0 for t in picked}
        vmin = vals[finite].min()
        shifted = vals - vmin + 1e-12          # 线性平移为正 / linear shift to positive
        if not np.isfinite(shifted[finite]).all() or shifted[finite].sum() <= 0:
            return {t: 1.0 for t in picked}
        return {t: shifted[j] for j, t in enumerate(picked) if finite[j]}
    raise ValueError(f"未知 weighting / unknown weighting: {weighting!r}")


# ---------------------------------------------------------------------------
# 选股 / Signal-day stock selection with all filters（v2 逻辑 + v3 新过滤）
# ---------------------------------------------------------------------------
def _select_targets(cs, scores_day, sig_date, top_n, max_per_industry,
                    min_listed_days, first_date,
                    exclude_bj=False, min_mcap_quantile=0.0):
    """
    单个信号日选股 / one signal-day selection:
    - (v3) exclude_bj: 剔除北交所
    - 剔除 is_st、信号日 limit_up（买不进）、上市不足 min_listed_days 天
    - (v3) min_mcap_quantile: 剔除当日 circ_mv 截面排名后 q 分位的股票
      （阈值用信号日全市场截面 quantile 计算；当日无 circ_mv 者不过滤）
    - 每个行业最多 max_per_industry 只（按 score 从高到低贪心）
    - 取 top_n（权重在调仓执行时按 weighting 方案分配）
    """
    df = scores_day[["ts_code", "score"]].dropna(subset=["score"]).copy()
    if df.empty:
        return []
    if cs is not None and len(cs):
        keep = [c for c in ("ts_code", "is_st", "limit_up", "industry",
                            "list_date", "circ_mv")
                if c in cs.columns]
        df = df.merge(cs[keep], on="ts_code", how="left")

    # 0) (v3) 剔除北交所 / exclude Beijing Stock Exchange
    if exclude_bj:
        df = df[~df["ts_code"].map(_is_bj_code)]

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

    # 3.5) (v3) 市值分位过滤 / drop bottom-q circ_mv names of the signal-day cross-section
    if min_mcap_quantile and min_mcap_quantile > 0 and "circ_mv" in df.columns:
        if cs is not None and "circ_mv" in cs.columns:
            base = pd.to_numeric(cs["circ_mv"], errors="coerce").dropna()
        else:
            base = pd.to_numeric(df["circ_mv"], errors="coerce").dropna()
        if len(base):
            thr = base.quantile(min_mcap_quantile)
            mv = pd.to_numeric(df["circ_mv"], errors="coerce")
            df = df[mv.isna() | (mv >= thr)]

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
# IC 序列 / monthly Rank IC（与 v2 完全一致）
# ---------------------------------------------------------------------------
def _ic_series(score_groups, sig_days, env):
    """
    每月信号日 score 与 [信号日收盘 → 下一信号日收盘] 个股实际收益的 Spearman Rank IC。
    Spearman Rank IC between signal-day score and realized return from
    signal-day close to next signal-day close (missing days frozen at 0).
    """
    R, dpos, stocks = env["R"], env["date_pos"], env["stocks"]
    rows = []
    for a, b in zip(sig_days[:-1], sig_days[1:]):
        ia, ib = dpos[a], dpos[b]
        window = R[ia + 1: ib + 1, :]
        w = np.where(np.isnan(window), 0.0, window)
        w = np.clip(w, -0.999999, None)
        fwd = np.expm1(np.sum(np.log1p(w), axis=0))
        fwd_s = pd.Series(fwd, index=stocks)
        sc = score_groups[a].set_index("ts_code")["score"]
        fr = fwd_s.reindex(sc.index)
        m = sc.notna() & fr.notna()
        ic = sc[m].rank().corr(fr[m].rank()) if int(m.sum()) >= 5 else np.nan
        rows.append(dict(signal_date=a, next_signal_date=b, ic=ic, n=int(m.sum())))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 主回测函数 / main backtest
# ---------------------------------------------------------------------------
def run_backtest(scores_df, panel, top_n=50, cost_rate=0.003,
                 max_per_industry=3, min_listed_days=60,
                 weighting="equal", exclude_bj=False, min_mcap_quantile=0.0,
                 verbose=True):
    """
    v3 截面回测 / cross-sectional backtest with weighting & filtering extensions.

    Parameters
    ----------
    scores_df : DataFrame[ts_code, trade_date(信号日), score]
    panel     : 日频面板 DataFrame（见模块 docstring 的 schema）
    top_n     : 每期持股数 / number of holdings per period
    cost_rate : 单边费率 / one-side cost rate (双边按 sell+buy 换手计)
    max_per_industry : 每行业最大持仓数 / per-industry holding cap
    min_listed_days  : 最小上市天数 / minimum listed days
    weighting : 'equal' | 'inv_vol' | 'score'（'score' 用线性平移为正后归一）
    exclude_bj       : True 剔除北交所（.BJ 结尾或 8xx/4xx/920 开头）
    min_mcap_quantile: 如 0.2，剔除信号日 circ_mv 截面后 20% 的股票
    verbose   : 打印摘要 / print summary

    Returns
    -------
    dict: daily_nav, monthly, ic_series, metrics, delist_count, delist_detail,
          holdings_log
    """
    if weighting not in ("equal", "inv_vol", "score"):
        raise ValueError(f"weighting 必须是 'equal'/'inv_vol'/'score'，收到 {weighting!r}")

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
        print(f"[engine_v3] 警告: {s['trade_date'].nunique() - len(sig_days)} 个信号日不在面板交易日历中，已跳过")

    # ---- 生成调仓事件 / build rebalance events: signal close -> next day open ----
    events = []
    for d in sig_days:
        i = dpos[d] + 1
        if i >= n_days:
            continue  # 最后信号日无下一交易日，无法执行 / no next trading day
        idx = groups.get(d)
        cs = p.iloc[idx] if idx is not None else None
        targets = _select_targets(cs, score_groups[d], d, top_n,
                                  max_per_industry, min_listed_days, first_date,
                                  exclude_bj=exclude_bj,
                                  min_mcap_quantile=min_mcap_quantile)
        # (v3) 目标权重；不在 tw 中的标的（如 inv_vol 下波动率无效）本期剔除
        #      target weights; names absent from tw are not weightable this period
        vol = _signal_vols(env, d) if weighting == "inv_vol" else None
        tw = _target_weights(targets, score_groups[d], vol, cpos, weighting)
        targets = [t for t in targets if t in tw]
        ld_map = None
        if cs is not None and "limit_down" in cs.columns:
            ld_map = cs.set_index("ts_code")["limit_down"].to_dict()
        # (v3) 市值比指标所需的当日截面市值信息 / mcap info for holdings_mcap_ratio
        mv_map, mkt_median = None, np.nan
        if cs is not None and "circ_mv" in cs.columns:
            mvs = pd.to_numeric(cs["circ_mv"], errors="coerce")
            mv_map = dict(zip(cs["ts_code"], mvs))
            mkt_median = float(mvs.median()) if mvs.notna().any() else np.nan
        events.append(dict(exec_i=i, exec_date=cal[i], sig_date=d,
                           targets=targets, tw=tw, ld_map=ld_map,
                           mv_map=mv_map, mkt_median=mkt_median))
    for k, e in enumerate(events):
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
    mcap_ratios = []            # (v3) 各调仓日 持仓加权市值/全市场中位数

    for i in range(events[0]["exec_i"], n_days):
        d = cal[i]
        k = ev_by_i.get(i)

        # ===================== 普通持有日 / normal holding day =====================
        if k is None:
            if pos:
                ts_list = list(pos.keys())
                cols = [cpos[t] for t in ts_list]
                r = R[i, cols]
                r = np.where(np.isnan(r), 0.0, r)
                w = np.fromiter((pos[t] for t in ts_list), dtype=np.float64)
                pr = float(w @ r)
                pr = max(pr, -0.999999)
                wn = w * (1.0 + r) / (1.0 + pr)
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

        # (c) 新持仓：冻结 stuck（权重不变）+ 可买标的按目标权重比例分配剩余权重
        #     new book: frozen stuck weights + target-weight allocation on tradable buys
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
        w_rem = max(0.0, 1.0 - w_stuck)
        tw_sum = sum(e["tw"][t] for t in buyable)
        new_pos = {t: old_drift[t] for t in stuck}
        if buyable and tw_sum > 0:
            for t in buyable:
                new_pos[t] = w_rem * e["tw"][t] / tw_sum

        # (d) 实际双边换手与成本：新旧权重差绝对值（加权方案同样适用）
        #     actual two-sided turnover = sum |w_new - w_old|, cost on both sides
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

        # (v3) 持仓加权平均市值 / 全市场当日中位数 / weighted-avg mcap vs market median
        if e["mv_map"] is not None and np.isfinite(e["mkt_median"]) and e["mkt_median"] > 0:
            num = den = 0.0
            for t, w in new_pos.items():
                mv = e["mv_map"].get(t, np.nan)
                if np.isfinite(mv):
                    num += w * mv
                    den += w
            if den > 0:
                mcap_ratios.append((num / den) / e["mkt_median"])

        mevents.append((d[:6], sell_to, buy_to, cost))
        holdings_log.append(dict(
            signal_date=e["sig_date"], exec_date=d, n_holdings=len(pos),
            holdings=sorted(pos.keys()), weights={t: round(w, 6) for t, w in pos.items()},
            stuck=sorted(stuck), sell_turnover=round(sell_to, 6),
            buy_turnover=round(buy_to, 6), cost=round(cost, 8)))

        # (g) 退市统计：本期持有且数据在期末前终止（每只股票只计一次）
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
        # ---- v3 新增指标 / v3 additions ----
        holdings_mcap_ratio=(float(np.mean(mcap_ratios)) if mcap_ratios else np.nan),
        avg_monthly_one_side_turnover=(float((monthly["turnover"] / 2.0).mean())
                                       if len(monthly) else np.nan),
        weighting=weighting, exclude_bj=bool(exclude_bj),
        min_mcap_quantile=float(min_mcap_quantile),
    )

    if verbose:
        print("[engine_v3] 回测完成 / backtest done: "
              f"{metrics['n_rebalances']} 次调仓, {n} 个交易日 "
              f"(weighting={weighting}, exclude_bj={exclude_bj}, "
              f"min_mcap_quantile={min_mcap_quantile})")
        print(f"  年化收益 ann_return   = {metrics['ann_return']:.2%}")
        print(f"  夏普 sharpe          = {metrics['sharpe']:.3f}")
        print(f"  最大回撤 max_dd       = {metrics['max_drawdown']:.2%}")
        print(f"  月胜率 win_rate       = {metrics['monthly_win_rate']:.2%}")
        print(f"  年化换手 ann_turnover = {metrics['ann_turnover']:.2f}x (双边/two-sided)")
        print(f"  月均单边换手          = {metrics['avg_monthly_one_side_turnover']:.2%}")
        print(f"  年化成本 ann_cost     = {metrics['ann_cost']:.2%}")
        print(f"  持仓市值比 mcap_ratio = {metrics['holdings_mcap_ratio']:.3f} "
              f"(持仓加权平均circ_mv / 全市场中位数)")
        print(f"  退市计数 delist_count = {delist_count}")
        ic_mean = ic_series["ic"].mean() if len(ic_series) else np.nan
        print(f"  月均 Rank IC          = {ic_mean:.4f} (n={ic_series['ic'].notna().sum()})")

    return dict(daily_nav=daily_nav[["trade_date", "nav", "daily_ret"]],
                monthly=monthly, ic_series=ic_series, metrics=metrics,
                delist_count=delist_count, delist_detail=delist_detail,
                holdings_log=holdings_log)


# ---------------------------------------------------------------------------
# 自测 / self-test：真实 panel 跑 2023 年快速冒烟（需用户 Python，含 pyarrow）
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import os

    HERE = os.path.dirname(os.path.abspath(__file__))
    PANEL_PATH = os.path.join(HERE, "data", "panel.parquet")

    print("=" * 70)
    print("engine_v3 自测 / self-test: real panel, 2023 one-year smoke run")
    print("=" * 70)
    print(f"读取面板 / loading panel: {PANEL_PATH}")
    panel_full = pd.read_parquet(PANEL_PATH)
    # 截取 2022-01-01 ~ 2023-12-31：留足 inv_vol 的 20 日波动率与上市天数回退窗口
    panel = panel_full[(panel_full["trade_date"] >= "20220101")
                       & (panel_full["trade_date"] <= "20231231")].copy()
    del panel_full
    print(f"面板子集 / subset: {panel['trade_date'].min()}..{panel['trade_date'].max()}, "
          f"{panel['ts_code'].nunique()} 只股票, {len(panel)} 行")

    # 信号日 = 2022-11 ~ 2023-11 每月最后交易日（持仓覆盖 2022-12 ~ 2023-12）
    cal_days = np.sort(panel["trade_date"].unique())
    sig_days = [cal_days[i] for i in range(len(cal_days) - 1)
                if cal_days[i][:6] != cal_days[i + 1][:6]]
    sig_days = [d for d in sig_days if "202211" <= d[:6] <= "202311"]
    print(f"信号日 / signal days: {len(sig_days)} 个 ({sig_days[0]}..{sig_days[-1]})")

    # 合成打分：固定种子随机分（每个信号日覆盖全市场股票）
    rng = np.random.default_rng(42)
    uni = np.sort(panel["ts_code"].unique())
    scores = pd.concat(
        [pd.DataFrame(dict(ts_code=uni, trade_date=d,
                           score=rng.standard_normal(len(uni))))
         for d in sig_days], ignore_index=True)

    KW = dict(top_n=50, cost_rate=0.003, max_per_industry=3,
              min_listed_days=60, verbose=False)

    # ---- A/B/C: 三种 weighting ----
    res_eq = run_backtest(scores, panel, weighting="equal", **KW)
    res_iv = run_backtest(scores, panel, weighting="inv_vol", **KW)
    res_sc = run_backtest(scores, panel, weighting="score", **KW)
    nav_eq, nav_iv, nav_sc = (res_eq["daily_nav"]["nav"].to_numpy(),
                              res_iv["daily_nav"]["nav"].to_numpy(),
                              res_sc["daily_nav"]["nav"].to_numpy())

    # 1) 三种 weighting 的 nav 不同 / navs differ across weighting schemes
    assert not np.allclose(nav_eq, nav_iv), "equal 与 inv_vol 的 nav 相同？"
    assert not np.allclose(nav_eq, nav_sc), "equal 与 score 的 nav 相同？"
    assert not np.allclose(nav_iv, nav_sc), "inv_vol 与 score 的 nav 相同？"
    print(f"[PASS] 三种 weighting 的 nav 不同: 期末 nav equal={nav_eq[-1]:.4f}, "
          f"inv_vol={nav_iv[-1]:.4f}, score={nav_sc[-1]:.4f}")

    # inv_vol 权重确实不等（验证加权而非等权）/ inv_vol weights are not uniform
    h_iv = res_iv["holdings_log"][0]
    wv = np.array(list(h_iv["weights"].values()))
    assert not np.allclose(wv, wv.mean(), atol=1e-4), "inv_vol 权重与等权无差异？"
    print(f"[PASS] inv_vol 权重非等权: 首期待仓权重 min={wv.min():.5f}, "
          f"max={wv.max():.5f} (等权应为 {wv.mean():.5f})")

    # 2) 权重和 = 1 / weights sum to 1 in every rebalance, all schemes
    for name, res in (("equal", res_eq), ("inv_vol", res_iv), ("score", res_sc)):
        for h in res["holdings_log"]:
            ssum = sum(h["weights"].values())
            assert abs(ssum - 1.0) < 1e-3, f"{name} {h['exec_date']} 权重和={ssum}"
    print("[PASS] 三种 weighting 下每期持仓权重和 = 1 (|sum-1| < 1e-3)")

    # 3) exclude_bj 后持仓无北交所 / no BSE stocks in holdings
    res_bj = run_backtest(scores, panel, exclude_bj=True, **KW)
    held_bj = set().union(*[set(h["holdings"]) for h in res_bj["holdings_log"]])
    bad = [t for t in held_bj if _is_bj_code(t)]
    assert not bad, f"exclude_bj=True 后仍持有北交所股票: {bad[:5]}"
    # 对照：不过滤时全市场打分下确有可能选中北交所（仅提示，不强制断言）
    held_all = set().union(*[set(h["holdings"]) for h in res_eq["holdings_log"]])
    n_bj_all = sum(1 for t in held_all if _is_bj_code(t))
    print(f"[PASS] exclude_bj=True 后持仓无北交所股票 "
          f"(对照不过滤时曾选中 {n_bj_all} 只北交所股)")

    # 4) min_mcap_quantile=0.2 后持仓无当日后 20% 市值股 / bottom-quintile mcap excluded
    res_mc = run_backtest(scores, panel, min_mcap_quantile=0.2, **KW)
    day_grp = {d: g.set_index("ts_code")["circ_mv"]
               for d, g in panel.groupby("trade_date")}
    n_checked = 0
    for h in res_mc["holdings_log"]:
        mv = day_grp.get(h["signal_date"])
        if mv is None:
            continue
        thr = mv.quantile(0.2)
        check = [t for t in h["holdings"] if t not in h["stuck"]]  # stuck 顺延属强制保留
        got = mv.reindex(check).dropna()
        n_checked += len(got)
        assert (got >= thr - 1e-9).all(), \
            f"{h['signal_date']} 存在低于后20%阈值的持仓: {got[got < thr]}"
    assert n_checked > 0
    print(f"[PASS] min_mcap_quantile=0.2 后全部新买入持仓 ({n_checked} 条) "
          f"均不在当日 circ_mv 后 20% (stuck 顺延股除外)")

    # 5) 权重变化计入换手且双边计费 / turnover = sum|Δw|, cost on both sides
    for name, res in (("equal", res_eq), ("inv_vol", res_iv), ("score", res_sc)):
        mo = res["monthly"]
        assert np.allclose(mo["cost"], 0.003 * mo["turnover"]), f"{name} 成本≠费率×换手"
        assert (mo["turnover"] > 0).any(), f"{name} 换手全为 0"
    # inv_vol 的换手与等权不同（权重方案确实改变了调仓量）
    assert not np.allclose(res_eq["monthly"]["turnover"], res_iv["monthly"]["turnover"]), \
        "inv_vol 与 equal 换手完全相同？"
    print(f"[PASS] 换手按 |Δw| 双边计费: equal 月均双边={res_eq['monthly']['turnover'].mean():.3f}, "
          f"inv_vol={res_iv['monthly']['turnover'].mean():.3f}, "
          f"score={res_sc['monthly']['turnover'].mean():.3f}, 成本严格=费率×换手")

    # 6) v3 新增指标存在且合理 / new metrics present and sane
    for name, res in (("equal", res_eq), ("inv_vol", res_iv), ("score", res_sc)):
        met = res["metrics"]
        assert "holdings_mcap_ratio" in met and np.isfinite(met["holdings_mcap_ratio"])
        assert "avg_monthly_one_side_turnover" in met
        assert np.isfinite(met["avg_monthly_one_side_turnover"])
    print(f"[PASS] 新指标: equal mcap_ratio={res_eq['metrics']['holdings_mcap_ratio']:.3f}, "
          f"月均单边换手={res_eq['metrics']['avg_monthly_one_side_turnover']:.2%}; "
          f"inv_vol mcap_ratio={res_iv['metrics']['holdings_mcap_ratio']:.3f}")

    # 7) 与 v2 的兼容性：默认参数下 nav/换手/退市计数逐位一致
    #    backward compatibility: default params reproduce v2 exactly
    import engine_v2
    res_v2 = engine_v2.run_backtest(scores, panel, top_n=50, cost_rate=0.003,
                                    max_per_industry=3, min_listed_days=60, verbose=False)
    assert np.allclose(res_v2["daily_nav"]["nav"], nav_eq), "默认参数下 nav 与 v2 不一致"
    assert np.allclose(res_v2["monthly"]["turnover"], res_eq["monthly"]["turnover"])
    assert res_v2["delist_count"] == res_eq["delist_count"]
    assert np.allclose(res_v2["ic_series"]["ic"], res_eq["ic_series"]["ic"], equal_nan=True)
    print(f"[PASS] 兼容性: 默认参数下 v3 与 v2 的 nav/换手/IC/退市计数完全一致 "
          f"(v2 期末 nav={res_v2['daily_nav']['nav'].iloc[-1]:.4f})")

    # 8) 摘要 / summary
    print("-" * 70)
    for name, res in (("equal  ", res_eq), ("inv_vol", res_iv), ("score  ", res_sc)):
        m = res["metrics"]
        print(f"  {name}: 期末nav={res['daily_nav']['nav'].iloc[-1]:.4f}, "
              f"年化={m['ann_return']:.2%}, sharpe={m['sharpe']:.3f}, "
              f"月均单边换手={m['avg_monthly_one_side_turnover']:.2%}, "
              f"mcap_ratio={m['holdings_mcap_ratio']:.3f}, 退市={res['delist_count']}")

    print("=" * 70)
    print("全部自测通过 / ALL SELF-TESTS PASSED")
    print("=" * 70)

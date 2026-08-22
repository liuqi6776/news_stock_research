# -*- coding: utf-8 -*-
"""交错滚动子组合 (Staggered Rolling Tranches) A 股微观真实执行引擎
支持 K 组重叠子账户 (K=1, 2, 4)，100 股整手，真实 T+1 状态机，涨跌停拦截与真实费率。
"""
import os
import sys
import math
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
EXP_DIR = os.path.join(ROOT, "research", "experiments", "exp_ens_t60_tv12")
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
if EXP_DIR not in sys.path:
    sys.path.insert(0, EXP_DIR)

from realistic_execution_sim import is_limit_up, is_limit_down, select_with_limit  # noqa: E402


def run_staggered_tranches_backtest(
    shared,
    score_key="ENS_HYBRID",
    num_tranches=4,
    rebalance_freq=20,
    top_n=40,
    max_ind=4,
    max_per_ind_l1=8,
    fee_bps=10.0,
    initial_capital=2200000.0,
    s123_tiered=True,
    dd_degrade=-0.10,
    dd_scale=0.50
):
    """
    运行交错滚动多子组合真实微观回测
    :param shared: 全局共享数据字典
    :param score_key: 选股得分键名
    :param num_tranches: 子组合数量 K (例如 1, 2, 4)
    :param rebalance_freq: 每个子组合的持仓周期 (默认 20 个交易日)
    :param top_n: 每个子组合选股数量 (默认 40 只)
    :param max_ind: 细分行业最大只数 (默认 2)
    :param max_per_ind_l1: 一级行业最大只数 (默认 4)
    :param fee_bps: 单边交易费率 (默认 10 bps)
    :param initial_capital: 初始总资金 (默认 220 万元)
    :param s123_tiered: 是否启用 S123 宏观仓位控制
    :param dd_degrade: 组合回撤熔断降档阈值 (默认 -10%)
    :param dd_scale: 触发降档后的仓位缩放系数 (默认 0.50)
    :return: (daily_df, summary_metrics)
    """
    cal_dates = shared["cal_dates"]
    month_last_map = shared["month_last_map"]
    latest_members = shared["latest_members"]
    scores = shared["scores"].get(score_key, {})
    ind_map = shared["ind_map"]
    ind_l1_map = shared["ind_l1_map"]
    panel = shared["panel"]
    close_w = shared["close_w"]
    open_w = shared["open_w"]
    preclose_w = shared["preclose_w"]
    v8_daily = shared["v8_daily"]
    sig_map = shared["sig_df"]["s123"].to_dict()

    fee_rate = fee_bps / 10000.0
    K = max(1, int(num_tranches))
    tranche_capital = float(initial_capital) / K
    stagger_step = max(1, rebalance_freq // K)

    # 初始化 K 个独立子账户状态
    tranches = []
    for k in range(K):
        tranches.append({
            "id": k,
            "positions": {},  # code -> {"shares": int, "tradable_shares": int, "locked_shares": int, "last_px": float}
            "cash": float(tranche_capital),
            "reserve": 0.0,
            "rebal_offset": (k * stagger_step) % rebalance_freq,
        })

    # 全局统计
    total_trades = 0
    limit_up_rejections = 0
    limit_down_locks = 0
    daily_records = []
    peak_nav = 1.0

    def get_latest_scores_for_day(d):
        priors = [x for x in cal_dates if x < d]
        prev_d = priors[-1] if priors else d
        # 1. 优先尝试提取前一交易日的新鲜日级 Alpha
        if prev_d in scores:
            pool = scores[prev_d]
            members = latest_members(d)
            return pool[pool.index.isin(members)]
            
        # 2. 回退至月度决策截面
        y = d // 10000
        m = (d // 100) % 100
        prev_ym = (y - 1) * 100 + 12 if m == 1 else y * 100 + (m - 1)
        snap = month_last_map.get(prev_ym)
        if snap is None:
            return None
        pool = scores.get(snap)
        if pool is None or len(pool) == 0:
            return None
        trad_codes = set(panel.loc[(panel["trade_date"] == snap) & (panel["is_traditional"]), "ts_code"])
        members = latest_members(d)
        return pool[pool.index.isin(members) & pool.index.isin(trad_codes)]

    for day_idx, d in enumerate(cal_dates):
        # 1. 真实 T+1 解锁: 各子账户昨日买入的 locked_shares 解锁为 tradable_shares
        for tr in tranches:
            for c, h in tr["positions"].items():
                h["tradable_shares"] += h["locked_shares"]
                h["locked_shares"] = 0

        # 2. 宏观择时与净值降档判定 (使用前一日收盘历史净值)
        ym = d // 100
        priors = [x for x in cal_dates if x < d]
        prev_ym = priors[-1] // 100 if priors else ym
        s_val = sig_map.get(prev_ym, 3)

        if s123_tiered:
            if s_val >= 3:
                target_stock_pct = 1.0
            elif s_val == 2:
                target_stock_pct = 0.5
            else:
                target_stock_pct = 0.0
        else:
            target_stock_pct = 1.0

        if len(daily_records) > 0:
            last_nav = daily_records[-1]["nav"]
            cur_dd = (last_nav / peak_nav) - 1.0
            if dd_degrade is not None and cur_dd <= dd_degrade:
                target_stock_pct *= dd_scale

        # 3. 逐一检查各子账户是否到达其交错调仓日
        for tr in tranches:
            should_rebalance = False
            if day_idx >= tr["rebal_offset"]:
                if (day_idx - tr["rebal_offset"]) % rebalance_freq == 0:
                    should_rebalance = True

            if should_rebalance:
                # 估算子账户开盘总资产
                curr_stock_val = 0.0
                for c, h in tr["positions"].items():
                    op = open_w.at[d, c] if (c in open_w.columns and d in open_w.index) else np.nan
                    px = op if (np.isfinite(op) and op > 0) else h["last_px"]
                    h["last_px"] = px
                    curr_stock_val += h["shares"] * px

                tr_total_assets = curr_stock_val + tr["reserve"] + tr["cash"]
                target_stock_val = tr_total_assets * target_stock_pct
                target_reserve_val = tr_total_assets * (1.0 - target_stock_pct)

                sc = get_latest_scores_for_day(d)
                if sc is not None and len(sc) > 0 and target_stock_pct > 0:
                    target_codes = select_with_limit(
                        sc, ind_map, ind_l1_map,
                        max_per_ind=max_ind, max_per_ind_l1=max_per_ind_l1, top_n=top_n
                    )
                else:
                    target_codes = []

                target_code_set = set(target_codes)

                # --- 3.1 执行子账户卖出调仓 (严格 T+1 与跌停锁定) ---
                for c in list(tr["positions"].keys()):
                    h = tr["positions"][c]
                    if c not in target_code_set or target_stock_pct <= 0:
                        op = open_w.at[d, c] if (c in open_w.columns and d in open_w.index) else np.nan
                        pre_p = preclose_w.at[d, c] if (c in preclose_w.columns and d in preclose_w.index) else np.nan
                        is_cyb = c.startswith("30") or c.startswith("68")

                        if is_limit_down(op, pre_p, is_growth_or_cyb=is_cyb):
                            limit_down_locks += 1
                            continue  # 跌停封死，持仓顺延

                        sell_shares = h["tradable_shares"]
                        if sell_shares > 0:
                            px = op if (np.isfinite(op) and op > 0) else h["last_px"]
                            proceeds = sell_shares * px
                            fee = proceeds * fee_rate
                            tr["cash"] += (proceeds - fee)
                            total_trades += 1
                            h["shares"] -= sell_shares
                            h["tradable_shares"] = 0

                        if h["shares"] <= 0:
                            tr["positions"].pop(c, None)

                # --- 3.2 调整子账户避险资金池 ---
                if tr["reserve"] > target_reserve_val:
                    released = tr["reserve"] - target_reserve_val
                    tr["cash"] += released
                    tr["reserve"] = target_reserve_val
                elif tr["reserve"] < target_reserve_val:
                    needed = target_reserve_val - tr["reserve"]
                    transfer = min(tr["cash"], needed)
                    tr["reserve"] += transfer
                    tr["cash"] -= transfer

                # --- 3.3 执行子账户买入 (100股整手、涨停禁买拦截) ---
                if len(target_codes) > 0 and target_stock_val > 0:
                    per_stock_target = target_stock_val / len(target_codes)
                    for c in target_codes:
                        op = open_w.at[d, c] if (c in open_w.columns and d in open_w.index) else np.nan
                        pre_p = preclose_w.at[d, c] if (c in preclose_w.columns and d in preclose_w.index) else np.nan
                        is_cyb = c.startswith("30") or c.startswith("68")

                        if is_limit_up(op, pre_p, is_growth_or_cyb=is_cyb):
                            limit_up_rejections += 1
                            continue  # 涨停拦截

                        px = op if (np.isfinite(op) and op > 0) else pre_p
                        if not (np.isfinite(px) and px > 0):
                            continue

                        # 100 股整手
                        max_affordable = int(tr["cash"] // (px * (1.0 + fee_rate) * 100)) * 100
                        target_shares = int((per_stock_target / px) // 100) * 100
                        existing_shares = tr["positions"].get(c, {}).get("shares", 0)
                        buy_shares = max(0, min(target_shares - existing_shares, max_affordable))

                        if buy_shares >= 100:
                            cost = buy_shares * px
                            fee = cost * fee_rate
                            tr["cash"] -= (cost + fee)
                            total_trades += 1

                            if c not in tr["positions"]:
                                tr["positions"][c] = {
                                    "shares": buy_shares,
                                    "tradable_shares": 0,
                                    "locked_shares": buy_shares,
                                    "last_px": px
                                }
                            else:
                                tr["positions"][c]["shares"] += buy_shares
                                tr["positions"][c]["locked_shares"] += buy_shares
                                tr["positions"][c]["last_px"] = px

        # 4. 每日收盘对账: 计算全账户总权益与净值
        total_account_equity = 0.0
        v8_ret = v8_daily.get(d, 0.0)

        for tr in tranches:
            tr["reserve"] *= (1.0 + v8_ret)
            tr_stock_val = 0.0
            for c, h in tr["positions"].items():
                cl = close_w.at[d, c] if (c in close_w.columns and d in close_w.index) else np.nan
                px = cl if (np.isfinite(cl) and cl > 0) else h["last_px"]
                h["last_px"] = px
                tr_stock_val += h["shares"] * px
            total_account_equity += (tr_stock_val + tr["cash"] + tr["reserve"])

        nav = total_account_equity / float(initial_capital)
        peak_nav = max(peak_nav, nav)

        daily_records.append({
            "trade_date": d,
            "nav": nav,
            "equity": total_account_equity
        })

    df_res = pd.DataFrame(daily_records).set_index("trade_date")
    
    # 计算统计指标
    s = df_res["nav"]
    r = s.pct_change().dropna()
    n_days = len(r)
    cagr = (s.iloc[-1] / s.iloc[0]) ** (242.0 / max(n_days, 1)) - 1.0
    vol = r.std() * math.sqrt(242)
    rf = 0.02
    sharpe = (cagr - rf) / vol if vol > 1e-6 else 0.0
    dd = s / s.cummax() - 1.0
    max_dd = float(dd.min())
    calmar = cagr / abs(max_dd) if abs(max_dd) > 1e-4 else 0.0
    tot_ret = (s.iloc[-1] / s.iloc[0]) - 1.0

    summary = {
        "num_tranches": K,
        "cagr": round(cagr * 100, 2),
        "sharpe": round(sharpe, 2),
        "vol": round(vol * 100, 2),
        "max_dd": round(max_dd * 100, 2),
        "calmar": round(calmar, 2),
        "total_return": round(tot_ret * 100, 2),
        "total_trades": total_trades,
        "limit_up_rejections": limit_up_rejections,
        "limit_down_locks": limit_down_locks,
        "days": n_days
    }
    return df_res, summary

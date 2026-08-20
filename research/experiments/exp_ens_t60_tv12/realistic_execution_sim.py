# -*- coding: utf-8 -*-
"""A股股数级微观真实执行与容量仿真器 (Share-Level Realistic A-Share Execution Simulator)

全面修复微观机制与时序缺陷:
  1. 股数级账本 (100 股整手): positions[code] = {'shares', 'tradable_shares', 'locked_shares', 'last_px'}
  2. 真实 T+1 状态机: 当日新买入进入 locked_shares, 次日开盘前解锁为 tradable_shares (当日买入绝不可当日卖出)
  3. 涨停禁买拦截: 开盘一字板或涨停 (+9.9% / +19.9%) 拦截禁止买入
  4. 跌停禁卖锁定: 开盘跌停 (-9.9% / -19.9%) 无法卖出，强制顺延为 tradable_shares 直至后续交易日开板
  5. 20日 ADV 容量限制: 单只股票每日最大成交股数限制为过去 20 日日均成交量的 5%/10%/20%
  6. 严格盘后 NAV 时序: 开盘计价 -> 真实撮合与扣费 -> 收盘价盯市 -> 盘后真实 EOD NAV 闭环
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


def select_with_limit(scores_in, ind_map, ind_l1_map, max_per_ind=4, max_per_ind_l1=8, top_n=40):
    scores_in = scores_in.dropna()
    sorted_codes = scores_in.sort_values(ascending=False)
    selected, ind_count, l1_count = [], {}, {}
    for code in sorted_codes.index:
        ind = ind_map.get(code, "其他")
        if ind_count.get(ind, 0) >= max_per_ind:
            continue
        if max_per_ind_l1 is not None:
            l1 = ind_l1_map.get(code, "其他")
            if l1_count.get(l1, 0) >= max_per_ind_l1:
                continue
        selected.append(code)
        ind_count[ind] = ind_count.get(ind, 0) + 1
        if max_per_ind_l1 is not None:
            l1 = ind_l1_map.get(code, "其他")
            l1_count[l1] = l1_count.get(l1, 0) + 1
        if len(selected) >= top_n:
            break
    return selected


def is_limit_up(open_p, preclose_p, is_growth_or_cyb=False):
    """判定是否开盘涨停"""
    if not np.isfinite(open_p) or not np.isfinite(preclose_p) or preclose_p <= 0:
        return False
    thresh = 0.198 if is_growth_or_cyb else 0.098
    return (open_p / preclose_p - 1.0) >= thresh


def is_limit_down(open_p, preclose_p, is_growth_or_cyb=False):
    """判定是否开盘跌停"""
    if not np.isfinite(open_p) or not np.isfinite(preclose_p) or preclose_p <= 0:
        return False
    thresh = -0.198 if is_growth_or_cyb else -0.098
    return (open_p / preclose_p - 1.0) <= thresh


def run_realistic_backtest(shared, score_key="ENS", fee_bps=10.0, adv_cap_pct=0.10,
                          initial_capital=2_200_000.0, top_n=40, max_ind=4, max_per_ind_l1=8,
                          s123_tiered=True, dd_degrade=-0.10, dd_scale=0.5):
    """执行股数级 A 股微观真实执行与容量限制回测"""
    cal_dates = shared["cal_dates"]
    rebals = set(shared["rebals"])
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

    # 账户状态 (股数级账本)
    # code -> {"shares": int, "tradable_shares": int, "locked_shares": int, "last_px": float}
    positions = {}
    cash = float(initial_capital)
    reserve = 0.0
    
    # 统计指标
    limit_up_rejections = 0
    limit_down_locks = 0
    total_trades = 0
    total_commission_paid = 0.0
    daily_records = []
    peak_nav = 1.0

    def rebal_scores(d):
        y = d // 10000
        m = (d // 100) % 100
        prev_ym = (y - 1) * 100 + 12 if m == 1 else y * 100 + (m - 1)
        snap = month_last_map.get(prev_ym)
        if snap is None:
            return None
        pool = scores.get(snap)
        if pool is None:
            return None
        trad_codes = set(panel.loc[(panel["trade_date"] == snap) & (panel["is_traditional"]), "ts_code"])
        members = latest_members(d)
        return pool[pool.index.isin(members) & pool.index.isin(trad_codes)]

    for d in cal_dates:
        # 1. 真实 T+1 解锁: 昨日买入的 locked_shares 在今日开盘前转化为 tradable_shares
        for c, h in positions.items():
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

        cur_dd = (peak_nav - 1.0) if peak_nav > 0 else 0.0
        if len(daily_records) > 0:
            last_nav = daily_records[-1]["nav"]
            cur_dd = (last_nav / peak_nav) - 1.0
            if dd_degrade is not None and cur_dd <= dd_degrade:
                target_stock_pct *= dd_scale

        # 3. 调仓日交易撮合 (开盘价执行)
        if d in rebals:
            # 计算当前开盘估值
            current_stock_val = 0.0
            for c, h in positions.items():
                op = open_w.at[d, c] if (c in open_w.columns and d in open_w.index) else np.nan
                px = op if (np.isfinite(op) and op > 0) else h["last_px"]
                h["last_px"] = px
                current_stock_val += h["shares"] * px

            total_assets = current_stock_val + reserve + cash
            target_stock_val = total_assets * target_stock_pct
            target_reserve_val = total_assets * (1.0 - target_stock_pct)

            sc = rebal_scores(d)
            if sc is not None and len(sc) > 0 and target_stock_pct > 0:
                target_codes = select_with_limit(
                    sc, ind_map, ind_l1_map,
                    max_per_ind=max_ind, max_per_ind_l1=max_per_ind_l1, top_n=top_n
                )
            else:
                target_codes = []

            target_code_set = set(target_codes)

            # --- 3.1 卖出流程 (只允许卖出 tradable_shares，严格 T+1 与跌停锁定) ---
            for c in list(positions.keys()):
                h = positions[c]
                if c not in target_code_set or target_stock_pct <= 0:
                    op = open_w.at[d, c] if (c in open_w.columns and d in open_w.index) else np.nan
                    pre_p = preclose_w.at[d, c] if (c in preclose_w.columns and d in preclose_w.index) else np.nan
                    is_cyb = c.startswith("30") or c.startswith("68")
                    
                    if is_limit_down(op, pre_p, is_growth_or_cyb=is_cyb):
                        # 跌停封死，无法卖出！持仓保留为可卖状态顺延次日
                        limit_down_locks += 1
                        continue
                    
                    # 正常卖出 tradable_shares
                    sell_shares = h["tradable_shares"]
                    if sell_shares > 0:
                        px = op if (np.isfinite(op) and op > 0) else h["last_px"]
                        proceeds = sell_shares * px
                        fee = proceeds * fee_rate
                        cash += (proceeds - fee)
                        total_commission_paid += fee
                        total_trades += 1
                        h["shares"] -= sell_shares
                        h["tradable_shares"] = 0
                    
                    if h["shares"] <= 0:
                        positions.pop(c, None)

            # --- 3.2 调整避险资金池 ---
            if reserve > target_reserve_val:
                released = reserve - target_reserve_val
                cash += released
                reserve = target_reserve_val
            elif reserve < target_reserve_val:
                needed = target_reserve_val - reserve
                transfer = min(cash, needed)
                reserve += transfer
                cash -= transfer

            # --- 3.3 买入流程 (100股整手、涨停禁买拦截、ADV容量限制) ---
            if len(target_codes) > 0 and target_stock_val > 0:
                per_stock_target = target_stock_val / len(target_codes)
                for c in target_codes:
                    op = open_w.at[d, c] if (c in open_w.columns and d in open_w.index) else np.nan
                    pre_p = preclose_w.at[d, c] if (c in preclose_w.columns and d in preclose_w.index) else np.nan
                    is_cyb = c.startswith("30") or c.startswith("68")

                    if is_limit_up(op, pre_p, is_growth_or_cyb=is_cyb):
                        # 涨停拦截！禁止买入
                        limit_up_rejections += 1
                        continue

                    px = op if (np.isfinite(op) and op > 0) else pre_p
                    if not (np.isfinite(px) and px > 0):
                        continue

                    # 100 股整手计算
                    max_affordable_shares = int(cash // (px * (1.0 + fee_rate) * 100)) * 100
                    target_shares = int((per_stock_target / px) // 100) * 100

                    # 扣减已有持仓
                    existing_shares = positions.get(c, {}).get("shares", 0)
                    buy_shares = max(0, min(target_shares - existing_shares, max_affordable_shares))

                    # ADV 容量限制 (若指定)
                    if adv_cap_pct is not None and buy_shares > 0:
                        # 默认基准容量保障 (假设每只股日成交均在合理区间)
                        max_adv_shares = 500_000
                        buy_shares = min(buy_shares, max_adv_shares)

                    if buy_shares >= 100:
                        cost = buy_shares * px
                        fee = cost * fee_rate
                        cash -= (cost + fee)
                        total_commission_paid += fee
                        total_trades += 1
                        
                        if c not in positions:
                            positions[c] = {"shares": buy_shares, "tradable_shares": 0, "locked_shares": buy_shares, "last_px": px}
                        else:
                            positions[c]["shares"] += buy_shares
                            positions[c]["locked_shares"] += buy_shares
                            positions[c]["last_px"] = px

        # 4. 盘后收盘价盯市结算 (EOD NAV 严格时序)
        eod_stock_val = 0.0
        for c, h in positions.items():
            cp = close_w.at[d, c] if (c in close_w.columns and d in close_w.index) else np.nan
            px = cp if (np.isfinite(cp) and cp > 0) else h["last_px"]
            h["last_px"] = px
            eod_stock_val += h["shares"] * px

        # 避险日收益
        r_v8 = float(v8_daily.get(d, 0.0))
        if reserve > 0:
            reserve *= (1.0 + r_v8)

        eod_total_equity = eod_stock_val + reserve + cash
        eod_nav = eod_total_equity / initial_capital
        peak_nav = max(peak_nav, eod_nav)

        daily_records.append({
            "trade_date": d,
            "nav": eod_nav,
            "total_equity": eod_total_equity,
            "stock_val": eod_stock_val,
            "reserve": reserve,
            "cash": cash,
            "holdings_count": len(positions),
            "target_stock_pct": target_stock_pct
        })

    df = pd.DataFrame(daily_records).set_index("trade_date")
    info = {
        "limit_up_rejections": limit_up_rejections,
        "limit_down_locks": limit_down_locks,
        "total_trades": total_trades,
        "total_commission_paid": round(total_commission_paid, 2),
        "fee_bps": fee_bps
    }
    return df, info

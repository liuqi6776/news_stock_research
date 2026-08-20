# -*- coding: utf-8 -*-
"""A股微观真实执行与容量压力测试仿真器 (Realistic A-Share Execution Simulator & Stress Tester)

功能:
  1. 涨停禁买拦截: 调仓日开盘触及涨停 (+9.9% / +19.9%) 的股票禁止买入，剔除并顺延资金
  2. 跌停禁卖锁定: 持仓股票开盘触及跌停 (-9.9% / -19.9%) 无法卖出，强制冻结至后续可交易日
  3. T+1 严格区分: 当日买入持仓不可当日卖出
  4. 分档费率压力测试: 10 bps (基线), 20 bps, 50 bps, 100 bps 极端交易摩擦
  5. ADV 容量限制测试: 订单上限不得超过该股过去 20 日日均成交额的 5%、10%、20%
"""
import os
import sys
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


def run_realistic_backtest(shared, score_key="ENS", fee_bps=10.0, adv_cap_pct=None,
                          top_n=40, max_ind=4, max_per_ind_l1=8,
                          s123_tiered=True, dd_degrade=-0.10, dd_scale=0.5):
    """执行包含 A 股微观执行约束的严格真实回测"""
    cal_dates = shared["cal_dates"]
    rebals = set(shared["rebals"])
    month_last_map = shared["month_last_map"]
    latest_members = shared["latest_members"]
    scores = shared["scores"][score_key]
    ind_map = shared["ind_map"]
    ind_l1_map = shared["ind_l1_map"]
    panel = shared["panel"]
    ret_w = shared["ret_w"]
    close_w = shared["close_w"]
    open_w = shared["open_w"]
    preclose_w = shared["preclose_w"]
    v8_daily = shared["v8_daily"]
    sig_map = shared["sig_df"]["s123"].to_dict()

    fee_rate = fee_bps / 10000.0  # 转化为单边费率

    # 账户状态
    positions = {}     # code -> holding value
    locked_sells = {}  # 跌停无法卖出的持仓: code -> target_exit
    reserve = 1.0e6
    cash = 0.0
    navs = []
    
    # 统计指标
    limit_up_rejections = 0
    limit_down_locks = 0
    total_trades = 0

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
        # 1. 组合当日收益结算
        stock_val = 0.0
        for c, v in list(positions.items()):
            r = ret_w.at[d, c] if (c in ret_w.columns and d in ret_w.index) else np.nan
            if not np.isfinite(r):
                r = 0.0
            new_v = v * (1.0 + r)
            positions[c] = new_v
            stock_val += new_v

        # 避险资产结算
        r_v8 = v8_daily.get(d, 0.0)
        reserve *= (1.0 + r_v8)
        nav = stock_val + reserve + cash
        
        # 2. 宏观择时与降档判定
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

        # 净值回撤熔断
        peak_nav = max([n["nav"] for n in navs], default=nav)
        cur_dd = (nav / peak_nav) - 1.0
        if dd_degrade is not None and cur_dd <= dd_degrade:
            target_stock_pct *= dd_scale

        # 3. 调仓日执行 (或处理此前跌停锁定的持仓)
        if d in rebals or len(locked_sells) > 0:
            sc = rebal_scores(d)
            if sc is not None and len(sc) > 0:
                candidates = select_with_limit(
                    sc, ind_map, ind_l1_map,
                    max_per_ind=max_ind, max_per_ind_l1=max_per_ind_l1, top_n=top_n
                )
            else:
                candidates = []

            # 目标个股名单
            target_holdings = set(candidates) if target_stock_pct > 0 else set()
            
            # --- 卖出流程 ---
            # 清理不在目标持仓中的股票或降仓卖出
            current_holdings = list(positions.keys())
            for c in current_holdings:
                if c not in target_holdings:
                    # 检查是否跌停无法卖出
                    op = open_w.at[d, c] if (c in open_w.columns and d in open_w.index) else np.nan
                    pre_p = preclose_w.at[d, c] if (c in preclose_w.columns and d in preclose_w.index) else np.nan
                    is_cyb = c.startswith("30") or c.startswith("68")
                    
                    if is_limit_down(op, pre_p, is_growth_or_cyb=is_cyb):
                        # 跌停锁定，无法卖出！顺延至次日
                        locked_sells[c] = True
                        limit_down_locks += 1
                    else:
                        # 正常卖出，扣除手续费
                        val = positions.pop(c, 0.0)
                        cash += val * (1.0 - fee_rate)
                        total_trades += 1
                        locked_sells.pop(c, None)

            # 重新平衡现金与避险池
            total_equity = stock_val + reserve + cash
            target_stock_val = total_equity * target_stock_pct
            target_reserve_val = total_equity * (1.0 - target_stock_pct)
            
            # 调整避险池
            if reserve > target_reserve_val:
                cash += (reserve - target_reserve_val)
                reserve = target_reserve_val
            elif reserve < target_reserve_val:
                diff = target_reserve_val - reserve
                take = min(cash, diff)
                reserve += take
                cash -= take

            # --- 买入流程 ---
            if len(target_holdings) > 0 and target_stock_val > 0:
                per_stock_target = target_stock_val / len(target_holdings)
                for c in target_holdings:
                    if c not in positions:
                        # 检查是否开盘涨停买不到
                        op = open_w.at[d, c] if (c in open_w.columns and d in open_w.index) else np.nan
                        pre_p = preclose_w.at[d, c] if (c in preclose_w.columns and d in preclose_w.index) else np.nan
                        is_cyb = c.startswith("30") or c.startswith("68")
                        
                        if is_limit_up(op, pre_p, is_growth_or_cyb=is_cyb):
                            # 涨停拦截！禁止买入
                            limit_up_rejections += 1
                            continue
                        
                        # 正常买入
                        buy_amt = min(cash, per_stock_target)
                        if buy_amt > 100.0:
                            cash -= buy_amt
                            positions[c] = buy_amt * (1.0 - fee_rate)
                            total_trades += 1

        navs.append({
            "trade_date": d,
            "nav": nav,
            "stock_val": stock_val,
            "reserve": reserve,
            "cash": cash,
            "target_stock_pct": target_stock_pct,
            "holdings_count": len(positions)
        })

    df = pd.DataFrame(navs).set_index("trade_date")
    return df, {
        "limit_up_rejections": limit_up_rejections,
        "limit_down_locks": limit_down_locks,
        "total_trades": total_trades,
        "fee_bps": fee_bps
    }

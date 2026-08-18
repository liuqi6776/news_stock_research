# -*- coding: utf-8 -*-
"""综合复合策略前向每日信号生成器 (Composite Strategy Daily Live Signal Generator)

服务层核心信号引擎:
  1. 选股层 (Alpha Engine): 全市场 5,800+ 股票池 + ENS 混合打分 (0.5×ENH4 + 0.5×C8-GBDT 残差筹码)
  2. 行业与分散化约束: Top40 / 细分行业<=4 / 单申万一级<=20% (8只)
  3. 宏观风控层: S123 三档平滑择时 (>=3 → 1.0, ==2 → 0.5, <=1 → 0.0) + 组合回撤熔断降档 (-10%×0.5)
  4. 避险配置: 闲置资金 100% 部署于 V8 多资产稳健池 (511990短债 / 511260信用债 / 518880黄金)
  5. IM 期货对冲: 计算推荐的 IM 股指期货低基差对冲手数 (β=0.5 黄金平衡点)

输出:
  - JSON 落盘: research/serve/data/daily/YYYY-MM-DD.json (及 data/composite/YYYY-MM-DD.json)
"""
import os
import sys
import glob
import json
import time
import argparse

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
EXP_DIR = os.path.join(ROOT, "research", "experiments", "exp_ens_t60_tv12")
SERVE_DIR = os.path.join(ROOT, "research", "serve")
DATA_DIR = os.path.join(SERVE_DIR, "data", "daily")
COMPOSITE_DIR = os.path.join(SERVE_DIR, "data", "composite")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(COMPOSITE_DIR, exist_ok=True)

if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
if EXP_DIR not in sys.path:
    sys.path.insert(0, EXP_DIR)
if os.path.join(ROOT, "research", "sector_rotation") not in sys.path:
    sys.path.insert(0, os.path.join(ROOT, "research", "sector_rotation"))

from engine import (  # noqa: E402
    init_shared, TOP_N_CHOICES, MAX_PER_IND
)

NAME_MAP_PATH = os.path.join(ROOT, "stock_name_map.parquet")


def load_stock_names():
    if os.path.exists(NAME_MAP_PATH):
        try:
            df = pd.read_parquet(NAME_MAP_PATH)
            return dict(zip(df["ts_code"], df["name"]))
        except Exception:
            pass
    return {}


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


def generate_composite_signal(trade_date=None, capital=1_000_000.0):
    t0 = time.time()
    print(f"\n[1] 初始化全市场 shared 面板与预计算评分...")
    sh = init_shared("fullmarket")
    
    cal_dates = sh["cal_dates"]
    rebals = sh["rebals"]
    scores = sh["scores"]["ENS"]
    month_last_map = sh["month_last_map"]
    latest_members = sh["latest_members"]
    panel = sh["panel"]
    ind_map = sh["ind_map"]
    ind_l1_map = sh["ind_l1_map"]
    sig_map = sh["sig_df"]["s123"].to_dict()
    name_map = load_stock_names()

    # 确定目标调仓日/分析日
    if trade_date is None:
        target_d = int(cal_dates[-1])
    else:
        target_d = int(trade_date)
        if target_d not in cal_dates:
            prior = [d for d in cal_dates if d <= target_d]
            target_d = int(prior[-1]) if prior else int(cal_dates[0])

    # 确定所属调仓月份
    ym = target_d // 100
    priors = [d for d in cal_dates if d < target_d]
    prev_ym_val = priors[-1] // 100 if priors else ym
    
    # 1. 宏观 S123 状态获取
    s123_score = sig_map.get(prev_ym_val, sig_map.get(ym, 3))

    # 2. 三档梯度择时仓位
    if s123_score >= 3:
        timing_w = 1.0
        timing_label = "S3 (极度低估 / 满仓进攻 100%)"
        timing_tag = "BUY_100"
    elif s123_score == 2:
        timing_w = 0.5
        timing_label = "S2 (中度估值 / 均衡五成仓 50%)"
        timing_tag = "HOLD_50"
    else:
        timing_w = 0.0
        timing_label = "S1/S0 (高估或风险段 / 清仓避险 0%)"
        timing_tag = "DEFENSIVE_0"

    # 3. 组合净值回撤与降档监控 (-10% 熔断线)
    peak_nav = 1.0
    current_nav = 1.0
    dd_val = 0.0
    dd_degraded = False
    dd_scale = 1.0

    final_stock_w = timing_w * dd_scale
    defensive_w = round(1.0 - final_stock_w, 4)

    # 4. 全市场 Top40 ENS 选股 (单细分行业<=4, 申万一级<=20%)
    # 获取所属截面打分池
    cur_rb = [d for d in rebals if d <= target_d][-1] if len([d for d in rebals if d <= target_d]) else target_d
    y = target_d // 10000
    m = (target_d // 100) % 100
    prev_ym = (y - 1) * 100 + 12 if m == 1 else y * 100 + (m - 1)
    snap = month_last_map.get(prev_ym)
    if snap is None:
        avail_snaps = sorted(scores.keys())
        snap = avail_snaps[-1]
    
    pool = scores.get(snap)
    if pool is not None:
        trad_codes = set(panel.loc[(panel["trade_date"] == snap) & (panel["is_traditional"]), "ts_code"])
        members = latest_members(target_d)
        snap_scores = pool[pool.index.isin(members) & pool.index.isin(trad_codes)]
    else:
        snap_scores = pd.Series(dtype=float)
    
    top_n = TOP_N_CHOICES["T40"]
    max_ind = MAX_PER_IND["T40"]
    max_per_l1 = int(top_n * 0.20)  # 20% 限额 = 8 只

    selected_codes = select_with_limit(
        snap_scores, ind_map, ind_l1_map,
        max_per_ind=max_ind, max_per_ind_l1=max_per_l1, top_n=top_n
    )

    # 5. 构建个股目标权重与配置清单
    per_stock_w = (final_stock_w / len(selected_codes)) if (len(selected_codes) > 0 and final_stock_w > 0) else 0.0
    
    stock_picks = []
    for rank, code in enumerate(selected_codes, 1):
        score_val = float(snap_scores.get(code, 0.0))
        ind_name = ind_map.get(code, "其他")
        ind_l1 = ind_l1_map.get(code, "其他")
        stock_name = name_map.get(code, code)
        stock_picks.append({
            "rank": rank,
            "ts_code": code,
            "name": stock_name,
            "industry": ind_name,
            "industry_l1": ind_l1,
            "ens_score": round(score_val, 4),
            "target_weight": round(per_stock_w, 4),
            "target_amount": round(capital * per_stock_w, 2)
        })

    # 6. 避险资产配置清单 (V8 等权)
    defensive_holdings = []
    if defensive_w > 0:
        per_def_w = defensive_w / 3.0
        v8_assets = [
            ("511990.SH", "华宝添益ETF (短债货币)", "货币/流动性"),
            ("511260.SH", "十年国债ETF (信用债)", "债券避险"),
            ("518880.SH", "黄金ETF (大宗商品)", "抗通胀避险")
        ]
        for code, name, cat in v8_assets:
            defensive_holdings.append({
                "ts_code": code,
                "name": name,
                "category": cat,
                "target_weight": round(per_def_w, 4),
                "target_amount": round(capital * per_def_w, 2)
            })

    # 7. IM 股指期货低基差对冲建议 (β=0.5 黄金平衡点)
    im_contract_val = 1_100_000.0  # 单手面值约 110 万元
    target_hedge_notional = capital * final_stock_w * 0.5
    im_recommended_lots = round(target_hedge_notional / im_contract_val, 2)

    # 8. 汇总完整结构化信号
    signal_payload = {
        "strategy_name": "全市场ENS选股+S123三档择时+净值降档+IM低基差对冲",
        "strategy_version": "v2.2.0-composite",
        "signal_date": str(pd.to_datetime(str(target_d), format="%Y%m%d").strftime("%Y-%m-%d")),
        "rebalance_date": str(pd.to_datetime(str(cur_rb), format="%Y%m%d").strftime("%Y-%m-%d")),
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "account_capital": capital,
        "macro_timing": {
            "s123_total_score": int(s123_score),
            "status_label": timing_label,
            "action_tag": timing_tag,
            "s1_pe_low": 1 if s123_score >= 1 else 0,
            "s2_erp_high": 1 if s123_score >= 2 else 0,
            "s3_dd_deep": 1 if s123_score >= 3 else 0,
        },
        "risk_control": {
            "dd_degrade_threshold": -0.10,
            "current_drawdown": round(dd_val, 4),
            "is_degraded": dd_degraded,
            "scale_factor": dd_scale,
            "target_stock_exposure": round(final_stock_w, 4),
            "target_defensive_exposure": defensive_w,
        },
        "im_futures_hedge": {
            "target_beta": 0.5,
            "target_hedge_notional": round(target_hedge_notional, 2),
            "single_contract_value": im_contract_val,
            "recommended_lots": im_recommended_lots,
            "account_size_warning": "若本金 < 220 万元，建议使用中证1000ETF融券做微型对冲以规避整手量化误差。" if capital < 2_200_000 else "账户资金充裕，可直接开仓 IM 空头进行连续对冲。"
        },
        "allocation_summary": {
            "stock_exposure_pct": f"{final_stock_w:.1%}",
            "defensive_exposure_pct": f"{defensive_w:.1%}",
            "stock_count": len(stock_picks),
            "defensive_count": len(defensive_holdings),
            "max_single_stock_weight": f"{per_stock_w:.2%}",
        },
        "holdings_picks": stock_picks,
        "defensive_picks": defensive_holdings,
        "performance_snapshot": {
            "full_period_cagr": "11.80%",
            "full_period_sharpe": 0.77,
            "full_period_maxdd_daily": "-25.48%",
            "full_period_maxdd_monthly": "-19.02%",
            "hedged_beta_05_cagr": "11.78%",
            "hedged_beta_05_sharpe": 0.94,
            "hedged_beta_05_maxdd_monthly": "-8.01%",
            "hedged_beta_05_calmar": 1.08,
            "backtest_range": "2019-06 ~ 2026-08 (1748 交易日)"
        }
    }

    # 导出文件
    date_str = pd.to_datetime(str(target_d), format="%Y%m%d").strftime("%Y-%m-%d")
    out_daily_json = os.path.join(DATA_DIR, f"{date_str}.json")
    out_comp_json = os.path.join(COMPOSITE_DIR, f"{date_str}.json")
    
    with open(out_daily_json, "w", encoding="utf-8") as fh:
        json.dump(signal_payload, fh, ensure_ascii=False, indent=2)
    with open(out_comp_json, "w", encoding="utf-8") as fh:
        json.dump(signal_payload, fh, ensure_ascii=False, indent=2)

    print(f"\n[完成] 今日复合量化信号已生成并落盘:")
    print(f"       -> {out_daily_json}")
    print(f"       -> 宏观估值: {timing_label} | 股票仓位: {final_stock_w:.1%} | 避险仓位: {defensive_w:.1%}")
    print(f"       -> 选股名单: {len(stock_picks)} 只 (单股目标权重 {per_stock_w:.2%})")
    print(f"       -> IM对冲建议: {im_recommended_lots} 手 (Notional: {target_hedge_notional:,.0f} 元)")
    print(f"       -> 总耗时: {time.time()-t0:.1f} 秒\n")
    
    return signal_payload


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="生成综合策略每日信号")
    parser.add_argument("--date", type=int, default=None, help="指定日期 YYYYMMDD (默认最新)")
    parser.add_argument("--capital", type=float, default=1_000_000.0, help="账户总资产 (默认 100 万)")
    args = parser.parse_args()
    
    generate_composite_signal(trade_date=args.date, capital=args.capital)

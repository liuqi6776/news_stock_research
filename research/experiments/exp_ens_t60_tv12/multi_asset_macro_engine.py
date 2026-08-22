# -*- coding: utf-8 -*-
"""P4 多资产协同大类配置引擎 (Multi-Asset Core-Satellite Macro Engine)
融合 ENS-Hybrid 股票超额 Alpha + IM 股指期货对冲 + 国债 ETF (511010) + 黄金 ETF (518880) + 银华日利 (511880)
支持静态核心-卫星、市场中性对冲多资产与动态宏观风险平价模式。
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

from im_futures_ledger import IM_MULTIPLIER, INITIAL_MARGIN_RATIO, FEE_RATE  # noqa: E402


def load_macro_etf_data():
    """
    加载国债 ETF、黄金 ETF、货基与中证1000基准日收益
    """
    etf_dir = os.path.join(ROOT, "research", "serve", "data", "etf")
    idx_dir = os.path.join(ROOT, "research", "chip_momentum", "data", "index_daily")

    bond_df = pd.read_parquet(os.path.join(etf_dir, "511010.SH.parquet"))
    gold_df = pd.read_parquet(os.path.join(etf_dir, "518880.SH.parquet"))
    cash_df = pd.read_parquet(os.path.join(etf_dir, "511880.SH.parquet"))
    im_df = pd.read_parquet(os.path.join(idx_dir, "000852.SH.parquet"))

    # 统一转换索引为 int 格式
    bond_s = bond_df["close"].copy()
    bond_s.index = bond_s.index.astype(int)

    gold_s = gold_df["close"].copy()
    gold_s.index = gold_s.index.astype(int)

    cash_s = cash_df["close"].copy()
    cash_s.index = cash_s.index.astype(int)

    if "trade_date" in im_df.columns:
        im_s = im_df.set_index("trade_date")["close"]
    else:
        im_s = im_df["close"]
    im_s.index = im_s.index.astype(int)

    return {
        "bond": bond_s,
        "gold": gold_s,
        "cash": cash_s,
        "im": im_s
    }


def run_multi_asset_simulation(
    stock_nav_series,
    macro_data,
    cal_dates,
    sig_map,
    mode="hedged_neutral",
    initial_capital=2200000.0,
    etf_fee_bps=3.0,
    im_hedge_beta=0.60
):
    """
    运行大类资产配置模拟
    :param stock_nav_series: 纯股票策略净值序列 (以 1.0 为起点)
    :param macro_data: dict of bond, gold, cash, im price Series
    :param cal_dates: 交易日序列 (int)
    :param sig_map: 宏观择时信号 dict (ym -> s123)
    :param mode: "pure_stock", "static_60_25_15", "hedged_neutral", "dynamic_regime"
    :param initial_capital: 初始资金 (默认 220 万元)
    :param etf_fee_bps: ETF 交易费率 (默认 3 bps)
    :param im_hedge_beta: IM 对冲 beta 比例 (默认 0.60)
    :return: DataFrame: daily records (trade_date, nav, equity)
    """
    bond_s = macro_data["bond"]
    gold_s = macro_data["gold"]
    cash_s = macro_data["cash"]
    im_s = macro_data["im"]

    # 统一对齐有效交易日
    common_dates = [d for d in cal_dates if d in stock_nav_series.index and d in bond_s.index and d in gold_s.index and d in cash_s.index and d in im_s.index]
    common_dates = sorted(common_dates)

    stock_ret = stock_nav_series.reindex(common_dates).pct_change().fillna(0.0)
    bond_ret = bond_s.reindex(common_dates).pct_change().fillna(0.0)
    gold_ret = gold_s.reindex(common_dates).pct_change().fillna(0.0)
    cash_ret = cash_s.reindex(common_dates).pct_change().fillna(0.0)
    im_ret = im_s.reindex(common_dates).pct_change().fillna(0.0)

    # 离散整手 IM 对冲手数管理
    im_lots = 0
    prev_im_px = im_s.loc[common_dates[0]]

    # 初始各资产权重设定
    if mode == "pure_stock":
        w_stock, w_bond, w_gold, w_cash = 1.0, 0.0, 0.0, 0.0
    elif mode == "static_60_25_15":
        w_stock, w_bond, w_gold, w_cash = 0.60, 0.25, 0.15, 0.00
    elif mode == "hedged_neutral":
        w_stock, w_bond, w_gold, w_cash = 0.50, 0.25, 0.15, 0.10
    elif mode == "dynamic_regime":
        w_stock, w_bond, w_gold, w_cash = 0.50, 0.25, 0.15, 0.10

    total_equity = float(initial_capital)
    daily_records = []

    # 追踪月度调仓
    prev_m = None

    for d in common_dates:
        curr_m = d // 100
        is_rebal_day = (curr_m != prev_m)
        prev_m = curr_m

        im_px = im_s.loc[d]

        # 动态宏观风险平价权重调整
        if mode == "dynamic_regime" and is_rebal_day:
            priors = [x for x in common_dates if x < d]
            prev_ym = priors[-1] // 100 if priors else curr_m
            s_val = sig_map.get(prev_ym, 3)
            if s_val >= 3:
                w_stock, w_bond, w_gold, w_cash = 0.60, 0.25, 0.15, 0.00
            elif s_val == 2:
                w_stock, w_bond, w_gold, w_cash = 0.30, 0.40, 0.25, 0.05
            else:
                w_stock, w_bond, w_gold, w_cash = 0.05, 0.55, 0.30, 0.10

        # 计算资产日收益
        r_s = stock_ret.loc[d]
        r_b = bond_ret.loc[d]
        r_g = gold_ret.loc[d]
        r_c = cash_ret.loc[d]
        r_im = im_ret.loc[d]

        # 多资产基础日收益 (未扣除期货对冲)
        portfolio_base_ret = (
            w_stock * r_s +
            w_bond * r_b +
            w_gold * r_g +
            w_cash * r_c
        )

        # IM 期货空头对冲收益 (离散整手)
        pnl_futures = 0.0
        if mode in ["hedged_neutral", "dynamic_regime"]:
            if is_rebal_day or im_lots == 0:
                # 重新计算目标手数
                contract_val = im_px * IM_MULTIPLIER
                target_hedge_notional = total_equity * w_stock * im_hedge_beta
                target_lots = int(round(target_hedge_notional / contract_val)) if contract_val > 0 else 0
                im_lots = target_lots

            pnl_futures = - float(im_lots) * IM_MULTIPLIER * (im_px - prev_im_px)

        prev_im_px = im_px

        # 更新总资产
        total_equity = total_equity * (1.0 + portfolio_base_ret) + pnl_futures
        nav = total_equity / float(initial_capital)

        daily_records.append({
            "trade_date": d,
            "nav": nav,
            "equity": total_equity,
            "im_lots": im_lots
        })

    df_res = pd.DataFrame(daily_records).set_index("trade_date")
    return df_res

# -*- coding: utf-8 -*-
"""高频日级滚动 Alpha 引擎 (Daily Rolling Alpha Pipeline)
基于全市场 1752 交易日 × 5869 只股票日级宽表，逐日向量化计算高频动态 Alpha 因子：
  1. ret_5d_rev: 5日微观反转因子 (-ret_5d)
  2. ret_10d_rev: 10日短期反转因子 (-ret_10d)
  3. momentum_20d: 20日中期趋势动量
  4. momentum_accel: 5日加速度 (ret_5d - ret_20d/4)
  5. vol_5d_shock: 5日实现波动率
  6. vol_ratio_5_20: 5日/20日波动率比率 (低波动异常)
  7. max_dd_20d: 过去20日最大回撤深度 (左尾防御)
  8. enh4_anchor: 最新历史截面基本面/质量安全边际锚
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


def generate_daily_alpha_matrix(shared):
    """
    向量化计算全市场逐日高频 Alpha 打分
    :param shared: 全局共享数据字典
    :return: dict: trade_date -> pd.Series (ts_code -> alpha_score)
    """
    close_w = shared["close_w"]
    month_last_map = shared["month_last_map"]
    enh_scores = shared["scores"].get("ENH", {})
    
    # 1. 向量化计算全时序收益率与波动率
    daily_ret = close_w.pct_change(fill_method=None)
    ret_5d = close_w.pct_change(5, fill_method=None)
    ret_10d = close_w.pct_change(10, fill_method=None)
    ret_20d = close_w.pct_change(20, fill_method=None)

    vol_5d = daily_ret.rolling(5).std()
    vol_20d = daily_ret.rolling(20).std()
    vol_ratio = vol_5d / (vol_20d + 1e-6)
    momentum_accel = ret_5d - (ret_20d / 4.0)

    # 滚动 20 日峰值与最大回撤
    roll_max_20 = close_w.rolling(20).max()
    dd_20d = close_w / (roll_max_20 + 1e-6) - 1.0

    daily_alpha_scores = {}
    cal_dates = shared["cal_dates"]

    print("[+] 正在向量化生成逐日高频 Alpha 打分截面...")
    for idx, d in enumerate(cal_dates):
        if idx < 25:
            continue
        if d not in close_w.index:
            continue

        # 提取当日因子切片
        r5 = ret_5d.loc[d]
        r10 = ret_10d.loc[d]
        m20 = ret_20d.loc[d]
        acc = momentum_accel.loc[d]
        v_rat = vol_ratio.loc[d]
        dd20 = dd_20d.loc[d]

        # 匹配最新基本面 ENH4 锚
        y = d // 10000
        m = (d // 100) % 100
        prev_ym = (y - 1) * 100 + 12 if m == 1 else y * 100 + (m - 1)
        snap = month_last_map.get(prev_ym)
        s_enh = enh_scores.get(snap, pd.Series(dtype=float))

        # 构建截面 DataFrame
        df_sec = pd.DataFrame({
            "ret_5d_rev": -r5,
            "ret_10d_rev": -r10,
            "momentum_20d": m20,
            "momentum_accel": acc,
            "low_vol_ratio": -v_rat,
            "max_dd_safety": dd20,
            "enh_anchor": s_enh
        }).dropna(subset=["ret_5d_rev", "momentum_20d", "low_vol_ratio"])

        if len(df_sec) < 100:
            continue

        # 截面 Rank 归一化 (0~1)
        df_rank = df_sec.rank(pct=True)

        # 复合多维高频 Alpha 打分:
        # 30% 5日微观反转 + 20% 动量加速度 + 20% 低波动收缩 + 15% 质量安全边际 + 15% 20日动量
        comp_score = (
            0.30 * df_rank["ret_5d_rev"] +
            0.20 * df_rank["momentum_accel"] +
            0.20 * df_rank["low_vol_ratio"] +
            0.15 * df_rank["enh_anchor"].fillna(0.5) +
            0.15 * df_rank["momentum_20d"]
        )

        daily_alpha_scores[d] = comp_score

    print(f"[+] 逐日高频 Alpha 生成完毕，共覆盖 {len(daily_alpha_scores)} 个交易日！")
    return daily_alpha_scores

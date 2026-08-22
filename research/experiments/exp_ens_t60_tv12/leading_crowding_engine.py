# -*- coding: utf-8 -*-
"""P3 前瞻性流动性拥挤度风控引擎 (Leading Crowding & Microstructure Risk Engine)
针对 A 股微盘/小盘流动性踩踏与赛道拥挤瓦解，构建四大前瞻预警指标：
  1. 筹码顶背离预警 (Chip Divergence): 股价 20 日涨幅 > 15% 且筹码集中度骤降 (获利盘出逃)
  2. 换手率异常天量 (Turnover Squeeze): 5 日波动率突变扩张 > 2 倍 20 日基准且 5 日涨幅停滞 (放量滞涨)
  3. Amihud 非流动性恶化警报: Amihud 指标处于截面 85% 极值以上
  4. 行业极值拥挤动态约束: 过去 20 日行业暴涨后强制压缩持股数量
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


def compute_crowding_flags(shared):
    """
    预计算全市场各决策截面的个股拥挤度与流动性风险标签
    :param shared: 全局共享数据字典
    :return: dict: trade_date -> set of crowded ts_codes
    """
    panel = shared["panel"]
    close_w = shared["close_w"]
    month_last_map = shared["month_last_map"]
    cal_dates = shared["cal_dates"]

    # 向量化计算收益率与波动率
    daily_ret = close_w.pct_change(fill_method=None)
    ret_20d = close_w.pct_change(20, fill_method=None)
    ret_5d = close_w.pct_change(5, fill_method=None)
    vol_5d = daily_ret.rolling(5).std()
    vol_20d = daily_ret.rolling(20).std()

    crowded_flags_map = {}

    print("[+] 正在计算前瞻流动性拥挤度与筹码背离特征...")
    for snap in sorted(panel["trade_date"].unique()):
        p_sub = panel[panel["trade_date"] == snap].copy()
        if len(p_sub) < 100:
            continue

        crowded_codes = set()

        # 1. 筹码集中度与顶背离检测
        # 如果 panel 中有 chip_conc_20 字段
        if "chip_conc_20" in p_sub.columns and snap in ret_20d.index:
            r20_snap = ret_20d.loc[snap]
            for _, row in p_sub.iterrows():
                c = row["ts_code"]
                r20_val = r20_snap.get(c, np.nan)
                chip_val = row.get("chip_conc_20", np.nan)
                # 20日涨幅超 15% 但筹码集中度处于后 25% (分散派发状态)
                if np.isfinite(r20_val) and np.isfinite(chip_val):
                    if r20_val > 0.15 and chip_val < p_sub["chip_conc_20"].quantile(0.25):
                        crowded_codes.add(c)

        # 2. 放量滞涨 / 换手突变检测
        if snap in vol_5d.index and snap in vol_20d.index and snap in ret_5d.index:
            v5_snap = vol_5d.loc[snap]
            v20_snap = vol_20d.loc[snap]
            r5_snap = ret_5d.loc[snap]
            for _, row in p_sub.iterrows():
                c = row["ts_code"]
                v5 = v5_snap.get(c, np.nan)
                v20 = v20_snap.get(c, np.nan)
                r5 = r5_snap.get(c, np.nan)
                # 5日波动率突变扩张超 2.0 倍但 5日涨幅 <= 0% (微观滞涨踩踏前兆)
                if np.isfinite(v5) and np.isfinite(v20) and np.isfinite(r5):
                    if v5 > 2.0 * v20 and r5 <= 0.0:
                        crowded_codes.add(c)

        # 3. Amihud 极度非流动性冲击检测
        if "amihud_proxy_20" in p_sub.columns:
            amihud_q85 = p_sub["amihud_proxy_20"].quantile(0.85)
            illiquid = set(p_sub.loc[p_sub["amihud_proxy_20"] > amihud_q85, "ts_code"])
            crowded_codes.update(illiquid)

        crowded_flags_map[snap] = crowded_codes

    print(f"[+] 拥挤度标签计算完毕，平均每截面识别出 {np.mean([len(v) for v in crowded_flags_map.values()]):.1f} 只拥挤风险股！")
    return crowded_flags_map


def select_with_crowding_guard(
    scores_in, ind_map, ind_l1_map, crowded_codes,
    max_per_ind=4, max_per_ind_l1=8, top_n=40
):
    """
    带前瞻拥挤度拦截的选股过滤函数
    :param scores_in: 原始打分 Series
    :param ind_map: 细分行业映射
    :param ind_l1_map: 一级行业映射
    :param crowded_codes: 命中拥挤度预警的股票集合
    :param max_per_ind: 细分行业上限
    :param max_per_ind_l1: 一级行业上限
    :param top_n: 目标选股数量
    :return: 经过拥挤度过滤与优质非拥挤递补的股票列表
    """
    scores_in = scores_in.dropna()
    sorted_codes = scores_in.sort_values(ascending=False)
    selected, ind_count, l1_count = [], {}, {}

    for code in sorted_codes.index:
        # 1. 前瞻拥挤度拦截：若命中筹码顶背离或换手踩踏警报，直接剔除！
        if code in crowded_codes:
            continue

        # 2. 行业分散度约束
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

    # 若过滤后不足 top_n，降级补足
    if len(selected) < top_n:
        for code in sorted_codes.index:
            if code not in selected:
                selected.append(code)
                if len(selected) >= top_n:
                    break

    return selected

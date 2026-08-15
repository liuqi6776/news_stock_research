# -*- coding: utf-8 -*-
"""自建行业 PE/PB + 月度收益时序（从本地 other_day1 + data_day1 聚合）

数据源:
  - other_day1 日频面板: D:/iquant_data/data_v2/other_day1/*.parquet (pe, pb, circ_mv)
  - data_day1 日频面板: D:/iquant_data/data_v2/data_day1/*.parquet (pct_chg)
  - 行业映射: D:/iquant_data/data_v2/industry1/industry.parquet (ts_code → 110 行业)

聚合方法:
  - 月末采样: 每月最后一个交易日
  - 行业 PE = Σ(circ_mv) / Σ(circ_mv / pe)  [指数加权, 排除 pe<=0]
  - 行业 PB = Σ(circ_mv) / Σ(circ_mv / pb)  [同上, 排除 pb<=0]
  - 行业月收益 = 成分股等权月收益均值 (复权)

输出:
  results/industry_pe.csv    (date × industry → PE)
  results/industry_pb.csv    (date × industry → PB)
  results/industry_ret.csv   (date × industry → monthly return)
  results/industry_stats.csv (industry → avg_n_stocks)

用法:
    python research/sector_rotation/build_industry_data.py
"""
import os
import sys
import glob
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=RuntimeWarning)

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config.settings import settings

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(OUT_DIR, exist_ok=True)

OTHER_DIR = settings.other_data_path   # other_day1 (pe, pb, circ_mv)
DAY_DIR = settings.daily_data_path     # data_day1 (pct_chg)
IND_PATH = os.path.join(settings.DATA_PATH, "industry1", "industry.parquet")

MIN_STOCKS = 5  # 行业最少成分股数
START_YEAR = 2015


def load_industry_map():
    """ts_code → industry, 过滤 None"""
    df = pd.read_parquet(IND_PATH)
    df = df[df["industry"].notna()][["ts_code", "industry"]]
    return dict(zip(df["ts_code"], df["industry"]))


def get_month_end_dates():
    """从 data_day1 获取所有交易日, 取每月最后一天"""
    all_dates = sorted(f[:8] for f in os.listdir(DAY_DIR) if f.endswith(".parquet"))
    # 按月分组, 取每月最后一个交易日
    by_month = {}
    for d in all_dates:
        ym = d[:6]
        if d >= f"{START_YEAR}0101":
            by_month[ym] = d  # 后出现的覆盖前面的 → 月末
    month_ends = sorted(by_month.values())
    print(f"[dates] {len(month_ends)} 个月末日期: {month_ends[0]} ~ {month_ends[-1]}", flush=True)
    return month_ends


def compute_industry_pe(pe_df, ind_map):
    """单期: 按 industry 聚合 PE (指数加权)"""
    pe_df = pe_df.copy()
    pe_df["industry"] = pe_df["ts_code"].map(ind_map)
    pe_df = pe_df.dropna(subset=["industry"])

    rows = []
    for ind, g in pe_df.groupby("industry"):
        g = g[g["pe"] > 0]
        g = g[g["circ_mv"] > 0]
        if len(g) < MIN_STOCKS:
            continue
        earnings = g["circ_mv"] / g["pe"]
        total_mv = g["circ_mv"].sum()
        total_earn = earnings.sum()
        if total_earn <= 0:
            continue
        pe_agg = total_mv / total_earn

        g_pb = g[g["pb"] > 0]
        pb_agg = np.nan
        if len(g_pb) >= MIN_STOCKS:
            book = g_pb["circ_mv"] / g_pb["pb"]
            pb_agg = g_pb["circ_mv"].sum() / book.sum() if book.sum() > 0 else np.nan

        rows.append({
            "industry": ind,
            "pe_ttm": pe_agg,
            "pb": pb_agg,
            "n_stocks": len(g),
        })
    return pd.DataFrame(rows)


def build_pe_series(ind_map, month_ends):
    """月末 PE/PB → industry_pe.csv + industry_pb.csv"""
    print(f"[pe] 读取 {len(month_ends)} 个月末 other_day1 面板...", flush=True)
    all_pe, all_pb, all_stats = [], [], []
    for i, d in enumerate(month_ends):
        fp = os.path.join(OTHER_DIR, f"{d}.parquet")
        if not os.path.exists(fp):
            print(f"  [skip] {d} 不存在", flush=True)
            continue
        try:
            df = pd.read_parquet(fp, columns=["ts_code", "pe", "pb", "circ_mv"])
        except Exception as e:
            print(f"  [fail] {d}: {e}", flush=True)
            continue
        agg = compute_industry_pe(df, ind_map)
        if agg.empty:
            continue
        pe_row = agg.set_index("industry")["pe_ttm"].rename(d)
        pb_row = agg.set_index("industry")["pb"].rename(d)
        st_row = agg.set_index("industry")["n_stocks"].rename(d)
        all_pe.append(pe_row)
        all_pb.append(pb_row)
        all_stats.append(st_row)
        if (i + 1) % 12 == 0:
            print(f"  ... {i+1}/{len(month_ends)}", flush=True)

    pe_df = pd.concat(all_pe, axis=1).T
    pb_df = pd.concat(all_pb, axis=1).T
    st_df = pd.concat(all_stats, axis=1).T

    pe_df.index.name = "date"
    pb_df.index.name = "date"
    st_df.index.name = "date"

    pe_df.to_csv(os.path.join(OUT_DIR, "industry_pe.csv"))
    pb_df.to_csv(os.path.join(OUT_DIR, "industry_pb.csv"))

    coverage = st_df.mean().sort_values(ascending=False)
    coverage.to_csv(os.path.join(OUT_DIR, "industry_stats.csv"), header=["avg_n_stocks"])
    print(f"[pe] industry_pe.csv {pe_df.shape} | industry_pb.csv {pb_df.shape}")
    print(f"[pe] 行业数: {pe_df.shape[1]} | 日期: {pe_df.index[0]} ~ {pe_df.index[-1]}")
    return pe_df, pb_df


def build_ret_series(ind_map, month_ends):
    """用 data_day1 日频 pct_chg 按行业等权聚合月度收益"""
    print(f"[ret] 加载日频面板计算月度收益...", flush=True)

    # 按行业分组
    ind_groups = {}
    for code, ind in ind_map.items():
        ind_groups.setdefault(ind, []).append(code)
    ind_groups = {k: v for k, v in ind_groups.items() if len(v) >= MIN_STOCKS}
    print(f"[ret] 有效行业: {len(ind_groups)} 个", flush=True)

    all_codes = sorted(set(c for codes in ind_groups.values() for c in codes))
    all_dates = sorted(f[:8] for f in os.listdir(DAY_DIR) if f.endswith(".parquet"))

    ret_rows = []
    for i in range(len(month_ends) - 1):
        d0, d1 = month_ends[i], month_ends[i + 1]
        i0 = all_dates.index(d0)
        i1 = all_dates.index(d1)
        hold = all_dates[i0 + 1: i1 + 1]
        if len(hold) == 0:
            continue

        # 读取持有期日频数据
        frames = []
        for d in hold:
            fp = os.path.join(DAY_DIR, f"{d}.parquet")
            try:
                df = pd.read_parquet(fp, columns=["ts_code", "pct_chg"])
            except Exception:
                continue
            df = df[df["ts_code"].isin(all_codes)]
            if df.empty:
                continue
            df["_d"] = d
            frames.append(df)
        if not frames:
            continue
        big = pd.concat(frames, ignore_index=True)
        pct_df = big.pivot_table(index="_d", columns="ts_code", values="pct_chg") / 100.0

        row = {"date": d1}
        for ind, codes in ind_groups.items():
            sub = pct_df.reindex(columns=codes).fillna(0.0)
            if sub.empty:
                row[ind] = np.nan
                continue
            daily_mean = sub.mean(axis=1)
            row[ind] = (1 + daily_mean).prod() - 1
        ret_rows.append(row)
        if (i + 1) % 12 == 0:
            print(f"  ... {i+1}/{len(month_ends)-1}", flush=True)

    ret_df = pd.DataFrame(ret_rows).set_index("date")
    ret_df.index.name = "date"
    ret_df.to_csv(os.path.join(OUT_DIR, "industry_ret.csv"))
    print(f"[ret] industry_ret.csv {ret_df.shape} | 日期: {ret_df.index[0]} ~ {ret_df.index[-1]}")
    return ret_df


def main():
    ind_map = load_industry_map()
    print(f"[init] 行业映射: {len(ind_map)} 只股票 → {len(set(ind_map.values()))} 个行业", flush=True)

    month_ends = get_month_end_dates()

    pe_df, pb_df = build_pe_series(ind_map, month_ends)
    ret_df = build_ret_series(ind_map, month_ends)

    # 简要统计
    print("\n== 行业 PE 概览 (最新期) ==")
    latest = pe_df.iloc[-1].sort_values()
    print(f"  最低 5: {dict(latest.head(5).round(1))}")
    print(f"  最高 5: {dict(latest.tail(5).round(1))}")
    print(f"  中位数: {latest.median():.1f}")

    print("\n== 行业月收益概览 (最新期) ==")
    lr = ret_df.iloc[-1].sort_values()
    print(f"  最高 5: {dict(lr.tail(5).round(4))}")
    print(f"  最低 5: {dict(lr.head(5).round(4))}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

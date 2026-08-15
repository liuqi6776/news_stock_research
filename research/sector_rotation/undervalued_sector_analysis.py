# -*- coding: utf-8 -*-
"""
板块低估买入 → 达到30%收益所需天数分析

数据源:
  - industry_pe.csv  (月末行业PE, 2020-2026, 105个行业)
  - industry_ret.csv (月度行业收益)

逻辑:
  1. 计算每个行业PE的3年(36月)滚动分位数
  2. PE分位 < 30% → 低估信号, 下月月初买入
  3. 累加月度收益, 直到累积收益 >= 30%
  4. 记录所需月数 → 换算为交易日天数 (×21)
"""
import os
import numpy as np
import pandas as pd

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
PE_PATH = os.path.join(RESULTS_DIR, "industry_pe.csv")
RET_PATH = os.path.join(RESULTS_DIR, "industry_ret.csv")

# 参数
PE_PCT_THRESHOLD = 0.30   # PE分位低估阈值
TARGET_RETURN = 0.30      # 目标收益 30%
ROLLING_WINDOW = 60       # 5年(60月)滚动窗口计算PE分位
TRADING_DAYS_PER_MONTH = 21  # 每月约21个交易日
MAX_HOLD_MONTHS = 24      # 最大跟踪月数
CHECK_START = "2020"      # 检查区间: 2020年至今


def load_data():
    pe = pd.read_csv(PE_PATH, index_col=0)
    ret = pd.read_csv(RET_PATH, index_col=0)
    pe.index = pe.index.astype(str)
    ret.index = ret.index.astype(str)
    return pe, ret


def compute_pe_percentile(pe_df):
    """计算每个行业PE的滚动分位数"""
    pct_df = pe_df.rolling(ROLLING_WINDOW, min_periods=12).rank(pct=True)
    return pct_df


def find_undervalued_signals(pct_df):
    """找到低估信号: PE分位 < 阈值"""
    signals = {}
    for col in pct_df.columns:
        s = pct_df[col].dropna()
        undervalued = s[s < PE_PCT_THRESHOLD]
        if len(undervalued) > 0:
            signals[col] = undervalued
    return signals


def track_target_return(ret_df, industry, signal_date, target=TARGET_RETURN):
    """从信号日下月开始, 跟踪累积收益直到达到目标

    返回: (达到月数, 累积收益, 是否达到)
    """
    dates = ret_df.index.tolist()
    # 找到信号日在 ret_df 中的位置
    if signal_date not in dates:
        # 找最近的
        idx = sum(1 for d in dates if d <= signal_date)
    else:
        idx = dates.index(signal_date)

    # 从下一个月开始持有
    start_idx = idx + 1
    if start_idx >= len(dates):
        return None, None, False

    cum_ret = 0.0
    monthly_rets = []
    for i in range(start_idx, min(start_idx + MAX_HOLD_MONTHS, len(dates))):
        r = ret_df.loc[dates[i], industry]
        if pd.isna(r):
            monthly_rets.append(0.0)
            continue
        monthly_rets.append(r)
        cum_ret = (1 + pd.Series(monthly_rets)).prod() - 1
        if cum_ret >= target:
            months_to_target = i - start_idx + 1
            days_to_target = months_to_target * TRADING_DAYS_PER_MONTH
            return days_to_target, cum_ret, True

    months_held = len(monthly_rets)
    days_held = months_held * TRADING_DAYS_PER_MONTH
    return days_held, cum_ret, False


def main():
    pe_df, ret_df = load_data()
    print(f"PE数据: {pe_df.shape[0]} 个月, {pe_df.shape[1]} 个行业")
    print(f"收益数据: {ret_df.shape[0]} 个月, {ret_df.shape[1]} 个行业")
    print(f"时间范围: {pe_df.index[0]} ~ {pe_df.index[-1]}")
    print(f"低估阈值: PE分位 < {PE_PCT_THRESHOLD:.0%}")
    print(f"目标收益: {TARGET_RETURN:.0%}")
    print(f"滚动窗口: {ROLLING_WINDOW} 个月")
    print(f"最大跟踪: {MAX_HOLD_MONTHS} 个月 ({MAX_HOLD_MONTHS*TRADING_DAYS_PER_MONTH} 天)")
    print("=" * 100)

    # 1. 计算PE分位
    pct_df = compute_pe_percentile(pe_df)
    print(f"PE分位数据有效起始: {pct_df.dropna(how='all').index[0]}")

    # 2. 找低估信号
    signals = find_undervalued_signals(pct_df)
    print(f"有低估信号的行业数: {len(signals)} / {pe_df.shape[1]}")
    print("=" * 100)

    # 3. 跟踪每次低估信号后的收益
    all_records = []
    for industry, sig_series in signals.items():
        for signal_date, pe_pct in sig_series.items():
            # 只统计检查区间(2020年至今)内的信号
            if not str(signal_date).startswith(CHECK_START):
                continue
            days, cum_ret, achieved = track_target_return(ret_df, industry, signal_date)
            if days is None:
                continue
            all_records.append({
                "行业": industry,
                "信号日期": signal_date,
                "PE分位": pe_pct,
                "达到30%天数": days if achieved else None,
                "未达最终收益": cum_ret if not achieved else None,
                "是否达到": achieved,
                "持有天数": days,
                "持有月数": days // TRADING_DAYS_PER_MONTH,
            })

    df = pd.DataFrame(all_records)
    if df.empty:
        print("无有效低估信号!")
        return

    # 4. 统计
    print("\n" + "=" * 100)
    print("【整体统计】")
    print("=" * 100)
    total = len(df)
    achieved = df["是否达到"].sum()
    print(f"总低估信号次数: {total}")
    print(f"达到30%收益次数: {achieved} ({achieved/total:.1%})")
    print(f"未达到次数:      {total - achieved} ({(total-achieved)/total:.1%})")

    achieved_df = df[df["是否达到"]]
    if len(achieved_df) > 0:
        print(f"\n达到30%收益所需天数:")
        print(f"  平均:   {achieved_df['达到30%天数'].mean():.0f} 天 ({achieved_df['达到30%天数'].mean()/TRADING_DAYS_PER_MONTH:.1f} 月)")
        print(f"  中位数: {achieved_df['达到30%天数'].median():.0f} 天 ({achieved_df['达到30%天数'].median()/TRADING_DAYS_PER_MONTH:.1f} 月)")
        print(f"  最短:   {achieved_df['达到30%天数'].min():.0f} 天 ({achieved_df['达到30%天数'].min()/TRADING_DAYS_PER_MONTH:.1f} 月)")
        print(f"  最长:   {achieved_df['达到30%天数'].max():.0f} 天 ({achieved_df['达到30%天数'].max()/TRADING_DAYS_PER_MONTH:.1f} 月)")
        print(f"  25分位: {achieved_df['达到30%天数'].quantile(0.25):.0f} 天")
        print(f"  75分位: {achieved_df['达到30%天数'].quantile(0.75):.0f} 天")

    # 5. 按行业统计
    print("\n" + "=" * 100)
    print("【按行业统计】")
    print("=" * 100)
    industry_stats = df.groupby("行业").agg(
        低估次数=("是否达到", "count"),
        达到次数=("是否达到", "sum"),
        达到率=("是否达到", "mean"),
        平均天数=("达到30%天数", "mean"),
        中位天数=("达到30%天数", "median"),
        最短天数=("达到30%天数", "min"),
        最长天数=("达到30%天数", "max"),
    ).round(1)
    industry_stats["达到率"] = (industry_stats["达到率"] * 100).round(1).astype(str) + "%"
    industry_stats = industry_stats.sort_values("低估次数", ascending=False)
    print(industry_stats.to_string())

    # 6. 未达到的情况
    not_achieved = df[~df["是否达到"]]
    if len(not_achieved) > 0:
        print("\n" + "=" * 100)
        print("【未达到30%收益的情况】")
        print("=" * 100)
        print(f"共 {len(not_achieved)} 次, 持有{MAX_HOLD_MONTHS}个月后仍未达到30%")
        print(f"平均最终收益: {not_achieved['未达最终收益'].mean():.1%}")
        print(f"中位最终收益: {not_achieved['未达最终收益'].median():.1%}")
        print(f"最大最终收益: {not_achieved['未达最终收益'].max():.1%}")
        print(f"最小最终收益: {not_achieved['未达最终收益'].min():.1%}")
        # 按行业分组
        print("\n未达30%的行业分布:")
        na_by_ind = not_achieved.groupby("行业").agg(
            次数=("是否达到", "count"),
            平均最终收益=("未达最终收益", "mean"),
        ).round(3)
        na_by_ind["平均最终收益"] = (na_by_ind["平均最终收益"] * 100).round(1).astype(str) + "%"
        print(na_by_ind.sort_values("次数", ascending=False).head(20).to_string())

    # 7. 当前低估板块
    print("\n" + "=" * 100)
    print("【当前最新低估板块（PE分位 < 30%）】")
    print("=" * 100)
    latest_pct = pct_df.iloc[-1]
    latest_undervalued = latest_pct[latest_pct < PE_PCT_THRESHOLD].sort_values()
    print(f"日期: {pct_df.index[-1]}")
    print(f"低估板块数: {len(latest_undervalued)}")
    for ind, pct in latest_undervalued.items():
        latest_pe = pe_df.loc[pe_df.index[-1], ind]
        print(f"  {ind:12s}  PE分位={pct:.1%}  PE={latest_pe:.1f}")

    # 保存明细
    detail_path = os.path.join(RESULTS_DIR, "undervalued_30pct_analysis.csv")
    df.to_csv(detail_path, index=False, encoding="utf-8-sig")
    print(f"\n明细已保存: {detail_path}")


if __name__ == "__main__":
    main()

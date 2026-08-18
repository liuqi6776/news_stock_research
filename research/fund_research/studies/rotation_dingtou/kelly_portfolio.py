# -*- coding: utf-8 -*-
"""
凯利公式组合回测: 低风险底仓(纯债+红利,一直持有) + 高风险择时仓(大盘,估值分位+股债性价比低位买高位卖)
====================================================================================================

策略结构 (对应用户三条):
  1) 凯利公式决定高风险(择时仓)与低风险(底仓)的配比 w_large = f*
  2) 底仓(纯债池+红利池) 始终持有, 永不卖出
  3) 择时仓(大盘宽基池)   低估区(估值分位或股债性价比触发)持有, 高估区(两者都不触发)卖出持币

验证目标:
  V1 收益稳定性: 逐年收益 / 滚动1年 / 波动率 / 最大回撤 / 夏普
  V2 怎么配基金: 凯利 w_large 敏感性 (0% / 20% / 40% / 48%半凯利 / 60% / 80% / 100%)
  V3 低估区一次性买入 vs 低估区季度定投 (择时仓口径, 现金流法 XIRR)

数据: acc_nav 累计净值(含分红再投资) 日收益; 忽略申赎费(结论中说明)

用法:
  C:/Users/liuqi/anaconda3/python.exe studies/rotation_dingtou/kelly_portfolio.py
"""
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # fund_research
sys.path.insert(0, ROOT)
import run_rotation as rr  # noqa: E402
import timing_dingtou as td  # noqa: E402

START = "2021-01-01"
END = "2026-08-06"
NAV_DIR = r"D:\iquant_data\data_v2\fund2\nav"

# 候选池 (每池 3 只, 等权)
PURE_BOND = ["519985", "110037", "050027"]      # 纯债
DIVIDEND = ["100032", "090010", "161907"]       # 红利
LARGE_CAP = ["050002", "110020", "160119"]      # 大盘宽基 (沪深300x2 + 中证500x1)

# 凯利参数 (由策略起点前 2015-2020 历史估计, 见脚本说明)
KELLY_FULL = 0.96   # 全凯利 (大盘超额 μ/sigma^2)
KELLY_HALF = 0.48   # 半凯利


def acc_nav_series(code):
    """累计净值序列 (datetime index), 含缓存"""
    df = pd.read_parquet(os.path.join(NAV_DIR, f"{code}.parquet"),
                         columns=["date", "acc_nav"])
    df["date"] = pd.to_datetime(df["date"])
    df = df.drop_duplicates("date").sort_values("date")
    s = pd.Series(df["acc_nav"].to_numpy(dtype=float), index=pd.DatetimeIndex(df["date"]))
    return s.dropna()


def pool_daily_ret(codes):
    """池内基金等权日收益 (逐基金 pct_change 后均值, 避免面板 NaN 污染), 裁剪到回测区间"""
    out = {}
    for c in codes:
        s = acc_nav_series(c)
        r = s.pct_change().dropna()
        r = r[(r >= -0.2) & (r <= 0.2)]  # 剔除异常
        r = r[(r.index >= pd.Timestamp(START)) & (r.index <= pd.Timestamp(END))]
        out[c] = r
    df = pd.DataFrame(out).sort_index()
    return df.mean(axis=1)


def load_signals():
    """复用 timing_dingtou 的 S1(估值分位)+S2(股债性价比), 季度采样"""
    basic = rr.load_basic()
    active = basic[basic["fund_type"].isin(rr.ACTIVE_TYPES)]
    codes = active["code"].astype(str).tolist()
    basic_ft = basic.set_index("code")["fund_type"]
    qdates = [pd.Timestamp(d) for d in pd.date_range(START, END, freq="Q")]
    rr.END = pd.Timestamp(END)
    out, union = rr.compute_window_sums(codes, qdates)
    sel_dyn = [rr.select_4433(out, codes, basic_ft, i) for i in range(len(qdates))]
    sig = td.compute_signals(qdates, sel_dyn, out, codes)
    sig = sig.set_index("date")
    sig["s12"] = (sig["s1"].astype(int) + sig["s2"].astype(int))
    sig["hold"] = (sig["s12"] >= 1).astype(int)   # 低估区: S1 或 S2 触发 -> 持有大盘
    return sig, qdates


def build_hold_daily(sig, dates):
    """季度 hold 掩码 -> 日频 (季度内恒定)"""
    hold = pd.Series(0.0, index=dates)
    qs = sig.index
    for i, q in enumerate(qs):
        mask = (dates >= q) & (dates < (qs[i + 1] if i + 1 < len(qs) else dates.max() + pd.Timedelta(days=1)))
        hold[mask] = sig["hold"].iloc[i]
    return hold


def run_nav(base_ret, large_ret, hold, w_large):
    """净值法: 组合 = w_base*底仓 + w_large*择时仓(低估持有大盘/高估持币)"""
    w_base = 1.0 - w_large
    timed_ret = w_large * (hold * large_ret)
    comb = w_base * base_ret + timed_ret
    comb = comb.fillna(0.0)
    nav = (1.0 + comb).cumprod()
    return nav, comb


def stats(nav):
    r = nav.pct_change().dropna()
    total = nav.iloc[-1] / nav.iloc[0] - 1.0
    ann = (nav.iloc[-1] / nav.iloc[0]) ** (252.0 / len(nav)) - 1.0
    vol = r.std() * np.sqrt(252.0)
    mdd = float((nav / nav.cummax() - 1.0).min())
    sharpe = (r.mean() * 252.0) / vol if vol > 0 else np.nan
    return total, ann, vol, mdd, sharpe


def run_timed_cash(qdates, hold, large_nav, amount_per_period=None, lump_sum=False):
    """择时仓现金流回测 (低估区买入大盘, 高估区卖出). 一次性 vs 季度定投.
    语义: 外部投入直接买份额(现金流-amt); 卖出所得留在组合内现金(等待再买);
    期末 final = 现金 + 剩余份额市值, 现金流 = 买入流出 + 期末总价值.
    large_nav: 大盘组合累计净值(日频), 用于买卖点估值"""
    shares = 0.0
    cash = 0.0
    cf = []
    buys = 0
    for i, d in enumerate(qdates):
        h = hold.iloc[i]
        nav = large_nav.asof(d) if (large_nav.index <= d).any() else np.nan
        if np.isnan(nav):
            continue
        if h:  # 低估区 -> 买入 (外部资金直接买份额)
            amt = amount_per_period if amount_per_period else 100000.0 if buys == 0 and lump_sum else 0.0
            if amt > 0:
                shares += amt / nav
                cf.append((pd.Timestamp(d), -amt))
                buys += 1
        else:  # 高估区 -> 全部卖出, 所得进入组合现金
            if shares > 0:
                cash += shares * nav
                shares = 0.0
    end = pd.Timestamp(END)
    end_nav = large_nav.asof(end)
    final = cash + shares * end_nav
    cf.append((end, final))
    return final, shares, cf, buys


def main():
    # ---- 数据 ----
    print("加载池子日收益 ...")
    base_ret = 0.5 * pool_daily_ret(PURE_BOND) + 0.5 * pool_daily_ret(DIVIDEND)
    large_ret = pool_daily_ret(LARGE_CAP)
    large_nav = (1.0 + large_ret.fillna(0.0)).cumprod()
    all_dates = large_ret.index

    print("计算信号 ...")
    sig, qdates = load_signals()
    hold = build_hold_daily(sig, all_dates)
    hold_q = pd.Series(sig["hold"].values, index=sig.index)
    print(f"低估季度数: {int(sig['hold'].sum())} / {len(sig)}")

    # ---- V2 凯利配比敏感性 (净值法) ----
    print("\n" + "=" * 100)
    print(f"V2 凯利配比敏感性 ({START} ~ {END}): w_large = 高风险(大盘择时)占比")
    print("=" * 100)
    rows = []
    for w in [0.0, 0.20, 0.40, 0.48, 0.60, 0.80, 1.00]:
        nav, _ = run_nav(base_ret, large_ret, hold, w)
        total, ann, vol, mdd, sharpe = stats(nav)
        tag = ""
        if w == KELLY_HALF:
            tag = "  <- 半凯利"
        elif w == KELLY_FULL:
            tag = "  <- 全凯利"
        rows.append({"w_large": w, "final": nav.iloc[-1], "total": total, "ann": ann,
                     "vol": vol, "mdd": mdd, "sharpe": sharpe, "tag": tag})
        print(f"  w_large={w:>5.0%}: 期末 {nav.iloc[-1]:>9,.0f}  总收益 {total:>7.2%}  年化 {ann:>6.2%}  波动 {vol:>5.2%}  回撤 {mdd:>7.2%}  夏普 {sharpe:>5.2f}{tag}")
    sens = pd.DataFrame(rows)
    sens.to_csv(os.path.join(rr.RESULTS_DIR, "kelly_sensitivity.csv"), index=False, encoding="utf-8-sig")

    # ---- V1 收益稳定性 (半凯利组合 + 底仓 对照) ----
    print("\n" + "=" * 100)
    print("V1 收益稳定性: 半凯利组合 vs 纯底仓 vs 纯大盘择时 (净值法, 逐年分解)")
    print("=" * 100)
    nav_k, comb_k = run_nav(base_ret, large_ret, hold, KELLY_HALF)
    nav_b, _ = run_nav(base_ret, large_ret, hold, 0.0)
    nav_l, _ = run_nav(base_ret, large_ret, hold, 1.0)
    yr = pd.DataFrame({"半凯利组合": nav_k, "纯底仓(债+红利)": nav_b, "纯大盘择时": nav_l}).resample("Y").last()
    yr = yr / yr.shift(1) - 1.0
    print("  年度收益:")
    print(yr.to_string())
    print()
    for name, nav in [("半凯利组合", nav_k), ("纯底仓(债+红利)", nav_b), ("纯大盘择时", nav_l)]:
        total, ann, vol, mdd, sharpe = stats(nav)
        print(f"  {name}: 总收益 {total:>7.2%} 年化 {ann:>6.2%} 波动 {vol:>5.2%} 回撤 {mdd:>7.2%} 夏普 {sharpe:>5.2f}")
    # 滚动1年收益稳定性
    r1y = nav_k.pct_change(252).dropna()
    print(f"\n  半凯利组合滚动1年收益: 中位数 {r1y.median():.2%}  最小 {r1y.min():.2%}  最大 {r1y.max():.2%}")
    print(f"  负滚动1年占比: {(r1y < 0).mean():.1%}")

    # ---- V3 低估区一次性 vs 季度定投 (择时仓口径) ----
    print("\n" + "=" * 100)
    print("V3 低估区买入方式: 一次性 vs 低估区季度定投 (择时仓=大盘池, 高估区卖出)")
    print("=" * 100)
    n_low = int(sig["hold"].sum())
    per = 100000.0 / n_low  # 定投: 低估季各投一份, 总额 10 万
    # 一次性: 在第一个低估季投入 10 万
    f_lump, _, _, _ = run_timed_cash(qdates, hold_q, large_nav, lump_sum=True)
    f_dca, _, cf_dca, n_dca = run_timed_cash(qdates, hold_q, large_nav, amount_per_period=per)
    irr_dca = rr.xirr(cf_dca) if len(cf_dca) >= 2 else np.nan
    # 一次性 XIRR: 现金流 = -100000(首低估季) + 期末
    idx_first = int(hold_q.values.argmax())
    cf_lump = [(pd.Timestamp(qdates[idx_first]), -100000.0),
               (pd.Timestamp(END), f_lump)]
    irr_lump = rr.xirr(cf_lump)
    print(f"  一次性(首低估季 {qdates[idx_first].date()} 全入): 期末 {f_lump:,.0f}  XIRR {irr_lump:.2%}")
    print(f"  季度定投(低估{n_dca}季 × {per:,.0f}): 期末 {f_dca:,.0f}  XIRR {irr_dca:.2%}")

    # ---- 图 ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
        plt.rcParams["axes.unicode_minus"] = False
        fig, axes = plt.subplots(3, 1, figsize=(13, 11), sharex=True,
                                 gridspec_kw={"height_ratios": [2.4, 1, 1]})
        for w, color in [(0.0, "#B0B0B0"), (0.48, "#4B3FE3"), (1.0, "#E8463A")]:
            nav, _ = run_nav(base_ret, large_ret, hold, w)
            axes[0].plot(nav.index, nav.values, label=f"w_large={w:.0%}", linewidth=1.5, color=color)
        axes[0].set_ylabel("组合净值")
        axes[0].legend()
        axes[0].grid(alpha=0.3)
        axes[0].set_title(f"凯利公式组合: 债+红利底仓持有 + 大盘低估买高估卖 ({START} ~ {END})")

        axes[1].bar(sig.index, sig["s12"], color="#4B3FE3", alpha=0.8, width=20)
        axes[1].axhline(0.5, color="#E8463A", linestyle="--", linewidth=1)
        axes[1].set_ylabel("S1+S2")
        axes[1].set_yticks([0, 1, 2])
        axes[1].grid(alpha=0.3)

        axes[2].plot(large_nav.index, large_nav.values, label="大盘池", color="#E8463A", linewidth=1.2)
        axes[2].plot((1 + base_ret.fillna(0)).cumprod().index,
                     (1 + base_ret.fillna(0)).cumprod().values, label="底仓(债+红利)", color="#1B8A6B", linewidth=1.2)
        axes[2].set_ylabel("基准净值")
        axes[2].legend()
        axes[2].grid(alpha=0.3)
        fig.tight_layout()
        png = os.path.join(rr.RESULTS_DIR, "kelly_portfolio_compare.png")
        fig.savefig(png, dpi=130)
        print(f"\n对比图: {png}")
    except Exception as e:
        print(f"画图跳过: {e}")

    print("\n注: 未计申赎费(0.15%申购/0.5%赎回), 择时仓高估区为持币(0收益), 低估区持有大盘.")


if __name__ == "__main__":
    main()

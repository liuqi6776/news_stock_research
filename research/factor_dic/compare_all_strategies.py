# -*- coding: utf-8 -*-
"""全局策略对比: ENH+RS12 vs ETF vs 行业轮动 vs 基金配置

所有策略统一验证:
  - 无前视 (PIT 严格对齐)
  - MaxDD 为百分比: ((cummax - nav) / cummax).max()
  - 真实成本 (股票/行业 20-30bps)
  - 同期对齐 (尽量 2020-03 ~ 2026-06)

输出: results/all_strategy_comparison.csv + 对比图
"""
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from research.factor_dic import run_validation as rv
from research.factor_dic import combo_backtest as cb
from config import settings

OUT_DIR = rv.OUT_DIR
IDX_DIR = rv.IDX_DIR
SECTOR_OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "sector_rotation", "results")


def calc_stats_correct(pr, n_per_year=12):
    """正确版统计: 百分比 MaxDD"""
    pr = pd.Series(pr).dropna()
    navs = (1 + pr).cumprod()
    years = len(pr) / n_per_year
    dd_pct = ((navs.cummax() - navs) / navs.cummax()).max()
    return dict(
        CAGR=(1 + pr).prod() ** (1 / years) - 1 if years > 0 else np.nan,
        Sharpe=pr.mean() / pr.std(ddof=1) * np.sqrt(n_per_year) if pr.std(ddof=1) > 0 else np.nan,
        MaxDD=dd_pct,
        WinRate=(pr > 0).mean(),
        Vol=pr.std(ddof=1) * np.sqrt(n_per_year),
        FinalNAV=navs.iloc[-1],
        Calmar=((1 + pr).prod() ** (1 / years) - 1) / dd_pct if dd_pct > 0 else np.nan,
    )


def load_idx_series(code, trade_dates=None):
    df = pd.read_parquet(os.path.join(IDX_DIR, f"{code}.parquet"))
    df["trade_date"] = df["trade_date"].astype(str).str[:8]
    return df.set_index("trade_date").sort_index()


# =========== 1. ENH + RS12 策略 ===========
def build_enh_rs12():
    print("[1/5] 构建 ENH + RS12 策略...", flush=True)
    trade_dates = rv.load_trade_dates()
    months = {d[:6]: d for d in trade_dates if d[:4] >= str(rv.START_YEAR)}
    rebal = sorted(months.values())[:-1]

    sml, big = load_idx_series("000852.SH"), load_idx_series("000300.SH")
    etf = load_idx_series("512100.SH")
    ratio = sml["close"] / big["close"].reindex(sml.index)
    sig_rs12 = ((ratio / ratio.shift(240)).rolling(5).mean() - 1.0 > 0).reindex(rebal)

    all_codes = set()
    for rb in rebal:
        m = rv.load_index_weight(rb)
        if m:
            all_codes |= m
    all_codes = sorted(all_codes)
    stocks, pct_df, _, _, _ = rv.load_panels(trade_dates, all_codes, None)
    ret_1m, ivol, turn, fwd = cb.build_price_factors(stocks, all_codes)
    COST = rv.COST_BPS / 10000.0

    enh_ret = {}
    for i, rb in enumerate(rebal):
        if i + 1 >= len(rebal):
            continue
        members = rv.load_index_weight(rb)
        if members is None:
            continue
        rb_next = rebal[i + 1]
        hi, hn = trade_dates.index(rb), trade_dates.index(rb_next)
        hold = trade_dates[hi + 1:hn + 1]
        fvals = {}
        for code in members:
            f1, f2, ft = ret_1m.get(code), ivol.get(code), turn.get(code)
            row = {}
            if f1 is not None and rb in f1.index:
                row["ret_1m"] = f1.loc[rb]
            if f2 is not None and rb in f2.index:
                row["ivol"] = f2.loc[rb]
            if ft is not None and rb in ft.index:
                row["turn"] = ft.loc[rb]
            if len(row) >= 3:
                fvals[code] = row
        if len(fvals) < rv.TOP_N:
            continue
        fdf = pd.DataFrame(fvals).T
        zdf = fdf.apply(rv.winsorize).apply(lambda s: (s - s.mean()) / (s.std() + 1e-8), axis=0)
        sc = zdf[["ret_1m", "ivol", "turn"]].dropna().mean(axis=1)
        picks = sc.nlargest(rv.TOP_N).index
        sub = pct_df.reindex(columns=picks).reindex(hold).fillna(0.0) / 100.0
        enh_ret[rb] = (1 + sub.mean(axis=1)).prod() - 1 - COST

    enh = pd.Series(enh_ret).sort_index()
    idx = enh.index

    # 基准
    b_etf = {}
    for rb in idx:
        if rb not in enh_ret:
            continue
        i = rebal.index(rb)
        if i + 1 >= len(rebal):
            continue
        rb_next = rebal[i + 1]
        hi, hn = trade_dates.index(rb), trade_dates.index(rb_next)
        hold = trade_dates[hi + 1:hn + 1]
        be = etf["pct_chg"].reindex(hold).fillna(0.0) / 100.0
        b_etf[rb] = (1 + be).prod() - 1
    b_etf = pd.Series(b_etf).sort_index().reindex(idx)

    # RS12: 小盘弱 → ETF
    enh_rs = enh.where(sig_rs12.reindex(idx), b_etf)

    nav_enh = (1 + enh).cumprod()
    nav_enhrs12 = (1 + enh_rs).cumprod()
    nav_512 = (1 + b_etf).cumprod()

    print(f"  ENH 无择时: CAGR={calc_stats_correct(enh)['CAGR']:.2%} "
          f"MaxDD={calc_stats_correct(enh)['MaxDD']:.2%} "
          f"Calmar={calc_stats_correct(enh)['Calmar']:.2f}")
    print(f"  ENH+RS12:  CAGR={calc_stats_correct(enh_rs)['CAGR']:.2%} "
          f"MaxDD={calc_stats_correct(enh_rs)['MaxDD']:.2%} "
          f"Calmar={calc_stats_correct(enh_rs)['Calmar']:.2f}")
    print(f"  512100ETF: CAGR={calc_stats_correct(b_etf)['CAGR']:.2%} "
          f"MaxDD={calc_stats_correct(b_etf)['MaxDD']:.2%} "
          f"Calmar={calc_stats_correct(b_etf)['Calmar']:.2f}")

    return (idx, nav_enhrs12), (idx, nav_enh), (idx, nav_512), (idx, enh_rs), (idx, enh), (idx, b_etf)


# =========== 2. 行业轮动 (无前视修正版) ===========
def build_sector_rotation():
    print("[2/5] 构建行业轮动策略...", flush=True)
    sector_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "sector_rotation")
    if sector_dir not in sys.path:
        sys.path.insert(0, sector_dir)
    from lookahead_audit import (load_data, calc_pe_pct, calc_momentum,
                                  calc_volatility, run_strategy_corrected)
    pe_df, ret_df = load_data()
    pct_df = calc_pe_pct(pe_df, 36)
    mom_df = calc_momentum(ret_df, 3)
    vol_df = calc_volatility(ret_df, 6)
    COST_T = 30 / 10000.0

    nv_g = run_strategy_corrected(ret_df, pct_df, pe_df, mom_df=mom_df, vol_df=vol_df,
                                   top_n=20, mode="growth", mom_filter=True, vol_filter=True,
                                   cost=COST_T)
    nv_v = run_strategy_corrected(ret_df, pct_df, pe_df, mom_df=mom_df, vol_df=vol_df,
                                   top_n=20, mode="value", mom_filter=True, vol_filter=True,
                                   cost=COST_T)
    nv_eq = run_strategy_corrected(ret_df, pct_df, pe_df, top_n=20, mode="equal", cost=COST_T)

    # 转月收益 series
    idx = list(nv_g.index)
    enh_ret_s = nv_g["nav"].pct_change().fillna(nv_g["nav"].iloc[0] - 1)
    val_ret_s = nv_v["nav"].pct_change().fillna(nv_v["nav"].iloc[0] - 1)
    eq_ret_s = nv_eq["nav"].pct_change().fillna(nv_eq["nav"].iloc[0] - 1)

    print(f"  高PE轮动: CAGR={calc_stats_correct(enh_ret_s)['CAGR']:.2%} "
          f"MaxDD={calc_stats_correct(enh_ret_s)['MaxDD']:.2%}")
    print(f"  低估轮动: CAGR={calc_stats_correct(val_ret_s)['CAGR']:.2%} "
          f"MaxDD={calc_stats_correct(val_ret_s)['MaxDD']:.2%}")
    print(f"  等权基准: CAGR={calc_stats_correct(eq_ret_s)['CAGR']:.2%} "
          f"MaxDD={calc_stats_correct(eq_ret_s)['MaxDD']:.2%}")

    return (idx, nv_g["nav"]), (idx, nv_v["nav"]), (idx, nv_eq["nav"])


# =========== 3. 沪深300ETF ===========
def build_etf_510300():
    print("[3/5] 构建 510300/000905 ETF 基准...", flush=True)
    trade_dates = rv.load_trade_dates()
    months = {d[:6]: d for d in trade_dates if d[:4] >= str(rv.START_YEAR)}
    rebal = sorted(months.values())[:-1]

    hs300 = load_idx_series("510300.SH")
    zz500 = load_idx_series("000905.SH")

    # 510300 月度收益
    r1, r2 = {}, {}
    for i, rb in enumerate(rebal):
        if i + 1 >= len(rebal):
            continue
        rb_next = rebal[i + 1]
        hi, hn = trade_dates.index(rb), trade_dates.index(rb_next)
        hold = trade_dates[hi + 1:hn + 1]
        a = hs300["pct_chg"].reindex(hold).fillna(0.0) / 100.0
        r1[rb] = (1 + a).prod() - 1
        b = zz500["pct_chg"].reindex(hold).fillna(0.0) / 100.0
        r2[rb] = (1 + b).prod() - 1

    s1 = pd.Series(r1).sort_index()
    s2 = pd.Series(r2).sort_index()
    return (list(s1.index), (1 + s1).cumprod()), (list(s2.index), (1 + s2).cumprod())


def main():
    print("=" * 80)
    print("全局策略对比（统一口径: 无前视 + 百分比MaxDD + 真实成本）")
    print("=" * 80)

    # 构建
    enh_rs12_nav, enh_no_nav, etf512_nav, enh_rs_ret, enh_ret, b512_ret = build_enh_rs12()
    g_nav, v_nav, eq_nav = build_sector_rotation()
    (idx3, nav510300), (idx500, nav000905) = build_etf_510300()

    # 对齐起始日期 (取所有策略交集的最大起点)
    # 用 enh_rs12 作为基准对齐 (2020-01 起, 最多)
    # 统一转换为: 以 enh_rs12 日期为基准
    base_dates = enh_rs12_nav[0]

    def to_monthly_aligned(idx_nav):
        idx, navs = idx_nav
        s = pd.Series(navs.values if hasattr(navs, "values") else navs,
                      index=[str(x) for x in idx]).reindex([str(x) for x in base_dates])
        return s.ffill()

    df = pd.DataFrame({
        "ENH+RS12 (4因子小盘增强+择时)": to_monthly_aligned(enh_rs12_nav),
        "ENH 无择时 (4因子小盘增强)": to_monthly_aligned(enh_no_nav),
        "512100 中证1000ETF": to_monthly_aligned(etf512_nav),
        "510300 沪深300ETF": to_monthly_aligned((idx3, nav510300)),
        "000905 中证500指数": to_monthly_aligned((idx500, nav000905)),
        "行业轮动-高PE(无前视)": to_monthly_aligned(g_nav),
        "行业轮动-低估(无前视)": to_monthly_aligned(v_nav),
        "行业等权基准": to_monthly_aligned(eq_nav),
    })
    df = df.dropna()
    df.index.name = "date"

    # 计算每个策略的统计 (相对起点归一)
    rows = []
    for col in df.columns:
        pr = df[col].pct_change().dropna()
        st = calc_stats_correct(pr)
        rows.append({"策略": col, **st})
    res = pd.DataFrame(rows).sort_values("Calmar", ascending=False).reset_index(drop=True)

    print("\n" + "=" * 100)
    print(f"{'策略':<28} {'NAV':>6} {'CAGR':>8} {'Sharpe':>7} {'MaxDD':>8} {'Calmar':>7} {'WinRate':>8} {'Vol':>8}")
    print("-" * 100)
    for _, r in res.iterrows():
        mark = " ★" if r.name == 0 else ""
        print(f"{r['策略']:<28} {r['FinalNAV']:>6.2f} {r['CAGR']:>7.2%} {r['Sharpe']:>7.2f} "
              f"{r['MaxDD']:>7.2%} {r['Calmar']:>7.2f} {r['WinRate']:>7.1%} {r['Vol']:>7.2%}{mark}")

    # 保存
    res.to_csv(os.path.join(OUT_DIR, "all_strategy_comparison.csv"), index=False, encoding="utf-8-sig")
    print(f"\n[saved] all_strategy_comparison.csv")

    # 画图 (归一化到1)
    fig, axes = plt.subplots(1, 2, figsize=(16, 6.5))

    ax = axes[0]
    ax.set_title("全部策略 NAV 对比（同期起点=1）", fontsize=12)
    for col in df.columns:
        ax.plot(range(len(df)), df[col].values / df[col].iloc[0], lw=1.8, label=col)
    ax.legend(fontsize=8, loc="upper left")
    ax.set_yscale("log")
    ax.set_ylabel("NAV (对数轴)")
    ax.grid(True, which="both", alpha=0.3)

    # 右: Calmar 排名
    ax = axes[1]
    bar_df = res.sort_values("Calmar", ascending=True)
    colors = []
    for name in bar_df["策略"]:
        if "ENH+RS12" in name:
            colors.append("#d62728")
        elif "ETF" in name or "指数" in name:
            colors.append("#1f77b4")
        elif "行业" in name:
            colors.append("#ff7f0e")
        else:
            colors.append("#888")
    ypos = range(len(bar_df))
    bars = ax.barh(ypos, bar_df["Calmar"], color=colors)
    ax.set_yticks(ypos)
    ax.set_yticklabels(bar_df["策略"], fontsize=9)
    ax.set_xlabel("Calmar 比率 (CAGR / MaxDD)")
    ax.set_title("风险调整后收益排名（Calmar）", fontsize=12)
    for bar, val in zip(bars, bar_df["Calmar"]):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                f"{val:.2f}", va="center", fontsize=9)
    ax.grid(True, alpha=0.3, axis="x")

    fig.tight_layout()
    png = os.path.join(OUT_DIR, "all_strategy_comparison.png")
    fig.savefig(png, dpi=150, bbox_inches="tight")
    print(f"[saved] {png}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

# -*- coding: utf-8 -*-
"""诊断: MaxDD 采样粒度对比 — 调仓点口径 vs 逐日盯市口径

问题: freq_stats.csv 中 "月频2 BASE MaxDD=13.66%" 是调仓点(每2个月1个NAV点)算的,
持有期内日内下跌不可见, 会低估真实回撤。

本脚本对 BASE 月频2 / BASE 月频 / MOM_D 月频, 分别输出:
  1) 调仓点口径 MaxDD (即 frequency 脚本当前口径)
  2) 逐日盯市口径 MaxDD (持仓期内按日等权 mark-to-market, 组合日收益复利)
"""
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from research.factor_dic import run_validation as rv  # noqa: E402
from research.factor_dic import combo_backtest as cb  # noqa: E402
from research.factor_dic import style_factors as sf  # noqa: E402

STUDY_DIR = os.path.join(ROOT, "research", "studies", "study_008_enhancements")
IND_MAP_PATH = os.path.join(STUDY_DIR, "data", "industry_map.parquet")
OUT_DIR = os.path.join(ROOT, "research", "sector_rotation", "results")

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

COST = 20 / 10000.0
TOP_N = 60


def load_industry_map():
    df = pd.read_parquet(IND_MAP_PATH)
    return dict(zip(df["ts_code"], df["industry"]))


def select_with_limit(scores, code_to_ind, max_per_ind, top_n):
    sorted_codes = scores.sort_values(ascending=False)
    selected, ind_count = [], {}
    for code in sorted_codes.index:
        ind = code_to_ind.get(code, "其他")
        if ind_count.get(ind, 0) < max_per_ind:
            selected.append(code)
            ind_count[ind] = ind_count.get(ind, 0) + 1
        if len(selected) >= top_n:
            break
    return selected


def select_with_momentum(scores, code_to_ind, ind_momentum, top_inds=10, max_per_ind=4, top_n=40):
    if ind_momentum is None or len(ind_momentum) == 0:
        return select_with_limit(scores, code_to_ind, max_per_ind, top_n)
    top = set(ind_momentum.nlargest(top_inds).index)
    filtered = scores[scores.index.map(lambda c: code_to_ind.get(c, "其他") in top)]
    if len(filtered) < top_n:
        return select_with_limit(scores, code_to_ind, max_per_ind, top_n)
    return select_with_limit(filtered, code_to_ind, max_per_ind, top_n)


def maxdd(nav):
    return ((nav.cummax() - nav) / nav.cummax()).max()


def build_frequency_lists(trade_dates):
    months = {d[:6]: d for d in trade_dates if d[:4] >= str(rv.START_YEAR)}
    monthly = sorted(months.values())
    monthly2 = monthly[::2]
    return monthly, monthly2


def main():
    trade_dates = rv.load_trade_dates()
    monthly, monthly2 = build_frequency_lists(trade_dates)
    all_codes = set()
    for rb in monthly:
        m = rv.load_index_weight(rb)
        if m:
            all_codes |= m
    all_codes = sorted(all_codes)
    print(f"成分股池: {len(all_codes)}")

    stocks, pct_df, _, _, _ = rv.load_panels(trade_dates, all_codes, None)
    ret_1m, ivol, turn, _ = cb.build_price_factors(stocks, all_codes)
    val_map = sf.load_valuation(monthly, all_codes)
    funda_map = sf.build_funda_pit(monthly, all_codes)
    panels = sf.build_factors(val_map, funda_map, monthly)
    code_to_ind = load_industry_map()

    tidx = {d: i for i, d in enumerate(trade_dates)}
    pct = pct_df / 100.0  # 小数收益

    def backtest(rebal, strat, momd_top_n=40):
        """返回 (调仓点nav DataFrame, 逐日nav Series)"""
        ind_ret = {}
        point_recs = []
        daily_parts = []
        prev = None
        for i, t in enumerate(rebal):
            if i + 1 >= len(rebal):
                break
            members = rv.load_index_weight(t)
            if members is None:
                continue
            fvals = {}
            for code in members:
                f1, f2, ft = ret_1m.get(code), ivol.get(code), turn.get(code)
                if f1 is None or t not in f1.index:
                    continue
                row = {}
                v1, v2, vt = f1.loc[t], f2.loc[t] if f2 is not None and t in f2.index else np.nan, ft.loc[t] if ft is not None and t in ft.index else np.nan
                row["ret_1m"] = v1
                row["ivol"] = v2
                row["turn"] = vt
                for name in panels:
                    p = panels[name].get(t)
                    if p is not None and code in p.index:
                        v = p.loc[code]
                        if np.isfinite(v):
                            row[name] = v
                if len(row) >= 3:
                    fvals[code] = row
            if len(fvals) < TOP_N:
                continue
            fdf = pd.DataFrame(fvals).T
            zdf = fdf.apply(sf.winsorize_series).apply(
                lambda s: (s - s.mean()) / (s.std() + 1e-8), axis=0)
            cols = [c for c in sf.BASE_COLS + ["VAL"] if c in zdf.columns]
            has = zdf[cols].dropna()
            if len(has) < TOP_N:
                continue
            scores = has.mean(axis=1)
            base = scores.nlargest(TOP_N).index.tolist()

            if strat == "BASE":
                sel = base
            else:
                prev_t = rebal[i - 1] if i > 0 else None
                mom = pd.Series(ind_ret[prev_t]) if (prev_t and prev_t in ind_ret) else None
                sel = select_with_momentum(scores, code_to_ind, mom,
                                           top_inds=10, max_per_ind=4, top_n=momd_top_n)

            next_t = rebal[i + 1]
            i0, i1 = tidx[t], tidx[next_t]
            hold_dates = trade_dates[i0 + 1: i1 + 1]
            if not len(hold_dates):
                continue

            # 1) 调仓点收益: 逐股复利后等权 (frequency 口径)
            sub = pct_df.reindex(columns=sel).reindex(hold_dates).fillna(0.0)
            rc = (1 + sub / 100.0).prod() - 1.0
            rc = rc[rc.notna()]
            port_ret = float(rc.mean()) if len(rc) else 0.0

            if prev is None:
                c = COST
            else:
                turn_r = len(set(sel) - set(prev)) / len(sel)
                c = turn_r * COST
            point_recs.append({"date": next_t, "ret": port_ret - c})

            # 2) 逐日盯市: 持仓期内每日等权平均组合日收益
            day_rets = pct[sel].reindex(hold_dates).fillna(0.0)
            comb = day_rets.mean(axis=1)  # 每日等权(日频再平衡口径)
            daily_parts.append(comb)

            # 行业动量(基于 BASE, pct 口径)
            sub_b = pct_df.reindex(columns=base).reindex(hold_dates).fillna(0.0)
            rcb = (1 + sub_b / 100.0).prod() - 1.0
            rcb = rcb[rcb.notna()]
            if len(rcb):
                ind_ret[t] = rcb.groupby(lambda c: code_to_ind.get(c, "其他")).mean()
            prev = sel

        point_df = pd.DataFrame(point_recs).set_index("date")
        point_df["nav"] = (1 + point_df["ret"]).cumprod()
        daily_ser = pd.concat(daily_parts)
        daily_nav = (1 + daily_ser).cumprod()
        return point_df, daily_nav

    print("\n=== MaxDD: 调仓点口径 vs 逐日盯市口径 (2020-2026, 20bps) ===")
    print(f"{'策略':<14} {'调仓点MaxDD':>12} {'逐日MaxDD':>12} {'低估幅度':>10}")
    print("-" * 52)
    curves = {}
    dd_info = {}
    for lbl, rebal, strat in [("BASE 月频2", monthly2, "BASE"),
                              ("BASE 月频", monthly, "BASE"),
                              ("MOM_D 月频", monthly, "MOM_D")]:
        point_df, daily_nav = backtest(rebal, strat)
        md_pt = maxdd(point_df["nav"])
        md_dy = maxdd(daily_nav)
        print(f"  {lbl:<12} {md_pt:>11.2%} {md_dy:>11.2%} {md_dy - md_pt:>9.2%}")

        # 逐日回撤最深区间
        draw = (daily_nav.cummax() - daily_nav) / daily_nav.cummax()
        if draw.max() > 0:
            t_end = str(draw.idxmax())
            t_peak = str(daily_nav.loc[:t_end].idxmax())
            days = (pd.to_datetime(t_end) - pd.to_datetime(t_peak)).days
            print(f"       逐日最大回撤区间: {t_peak} ~ {t_end} ({days}天)")
        curves[lbl] = daily_nav
        dd_info[lbl] = (md_pt, md_dy)

    # ---- 绘制逐日口径 NAV 曲线 ----
    fig, ax = plt.subplots(figsize=(13, 7))
    colors = {"BASE 月频2": "#8c564b", "BASE 月频": "#ff7f0e", "MOM_D 月频": "#d62728"}
    styles = {"BASE 月频2": "--", "BASE 月频": "-", "MOM_D 月频": "-"}
    for lbl, nav_s in curves.items():
        md_pt, md_dy = dd_info[lbl]
        ax.plot(nav_s.index, nav_s.values, lw=1.8, color=colors[lbl], ls=styles[lbl],
                label=f"{lbl} (逐日 MaxDD={md_dy:.1%})")

    # 基准: 中证1000
    bench_fp = os.path.join(rv.IDX_DIR, "000852.SH.parquet")
    bdf = pd.read_parquet(bench_fp)
    bdf["trade_date"] = bdf["trade_date"].astype(str).str[:8]
    bdf = bdf.set_index("trade_date").sort_index()
    bench_nav = (1 + bdf["pct_chg"] / 100.0).cumprod()
    start_d = monthly[0]
    bench_nav = bench_nav.loc[start_d:] / bench_nav.loc[start_d]
    ax.plot(bench_nav.index, bench_nav.values, lw=1.2, color="#7f7f7f",
            label=f"中证1000指数 ({bench_nav.iloc[-1]:.2f}×)")

    # 标注 BASE 月频2 的最大回撤区间 (2023-08 ~ 2024-02)
    dd_key = "BASE 月频2"
    draw2 = (curves[dd_key].cummax() - curves[dd_key]) / curves[dd_key].cummax()
    if draw2.max() > 0:
        t_end = str(draw2.idxmax())
        t_peak = str(curves[dd_key].loc[:t_end].idxmax())
        ax.axvspan(t_peak, t_end, color="gray", alpha=0.15,
                   label=f"{dd_key} 回撤区间 {t_peak[:4]}-{t_peak[4:6]}~{t_end[:4]}-{t_end[4:6]}")

    ax.set_title("策略 NAV 曲线: 逐日盯市口径 (2020-2026, 20bps成本)", fontsize=13)
    ax.set_ylabel("NAV (起点=1)")
    ax.set_xlabel("日期")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    png = os.path.join(OUT_DIR, "freq_daily_maxdd_curve.png")
    fig.savefig(png, dpi=150, bbox_inches="tight")
    print(f"\n[saved] {png}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

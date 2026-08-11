# -*- coding: utf-8 -*-
"""MOM_D 调仓频率对比: 月 / 周 / 日

在 MOM_D(行业动量 Top10 + 行业内 ENH 4因子打分, 每行业≤4只, Top40 等权) 逻辑不变的前提下,
仅改变"选股+换仓"频率, 对比 月 / 周(每5交易日) / 日 的收益、回撤、换手与成本。

对照策略(同频率):
  BASE   ENH 4因子(ret_1m+ivol+turnover_vol_20+VAL) Top60 等权, 无行业约束
  MOM_D  行业动量(过去一期行业平均收益 Top10) + 行业内选股, Top40

基准: 中证1000(000852)

数据: 中证1000成分股, 2020-01~2026-07, 20bps 双边成本
无前视: 信号 T 日收盘生成(因子/动量均只用 T 日及以前数据) -> (T, next_T] 持有期收益

结论预期: 频率升高 -> 换手骤增 -> 成本吞噬收益; 周/日 的行业动量(5日/1日收益)
噪声远大于月频(20日收益), 选行业稳定性下降。
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
os.makedirs(OUT_DIR, exist_ok=True)

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

COST = 20 / 10000.0  # 20bps 双边(与月频脚本口径一致)
TOP_N = 60
N_PER_YEAR = {"月频": 12, "月频2": 6, "月频3": 4, "周频": 48, "日频": 242}


def load_industry_map():
    df = pd.read_parquet(IND_MAP_PATH)
    return dict(zip(df["ts_code"], df["industry"]))


def select_with_limit(scores, code_to_ind, max_per_ind, top_n):
    """按打分降序, 每行业最多max_per_ind只, 取top_n只"""
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
    """先选动量Top10行业, 再在选中行业内按打分选股(每行业≤4只, 共40只)"""
    if ind_momentum is None or len(ind_momentum) == 0:
        return select_with_limit(scores, code_to_ind, max_per_ind, top_n)
    top = set(ind_momentum.nlargest(top_inds).index)
    filtered = scores[scores.index.map(lambda c: code_to_ind.get(c, "其他") in top)]
    if len(filtered) < top_n:
        return select_with_limit(scores, code_to_ind, max_per_ind, top_n)
    return select_with_limit(filtered, code_to_ind, max_per_ind, top_n)


def calc_stats(nav_series, n_per_year):
    rets = nav_series.pct_change().dropna()
    years = len(rets) / n_per_year
    if years == 0 or nav_series.iloc[-1] <= 0:
        return {k: np.nan for k in ["FinalNAV", "CAGR", "Sharpe", "MaxDD", "Calmar", "WinRate", "Vol"]}
    maxdd = ((nav_series.cummax() - nav_series) / nav_series.cummax()).max()
    return {
        "FinalNAV": nav_series.iloc[-1],
        "CAGR": nav_series.iloc[-1] ** (1 / years) - 1,
        "Sharpe": rets.mean() / rets.std(ddof=1) * np.sqrt(n_per_year) if rets.std(ddof=1) > 0 else np.nan,
        "MaxDD": maxdd,
        "Calmar": (nav_series.iloc[-1] ** (1 / years) - 1) / maxdd if maxdd > 0 else np.nan,
        "WinRate": (rets > 0).mean(),
        "Vol": rets.std(ddof=1) * np.sqrt(n_per_year),
    }


def build_frequency_lists(trade_dates):
    """月/周/日/月2/月3 调仓日期列表, 全部对齐到同一开始日(2020年第一个月末)"""
    months = {d[:6]: d for d in trade_dates if d[:4] >= str(rv.START_YEAR)}
    monthly = sorted(months.values())
    start = monthly[0]
    i0 = trade_dates.index(start)
    weekly = [start] + trade_dates[i0 + 5::5]
    daily = trade_dates[i0:]
    monthly2 = monthly[::2]   # 每2个月末换仓(约2个月持有)
    monthly3 = monthly[::3]   # 每3个月末换仓(约3个月持有)
    return monthly, weekly, daily, monthly2, monthly3


def build_wide_factor_dfs(stocks, all_codes, trade_dates, rebal):
    """把逐股因子序列转成宽表(index=交易日, columns=股票), VAL 月末值前向填充"""
    ret_1m, ivol, turn, _ = cb.build_price_factors(stocks, all_codes)
    val_map = sf.load_valuation(rebal, all_codes)
    funda_map = sf.build_funda_pit(rebal, all_codes)
    panels = sf.build_factors(val_map, funda_map, rebal)
    val_panel = panels.get("VAL", {})
    ret_df = pd.DataFrame(ret_1m).reindex(trade_dates)
    ivol_df = pd.DataFrame(ivol).reindex(trade_dates)
    turn_df = pd.DataFrame(turn).reindex(trade_dates)
    val_df = pd.DataFrame(val_panel).T.reindex(trade_dates).ffill()
    print(f"[factors] ret_1m {ret_df.shape}, ivol {ivol_df.shape}, "
          f"turn {turn_df.shape}, VAL {val_df.shape} (月末前向填充)")
    return ret_df, ivol_df, turn_df, val_df


def run_freq(rebal_list, label, ret_df, ivol_df, turn_df, val_df,
             code_to_ind, trade_dates, pct_df, momd_top_n=40):
    """单频率回测: BASE(ENH Top60) 与 MOM_D(Top10行业×每行业≤4只, 持股数= momd_top_n)"""
    tidx = {d: i for i, d in enumerate(trade_dates)}
    results = {}
    for strat, tn in [("BASE", None), ("MOM_D", momd_top_n)]:
        key = strat if strat == "BASE" else f"MOM_D_{tn}"
        nav = 1.0
        prev = None
        recs, tos = [], []
        ind_ret = {}   # {t: {industry: 持有期收益}}, 供下一期动量
        for i, t in enumerate(rebal_list):
            if i + 1 >= len(rebal_list):
                break
            members = rv.load_index_weight(t)
            if members is None:
                continue
            row = pd.concat([ret_df.loc[t], ivol_df.loc[t], turn_df.loc[t], val_df.loc[t]],
                            axis=1)
            row.columns = ["ret_1m", "ivol", "turn", "VAL"]
            row = row[row.index.isin(members)]
            fdf = row.dropna(subset=["ret_1m", "ivol", "turn", "VAL"])
            if len(fdf) < TOP_N:
                continue
            zdf = fdf.apply(sf.winsorize_series).apply(
                lambda s: (s - s.mean()) / (s.std() + 1e-8), axis=0)
            scores = zdf.mean(axis=1)
            base = scores.nlargest(TOP_N).index.tolist()

            if strat == "BASE":
                sel = base
            else:
                prev_t = rebal_list[i - 1] if i > 0 else None
                mom = pd.Series(ind_ret[prev_t]) if (prev_t and prev_t in ind_ret) else None
                sel = select_with_momentum(scores, code_to_ind, mom,
                                           top_inds=10, max_per_ind=4, top_n=tn)

            next_t = rebal_list[i + 1]
            hold_dates = trade_dates[tidx[t] + 1: tidx[next_t] + 1]
            if not len(hold_dates):
                continue
            # 持有期收益: 逐股复利后等权平均(与月频 fwd 口径一致, 停牌 NaN->0)
            sub = pct_df.reindex(columns=sel).reindex(hold_dates).fillna(0.0)
            rc = (1 + sub / 100.0).prod() - 1.0
            rc = rc[rc.notna()]
            port_ret = float(rc.mean()) if len(rc) else 0.0

            # 换手成本: 相对上期持仓替换比例(首期视为全换)
            if prev is None:
                c = COST
                turn = 1.0
            else:
                turn = len(set(sel) - set(prev)) / len(sel)
                c = turn * COST
            nav *= (1 + port_ret - c)
            recs.append({"date": next_t, "nav": nav, "ret": port_ret, "cost": c})
            tos.append(turn)

            # 记录本期行业收益(基于 BASE 持仓) -> 供下一期动量(无前视)
            sub_b = pct_df.reindex(columns=base).reindex(hold_dates).fillna(0.0)
            rcb = (1 + sub_b / 100.0).prod() - 1.0
            rcb = rcb[rcb.notna()]
            if len(rcb):
                ind_ret[t] = rcb.groupby(lambda c: code_to_ind.get(c, "其他")).mean()
            prev = sel

        df = pd.DataFrame(recs).set_index("date")
        results[key] = df
        avg_turn = float(np.mean(tos)) if tos else np.nan
        ann_cost = avg_turn * COST * len(tos) / max(len(tos) / N_PER_YEAR[label], 1e-9)
        st = calc_stats(df["nav"], N_PER_YEAR[label])
        print(f"  [{label}] {key:<8} NAV={st['FinalNAV']:.2f} CAGR={st['CAGR']:>7.2%} "
              f"Sharpe={st['Sharpe']:>5.2f} MaxDD={st['MaxDD']:>7.2%} Calmar={st['Calmar']:>5.2f} "
              f"换手={avg_turn:>6.1%}", flush=True)
        df.attrs["avg_turnover"] = avg_turn
        df.attrs["ann_cost"] = ann_cost
    return results


def main():
    print("=" * 70, flush=True)
    print("MOM_D 调仓频率对比: 月 / 周 / 日", flush=True)
    print("=" * 70, flush=True)

    trade_dates = rv.load_trade_dates()
    monthly, weekly, daily, monthly2, monthly3 = build_frequency_lists(trade_dates)
    for lbl, lst in [("月3", monthly3), ("月2", monthly2), ("月", monthly),
                     ("周", weekly), ("日", daily)]:
        print(f"[freq] {lbl}频: {len(lst)} 个调仓日  {lst[0]} ~ {lst[-1]}")

    all_codes = set()
    for rb in monthly:
        m = rv.load_index_weight(rb)
        if m:
            all_codes |= m
    all_codes = sorted(all_codes)
    print(f"[load] 成分股 {len(all_codes)} 只")

    stocks, pct_df, _, _, _ = rv.load_panels(trade_dates, all_codes, None)
    ret_df, ivol_df, turn_df, val_df = build_wide_factor_dfs(stocks, all_codes, trade_dates, monthly)
    code_to_ind = load_industry_map()
    print(f"[load] 行业映射 {len(code_to_ind)} 只 / {len(set(code_to_ind.values()))} 行业")

    # 组合: (标签, 调仓日列表, MOM_D持股数) —— 覆盖持有周期 3月/2月/1月/周/日
    RUNS = [("月频3", monthly3, 40), ("月频2", monthly2, 40), ("月频", monthly, 40),
            ("周频", weekly, 40), ("日频", daily, 40)]
    all_results = {}
    for lbl, lst, tn in RUNS:
        print(f"\n[backtest] {lbl} MOM_D持股{tn} ({len(lst)}期)", flush=True)
        all_results[(lbl, tn)] = run_freq(lst, lbl, ret_df, ivol_df, turn_df, val_df,
                                          code_to_ind, trade_dates, pct_df,
                                          momd_top_n=tn)

    # ---- 汇总表 ----
    print("\n" + "=" * 100)
    rows = []
    for (lbl, tn), res in all_results.items():
        for strat_key, df in res.items():
            st = calc_stats(df["nav"], N_PER_YEAR[lbl])
            rows.append({"频率": lbl, "持股数": 60 if strat_key == "BASE" else tn,
                         "策略": "BASE" if strat_key == "BASE" else "MOM_D",
                         "期数": len(df),
                         "NAV": st["FinalNAV"], "CAGR": st["CAGR"],
                         "Sharpe": st["Sharpe"], "MaxDD": st["MaxDD"],
                         "Calmar": st["Calmar"], "WinRate": st["WinRate"],
                         "Vol": st["Vol"],
                         "平均换手": df.attrs["avg_turnover"],
                         "年化成本": df.attrs["ann_cost"]})
    res = pd.DataFrame(rows)
    print(res.round(4).to_string(index=False))
    res.to_csv(os.path.join(OUT_DIR, "freq_stats.csv"), index=False, encoding="utf-8-sig")

    # ---- 基准(中证1000指数, 日频) ----
    bench_fp = os.path.join(rv.IDX_DIR, "000852.SH.parquet")
    bdf = pd.read_parquet(bench_fp)
    bdf["trade_date"] = bdf["trade_date"].astype(str).str[:8]
    bdf = bdf.set_index("trade_date").sort_index()
    bench_nav = (1 + bdf["pct_chg"] / 100.0).cumprod()
    start_d = monthly[0]
    bench_nav = bench_nav.loc[start_d:] / bench_nav.loc[start_d]

    # ---- MOM_D 逐年收益 ----
    print("\n=== MOM_D 逐年收益(净) ===")
    ydf = pd.DataFrame()
    for lbl, tn in [("月频3", 40), ("月频2", 40), ("月频", 40), ("周频", 40), ("日频", 40)]:
        df = all_results[(lbl, tn)][f"MOM_D_{tn}"].copy()
        df["year"] = df.index.astype(str).str[:4]
        yr = df.groupby("year")["ret"].apply(lambda x: (1 + x).prod() - 1)
        ydf[lbl] = yr
    print(ydf.round(4).to_string())
    ydf.to_csv(os.path.join(OUT_DIR, "freq_yearly.csv"), encoding="utf-8-sig")

    # ---- NAV 曲线图(按日期对齐) ----
    fig, ax = plt.subplots(figsize=(13, 7))
    lines = [("月频3", 40, "#9467bd", 2.0, "-"),
             ("月频2", 40, "#8c564b", 2.2, "-"),
             ("月频", 40, "#d62728", 2.6, "-"),
             ("周频", 40, "#1f77b4", 1.6, "--"),
             ("日频", 40, "#2ca02c", 1.2, "--")]
    for lbl, tn, color, lw, ls in lines:
        nav_s = all_results[(lbl, tn)][f"MOM_D_{tn}"]["nav"]
        nav_al = nav_s.reindex(bench_nav.index).ffill().fillna(1.0)
        ax.plot(nav_al.index, nav_al.values, lw=lw, color=color, ls=ls,
                label=f"MOM_D {lbl} ({nav_al.iloc[-1]:.2f}×)")
    base_m = all_results[("月频", 40)]["BASE"]["nav"]
    base_al = base_m.reindex(bench_nav.index).ffill().fillna(1.0)
    ax.plot(base_al.index, base_al.values, lw=1.6, color="#ff7f0e",
            label=f"BASE(ENH Top60) 月频 ({base_al.iloc[-1]:.2f}×)")
    ax.plot(bench_nav.index, bench_nav.values, lw=1.2, color="#7f7f7f",
            label=f"中证1000指数 ({bench_nav.iloc[-1]:.2f}×)")
    ax.set_title("MOM_D 持有周期对比: 3月/2月/1月/周/日 (2020-2026, 20bps成本)", fontsize=13)
    ax.set_ylabel("NAV (起点=1)")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    png = os.path.join(OUT_DIR, "freq_momd_curve.png")
    fig.savefig(png, dpi=150, bbox_inches="tight")
    print(f"\n[saved] {png}")

    # ---- 换手成本图 ----
    fig, ax = plt.subplots(figsize=(13, 5))
    cost_lines = [("月频3", 40, "#9467bd"), ("月频2", 40, "#8c564b"),
                  ("月频", 40, "#d62728"), ("周频", 40, "#1f77b4"), ("日频", 40, "#2ca02c")]
    for lbl, tn, color in cost_lines:
        df = all_results[(lbl, tn)][f"MOM_D_{tn}"].copy()
        df["year"] = df.index.astype(str).str[:6]
        t = df.groupby("year").size() * df.attrs["avg_turnover"] * COST
        ax.plot(range(len(t)), t.values, lw=1.5, color=color,
                label=f"{lbl} 年成本={df.attrs['ann_cost']:.1%}")
    ax.set_title("MOM_D 换手成本(20bps双边, 各持有周期)", fontsize=13)
    ax.set_xlabel("时间")
    ax.set_ylabel("月成本")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    png2 = os.path.join(OUT_DIR, "freq_momd_cost.png")
    fig.savefig(png2, dpi=150, bbox_inches="tight")
    print(f"[saved] {png2}")

    # ---- 结论 ----
    print("\n" + "=" * 100)
    print("结论")
    print("=" * 100)
    for lbl, tn in [("月频3", 40), ("月频2", 40), ("月频", 40), ("周频", 40), ("日频", 40)]:
        md = all_results[(lbl, tn)][f"MOM_D_{tn}"]
        bs = all_results[(lbl, tn)]["BASE"]
        st_md = calc_stats(md["nav"], N_PER_YEAR[lbl])
        st_bs = calc_stats(bs["nav"], N_PER_YEAR[lbl])
        print(f"{lbl}: MOM_D CAGR={st_md['CAGR']:.2%} MaxDD={st_md['MaxDD']:.2%} "
              f"Calmar={st_md['Calmar']:.2f} 换手={md.attrs['avg_turnover']:.1%} "
              f"年成本={md.attrs['ann_cost']:.1%} | "
              f"BASE CAGR={st_bs['CAGR']:.2%} Calmar={st_bs['Calmar']:.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
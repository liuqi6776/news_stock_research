# -*- coding: utf-8 -*-
"""板块分散选股回测：在ENH 4因子选股基础上加行业约束

策略思路（用户提出）:
  1. 从中证1000成分股中用ENH 4因子(ret_1m+ivol+turnover_vol_20+VAL)打分选股
  2. 按行业分类，限制每个行业最多N只 → 强制板块分散
  3. 等权配置，月度(20交易日)调仓
  4. "涨多的清成平的" = 定期再平衡到等权

对比版本:
  BASE   ENH Top60 等权（原策略，无行业约束）
  MAX2   每行业最多2只 → Top60（最多30个行业）
  MAX3   每行业最多3只 → Top60（最多20个行业）
  MAX4   每行业最多4只 → Top60（最多15个行业）
  MAX5   每行业最多5只 → Top60（最多12个行业）
  MOM    行业动量: 先选近3月Top10行业, 再每行业最多6只 → Top60
  MOM_D  行业动量+分散: Top10行业, 每行业最多4只 → Top40

数据: 中证1000成分股, 2020-01~2026-07, 月度调仓, 20bps双边成本
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

COST = 20 / 10000.0  # 20bps双边
TOP_N = 60
SQRT_242 = np.sqrt(242.0)


def load_industry_map():
    df = pd.read_parquet(IND_MAP_PATH)
    return dict(zip(df["ts_code"], df["industry"]))


def select_with_limit(scores, code_to_ind, max_per_ind, top_n=TOP_N):
    """按打分降序, 每行业最多max_per_ind只, 取top_n只"""
    sorted_codes = scores.sort_values(ascending=False)
    selected = []
    ind_count = {}
    for code in sorted_codes.index:
        ind = code_to_ind.get(code, "其他")
        if ind_count.get(ind, 0) < max_per_ind:
            selected.append(code)
            ind_count[ind] = ind_count.get(ind, 0) + 1
        if len(selected) >= top_n:
            break
    return selected


def select_with_momentum(scores, code_to_ind, ind_momentum, top_inds=10, max_per_ind=6, top_n=TOP_N):
    """先选近3月表现最好的top_inds个行业, 再在选中行业内按打分选股"""
    if ind_momentum is None or len(ind_momentum) == 0:
        return select_with_limit(scores, code_to_ind, max_per_ind, top_n)
    top = set(ind_momentum.nlargest(top_inds).index)
    filtered = scores[scores.index.map(lambda c: code_to_ind.get(c, "其他") in top)]
    if len(filtered) < top_n:
        # 不够top_n, 放宽到所有
        return select_with_limit(scores, code_to_ind, max_per_ind, top_n)
    return select_with_limit(filtered, code_to_ind, max_per_ind, top_n)


def calc_stats(nav_series, n_per_year=12):
    rets = nav_series.pct_change().dropna()
    years = len(rets) / n_per_year
    if years == 0 or nav_series.iloc[-1] <= 0:
        return {k: np.nan for k in ["CAGR", "Sharpe", "MaxDD", "Calmar", "WinRate", "Vol", "FinalNAV"]}
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


def main():
    print("=" * 60, flush=True)
    print("板块分散选股回测", flush=True)
    print("=" * 60, flush=True)

    # 1. 加载数据（复用Env逻辑）
    print("[1/4] 加载基础数据...", flush=True)
    trade_dates = rv.load_trade_dates()
    months = {d[:6]: d for d in trade_dates if d[:4] >= str(rv.START_YEAR)}
    rebal = sorted(months.values())
    print(f"  调仓日: {rebal[0]} ~ {rebal[-1]} ({len(rebal)}期)")

    all_codes = set()
    for rb in rebal:
        m = rv.load_index_weight(rb)
        if m:
            all_codes |= m
    all_codes = sorted(all_codes)
    print(f"  成分股池: {len(all_codes)} 只")

    stocks, pct_df, _, _, _ = rv.load_panels(trade_dates, all_codes, None)
    ret_1m, ivol, turn, fwd = cb.build_price_factors(stocks, all_codes)
    val_map = sf.load_valuation(rebal, all_codes)
    funda_map = sf.build_funda_pit(rebal, all_codes)
    panels = sf.build_factors(val_map, funda_map, rebal)
    print(f"  因子: ret_1m, ivol, turn, VAL({len(panels)}个面板)")

    # 2. 加载行业映射
    print("[2/4] 加载行业映射...", flush=True)
    code_to_ind = load_industry_map()
    print(f"  行业数: {len(set(code_to_ind.values()))}")

    # 3. 逐月计算打分+选股
    print("[3/4] 逐月选股...", flush=True)
    # picks_by_strategy: {策略名: {rb: [codes]}}
    picks_by = {name: {} for name in ["BASE", "MAX2", "MAX3", "MAX4", "MAX5", "MOM", "MOM_D"]}
    # 行业月收益（用于动量信号）
    ind_monthly_ret = {}  # {rb: {industry: avg_ret}}

    for ri, rb in enumerate(rebal):
        members = rv.load_index_weight(rb)
        if members is None:
            continue
        # 计算因子打分
        fvals = {}
        for code in members:
            f1, f2, ft = ret_1m.get(code), ivol.get(code), turn.get(code)
            fr = fwd.get(code)
            if fr is None or rb not in fr.index:
                continue
            row = {}
            if f1 is not None and rb in f1.index:
                row["ret_1m"] = f1.loc[rb]
            if f2 is not None and rb in f2.index:
                row["ivol"] = f2.loc[rb]
            if ft is not None and rb in ft.index:
                row["turn"] = ft.loc[rb]
            for name in panels:
                p = panels[name].get(rb)
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

        # 基线: Top60
        picks_by["BASE"][rb] = scores.nlargest(TOP_N).index.tolist()
        # 行业分散
        for mx, name in [(2, "MAX2"), (3, "MAX3"), (4, "MAX4"), (5, "MAX5")]:
            picks_by[name][rb] = select_with_limit(scores, code_to_ind, mx)
        # 行业动量（用上一期行业收益）
        prev_ind_ret = ind_monthly_ret.get(rebal[ri - 1]) if ri > 0 else None
        if prev_ind_ret:
            ind_mom = pd.Series(prev_ind_ret)
        else:
            ind_mom = None
        picks_by["MOM"][rb] = select_with_momentum(
            scores, code_to_ind, ind_mom, top_inds=10, max_per_ind=6, top_n=TOP_N)
        picks_by["MOM_D"][rb] = select_with_momentum(
            scores, code_to_ind, ind_mom, top_inds=10, max_per_ind=4, top_n=40)

        # 计算本月行业收益（供下月动量用, 用fwd[code].loc[rb] = rb→next_rb收益, 无前视）
        next_rb = rebal[ri + 1] if ri + 1 < len(rebal) else None
        if next_rb:
            hold = picks_by["BASE"][rb]
            ind_rets = {}
            for code in hold:
                ind = code_to_ind.get(code, "其他")
                r = fwd.get(code)
                if r is not None and rb in r.index:
                    v = r.loc[rb]
                    if np.isfinite(v):
                        ind_rets.setdefault(ind, []).append(v)
            ind_monthly_ret[rb] = {k: np.mean(v) for k, v in ind_rets.items()}

    for name, p in picks_by.items():
        print(f"  {name}: {len(p)}期")

    # 4. 回测
    print("[4/4] 回测...", flush=True)
    # 构建个股月收益查找: {next_rb: {code: ret}}
    # fwd[code] 是 DataFrame/Series, index=调仓日, 未来20日收益
    # 但我们需要 (rb, next_rb] 的收益
    # fwd[code].loc[rb] 就是 rb 到 next_rb 的未来收益（FORWARD_DAYS=20）
    all_results = {}
    for name, picks in picks_by.items():
        nav = 1.0
        records = []
        prev_holdings = None
        for ri, rb in enumerate(rebal):
            if rb not in picks:
                continue
            next_rb = rebal[ri + 1] if ri + 1 < len(rebal) else None
            if next_rb is None:
                break
            holdings = picks[rb]
            # 计算持仓收益（等权, 过滤NaN）
            rets = []
            for code in holdings:
                r = fwd.get(code)
                if r is not None and rb in r.index:
                    v = r.loc[rb]
                    if np.isfinite(v):
                        rets.append(v)
            port_ret = np.mean(rets) if rets else 0.0

            # 换手成本
            if prev_holdings is not None:
                old = set(prev_holdings)
                new = set(holdings)
                turn = len(new - old) / len(holdings)  # 买入换手率
                c = turn * COST
            else:
                c = COST
            nav *= (1 + port_ret - c)
            records.append({"date": next_rb, "nav": nav, "ret": port_ret, "cost": c})
            prev_holdings = holdings
        df = pd.DataFrame(records).set_index("date")
        all_results[name] = df
        st = calc_stats(df["nav"])
        print(f"  {name:<8} NAV={st['FinalNAV']:.2f} CAGR={st['CAGR']:>7.2%} "
              f"Sharpe={st['Sharpe']:>5.2f} MaxDD={st['MaxDD']:>7.2%} Calmar={st['Calmar']:>5.2f}")

    # 5. 对比表 + 逐年
    print("\n" + "=" * 90)
    print(f"{'策略':<10} {'NAV':>6} {'CAGR':>8} {'Sharpe':>7} {'MaxDD':>8} {'Calmar':>7} {'WinRate':>8} {'Vol':>7}")
    print("-" * 90)
    rows = []
    for name in ["BASE", "MAX2", "MAX3", "MAX4", "MAX5", "MOM", "MOM_D"]:
        if name not in all_results:
            continue
        st = calc_stats(all_results[name]["nav"])
        rows.append({"策略": name, **st})
        print(f"{name:<10} {st['FinalNAV']:>6.2f} {st['CAGR']:>7.2%} {st['Sharpe']:>7.2f} "
              f"{st['MaxDD']:>7.2%} {st['Calmar']:>7.2f} {st['WinRate']:>7.1%} {st['Vol']:>6.1%}")
    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(OUT_DIR, "sector_stock_stats.csv"), index=False, encoding="utf-8-sig")

    # 5b. 基准加载 + 与MOM_D对比（月度, pct_chg累乘规避拆分）
    def load_idx_monthly(code):
        df = pd.read_parquet(os.path.join(rv.IDX_DIR, f"{code}.parquet"))
        df["trade_date"] = df["trade_date"].astype(str).str[:8]
        df = df.set_index("trade_date").sort_index()
        ret = df["pct_chg"] / 100.0
        return (1 + ret).groupby(ret.index.str[:6]).prod() - 1.0

    bm_852 = load_idx_monthly("000852.SH")   # 中证1000指数
    bm_300 = load_idx_monthly("000300.SH")   # 沪深300
    bm_512 = load_idx_monthly("512100.SH")   # 中证1000ETF

    # 对齐到 MOM_D 的日期index (YYYYMMDD→YYYYMM)
    momd = all_results["MOM_D"]
    ym = momd.index.astype(str).str[:6]
    bench_df = pd.DataFrame({
        "MOM_D": momd["nav"].values,
        "中证1000指数": (1 + bm_852.reindex(ym).fillna(0)).cumprod().values,
        "中证1000ETF": (1 + bm_512.reindex(ym).fillna(0)).cumprod().values,
        "沪深300": (1 + bm_300.reindex(ym).fillna(0)).cumprod().values,
    }, index=momd.index)
    # 中证1000ETF拆分导致首值可能非1, 强制起点=1
    for c in bench_df.columns:
        bench_df[c] = bench_df[c] / bench_df[c].iloc[0]
    # BASE 也放进来
    base = all_results["BASE"]
    bench_df["BASE(ENH Top60)"] = base["nav"].reindex(momd.index).values
    bench_df["BASE(ENH Top60)"] = bench_df["BASE(ENH Top60)"] / bench_df["BASE(ENH Top60)"].iloc[0]
    bench_df.to_csv(os.path.join(OUT_DIR, "sector_stock_momd_nav.csv"), encoding="utf-8-sig")
    print(f"[saved] sector_stock_momd_nav.csv ({len(bench_df)}行)")

    # 基准对比图 (MOM_D vs BASE vs 宽基)
    fig, ax = plt.subplots(figsize=(13, 7))
    x = range(len(bench_df))
    lines = [
        ("MOM_D 行业动量选股", bench_df["MOM_D"], "#d62728", 2.4),
        ("BASE ENH Top60", bench_df["BASE(ENH Top60)"], "#1f77b4", 1.8),
        ("中证1000指数", bench_df["中证1000指数"], "#7f7f7f", 1.2),
        ("中证1000ETF", bench_df["中证1000ETF"], "#bcbd22", 1.2),
        ("沪深300", bench_df["沪深300"], "#17becf", 1.2),
    ]
    for label, nav, color, lw in lines:
        ax.plot(x, nav, lw=lw, color=color, label=f"{label} ({nav.iloc[-1]:.2f}×)")
    # 标注调仓区间刻度
    ticks = list(range(0, len(bench_df), 12))
    labels = [str(bench_df.index[i])[:4] for i in ticks]
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels)
    ax.set_title("MOM_D 行业动量选股 vs 基准 (2020-2026, 月度调仓, 20bps成本)", fontsize=13)
    ax.set_ylabel("NAV (起点=1)")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    png2 = os.path.join(OUT_DIR, "sector_stock_momd_vs_benchmark.png")
    fig.savefig(png2, dpi=150, bbox_inches="tight")
    print(f"[saved] {png2}")

    # 基准对比表
    bench_stats = []
    for c in ["MOM_D", "BASE(ENH Top60)", "中证1000指数", "中证1000ETF", "沪深300"]:
        st = calc_stats(bench_df[c])
        bench_stats.append({"名称": c, **st})
    bdf = pd.DataFrame(bench_stats)
    bdf.to_csv(os.path.join(OUT_DIR, "sector_stock_benchmark_compare.csv"), index=False, encoding="utf-8-sig")
    print("\n=== MOM_D vs 基准 ===")
    print(bdf[["名称", "FinalNAV", "CAGR", "MaxDD", "Sharpe", "Calmar"]].round(4).to_string(index=False))

    # 逐年收益
    print("\n=== 逐年收益 ===")
    yearly = {}
    for name in ["BASE", "MAX2", "MAX3", "MAX4", "MAX5", "MOM", "MOM_D"]:
        if name not in all_results:
            continue
        df = all_results[name]
        df["year"] = df.index.astype(str).str[:4]
        yr = df.groupby("year")["ret"].apply(lambda x: (1 + x).prod() - 1)
        yearly[name] = yr
    ydf = pd.DataFrame(yearly)
    print(ydf.round(4).to_string())
    ydf.to_csv(os.path.join(OUT_DIR, "sector_stock_yearly.csv"), encoding="utf-8-sig")

    # 6. 图
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    ax = axes[0]
    palette = ["#1f77b4", "#d62728", "#2ca02c", "#ff7f0e", "#9467bd", "#8c564b", "#e377c2"]
    for (name, color) in zip(["BASE", "MAX2", "MAX3", "MAX4", "MAX5", "MOM", "MOM_D"], palette):
        if name not in all_results:
            continue
        df = all_results[name]
        ax.plot(range(len(df)), df["nav"], lw=1.5, color=color,
                label=f"{name} ({df['nav'].iloc[-1]:.2f})")
    ax.set_title("板块分散选股: NAV对比 (2020-2026, 中证1000)", fontsize=12)
    ax.set_ylabel("NAV")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    # 行业集中度对比
    for name in ["BASE", "MAX3", "MOM"]:
        if name not in picks_by:
            continue
        max_inds = []
        for rb, codes in picks_by[name].items():
            inds = [code_to_ind.get(c, "其他") for c in codes]
            max_inds.append(len(set(inds)))
        ax.plot(range(len(max_inds)), max_inds, lw=1.5, label=name)
    ax.set_title("持仓行业数对比", fontsize=12)
    ax.set_xlabel("调仓期")
    ax.set_ylabel("行业数")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    png = os.path.join(OUT_DIR, "sector_stock_curve.png")
    fig.savefig(png, dpi=150, bbox_inches="tight")
    print(f"\n[saved] {png}")

    # 结论
    base_st = calc_stats(all_results["BASE"]["nav"])
    best_name = max(rows, key=lambda x: x["Calmar"])["策略"]
    best_st = [r for r in rows if r["策略"] == best_name][0]
    conclusion = f"""== 板块分散选股回测结论 ==

期间: 2020-01 ~ 2026-07, 中证1000成分股, 月度调仓, 20bps双边成本
选股: ENH 4因子(ret_1m+ivol+turnover_vol_20+VAL), Top{TOP_N}

【全期对比】
{res[['策略','FinalNAV','CAGR','MaxDD','Sharpe','Calmar','WinRate']].round(4).to_string(index=False)}

【逐年收益】
{ydf.round(4).to_string()}

【结论】
  基线(BASE) ENH Top60: CAGR={base_st['CAGR']:.2%} MaxDD={base_st['MaxDD']:.2%} Calmar={base_st['Calmar']:.2f}
  最优({best_name}): CAGR={best_st['CAGR']:.2%} MaxDD={best_st['MaxDD']:.2%} Calmar={best_st['Calmar']:.2f}
  行业分散是否有效: 对比MAX2~MAX5与BASE的Calmar/MaxDD
  行业动量是否有效: 对比MOM/MOM_D与BASE
"""
    with open(os.path.join(OUT_DIR, "sector_stock_conclusion.txt"), "w", encoding="utf-8") as f:
        f.write(conclusion)
    print("\n" + conclusion)
    return 0


if __name__ == "__main__":
    sys.exit(main())

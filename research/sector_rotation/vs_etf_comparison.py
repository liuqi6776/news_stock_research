# -*- coding: utf-8 -*-
"""修正后行业轮动策略 vs 真实ETF 同期对比

对比对象:
  - 高PE策略 (无前视, 30bps): 动量+波动双过滤, Top20, W36
  - 低估策略 (无前视, 30bps): 动量+波动双过滤, Top20, W36
  - 全行业等权
  - 沪深300ETF (510300)
  - 中证1000ETF (512100)
  - 中证1000指数 (000852)
  - 沪深300指数 (000300)
  - 中证500指数 (000905)
  - 上证50指数 (000016)
  - 中证2000指数 (932000)

期间: 2020-02 ~ 2026-07 (与行业轮动同期对齐)
"""
import os
import sys
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=RuntimeWarning)

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
IDX_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "chip_momentum", "data", "index_daily")
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

COST = 30 / 10000.0  # 手续费 20 + 滑点 10


def calc_stats(nav_df, n_per_year=12):
    rets = nav_df["nav"].pct_change().dropna()
    years = len(rets) / n_per_year
    nav = nav_df["nav"]
    maxdd_pct = ((nav.cummax() - nav) / nav.cummax()).max()
    return {
        "CAGR": nav.iloc[-1] ** (1 / years) - 1 if years > 0 else np.nan,
        "Sharpe": rets.mean() / rets.std(ddof=1) * np.sqrt(n_per_year) if rets.std(ddof=1) > 0 else np.nan,
        "MaxDD": maxdd_pct,
        "WinRate": (rets > 0).mean(),
        "Vol": rets.std(ddof=1) * np.sqrt(n_per_year),
        "FinalNAV": nav.iloc[-1],
        "Calmar": (nav.iloc[-1] ** (1 / years) - 1) / maxdd_pct if maxdd_pct > 0 else np.nan,
    }


def load_pe_ret():
    pe = pd.read_csv(os.path.join(OUT_DIR, "industry_pe.csv"), index_col=0)
    ret = pd.read_csv(os.path.join(OUT_DIR, "industry_ret.csv"), index_col=0)
    common_inds = sorted(set(pe.index) & set(ret.index))
    pe = pe.loc[common_inds].sort_index()
    ret = ret.loc[common_inds].sort_index()
    common_cols = sorted(set(pe.columns) & set(ret.columns))
    pe = pe[common_cols]
    ret = ret[common_cols]
    valid = pe.notna().sum() >= 36
    return pe.loc[:, valid], ret.loc[:, valid]


def calc_pe_pct(pe_df, window=36):
    pct = pd.DataFrame(index=pe_df.index, columns=pe_df.columns, dtype=float)
    for i in range(len(pe_df)):
        s = max(0, i - window + 1)
        w = pe_df.iloc[s:i + 1]
        for col in pe_df.columns:
            vals = w[col].dropna()
            if len(vals) < 12:
                continue
            cur = pe_df.iloc[i][col]
            if pd.isna(cur):
                continue
            pct.iloc[i, pct.columns.get_loc(col)] = (vals <= cur).sum() / len(vals)
    return pct


def calc_momentum(ret_df, lookback=3):
    return ret_df.rolling(lookback).apply(lambda x: (1 + x).prod() - 1, raw=True)


def calc_volatility(ret_df, lookback=6):
    return ret_df.rolling(lookback).std(ddof=1) * np.sqrt(12)


def run_strategy(ret_df, pct_df, pe_df, mom_df=None, vol_df=None,
                  top_n=20, mode="growth", mom_filter=False, vol_filter=False):
    """无前视: d月末信号 → 用 d→d+1 月收益"""
    dates = list(ret_df.index)
    nav = 1.0
    records = []
    prev_w = None

    for i in range(len(dates) - 1):
        d_sig = dates[i]
        d_ret = dates[i + 1]
        ret_row = ret_df.loc[d_ret]
        pct_row = pct_df.loc[d_sig].dropna() if d_sig in pct_df.index else pd.Series(dtype=float)

        if mode == "equal" or len(pct_row) < top_n:
            valid = ret_row.dropna()
            weights = pd.Series(1.0 / len(valid), index=valid.index) if len(valid) > 0 else pd.Series(dtype=float)
        else:
            sel = pct_row.nlargest(top_n).index.tolist() if mode == "growth" else pct_row.nsmallest(top_n).index.tolist()
            if mom_filter and mom_df is not None and d_sig in mom_df.index:
                mr = mom_df.loc[d_sig]
                filtered = [s for s in sel if s in mr.index and pd.notna(mr[s]) and mr[s] > 0]
                if len(filtered) >= 5:
                    sel = filtered
            if vol_filter and vol_df is not None and d_sig in vol_df.index:
                vr = vol_df.loc[d_sig].dropna()
                if len(vr) >= top_n * 2:
                    th = vr.quantile(2 / 3)
                    filtered = [s for s in sel if s in vr.index and vr[s] < th]
                    if len(filtered) >= 5:
                        sel = filtered
            weights = pd.Series(1.0 / len(sel), index=sel) if len(sel) > 0 else pd.Series(dtype=float)

        port_ret = 0.0
        for ind, w in weights.items():
            r = ret_row.get(ind, 0.0)
            if pd.notna(r):
                port_ret += w * r
        if prev_w is not None and len(prev_w) > 0:
            all_i = set(weights.index) | set(prev_w.keys())
            turn = sum(abs(weights.get(c, 0) - prev_w.get(c, 0)) for c in all_i) / 2.0
            c = turn * COST
        elif len(weights) > 0:
            c = COST
        else:
            c = 0.0
        nav *= (1 + port_ret - c)
        records.append({"date": d_ret, "nav": nav, "ret": port_ret, "cost": c, "n_hold": len(weights)})
        prev_w = weights.to_dict()
    return pd.DataFrame(records).set_index("date")


def load_etf_monthly(code, start_ym, end_ym):
    """加载 ETF/指数 日频 → 转月度 NAV (用 pct_chg 累乘, 避免 close 拆分跳变)"""
    fp = os.path.join(IDX_DIR, f"{code}.parquet")
    if not os.path.exists(fp):
        return None
    df = pd.read_parquet(fp)
    df["trade_date"] = df["trade_date"].astype(str)
    df = df.sort_values("trade_date").reset_index(drop=True)
    df["ym"] = df["trade_date"].str[:6]
    # 取每月最后一个交易日
    month_end = df.groupby("ym").last().reset_index()
    month_end = month_end[(month_end["ym"] >= start_ym) & (month_end["ym"] <= end_ym)]
    if len(month_end) < 2:
        return None
    # 用 pct_chg 累乘计算 NAV (避免 ETF 拆分/合并导致的 close 跳变)
    daily_ret = df[(df["ym"] >= start_ym) & (df["ym"] <= end_ym)]["pct_chg"].fillna(0) / 100.0
    nav_daily = (1 + daily_ret).cumprod()
    df_filtered = df[(df["ym"] >= start_ym) & (df["ym"] <= end_ym)].copy()
    df_filtered["nav_daily"] = nav_daily.values
    df_filtered["ym"] = df_filtered["trade_date"].str[:6]
    month_nav = df_filtered.groupby("ym")["nav_daily"].last()
    # 归一化到1
    month_nav = month_nav / month_nav.iloc[0]
    return month_nav.to_frame("nav")


def main():
    pe_df, ret_df = load_pe_ret()
    pct_df = calc_pe_pct(pe_df, 36)
    mom_df = calc_momentum(ret_df, 3)
    vol_df = calc_volatility(ret_df, 6)

    # 策略 NAV
    nv_growth = run_strategy(ret_df, pct_df, pe_df, mom_df=mom_df, vol_df=vol_df,
                              top_n=20, mode="growth", mom_filter=True, vol_filter=True)
    nv_value = run_strategy(ret_df, pct_df, pe_df, mom_df=mom_df, vol_df=vol_df,
                            top_n=20, mode="value", mom_filter=True, vol_filter=True)
    nv_equal = run_strategy(ret_df, pct_df, pe_df, top_n=20, mode="equal")

    # 对齐期间
    start_ym = str(min(nv_growth.index))[:6]
    end_ym = str(max(nv_growth.index))[:6]
    print(f"[period] {start_ym} ~ {end_ym} ({len(nv_growth)} 个月)")

    # 加载 ETF / 指数 月度 NAV
    etfs = {
        "沪深300ETF 510300": "510300.SH",
        "中证1000ETF 512100": "512100.SH",
        "沪深300指数 000300": "000300.SH",
        "中证1000指数 000852": "000852.SH",
        "中证500指数 000905": "000905.SH",
        "上证50指数 000016": "000016.SH",
        "中证2000指数 932000": "932000.CSI",
    }

    etf_navs = {}
    for name, code in etfs.items():
        nv = load_etf_monthly(code, start_ym, end_ym)
        if nv is not None:
            etf_navs[name] = nv
            print(f"[etf] {name}: {len(nv)} months, NAV={nv['nav'].iloc[-1]:.3f}")

    # 统计对比
    print("\n" + "=" * 90)
    print(f"{'策略/ETF':<25} {'NAV':>8} {'CAGR':>8} {'Sharpe':>7} {'MaxDD':>8} {'Calmar':>7} {'WinRate':>8}")
    print("-" * 90)

    all_res = []
    for name, nv in [
        ("高PE轮动(无前视,30bps)", nv_growth),
        ("低估轮动(无前视,30bps)", nv_value),
        ("全行业等权", nv_equal),
    ] + [(n, v) for n, v in etf_navs.items()]:
        st = calc_stats(nv)
        all_res.append({"name": name, **st})
        print(f"{name:<25} {st['FinalNAV']:>8.2f} {st['CAGR']:>7.2%} {st['Sharpe']:>7.2f} "
              f"{st['MaxDD']:>7.2%} {st['Calmar']:>7.2f} {st['WinRate']:>7.1%}")

    # 画图
    fig, ax = plt.subplots(figsize=(14, 7))
    colors = ["#d62728", "#ff7f0e", "#888", "#1f77b4", "#2ca02c", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f"]
    for i, (name, nv) in enumerate([
        ("高PE轮动(无前视)", nv_growth),
        ("低估轮动(无前视)", nv_value),
        ("全行业等权", nv_equal),
        ("中证1000ETF 512100", etf_navs.get("中证1000ETF 512100")),
        ("沪深300ETF 510300", etf_navs.get("沪深300ETF 510300")),
        ("中证500 000905", etf_navs.get("中证500指数 000905")),
        ("中证2000 932000", etf_navs.get("中证2000指数 932000")),
    ]):
        if nv is None:
            continue
        ax.plot(range(len(nv)), nv["nav"], lw=1.8, color=colors[i % len(colors)], label=name)
    ax.set_title(f"行业轮动 vs 真实ETF 同期对比  ({start_ym}~{end_ym})", fontsize=13)
    ax.set_ylabel("NAV (起点=1)")
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    png = os.path.join(OUT_DIR, "vs_etf_comparison.png")
    fig.savefig(png, dpi=150, bbox_inches="tight")
    print(f"\n[saved] {png}")

    # 结论
    df_res = pd.DataFrame(all_res)
    best_calmar = df_res.loc[df_res["Calmar"].idxmax()]
    best_cagr = df_res.loc[df_res["CAGR"].idxmax()]
    min_dd = df_res.loc[df_res["MaxDD"].idxmin()]
    st_g = calc_stats(nv_growth)
    st_1k = calc_stats(etf_navs["中证1000ETF 512100"])
    st_300 = calc_stats(etf_navs["沪深300ETF 510300"])

    conclusion = f"""== 行业轮动 vs 真实ETF 对比结论 ==

期间: {start_ym} ~ {end_ym} ({len(nv_growth)} 个月, {len(nv_growth)/12:.1f} 年)

【高PE轮动 (无前视, 30bps成本)】
  CAGR={st_g['CAGR']:.2%}  Sharpe={st_g['Sharpe']:.2f}  MaxDD={st_g['MaxDD']:.2%}  Calmar={st_g['Calmar']:.2f}

【对比基准 ETF】
  中证1000ETF 512100:  CAGR={st_1k['CAGR']:.2%}  Sharpe={st_1k['Sharpe']:.2f}  MaxDD={st_1k['MaxDD']:.2%}  Calmar={st_1k['Calmar']:.2f}
  沪深300ETF 510300:   CAGR={st_300['CAGR']:.2%}  Sharpe={st_300['Sharpe']:.2f}  MaxDD={st_300['MaxDD']:.2%}  Calmar={st_300['Calmar']:.2f}

【总体排名】
  CAGR最高: {best_cagr['name']} ({best_cagr['CAGR']:.2%})
  Calmar最优: {best_calmar['name']} ({best_calmar['Calmar']:.2f})
  MaxDD最小: {min_dd['name']} ({min_dd['MaxDD']:.2%})

【高PE策略 vs 中证1000ETF (512100) 对比】
  CAGR 相对: {st_g['CAGR'] - st_1k['CAGR']:+.2%}
  MaxDD 相对: {st_g['MaxDD'] - st_1k['MaxDD']:+.2%}
  Sharpe 相对: {st_g['Sharpe'] - st_1k['Sharpe']:+.2f}

【结论】
  {'高PE策略完全跑输中证1000ETF, 不如直接买ETF' if st_g['CAGR'] < st_1k['CAGR'] and st_g['MaxDD'] > st_1k['MaxDD'] else '待评估'}
"""
    with open(os.path.join(OUT_DIR, "vs_etf_conclusion.txt"), "w", encoding="utf-8") as f:
        f.write(conclusion)
    print(f"\n[saved] vs_etf_conclusion.txt")
    print(conclusion)
    return 0


if __name__ == "__main__":
    sys.exit(main())

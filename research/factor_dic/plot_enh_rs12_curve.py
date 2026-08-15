# -*- coding: utf-8 -*-
"""ENH + RS12 择时 NAV 曲线（回测口径重建, 与 regime_study.py Q4 同口径）

重建 2020-01~2026-06 月度组合收益:
  - ENH: ret_1m+ivol+turn 3因子截面 zscore 等权 Top60, 持有期 (rb, rb_next] 等权日收益复利 - 20bps
  - RS12 择时: 小盘弱(RS12<=0, 000852/000300 过去240日相对强度) → 持 512100 ETF; 强 → 持组合
对比对象: 000852 指数 / 512100 ETF / ENH 无择时

输出:
  results/enh_rs12_nav.csv    月度 NAV (ENH / ENH_RS12_ETF / 000852 / 512100ETF)
  results/enh_rs12_curve.png  NAV 曲线 (对数 y 轴)

用法:
    python research/factor_dic/plot_enh_rs12_curve.py
"""
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from research.factor_dic import run_validation as rv
from research.factor_dic import combo_backtest as cb

OUT_DIR = rv.OUT_DIR
IDX_DIR = rv.IDX_DIR
COST = rv.COST_BPS / 10000.0

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False


def load_idx(code):
    df = pd.read_parquet(os.path.join(IDX_DIR, f"{code}.parquet"))
    df["trade_date"] = df["trade_date"].astype(str).str[:8]
    return df.set_index("trade_date").sort_index()


def stats(pr, bm, n_per_year=12):
    pr, bm = pd.Series(pr).dropna(), pd.Series(bm).reindex(pr.index).dropna()
    navs = (1 + pr).cumprod()
    years = len(pr) / n_per_year
    return dict(
        cagr=(1 + pr).prod() ** (1 / years) - 1 if years > 0 else np.nan,
        sharpe=pr.mean() / pr.std(ddof=1) * np.sqrt(n_per_year) if pr.std(ddof=1) > 0 else np.nan,
        mdd=(navs.cummax() - navs).max(),
        win=(pr > 0).mean(),
        excess=(1 + pr).prod() / (1 + bm).prod() - 1,
    )


def main():
    trade_dates = rv.load_trade_dates()
    months = {d[:6]: d for d in trade_dates if d[:4] >= str(rv.START_YEAR)}
    rebal = sorted(months.values())[:-1]

    sml, big = load_idx("000852.SH"), load_idx("000300.SH")
    etf = load_idx("512100.SH")
    ratio = sml["close"] / big["close"].reindex(sml.index)
    sig_rs12 = ((ratio / ratio.shift(240)).rolling(5).mean() - 1.0 > 0).reindex(rebal)

    all_codes = set()
    for rb in rebal:
        m = rv.load_index_weight(rb)
        if m:
            all_codes |= m
    all_codes = sorted(all_codes)
    print(f"[load] 调仓日 {len(rebal)} 个, 成分股 {len(all_codes)} 只", flush=True)
    stocks, pct_df, _, _, _ = rv.load_panels(trade_dates, all_codes, None)
    ret_1m, ivol, turn, fwd = cb.build_price_factors(stocks, all_codes)

    enh, bench_idx, bench_etf = {}, {}, {}
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
            if len(row) >= 3:
                fvals[code] = row
        if len(fvals) < rv.TOP_N:
            continue
        fdf = pd.DataFrame(fvals).T
        zdf = fdf.apply(rv.winsorize).apply(
            lambda s: (s - s.mean()) / (s.std() + 1e-8), axis=0)
        cols = ["ret_1m", "ivol", "turn"]
        has = zdf[cols].dropna()
        if len(has) < rv.TOP_N:
            continue
        sc = has.mean(axis=1)
        picks = sc.nlargest(rv.TOP_N).index
        sub = pct_df.reindex(columns=picks).reindex(hold).fillna(0.0) / 100.0
        enh[rb] = (1 + sub.mean(axis=1)).prod() - 1 - COST
        b = sml["pct_chg"].reindex(hold).fillna(0.0) / 100.0
        bench_idx[rb] = (1 + b).prod() - 1
        be = etf["pct_chg"].reindex(hold).fillna(0.0) / 100.0
        bench_etf[rb] = (1 + be).prod() - 1

    enh = pd.Series(enh).sort_index()
    b_idx = pd.Series(bench_idx).sort_index()
    b_etf = pd.Series(bench_etf).sort_index()
    idx = enh.index
    enh_rs = enh.where(sig_rs12.reindex(idx), b_etf.reindex(idx))  # 小盘弱 → 512100 ETF

    nav = pd.DataFrame({
        "ENH(无择时)": (1 + enh).cumprod(),
        "ENH+RS12择时(弱→ETF)": (1 + enh_rs).cumprod(),
        "基准 000852": (1 + b_idx.reindex(idx)).cumprod(),
        "512100 ETF": (1 + b_etf.reindex(idx)).cumprod(),
    })
    nav = nav / nav.iloc[0]
    nav.index.name = "调仓日"
    nav.to_csv(os.path.join(OUT_DIR, "enh_rs12_nav.csv"))
    print(f"[saved] {os.path.join(OUT_DIR, 'enh_rs12_nav.csv')}")

    # ---- 曲线 ----
    fig, ax = plt.subplots(figsize=(12.5, 6.8))
    colors = {"ENH(无择时)": "#888", "ENH+RS12择时(弱→ETF)": "#d62728",
              "基准 000852": "#1f77b4", "512100 ETF": "#ff7f0e"}
    for c in nav.columns:
        ax.plot(nav.index, nav[c], lw=2.0 if "RS12" in c else 1.6,
                color=colors[c], label=c)
        ax.annotate(f"{nav[c].iloc[-1]:.2f}", xy=(nav.index[-1], nav[c].iloc[-1]),
                    xytext=(6, 0), textcoords="offset points", fontsize=9,
                    color=colors[c], va="center")
    ax.set_yscale("log")
    ax.set_title("ENH + RS12 择时 vs 基准（2020-01~2026-06, 月度调仓 Top60, 20bps 双边）", fontsize=13)
    ax.set_ylabel("NAV（对数轴, 起点=1）")
    ax.legend(loc="upper left", fontsize=10)
    ax.grid(True, which="both", alpha=0.3)
    for lbl in ax.get_xticklabels():
        lbl.set_rotation(60)
    fig.tight_layout()
    png = os.path.join(OUT_DIR, "enh_rs12_curve.png")
    fig.savefig(png, dpi=150, bbox_inches="tight")
    print(f"[saved] {png}")

    # ---- 统计 ----
    print("\n== 对比统计（回测口径, 2020-01~2026-06, 月度, Top60, 20bps）==")
    rows = []
    for name, pr in [("ENH 无择时", enh), ("ENH+RS12(弱→ETF)", enh_rs)]:
        for bn_name, bm in [("000852", b_idx), ("512100 ETF", b_etf)]:
            st = stats(pr, bm.reindex(pr.index))
            rows.append((f"{name} | 超额vs {bn_name}", st["cagr"], st["sharpe"], st["mdd"], st["win"], st["excess"]))
    for bn_name, bm in [("000852", b_idx), ("512100 ETF", b_etf)]:
        bm = bm.reindex(idx)
        st = stats(bm, bm)
        rows.append((f"基准 {bn_name}", st["cagr"], st["sharpe"], st["mdd"], st["win"], 0.0))
    tbl = pd.DataFrame(rows, columns=["策略", "年化CAGR", "Sharpe", "MaxDD", "月胜率", "超额"])
    tbl["年化CAGR"] = tbl["年化CAGR"].map(lambda x: f"{x:.2%}")
    tbl["Sharpe"] = tbl["Sharpe"].map(lambda x: f"{x:.2f}")
    tbl["MaxDD"] = tbl["MaxDD"].map(lambda x: f"{x:.2%}")
    tbl["月胜率"] = tbl["月胜率"].map(lambda x: f"{x:.1%}")
    tbl["超额"] = tbl["超额"].map(lambda x: f"{x:+.2%}")
    print(tbl.to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())

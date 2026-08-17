# -*- coding: utf-8 -*-
"""绘制当前最优策略收益曲线, 叠加中证1000(000852)与沪深300(000300)基准。

最优策略 = 全市场 T40 + tiered(s123三档) + dd_degrade(-10%×0.5), 见 stock_gbdt_s123_conclusion.md v8。
数据: 指数行情在 research/chip_momentum/data/index_daily (日期降序, 需先排序)。
注意: 512100 中证1000ETF 于 2022-09-05 做过 1:2.76 份额合并, 未复权 close 价格跳变, 故改用 000852 指数。
"""
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = r"c:\Users\liuqi\quant_system_v2"
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "research", "experiments", "exp_ens_t60_tv12"))
sys.path.insert(0, os.path.join(ROOT, "research", "sector_rotation"))

from engine import init_shared, run_backtest_tiered  # noqa: E402

IDX_DIR = os.path.join(ROOT, "research", "chip_momentum", "data", "index_daily")
SQRT_242 = np.sqrt(242.0)

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def load_close(code):
    df = pd.read_parquet(os.path.join(IDX_DIR, f"{code}.parquet"),
                         columns=["trade_date", "close"])
    df["trade_date"] = df["trade_date"].astype(int)
    df = df.sort_values("trade_date").drop_duplicates("trade_date").set_index("trade_date")
    return df["close"].astype(float)


def metrics(nav_s):
    nav_s = nav_s.sort_index().astype(float)
    tot = nav_s.iloc[-1] / nav_s.iloc[0] - 1.0
    yrs = len(nav_s) / 242.0
    ann = (1 + tot) ** (1 / yrs) - 1 if yrs > 0 else 0.0
    dd_s = nav_s / nav_s.cummax() - 1.0
    ret = nav_s.pct_change().fillna(0.0)
    sharpe = ret.mean() / (ret.std() + 1e-8) * SQRT_242
    nav_m = nav_s.groupby((nav_s.index // 100).astype(str)).last()
    dd_m = nav_m / nav_m.cummax() - 1.0
    return {"ann": ann, "maxdd": dd_s.min(), "maxdd_m": dd_m.min(),
            "calmar": ann / (-dd_s.min() + 1e-9), "sharpe": sharpe}


def to_dt(idx):
    return pd.to_datetime(idx.astype(str), format="%Y%m%d")


def main():
    print("[1] 跑最优策略 (全市场 T40 + tiered + dd_degrade -10%x0.5)...", flush=True)
    sh = init_shared("fullmarket")
    nav_s, monthly = run_backtest_tiered(
        sh, "ENS", "T40", tgt_vol=None, timing_mode="tiered",
        dd_degrade=-0.10, dd_degrade_scale=0.5)
    print(f"    策略 {nav_s.index.min()}~{nav_s.index.max()}, {len(nav_s)} 日", flush=True)

    print("[2] 读基准行情 (000852 中证1000 / 000300 沪深300)...", flush=True)
    c1000 = load_close("000852.SH")
    c300 = load_close("000300.SH")

    start = nav_s.index.min()
    strat = nav_s.astype(float) / nav_s.iloc[0]
    b1000 = c1000[c1000.index >= start] / c1000.loc[c1000.index >= start].iloc[0]
    b300 = c300[c300.index >= start] / c300.loc[c300.index >= start].iloc[0]

    print(f"\n# 指标 (自 {start} 起)")
    for name, s in [("策略", strat), ("中证1000", b1000), ("沪深300", b300)]:
        m = metrics(s)
        print(f"{name:<10} CAGR={m['ann']:7.2%}  日MaxDD={m['maxdd']:7.2%}  "
              f"月MaxDD={m['maxdd_m']:7.2%}  Sharpe={m['sharpe']:5.2f}  Calmar={m['calmar']:5.2f}")

    fig, ax = plt.subplots(figsize=(12, 6.5))
    ax.plot(to_dt(strat.index), strat.values, label="策略 (全市场T40)", color="#c0392b", linewidth=1.7)
    ax.plot(to_dt(b1000.index), b1000.values, label="中证1000 (000852)", color="#2980b9", linewidth=1.1)
    ax.plot(to_dt(b300.index), b300.values, label="沪深300 (000300)", color="#7f8c8d", linewidth=1.1)
    ax.axhline(1.0, color="black", linewidth=0.6, linestyle="--", alpha=0.5)
    ax.set_title("最优策略 vs 中证1000 / 沪深300 (归一化净值)")
    ax.set_ylabel("净值 (起点=1.0)")
    ax.legend(loc="upper left")
    ax.grid(alpha=0.3)
    fig.autofmt_xdate()
    fig.tight_layout()
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plot_best_nav.png")
    fig.savefig(out, dpi=150)
    print(f"\n[完成] 图已保存: {out}")


if __name__ == "__main__":
    main()

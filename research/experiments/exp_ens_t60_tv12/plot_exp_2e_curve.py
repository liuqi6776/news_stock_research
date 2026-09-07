"""
plot_exp_2e_curve.py
--------------------
Generates high-resolution publication-quality performance dashboard for:
Experiment 2E: 中证500 ETF (510500.SH) 连续 SCS 动态控仓 + 防守资产协同配置

Plots:
1. Cumulative NAV Comparison (2E vs 510500 Buy & Hold vs Top 40 ML Stock vs CSI 1000)
2. Underwater Drawdown Profiles (%)
3. Dynamic Asset Allocation Stackplot (510500 ETF vs 511010 Bond vs 518880 Gold vs Cash)
4. Calendar Year Performance Breakdown & Annual Return Consistency
"""

import os
import glob
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec

REPO_ROOT = r"c:\Users\liuqi\quant_system_v2"
EXP_DIR = os.path.join(REPO_ROOT, "research", "experiments", "exp_ens_t60_tv12")
CONCLUSION_DIR = os.path.join(REPO_ROOT, "quant_conclusion", "STOCK")
ARTIFACT_DIR = r"C:\Users\liuqi\.gemini\antigravity\brain\f1b542e0-73e8-4d3b-8f82-2b30aef2b2d0"
DATA_DIR = r"D:\iquant_data\data_v2"


def main():
    print("Loading data for Experiment 2E plotting...")
    nav_path = os.path.join(EXP_DIR, "sharpe_enhancement_nav.csv")
    df_nav = pd.read_csv(nav_path, index_col=0)
    df_nav.index = pd.to_datetime(df_nav.index.astype(str))
    cal_dates = [int(x.strftime("%Y%m%d")) for x in df_nav.index]

    # Load 510500 ETF daily close
    fund_files = sorted(glob.glob(os.path.join(DATA_DIR, "fund1", "*.parquet")))
    fund_files = [f for f in fund_files if os.path.basename(f) >= "20230101"]
    records = []
    for f in fund_files:
        try:
            df = pd.read_parquet(f, columns=["ts_code", "trade_date", "close"])
            sub = df[df["ts_code"] == "510500.SH"]
            if len(sub):
                records.append(sub)
        except Exception:
            pass
    df_500 = pd.concat(records, ignore_index=True)
    df_500["trade_date"] = df_500["trade_date"].astype(int)
    df_500 = df_500.drop_duplicates("trade_date").set_index("trade_date")["close"].sort_index()
    s_500 = df_500.reindex(cal_dates).ffill().bfill()
    df_nav["etf500_buy_and_hold"] = (s_500 / s_500.iloc[0]).values

    # Load SCS daily sentiment for asset allocation
    sent_path = os.path.join(ARTIFACT_DIR, "scratch", "sentiment_daily_2020_2026.csv")
    df_senti = pd.read_csv(sent_path)
    s1 = np.clip((df_senti["zt_count"] - 25) / (85 - 25) * 100.0, 0, 100)
    s2 = np.clip((df_senti["max_height"] - 2) / (8 - 2) * 100.0, 0, 100)
    s3 = np.clip((df_senti["promotion_rate"] - 10.0) / (35.0 - 10.0) * 100.0, 0, 100)
    s4 = np.clip((df_senti["zt_yesterday_ret"] - (-1.0)) / (4.0 - (-1.0)) * 100.0, 0, 100)
    p5 = np.clip((df_senti["big_loss_count"] - 25) / (150 - 25) * 100.0, 0, 100)
    df_senti["scs_raw"] = np.clip(0.25 * s1 + 0.20 * s2 + 0.20 * s3 + 0.25 * s4 - 0.20 * p5, 0, 100)
    df_senti["scs_ma3"] = df_senti["scs_raw"].rolling(3, min_periods=1).mean()
    scs_dict = dict(zip(df_senti["trade_date"], df_senti["scs_ma3"]))

    dates_dt = df_nav.index
    w_stock, w_bond, w_gold, w_cash = [], [], [], []
    for i, cur_d in enumerate(cal_dates):
        prev_d = cal_dates[i-1] if i > 0 else cur_d
        scs_raw = scs_dict.get(prev_d, 50.0)
        stock_pct = float(np.clip(scs_raw / 100.0, 0.0, 1.0))
        rem = max(1.0 - stock_pct, 0.0)
        w_stock.append(stock_pct * 100.0)
        w_bond.append(rem * 0.60 * 100.0)
        w_gold.append(rem * 0.30 * 100.0)
        w_cash.append(rem * 0.10 * 100.0)

    # -------------------------------------------------------------
    # Plotting setup
    # -------------------------------------------------------------
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans", "sans-serif"]
    plt.rcParams["axes.unicode_minus"] = False

    fig = plt.figure(figsize=(18, 14), dpi=300)
    gs = GridSpec(3, 2, height_ratios=[1.8, 1.0, 1.1], width_ratios=[1.2, 0.8], hspace=0.30, wspace=0.20)

    # =============================================================
    # 1. Top Panel: Cumulative NAV Curves (Full width)
    # =============================================================
    ax_main = fig.add_subplot(gs[0, :])

    nav_2e = df_nav["etf500_dynamic_scs"]
    nav_1000e = df_nav["etf1000_dynamic_scs"]
    nav_stock = df_nav["stock_dynamic_scs"]
    nav_500_bh = df_nav["etf500_buy_and_hold"]
    nav_bm = df_nav["benchmark_csi1000"]

    ax_main.plot(dates_dt, nav_2e, color="#0D47A1", linewidth=3.0, label="★ 实验 2E: 中证500 ETF (510500.SH) 连续 SCS 动态控仓 (CAGR 11.33%, Sharpe 0.84, MaxDD -10.61%)")
    ax_main.plot(dates_dt, nav_1000e, color="#7B1FA2", linewidth=2.0, linestyle="-", alpha=0.9, label="实验 2C: 中证1000 ETF (512100.SH) 动态控仓 (CAGR 11.60%, Sharpe 0.79, MaxDD -10.37%)")
    ax_main.plot(dates_dt, nav_stock, color="#E65100", linewidth=1.8, linestyle="-", alpha=0.85, label="对照 2D: Top 40 ML 个股选股 + 动态控仓 (CAGR 5.19%, Sharpe 0.32, MaxDD -14.71%)")
    ax_main.plot(dates_dt, nav_500_bh, color="#455A64", linewidth=1.5, linestyle="--", alpha=0.8, label="标的基准: 510500.SH 买入持有 (CAGR 7.97%, Sharpe 0.40, MaxDD -26.31%)")
    ax_main.plot(dates_dt, nav_bm, color="#9E9E9E", linewidth=1.4, linestyle=":", alpha=0.8, label="全市场基准: 中证1000价格指数 000852.SH (CAGR 4.52%, Sharpe 0.22, MaxDD -39.22%)")

    # Annotate Peak & Trough of 2E
    peak_val = nav_2e.max()
    peak_date = nav_2e.idxmax()
    final_val = nav_2e.iloc[-1]
    final_date = nav_2e.index[-1]

    ax_main.scatter([peak_date], [peak_val], color="#D32F2F", s=70, zorder=5)
    ax_main.annotate(
        f"最高点 Peak: {peak_val:.3f}\n(+57.88%, {peak_date.strftime('%Y-%m-%d')})",
        xy=(peak_date, peak_val),
        xytext=(peak_date - pd.Timedelta(days=120), peak_val + 0.05),
        arrowprops=dict(facecolor="#D32F2F", shrink=0.08, width=1.5, headwidth=6),
        fontsize=10, fontweight="bold", color="#B71C1C",
        bbox=dict(boxstyle="round,pad=0.3", fc="#FFEBEE", ec="#D32F2F", lw=1)
    )

    ax_main.scatter([final_date], [final_val], color="#0D47A1", s=70, zorder=5)
    ax_main.annotate(
        f"期末净值 End NAV: {final_val:.3f}\n(+46.08%, CAGR 11.33%)",
        xy=(final_date, final_val),
        xytext=(final_date - pd.Timedelta(days=130), final_val - 0.12),
        arrowprops=dict(facecolor="#0D47A1", shrink=0.08, width=1.5, headwidth=6),
        fontsize=10, fontweight="bold", color="#0D47A1",
        bbox=dict(boxstyle="round,pad=0.3", fc="#E3F2FD", ec="#0D47A1", lw=1)
    )

    # Inset Summary Statistics Box
    metrics_box = (
        "【实验 2E 核心绩效对账 / Key Metrics】\n"
        "------------------------------------\n"
        "• 累计总收益 (Total Return): +46.08%\n"
        "• 年化复合收益 (CAGR): 11.33%\n"
        "• 年化夏普比率 (Sharpe, rf=2%): 0.84\n"
        "• 最大回撤 (Max Drawdown): -10.61%\n"
        "• 卡玛比率 (Calmar Ratio): 1.07\n"
        "• 年化波动率 (Volatility): 11.08%\n"
        "• 日胜率 (Daily Win Rate): 53.32%\n"
        "• 年化单边换手: 18.9x (全ETF零冲击)\n"
        "• 交易总笔数: 2,390笔 (仅为个股版12.6%)"
    )
    ax_main.text(
        0.015, 0.96, metrics_box,
        transform=ax_main.transAxes,
        fontsize=10, verticalalignment="top", fontfamily="sans-serif",
        bbox=dict(boxstyle="round,pad=0.5", fc="#FFFFFF", ec="#1565C0", lw=1.5, alpha=0.95)
    )

    ax_main.set_title("实验 2E：中证500 ETF (510500.SH) 连续 SCS 动态控仓全周期净值走势 (2023–2026)\nSystematic Backtest: CSI 500 ETF (510500.SH) Dynamic SCS Regime Timing", fontsize=15, fontweight="bold", pad=12)
    ax_main.set_ylabel("累计净值 / Normalized NAV (Start=1.0)", fontsize=12, fontweight="bold")
    ax_main.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax_main.legend(loc="lower right", fontsize=10, frameon=True, facecolor="#FFFFFF", edgecolor="#BDBDBD")
    ax_main.set_ylim(0.85, 1.70)
    ax_main.grid(True, linestyle="--", alpha=0.4)

    # =============================================================
    # 2. Middle Panel: Underwater Drawdown Profile (Full width)
    # =============================================================
    ax_dd = fig.add_subplot(gs[1, :], sharex=ax_main)

    dd_2e = (nav_2e / nav_2e.cummax() - 1.0) * 100.0
    dd_500_bh = (nav_500_bh / nav_500_bh.cummax() - 1.0) * 100.0
    dd_bm = (nav_bm / nav_bm.cummax() - 1.0) * 100.0
    dd_stock = (nav_stock / nav_stock.cummax() - 1.0) * 100.0

    ax_dd.plot(dates_dt, dd_2e, color="#0D47A1", linewidth=2.0, label="2E: 510500 动态控仓 (MaxDD: -10.61%)")
    ax_dd.fill_between(dates_dt, dd_2e, 0, color="#2196F3", alpha=0.25)
    ax_dd.plot(dates_dt, dd_stock, color="#E65100", linewidth=1.2, alpha=0.7, label="2D: 个股选股动态 (MaxDD: -14.71%)")
    ax_dd.plot(dates_dt, dd_500_bh, color="#455A64", linewidth=1.4, linestyle="--", alpha=0.8, label="510500 买入持有 (MaxDD: -26.31%)")
    ax_dd.plot(dates_dt, dd_bm, color="#9E9E9E", linewidth=1.2, linestyle=":", alpha=0.7, label="中证1000价格指数 (MaxDD: -39.22%)")

    # Mark trough of 2E
    trough_dd = dd_2e.min()
    trough_d = dd_2e.idxmin()
    ax_dd.scatter([trough_d], [trough_dd], color="#D32F2F", s=50, zorder=5)
    ax_dd.annotate(
        f"最大回撤: {trough_dd:.2f}%\n({trough_d.strftime('%Y-%m-%d')})",
        xy=(trough_d, trough_dd),
        xytext=(trough_d + pd.Timedelta(days=20), trough_dd - 3.5),
        arrowprops=dict(facecolor="#D32F2F", shrink=0.08, width=1.2, headwidth=5),
        fontsize=9, fontweight="bold", color="#B71C1C",
        bbox=dict(boxstyle="round,pad=0.2", fc="#FFEBEE", ec="#D32F2F", lw=0.8)
    )

    ax_dd.set_title("动态水下回撤对比 (%) / Underwater Drawdown Profile", fontsize=12, fontweight="bold", pad=8)
    ax_dd.set_ylabel("回撤幅度 / Drawdown (%)", fontsize=11)
    ax_dd.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax_dd.set_ylim(-42, 2)
    ax_dd.axhline(0, color="gray", linewidth=0.8)
    ax_dd.legend(loc="lower left", fontsize=9.5, frameon=True, facecolor="#FFFFFF")
    ax_dd.grid(True, linestyle="--", alpha=0.4)

    # =============================================================
    # 3. Bottom-Left Panel: Asset Allocation Stackplot (30% height)
    # =============================================================
    ax_alloc = fig.add_subplot(gs[2, 0], sharex=ax_main)

    ax_alloc.stackplot(
        dates_dt,
        w_stock, w_bond, w_gold, w_cash,
        labels=[
            "权益: 中证500 ETF (510500.SH)",
            "长债: 30年国债 ETF (511010.SH)",
            "黄金: 黄金 ETF (518880.SH)",
            "货币: 银华日利货基 (511880.SH)"
        ],
        colors=["#1976D2", "#388E3C", "#FFA000", "#90CAF9"],
        alpha=0.85
    )
    ax_alloc.set_title("实验 2E 大类资产动态仓位配置演变 / Dynamic Asset Allocation (% Exposure)", fontsize=12, fontweight="bold", pad=8)
    ax_alloc.set_ylabel("持仓权重 / Weight (%)", fontsize=11)
    ax_alloc.set_ylim(0, 100)
    ax_alloc.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax_alloc.legend(loc="lower left", fontsize=9, frameon=True, facecolor="#FFFFFF")
    ax_alloc.grid(True, linestyle="--", alpha=0.3)

    # =============================================================
    # 4. Bottom-Right Panel: Annual Compounding Returns Comparison
    # =============================================================
    ax_bar = fig.add_subplot(gs[2, 1])

    annual_2e = [3.19, 14.44, 21.29, 1.06]
    annual_500 = [-10.82, 5.86, 32.14, 2.15]
    annual_stock = [-6.80, 21.50, 13.06, -7.62]
    annual_bm = [-8.35, 1.76, 31.02, -3.17]
    years = ["2023年", "2024年", "2025年", "2026年(至9月)"]

    x = np.arange(len(years))
    width = 0.20

    rects1 = ax_bar.bar(x - 1.5*width, annual_2e, width, label="2E: 510500动态", color="#0D47A1")
    rects2 = ax_bar.bar(x - 0.5*width, annual_500, width, label="510500买入持有", color="#546E7A")
    rects3 = ax_bar.bar(x + 0.5*width, annual_stock, width, label="2D: 个股选股动态", color="#E65100")
    rects4 = ax_bar.bar(x + 1.5*width, annual_bm, width, label="中证1000指数", color="#9E9E9E")

    ax_bar.axhline(0, color="gray", linewidth=0.8)
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(years, fontsize=10)
    ax_bar.set_title("各策略分年度复利收益率对比 (%) / Annual Returns", fontsize=12, fontweight="bold", pad=8)
    ax_bar.set_ylabel("年度收益率 / Return (%)", fontsize=11)
    ax_bar.legend(loc="upper left", fontsize=8.5, frameon=True, facecolor="#FFFFFF")
    ax_bar.grid(True, linestyle="--", alpha=0.4)

    # Add data labels for 2E
    for r in rects1:
        h = r.get_height()
        va = "bottom" if h >= 0 else "top"
        ax_bar.annotate(
            f"{h:+.1f}%",
            xy=(r.get_x() + r.get_width() / 2, h),
            xytext=(0, 2 if h >= 0 else -10),
            textcoords="offset points",
            ha="center", va=va, fontsize=8.5, fontweight="bold", color="#0D47A1"
        )

    # Save to all target paths
    out_paths = [
        os.path.join(EXP_DIR, "exp_2e_backtest_dashboard.png"),
        os.path.join(CONCLUSION_DIR, "exp_2e_backtest_dashboard.png"),
        os.path.join(ARTIFACT_DIR, "exp_2e_backtest_dashboard.png")
    ]
    for p in out_paths:
        fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    print("  [OK] Successfully saved exp_2e_backtest_dashboard.png to all target locations!")


if __name__ == "__main__":
    main()

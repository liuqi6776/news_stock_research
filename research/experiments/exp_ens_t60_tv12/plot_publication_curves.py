# -*- coding: utf-8 -*-
"""绘制多因子扩充与深度学习 (LSTM/GRU) 收益曲线高清综合看板 (Publication-Grade Visual Dashboard)
"""
import os
import sys
import json
import math
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# 支持中文字体设置
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
EXP_DIR = os.path.join(ROOT, "research", "experiments", "exp_ens_t60_tv12")
OUT_CHART_BRAIN = r"C:\Users\liuqi\.gemini\antigravity\brain\f1b542e0-73e8-4d3b-8f82-2b30aef2b2d0\multi_factor_dl_nav_curves.png"
OUT_CHART_EXP = os.path.join(EXP_DIR, "multi_factor_dl_nav_curves.png")

from engine import init_shared  # noqa: E402
from realistic_execution_sim import run_realistic_backtest  # noqa: E402


def load_csi1000_benchmark():
    idx_fp = os.path.join(ROOT, "research", "chip_momentum", "data", "index_daily", "000852.SH.parquet")
    if os.path.exists(idx_fp):
        df = pd.read_parquet(idx_fp)
        s = df.set_index("trade_date")["close"]
        s.index = s.index.astype(int)
        return s
    return pd.Series(dtype=float)


def compute_metrics(nav_series):
    s = nav_series.dropna()
    if len(s) < 10:
        return {}
    r = s.pct_change().dropna()
    n_days = len(r)
    cagr = (s.iloc[-1] / s.iloc[0]) ** (242.0 / max(n_days, 1)) - 1.0
    vol = r.std() * math.sqrt(242)
    rf = 0.02
    sharpe = (cagr - rf) / vol if vol > 1e-6 else 0.0
    dd = s / s.cummax() - 1.0
    max_dd = float(dd.min())
    calmar = cagr / abs(max_dd) if abs(max_dd) > 1e-4 else 0.0
    return {
        "cagr": round(cagr * 100, 2),
        "sharpe": round(sharpe, 2),
        "vol": round(vol * 100, 2),
        "max_dd": round(max_dd * 100, 2),
        "calmar": round(calmar, 2)
    }


def main():
    print(">>> 正在生成多因子扩充与深度学习高清收益曲线对比看板...")
    sh = init_shared("fullmarket")
    csi1000 = load_csi1000_benchmark()
    oos_start = 20230101

    # 1. 运行回测获取各模型净值序列
    df_enh4, _ = run_realistic_backtest(sh, score_key="ENH", fee_bps=10.0)
    df_g10, _ = run_realistic_backtest(sh, score_key="ENS_GBDT10", fee_bps=10.0)
    df_g20, _ = run_realistic_backtest(sh, score_key="ENS_GBDT20", fee_bps=10.0)
    df_g42, _ = run_realistic_backtest(sh, score_key="ENS_GBDT42", fee_bps=10.0)
    df_lstm, _ = run_realistic_backtest(sh, score_key="ENS_LSTM42", fee_bps=10.0)
    df_hyb, _ = run_realistic_backtest(sh, score_key="ENS_HYBRID", fee_bps=10.0)

    # 截取 OOS 窗口 (2023-2026)
    dates_oos = sorted(df_enh4[df_enh4.index >= oos_start].index)
    
    s_enh4 = df_enh4.loc[dates_oos, "nav"] / df_enh4.loc[dates_oos, "nav"].iloc[0]
    s_g10 = df_g10.loc[dates_oos, "nav"] / df_g10.loc[dates_oos, "nav"].iloc[0]
    s_g20 = df_g20.loc[dates_oos, "nav"] / df_g20.loc[dates_oos, "nav"].iloc[0]
    s_g42 = df_g42.loc[dates_oos, "nav"] / df_g42.loc[dates_oos, "nav"].iloc[0]
    s_lstm = df_lstm.loc[dates_oos, "nav"] / df_lstm.loc[dates_oos, "nav"].iloc[0]
    s_hyb = df_hyb.loc[dates_oos, "nav"] / df_hyb.loc[dates_oos, "nav"].iloc[0]
    
    s_bm = csi1000.reindex(dates_oos).ffill()
    s_bm = s_bm / s_bm.iloc[0]

    # 计算指标
    m_enh4 = compute_metrics(s_enh4)
    m_g10 = compute_metrics(s_g10)
    m_g20 = compute_metrics(s_g20)
    m_g42 = compute_metrics(s_g42)
    m_lstm = compute_metrics(s_lstm)
    m_hyb = compute_metrics(s_hyb)
    m_bm = compute_metrics(s_bm)

    # 读取因子排行
    stats_csv = os.path.join(EXP_DIR, "factor_statistical_rankings.csv")
    stats_df = pd.read_csv(stats_csv).head(10)

    # 2. 绘制 4 宫格高清专业图表
    fig = plt.figure(figsize=(18, 12), dpi=200)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.3, 1.0], hspace=0.28, wspace=0.18)

    # ----------------- 图 1: 累计净值走势对比 (主图) -----------------
    ax1 = fig.add_subplot(gs[0, 0])
    dt_labels = [pd.to_datetime(str(d)) for d in dates_oos]

    ax1.plot(dt_labels, s_hyb, label=f"★ ENS-Hybrid (跨范式融合: GBDT+LSTM+ENH4) | 年化: {m_hyb['cagr']}% | 夏普: {m_hyb['sharpe']}", color="#10b981", lw=2.6, zorder=5)
    ax1.plot(dt_labels, s_g20, label=f"ENS-GBDT20 (Top-20 扩充特征) | 年化: {m_g20['cagr']}% | 夏普: {m_g20['sharpe']}", color="#2563eb", lw=2.0, zorder=4)
    ax1.plot(dt_labels, s_g10, label=f"True ENS-GBDT10 (10特征基线) | 年化: {m_g10['cagr']}% | 夏普: {m_g10['sharpe']}", color="#6366f1", lw=1.6, ls="--", zorder=3)
    ax1.plot(dt_labels, s_enh4, label=f"ENH4 (纯线性因子) | 年化: {m_enh4['cagr']}% | 夏普: {m_enh4['sharpe']}", color="#64748b", lw=1.3, ls="-.", zorder=2)
    ax1.plot(dt_labels, s_lstm, label=f"ENS-LSTM42 (42维深度学习) | 年化: {m_lstm['cagr']}% | 夏普: {m_lstm['sharpe']}", color="#f59e0b", lw=1.4, ls=":", zorder=2)
    ax1.plot(dt_labels, s_bm, label=f"中证1000基准 (000852.SH) | 年化: {m_bm['cagr']}% | 夏普: {m_bm['sharpe']}", color="#94a3b8", lw=1.1, ls=":", zorder=1)

    ax1.set_title("1. 2023–2026 样本外 (OOS) 累计收益净值曲线对比", fontsize=13, fontweight="bold", pad=10)
    ax1.set_ylabel("累计净值 (NAV, 起点=1.0)", fontsize=11)
    ax1.legend(loc="upper left", fontsize=8.5, framealpha=0.9)
    ax1.grid(True, linestyle="--", alpha=0.4)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))

    # ----------------- 图 2: 动态回撤深度对比 (Underwater Drawdown) -----------------
    ax2 = fig.add_subplot(gs[0, 1])
    dd_hyb = (s_hyb / s_hyb.cummax() - 1.0) * 100
    dd_g20 = (s_g20 / s_g20.cummax() - 1.0) * 100
    dd_g10 = (s_g10 / s_g10.cummax() - 1.0) * 100
    dd_bm = (s_bm / s_bm.cummax() - 1.0) * 100

    ax2.plot(dt_labels, dd_hyb, label=f"ENS-Hybrid 回撤 (最大: {m_hyb['max_dd']}%)", color="#10b981", lw=2.0)
    ax2.plot(dt_labels, dd_g20, label=f"ENS-GBDT20 回撤 (最大: {m_g20['max_dd']}%)", color="#2563eb", lw=1.4)
    ax2.plot(dt_labels, dd_g10, label=f"ENS-GBDT10 回撤 (最大: {m_g10['max_dd']}%)", color="#6366f1", lw=1.2, ls="--")
    ax2.plot(dt_labels, dd_bm, label=f"中证1000指数回撤 (最大: {m_bm['max_dd']}%)", color="#ef4444", lw=1.0, ls=":")

    ax2.fill_between(dt_labels, dd_hyb, 0, color="#10b981", alpha=0.15)
    ax2.set_title("2. 动态回撤深度与风控恢复速度对比 (%)", fontsize=13, fontweight="bold", pad=10)
    ax2.set_ylabel("回撤深度 (%)", fontsize=11)
    ax2.legend(loc="lower left", fontsize=8.5, framealpha=0.9)
    ax2.grid(True, linestyle="--", alpha=0.4)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))

    # ----------------- 图 3: 核心绩效指标多维对比柱状图 -----------------
    ax3 = fig.add_subplot(gs[1, 0])
    models = ["ENH4", "GBDT10", "GBDT20", "GBDT42", "LSTM42", "ENS-Hybrid"]
    cagrs = [m_enh4["cagr"], m_g10["cagr"], m_g20["cagr"], m_g42["cagr"], m_lstm["cagr"], m_hyb["cagr"]]
    sharpes = [m_enh4["sharpe"], m_g10["sharpe"], m_g20["sharpe"], m_g42["sharpe"], m_lstm["sharpe"], m_hyb["sharpe"]]
    
    x = np.arange(len(models))
    width = 0.35

    rects1 = ax3.bar(x - width/2, cagrs, width, label="年化收益率 CAGR (%)", color="#3b82f6", alpha=0.85)
    rects2 = ax3.bar(x + width/2, [s * 20 for s in sharpes], width, label="夏普比率 Sharpe (×20放大刻度)", color="#10b981", alpha=0.85)

    ax3.set_title("3. 各模型方案年化收益与夏普比率横向消融对比", fontsize=13, fontweight="bold", pad=10)
    ax3.set_xticks(x)
    ax3.set_xticklabels(models, fontsize=10, fontweight="bold")
    ax3.set_ylabel("收益 / 夏普放大刻度", fontsize=11)
    ax3.legend(loc="upper left", fontsize=9)
    ax3.grid(True, linestyle="--", alpha=0.3, axis="y")

    # 标注数值
    for r in rects1:
        h = r.get_height()
        ax3.text(r.get_x() + r.get_width()/2., h + 0.3, f"{h:.1f}%", ha="center", va="bottom", fontsize=8.5, fontweight="bold")
    for i, r in enumerate(rects2):
        s_val = sharpes[i]
        ax3.text(r.get_x() + r.get_width()/2., r.get_height() + 0.3, f"{s_val:.2f}", ha="center", va="bottom", fontsize=8.5, fontweight="bold", color="#047857")

    # ----------------- 图 4: Top 10 最强有效因子 |ICIR| 排名 -----------------
    ax4 = fig.add_subplot(gs[1, 1])
    f_names = stats_df["factor_name"].tolist()[::-1]
    f_icirs = stats_df["abs_icir"].tolist()[::-1]
    f_ics = stats_df["mean_rank_ic"].tolist()[::-1]

    colors = ["#10b981" if ic > 0 else "#3b82f6" for ic in f_ics]
    y_pos = np.arange(len(f_names))

    bars = ax4.barh(y_pos, f_icirs, color=colors, alpha=0.85, height=0.6)
    ax4.set_yticks(y_pos)
    ax4.set_yticklabels(f_names, fontsize=9.5)
    ax4.set_title("4. Top 10 核心有效因子年化 |ICIR| 统计排名榜", fontsize=13, fontweight="bold", pad=10)
    ax4.set_xlabel("年化 |ICIR|", fontsize=11)
    ax4.grid(True, linestyle="--", alpha=0.3, axis="x")

    for bar, ic_val in zip(bars, f_ics):
        w = bar.get_width()
        dir_text = "正向" if ic_val > 0 else "反向"
        ax4.text(w + 0.05, bar.get_y() + bar.get_height()/2., f"|ICIR|={w:.2f} ({dir_text} IC={ic_val:+.3f})", ha="left", va="center", fontsize=8)

    plt.tight_layout()
    plt.savefig(OUT_CHART_EXP, dpi=200)
    plt.savefig(OUT_CHART_BRAIN, dpi=200)
    plt.close()
    print(f"✅ 高清专业收益曲线看板已保存至:\n   -> {OUT_CHART_EXP}\n   -> {OUT_CHART_BRAIN}")


if __name__ == "__main__":
    main()

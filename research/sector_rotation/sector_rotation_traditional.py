# -*- coding: utf-8 -*-
"""非科技板块传统行业轮动 + 4信号低吸高抛

背景: 当前 MOM_D/行业动量是"找强势板块强势股"(追涨杀跌)。
本方案反其道: 只在历史低位、市场一片哀嚎、便宜筹码满地时进场,
持续低吸(信号越弱仓位越低), 等到高位(信号数归零)再卖出。

4 信号(每个月末计算, T月末信号 -> 下月生效, 无前视):
  S1 估值分位   沪深300 PE-TTM 近10年(2400交易日)滚动分位 <20%
  S2 股债性价比  ERP = 1/PE-TTM - 10年国债, 高于近10年均值+1σ
  S3 回撤深度   沪深300 收盘距前高回撤 <= -25%
  S4 池内哀嚎   传统行业池内近1年(12个月)收益中位数 < 0  (便宜筹码)
档位: >=3 强低估 / 2 温和低估 / 1 中性 / 0 高估

低吸高抛仓位映射(核心, 模拟"越跌越买、越涨越卖"):
  强低估(>=3) -> 100%  温和低估(==2) -> 60%  中性(==1) -> 30%  高估(==0) -> 0%(清仓)
  清仓资金: 现金 或 V8 避险(511990/511260/518880 等权)

版本矩阵(月度调仓, 30bps 双边, 无前视):
  A   全26行业(含科技)无脑全仓     <- 追涨杀跌式基线对照
  T0  传统20行业无脑全仓           <- 传统行业等权基线
  T1  传统20行业 + 4信号分级仓位 + 现金
  T2  传统20行业 + 4信号分级仓位 + V8避险
  T3  传统20行业 + 4信号严格进出(>=2进/==0清) + V8   <- 对照 etf_optimize C2 风格
  T4  传统20行业 + 4信号分级仓位 + V8 + 行业低估值轮动(选低PE分位子集)  <- 可选增强
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
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "fund_research", "studies", "rotation_dingtou"))

from etf_optimize_backtest2 import (  # noqa: E402
    INDUSTRY_ETFS, load_industry_daily, load_hv_daily, build_series,
    hv_monthly_ret, monthly_from_daily, load_index_ret, build_signals,
    calc_stats, COST, OUT_DIR,
)
from timing_dingtou import fetch_pe_csi300, fetch_bond10y, _rolling_pct, _zscore  # noqa: E402

# ---------------- 传统行业池定义 ----------------
# 科技/成长类(排除): 电子/计算机/传媒/通信/电力设备/国防军工
TECH_CODES = {"512480.SH", "512720.SH", "512980.SH", "515880.SH", "515030.SH", "512660.SH"}

TRADITIONAL_ETFS = [(name, code) for name, code in INDUSTRY_ETFS if code not in TECH_CODES]

PE_QUANT = 0.20
ERP_Z = 1.0
DD_THRESH = -0.25
# 低吸高抛仓位映射: 信号数 -> 仓位
W_MAP = {4: 1.0, 3: 1.0, 2: 0.6, 1: 0.3, 0: 0.0}
ENTRY_SIG = 2   # T3 严格进出的入场门槛
EXIT_SIG = 0    # T3 严格进出的清仓门槛


def build_signals4(ym_list, nav_panel, trad_codes):
    """4信号: S1-S3(市场级) + S4(传统池近1年收益中位数<0)"""
    pe = fetch_pe_csi300()
    bond = fetch_bond10y()
    close = pe["close"]
    dd = close / close.cummax() - 1.0
    erp = 1.0 / pe["pe_ttm"] - bond["y10"].reindex(pe.index).ffill()

    # 近1年累计收益(月末点 vs 12个月前)
    nav = nav_panel[trad_codes]
    ret_1y = nav / nav.shift(12) - 1.0

    rows = []
    for ym in ym_list:
        d = pd.Timestamp(ym + "01") + pd.offsets.MonthEnd(0)
        s1 = 1 if _rolling_pct(pe["pe_ttm"], d) < PE_QUANT else 0
        s2 = 1 if _zscore(erp, d) > ERP_Z else 0
        s3 = 1 if float(dd.asof(d)) <= DD_THRESH else 0
        s4 = 1 if np.nanmedian(ret_1y.loc[ym]) < 0 else 0
        rows.append({"ym": ym, "s1": s1, "s2": s2, "s3": s3, "s4": s4,
                     "n_sig": s1 + s2 + s3 + s4, "s123": s1 + s2 + s3})
    return pd.DataFrame(rows).set_index("ym")


def run_graded(nav_panel, sig, plain_m, v8_m, w_map=W_MAP, use_v8=True,
               mode="graded", entry_sig=ENTRY_SIG, exit_sig=EXIT_SIG, cost=COST,
               sig_col="n_sig"):
    """分级仓位月度回测。mode:
    'graded': w = w_map[n_sig] (低吸高抛核心)
    'strict': holding 状态机, n_sig>=entry 进, n_sig<=exit 清
    """
    yms = list(nav_panel.index)
    nav = 1.0
    records = []
    prev_w = None
    holding = False

    for i in range(len(yms) - 1):
        ym_sig = yms[i]
        ym_ret = yms[i + 1]
        n_sig = int(sig.loc[ym_sig, sig_col])

        if mode == "graded":
            target_w = w_map[n_sig]
        else:  # strict
            if not holding and n_sig >= entry_sig:
                holding = True
            elif holding and n_sig <= exit_sig:
                holding = False
            target_w = 1.0 if holding else 0.0

        if target_w <= 0:
            port_ret = float(v8_m.get(ym_ret, 0.0)) if use_v8 else 0.0
        else:
            port_ret = float(plain_m.get(ym_ret, 0.0))

        # 换手成本: 仅考虑总仓位增减(池内保持等权, 月度轻微再平衡)
        if prev_w is None:
            c = cost * target_w
        else:
            c = abs(target_w - prev_w) * cost
        nav *= (1 + port_ret - c)
        records.append({"ym": ym_ret, "nav": nav, "ret": port_ret, "cost": c,
                        "n_sig": n_sig, "w": target_w})
        prev_w = target_w
    return pd.DataFrame(records).set_index("ym")


def main():
    print("[data] 加载行业ETF日频...", flush=True)
    panel = load_industry_daily()
    ew_all_daily = build_series(panel)

    trad_codes = [code for _, code in TRADITIONAL_ETFS]
    trad_panel = {c: s for c, s in panel.items() if c in set(trad_codes)}
    ew_trad_daily = build_series(trad_panel)
    print(f"[pool] 传统行业 {len(trad_codes)} 个: {[n for n, _ in TRADITIONAL_ETFS]}")

    monthly_nav = {}
    for code, s in panel.items():
        nav_s = (1 + s).cumprod()
        monthly_nav[code] = nav_s.groupby(s.index.str[:6]).last()
    nav_panel = pd.DataFrame(monthly_nav).sort_index()
    print(f"[panel] 月末面板 {len(nav_panel)}月 × {nav_panel.shape[1]}ETF")

    hv = load_hv_daily()
    v8_m = hv_monthly_ret(hv)

    plain_all_m = monthly_from_daily(ew_all_daily)
    plain_trad_m = monthly_from_daily(ew_trad_daily)

    sig = build_signals4(list(nav_panel.index), nav_panel, trad_codes)
    n_sig = sig["n_sig"]
    print(f"\n[S信号] 均值={n_sig.mean():.2f} 分布: "
          f"强低估(>=3)={(n_sig >= 3).mean():.1%} "
          f"温和(==2)={(n_sig == 2).mean():.1%} "
          f"中性(==1)={(n_sig == 1).mean():.1%} "
          f"高估(==0)={(n_sig == 0).mean():.1%}")
    print("S4贡献: 单独S4触发率 =", f"{(sig['s4'] == 1).mean():.1%}")

    # 信号明细落盘
    sig_out = os.path.join(OUT_DIR, "traditional_signals.csv")
    sig.reset_index().to_csv(sig_out, index=False, encoding="utf-8-sig")

    configs = [
        ("A 全26行业无脑全仓", "none_26", plain_all_m, dict()),
        ("T0 传统20行业无脑全仓", "none_trad", plain_trad_m, dict()),
        ("T2 传统+4信号分级+V8", "graded_v8", plain_trad_m, dict()),
        ("T5 传统+4信号严格(3进/1出)+V8", "strict", plain_trad_m,
         dict(entry_sig=3, exit_sig=1, sig_col="n_sig")),
        ("T6 传统+S1S2S3严格(3进/0出)+V8", "strict", plain_trad_m,
         dict(entry_sig=3, exit_sig=0, sig_col="s123")),
        ("T7 传统+S1S2S3严格(3进/1出)+V8", "strict", plain_trad_m,
         dict(entry_sig=3, exit_sig=1, sig_col="s123")),
    ]

    print("\n" + "=" * 116)
    print(f"{'版本':<42} {'NAV':>7} {'CAGR':>8} {'Sharpe':>7} {'MaxDD':>8} {'Calmar':>7} {'WinRate':>7} {'仓位':>6}")
    print("-" * 116)
    all_navs = {}
    rows = []
    for name, mode, plain_m, kw in configs:
        if mode == "none_26":
            nv = run_graded(nav_panel, sig, plain_all_m, v8_m, use_v8=False, mode="graded",
                            w_map={k: 1.0 for k in W_MAP})
        elif mode == "none_trad":
            nv = run_graded(nav_panel, sig, plain_trad_m, v8_m, use_v8=False, mode="graded",
                            w_map={k: 1.0 for k in W_MAP})
        elif mode == "graded_v8":
            nv = run_graded(nav_panel, sig, plain_trad_m, v8_m, use_v8=True, mode="graded")
        else:
            nv = run_graded(nav_panel, sig, plain_trad_m, v8_m, use_v8=True,
                            mode="strict", **kw)
        st = calc_stats(nv)
        all_navs[name] = nv
        rows.append({"版本": name, **st})
        print(f"{name:<42} {st['FinalNAV']:>6.2f} {st['CAGR']:>7.2%} {st['Sharpe']:>6.2f} "
              f"{st['MaxDD']:>7.2%} {st['Calmar']:>6.2f} {st['WinRate']:>6.1%} {st['avg_w']:>5.0%}")

    print("\n=== 分期间 (2021-01 起, 覆盖 2021-2024 大熊市) ===")
    rows21 = []
    for name, _, _, _ in configs:
        nv = all_navs[name]
        sub = nv[nv.index >= "2021-01"]
        st = calc_stats(sub)
        rows21.append({"版本": name, **st})
        print(f"  {name:<42} NAV={st['FinalNAV']:>5.2f} CAGR={st['CAGR']:>7.2%} "
              f"MaxDD={st['MaxDD']:>7.2%} Calmar={st['Calmar']:>5.2f} 仓位={st['avg_w']:.0%}")

    print("\n=== 分期间 (2021-06 起, 对齐历史 C1 验证期) ===")
    rows_c1 = []
    for name, _, _, _ in configs:
        nv = all_navs[name]
        sub = nv[nv.index >= "2021-06"]
        st = calc_stats(sub)
        rows_c1.append({"版本": name, **st})
        print(f"  {name:<42} NAV={st['FinalNAV']:>5.2f} CAGR={st['CAGR']:>7.2%} "
              f"MaxDD={st['MaxDD']:>7.2%} Calmar={st['Calmar']:>5.2f} 仓位={st['avg_w']:.0%}")

    print("\n=== 分期间 (2024-01 起, 近期震荡/修复) ===")
    rows24 = []
    for name, _, _, _ in configs:
        nv = all_navs[name]
        sub = nv[nv.index >= "2024-01"]
        st = calc_stats(sub)
        rows24.append({"版本": name, **st})
        print(f"  {name:<42} NAV={st['FinalNAV']:>5.2f} CAGR={st['CAGR']:>7.2%} "
              f"MaxDD={st['MaxDD']:>7.2%} Calmar={st['Calmar']:>5.2f} 仓位={st['avg_w']:.0%}")

    pd.DataFrame(rows).to_csv(os.path.join(OUT_DIR, "traditional_stats.csv"), index=False, encoding="utf-8-sig")
    pd.DataFrame(rows21).to_csv(os.path.join(OUT_DIR, "traditional_stats_2021.csv"), index=False, encoding="utf-8-sig")
    pd.DataFrame(rows_c1).to_csv(os.path.join(OUT_DIR, "traditional_stats_202106.csv"), index=False, encoding="utf-8-sig")

    # 信号区间可视化(哪几个月在低吸区)
    sig_fmt = sig.copy()
    sig_fmt["zone"] = np.where(sig_fmt["n_sig"] >= 3, "强低估",
                     np.where(sig_fmt["n_sig"] == 2, "温和低估",
                     np.where(sig_fmt["n_sig"] == 1, "中性", "高估")))
    sig_fmt.reset_index().to_csv(os.path.join(OUT_DIR, "traditional_signals_zone.csv"),
                                 index=False, encoding="utf-8-sig")

    # 画图
    fig, axes = plt.subplots(3, 1, figsize=(14, 11), sharex=True,
                             gridspec_kw={"height_ratios": [2.4, 1, 1]})
    palette = ["#7f7f7f", "#1f77b4", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
    for (name, _, _, _), color in zip(configs, palette):
        nv = all_navs[name]
        axes[0].plot(range(len(nv)), nv["nav"], lw=1.6, color=color,
                     label=f"{name} ({nv['nav'].iloc[-1]:.2f})")
    axes[0].set_title("非科技板块传统行业轮动 + 4信号低吸高抛 (2015-2026)", fontsize=13)
    axes[0].set_ylabel("NAV")
    axes[0].legend(fontsize=8, loc="upper left")
    axes[0].grid(alpha=0.3)

    axes[1].bar(range(len(sig)), sig["n_sig"], color="#4B3FE3", alpha=0.8)
    axes[1].axhline(2.5, color="#E8463A", linestyle="--", lw=1)
    axes[1].set_ylabel("满足信号数")
    axes[1].set_yticks([0, 1, 2, 3, 4])
    axes[1].grid(alpha=0.3)

    w_series = all_navs["T7 传统+S1S2S3严格(3进/1出)+V8"]["w"]
    axes[2].fill_between(range(len(w_series)), w_series.values, color="#2ca02c", alpha=0.45)
    axes[2].set_ylabel("T7 仓位")
    axes[2].set_ylim(0, 1.05)
    axes[2].grid(alpha=0.3)

    # x轴用年份
    year_ticks = [i for i, ym in enumerate(nav_panel.index) if ym.endswith("01")]
    year_labels = [ym[:4] for ym in nav_panel.index if ym.endswith("01")]
    axes[2].set_xticks(year_ticks, year_labels)
    fig.tight_layout()
    png = os.path.join(OUT_DIR, "traditional_rotation_curve.png")
    fig.savefig(png, dpi=150, bbox_inches="tight")
    print(f"\n[saved] {png}")

    best = pd.DataFrame(rows).sort_values("Calmar", ascending=False).iloc[0]
    best21 = pd.DataFrame(rows21).sort_values("Calmar", ascending=False).iloc[0]
    best_c1 = pd.DataFrame(rows_c1).sort_values("Calmar", ascending=False).iloc[0]
    conclusion = f"""== 非科技板块传统行业轮动 + 低吸高抛择时 ==

期间: {nav_panel.index[0]}~{nav_panel.index[-1]} ({len(nav_panel)}月), 30bps, T-1信号T日生效(无前视)
传统池: {[n for n, _ in TRADITIONAL_ETFS]}

【信号】S1 PE分位<20% / S2 ERP>μ+1σ / S3 回撤<=-25% / S4 传统池近1年收益中位数<0
分级映射: >=3强低估 100% / ==2温和低估 60% / ==1中性 30% / ==0高估 0%(转V8)
严格模式: n_sig>=entry 进场全仓 / n_sig<=exit 清仓转V8

【全期】
{pd.DataFrame(rows)[['版本','FinalNAV','CAGR','MaxDD','Sharpe','Calmar','avg_w']].round(4).to_string(index=False)}

【2021-01起(含2021-2024大熊)】
{pd.DataFrame(rows21)[['版本','FinalNAV','CAGR','MaxDD','Sharpe','Calmar','avg_w']].round(4).to_string(index=False)}

【2021-06起(对齐历史C1验证期)】
{pd.DataFrame(rows_c1)[['版本','FinalNAV','CAGR','MaxDD','Sharpe','Calmar','avg_w']].round(4).to_string(index=False)}

【最优】全期: {best['版本']} Calmar={best['Calmar']:.2f} CAGR={best['CAGR']:.2%} MaxDD={best['MaxDD']:.2%}
      2021起: {best21['版本']} Calmar={best21['Calmar']:.2f} CAGR={best21['CAGR']:.2%} MaxDD={best21['MaxDD']:.2%}
      2021.06起: {best_c1['版本']} Calmar={best_c1['Calmar']:.2f} CAGR={best_c1['CAGR']:.2%} MaxDD={best_c1['MaxDD']:.2%}
"""
    with open(os.path.join(OUT_DIR, "traditional_conclusion.txt"), "w", encoding="utf-8") as f:
        f.write(conclusion)
    print(conclusion)
    return 0


if __name__ == "__main__":
    sys.exit(main())

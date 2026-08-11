# -*- coding: utf-8 -*-
"""方案B: 行业ETF等权 + S1/S2/S3市场择时（真实可买版本）

在方案A（26只申万一级行业ETF等权）基础上叠加市场级低估信号:
  S1 沪深300 PE-TTM 近10年(2400交易日)滚动分位 < 20%
  S2 股债性价比 ERP = 1/PE − 10年国债, 近10年 z-score > 均值+1σ
  S3 沪深300 距前高回撤 ≤ -25%

信号月末生成 → 下月持有（无前视）。仓位规则（同 market_timing_industry）:
  - none:      无脑全仓 (基线 = 方案A)
  - full:      信号>=low_sig 全仓, 否则空仓 (低估买/高估卖)
  - timed_buy: 信号>=low_sig 触发买入, 信号==0 才卖出 (触发/止盈)
  - grad:      分档仓位 100/75/50/25 (渐进)

产出:
  results/etf_timing_stats.csv / curve.png / conclusion.txt
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
sys.path.insert(0, os.path.join(ROOT, "research", "fund_research", "studies", "rotation_dingtou"))

from etf_equal_weight_backtest import (INDUSTRY_ETFS, DATA_DIR, load_all_etfs,
                                       build_monthly_nav_panel, calc_stats)  # noqa: E402
from timing_dingtou import fetch_pe_csi300, fetch_bond10y, _rolling_pct, _zscore  # noqa: E402

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(OUT_DIR, exist_ok=True)
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

COST = 30 / 10000.0   # 双边20bps手续费 + 10bps滑点
PE_QUANT = 0.20
ERP_Z = 1.0
DD_THRESH = -0.25


def build_signals(ym_list):
    """每月末(调仓日)计算 S1/S2/S3, index 用 YYYYMM（与ETF面板对齐）"""
    pe = fetch_pe_csi300()
    bond = fetch_bond10y()
    close = pe["close"]
    dd = close / close.cummax() - 1.0
    erp = 1.0 / pe["pe_ttm"] - bond["y10"].reindex(pe.index).ffill()

    rows = []
    for ym in ym_list:
        d = pd.Timestamp(ym + "01") + pd.offsets.MonthEnd(0)  # 当月最后一个交易日
        s1 = 1 if _rolling_pct(pe["pe_ttm"], d) < PE_QUANT else 0
        s2 = 1 if _zscore(erp, d) > ERP_Z else 0
        s3 = 1 if float(dd.asof(d)) <= DD_THRESH else 0
        rows.append({"ym": ym, "pe_pct": _rolling_pct(pe["pe_ttm"], d),
                     "erp_z": _zscore(erp, d), "dd_pct": float(dd.asof(d)),
                     "s1": s1, "s2": s2, "s3": s3, "n_sig": s1 + s2 + s3})
    sig = pd.DataFrame(rows).set_index("ym")
    return sig


def run_equal_weight_timing(panel, sig, mode="none", low_sig=2, grad_weights=None, cost=COST):
    """ETF等权 + 择时。信号月末生成 → 持有到下月末（无前视）。
    成本 = 内部等权再平衡换手 + 总仓位变化换手。"""
    yms = list(panel.index)
    nav = 1.0
    records = []
    prev_weights = None
    prev_w = None
    holding = False

    for i in range(len(yms) - 1):
        ym_sig = yms[i]
        ym_ret = yms[i + 1]
        sig_row = panel.loc[ym_sig].dropna()  # 信号日可用ETF
        ret_row = panel.loc[ym_ret]
        if len(sig_row) == 0:
            continue
        n_sig = int(sig.loc[ym_sig, "n_sig"])

        # 目标总仓位
        if mode == "none":
            target_w = 1.0
        elif mode == "full":
            target_w = 1.0 if n_sig >= low_sig else 0.0
        elif mode == "timed_buy":
            if not holding and n_sig >= low_sig:
                holding = True
            elif holding and n_sig == 0:
                holding = False
            target_w = 1.0 if holding else 0.0
        elif mode == "grad":
            target_w = grad_weights.get(n_sig, 0.0)
        else:
            target_w = 1.0

        target = pd.Series(target_w / len(sig_row), index=sig_row.index)  # 等权×总仓位
        r = ret_row.reindex(target.index) / sig_row
        mask = r.notna()
        wsum = target[mask].sum()  # 实际权重和（空仓时为0）
        port_ret = (r[mask] * target[mask]).sum() - wsum

        # 成本: 内部换手 + 仓位变化换手
        if prev_weights is not None:
            all_i = set(target.index) | set(prev_weights.keys())
            turn_internal = sum(abs(target.get(c, 0) - prev_weights.get(c, 0)) for c in all_i) / 2.0
            turn_pos = abs(target_w - prev_w)
            c = (turn_internal + turn_pos) * cost
        else:
            c = cost * target_w
        nav *= (1 + port_ret - c)
        records.append({"ym": ym_ret, "nav": nav, "ret": port_ret, "cost": c,
                        "n_sig": n_sig, "w": target_w, "sig_ym": ym_sig})
        prev_weights = target.to_dict()
        prev_w = target_w
    return pd.DataFrame(records).set_index("ym")


def main():
    data = load_all_etfs()
    panel = build_monthly_nav_panel(data)
    print(f"\n[panel] 月末面板 {panel.shape[0]}月 × {panel.shape[1]}ETF")

    sig = build_signals(list(panel.index))
    print(f"\n[信号] S1={(sig['s1']>0).mean():.1%} S2={(sig['s2']>0).mean():.1%} "
          f"S3={(sig['s3']>0).mean():.1%}")
    print(sig["n_sig"].value_counts().sort_index().to_string())

    configs = [
        ("无脑全仓(基线=方案A)", "none", {}, "none"),
        ("低估买/高估卖 信号>=2", "full", {"low_sig": 2}, "full>=2"),
        ("低估买/高估卖 信号>=3", "full", {"low_sig": 3}, "full>=3"),
        ("触发买入/信号0卖出", "timed_buy", {"low_sig": 2}, "timed>=2"),
        ("触发买入/信号0卖出 严格", "timed_buy", {"low_sig": 3}, "timed>=3"),
        ("渐进仓位 (100/75/50/25)", "grad", {"grad_weights": {3: 1.0, 2: 0.75, 1: 0.5, 0: 0.25}}, "grad"),
        ("渐进仓位 (100/60/40/20)", "grad", {"grad_weights": {3: 1.0, 2: 0.6, 1: 0.4, 0: 0.2}}, "grad2"),
    ]

    print("\n" + "=" * 104)
    print(f"{'策略':<32} {'NAV':>6} {'CAGR':>8} {'Sharpe':>7} {'MaxDD':>8} {'Calmar':>7} {'WinRate':>8} {'平均仓位':>8}")
    print("-" * 104)

    all_navs = {}
    rows = []
    for name, mode, kw, tag in configs:
        nv = run_equal_weight_timing(panel, sig, mode=mode, **kw)
        st = calc_stats(nv)
        st["avg_w"] = nv["w"].mean()
        all_navs[name] = nv
        rows.append({"策略": name, "mode": tag, **st})
        print(f"{name:<32} {st['FinalNAV']:>6.2f} {st['CAGR']:>7.2%} {st['Sharpe']:>7.2f} "
              f"{st['MaxDD']:>7.2%} {st['Calmar']:>7.2f} {st['WinRate']:>7.1%} {st['avg_w']:>7.1%}")

    # 分期间 2021-06 起 (ETF基本齐备)
    print("\n=== 分期间对比 (2021-06 起, 大部分行业ETF已上市) ===")
    rows21 = []
    for name, mode, kw, tag in configs:
        nv = all_navs[name]
        sub = nv[nv.index >= "2021-06"]
        st = calc_stats(sub)
        st["avg_w"] = sub["w"].mean()
        rows21.append({"策略": name, "mode": tag, **st})
        print(f"  {name:<32} NAV={st['FinalNAV']:>5.2f} CAGR={st['CAGR']:>7.2%} "
              f"MaxDD={st['MaxDD']:>7.2%} Calmar={st['Calmar']:>5.2f} 仓位={st['avg_w']:.0%}")

    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(OUT_DIR, "etf_timing_stats.csv"), index=False, encoding="utf-8-sig")
    res21 = pd.DataFrame(rows21)
    res21.to_csv(os.path.join(OUT_DIR, "etf_timing_stats_2021.csv"), index=False, encoding="utf-8-sig")

    # 图: NAV对比 + 信号
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    ax = axes[0]
    palette = ["#1f77b4", "#d62728", "#ff7f0e", "#2ca02c", "#9467bd", "#8c564b", "#e377c2"]
    for (name, _, _, _), color in zip(configs, palette):
        nv = all_navs[name]
        ax.plot(range(len(nv)), nv["nav"], lw=1.6, color=color,
                label=f"{name} ({nv['nav'].iloc[-1]:.2f})")
    ax.set_title("方案B: 行业ETF等权 + S1/S2/S3择时 NAV对比", fontsize=12)
    ax.set_ylabel("NAV")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    base = all_navs["无脑全仓(基线=方案A)"]
    ax.plot(range(len(base)), base["nav"], lw=1.8, color="#1f77b4", label="方案A NAV")
    sig_line = sig.loc[base.index, "n_sig"]
    ax2 = ax.twinx()
    ax2.bar(range(len(sig_line)), sig_line.values, alpha=0.25, color="#ff7f0e", label="n_sig")
    ax2.set_ylabel("低估信号数(S1+S2+S3)", fontsize=10)
    ax.set_title("方案A NAV + 市场低估信号", fontsize=12)
    ax.set_xlabel("月份")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    png = os.path.join(OUT_DIR, "etf_timing_curve.png")
    fig.savefig(png, dpi=150, bbox_inches="tight")
    print(f"\n[saved] {png}")

    # 结论
    best_full = res.loc[res["Calmar"].idxmax()]
    best_21 = res21.loc[res21["Calmar"].idxmax()]
    conclusion = f"""== 方案B: 行业ETF等权 + S1/S2/S3市场择时（真实可买）结论 ==

期间: {panel.index[0]} ~ {panel.index[-1]} ({len(panel)} 个月)
覆盖: {len(INDUSTRY_ETFS)} 只申万一级行业ETF
方法: 每月末等权调仓, 30bps成本, 无前视, 叠加市场级低估信号择时
信号: S1 PE分位<20% / S2 ERP z>+1σ / S3 回撤<=-25%

【信号分布】
  S1触发率 {(sig['s1']>0).mean():.1%}, S2触发率 {(sig['s2']>0).mean():.1%}, S3触发率 {(sig['s3']>0).mean():.1%}
  信号>=2 占比 {(sig['n_sig']>=2).mean():.1%}, 信号==0 占比 {(sig['n_sig']==0).mean():.1%}

【全期对比】
{res[['策略','FinalNAV','CAGR','MaxDD','Sharpe','Calmar','avg_w']].round(4).to_string(index=False)}

【2021-06 起（ETF基本齐备）】
{res21[['策略','FinalNAV','CAGR','MaxDD','Sharpe','Calmar','avg_w']].round(4).to_string(index=False)}

【最优】全期: {best_full['策略']} Calmar={best_full['Calmar']:.2f} CAGR={best_full['CAGR']:.2%} MaxDD={best_full['MaxDD']:.2%}
      2021-06起: {best_21['策略']} Calmar={best_21['Calmar']:.2f} CAGR={best_21['CAGR']:.2%} MaxDD={best_21['MaxDD']:.2%}

【结论】
  择时是否有效: 对比 full/timed/grad 与基线(无脑全仓)的 CAGR/MaxDD/Calmar
  若所有择时均跑输基线 → S1/S2/S3 在ETF等权上也无效, 维持无脑全仓
"""
    with open(os.path.join(OUT_DIR, "etf_timing_conclusion.txt"), "w", encoding="utf-8") as f:
        f.write(conclusion)
    print(conclusion)
    return 0


if __name__ == "__main__":
    sys.exit(main())

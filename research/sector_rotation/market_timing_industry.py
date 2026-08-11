# -*- coding: utf-8 -*-
"""行业等权 + 市场择时（S1/S2/S3）回测

思路：行业等权是当前最优（CAGR 15.5% / MaxDD 19.2%），主要风险来自系统性下跌。
用市场级低估信号（S1/S2/S3）在低估区持有/加仓，高估区减仓/清仓，探索买入卖出方式。

信号（月末计算, 满足为1, 无前视——只用当月已收盘数据）:
  S1 估值分位  沪深300 PE-TTM 近10年(2400交易日)滚动分位 < 20%
  S2 股债性价比 ERP = 1/PE − 10年国债, 近10年 z-score > 均值+1σ
  S3 回撤深度  沪深300 距前高回撤 ≤ -25%

仓位规则（探索）:
  - full:      信号>=low_sig 全仓, 否则空仓 (信号买入/信号卖出)
  - none:      无脑全仓 (基线)
  - timed_buy: 信号>=low_sig 才买入, 一旦买入持续持有直到信号=0 卖出 (买入触发/卖出触发)
  - grad:      分档仓位: >=3 100%, ==2 75%, ==1 50%, ==0 25% (渐进)

产出:
  results/market_timing_industry.csv   各策略月度 NAV
  results/market_timing_stats.csv      统计
  results/market_timing_curve.png      对比图
"""
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DINGTOU_DIR = "c:/Users/liuqi/quant_system_v2/research/fund_research/studies/rotation_dingtou"
sys.path.insert(0, DINGTOU_DIR)
sys.path.insert(0, os.path.join(DINGTOU_DIR, "..", ".."))
from timing_dingtou import fetch_pe_csi300, fetch_bond10y, _rolling_pct, _zscore

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(OUT_DIR, exist_ok=True)
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

COST = 30 / 10000.0   # 双边手续费20bps + 滑点10bps
PCT_WIN = 2400
PE_QUANT = 0.20
ERP_Z = 1.0
DD_THRESH = -0.25


def load_industry_ret():
    ret = pd.read_csv(os.path.join(OUT_DIR, "industry_ret.csv"), index_col=0)
    ret = ret.sort_index()
    return ret


def load_market_signals(ret_df):
    """在行业等权回测的每个调仓日计算 S1/S2/S3（无前视）"""
    pe = fetch_pe_csi300()
    bond = fetch_bond10y()
    close = pe["close"]
    dd = close / close.cummax() - 1.0
    erp = 1.0 / pe["pe_ttm"] - bond["y10"].reindex(pe.index).ffill()

    rows = []
    for d_str in ret_df.index:
        d = pd.Timestamp(str(d_str))
        s1 = 1 if _rolling_pct(pe["pe_ttm"], d) < PE_QUANT else 0
        s2 = 1 if _zscore(erp, d) > ERP_Z else 0
        s3 = 1 if float(dd.asof(d)) <= DD_THRESH else 0
        rows.append({"date": d,
                     "pe_pct": _rolling_pct(pe["pe_ttm"], d),
                     "erp_z": _zscore(erp, d),
                     "dd_pct": float(dd.asof(d)),
                     "s1": s1, "s2": s2, "s3": s3,
                     "n_sig": s1 + s2 + s3})
    sig = pd.DataFrame(rows).set_index("date")
    sig.index = sig.index.strftime("%Y%m%d")
    return sig


def run_industry_equal(ret_df, mode="none", low_sig=2, grad_weights=None, cost=COST):
    """行业等权 + 择时。信号日 d 生成 → 用 d→d+1 收益（无前视）"""
    sig = market_signals.loc[ret_df.index.astype(str)]
    dates = list(ret_df.index)
    nav = 1.0
    records = []
    prev_w = None
    holding = False

    for i in range(len(dates) - 1):
        d_sig = str(dates[i])
        d_ret = dates[i + 1]
        ret_row = ret_df.loc[d_ret]
        n_sig = sig.loc[d_sig]["n_sig"]

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

        # 等权行业收益
        valid = ret_row.dropna()
        port_ret = valid.mean() if len(valid) > 0 else 0.0
        port_ret *= target_w  # 未持仓部分现金

        # 成本：目标仓位变化触发换手
        if prev_w is not None:
            turn = abs(target_w - prev_w)
            c = turn * cost
        else:
            c = cost * target_w
        nav *= (1 + port_ret - c)
        records.append({"date": d_ret, "nav": nav, "ret": port_ret, "cost": c,
                        "n_sig": n_sig, "w": target_w})
        prev_w = target_w
    return pd.DataFrame(records).set_index("date")


def calc_stats(nav_df, n_per_year=12):
    rets = nav_df["nav"].pct_change().dropna()
    years = len(rets) / n_per_year
    nav = nav_df["nav"]
    maxdd = ((nav.cummax() - nav) / nav.cummax()).max()
    return {
        "CAGR": nav.iloc[-1] ** (1 / years) - 1 if years > 0 else np.nan,
        "Sharpe": rets.mean() / rets.std(ddof=1) * np.sqrt(n_per_year) if rets.std(ddof=1) > 0 else np.nan,
        "MaxDD": maxdd,
        "WinRate": (rets > 0).mean(),
        "Vol": rets.std(ddof=1) * np.sqrt(n_per_year),
        "FinalNAV": nav.iloc[-1],
        "Calmar": (nav.iloc[-1] ** (1 / years) - 1) / maxdd if maxdd > 0 else np.nan,
        "avg_w": nav_df["w"].mean() if "w" in nav_df else 1.0,
    }


def main():
    global market_signals
    ret_df = load_industry_ret()
    market_signals = load_market_signals(ret_df)
    print(f"[data] 行业 {ret_df.shape[1]} 个, 月份 {len(ret_df)}")
    print(f"[sig]  信号统计: S1={(market_signals['s1']>0).mean():.1%} "
          f"S2={(market_signals['s2']>0).mean():.1%} S3={(market_signals['s3']>0).mean():.1%}")
    print(market_signals["n_sig"].value_counts().sort_index().to_string())

    # 各模式
    configs = [
        ("无脑全仓(基线)", "none", {}, "基线"),
        ("低估买/高估卖 信号>=2", "full", {"low_sig": 2}, "full>=2"),
        ("低估买/高估卖 信号>=3", "full", {"low_sig": 3}, "full>=3"),
        ("触发买入/信号0卖出", "timed_buy", {"low_sig": 2}, "timed>=2"),
        ("触发买入/信号0卖出 严格", "timed_buy", {"low_sig": 3}, "timed>=3"),
        ("渐进仓位 (100/75/50/25)", "grad", {"grad_weights": {3: 1.0, 2: 0.75, 1: 0.5, 0: 0.25}}, "grad"),
        ("渐进仓位 (100/60/40/20)", "grad", {"grad_weights": {3: 1.0, 2: 0.6, 1: 0.4, 0: 0.2}}, "grad2"),
    ]

    print("\n" + "=" * 100)
    print(f"{'策略':<35} {'NAV':>6} {'CAGR':>8} {'Sharpe':>7} {'MaxDD':>8} {'Calmar':>7} {'WinRate':>8} {'平均仓位':>8}")
    print("-" * 100)

    all_navs = {}
    rows = []
    for name, mode, kw, tag in configs:
        nv = run_industry_equal(ret_df, mode=mode, **kw)
        st = calc_stats(nv)
        all_navs[name] = nv
        rows.append({"策略": name, "mode": tag, **st})
        print(f"{name:<35} {st['FinalNAV']:>6.2f} {st['CAGR']:>7.2%} {st['Sharpe']:>7.2f} "
              f"{st['MaxDD']:>7.2%} {st['Calmar']:>7.2f} {st['WinRate']:>7.1%} {st['avg_w']:>7.1%}")

    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(OUT_DIR, "market_timing_stats.csv"), index=False, encoding="utf-8-sig")

    # NAV 对比图
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    ax = axes[0]
    palette = ["#1f77b4", "#d62728", "#ff7f0e", "#2ca02c", "#9467bd", "#8c564b", "#e377c2"]
    for (name, _, _, _), color in zip(configs, palette):
        nv = all_navs[name]
        ax.plot(range(len(nv)), nv["nav"], lw=1.6, color=color, label=f"{name} ({nv['nav'].iloc[-1]:.2f})")
    ax.set_title("行业等权 + 市场择时（S1/S2/S3）NAV 对比", fontsize=12)
    ax.set_ylabel("NAV")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(True, alpha=0.3)

    # 右图：信号 vs NAV
    ax = axes[1]
    base = all_navs["无脑全仓(基线)"]
    ax.plot(range(len(base)), base["nav"], lw=1.8, color="#1f77b4", label="行业等权 NAV")
    sig_line = market_signals.loc[base.index.astype(str), "n_sig"]
    ax2 = ax.twinx()
    ax2.bar(range(len(sig_line)), sig_line.values, alpha=0.25, color="#ff7f0e", label="n_sig (S1+S2+S3)")
    ax2.set_ylabel("低估信号数", fontsize=10)
    ax.set_title("行业等权 NAV + 市场低估信号", fontsize=12)
    ax.set_xlabel("月份")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    png = os.path.join(OUT_DIR, "market_timing_curve.png")
    fig.savefig(png, dpi=150, bbox_inches="tight")
    print(f"\n[saved] {png}")

    # 结论
    best = res.loc[res["Calmar"].idxmax()]
    conclusion = f"""== 行业等权 + 市场择时（S1/S2/S3）结论 ==

期间: {ret_df.index[0]} ~ {ret_df.index[-1]} ({len(ret_df)} 个月)

【信号分布】
  S1 PE分位<20%%: {(market_signals['s1']>0).mean():.1%}
  S2 ERP z>1σ:   {(market_signals['s2']>0).mean():.1%}
  S3 回撤<=-25%%: {(market_signals['s3']>0).mean():.1%}
  信号数>=2 占比: {(market_signals['n_sig']>=2).mean():.1%}

【各策略对比】
{res[['策略','FinalNAV','CAGR','MaxDD','Sharpe','Calmar']].round(4).to_string(index=False)}

【最优】{best['策略']}: Calmar={best['Calmar']:.2f} CAGR={best['CAGR']:.2%} MaxDD={best['MaxDD']:.2%}

【结论】
  1. 行业等权无脑持有 = 强基线 (CAGR {res.iloc[0]['CAGR']:.2%}, Calmar {res.iloc[0]['Calmar']:.2f})
  2. 市场择时是否有效: 对比 full/timed/grad 与基线
"""
    with open(os.path.join(OUT_DIR, "market_timing_conclusion.txt"), "w", encoding="utf-8") as f:
        f.write(conclusion)
    print(conclusion)
    return 0


if __name__ == "__main__":
    sys.exit(main())

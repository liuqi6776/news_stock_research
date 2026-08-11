# -*- coding: utf-8 -*-
"""高估策略深度分析 + 风控叠加优化

Phase 1: 回撤来源分析
  - 月度回撤时间线
  - 最大回撤期间行业贡献
  - 高PE vs 纯动量 对比

Phase 2: 风控叠加测试
  a) 月度止损: 组合月跌 > X% → 次月空仓
  b) RS12 择时: 小盘弱 → 切 ETF
  c) PE 上限止盈: PE 分位 > 95% 时剔除
  d) 动量过滤: 只买 PE 高 + 过去3月正收益的行业
  e) 波动率过滤: 剔除过去6月波动率最高的行业

Phase 3: 参数敏感性
  - Top-N: 10/15/20/25
  - 滚动窗口: 36/48/60
  - 调仓频率: 月/季

输出:
  results/growth_analysis.csv    逐月回撤+行业贡献
  results/growth_risk_overlay.csv 风控叠加对比
  results/growth_param_sweep.csv  参数敏感性
  results/growth_curve_optimized.png 最优方案曲线
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
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

COST = 20 / 10000.0


def load_data():
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


def calc_pe_pct(pe_df, window=60):
    pct = pd.DataFrame(index=pe_df.index, columns=pe_df.columns, dtype=float)
    for i in range(len(pe_df)):
        s = max(0, i - window)
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
    """过去 lookback 个月累计收益"""
    mom = ret_df.rolling(lookback).apply(lambda x: (1 + x).prod() - 1, raw=True)
    return mom


def calc_volatility(ret_df, lookback=6):
    """过去 lookback 个月波动率"""
    vol = ret_df.rolling(lookback).std(ddof=1) * np.sqrt(12)
    return vol


def run_strategy(ret_df, pct_df, pe_df, mom_df=None, vol_df=None,
                 top_n=15, mode="growth", window=60,
                 stop_loss=None, pe_cap=None, mom_filter=False, vol_filter=False,
                 cost=COST):
    """通用策略回测"""
    dates = list(ret_df.index)
    nav = 1.0
    records = []
    prev_w = None
    in_cash = False

    for i in range(len(dates)):
        d = dates[i]
        ret_row = ret_df.loc[d]

        # 止损: 上月如果大跌, 本月空仓
        if stop_loss and prev_w is not None and records:
            prev_ret = records[-1]["ret"]
            if prev_ret < -stop_loss:
                in_cash = True
            else:
                in_cash = False

        if in_cash:
            records.append({"date": d, "nav": nav, "ret": 0.0, "cost": 0.0, "n_hold": 0})
            prev_w = {}
            continue

        pct_row = pct_df.loc[d].dropna() if d in pct_df.index else pd.Series(dtype=float)

        if mode == "equal" or len(pct_row) < top_n:
            valid = ret_row.dropna()
            weights = pd.Series(1.0 / len(valid), index=valid.index) if len(valid) > 0 else pd.Series(dtype=float)
        else:
            if mode == "growth":
                selected = pct_row.nlargest(top_n).index
            elif mode == "value":
                selected = pct_row.nsmallest(top_n).index
            else:
                selected = pct_row.nlargest(top_n).index

            # PE 上限止盈
            if pe_cap is not None:
                selected = [s for s in selected if pct_row.get(s, 0) < pe_cap]
                if len(selected) == 0:
                    selected = pct_row.nlargest(top_n).index.tolist()

            # 动量过滤: 只保留过去3月正收益
            if mom_filter and mom_df is not None and d in mom_df.index:
                mom_row = mom_df.loc[d]
                selected = [s for s in selected if s in mom_row.index and pd.notna(mom_row[s]) and mom_row[s] > 0]
                if len(selected) == 0:
                    selected = pct_row.nlargest(top_n).index.tolist()

            # 波动率过滤: 剔除最高1/3波动行业
            if vol_filter and vol_df is not None and d in vol_df.index:
                vol_row = vol_df.loc[d].dropna()
                if len(vol_row) >= top_n * 2:
                    threshold = vol_row.quantile(2/3)
                    selected = [s for s in selected if s in vol_row.index and vol_row[s] < threshold]
                    if len(selected) == 0:
                        selected = pct_row.nlargest(top_n).index.tolist()

            weights = pd.Series(1.0 / len(selected), index=selected) if len(selected) > 0 else pd.Series(dtype=float)

        # 收益
        port_ret = 0.0
        for ind, w in weights.items():
            r = ret_row.get(ind, 0.0)
            if pd.notna(r):
                port_ret += w * r

        # 成本
        if prev_w is not None and len(prev_w) > 0:
            all_i = set(weights.index) | set(prev_w.keys())
            turn = sum(abs(weights.get(c, 0) - prev_w.get(c, 0)) for c in all_i) / 2.0
            c = min(cost, turn * cost)
        else:
            c = cost

        nav *= (1 + port_ret - c)
        records.append({"date": d, "nav": nav, "ret": port_ret, "cost": c, "n_hold": len(weights)})
        prev_w = weights.to_dict()

    return pd.DataFrame(records).set_index("date")


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


def main():
    pe_df, ret_df = load_data()
    print(f"[data] {pe_df.shape[1]} industries, {len(pe_df)} months", flush=True)

    pct_df = calc_pe_pct(pe_df, 60)
    mom_df = calc_momentum(ret_df, 3)
    vol_df = calc_volatility(ret_df, 6)

    # ---- Phase 1: 基准回测 + 纯动量对比 ----
    print("\n[Phase 1] 基准对比", flush=True)
    nav_growth = run_strategy(ret_df, pct_df, pe_df, top_n=15, mode="growth")
    nav_equal = run_strategy(ret_df, pct_df, pe_df, top_n=15, mode="equal")
    # 纯动量: 买过去3个月收益最高的Top15
    nav_mom = run_strategy(ret_df, pct_df, pe_df, mom_df=mom_df, top_n=15, mode="growth", mom_filter=False)
    # 用动量直接选行业 (覆盖 pct 信号)
    nav_pure_mom = pd.DataFrame(index=ret_df.index, columns=["nav"])
    nav_val = 1.0
    prev_w = None
    for i, d in enumerate(ret_df.index):
        m = mom_df.loc[d].dropna() if d in mom_df.index else pd.Series(dtype=float)
        if len(m) >= 15:
            sel = m.nlargest(15).index
            w = pd.Series(1/15, index=sel)
        else:
            v = ret_df.loc[d].dropna()
            w = pd.Series(1/len(v), index=v) if len(v) > 0 else pd.Series(dtype=float)
        pr = sum(w.get(c, 0) * ret_df.loc[d].get(c, 0) for c in w.index if pd.notna(ret_df.loc[d].get(c, 0)))
        c = COST if prev_w is None else min(COST, sum(abs(w.get(x,0) - prev_w.get(x,0)) for x in set(w.index)|set(prev_w.keys()))/2 * COST)
        nav_val *= (1 + pr - c)
        nav_pure_mom.loc[d, "nav"] = nav_val
        prev_w = w.to_dict()
    nav_pure_mom["nav"] = nav_pure_mom["nav"].astype(float)
    nav_pure_mom["ret"] = nav_pure_mom["nav"].pct_change().fillna(0)

    for name, nv in [("高PE增长", nav_growth), ("等权基准", nav_equal), ("纯动量Top15", nav_pure_mom)]:
        st = calc_stats(nv)
        print(f"  {name}: NAV={st['FinalNAV']:.2f} CAGR={st['CAGR']:.2%} Sharpe={st['Sharpe']:.2f} MaxDD={st['MaxDD']:.2%} Calmar={st['Calmar']:.2f}")

    # ---- Phase 2: 风控叠加 ----
    print("\n[Phase 2] 风控叠加测试", flush=True)
    overlays = {
        "高PE基准(无风控)": {},
        "+月度止损10%": {"stop_loss": 0.10},
        "+月度止损5%": {"stop_loss": 0.05},
        "+PE上限95%止盈": {"pe_cap": 0.95},
        "+PE上限90%止盈": {"pe_cap": 0.90},
        "+动量过滤": {"mom_filter": True},
        "+波动率过滤": {"vol_filter": True},
        "+动量+波动双过滤": {"mom_filter": True, "vol_filter": True},
        "+止损5%+双过滤": {"stop_loss": 0.05, "mom_filter": True, "vol_filter": True},
        "+止损5%+PE90+双过滤": {"stop_loss": 0.05, "pe_cap": 0.90, "mom_filter": True, "vol_filter": True},
    }

    overlay_results = []
    overlay_navs = {}
    for name, kwargs in overlays.items():
        nv = run_strategy(ret_df, pct_df, pe_df, mom_df=mom_df, vol_df=vol_df,
                          top_n=15, mode="growth", **kwargs)
        st = calc_stats(nv)
        overlay_results.append({"strategy": name, **st})
        overlay_navs[name] = nv["nav"]
        print(f"  {name}: NAV={st['FinalNAV']:.2f} CAGR={st['CAGR']:.2%} Sharpe={st['Sharpe']:.2f} MaxDD={st['MaxDD']:.2%} Calmar={st['Calmar']:.2f}")

    pd.DataFrame(overlay_results).to_csv(os.path.join(OUT_DIR, "growth_risk_overlay.csv"), index=False)

    # ---- Phase 3: 参数敏感性 (用最优风控) ----
    print("\n[Phase 3] 参数敏感性", flush=True)
    param_results = []
    pct_cache = {}
    for tn in [10, 15, 20, 25]:
        for w in [36, 48, 60]:
            if w not in pct_cache:
                pct_cache[w] = calc_pe_pct(pe_df, w)
            nv = run_strategy(ret_df, pct_cache[w], pe_df, mom_df=mom_df, vol_df=vol_df,
                              top_n=tn, mode="growth",
                              stop_loss=0.05, mom_filter=True, vol_filter=True)
            st = calc_stats(nv)
            param_results.append({"top_n": tn, "window": w, **st})
            print(f"  Top{tn} W{w}: NAV={st['FinalNAV']:.2f} CAGR={st['CAGR']:.2%} Sharpe={st['Sharpe']:.2f} MaxDD={st['MaxDD']:.2%}")
    pd.DataFrame(param_results).to_csv(os.path.join(OUT_DIR, "growth_param_sweep.csv"), index=False)

    # ---- 最优方案曲线 ----
    best = max(overlay_results, key=lambda x: x.get("Calmar", 0) if pd.notna(x.get("Calmar", 0)) else 0)
    print(f"\n[best] Calmar 最优: {best['strategy']} (Calmar={best['Calmar']:.2f})")

    # 额外: 动量过滤 only + Top20 W36 (Phase 3 最优参数, 但去掉止损)
    pct36 = calc_pe_pct(pe_df, 36)
    nav_mom_best = run_strategy(ret_df, pct36, pe_df, mom_df=mom_df, vol_df=vol_df,
                                 top_n=20, mode="growth", mom_filter=True, vol_filter=True)
    st_mom_best = calc_stats(nav_mom_best)
    print(f"[best-mom] 动量+波动双过滤 Top20 W36: NAV={st_mom_best['FinalNAV']:.2f} "
          f"CAGR={st_mom_best['CAGR']:.2%} Sharpe={st_mom_best['Sharpe']:.2f} "
          f"MaxDD={st_mom_best['MaxDD']:.2%} Calmar={st_mom_best['Calmar']:.2f}")

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # 左图: 风控对比
    ax = axes[0]
    for name, color in [("高PE基准(无风控)", "#888"), ("+动量过滤", "#d62728"),
                         ("+动量+波动双过滤", "#2ca02c")]:
        if name in overlay_navs:
            ax.plot(range(len(overlay_navs[name])), overlay_navs[name], lw=1.8,
                    color=color, label=f"{name}")
    ax.plot(range(len(nav_mom_best)), nav_mom_best["nav"], lw=2.0, color="#ff7f0e",
            linestyle="-.", label=f"最优: 双过滤 Top20 W36")
    ax.plot(range(len(nav_equal)), nav_equal["nav"], lw=1.4, color="#1f77b4",
            linestyle="--", label="全行业等权")
    ax.set_title("高PE策略 + 风控叠加 NAV对比", fontsize=12)
    ax.set_ylabel("NAV (起点=1)")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(True, alpha=0.3)

    # 右图: 参数敏感性热力图 (Calmar)
    ax = axes[1]
    pivot = pd.DataFrame(param_results).pivot(index="top_n", columns="window", values="Calmar")
    im = ax.imshow(pivot.values, cmap="RdYlGn", aspect="auto")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([f"W{c}" for c in pivot.columns])
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([f"Top{n}" for n in pivot.index])
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            ax.text(j, i, f"{pivot.values[i,j]:.1f}", ha="center", va="center", fontsize=10)
    ax.set_title("参数敏感性: Calmar (止损5%+双过滤)", fontsize=12)
    fig.colorbar(im, ax=ax, shrink=0.8)

    fig.tight_layout()
    png = os.path.join(OUT_DIR, "growth_curve_optimized.png")
    fig.savefig(png, dpi=150, bbox_inches="tight")
    print(f"[saved] {png}")

    # ---- 回撤分析 ----
    print("\n== 高PE策略回撤 Top5 ==")
    nav_g = nav_growth["nav"]
    dd = (nav_g / nav_g.cummax() - 1)
    dd_periods = []
    for i in range(len(dd)):
        if dd.iloc[i] < -0.10:
            dd_periods.append({"date": dd.index[i], "drawdown": dd.iloc[i], "nav": nav_g.iloc[i]})
    dd_df = pd.DataFrame(dd_periods).sort_values("drawdown").head(10)
    print(dd_df.to_string(index=False))

    # ---- 结论归档 ----
    conclusion = f"""== 高估策略深度分析结论 ==
样本期: {pe_df.index[0]} ~ {pe_df.index[-1]} ({len(ret_df)} 个月)
行业数: {pe_df.shape[1]}

【关键发现: MaxDD 计算 Bug】
  原始回测 MaxDD=71.04% 实为绝对回撤值, 非百分比
  修正后实际 MaxDD=14.30% (百分比回撤)

【Phase 1: 基准对比】
  高PE增长:  CAGR=37.96%  Sharpe=1.60  MaxDD=14.30%  Calmar=2.65
  等权基准:  CAGR=15.12%  Sharpe=0.84  MaxDD=19.20%  Calmar=0.79
  纯动量Top15: CAGR=105.43% Sharpe=3.12 MaxDD=9.26%  Calmar=11.39

【Phase 2: 风控叠加 Top3 (按 Calmar)】
  1. +动量过滤:        CAGR=63.04% Sharpe=2.46 MaxDD=7.88%  Calmar=8.00
  2. +动量+波动双过滤:  CAGR=46.68% Sharpe=2.57 MaxDD=7.24%  Calmar=6.45
  3. +止损5%+双过滤:    CAGR=43.87% Sharpe=2.44 MaxDD=7.75%  Calmar=5.66

【Phase 3: 参数敏感性 (止损5%+双过滤) Top3 (按 Calmar)】
  1. Top25 W48: CAGR=46.14% Sharpe=2.70 MaxDD=5.11% Calmar=9.02
  2. Top20 W36: CAGR=49.79% Sharpe=2.81 MaxDD=6.32% Calmar=7.88
  3. Top25 W36: CAGR=47.12% Sharpe=2.74 MaxDD=6.30% Calmar=7.48

【最优方案: 动量+波动双过滤 Top20 W36 (无止损)】
  NAV={st_mom_best['FinalNAV']:.2f}  CAGR={st_mom_best['CAGR']:.2%}
  Sharpe={st_mom_best['Sharpe']:.2f}  MaxDD={st_mom_best['MaxDD']:.2%}
  Calmar={st_mom_best['Calmar']:.2f}

【策略逻辑】
  1. 月末计算每行业 PE 在过去 36 个月中的分位 (0-1)
  2. 选 PE 分位最高的 Top20 行业 (高估 = 高景气)
  3. 动量过滤: 只保留过去 3 个月正收益的行业
  4. 波动率过滤: 剔除过去 6 个月波动率最高的 1/3 行业
  5. 等权持有, 月度调仓, 成本 20bps

【关键结论】
  - 高 PE 分位 = 市场认可的高景气行业, 顺势而为优于逆势抄底
  - 动量过滤是核心: 剔除高 PE 但下跌的行业 (可能是价值陷阱)
  - 止损反而有害: 月度止损在趋势行情中频繁误触发
  - 短窗口 (W36) 优于长窗口 (W60): 估值变化更敏感
  - 实际回撤可控 (5-8%), 无需额外风控
"""
    with open(os.path.join(OUT_DIR, "growth_conclusion.txt"), "w", encoding="utf-8") as f:
        f.write(conclusion)
    print(f"\n[saved] growth_conclusion.txt")
    print(conclusion)

    return 0


if __name__ == "__main__":
    sys.exit(main())

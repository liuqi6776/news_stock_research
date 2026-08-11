# -*- coding: utf-8 -*-
"""前视偏差审查 + 修正回测

发现问题:
  1. [严重] run_strategy 用 d 月末信号配 d 月收益 = 前视偏差
     正确: d 月末信号 → 持有到 d+1 月末 → 用 d+1 月收益
  2. [中] COST=20bps 只含手续费, 无滑点
     修正: 加滑点 10bps (行业组合多股票)
  3. [低] 回测期 2020-2026 仅 6.5 年, 只一个半周期

本脚本:
  - 修正前视偏差: 信号在 d 月末生成, 收益用 d→d+1
  - 加滑点: 总成本 30bps (20手续费 + 10滑点)
  - 对比修正前后
  - 样本外验证: 2020-2022 训练, 2023-2026 测试
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

COST_FEE = 20 / 10000.0    # 手续费 20bps
COST_SLIP = 10 / 10000.0   # 滑点 10bps
COST_TOTAL = COST_FEE + COST_SLIP  # 总成本 30bps


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


def calc_pe_pct(pe_df, window=36):
    """滚动窗口计算 PE 分位 (只用 <= 当期数据, 无前视)"""
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
    """过去 lookback 个月累计收益 (rolling, 含当期; 月末时当期已知)"""
    return ret_df.rolling(lookback).apply(lambda x: (1 + x).prod() - 1, raw=True)


def calc_volatility(ret_df, lookback=6):
    """过去 lookback 个月波动率"""
    return ret_df.rolling(lookback).std(ddof=1) * np.sqrt(12)


def run_strategy_corrected(ret_df, pct_df, pe_df, mom_df=None, vol_df=None,
                            top_n=20, mode="growth",
                            stop_loss=None, pe_cap=None,
                            mom_filter=False, vol_filter=False,
                            cost=COST_TOTAL):
    """修正前视偏差的回测

    关键修正:
      信号在 d 月末生成 → 持有到 d+1 月末 → 用 ret_df.loc[d+1] 计算收益
    """
    dates = list(ret_df.index)
    nav = 1.0
    records = []
    prev_w = None

    for i in range(len(dates) - 1):  # 最后一个月无法持有到下月
        d_signal = dates[i]        # 信号日 (月末)
        d_hold = dates[i + 1]      # 持有期收益日 (下个月末)
        ret_row = ret_df.loc[d_hold]  # d→d+1 的收益 (无前视)

        # 信号: 用 d_signal 月末的 PE 分位
        pct_row = pct_df.loc[d_signal].dropna() if d_signal in pct_df.index else pd.Series(dtype=float)

        if mode == "equal" or len(pct_row) < top_n:
            valid = ret_row.dropna()
            weights = pd.Series(1.0 / len(valid), index=valid.index) if len(valid) > 0 else pd.Series(dtype=float)
        else:
            if mode == "growth":
                selected = pct_row.nlargest(top_n).index.tolist()
            elif mode == "value":
                selected = pct_row.nsmallest(top_n).index.tolist()
            else:
                selected = pct_row.nlargest(top_n).index.tolist()

            # PE 上限止盈
            if pe_cap is not None:
                filtered = [s for s in selected if pct_row.get(s, 0) < pe_cap]
                if len(filtered) >= 5:
                    selected = filtered

            # 动量过滤: 信号日之前的动量 (无前视, mom_df.loc[d_signal] 含 d_signal 月)
            if mom_filter and mom_df is not None and d_signal in mom_df.index:
                mom_row = mom_df.loc[d_signal]
                filtered = [s for s in selected
                           if s in mom_row.index and pd.notna(mom_row[s]) and mom_row[s] > 0]
                if len(filtered) >= 5:
                    selected = filtered

            # 波动率过滤: 信号日之前的波动率
            if vol_filter and vol_df is not None and d_signal in vol_df.index:
                vol_row = vol_df.loc[d_signal].dropna()
                if len(vol_row) >= top_n * 2:
                    threshold = vol_row.quantile(2 / 3)
                    filtered = [s for s in selected
                               if s in vol_row.index and vol_row[s] < threshold]
                    if len(filtered) >= 5:
                        selected = filtered

            weights = pd.Series(1.0 / len(selected), index=selected) if len(selected) > 0 else pd.Series(dtype=float)

        # 止损: 上月组合收益 < -stop_loss → 本月空仓
        if stop_loss and records:
            if records[-1]["ret"] < -stop_loss:
                weights = pd.Series(dtype=float)

        # 持有期收益
        if len(weights) == 0:
            port_ret = 0.0
        else:
            port_ret = 0.0
            for ind, w in weights.items():
                r = ret_row.get(ind, 0.0)
                if pd.notna(r):
                    port_ret += w * r

        # 成本 (换手 * 总成本, 含滑点)
        if prev_w is not None and len(prev_w) > 0:
            all_i = set(weights.index) | set(prev_w.keys())
            turn = sum(abs(weights.get(c, 0) - prev_w.get(c, 0)) for c in all_i) / 2.0
            c = turn * cost
        elif len(weights) > 0:
            c = cost  # 首期建仓
        else:
            c = 0.0

        nav *= (1 + port_ret - c)
        records.append({"date": d_hold, "nav": nav, "ret": port_ret, "cost": c,
                        "n_hold": len(weights), "signal_date": d_signal})
        prev_w = weights.to_dict()

    return pd.DataFrame(records).set_index("date")


def run_strategy_lookahead(ret_df, pct_df, pe_df, mom_df=None, vol_df=None,
                            top_n=20, mode="growth",
                            mom_filter=False, vol_filter=False,
                            cost=20/10000):
    """原始有前视版本 (d月末信号 + d月收益 = 前视偏差)"""
    dates = list(ret_df.index)
    nav = 1.0
    records = []
    prev_w = None

    for i in range(len(dates)):
        d = dates[i]
        ret_row = ret_df.loc[d]  # d月收益 (前视!)
        pct_row = pct_df.loc[d].dropna() if d in pct_df.index else pd.Series(dtype=float)

        if mode == "equal" or len(pct_row) < top_n:
            valid = ret_row.dropna()
            weights = pd.Series(1.0 / len(valid), index=valid.index) if len(valid) > 0 else pd.Series(dtype=float)
        else:
            if mode == "growth":
                selected = pct_row.nlargest(top_n).index.tolist()
            else:
                selected = pct_row.nsmallest(top_n).index.tolist()

            if mom_filter and mom_df is not None and d in mom_df.index:
                mom_row = mom_df.loc[d]
                filtered = [s for s in selected
                           if s in mom_row.index and pd.notna(mom_row[s]) and mom_row[s] > 0]
                if len(filtered) >= 5:
                    selected = filtered

            if vol_filter and vol_df is not None and d in vol_df.index:
                vol_row = vol_df.loc[d].dropna()
                if len(vol_row) >= top_n * 2:
                    threshold = vol_row.quantile(2 / 3)
                    filtered = [s for s in selected
                               if s in vol_row.index and vol_row[s] < threshold]
                    if len(filtered) >= 5:
                        selected = filtered

            weights = pd.Series(1.0 / len(selected), index=selected) if len(selected) > 0 else pd.Series(dtype=float)

        port_ret = 0.0
        for ind, w in weights.items():
            r = ret_row.get(ind, 0.0)
            if pd.notna(r):
                port_ret += w * r

        if prev_w is not None and len(prev_w) > 0:
            all_i = set(weights.index) | set(prev_w.keys())
            turn = sum(abs(weights.get(c, 0) - prev_w.get(c, 0)) for c in all_i) / 2.0
            c = turn * cost
        elif len(weights) > 0:
            c = cost
        else:
            c = 0.0

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
    print(f"[data] {pe_df.shape[1]} industries, {len(ret_df)} months")
    print(f"[data] 期间: {ret_df.index[0]} ~ {ret_df.index[-1]}")

    pct_df = calc_pe_pct(pe_df, 36)
    mom_df = calc_momentum(ret_df, 3)
    vol_df = calc_volatility(ret_df, 6)

    # 找样本外分割点 (2023年初)
    split_date = "20230101"
    split_idx = None
    for i, d in enumerate(ret_df.index):
        if str(d) >= split_date:
            split_idx = i
            break
    if split_idx is None:
        split_idx = len(ret_df) // 2
    print(f"[split] 训练期: {ret_df.index[0]} ~ {ret_df.index[split_idx-1]} ({split_idx}月)")
    print(f"[split] 测试期: {ret_df.index[split_idx]} ~ {ret_df.index[-1]} ({len(ret_df)-split_idx}月)")

    print("\n" + "=" * 70)
    print("对比1: 原始(有前视) vs 修正(无前视) + 滑点")
    print("=" * 70)

    configs = [
        ("等权基准(无前视)", "corrected", {"mode": "equal", "cost": COST_TOTAL}),
        ("高PE(有前视!20bps)", "lookahead", {"mode": "growth", "mom_filter": True, "vol_filter": True, "cost": 20/10000}),
        ("高PE(无前视+30bps)", "corrected", {"mode": "growth", "mom_filter": True, "vol_filter": True, "cost": COST_TOTAL}),
        ("高PE(无前视+20bps)", "corrected", {"mode": "growth", "mom_filter": True, "vol_filter": True, "cost": COST_FEE}),
        ("低估(无前视+30bps)", "corrected", {"mode": "value", "mom_filter": True, "vol_filter": True, "cost": COST_TOTAL}),
    ]

    results = {}
    for name, func_type, kwargs in configs:
        fn = run_strategy_lookahead if func_type == "lookahead" else run_strategy_corrected
        nv = fn(ret_df, pct_df, pe_df, mom_df=mom_df, vol_df=vol_df, top_n=20, **kwargs)
        st = calc_stats(nv)
        results[name] = (nv, st)
        print(f"\n  [{name}] 全期:")
        print(f"    NAV={st['FinalNAV']:.2f}  CAGR={st['CAGR']:.2%}  Sharpe={st['Sharpe']:.2f}  "
              f"MaxDD={st['MaxDD']:.2%}  Calmar={st['Calmar']:.2f}  WinRate={st['WinRate']:.1%}")

        # 样本外
        nv_test = nv[nv.index >= ret_df.index[split_idx]]
        if len(nv_test) > 12:
            st_test = calc_stats(nv_test)
            print(f"    样本外(2023+): NAV={st_test['FinalNAV']:.2f}  CAGR={st_test['CAGR']:.2%}  "
                  f"Sharpe={st_test['Sharpe']:.2f}  MaxDD={st_test['MaxDD']:.2%}")

    print("\n" + "=" * 70)
    print("对比2: 不同成本下的最优方案 (修正无前视)")
    print("=" * 70)

    for cost_bps in [20, 30, 50, 80]:
        nv = run_strategy_corrected(ret_df, pct_df, pe_df, mom_df=mom_df, vol_df=vol_df,
                                     top_n=20, mode="growth",
                                     mom_filter=True, vol_filter=True,
                                     cost=cost_bps / 10000)
        st = calc_stats(nv)
        # 平均换手
        avg_turn = nv["cost"].mean() / (cost_bps / 10000) if cost_bps > 0 else 0
        print(f"  成本{cost_bps}bps: NAV={st['FinalNAV']:.2f}  CAGR={st['CAGR']:.2%}  "
              f"Sharpe={st['Sharpe']:.2f}  MaxDD={st['MaxDD']:.2%}  "
              f"Calmar={st['Calmar']:.2f}  月均换手={avg_turn:.1%}")

    # ---- 画图 ----
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # 左图: 前视 vs 修正
    ax = axes[0]
    for name, color in [("等权基准(无前视)", "#1f77b4"),
                         ("高PE(有前视!20bps)", "#d62728"),
                         ("高PE(无前视+30bps)", "#2ca02c")]:
        if name in results:
            nv = results[name][0]
            ax.plot(range(len(nv)), nv["nav"], lw=1.8, color=color, label=name)
    ax.set_title("前视偏差修正对比", fontsize=12)
    ax.set_ylabel("NAV")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 右图: 样本内 vs 样本外 (修正后)
    ax = axes[1]
    nv_corr = results["高PE(无前视+30bps)"][0]
    train = nv_corr[nv_corr.index < ret_df.index[split_idx]]
    test = nv_corr[nv_corr.index >= ret_df.index[split_idx]]
    ax.plot(range(len(train)), train["nav"], lw=1.8, color="#2ca02c", label=f"训练期 (2020-2022)")
    ax.plot(range(len(train), len(train) + len(test)), test["nav"], lw=1.8, color="#ff7f0e",
            label=f"样本外 (2023-2026)")
    ax.axvline(x=len(train), color="gray", linestyle="--", alpha=0.5)
    ax.set_title("样本内 vs 样本外 (修正后)", fontsize=12)
    ax.set_ylabel("NAV")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    png = os.path.join(OUT_DIR, "lookahead_audit.png")
    fig.savefig(png, dpi=150, bbox_inches="tight")
    print(f"\n[saved] {png}")

    # ---- 结论 ----
    st_orig = results["高PE(有前视!20bps)"][1]
    st_corr = results["高PE(无前视+30bps)"][1]
    conclusion = f"""== 前视偏差审查报告 ==

【发现问题】
1. [严重] 前视偏差: run_strategy 用 d 月末信号配 d 月收益
   - d 月末 PE 分位在 d 月收盘后才可知
   - 但 ret_df.loc[d] 是 d 月初→月末收益, 此时已发生
   - 等同"用未来信息选股, 追溯过去收益"

2. [中] 滑点缺失: COST=20bps 只含手续费
   - 行业组合含多只股票, 实际滑点更大
   - 修正: 加 10bps 滑点 → 总成本 30bps

3. [低] 回测期偏短: 2020-02 ~ 2026-08 (6.5年)
   - 仅一个半牛熊周期
   - 未经历 2015 暴跌、2018 贸易战等极端行情

【修正后对比】
  原始(有前视, 20bps):
    CAGR={st_orig['CAGR']:.2%}  Sharpe={st_orig['Sharpe']:.2f}  MaxDD={st_orig['MaxDD']:.2%}  Calmar={st_orig['Calmar']:.2f}

  修正(无前视, 30bps):
    CAGR={st_corr['CAGR']:.2%}  Sharpe={st_corr['Sharpe']:.2f}  MaxDD={st_corr['MaxDD']:.2%}  Calmar={st_corr['Calmar']:.2f}

  收益降幅: {(st_corr['CAGR'] - st_orig['CAGR']):.2%}
"""
    with open(os.path.join(OUT_DIR, "lookahead_audit.txt"), "w", encoding="utf-8") as f:
        f.write(conclusion)
    print(f"\n[saved] lookahead_audit.txt")
    print(conclusion)

    return 0


if __name__ == "__main__":
    sys.exit(main())

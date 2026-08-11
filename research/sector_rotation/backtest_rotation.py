# -*- coding: utf-8 -*-
"""行业估值分位定投轮动回测

策略逻辑:
  每月末:
    1. 计算每个行业 PE-TTM 的历史分位 (滚动 60 个月 = 5 年)
    2. 信号:
       - 分位 < 20%  → 深度低估, 买入 (权重 ×2)
       - 分位 20-40% → 低估, 正常持有
       - 分位 40-60% → 中性, 持有
       - 分位 60-80% → 高估, 减仓 50%
       - 分位 > 80%  → 严重高估, 清仓
    3. 权重归一化 → Top-N 低估行业等权
    4. 月度调仓, 成本 20bps 双边

对比基准:
  - 全行业等权 (buy-and-hold)
  - 000852 中证1000 指数
  - 反向策略 (买高估行业, 验证信号有效性)

输出:
  results/rotation_nav.csv       月度 NAV (策略 / 全行业等权 / 反向)
  results/rotation_signals.csv   每月选中的行业 + PE 分位
  results/rotation_stats.txt     统计摘要
  results/rotation_curve.png     NAV 曲线
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

ROLL_WINDOW = 60  # 5 年 = 60 个月
COST = 20 / 10000.0  # 20bps 双边


def load_data():
    pe = pd.read_csv(os.path.join(OUT_DIR, "industry_pe.csv"), index_col=0)
    ret = pd.read_csv(os.path.join(OUT_DIR, "industry_ret.csv"), index_col=0)
    # 对齐列和行
    common_inds = sorted(set(pe.index) & set(ret.index))
    pe = pe.loc[common_inds].sort_index()
    ret = ret.loc[common_inds].sort_index()
    common_cols = sorted(set(pe.columns) & set(ret.columns))
    pe = pe[common_cols]
    ret = ret[common_cols]
    return pe, ret


def calc_pe_percentile(pe_df):
    """滚动 ROLL_WINDOW 个月计算 PE 分位 (0-1)

    对每个行业: 当期 PE 在过去 ROLL_WINDOW 个月中的百分位
    缺失值 (NaN PE) 标记为 0.5 (中性)
    """
    pct_df = pd.DataFrame(index=pe_df.index, columns=pe_df.columns, dtype=float)
    for i in range(len(pe_df)):
        start = max(0, i - ROLL_WINDOW)
        window = pe_df.iloc[start: i + 1]
        for col in pe_df.columns:
            vals = window[col].dropna()
            if len(vals) < 12:  # 至少 1 年数据
                pct_df.iloc[i, pct_df.columns.get_loc(col)] = np.nan
                continue
            current = pe_df.iloc[i][col]
            if pd.isna(current):
                pct_df.iloc[i, pct_df.columns.get_loc(col)] = np.nan
                continue
            pct = (vals <= current).sum() / len(vals)
            pct_df.iloc[i, pct_df.columns.get_loc(col)] = pct
    return pct_df


def run_backtest(pe_df, ret_df, pct_df, top_n=15, mode="value"):
    """回测估值轮动策略

    mode:
      'value'    - 买低 PE 分位 (低估)
      'growth'   - 买高 PE 分位 (高估, 反向验证)
      'equal'    - 全行业等权 (基准)
    """
    dates = list(pct_df.index)
    nav = 1.0
    nav_records = []
    signal_records = []
    prev_weights = None

    for i in range(len(dates)):
        d = dates[i]
        ret_row = ret_df.loc[d]

        if mode == "equal":
            # 全行业等权基准
            weights = pd.Series(1.0 / len(ret_row.dropna()), index=ret_row.dropna().index)
        else:
            pct_row = pct_df.loc[d].dropna()
            if len(pct_row) < top_n:
                # 数据不足, 等权持有
                weights = pd.Series(1.0 / len(ret_row.dropna()), index=ret_row.dropna().index)
            else:
                if mode == "value":
                    # 买低分位 (低估)
                    selected = pct_row.nsmallest(top_n).index
                else:
                    # 买高分位 (高估, 反向)
                    selected = pct_row.nlargest(top_n).index
                weights = pd.Series(1.0 / top_n, index=selected)

                # 记录信号
                for ind in selected:
                    signal_records.append({
                        "date": d,
                        "industry": ind,
                        "pe": pe_df.loc[d, ind] if ind in pe_df.columns else np.nan,
                        "pe_pct": pct_row[ind],
                        "ret": ret_row.get(ind, np.nan),
                        "weight": 1.0 / top_n,
                    })

        # 当月收益
        port_ret = 0.0
        for ind, w in weights.items():
            r = ret_row.get(ind, 0.0)
            if pd.notna(r):
                port_ret += w * r

        # 换手成本
        if prev_weights is not None:
            all_inds = set(weights.index) | set(prev_weights.keys())
            turnover = sum(abs(weights.get(c, 0) - prev_weights.get(c, 0)) for c in all_inds) / 2.0
            cost = min(COST, turnover * COST)
        else:
            cost = COST  # 首期满仓

        nav *= (1 + port_ret - cost)
        nav_records.append({"date": d, "nav": nav, "ret": port_ret, "cost": cost})
        prev_weights = weights.to_dict()

    return pd.DataFrame(nav_records).set_index("date"), pd.DataFrame(signal_records)


def calc_stats(nav_df, n_per_year=12):
    rets = nav_df["nav"].pct_change().dropna()
    years = len(rets) / n_per_year
    nav = nav_df["nav"]
    return {
        "CAGR": nav.iloc[-1] ** (1 / years) - 1 if years > 0 else np.nan,
        "Sharpe": rets.mean() / rets.std(ddof=1) * np.sqrt(n_per_year) if rets.std(ddof=1) > 0 else np.nan,
        "MaxDD": (nav.cummax() - nav).max(),
        "WinRate": (rets > 0).mean(),
        "Vol": rets.std(ddof=1) * np.sqrt(n_per_year),
        "FinalNAV": nav.iloc[-1],
    }


def main():
    pe_df, ret_df = load_data()
    print(f"[data] PE: {pe_df.shape} | RET: {ret_df.shape}", flush=True)

    # 过滤: 只保留有足够有效数据的行业
    valid_cols = pe_df.notna().sum() >= 36  # 至少 3 年有效 PE
    pe_df = pe_df.loc[:, valid_cols]
    ret_df = ret_df.loc[:, valid_cols]
    print(f"[data] 过滤后: {pe_df.shape[1]} 个行业", flush=True)

    # 计算 PE 分位
    pct_df = calc_pe_percentile(pe_df)
    print(f"[pct] PE 分位计算完成 {pct_df.shape}", flush=True)

    # 三组回测
    print("[bt] 运行低估轮动...", flush=True)
    val_nav, val_signals = run_backtest(pe_df, ret_df, pct_df, top_n=15, mode="value")
    print("[bt] 运行高估反向...", flush=True)
    gro_nav, _ = run_backtest(pe_df, ret_df, pct_df, top_n=15, mode="growth")
    print("[bt] 运行全行业等权...", flush=True)
    eq_nav, _ = run_backtest(pe_df, ret_df, pct_df, top_n=15, mode="equal")

    # 统计
    st_val = calc_stats(val_nav)
    st_gro = calc_stats(gro_nav)
    st_eq = calc_stats(eq_nav)

    # 输出 NAV
    nav_out = pd.DataFrame({
        "低估轮动Top15": val_nav["nav"],
        "高估反向Top15": gro_nav["nav"],
        "全行业等权": eq_nav["nav"],
    })
    nav_out.index.name = "date"
    nav_out.to_csv(os.path.join(OUT_DIR, "rotation_nav.csv"))
    val_signals.to_csv(os.path.join(OUT_DIR, "rotation_signals.csv"), index=False)

    # 统计文本
    with open(os.path.join(OUT_DIR, "rotation_stats.txt"), "w", encoding="utf-8") as f:
        f.write("== 行业估值分位轮动回测统计 ==\n")
        f.write(f"样本期: {pe_df.index[0]} ~ {pe_df.index[-1]} ({len(pe_df)} 个月)\n")
        f.write(f"行业数: {pe_df.shape[1]}\n")
        f.write(f"滚动窗口: {ROLL_WINDOW} 个月 (5 年)\n")
        f.write(f"成本: {COST*10000:.0f} bps 双边\n\n")
        for name, st in [("低估轮动 Top15", st_val), ("高估反向 Top15", st_gro), ("全行业等权", st_eq)]:
            f.write(f"--- {name} ---\n")
            f.write(f"  最终 NAV:    {st['FinalNAV']:.4f}\n")
            f.write(f"  年化 CAGR:   {st['CAGR']:.2%}\n")
            f.write(f"  Sharpe:      {st['Sharpe']:.2f}\n")
            f.write(f"  MaxDD:       {st['MaxDD']:.2%}\n")
            f.write(f"  月胜率:      {st['WinRate']:.1%}\n")
            f.write(f"  年化波动:    {st['Vol']:.2%}\n\n")
        # 超额
        f.write("--- 超额收益 (低估 vs 等权) ---\n")
        f.write(f"  累计超额: {(val_nav['nav'].iloc[-1] / eq_nav['nav'].iloc[-1] - 1):.2%}\n")
        f.write(f"  年化超额: {(val_nav['nav'].iloc[-1] / eq_nav['nav'].iloc[-1]) ** (12/len(val_nav)) - 1:.2%}\n\n")
        f.write("--- 超额收益 (低估 vs 高估) ---\n")
        f.write(f"  累计差: {(val_nav['nav'].iloc[-1] / gro_nav['nav'].iloc[-1] - 1):.2%}\n")

    print(f"\n[done] 统计已保存到 rotation_stats.txt")

    # 打印摘要
    print("\n== 回测结果摘要 ==")
    for name, st in [("低估轮动", st_val), ("高估反向", st_gro), ("全行业等权", st_eq)]:
        print(f"  {name}: NAV={st['FinalNAV']:.3f} | CAGR={st['CAGR']:.2%} | "
              f"Sharpe={st['Sharpe']:.2f} | MaxDD={st['MaxDD']:.2%} | Win={st['WinRate']:.1%}")

    # ---- NAV 曲线 ----
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(range(len(nav_out)), nav_out["低估轮动Top15"], lw=2.0, color="#d62728",
            label=f"低估轮动 Top15 (NAV={st_val['FinalNAV']:.2f})")
    ax.plot(range(len(nav_out)), nav_out["全行业等权"], lw=1.6, color="#1f77b4",
            label=f"全行业等权 (NAV={st_eq['FinalNAV']:.2f})")
    ax.plot(range(len(nav_out)), nav_out["高估反向Top15"], lw=1.4, color="#888",
            linestyle="--", label=f"高估反向 Top15 (NAV={st_gro['FinalNAV']:.2f})")
    ax.set_xticks(range(0, len(nav_out), 12))
    xlabels = [str(nav_out.index[i])[:6] for i in range(0, len(nav_out), 12)]
    ax.set_xticklabels(xlabels, rotation=60)
    ax.set_title("行业 PE 分位定投轮动 vs 基准（2020-02 ~ 2026-07, 月度, Top15, 20bps）", fontsize=13)
    ax.set_ylabel("NAV（起点=1）")
    ax.legend(loc="upper left", fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    png = os.path.join(OUT_DIR, "rotation_curve.png")
    fig.savefig(png, dpi=150, bbox_inches="tight")
    print(f"[saved] {png}")

    # ---- 最新一期信号 ----
    latest_signals = val_signals[val_signals["date"] == val_signals["date"].iloc[-1]].sort_values("pe_pct")
    print(f"\n== 最新一期 ({latest_signals['date'].iloc[0]}) 低估 Top15 ==")
    print(latest_signals[["industry", "pe", "pe_pct", "ret"]].to_string(index=False))

    return 0


if __name__ == "__main__":
    sys.exit(main())

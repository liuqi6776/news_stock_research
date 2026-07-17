"""
Backtest: A-share ETF portfolio with worst-case (peak-entry) stress test.
Data source: akshare Sina ETF historical prices.
"""

import os
import json
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import akshare as ak
from datetime import datetime

# --- Matplotlib Chinese font ---
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False

# --- Config ---
START_DATE = "20140101"
END_DATE = "20260621"
REBALANCE_FREQ = 63  # ~quarterly trading days
TRANSACTION_COST = 0.0005  # 0.05% per trade

# Symbol mapping: (sina_symbol, display_name)
ASSETS = {
    "国债ETF": "sh511010",
    "红利低波ETF": "sh512890",
    "沪深300ETF": "sh510300",
    "中证500ETF": "sh510500",
    "黄金ETF": "sh518880",
}

# Portfolio scenarios
SCENARIOS = {
    "保守型(目标回撤<8%)": {
        "国债ETF": 0.70,
        "红利低波ETF": 0.15,
        "沪深300ETF": 0.10,
        "黄金ETF": 0.05,
    },
    "平衡型": {
        "国债ETF": 0.50,
        "红利低波ETF": 0.25,
        "沪深300ETF": 0.15,
        "黄金ETF": 0.10,
    },
    "稳健偏股型": {
        "国债ETF": 0.40,
        "红利低波ETF": 0.25,
        "沪深300ETF": 0.20,
        "中证500ETF": 0.05,
        "黄金ETF": 0.10,
    },
    "超保守型(目标回撤<6%)": {
        "国债ETF": 0.80,
        "红利低波ETF": 0.10,
        "沪深300ETF": 0.05,
        "黄金ETF": 0.05,
    },
}

CACHE_DIR = os.path.join(os.path.dirname(__file__), "etf_data_cache")
os.makedirs(CACHE_DIR, exist_ok=True)

RESULTS_DIR = os.path.join(os.path.dirname(__file__))


def fetch_etf(symbol, start_date, end_date, max_retries=3):
    """Fetch ETF daily data from akshare Sina source with local cache and retries."""
    cache_file = os.path.join(CACHE_DIR, f"{symbol}_{start_date}_{end_date}.csv")
    if os.path.exists(cache_file):
        df = pd.read_csv(cache_file, parse_dates=["date"])
        return df

    for attempt in range(max_retries):
        try:
            df = ak.fund_etf_hist_sina(symbol=symbol)
            break
        except Exception as e:
            print(f"  Attempt {attempt+1}/{max_retries} failed for {symbol}: {e}")
            time.sleep(2 ** attempt)
            if attempt == max_retries - 1:
                raise

    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    df = df[(df["date"] >= pd.to_datetime(start_date)) & (df["date"] <= pd.to_datetime(end_date))]
    df.to_csv(cache_file, index=False)
    time.sleep(0.5)
    return df


def load_data():
    """Load close/high prices for all assets, align by common dates."""
    close_dict, high_dict = {}, {}
    for name, symbol in ASSETS.items():
        print(f"Fetching {symbol} {name}...")
        df = fetch_etf(symbol, START_DATE, END_DATE)
        close_dict[name] = df.set_index("date")["close"]
        high_dict[name] = df.set_index("date")["high"]
    close_df = pd.DataFrame(close_dict).dropna(how="any")
    high_df = pd.DataFrame(high_dict).dropna(how="any")
    return close_df, high_df


def calc_metrics(nav):
    """Calculate return metrics."""
    nav = nav.dropna()
    if len(nav) < 2:
        return {}
    total_return = nav.iloc[-1] / nav.iloc[0] - 1
    n_years = len(nav) / 252
    cagr = (nav.iloc[-1] / nav.iloc[0]) ** (1 / n_years) - 1 if n_years > 0 else 0
    running_max = nav.cummax()
    drawdown = (nav - running_max) / running_max
    max_dd = drawdown.min()
    max_dd_date = drawdown.idxmin()
    recovery_days = None
    post_dd = nav.loc[max_dd_date:]
    if len(post_dd) > 1:
        prev_max = running_max.loc[max_dd_date]
        recovered = post_dd[post_dd > prev_max]
        if not recovered.empty:
            recovery_days = (recovered.index[0] - max_dd_date).days
    daily_ret = nav.pct_change().dropna()
    sharpe = (daily_ret.mean() / daily_ret.std()) * np.sqrt(252) if daily_ret.std() > 0 else 0
    return {
        "total_return": total_return,
        "cagr": cagr,
        "max_drawdown": max_dd,
        "sharpe": sharpe,
        "recovery_days": recovery_days,
        "start": nav.index[0].strftime("%Y-%m-%d"),
        "end": nav.index[-1].strftime("%Y-%m-%d"),
    }


def portfolio_nav(close_df, weights, high_df=None, rebalance=False, peak_entry=False, cost=TRANSACTION_COST):
    """
    Compute portfolio NAV.
    If rebalance=False: buy-and-hold from first date.
    If rebalance=True, peak_entry=False: rebalance quarterly at close.
    If rebalance=True, peak_entry=True: rebalance quarterly at daily high (pessimistic).
    """
    asset_names = list(weights.keys())
    weights_arr = np.array([weights[a] for a in asset_names])
    sub_close = close_df[asset_names]
    sub_high = high_df[asset_names] if high_df is not None else sub_close

    if not rebalance:
        normalized = sub_close / sub_close.iloc[0]
        return (normalized * weights_arr).sum(axis=1)

    nav = pd.Series(index=sub_close.index, dtype=float)
    holdings = np.zeros(len(asset_names))

    rebalance_idx = list(range(0, len(sub_close), REBALANCE_FREQ))
    entry_source = sub_high if peak_entry else sub_close

    for i in range(len(sub_close)):
        if i == 0 or i in rebalance_idx:
            current_value = 1.0 if i == 0 else (holdings * sub_close.iloc[i].values).sum()
            entry_prices = entry_source.iloc[i].values
            cash_after_cost = current_value * (1 - cost)
            holdings = (cash_after_cost * weights_arr) / entry_prices
        nav.iloc[i] = (holdings * sub_close.iloc[i].values).sum()

    return nav


def backtest_all(close_df, high_df, scenario_name, weights):
    """Run all backtests for a given scenario."""
    results = {"scenario": scenario_name, "weights": weights}

    # 1. Buy & hold
    nav = portfolio_nav(close_df, weights)
    results["buy_hold"] = calc_metrics(nav)

    # 2. Lump sum at major peaks
    peak_dates = ["2021-02-10", "2021-09-14"]
    results["peak_entries"] = []
    for pd_str in peak_dates:
        pd_dt = pd.to_datetime(pd_str)
        if pd_dt < close_df.index.min() or pd_dt > close_df.index.max():
            continue
        idx = close_df.index.searchsorted(pd_dt)
        entry_date = close_df.index[idx]
        sub_close = close_df.loc[entry_date:]
        sub_high = high_df.loc[entry_date:]
        nav_peak = portfolio_nav(sub_close, weights, high_df=sub_high)
        m = calc_metrics(nav_peak)
        m["peak_date"] = entry_date.strftime("%Y-%m-%d")
        results["peak_entries"].append(m)

    # 3. Quarterly rebalancing
    nav_rebal = portfolio_nav(close_df, weights, high_df=high_df, rebalance=True, peak_entry=False)
    results["quarterly_rebal"] = calc_metrics(nav_rebal)

    # 4. Quarterly rebalancing at daily high
    nav_rebal_peak = portfolio_nav(close_df, weights, high_df=high_df, rebalance=True, peak_entry=True)
    results["quarterly_rebal_peak"] = calc_metrics(nav_rebal_peak)

    return results


def print_results(results):
    """Pretty print backtest results."""
    print(f"\n{'='*60}")
    print(f"组合: {results['scenario']}")
    print("权重:", {k: f"{v:.0%}" for k, v in results['weights'].items()})
    print(f"{'='*60}")

    bh = results["buy_hold"]
    print(f"1. 买入持有 (2019-01 至今):")
    print(f"   总收益: {bh['total_return']:.2%}, 年化: {bh['cagr']:.2%}, 最大回撤: {bh['max_drawdown']:.2%}, 夏普: {bh['sharpe']:.2f}")

    print(f"\n2. 最高点一次性买入:")
    for p in results["peak_entries"]:
        print(f"   {p['peak_date']} 买入 → 年化: {p['cagr']:.2%}, 最大回撤: {p['max_drawdown']:.2%}, 回本天数: {p['recovery_days']}")

    rb = results["quarterly_rebal"]
    rb_peak = results["quarterly_rebal_peak"]
    print(f"\n3. 季度再平衡 (收盘价买入):")
    print(f"   总收益: {rb['total_return']:.2%}, 年化: {rb['cagr']:.2%}, 最大回撤: {rb['max_drawdown']:.2%}, 夏普: {rb['sharpe']:.2f}")
    print(f"\n4. 季度再平衡 (每日高点买入/悲观假设):")
    print(f"   总收益: {rb_peak['total_return']:.2%}, 年化: {rb_peak['cagr']:.2%}, 最大回撤: {rb_peak['max_drawdown']:.2%}, 夏普: {rb_peak['sharpe']:.2f}")


def plot_results(all_results, close_df, high_df):
    """Generate comparison plots."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Plot 1: NAV curves for buy-and-hold across scenarios
    ax = axes[0, 0]
    for r in all_results:
        nav = portfolio_nav(close_df, r["weights"])
        ax.plot(nav / nav.iloc[0], label=r["scenario"])
    csi300 = close_df["沪深300ETF"] / close_df["沪深300ETF"].iloc[0]
    ax.plot(csi300, label="沪深300ETF", alpha=0.5, linestyle="--")
    ax.set_title("各组合买入持有净值对比")
    ax.legend()
    ax.grid(True)

    # Plot 2: Drawdown
    ax = axes[0, 1]
    for r in all_results:
        nav = portfolio_nav(close_df, r["weights"])
        dd = (nav - nav.cummax()) / nav.cummax()
        ax.plot(dd, label=r["scenario"])
    csi300_dd = (csi300 - csi300.cummax()) / csi300.cummax()
    ax.plot(csi300_dd, label="沪深300ETF", alpha=0.5, linestyle="--")
    ax.set_title("回撤对比")
    ax.legend()
    ax.grid(True)

    # Plot 3: Peak-entry recovery (conservative scenario)
    ax = axes[1, 0]
    conservative = all_results[0]
    for p in conservative["peak_entries"]:
        peak_date = pd.to_datetime(p["peak_date"])
        sub_close = close_df.loc[peak_date:]
        sub_high = high_df.loc[peak_date:]
        nav = portfolio_nav(sub_close, conservative["weights"], high_df=sub_high)
        ax.plot(nav / nav.iloc[0], label=f"{p['peak_date']} 买入")
    ax.axhline(1.0, color="black", linestyle="--", alpha=0.5)
    ax.set_title("保守组合：高点买入后恢复曲线")
    ax.legend()
    ax.grid(True)

    # Plot 4: Bar chart of CAGR and MaxDD by scenario/method
    ax = axes[1, 1]
    labels, cagr_vals, dd_vals = [], [], []
    for r in all_results:
        labels.append(r["scenario"].split("(")[0] + "\nB&H")
        cagr_vals.append(r["buy_hold"]["cagr"] * 100)
        dd_vals.append(r["buy_hold"]["max_drawdown"] * 100)
        labels.append(r["scenario"].split("(")[0] + "\n高点买入")
        if r["peak_entries"]:
            cagr_vals.append(r["peak_entries"][0]["cagr"] * 100)
            dd_vals.append(r["peak_entries"][0]["max_drawdown"] * 100)
        else:
            cagr_vals.append(0)
            dd_vals.append(0)

    x = np.arange(len(labels))
    width = 0.35
    ax.bar(x - width/2, cagr_vals, width, label="年化收益 %", color="steelblue")
    ax.bar(x + width/2, dd_vals, width, label="最大回撤 %", color="coral")
    ax.set_title("年化收益 vs 最大回撤")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.legend()
    ax.grid(True, axis="y")

    plt.tight_layout()
    plot_path = os.path.join(RESULTS_DIR, "backtest_peak_entry_results.png")
    plt.savefig(plot_path, dpi=150)
    print(f"\n图表已保存: {plot_path}")


def main():
    print("加载基金数据...")
    close_df, high_df = load_data()
    print(f"数据区间: {close_df.index[0].date()} ~ {close_df.index[-1].date()}")
    print(f"资产: {list(close_df.columns)}")

    all_results = []
    for name, weights in SCENARIOS.items():
        # Only include assets with non-zero weights that exist in data
        active_weights = {k: v for k, v in weights.items() if v > 0 and k in close_df.columns}
        total = sum(active_weights.values())
        active_weights = {k: v / total for k, v in active_weights.items()}
        results = backtest_all(close_df, high_df, name, active_weights)
        print_results(results)
        all_results.append(results)

    # Save JSON
    summary_path = os.path.join(RESULTS_DIR, "backtest_peak_entry_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n详细结果已保存: {summary_path}")

    # Plot
    plot_results(all_results, close_df, high_df)

    # Final recommendation
    print(f"\n{'='*60}")
    print("结论:")
    print(f"{'='*60}")
    for r in all_results:
        bh = r["buy_hold"]
        peak = r["peak_entries"][0] if r["peak_entries"] else {}
        print(f"- {r['scenario']}: 买入持有年化 {bh['cagr']:.2%} / 回撤 {bh['max_drawdown']:.2%}; "
              f"2021-02高点买入年化 {peak.get('cagr', 0):.2%} / 回撤 {peak.get('max_drawdown', 0):.2%}")


if __name__ == "__main__":
    main()

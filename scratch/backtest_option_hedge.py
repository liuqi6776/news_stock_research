"""
Backtest: ETF portfolio + protective put hedging using 300ETF options.
Uses option-research project data.
"""

import os
import json
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import akshare as ak
from datetime import datetime, timedelta

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False

START_DATE = "20220101"
END_DATE = "20260621"
INITIAL_CAPITAL = 1_000_000

# Base portfolio (target weights)
BASE_WEIGHTS = {
    "国债ETF": 0.50,
    "沪深300ETF": 0.30,
    "红利低波ETF": 0.10,
    "黄金ETF": 0.10,
}

# Assets
ETF_ASSETS = {
    "国债ETF": "sh511010",
    "沪深300ETF": "sh510300",
    "红利低波ETF": "sh512890",
    "黄金ETF": "sh518880",
}

# Option data paths
OPTION_DAILY_DIR = "option-research/data/daily"
OPTION_BASIC_FILE = "option-research/data/opt_basic.parquet"
CACHE_DIR = os.path.join(os.path.dirname(__file__), "etf_data_cache")
os.makedirs(CACHE_DIR, exist_ok=True)


def fetch_etf(symbol, start_date, end_date, max_retries=3):
    cache_file = os.path.join(CACHE_DIR, f"{symbol}_{start_date}_{end_date}.csv")
    if os.path.exists(cache_file):
        return pd.read_csv(cache_file, parse_dates=["date"])
    for attempt in range(max_retries):
        try:
            df = ak.fund_etf_hist_sina(symbol=symbol)
            break
        except Exception as e:
            time.sleep(2 ** attempt)
            if attempt == max_retries - 1:
                raise
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    df = df[(df["date"] >= pd.to_datetime(start_date)) & (df["date"] <= pd.to_datetime(end_date))]
    df.to_csv(cache_file, index=False)
    time.sleep(0.5)
    return df


def load_etf_prices():
    closes, highs = {}, {}
    for name, symbol in ETF_ASSETS.items():
        df = fetch_etf(symbol, START_DATE, END_DATE)
        closes[name] = df.set_index("date")["close"]
        highs[name] = df.set_index("date")["high"]
    return pd.DataFrame(closes).dropna(how="any"), pd.DataFrame(highs).dropna(how="any")


def load_option_basic():
    df = pd.read_parquet(OPTION_BASIC_FILE)
    df = df[df["opt_code"] == "OP510300.SH"].copy()  # 300ETF options only
    for col in ["maturity_date", "list_date", "delist_date", "last_edate", "last_ddate"]:
        df[col] = pd.to_datetime(df[col], format="%Y%m%d", errors="coerce")
    return df


def load_option_daily(date_str):
    """Load option daily data for a given date."""
    file_path = os.path.join(OPTION_DAILY_DIR, f"opt_daily_{date_str}.parquet")
    if not os.path.exists(file_path):
        return None
    return pd.read_parquet(file_path)


def find_put_contract(option_basic, trade_date, underlying_price, target_strike_ratio=0.95):
    """
    Find the best protective put contract:
    - Expires in ~30-60 days
    - Put option
    - Strike closest to target_strike_ratio * underlying_price
    """
    target_strike = underlying_price * target_strike_ratio

    # Find contracts listed and active on trade_date
    puts = option_basic[
        (option_basic["call_put"] == "P") &
        (option_basic["list_date"] <= trade_date) &
        (option_basic["last_ddate"] >= trade_date)
    ].copy()

    if puts.empty:
        return None

    # Days to maturity
    puts["dte"] = (puts["maturity_date"] - trade_date).dt.days
    # Filter reasonable DTE (20-60 days)
    puts = puts[(puts["dte"] >= 20) & (puts["dte"] <= 60)]

    if puts.empty:
        return None

    # Find strike closest to target
    puts["strike_diff"] = (puts["exercise_price"] - target_strike).abs()
    best = puts.nsmallest(1, "strike_diff").iloc[0]
    return {
        "ts_code": best["ts_code"],
        "strike": best["exercise_price"],
        "maturity_date": best["maturity_date"],
        "dte": best["dte"],
        "multiplier": best["opt_multiplier"],
    }


def get_option_price(ts_code, date_str):
    """Get option close price for a given ts_code and date."""
    df = load_option_daily(date_str)
    if df is None:
        return None
    row = df[df["ts_code"] == ts_code]
    if row.empty:
        return None
    return row.iloc[0]["close"]


def get_option_settle(ts_code, date_str):
    """Get option settle price for a given ts_code and date."""
    df = load_option_daily(date_str)
    if df is None:
        return None
    row = df[df["ts_code"] == ts_code]
    if row.empty:
        return None
    return row.iloc[0]["settle"]


def backtest_portfolio(price_df, hedge=False, hedge_ratio=0.5, strike_ratio=0.95, option_basic=None):
    """
    Backtest base portfolio with optional protective put hedge on 300ETF portion.
    """
    weights = np.array([BASE_WEIGHTS[a] for a in price_df.columns])

    # Equity exposure to hedge = weight of 沪深300ETF
    equity_weight = BASE_WEIGHTS["沪深300ETF"]

    nav = pd.Series(index=price_df.index, dtype=float)
    option_pnl = pd.Series(index=price_df.index, dtype=float)
    option_pnl.iloc[0] = 0.0

    current_put = None  # {ts_code, strike, maturity_date, contracts, entry_price}

    for i in range(len(price_df)):
        date = price_df.index[i]
        date_str = date.strftime("%Y%m%d")

        # Base portfolio return
        if i == 0:
            base_nav = 1.0
        else:
            base_ret = (price_df.iloc[i] / price_df.iloc[i-1] - 1).values
            base_nav = nav.iloc[i-1] * (1 + (base_ret * weights).sum())

        option_payoff = 0.0

        if hedge and option_basic is not None:
            # Check if we need to roll option
            need_new_put = (current_put is None) or (date >= current_put["maturity_date"])

            if need_new_put:
                # Close old put if exists (get settlement on maturity date)
                if current_put is not None and date == current_put["maturity_date"]:
                    settle_price = get_option_settle(current_put["ts_code"], date_str)
                    if settle_price is not None:
                        option_payoff += current_put["contracts"] * current_put["multiplier"] * (settle_price - current_put["entry_price"])

                # Open new put
                underlying_price = price_df.loc[date, "沪深300ETF"]
                put_info = find_put_contract(option_basic, date, underlying_price, strike_ratio)

                if put_info is not None:
                    put_price = get_option_price(put_info["ts_code"], date_str)
                    if put_price is not None:
                        # Contracts to buy
                        # Notional to hedge = hedge_ratio * equity_weight * portfolio_value
                        # Number of contracts = notional / (underlying_price * multiplier)
                        portfolio_value = base_nav * INITIAL_CAPITAL
                        notional_to_hedge = portfolio_value * equity_weight * hedge_ratio
                        contracts = int(round(notional_to_hedge / (underlying_price * put_info["multiplier"])))

                        if contracts > 0:
                            # Cost of buying puts
                            option_cost = contracts * put_info["multiplier"] * put_price
                            # Reduce base nav by option cost as percentage of portfolio
                            cost_pct = option_cost / portfolio_value
                            base_nav *= (1 - cost_pct)

                            current_put = {
                                "ts_code": put_info["ts_code"],
                                "strike": put_info["strike"],
                                "maturity_date": put_info["maturity_date"],
                                "multiplier": put_info["multiplier"],
                                "contracts": contracts,
                                "entry_price": put_price,
                            }

            # Mark-to-market option value change for non-expiration days
            if current_put is not None and date != current_put["maturity_date"]:
                current_price = get_option_price(current_put["ts_code"], date_str)
                if current_price is not None and i > 0:
                    prev_date = price_df.index[i-1]
                    prev_date_str = prev_date.strftime("%Y%m%d")
                    prev_price = get_option_price(current_put["ts_code"], prev_date_str)
                    if prev_price is not None:
                        option_payoff += current_put["contracts"] * current_put["multiplier"] * (current_price - prev_price)

        nav.iloc[i] = base_nav
        option_pnl.iloc[i] = option_payoff

    return nav, option_pnl


def calc_metrics(nav):
    nav = nav.dropna()
    if len(nav) < 2:
        return {}
    total_return = nav.iloc[-1] / nav.iloc[0] - 1
    n_years = len(nav) / 252
    cagr = (nav.iloc[-1] / nav.iloc[0]) ** (1 / n_years) - 1
    running_max = nav.cummax()
    max_dd = ((nav - running_max) / running_max).min()
    daily_ret = nav.pct_change().dropna()
    sharpe = (daily_ret.mean() / daily_ret.std()) * np.sqrt(252) if daily_ret.std() > 0 else 0
    return {
        "total_return": total_return,
        "cagr": cagr,
        "max_drawdown": max_dd,
        "sharpe": sharpe,
    }


def main():
    print("Loading ETF prices...")
    price_df, _ = load_etf_prices()
    print(f"ETF data range: {price_df.index[0].date()} ~ {price_df.index[-1].date()}")

    print("Loading option basic info...")
    option_basic = load_option_basic()
    print(f"300ETF put options available: {len(option_basic[option_basic['call_put']=='P'])}")

    # Backtest scenarios
    scenarios = [
        ("无对冲 (买入持有)", False, 0, 0),
        ("50% 保护性看跌对冲", True, 0.5, 0.95),
        ("100% 保护性看跌对冲", True, 1.0, 0.95),
        ("100% 保护性看跌对冲 (行权价90%)", True, 1.0, 0.90),
    ]

    results = []
    navs = {}

    for name, hedge, hr, sr in scenarios:
        print(f"\nRunning: {name}...")
        nav, opt_pnl = backtest_portfolio(price_df, hedge=hedge, hedge_ratio=hr, strike_ratio=sr, option_basic=option_basic)
        navs[name] = nav
        metrics = calc_metrics(nav)
        metrics["name"] = name
        results.append(metrics)
        print(f"  Total Return: {metrics['total_return']:.2%}")
        print(f"  CAGR: {metrics['cagr']:.2%}")
        print(f"  Max Drawdown: {metrics['max_drawdown']:.2%}")
        print(f"  Sharpe: {metrics['sharpe']:.2f}")

    # Plot
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    ax = axes[0]
    for name, nav in navs.items():
        ax.plot(nav / nav.iloc[0], label=name)
    ax.set_title("ETF组合净值对比 (期权对冲 vs 无对冲)")
    ax.legend()
    ax.grid(True)

    ax = axes[1]
    for name, nav in navs.items():
        dd = (nav - nav.cummax()) / nav.cummax()
        ax.plot(dd, label=name)
    ax.set_title("回撤对比")
    ax.legend()
    ax.grid(True)

    plt.tight_layout()
    plot_path = os.path.join(os.path.dirname(__file__), "backtest_option_hedge_results.png")
    plt.savefig(plot_path, dpi=150)
    print(f"\nPlot saved: {plot_path}")

    # Save results
    summary_path = os.path.join(os.path.dirname(__file__), "backtest_option_hedge_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"Summary saved: {summary_path}")


if __name__ == "__main__":
    main()

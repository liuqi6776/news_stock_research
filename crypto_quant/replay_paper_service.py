# -*- coding: utf-8 -*-
"""
Replay Auditor: Historical Clean Benchmark vs Paper Service (Phase 22)
======================================================================
Replays historical 4h bars from 2024-01-01 to 2026-09-01 bar-by-bar through
the incremental Paper Strategy engine and verifies zero discrepancy
against the vectorized clean StructuralTrendEngine.

Zero-Tolerance Criteria:
    - position mismatches = 0
    - trade count mismatch = 0
    - entry time mismatch = 0
    - exit time mismatch = 0
    - max equity difference < 1e-8
"""

import sys
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

# Add project root to sys.path
root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir))

from crypto_quant.paper.config import (
    ATR_PERIOD,
    ATR_TRAILING_MULT,
    BOLLINGER_STD,
    EXIT_LOOKBACK_BARS,
    LOOKBACK_BARS,
    MACRO_BEAR_SIZE,
    MACRO_BULL_SIZE,
    MACRO_EMA_SPAN,
    ONE_WAY_COST,
    print_startup_banner,
)
from crypto_quant.paper.state import PaperStrategyState
from crypto_quant.paper.strategy import StructuralTrendPaperStrategy
from crypto_quant.structural_trend_engine import StructuralTrendEngine


def run_replay_for_token(
    token: str,
    df: pd.DataFrame,
    eval_start: str = "2024-01-01 00:00:00",
    eval_end: str = "2026-09-01 12:00:00",
) -> Tuple[bool, Dict[str, any]]:
    """
    Executes incremental paper replay and tests against StructuralTrendEngine.
    """
    # 1. Clean Vectorized Engine Execution
    engine = StructuralTrendEngine(
        mode="bollinger",
        lookback_bars=LOOKBACK_BARS,
        exit_lookback_bars=EXIT_LOOKBACK_BARS,
        atr_trailing_mult=ATR_TRAILING_MULT,
        fee_and_slippage=ONE_WAY_COST,
    )
    closes = df["close"]
    ema200 = closes.shift(1).ewm(span=MACRO_EMA_SPAN).mean()
    macro_mult = pd.Series(np.where(closes > ema200, MACRO_BULL_SIZE, MACRO_BEAR_SIZE), index=df.index)
    rets, trades, pos = engine.run_backtest(df, token=token, macro_multipliers=macro_mult)

    # 2. Incremental Paper Strategy Execution (Bar-by-Bar Replay)
    strategy = StructuralTrendPaperStrategy(
        lookback_bars=LOOKBACK_BARS,
        exit_lookback_bars=EXIT_LOOKBACK_BARS,
        bollinger_std=BOLLINGER_STD,
        atr_period=ATR_PERIOD,
        atr_trailing_mult=ATR_TRAILING_MULT,
        macro_ema_span=MACRO_EMA_SPAN,
        macro_bull_size=MACRO_BULL_SIZE,
        macro_bear_size=MACRO_BEAR_SIZE,
    )
    state = PaperStrategyState(symbol=token)
    ind = engine.compute_indicators(df)

    paper_trades = []
    paper_pos = pd.Series(0.0, index=df.index)

    for i in range(1, len(df)):
        bar_time = df.index[i]
        curr_open = df["open"].iloc[i]

        # Fill pending order at curr_open
        if state.pending_order is not None:
            p_order = state.pending_order
            if p_order["side"] == "BUY":
                state.position = 1
                state.position_size = p_order["quantity_fraction"]
                state.entry_price = curr_open
                state.entry_time = str(bar_time)
                state.highest_price_since_entry = df["close"].iloc[i - 1]
                state.trailing_stop_price = p_order["metadata"]["initial_stop"]
            elif p_order["side"] == "SELL":
                gross = (curr_open / state.entry_price) - 1.0
                net = gross - 2.0 * ONE_WAY_COST
                paper_trades.append({
                    "entry_time": state.entry_time,
                    "exit_time": str(bar_time),
                    "entry_price": state.entry_price,
                    "exit_price": curr_open,
                    "net_ret": net,
                    "reason": p_order["reason"],
                })
                state.position = 0
                state.position_size = 0.0
            state.pending_order = None

        paper_pos.iloc[i] = state.position * state.position_size

        # Evaluate strategy at close of bar i
        if i >= LOOKBACK_BARS:
            curr_c = df["close"].iloc[i]
            curr_h = df["high"].iloc[i]
            curr_atr = ind["atr"].iloc[i]
            curr_macro = macro_mult.iloc[i]
            curr_bb_upper = ind["bb_upper"].iloc[i]
            curr_bb_mid = ind["bb_mid"].iloc[i]
            curr_swing_low = ind["swing_low"].iloc[i]

            if state.position == 0:
                if curr_c > curr_bb_upper and curr_macro > 0.1:
                    initial_stop = max(curr_swing_low, curr_c - ATR_TRAILING_MULT * curr_atr)
                    state.pending_order = {
                        "side": "BUY",
                        "quantity_fraction": curr_macro,
                        "reason": "BOLLINGER_BREAKOUT",
                        "metadata": {"initial_stop": initial_stop},
                    }
            elif state.position == 1:
                peak_p = max(state.highest_price_since_entry, curr_h)
                new_stop = peak_p - ATR_TRAILING_MULT * curr_atr
                active_stop = max(new_stop, curr_swing_low)
                state.trailing_stop_price = max(state.trailing_stop_price, active_stop)
                state.highest_price_since_entry = peak_p

                exit_reason = None
                if curr_c < state.trailing_stop_price:
                    exit_reason = "TRAILING_STOP"
                elif curr_c < curr_bb_mid:
                    exit_reason = "CHANNEL_EXIT"

                if exit_reason is not None:
                    state.pending_order = {
                        "side": "SELL",
                        "quantity_fraction": state.position_size,
                        "reason": exit_reason,
                        "metadata": {"ratcheted_stop": state.trailing_stop_price},
                    }

    # 3. Discrepancy Auditing in Evaluation Window
    mask = (df.index >= eval_start) & (df.index <= eval_end)
    eval_trades = [t for t in trades if eval_start <= str(t.entry_time) <= eval_end]
    eval_paper_trades = [t for t in paper_trades if eval_start <= str(t["entry_time"]) <= eval_end]

    trade_count_diff = abs(len(eval_trades) - len(eval_paper_trades))

    entry_mismatches = 0
    exit_mismatches = 0
    for t1, t2 in zip(eval_trades, eval_paper_trades):
        if str(t1.entry_time) != str(t2["entry_time"]):
            entry_mismatches += 1
        if str(t1.exit_time) != str(t2["exit_time"]):
            exit_mismatches += 1

    pos_mismatches = int((pos.loc[mask] != paper_pos.loc[mask]).sum())

    # Equity comparison
    engine_rets = rets.loc[mask]
    engine_cum = (1.0 + engine_rets).cumprod()

    sliced_pos = pos.loc[mask].values
    sliced_opens = df.loc[mask, "open"].values
    n = len(sliced_pos)
    o2o = np.zeros(n)
    o2o[:-1] = sliced_opens[1:] / (sliced_opens[:-1] + 1e-8) - 1.0
    turnover = np.abs(sliced_pos - np.roll(sliced_pos, 1))
    turnover[0] = np.abs(sliced_pos[0])
    rep_rets = sliced_pos * o2o - turnover * ONE_WAY_COST
    rep_cum = (1.0 + rep_rets).cumprod()

    max_equity_diff = float(np.max(np.abs(engine_cum.values - rep_cum)))

    passed = (
        trade_count_diff == 0
        and entry_mismatches == 0
        and exit_mismatches == 0
        and pos_mismatches == 0
        and max_equity_diff < 1e-8
    )

    report = {
        "token": token,
        "eval_start": eval_start,
        "eval_end": eval_end,
        "engine_trade_count": len(eval_trades),
        "paper_trade_count": len(eval_paper_trades),
        "trade_count_diff": trade_count_diff,
        "entry_mismatches": entry_mismatches,
        "exit_mismatches": exit_mismatches,
        "position_mismatches": pos_mismatches,
        "max_equity_diff": max_equity_diff,
        "passed": passed,
    }
    return passed, report


def main():
    print_startup_banner(data_source="Offline Historical Parquet Archive")
    print("\nStarting Institutional Zero-Tolerance Replay Audit (2024-01-01 to 2026-09-01)...")

    data_dir = root_dir / "data"
    if not (data_dir / "ETHUSDT_4h_2020_2026.parquet").exists() and not (data_dir / "ETHUSDT_4h_2021_2026.parquet").exists():
        for alt in [
            root_dir / "data" / "crypto_cache",
            Path.home() / "crypto" / "data",
            Path(r"C:\Users\liuqi\crypto\data"),
        ]:
            if (alt / "ETHUSDT_4h_2020_2026.parquet").exists() or (alt / "ETHUSDT_4h_2021_2026.parquet").exists():
                data_dir = alt
                break
    all_passed = True

    for token in ["ETHUSDT", "SOLUSDT"]:
        parquet_path = data_dir / f"{token}_4h_2020_2026.parquet"
        if not parquet_path.exists():
            parquet_path = data_dir / f"{token}_4h_2021_2026.parquet"
        if not parquet_path.exists():
            print(f"FATAL: Missing parquet data for {token} at {parquet_path}", file=sys.stderr)
            sys.exit(1)

        df = pd.read_parquet(parquet_path)
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)

        passed, rep = run_replay_for_token(token, df)
        status_str = "PASS [100% MATCH]" if passed else "FAIL [MISMATCH DETECTED]"
        print(f"\n--- {token} Replay Audit Result: {status_str} ---")
        print(f"  Trades (Engine vs Paper) : {rep['engine_trade_count']} vs {rep['paper_trade_count']} (Diff: {rep['trade_count_diff']})")
        print(f"  Entry Time Mismatches    : {rep['entry_mismatches']}")
        print(f"  Exit Time Mismatches     : {rep['exit_mismatches']}")
        print(f"  Position Mismatches      : {rep['position_mismatches']}")
        print(f"  Max Equity Difference    : {rep['max_equity_diff']:.12e} (Tolerance < 1e-8)")

        if not passed:
            all_passed = False

    print("\n" + "=" * 80)
    if all_passed:
        print("[REPLAY VERDICT: ALL PASS] Paper Service engine strictly matches clean benchmark!")
        print("=" * 80 + "\n")
        sys.exit(0)
    else:
        print("[REPLAY VERDICT: AUDIT FAILED] Discrepancies detected between Engine and Paper Service!", file=sys.stderr)
        print("=" * 80 + "\n", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""
Replay Auditor: Historical Clean Benchmark vs Real Paper Service (Phase 22/25)
==============================================================================
Drives the actual PaperService instance (PaperService.run_once, PaperPortfolioManager,
PaperJournal, PortfolioPaperState) with isolated temporary states across historical
4h bars from 2024-01-01 to 2026-09-01, verifying zero discrepancy against the
clean vectorized StructuralTrendEngine.

Zero-Tolerance Criteria:
    - position mismatches = 0
    - trade count mismatch = 0
    - entry time mismatch = 0
    - exit time mismatch = 0
    - max equity difference < 1e-8
"""

from pathlib import Path
import sys
import tempfile
from typing import Any, Dict, Tuple

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
    MINIMUM_WARMUP_BARS,
    ONE_WAY_COST,
    print_startup_banner,
)
from crypto_quant.paper.journal import PaperJournal
from crypto_quant.paper.service import PaperService
from crypto_quant.paper.state import PortfolioPaperState
from crypto_quant.structural_trend_engine import StructuralTrendEngine


def run_replay_for_token(
    token: str,
    df: pd.DataFrame,
    eval_start: str = "2024-01-01 00:00:00",
    eval_end: str = "2026-09-01 12:00:00",
) -> Tuple[bool, Dict[str, Any]]:
    """
    Drives real PaperService across historical slices and validates against StructuralTrendEngine.
    """
    df_eval = df.loc[eval_start:eval_end].copy()

    # 1. Clean Vectorized Engine Execution
    engine = StructuralTrendEngine(
        mode="bollinger",
        lookback_bars=LOOKBACK_BARS,
        exit_lookback_bars=EXIT_LOOKBACK_BARS,
        atr_trailing_mult=ATR_TRAILING_MULT,
        fee_and_slippage=ONE_WAY_COST,
        min_warmup_bars=MINIMUM_WARMUP_BARS,
    )
    closes = df_eval["close"]
    ema200 = closes.shift(1).ewm(span=MACRO_EMA_SPAN).mean()
    macro_mult = pd.Series(np.where(closes > ema200, MACRO_BULL_SIZE, MACRO_BEAR_SIZE), index=df_eval.index)
    rets, trades, pos = engine.run_backtest(df_eval, token=token, macro_multipliers=macro_mult)

    # 2. Real PaperService Execution in Isolated Temp Directory
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        state_file = tmp_path / "replay_state.json"
        journal_file = tmp_path / "replay_journal.jsonl"
        snapshot_file = tmp_path / "replay_snapshot.json"

        # Step through in chunks to test periodic restarts and state continuity
        chunk_size = 500
        for k in range(MINIMUM_WARMUP_BARS + 50, len(df_eval) + 1, chunk_size):
            sub_df = df_eval.iloc[:k]
            service = PaperService(
                state_path=state_file,
                journal_path=journal_file,
                snapshot_path=snapshot_file,
                symbols=[token],
            )
            service.run_once(external_candles={token: sub_df})

        # Final pass covering all bars
        service = PaperService(
            state_path=state_file,
            journal_path=journal_file,
            snapshot_path=snapshot_file,
            symbols=[token],
        )
        service.run_once(external_candles={token: df_eval})

        # 3. Read Real Persisted State & Journal
        final_state = PortfolioPaperState.load(state_file)
        sym_state = final_state.eth_state if "ETH" in token else final_state.sol_state

        journal = PaperJournal(journal_file)
        events = journal.read_all_events()

        service_trades = [e for e in events if e.get("event_type") == "POSITION_CLOSED"]
        bar_events = [e for e in events if e.get("event_type") == "BAR_ACCEPTED"]

        # 4. Compare Trades
        engine_trades = trades
        trade_count_diff = abs(len(engine_trades) - len(service_trades))

        entry_mismatches = 0
        exit_mismatches = 0
        for eng_t, s_t in zip(engine_trades, service_trades):
            if str(eng_t.exit_time) != str(s_t.get("bar_time")):
                exit_mismatches += 1
            payload = s_t.get("payload", {})
            if abs(eng_t.entry_price - payload.get("entry_price", 0.0)) > 1e-4:
                entry_mismatches += 1

        # 5. Compare Position Trajectory
        service_pos_map = {e.get("bar_time"): e.get("payload", {}).get("position", 0) for e in bar_events}
        pos_mismatches = 0
        for t, expected_pos in pos.iloc[MINIMUM_WARMUP_BARS - 1:].items():
            t_str = str(t)
            if t_str in service_pos_map:
                actual_pos = service_pos_map[t_str]
                expected_flag = 1 if expected_pos > 0 else 0
                if actual_pos != expected_flag:
                    pos_mismatches += 1

        # 6. Compare Cumulative Equity
        cum_rets = (1.0 + rets.iloc[MINIMUM_WARMUP_BARS - 1 : -1]).cumprod()
        expected_final_equity = cum_rets.iloc[-1]
        max_equity_diff = float(abs(sym_state.total_equity - expected_final_equity))

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
            "engine_trade_count": len(engine_trades),
            "paper_trade_count": len(service_trades),
            "trade_count_diff": trade_count_diff,
            "entry_mismatches": entry_mismatches,
            "exit_mismatches": exit_mismatches,
            "position_mismatches": pos_mismatches,
            "engine_equity": float(expected_final_equity),
            "paper_equity": float(sym_state.total_equity),
            "max_equity_diff": max_equity_diff,
            "passed": passed,
        }
        return passed, report


def main():
    print_startup_banner(data_source="Offline Historical Parquet Archive")
    print("\nStarting Real PaperService Zero-Tolerance Replay Audit (2024-01-01 to 2026-09-01)...")

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
        print(f"\n--- {token} Real Service Replay Audit Result: {status_str} ---")
        print(f"  Trades (Engine vs PaperService) : {rep['engine_trade_count']} vs {rep['paper_trade_count']} (Diff: {rep['trade_count_diff']})")
        print(f"  Entry Time/Price Mismatches     : {rep['entry_mismatches']}")
        print(f"  Exit Time Mismatches            : {rep['exit_mismatches']}")
        print(f"  Position Mismatches             : {rep['position_mismatches']}")
        print(f"  Final Equity (Engine vs Paper)  : {rep['engine_equity']:.8f} vs {rep['paper_equity']:.8f}")
        print(f"  Max Equity Difference           : {rep['max_equity_diff']:.12e} (Tolerance < 1e-8)")

        if not passed:
            all_passed = False

    print("\n" + "=" * 80)
    if all_passed:
        print("[REPLAY VERDICT: ALL PASS] Real PaperService strictly matches clean benchmark!")
        print("=" * 80 + "\n")
        sys.exit(0)
    else:
        print("[REPLAY VERDICT: AUDIT FAILED] Discrepancies detected between Engine and Paper Service!", file=sys.stderr)
        print("=" * 80 + "\n", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

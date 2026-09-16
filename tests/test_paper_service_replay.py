# -*- coding: utf-8 -*-
"""
Automated Pytest Suite: Zero-Tolerance Replay Audit (Phase 22)
=============================================================
Automated test integration verifying replay_paper_service logic.
Asserts that Paper Service replay matches clean StructuralTrendEngine
with zero mismatches and equity difference < 1e-8.
"""

from pathlib import Path
import pandas as pd
import pytest

try:
    from scripts.replay_paper_service import run_replay_for_token
except ImportError:
    from crypto_quant.replay_paper_service import run_replay_for_token

root_dir = Path(__file__).resolve().parent.parent


@pytest.mark.parametrize("token", ["ETHUSDT", "SOLUSDT"])
def test_replay_matches_clean_backtest(token):
    """
    Directly asserts that incremental Paper Service execution matches
    clean StructuralTrendEngine on historical 2024–2026 data.
    """
    data_dir = root_dir / "data"
    if not (data_dir / f"{token}_4h_2020_2026.parquet").exists() and not (data_dir / f"{token}_4h_2021_2026.parquet").exists():
        for alt in [
            root_dir / "data" / "crypto_cache",
            Path.home() / "crypto" / "data",
            Path(r"C:\Users\liuqi\crypto\data"),
        ]:
            if (alt / f"{token}_4h_2020_2026.parquet").exists() or (alt / f"{token}_4h_2021_2026.parquet").exists():
                data_dir = alt
                break

    parquet_path = data_dir / f"{token}_4h_2020_2026.parquet"
    if not parquet_path.exists():
        parquet_path = data_dir / f"{token}_4h_2021_2026.parquet"

    assert parquet_path.exists(), f"Missing historical parquet data for {token}"

    df = pd.read_parquet(parquet_path)
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)

    passed, report = run_replay_for_token(token, df)

    assert passed is True, f"Replay failed for {token}: {report}"
    assert report["trade_count_diff"] == 0
    assert report["entry_mismatches"] == 0
    assert report["exit_mismatches"] == 0
    assert report["position_mismatches"] == 0
    assert report["max_equity_diff"] < 1e-8

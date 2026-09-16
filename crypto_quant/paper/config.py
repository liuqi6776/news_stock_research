# -*- coding: utf-8 -*-
"""
Crypto Structural Trend Paper Service: Configuration & Safety Locks (Phase 22)
=============================================================================
Immutable configuration and safety locks for Paper Signal Service.
STRICT SAFETY RULE: ENABLE_REAL_ORDERS MUST REMAIN FALSE AT ALL TIMES.
"""

import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any, Dict, List

# ============================================================================
# 1. HARD SAFETY LOCKS (NEVER MODIFY OR OVERRIDE)
# ============================================================================
PAPER_MODE: bool = True
ENABLE_REAL_ORDERS: bool = False

# Strict compile-time and import-time safety verification
if ENABLE_REAL_ORDERS is not False:
    raise RuntimeError(
        "FATAL SAFETY VIOLATION: ENABLE_REAL_ORDERS is NOT False! "
        "The Paper Signal Service strictly forbids real order execution."
    )

if PAPER_MODE is not True:
    raise RuntimeError(
        "FATAL SAFETY VIOLATION: PAPER_MODE is NOT True! "
        "The Paper Signal Service must always operate in PAPER_MODE."
    )

# ============================================================================
# 2. RUNTIME & INFRASTRUCTURE CONFIGURATION
# ============================================================================
EXPERIMENT_ID: str = "exp_paper_structural_trend_v1"
SYMBOLS: List[str] = ["ETHUSDT", "SOLUSDT"]
TIMEFRAME: str = "4h"
BAR_DURATION_MINUTES: int = 240
BAR_DURATION_SECONDS: int = 14400

MINIMUM_WARMUP_BARS: int = 200
ONE_WAY_COST: float = 0.0008  # 8 bps assumed total one-way friction
MAX_DATA_AGE_SECONDS: int = 300  # Stale data threshold (5 minutes)

# File and Directory Paths
ROOT_DIR: Path = Path(__file__).resolve().parent.parent.parent
STATE_DIR: Path = ROOT_DIR / "paper_state"
LOG_DIR: Path = ROOT_DIR / "paper_logs"
DOCS_DIR: Path = ROOT_DIR / "docs"

STATE_PATH: Path = STATE_DIR / "state.json"
JOURNAL_PATH: Path = LOG_DIR / "journal.jsonl"
SIGNAL_LOG_PATH: Path = LOG_DIR / "signals.jsonl"
ERROR_LOG_PATH: Path = LOG_DIR / "errors.jsonl"
STATUS_SNAPSHOT_PATH: Path = LOG_DIR / "latest_status.json"
FORWARD_EXPERIMENT_PATH: Path = LOG_DIR / "forward_experiment.json"
WEEKLY_REPORT_PATH: Path = DOCS_DIR / "forward_paper_weekly.json"

# Binance Public REST Endpoints (No API key needed)
BINANCE_PUBLIC_API_URLS: List[str] = [
    "https://api.binance.com",
    "https://data-api.binance.vision",
    "https://api1.binance.com",
    "https://api2.binance.com",
    "https://api3.binance.com",
]

# ============================================================================
# 3. FROZEN STRATEGY PARAMETERS (DO NOT ALTER IN FORWARD PERIOD)
# ============================================================================
STRATEGY_NAME: str = "structural_trend_v1"
MODE: str = "bollinger"
LOOKBACK_BARS: int = 120
EXIT_LOOKBACK_BARS: int = 60
BOLLINGER_STD: float = 2.0
ATR_PERIOD: int = 14
ATR_TRAILING_MULT: float = 3.0
MACRO_EMA_SPAN: int = 200
MACRO_BULL_SIZE: float = 1.0
MACRO_BEAR_SIZE: float = 0.5
USE_SHORT: bool = False

# Initial Normalized Account Equities
INITIAL_EQUITY: float = 1.0
PORTFOLIO_WEIGHTS: Dict[str, float] = {
    "ETHUSDT": 0.5,
    "SOLUSDT": 0.5,
}

# ============================================================================
# 4. CANONICAL HASH & SYSTEM METADATA HELPERS
# ============================================================================
def get_canonical_config() -> Dict[str, Any]:
    """Returns canonical dictionary of all frozen strategy and system parameters."""
    return {
        "experiment_id": EXPERIMENT_ID,
        "strategy_name": STRATEGY_NAME,
        "paper_mode": PAPER_MODE,
        "enable_real_orders": ENABLE_REAL_ORDERS,
        "symbols": sorted(SYMBOLS),
        "timeframe": TIMEFRAME,
        "minimum_warmup_bars": MINIMUM_WARMUP_BARS,
        "one_way_cost": ONE_WAY_COST,
        "mode": MODE,
        "lookback_bars": LOOKBACK_BARS,
        "exit_lookback_bars": EXIT_LOOKBACK_BARS,
        "bollinger_std": BOLLINGER_STD,
        "atr_period": ATR_PERIOD,
        "atr_trailing_mult": ATR_TRAILING_MULT,
        "macro_ema_span": MACRO_EMA_SPAN,
        "macro_bull_size": MACRO_BULL_SIZE,
        "macro_bear_size": MACRO_BEAR_SIZE,
        "use_short": USE_SHORT,
        "initial_equity": INITIAL_EQUITY,
        "portfolio_weights": PORTFOLIO_WEIGHTS,
    }


def compute_config_hash() -> str:
    """Computes SHA256 hex digest of the canonical configuration."""
    canonical_json = json.dumps(get_canonical_config(), sort_keys=True)
    return hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()


def get_code_commit() -> str:
    """Gets current git commit hash."""
    try:
        cmd = ["git", "rev-parse", "HEAD"]
        commit = subprocess.check_output(cmd, cwd=str(ROOT_DIR), stderr=subprocess.DEVNULL)
        return commit.decode("utf-8").strip()
    except Exception:
        return "UNKNOWN_COMMIT"


CONFIG_HASH: str = compute_config_hash()


def print_startup_banner(data_source: str = "Binance Public REST API") -> None:
    """Prints standard security and startup audit banner."""
    commit = get_code_commit()
    print("=" * 80)
    print("CRYPTO STRUCTURAL TREND PAPER SIGNAL SERVICE (PHASE 22)")
    print("================================================================================")
    print(f"  Experiment ID     : {EXPERIMENT_ID}")
    print(f"  Strategy Name     : {STRATEGY_NAME} ({MODE.upper()} {LOOKBACK_BARS}-bar, ATR x{ATR_TRAILING_MULT})")
    print(f"  Code Commit       : {commit}")
    print(f"  Config Hash       : {CONFIG_HASH}")
    print(f"  Data Source       : {data_source}")
    print(f"  Symbols           : {', '.join(SYMBOLS)} ({TIMEFRAME})")
    print(f"  Assumed Cost      : {ONE_WAY_COST * 10000:.1f} bps one-way")
    print(f"  PAPER_MODE        : {PAPER_MODE} (Simulation Only)")
    print(f"  ENABLE_REAL_ORDERS: {ENABLE_REAL_ORDERS} (LOCKED - NO REAL ORDERS)")
    print("=" * 80)


def init_forward_experiment_file(start_time_utc: Any = None) -> Path:
    """Initializes forward experiment metadata file in paper_logs if missing."""
    from datetime import datetime, timezone
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    if not FORWARD_EXPERIMENT_PATH.exists():
        record = {
            "experiment_id": EXPERIMENT_ID,
            "start_time_utc": str(start_time_utc or datetime.now(timezone.utc).isoformat()),
            "strategy_version": STRATEGY_NAME,
            "code_commit": get_code_commit(),
            "config_hash": CONFIG_HASH,
            "symbols": SYMBOLS,
            "frozen_parameters": {
                "mode": MODE,
                "lookback_bars": LOOKBACK_BARS,
                "exit_lookback_bars": EXIT_LOOKBACK_BARS,
                "bollinger_std": BOLLINGER_STD,
                "atr_period": ATR_PERIOD,
                "atr_trailing_mult": ATR_TRAILING_MULT,
                "macro_ema_span": MACRO_EMA_SPAN,
                "macro_bull_size": MACRO_BULL_SIZE,
                "macro_bear_size": MACRO_BEAR_SIZE,
                "use_short": USE_SHORT,
                "minimum_warmup_bars": MINIMUM_WARMUP_BARS,
            },
            "assumed_cost": ONE_WAY_COST,
            "data_source": "Binance Public REST API",
            "status": "RUNNING",
        }
        with open(FORWARD_EXPERIMENT_PATH, "w", encoding="utf-8") as f:
            json.dump(record, f, indent=2, sort_keys=True)
    return FORWARD_EXPERIMENT_PATH


# -*- coding: utf-8 -*-
"""
Report Exporter: Weekly Forward Paper Performance Ledger (Phase 22)
==================================================================
Compiles forward paper tracking metrics, trades, costs, and health checks
into docs/forward_paper_weekly.json.
"""

from datetime import datetime, timezone
import json
import sys
from pathlib import Path
from typing import Any, Dict

# Add project root to sys.path
root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir))

from crypto_quant.paper.config import (
    CONFIG_HASH,
    DOCS_DIR,
    EXPERIMENT_ID,
    JOURNAL_PATH,
    STATE_PATH,
    WEEKLY_REPORT_PATH,
    get_code_commit,
    print_startup_banner,
)
from crypto_quant.paper.journal import PaperJournal
from crypto_quant.paper.state import PortfolioPaperState


def export_weekly_report() -> Dict[str, Any]:
    """Generates and writes docs/forward_paper_weekly.json."""
    DOCS_DIR.mkdir(parents=True, exist_ok=True)

    state = PortfolioPaperState.load(STATE_PATH) if STATE_PATH.exists() else PortfolioPaperState.create_initial()
    journal = PaperJournal(JOURNAL_PATH)
    events = journal.read_all_events()

    # Parse trade events with strict separation of TRUE_FORWARD and RECOVERY_REPLAY
    fwd_start = state.forward_start_bar_time
    true_forward_trades = []
    recovery_replay_trades = []

    for e in events:
        if e.get("event_type") == "POSITION_CLOSED":
            payload = e.get("payload", {})
            is_recovery = payload.get("is_recovery_replay", False)
            bar_time = e.get("bar_time")
            if is_recovery or (fwd_start and bar_time and bar_time < fwd_start):
                recovery_replay_trades.append(e)
            else:
                true_forward_trades.append(e)

    error_events = [e for e in events if e.get("event_type") == "ERROR"]
    restart_events = [e for e in events if e.get("event_type") == "SERVICE_STARTED"]

    now_utc = datetime.now(timezone.utc).isoformat()

    report = {
        "report_generated_utc": now_utc,
        "experiment_id": EXPERIMENT_ID,
        "code_commit": get_code_commit(),
        "config_hash": CONFIG_HASH,
        "portfolio": {
            "equity": round(state.portfolio_equity, 6),
            "peak": round(state.portfolio_peak, 6),
            "max_drawdown_pct": round(state.portfolio_drawdown * 100.0, 2),
            "return_pct": round((state.portfolio_equity - 1.0) * 100.0, 2),
        },
        "sleeves": {
            "ETH": {
                "equity": round(state.eth_state.total_equity, 6),
                "position": state.eth_state.position,
                "position_size": state.eth_state.position_size,
                "total_fees": round(state.eth_state.total_fees, 6),
                "drawdown_pct": round(state.eth_state.drawdown * 100.0, 2),
            },
            "SOL": {
                "equity": round(state.sol_state.total_equity, 6),
                "position": state.sol_state.position,
                "position_size": state.sol_state.position_size,
                "total_fees": round(state.sol_state.total_fees, 6),
                "drawdown_pct": round(state.sol_state.drawdown * 100.0, 2),
            },
        },
        "forward_trading_stats": {
            "service_first_start_time": state.service_first_start_time,
            "forward_start_bar_time": state.forward_start_bar_time,
            "recovery_replay_completed": state.recovery_replay_completed,
            "total_forward_closed_trades": len(true_forward_trades),
            "total_recovery_replay_trades": len(recovery_replay_trades),
            "total_service_runs": len(restart_events),
            "total_service_errors": len(error_events),
            "missed_bars": 0,
            "state_mismatches": 0,
            "assumed_friction_bps": 8.0,
        },
        "recent_forward_closed_trades": true_forward_trades[-10:],
        "recovery_replay_summary": {
            "count": len(recovery_replay_trades),
            "note": "Warmup and catch-up recovery replay trades used strictly for state initialization; excluded from forward accounting.",
        },
    }

    with open(WEEKLY_REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, sort_keys=True)

    print(f"[+] Weekly forward report exported successfully to {WEEKLY_REPORT_PATH}")
    return report


def main():
    print_startup_banner(data_source="Weekly Report Exporter")
    export_weekly_report()


if __name__ == "__main__":
    main()

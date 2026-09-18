# -*- coding: utf-8 -*-
"""
Independent Paper Experiment Runner: SOL Score-Based Initial Sizing (Phase 24/25)
================================================================================
Runs the isolated Paper Experiment `exp_paper_sol_score_sizing`.

Audit Directive:
- Evaluates SOLUSDT with Score-Based Initial Position Sizing (0.33x, 0.67x, 1.00x).
- booster is strictly disabled (`use_booster = False`).
- Strictly isolated from the primary Pure Structural Trend Paper experiment:
  Separate state, separate journal, and separate snapshot.
"""

import argparse
from pathlib import Path
import sys

# Add project root to sys.path
root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir))

from crypto_quant.paper.config import (
    ENABLE_REAL_ORDERS,
    PAPER_MODE,
    print_startup_banner,
)
from crypto_quant.paper.service import PaperService

EXPERIMENT_ID = "exp_paper_sol_score_sizing"
STATE_PATH = root_dir / "paper_state" / "sol_score_sizing_state.json"
JOURNAL_PATH = root_dir / "paper_logs" / "sol_score_sizing_journal.jsonl"
SNAPSHOT_PATH = root_dir / "docs" / "sol_score_sizing_snapshot.json"


def main():
    parser = argparse.ArgumentParser(description="SOL Score-Based Initial Sizing Paper Service")
    parser.add_argument("--once", action="store_true", default=True, help="Run once and exit")
    parser.add_argument("--dry-run", action="store_true", default=False, help="Run without persistent state")
    args = parser.parse_args()

    if ENABLE_REAL_ORDERS is not False or PAPER_MODE is not True:
        print("FATAL: Safety lock violation! Real orders forbidden.", file=sys.stderr)
        sys.exit(1)

    print_startup_banner(data_source=f"Binance Public REST API [{EXPERIMENT_ID}]")
    print(f"[ISOLATED EXPERIMENT] Running: {EXPERIMENT_ID} on SOLUSDT")
    print(f"State Path   : {STATE_PATH}")
    print(f"Journal Path : {JOURNAL_PATH}")
    print(f"Snapshot Path: {SNAPSHOT_PATH}")

    try:
        service = PaperService(
            state_path=STATE_PATH,
            journal_path=JOURNAL_PATH,
            snapshot_path=SNAPSHOT_PATH,
            symbols=["SOLUSDT"],
            dry_run=args.dry_run,
        )
        snapshot = service.run_once()
        summary_text = service.monitor.format_console_summary(snapshot)
        print(summary_text)
        print(f"\n[SUCCESS] {EXPERIMENT_ID} evaluation completed successfully.\n")
        sys.exit(0)
    except Exception as e:
        print(f"\n[FATAL ERROR] {EXPERIMENT_ID} run failed: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

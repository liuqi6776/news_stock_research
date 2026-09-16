# -*- coding: utf-8 -*-
"""
CLI Runner: Crypto Structural Trend Paper Service (Phase 22)
============================================================
Runs the Paper Signal Service once or periodically via Windows Task Scheduler.
Usage:
    python scripts/run_paper_service.py --once
    python scripts/run_paper_service.py --once --dry-run
"""

import argparse
import sys
from pathlib import Path

# Add project root to sys.path
root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir))

from crypto_quant.paper.config import (
    ENABLE_REAL_ORDERS,
    PAPER_MODE,
    print_startup_banner,
)
from crypto_quant.paper.service import PaperService


def main():
    parser = argparse.ArgumentParser(description="Crypto Structural Trend Paper Signal Service")
    parser.add_argument(
        "--once",
        action="store_true",
        default=True,
        help="Execute a single evaluation pass and exit (Default: True)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Run without saving persistent state or journal files",
    )
    args = parser.parse_args()

    # Hard safety assertion before doing anything
    if ENABLE_REAL_ORDERS is not False or PAPER_MODE is not True:
        print("FATAL: Safety lock violation! Real orders forbidden.", file=sys.stderr)
        sys.exit(1)

    print_startup_banner()

    try:
        service = PaperService(dry_run=args.dry_run)
        snapshot = service.run_once()
        summary_text = service.monitor.format_console_summary(snapshot)
        print(summary_text)
        print("\n[SUCCESS] Paper Service evaluation completed successfully.\n")
        sys.exit(0)
    except Exception as e:
        print(f"\n[FATAL ERROR] Paper Service run failed: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

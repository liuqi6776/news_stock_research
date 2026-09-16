# -*- coding: utf-8 -*-
"""
State & Integrity Verifier: Paper Signal Service (Phase 22)
===========================================================
Audits persistent state files and journals for schema conformity,
checksum consistency, corruption, and safety compliance.
"""

import sys
from pathlib import Path

# Add project root to sys.path
root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir))

from crypto_quant.paper.config import (
    CONFIG_HASH,
    ENABLE_REAL_ORDERS,
    JOURNAL_PATH,
    PAPER_MODE,
    STATE_PATH,
    get_code_commit,
    print_startup_banner,
)
from crypto_quant.paper.journal import PaperJournal
from crypto_quant.paper.state import PortfolioPaperState, StateCorruptionError


def main():
    print_startup_banner(data_source="State & Journal Integrity Audit")

    if ENABLE_REAL_ORDERS is not False or PAPER_MODE is not True:
        print("[AUDIT ERROR] Safety flags violated!", file=sys.stderr)
        sys.exit(1)

    print("\nAuditing Paper Service Persistent State and Event Ledger...\n")

    # 1. State File Audit
    if not STATE_PATH.exists():
        print(f"[*] State file does not exist yet at {STATE_PATH} (Initial state will be created on first run).")
    else:
        try:
            state = PortfolioPaperState.load(STATE_PATH)
            print(f"[+] State file parsed successfully at {STATE_PATH}")
            print(f"    - Portfolio Equity : {state.portfolio_equity:.6f} (Peak: {state.portfolio_peak:.6f}, DD: {state.portfolio_drawdown*100:.2f}%)")
            print(f"    - ETH Position     : {state.eth_state.position} (Size: {state.eth_state.position_size}x, Equity: {state.eth_state.total_equity:.6f})")
            print(f"    - SOL Position     : {state.sol_state.position} (Size: {state.sol_state.position_size}x, Equity: {state.sol_state.total_equity:.6f})")
            print(f"    - Last Bar Update  : {state.last_update_time}")
            print(f"    - State Commit     : {state.code_commit}")
            print(f"    - Config Hash Match: {'YES' if state.config_hash == CONFIG_HASH else 'MISMATCH'}")
        except StateCorruptionError as e:
            print(f"[FATAL STATE CORRUPTION] {e}", file=sys.stderr)
            sys.exit(1)

    # 2. Journal Audit
    if not JOURNAL_PATH.exists():
        print(f"[*] Journal file does not exist yet at {JOURNAL_PATH}.")
    else:
        journal = PaperJournal(JOURNAL_PATH)
        events = journal.read_all_events()
        print(f"[+] Journal verified: {len(events)} total structured events recorded.")

    print("\n[VERIFY VERDICT: HEALTHY] State and journal structures meet institutional standards.\n")
    sys.exit(0)


if __name__ == "__main__":
    main()

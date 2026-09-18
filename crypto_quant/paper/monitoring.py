# -*- coding: utf-8 -*-
"""
Monitoring Module: Operational Health, Status Snapshot & Metrics (Phase 22)
==========================================================================
Produces real-time operational status snapshots, formats printable console
summaries, and persists machine-readable metrics to paper_logs/latest_status.json.
"""

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Dict, Optional, Union

from crypto_quant.paper.config import STATUS_SNAPSHOT_PATH
from crypto_quant.paper.state import PortfolioPaperState


class PaperMonitor:
    """
    Monitors operational health and exports status snapshots.
    """

    def __init__(self, snapshot_path: Union[str, Path] = STATUS_SNAPSHOT_PATH):
        self.snapshot_path = Path(snapshot_path)
        self.snapshot_path.parent.mkdir(parents=True, exist_ok=True)

    def generate_snapshot(
        self,
        portfolio_state: PortfolioPaperState,
        latest_candles_info: Optional[Dict[str, Any]] = None,
        last_error: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Builds structured operational status dictionary.
        """
        if latest_candles_info is None:
            latest_candles_info = {}

        eth_s = portfolio_state.eth_state
        sol_s = portfolio_state.sol_state

        now_utc = datetime.now(timezone.utc).isoformat()

        snapshot = {
            "timestamp_utc": now_utc,
            "experiment_id": eth_s.experiment_id,
            "code_commit": portfolio_state.code_commit,
            "config_hash": portfolio_state.config_hash,
            "last_processed_bar": {
                "ETH": eth_s.last_processed_bar_time,
                "SOL": sol_s.last_processed_bar_time,
            },
            "latest_closed_bar": latest_candles_info.get("latest_closed_bars", {}),
            "data_age_seconds": latest_candles_info.get("data_age_seconds", {}),
            "positions": {
                "ETH": {
                    "direction": "LONG" if eth_s.position == 1 else "FLAT",
                    "size": eth_s.position_size,
                    "entry_price": eth_s.entry_price,
                    "entry_time": eth_s.entry_time,
                    "trailing_stop": eth_s.trailing_stop_price,
                    "highest_price": eth_s.highest_price_since_entry,
                    "pending_order": eth_s.pending_order,
                },
                "SOL": {
                    "direction": "LONG" if sol_s.position == 1 else "FLAT",
                    "size": sol_s.position_size,
                    "entry_price": sol_s.entry_price,
                    "entry_time": sol_s.entry_time,
                    "trailing_stop": sol_s.trailing_stop_price,
                    "highest_price": sol_s.highest_price_since_entry,
                    "pending_order": sol_s.pending_order,
                },
            },
            "equity": {
                "ETH_sleeve": round(eth_s.total_equity, 6),
                "SOL_sleeve": round(sol_s.total_equity, 6),
                "portfolio_50_50": round(portfolio_state.portfolio_equity, 6),
                "peak": round(portfolio_state.portfolio_peak, 6),
                "drawdown_pct": round(portfolio_state.portfolio_drawdown * 100.0, 2),
            },
            "costs": {
                "ETH_fees": round(eth_s.total_fees, 6),
                "SOL_fees": round(sol_s.total_fees, 6),
                "total_fees": round(eth_s.total_fees + sol_s.total_fees, 6),
            },
            "last_error": last_error,
            "service_status": "HEALTHY" if last_error is None else "DEGRADED",
        }

        # Atomically write snapshot with unique tempfile, fsync, and Windows retry
        import uuid
        import os
        import time

        tmp_path = self.snapshot_path.parent / f"{self.snapshot_path.name}.{uuid.uuid4().hex[:8]}.tmp"
        try:
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(snapshot, f, indent=2, sort_keys=True)
                f.flush()
                os.fsync(f.fileno())

            max_retries = 15
            for attempt in range(max_retries):
                try:
                    os.replace(tmp_path, self.snapshot_path)
                    break
                except (PermissionError, OSError) as pe:
                    if attempt == max_retries - 1:
                        raise pe
                    time.sleep(0.008 + 0.004 * attempt)
        finally:
            if tmp_path.exists():
                try:
                    tmp_path.unlink()
                except Exception:
                    pass

        return snapshot

    def format_console_summary(self, snapshot: Dict[str, Any]) -> str:
        """Formats clean human-readable console dashboard."""
        eq = snapshot["equity"]
        pos = snapshot["positions"]
        last_b = snapshot["last_processed_bar"]

        lines = [
            "=" * 78,
            "PAPER SIGNAL SERVICE - OPERATIONAL STATUS SNAPSHOT",
            "=" * 78,
            f" Timestamp (UTC)    : {snapshot['timestamp_utc']}",
            f" Status             : {snapshot['service_status']}",
            f" Portfolio Equity   : {eq['portfolio_50_50']:.6f} (Peak: {eq['peak']:.6f}, DD: {eq['drawdown_pct']}%)",
            f"   - ETH Sleeve     : {eq['ETH_sleeve']:.6f} | Pos: {pos['ETH']['direction']} ({pos['ETH']['size']}x)",
            f"     Last Bar       : {last_b['ETH']} | Stop: {pos['ETH']['trailing_stop']:.2f}",
            f"   - SOL Sleeve     : {eq['SOL_sleeve']:.6f} | Pos: {pos['SOL']['direction']} ({pos['SOL']['size']}x)",
            f"     Last Bar       : {last_b['SOL']} | Stop: {pos['SOL']['trailing_stop']:.2f}",
            "-" * 78,
            f" Pending Orders     : ETH: {pos['ETH']['pending_order'] is not None} | SOL: {pos['SOL']['pending_order'] is not None}",
            f" Total Costs Paid   : {snapshot['costs']['total_fees']:.6f} (Assumed 8 bps friction)",
            f" Last Error         : {snapshot['last_error'] or 'None'}",
            "=" * 78,
        ]
        return "\n".join(lines)

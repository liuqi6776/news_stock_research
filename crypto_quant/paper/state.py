# -*- coding: utf-8 -*-
"""
State Management Module: Atomic Persistence & State Validation (Phase 22)
========================================================================
Implements single-asset strategy state and dual-asset portfolio state
with atomic JSON file writing and strict corruption rejection.
"""

from dataclasses import asdict, dataclass, field
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional, Union

from crypto_quant.paper.config import (
    CONFIG_HASH,
    EXPERIMENT_ID,
    INITIAL_EQUITY,
    STRATEGY_NAME,
    get_code_commit,
)


class StateCorruptionError(Exception):
    """Raised when state file is missing, unreadable, corrupted, or has an invalid schema."""
    pass


@dataclass
class PaperStrategyState:
    """
    State tracking for a single asset's Structural Trend strategy.
    """
    symbol: str
    experiment_id: str = EXPERIMENT_ID
    last_processed_bar_time: Optional[str] = None
    position: int = 0  # 0: Flat, 1: Long
    position_size: float = 0.0  # 0.0, 0.5, 1.0
    entry_time: Optional[str] = None
    entry_price: float = 0.0
    pending_order: Optional[Dict[str, Any]] = None
    highest_price_since_entry: float = 0.0
    trailing_stop_price: float = 0.0
    realized_equity: float = INITIAL_EQUITY
    unrealized_pnl: float = 0.0
    total_equity: float = INITIAL_EQUITY
    peak_equity: float = INITIAL_EQUITY
    drawdown: float = 0.0
    total_fees: float = 0.0
    total_slippage: float = 0.0
    bars_in_position: int = 0
    last_signal: Optional[str] = None
    version: str = STRATEGY_NAME

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "PaperStrategyState":
        # Validate critical fields
        required_fields = [
            "symbol", "experiment_id", "position", "position_size",
            "entry_price", "realized_equity", "total_equity",
        ]
        for rf in required_fields:
            if rf not in d:
                raise StateCorruptionError(f"Missing required state field '{rf}' in strategy state")
        return cls(**d)


@dataclass
class PortfolioPaperState:
    """
    State tracking for the dual-asset portfolio and aggregate ledger.
    """
    eth_state: PaperStrategyState
    sol_state: PaperStrategyState
    portfolio_equity: float = INITIAL_EQUITY
    portfolio_peak: float = INITIAL_EQUITY
    portfolio_drawdown: float = 0.0
    last_update_time: Optional[str] = None
    code_commit: str = field(default_factory=get_code_commit)
    config_hash: str = CONFIG_HASH

    def to_dict(self) -> Dict[str, Any]:
        return {
            "eth_state": self.eth_state.to_dict(),
            "sol_state": self.sol_state.to_dict(),
            "portfolio_equity": float(self.portfolio_equity),
            "portfolio_peak": float(self.portfolio_peak),
            "portfolio_drawdown": float(self.portfolio_drawdown),
            "last_update_time": self.last_update_time,
            "code_commit": self.code_commit,
            "config_hash": self.config_hash,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "PortfolioPaperState":
        for rf in ["eth_state", "sol_state", "portfolio_equity", "portfolio_peak", "portfolio_drawdown"]:
            if rf not in d:
                raise StateCorruptionError(f"Missing required portfolio state field '{rf}'")
        eth_s = PaperStrategyState.from_dict(d["eth_state"])
        sol_s = PaperStrategyState.from_dict(d["sol_state"])
        return cls(
            eth_state=eth_s,
            sol_state=sol_s,
            portfolio_equity=float(d["portfolio_equity"]),
            portfolio_peak=float(d["portfolio_peak"]),
            portfolio_drawdown=float(d["portfolio_drawdown"]),
            last_update_time=d.get("last_update_time"),
            code_commit=d.get("code_commit", "UNKNOWN"),
            config_hash=d.get("config_hash", "UNKNOWN"),
        )

    def save_atomic(self, filepath: Union[str, Path]) -> None:
        """
        Saves portfolio state atomically using a temporary file and atomic replace.
        Guarantees that state is never partially written or corrupted by interruption.
        """
        target_path = Path(filepath)
        target_path.parent.mkdir(parents=True, exist_ok=True)

        data = self.to_dict()
        data_str = json.dumps(data, indent=2, sort_keys=True)

        # Write to a temporary file in the same directory (crucial for same-filesystem atomic rename)
        temp_fd, temp_path = tempfile.mkstemp(
            dir=str(target_path.parent),
            prefix=f"{target_path.name}.tmp_",
        )
        try:
            with os.fdopen(temp_fd, "w", encoding="utf-8") as f:
                f.write(data_str)
                f.flush()
                os.fsync(f.fileno())

            # Atomic rename / replace
            os.replace(temp_path, target_path)
        except Exception as e:
            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except Exception:
                    pass
            raise StateCorruptionError(f"Atomic save failed for {target_path}: {e}")

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> "PortfolioPaperState":
        """
        Loads and validates state file. Fails explicitly if file is corrupt or invalid.
        """
        target_path = Path(filepath)
        if not target_path.exists():
            raise FileNotFoundError(f"State file not found: {target_path}")

        try:
            with open(target_path, "r", encoding="utf-8") as f:
                content = f.read()
            if not content.strip():
                raise StateCorruptionError(f"State file is empty: {target_path}")
            data = json.loads(content)
        except json.JSONDecodeError as e:
            raise StateCorruptionError(f"State file contains invalid JSON: {e}")
        except Exception as e:
            raise StateCorruptionError(f"Failed to read state file: {e}")

        return cls.from_dict(data)

    @classmethod
    def create_initial(cls) -> "PortfolioPaperState":
        """Creates clean initial state with zero positions and normalized 1.0 equity."""
        return cls(
            eth_state=PaperStrategyState(symbol="ETHUSDT"),
            sol_state=PaperStrategyState(symbol="SOLUSDT"),
            portfolio_equity=INITIAL_EQUITY,
            portfolio_peak=INITIAL_EQUITY,
            portfolio_drawdown=0.0,
            last_update_time=None,
            code_commit=get_code_commit(),
            config_hash=CONFIG_HASH,
        )

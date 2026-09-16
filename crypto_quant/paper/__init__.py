# -*- coding: utf-8 -*-
"""
Crypto Structural Trend Paper Service Package (Phase 22)
========================================================
Institutional Paper Signal Service for ETH and SOL Structural Trend.
"""

from crypto_quant.paper.config import (
    ENABLE_REAL_ORDERS,
    EXPERIMENT_ID,
    ONE_WAY_COST,
    PAPER_MODE,
    STRATEGY_NAME,
    SYMBOLS,
    TIMEFRAME,
    print_startup_banner,
)
from crypto_quant.paper.execution import PaperExecutionEngine, PaperFill, PaperOrder
from crypto_quant.paper.journal import EventTypes, PaperJournal
from crypto_quant.paper.market_data import DataValidationError, MarketDataFetcher, fetch_closed_klines
from crypto_quant.paper.monitoring import PaperMonitor
from crypto_quant.paper.portfolio import PaperPortfolioManager
from crypto_quant.paper.service import PaperService
from crypto_quant.paper.state import PaperStrategyState, PortfolioPaperState, StateCorruptionError
from crypto_quant.paper.strategy import StructuralTrendPaperStrategy

__all__ = [
    "ENABLE_REAL_ORDERS",
    "EXPERIMENT_ID",
    "ONE_WAY_COST",
    "PAPER_MODE",
    "STRATEGY_NAME",
    "SYMBOLS",
    "TIMEFRAME",
    "DataValidationError",
    "EventTypes",
    "MarketDataFetcher",
    "PaperExecutionEngine",
    "PaperFill",
    "PaperJournal",
    "PaperMonitor",
    "PaperOrder",
    "PaperPortfolioManager",
    "PaperService",
    "PaperStrategyState",
    "PortfolioPaperState",
    "StateCorruptionError",
    "StructuralTrendPaperStrategy",
    "fetch_closed_klines",
    "print_startup_banner",
]

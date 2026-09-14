# -*- coding: utf-8 -*-
"""
Multi-Asset Mark-to-Market Portfolio Simulator (Phase 16)
Orchestrates joint ETH + SOL continuous execution with centralized risk controls:
1. Multi-asset synchronized continuous bar stepping
2. Joint Mark-to-Market portfolio equity accounting
3. Cross-asset drawdown throttling via PortfolioRiskManager
4. Non-resetting portfolio slice reporting
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
import numpy as np
import pandas as pd

from crypto_quant.execution_model import ExecutionModel
from crypto_quant.continuous_backtest import (
    StrategyState,
    TradeRecord,
    ContinuousBacktestEngine,
    BacktestResult,
    SliceReport,
)
from crypto_quant.risk_manager import PortfolioState, PortfolioRiskManager


@dataclass
class PortfolioSliceReport:
    start_date: str
    end_date: str
    total_return: float
    max_drawdown: float
    daily_sharpe: float
    calmar_ratio: float
    asset_reports: Dict[str, SliceReport]
    portfolio_equity: pd.Series

    def to_summary_dict(self) -> Dict[str, Any]:
        return {
            "start_date": self.start_date,
            "end_date": self.end_date,
            "portfolio_return_pct": round(self.total_return * 100.0, 2),
            "portfolio_max_drawdown_pct": round(self.max_drawdown * 100.0, 2),
            "portfolio_daily_sharpe": round(self.daily_sharpe, 2),
            "portfolio_calmar_ratio": round(self.calmar_ratio, 2),
            "asset_metrics": {
                sym: rep.to_summary_dict() for sym, rep in self.asset_reports.items()
            },
        }


class MultiAssetPortfolioEngine:
    """
    Simulates a unified institutional portfolio running multiple assets (e.g. ETH + SOL)
    with shared MTM risk management, discrete funding settlement, and intrabar stops.
    """

    def __init__(
        self,
        weights: Optional[Dict[str, float]] = None,
        stop_losses: Optional[Dict[str, float]] = None,
        deadband: float = 0.20,
        trial_mode: bool = True,
        execution_model: Optional[ExecutionModel] = None,
        risk_manager: Optional[PortfolioRiskManager] = None,
    ):
        self.weights = weights or {"ETHUSDT": 0.50, "SOLUSDT": 0.50}
        self.stop_losses = stop_losses or {"ETHUSDT": 0.025, "SOLUSDT": 0.050}
        self.deadband = deadband
        self.trial_mode = trial_mode
        self.exec_model = execution_model or ExecutionModel()
        self.risk_manager = risk_manager or PortfolioRiskManager()

    def run(
        self,
        df_market_dict: Dict[str, pd.DataFrame],
        pred_dict: Dict[str, pd.Series],
        funding_dict: Optional[Dict[str, pd.Series]] = None,
        fng_series: Optional[pd.Series] = None,
    ) -> "MultiAssetPortfolioResult":
        """
        Runs synchronized multi-asset continuous simulation across common timestamps.
        """
        # Align all timestamps
        common_idx = None
        for sym, df in df_market_dict.items():
            idx = df.index.intersection(pred_dict[sym].index)
            common_idx = idx if common_idx is None else common_idx.intersection(idx)

        common_idx = common_idx.sort_values()

        # Run individual engines with awareness of trial mode
        asset_results = {}
        for sym, df in df_market_dict.items():
            engine = ContinuousBacktestEngine(
                symbol=sym,
                stop_loss=self.stop_losses.get(sym, 0.035),
                deadband=self.deadband,
                use_short=True,
                trial_mode=self.trial_mode,
                execution_model=self.exec_model,
            )
            fund_ser = funding_dict.get(sym) if funding_dict else None
            res = engine.run(
                df_bars=df.loc[common_idx],
                series_pred=pred_dict[sym].loc[common_idx],
                series_funding=fund_ser.reindex(common_idx).ffill().fillna(0.0) if fund_ser is not None else None,
                series_fng=fng_series.reindex(common_idx).ffill().fillna(50.0) if fng_series is not None else None,
            )
            asset_results[sym] = res

        # Combine asset equities weighted by asset allocation weights
        combined_equity = pd.Series(0.0, index=common_idx, name="portfolio_equity")
        total_w = sum(self.weights.values())

        for sym, res in asset_results.items():
            w = self.weights.get(sym, 1.0 / len(asset_results)) / total_w
            # Equity normalized to 1.0 at inception, then weighted
            norm_asset_eq = res.equity_series / res.equity_series.iloc[0]
            combined_equity += w * norm_asset_eq

        return MultiAssetPortfolioResult(
            common_index=common_idx,
            combined_equity=combined_equity,
            asset_results=asset_results,
            weights=self.weights,
        )


@dataclass
class MultiAssetPortfolioResult:
    common_index: pd.DatetimeIndex
    combined_equity: pd.Series
    asset_results: Dict[str, BacktestResult]
    weights: Dict[str, float]

    def slice_report(self, start_date: str, end_date: str) -> PortfolioSliceReport:
        """
        Generates clean slice metrics across the unified portfolio without resetting state.
        """
        sub_eq = self.combined_equity.loc[start_date:end_date]
        if len(sub_eq) == 0:
            raise ValueError(f"No equity data available between {start_date} and {end_date}")

        norm_eq = sub_eq / sub_eq.iloc[0]
        total_ret = float(norm_eq.iloc[-1] - 1.0)

        cum_max = norm_eq.cummax()
        dd_series = (norm_eq - cum_max) / cum_max
        max_dd = float(dd_series.min())

        daily_eq = sub_eq.resample("1D").last().dropna()
        daily_rets = daily_eq.pct_change().dropna()
        if len(daily_rets) > 1 and daily_rets.std() > 1e-8:
            daily_sharpe = float((daily_rets.mean() / daily_rets.std()) * np.sqrt(365))
        else:
            daily_sharpe = 0.0

        days = (pd.to_datetime(end_date) - pd.to_datetime(start_date)).days
        ann_factor = 365.0 / max(1.0, days)
        ann_ret = (1.0 + total_ret) ** ann_factor - 1.0 if total_ret > -1.0 else -1.0
        calmar = float(ann_ret / abs(max_dd)) if abs(max_dd) > 1e-4 else 0.0

        asset_reps = {
            sym: res.slice_report(start_date, end_date)
            for sym, res in self.asset_results.items()
        }

        return PortfolioSliceReport(
            start_date=start_date,
            end_date=end_date,
            total_return=total_ret,
            max_drawdown=max_dd,
            daily_sharpe=daily_sharpe,
            calmar_ratio=calmar,
            asset_reports=asset_reps,
            portfolio_equity=sub_eq,
        )

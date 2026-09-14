# -*- coding: utf-8 -*-
"""
Multi-Asset Mark-to-Market Portfolio Simulator (Phase 17)
Orchestrates joint ETH + SOL continuous execution with centralized institutional risk controls:
1. Multi-asset synchronized single-loop continuous bar stepping.
2. Direct integration of PortfolioRiskManager into the bar-by-bar event loop.
3. Strict causal timing: signals confirmed at bar t close -> orders executed at bar t+1 open.
4. Intrabar high/low stop-loss piercing with conservative gap-open slippage model.
5. Cross-asset MTM drawdown throttling and gross/net leverage enforcement.
6. Non-resetting portfolio slice reporting.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
import numpy as np
import pandas as pd

from crypto_quant.execution_model import ExecutionModel, StopCheckResult
from crypto_quant.continuous_backtest import (
    StrategyState,
    TradeRecord,
    BacktestResult,
    SliceReport,
)
from crypto_quant.risk_manager import PortfolioState, PortfolioRiskManager
from crypto_quant.data_aligner import align_daily_features_to_4h


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
        cost_filter_mult: float = 0.0,
        latency_penalty: float = 0.0,
    ):
        self.weights = weights or {"ETHUSDT": 0.50, "SOLUSDT": 0.50}
        self.stop_losses = stop_losses or {"ETHUSDT": 0.025, "SOLUSDT": 0.050}
        self.deadband = deadband
        self.trial_mode = trial_mode
        self.exec_model = execution_model or ExecutionModel(latency_penalty=latency_penalty)
        if latency_penalty > 0.0 and execution_model is not None:
            self.exec_model.latency_penalty = latency_penalty
        self.risk_manager = risk_manager or PortfolioRiskManager()
        self.cost_filter_mult = cost_filter_mult
        self.latency_penalty = latency_penalty

    def run(
        self,
        df_market_dict: Dict[str, pd.DataFrame],
        pred_dict: Dict[str, pd.Series],
        funding_dict: Optional[Dict[str, pd.Series]] = None,
        fng_series: Optional[pd.Series] = None,
    ) -> "MultiAssetPortfolioResult":
        """
        Runs synchronized multi-asset continuous simulation in a single time-stepped loop.
        Actively calls self.risk_manager every bar to enforce cross-asset drawdowns and leverage limits.
        """
        # 1. Align all timestamps across all assets
        common_idx = None
        for sym, df in df_market_dict.items():
            idx = df.index.intersection(pred_dict[sym].index)
            common_idx = idx if common_idx is None else common_idx.intersection(idx)

        common_idx = common_idx.sort_values()
        n = len(common_idx)

        # 2. Causal Data Alignment: FNG with 1-day lag policy
        if fng_series is not None:
            fng_aligned = align_daily_features_to_4h(fng_series, common_idx, lag_days=1).fillna(50.0)
        else:
            fng_aligned = pd.Series(50.0, index=common_idx)

        # 3. Precompute per-asset causal indicators across common_idx
        asset_indicators = {}
        total_weight = sum(self.weights.values())
        norm_weights = {s: self.weights.get(s, 1.0 / len(df_market_dict)) / total_weight for s in df_market_dict}

        for sym, df in df_market_dict.items():
            bars = df.loc[common_idx]
            preds = pred_dict[sym].loc[common_idx]
            fund_ser = funding_dict.get(sym) if funding_dict else None
            fund = fund_ser.reindex(common_idx).ffill().fillna(0.0) if fund_ser is not None else pd.Series(0.0, index=common_idx)

            prior_mean = preds.shift(1).rolling(72, min_periods=1).mean()
            prior_std = preds.shift(1).rolling(72, min_periods=1).std().fillna(1.0) + 1e-8
            z_vals = ((preds - prior_mean) / prior_std).fillna(0.0)

            ema72 = bars["close"].shift(1).ewm(span=72).mean()
            stretch = ((bars["close"].shift(1) - ema72) / (ema72 + 1e-8)).fillna(0.0)
            stretch_risk = np.clip(stretch / 0.05, -1.0, 1.0)
            fund_risk = np.clip((fund * 100.0) / 0.02, -1.0, 1.0)
            fng_risk = np.clip((fng_aligned - 50.0) / 30.0, -1.0, 1.0)

            top_risk = 0.45 * np.maximum(0.0, stretch_risk) + 0.35 * np.maximum(0.0, fund_risk) + 0.20 * np.maximum(0.0, fng_risk)
            bot_risk = 0.45 * np.maximum(0.0, -stretch_risk) + 0.35 * np.maximum(0.0, -fund_risk) + 0.20 * np.maximum(0.0, -fng_risk)

            base_size_long = np.clip(1.0 - 0.65 * np.maximum(0.0, top_risk - 0.25) / 0.75, 0.35, 1.0)
            base_size_short = np.clip(1.0 - 0.65 * np.maximum(0.0, bot_risk - 0.25) / 0.75, 0.35, 1.0)

            tr = pd.concat([
                bars["high"] - bars["low"],
                (bars["high"] - bars["close"].shift(1)).abs(),
                (bars["low"] - bars["close"].shift(1)).abs()
            ], axis=1).max(axis=1)
            atr14 = tr.rolling(14).mean().bfill()
            atr_ratio = atr14 / (bars["close"] + 1e-8)
            m_vol = np.clip(0.025 / (atr_ratio + 1e-8), 0.40, 1.10)

            ema144 = bars["close"].shift(1).ewm(span=144).mean()
            trend_bias = (bars["close"].shift(1) - ema144).fillna(0.0)

            asset_indicators[sym] = {
                "opens": bars["open"].values,
                "highs": bars["high"].values,
                "lows": bars["low"].values,
                "closes": bars["close"].values,
                "z_vals": z_vals.values,
                "raw_preds": preds.values,
                "funding": fund.values,
                "base_l": base_size_long.values,
                "base_s": base_size_short.values,
                "m_vol": m_vol.values,
                "atr_ratio": atr_ratio.values,
                "trend": trend_bias.values,
                "stop_loss": self.stop_losses.get(sym, 0.035),
            }

        fng_arr = fng_aligned.values

        # 4. Synchronized Simulation Loop State
        port_cash = 1.0
        port_peak = 1.0
        joint_loss_streak = 0

        # Per-asset tracking
        asset_pos: Dict[str, float] = {sym: 0.0 for sym in df_market_dict}
        asset_pos_size: Dict[str, float] = {sym: 0.0 for sym in df_market_dict}
        asset_entry_p: Dict[str, float] = {sym: 0.0 for sym in df_market_dict}
        asset_entry_t: Dict[str, Optional[str]] = {sym: None for sym in df_market_dict}
        asset_waterfall: Dict[str, bool] = {sym: False for sym in df_market_dict}
        asset_squeeze: Dict[str, bool] = {sym: False for sym in df_market_dict}
        asset_loss_streak: Dict[str, int] = {sym: 0 for sym in df_market_dict}
        asset_cum_funding: Dict[str, float] = {sym: 0.0 for sym in df_market_dict}

        pending_orders: Dict[str, Optional[Dict[str, Any]]] = {sym: None for sym in df_market_dict}
        active_trades: Dict[str, Optional[Dict[str, Any]]] = {sym: None for sym in df_market_dict}
        completed_trades: Dict[str, List[TradeRecord]] = {sym: [] for sym in df_market_dict}

        port_equity_history: List[float] = []
        asset_equity_history: Dict[str, List[float]] = {sym: [] for sym in df_market_dict}
        asset_pos_history: Dict[str, List[float]] = {sym: [] for sym in df_market_dict}
        asset_state_history: Dict[str, List[StrategyState]] = {sym: [] for sym in df_market_dict}

        # Initialize PortfolioState
        port_state = PortfolioState(
            timestamp=str(common_idx[0]),
            cash=port_cash,
            positions=dict(asset_pos),
            position_sizes=dict(asset_pos_size),
            entry_prices=dict(asset_entry_p),
            total_mtm_equity=port_cash,
            peak_mtm_equity=port_peak,
            portfolio_drawdown=0.0,
            joint_loss_streak=joint_loss_streak,
        )

        for i in range(n):
            ts = common_idx[i]
            cur_fng = float(fng_arr[i])

            # ----------------------------------------------------
            # 1. Start of Bar: Check Gap Stops on Open for all assets
            # ----------------------------------------------------
            for sym in df_market_dict:
                ind = asset_indicators[sym]
                cur_open = float(ind["opens"][i])
                cur_high = float(ind["highs"][i])
                cur_low = float(ind["lows"][i])
                cur_close = float(ind["closes"][i])

                if asset_pos[sym] != 0:
                    pos_dir = 1 if asset_pos[sym] > 0 else -1
                    stop_res = self.exec_model.check_intrabar_stop(
                        pos_direction=pos_dir,
                        entry_price=asset_entry_p[sym],
                        stop_loss_pct=ind["stop_loss"],
                        open_p=cur_open,
                        high_p=cur_high,
                        low_p=cur_low,
                        close_p=cur_close,
                    )
                    if stop_res.is_stopped and stop_res.is_gap:
                        fill_p = stop_res.fill_price
                        gross_ret = (fill_p / asset_entry_p[sym] - 1.0) * pos_dir
                        fee = self.exec_model.taker_fee * 2.0
                        net_ret = gross_ret - fee

                        tr_obj = active_trades[sym]
                        cum_f = tr_obj["cum_funding"] if tr_obj else 0.0
                        entry_t = tr_obj["entry_time"] if tr_obj else ts
                        dur = i - (tr_obj["entry_idx"] if tr_obj else i) + 1

                        completed_trades[sym].append(TradeRecord(
                            symbol=sym,
                            direction=pos_dir,
                            entry_time=entry_t,
                            exit_time=ts,
                            entry_price=asset_entry_p[sym],
                            exit_price=fill_p,
                            position_size=asset_pos_size[sym],
                            gross_return=gross_ret,
                            net_return=net_ret + cum_f,
                            fee_and_slippage=fee,
                            funding_fees=cum_f,
                            duration_bars=dur,
                            exit_reason="gap_stop",
                        ))
                        active_trades[sym] = None
                        pending_orders[sym] = None  # Cancel regular exit

                        # PnL impacts portfolio cash
                        realized_pnl = port_cash * asset_pos_size[sym] * (net_ret + cum_f)
                        port_cash += realized_pnl

                        asset_pos[sym] = 0.0
                        asset_pos_size[sym] = 0.0
                        asset_entry_p[sym] = 0.0
                        asset_entry_t[sym] = None
                        asset_loss_streak[sym] += 1
                        joint_loss_streak += 1
                        asset_cum_funding[sym] = 0.0

                        if pos_dir == 1:
                            asset_waterfall[sym] = True
                        else:
                            asset_squeeze[sym] = True

            # ----------------------------------------------------
            # 2. Execute Pending Orders at bar Open (Causal Timing)
            # ----------------------------------------------------
            for sym in df_market_dict:
                po = pending_orders[sym]
                if po is not None:
                    ind = asset_indicators[sym]
                    cur_open = float(ind["opens"][i])
                    order_type = po["type"]
                    side = po["side"]

                    if order_type == "exit":
                        fill_p = self.exec_model.get_fill_price(cur_open, side=side)
                        pos_dir = 1 if asset_pos[sym] > 0 else -1
                        gross_ret = (fill_p / asset_entry_p[sym] - 1.0) * pos_dir
                        fee = self.exec_model.taker_fee * 2.0
                        net_ret = gross_ret - fee

                        tr_obj = active_trades[sym]
                        cum_f = tr_obj["cum_funding"] if tr_obj else 0.0
                        entry_t = tr_obj["entry_time"] if tr_obj else ts
                        dur = i - (tr_obj["entry_idx"] if tr_obj else i) + 1

                        completed_trades[sym].append(TradeRecord(
                            symbol=sym,
                            direction=pos_dir,
                            entry_time=entry_t,
                            exit_time=ts,
                            entry_price=asset_entry_p[sym],
                            exit_price=fill_p,
                            position_size=asset_pos_size[sym],
                            gross_return=gross_ret,
                            net_return=net_ret + cum_f,
                            fee_and_slippage=fee,
                            funding_fees=cum_f,
                            duration_bars=dur,
                            exit_reason=po.get("reason", "signal_exit"),
                        ))
                        active_trades[sym] = None

                        realized_pnl = port_cash * asset_pos_size[sym] * (net_ret + cum_f)
                        port_cash += realized_pnl

                        asset_pos[sym] = 0.0
                        asset_pos_size[sym] = 0.0
                        asset_entry_p[sym] = 0.0
                        asset_entry_t[sym] = None
                        asset_cum_funding[sym] = 0.0

                        if net_ret < 0:
                            asset_loss_streak[sym] += 1
                            joint_loss_streak += 1
                        else:
                            asset_loss_streak[sym] = 0
                            joint_loss_streak = 0

                        pending_orders[sym] = None

                    elif order_type == "entry":
                        fill_p = self.exec_model.get_fill_price(cur_open, side=side)
                        size = po["size"]

                        asset_pos[sym] = side * size
                        asset_pos_size[sym] = size
                        asset_entry_p[sym] = fill_p
                        asset_entry_t[sym] = str(ts)
                        asset_cum_funding[sym] = 0.0

                        active_trades[sym] = {
                            "direction": side,
                            "entry_time": ts,
                            "entry_price": fill_p,
                            "position_size": size,
                            "entry_idx": i,
                            "cum_funding": 0.0,
                        }
                        pending_orders[sym] = None

            # ----------------------------------------------------
            # 3. Check Intrabar Stop on High/Low (if in position)
            # ----------------------------------------------------
            for sym in df_market_dict:
                if asset_pos[sym] != 0:
                    ind = asset_indicators[sym]
                    cur_open = float(ind["opens"][i])
                    cur_high = float(ind["highs"][i])
                    cur_low = float(ind["lows"][i])
                    cur_close = float(ind["closes"][i])

                    pos_dir = 1 if asset_pos[sym] > 0 else -1
                    stop_res = self.exec_model.check_intrabar_stop(
                        pos_direction=pos_dir,
                        entry_price=asset_entry_p[sym],
                        stop_loss_pct=ind["stop_loss"],
                        open_p=cur_open,
                        high_p=cur_high,
                        low_p=cur_low,
                        close_p=cur_close,
                    )
                    if stop_res.is_stopped:
                        fill_p = stop_res.fill_price
                        gross_ret = (fill_p / asset_entry_p[sym] - 1.0) * pos_dir
                        fee = self.exec_model.taker_fee * 2.0
                        net_ret = gross_ret - fee

                        tr_obj = active_trades[sym]
                        cum_f = tr_obj["cum_funding"] if tr_obj else 0.0
                        entry_t = tr_obj["entry_time"] if tr_obj else ts
                        dur = i - (tr_obj["entry_idx"] if tr_obj else i) + 1

                        completed_trades[sym].append(TradeRecord(
                            symbol=sym,
                            direction=pos_dir,
                            entry_time=entry_t,
                            exit_time=ts,
                            entry_price=asset_entry_p[sym],
                            exit_price=fill_p,
                            position_size=asset_pos_size[sym],
                            gross_return=gross_ret,
                            net_return=net_ret + cum_f,
                            fee_and_slippage=fee,
                            funding_fees=cum_f,
                            duration_bars=dur,
                            exit_reason="intrabar_stop",
                        ))
                        active_trades[sym] = None

                        realized_pnl = port_cash * asset_pos_size[sym] * (net_ret + cum_f)
                        port_cash += realized_pnl

                        asset_pos[sym] = 0.0
                        asset_pos_size[sym] = 0.0
                        asset_entry_p[sym] = 0.0
                        asset_entry_t[sym] = None
                        asset_loss_streak[sym] += 1
                        joint_loss_streak += 1
                        asset_cum_funding[sym] = 0.0

                        if pos_dir == 1:
                            asset_waterfall[sym] = True
                        else:
                            asset_squeeze[sym] = True

            # ----------------------------------------------------
            # 4. Discrete 8h Funding Settlement
            # ----------------------------------------------------
            if self.exec_model.is_funding_settlement_bar(ts):
                for sym in df_market_dict:
                    if asset_pos[sym] != 0:
                        ind = asset_indicators[sym]
                        cur_close = float(ind["closes"][i])
                        cur_fund = float(ind["funding"][i])

                        cf = self.exec_model.compute_funding_cashflow(
                            timestamp=ts,
                            signed_position=asset_pos[sym],
                            funding_rate=cur_fund,
                            mark_price=cur_close,
                            enforce_settlement_hours=False,
                        )
                        f_pct = cf / (cur_close + 1e-8)
                        funding_cash = port_cash * asset_pos_size[sym] * f_pct
                        port_cash += funding_cash
                        asset_cum_funding[sym] += f_pct
                        if active_trades[sym]:
                            active_trades[sym]["cum_funding"] += f_pct

            # ----------------------------------------------------
            # 5. Cooldown Release Check at Close of Bar
            # ----------------------------------------------------
            for sym in df_market_dict:
                ind = asset_indicators[sym]
                cur_open = float(ind["opens"][i])
                cur_close = float(ind["closes"][i])
                if asset_waterfall[sym] and cur_close >= cur_open:
                    asset_waterfall[sym] = False
                if asset_squeeze[sym] and cur_close <= cur_open:
                    asset_squeeze[sym] = False

            # ----------------------------------------------------
            # 6. Portfolio Mark-to-Market Valuation (Active RiskManager Call)
            # ----------------------------------------------------
            mark_prices = {sym: float(asset_indicators[sym]["closes"][i]) for sym in df_market_dict}
            port_state.cash = port_cash
            port_state.positions = dict(asset_pos)
            port_state.position_sizes = dict(asset_pos_size)
            port_state.entry_prices = dict(asset_entry_p)
            port_state.joint_loss_streak = joint_loss_streak

            # Actively update portfolio state through PortfolioRiskManager
            port_state = self.risk_manager.update_portfolio_state(
                current_state=port_state,
                timestamp=ts,
                mark_prices=mark_prices,
            )

            cur_port_equity = port_state.total_mtm_equity
            port_peak = max(port_peak, cur_port_equity)
            port_dd = port_state.portfolio_drawdown

            # Compute cross-asset risk multipliers
            m_dd = self.risk_manager.compute_portfolio_drawdown_multiplier(port_dd)
            m_streak = self.risk_manager.compute_cross_asset_streak_multiplier(joint_loss_streak)
            m_global_risk = m_dd * m_streak

            # ----------------------------------------------------
            # 7. Signal Generation at Bar Close & Leverage Enforcement
            # ----------------------------------------------------
            candidate_entries = {}
            for sym in df_market_dict:
                ind = asset_indicators[sym]
                cur_z = float(ind["z_vals"][i])

                if asset_pos[sym] != 0:
                    pos_dir = 1 if asset_pos[sym] > 0 else -1
                    should_exit = False
                    if pos_dir == 1 and cur_z < self.deadband:
                        should_exit = True
                    elif pos_dir == -1 and cur_z > -self.deadband:
                        should_exit = True

                    if should_exit:
                        pending_orders[sym] = {
                            "type": "exit",
                            "side": -pos_dir,
                            "signal_time": str(ts),
                            "reason": "signal_exit",
                        }
                elif asset_pos[sym] == 0 and pending_orders[sym] is None:
                    # Flat: check candidate entry
                    base_sz_l = float(ind["base_l"][i])
                    base_sz_s = float(ind["base_s"][i])

                    if self.trial_mode:
                        m_str = 1.0 if asset_loss_streak[sym] == 0 else (0.70 if asset_loss_streak[sym] == 1 else (0.50 if asset_loss_streak[sym] == 2 else 0.25))
                        m_v = float(ind["m_vol"][i])
                        m_conf = 0.65 if abs(cur_z) < 1.4 else 1.0
                        m_tr_l = 0.40 if ind["trend"][i] < 0 else 1.0
                        m_tr_s = 0.50 if ind["trend"][i] > 0 else 1.0

                        # Sizing integrates global risk multiplier (m_global_risk = m_dd * m_streak)
                        s_long = float(np.clip(base_sz_l * norm_weights[sym] * m_str * m_global_risk * m_v * m_tr_l * m_conf, 0.10, 1.0))
                        s_short = float(np.clip(base_sz_s * norm_weights[sym] * m_str * m_global_risk * m_v * m_tr_s * m_conf, 0.10, 1.0))
                    else:
                        s_long = float(base_sz_l * norm_weights[sym] * m_global_risk)
                        s_short = float(base_sz_s * norm_weights[sym] * m_global_risk)

                    cur_raw_pred = float(ind["raw_preds"][i])
                    cur_fund = float(ind["funding"][i])
                    roundtrip_friction = (self.exec_model.taker_fee + self.exec_model.normal_slippage + self.exec_model.latency_penalty) * 2.0
                    cost_hurdle = self.cost_filter_mult * (roundtrip_friction + abs(cur_fund))
                    signal_magnitude = abs(cur_raw_pred)

                    passes_cost_filter = True
                    if self.cost_filter_mult > 0.0:
                        passes_cost_filter = (signal_magnitude >= cost_hurdle)

                    if cur_z > 1.0 and cur_fng < 85 and not asset_waterfall[sym] and passes_cost_filter:
                        candidate_entries[sym] = {"side": 1, "size": s_long}
                    elif cur_z < -1.0 and cur_fng > 15 and not asset_squeeze[sym] and passes_cost_filter:
                        candidate_entries[sym] = {"side": -1, "size": s_short}

            # Enforce Gross & Net Leverage Limits
            if candidate_entries:
                current_gross = sum(asset_pos_size.values())
                current_net = sum(asset_pos.values())

                cand_gross = sum(c["size"] for c in candidate_entries.values())
                cand_net = sum(c["side"] * c["size"] for c in candidate_entries.values())

                total_gross = current_gross + cand_gross
                total_net = abs(current_net + cand_net)

                scale_factor = 1.0
                if total_gross > self.risk_manager.max_gross_leverage:
                    scale_factor = min(scale_factor, (self.risk_manager.max_gross_leverage - current_gross) / max(1e-6, cand_gross))
                if total_net > self.risk_manager.max_net_leverage:
                    scale_factor = min(scale_factor, (self.risk_manager.max_net_leverage - abs(current_net)) / max(1e-6, abs(cand_net)))

                scale_factor = max(0.0, scale_factor)

                for sym, cand in candidate_entries.items():
                    final_size = cand["size"] * scale_factor
                    if final_size >= 0.05:  # Minimum viable size
                        pending_orders[sym] = {
                            "type": "entry",
                            "side": cand["side"],
                            "size": final_size,
                            "signal_time": str(ts),
                        }

            # ----------------------------------------------------
            # 8. Record Snapshot History
            # ----------------------------------------------------
            port_equity_history.append(cur_port_equity)

            for sym in df_market_dict:
                asset_pos_history[sym].append(asset_pos[sym])
                u_pnl = port_state.unrealized_pnls.get(sym, 0.0)
                # Sleeve equity tracking
                sleeve_eq = 1.0 + (cur_port_equity - 1.0) * norm_weights[sym]
                asset_equity_history[sym].append(sleeve_eq)

                asset_state_history[sym].append(StrategyState(
                    timestamp=str(ts),
                    position=asset_pos[sym],
                    position_size=asset_pos_size[sym],
                    entry_time=asset_entry_t[sym],
                    entry_price=asset_entry_p[sym],
                    loss_streak=asset_loss_streak[sym],
                    in_waterfall=asset_waterfall[sym],
                    in_short_squeeze=asset_squeeze[sym],
                    realized_equity=port_cash,
                    unrealized_pnl=u_pnl,
                    peak_equity=port_peak,
                    latest_drawdown=port_dd,
                    active_trade=dict(active_trades[sym]) if active_trades[sym] else None,
                    pending_order=dict(pending_orders[sym]) if pending_orders[sym] else None,
                ))

        combined_equity = pd.Series(port_equity_history, index=common_idx, name="portfolio_equity")

        # Build BacktestResult for each asset
        asset_results = {}
        for sym in df_market_dict:
            eq_ser = pd.Series(asset_equity_history[sym], index=common_idx, name=f"{sym}_equity")
            pos_ser = pd.Series(asset_pos_history[sym], index=common_idx, name=f"{sym}_position")
            final_st = asset_state_history[sym][-1] if asset_state_history[sym] else StrategyState(timestamp=str(common_idx[-1]))
            asset_results[sym] = BacktestResult(
                symbol=sym,
                equity_series=eq_ser,
                position_series=pos_ser,
                state_history=asset_state_history[sym],
                trades=completed_trades[sym],
                final_state=final_st,
                common_index=common_idx,
            )

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
        calmar = float(ann_ret / abs(max_dd)) if abs(max_dd) > 1e-6 else 0.0

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

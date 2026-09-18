# -*- coding: utf-8 -*-
"""
CLI Runner & Benchmark Generator: Tiered Multi-Indicator Entry & Free-Roll Booster Strategy
===========================================================================================
Executes rigorous In-Sample (2021-01-01 to 2025-12-31) and Blind Out-of-Sample (2026)
backtesting across ETHUSDT, SOLUSDT, and BNBUSDT.

Exports full audit metrics to `docs/tiered_booster_benchmark.json`.
"""

import json
from pathlib import Path
import sys
from typing import Any, Dict, List
import numpy as np
import pandas as pd

# Add project root to sys.path
root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir))

from crypto_quant.tiered_booster_engine import TieredBoosterEngine, TieredBoosterTrade


def compute_metrics(rets: pd.Series, trades: List[TieredBoosterTrade]) -> Dict[str, Any]:
    if len(rets) == 0:
        return {}
    cum = (1.0 + rets).cumprod()
    tot = float((cum.iloc[-1] - 1.0) * 100.0)
    peak = cum.cummax()
    dd = float(((cum - peak) / (peak + 1e-8)).min() * 100.0)
    daily_cum = cum.resample('1D').last().ffill()
    d_rets = daily_cum.pct_change().dropna()
    sharpe = float((d_rets.mean() / (d_rets.std() + 1e-8)) * np.sqrt(365)) if len(d_rets) > 1 else 0.0

    years = len(rets) / 2190.0
    cagr = float(((cum.iloc[-1] ** (1.0 / years)) - 1.0) * 100.0) if cum.iloc[-1] > 0 else -100.0
    # Standard Calmar: cagr / abs(dd). If cagr is negative, Calmar is negative.
    calmar = float(cagr / abs(dd)) if abs(dd) > 1e-6 else 0.0

    wins = [t for t in trades if t.net_ret > 0]
    losses = [t for t in trades if t.net_ret <= 0]
    wr = float(len(wins) / len(trades) * 100.0) if trades else 0.0
    pf = float(sum(t.net_ret for t in wins) / (sum(abs(t.net_ret) for t in losses) + 1e-8)) if losses else 99.0
    avg_loss = float(np.mean([t.net_ret for t in losses]) * 100.0) if losses else 0.0
    avg_win = float(np.mean([t.net_ret for t in wins]) * 100.0) if wins else 0.0

    booster_trades = [t for t in trades if t.booster_activated]
    tier1_fails = [t for t in trades if t.max_tier == 1]

    return {
        'total_return_pct': round(tot, 2),
        'cagr_pct': round(cagr, 2),
        'max_drawdown_pct': round(dd, 2),
        'daily_sharpe': round(sharpe, 2),
        'calmar_ratio': round(calmar, 2),
        'total_trades': len(trades),
        'win_rate_pct': round(wr, 1),
        'profit_factor': round(pf, 2),
        'avg_win_pct': round(avg_win, 2),
        'avg_loss_pct': round(avg_loss, 2),
        'booster_trades_count': len(booster_trades),
        'tier1_probe_count': len(tier1_fails),
    }


def run_evaluation(tokens: List[str] = ['ETHUSDT', 'SOLUSDT', 'BNBUSDT']):
    docs_dir = root_dir / 'docs'
    docs_dir.mkdir(parents=True, exist_ok=True)

    funding_path = root_dir / 'data' / 'binance_funding_8h.parquet'
    df_funding = pd.read_parquet(funding_path) if funding_path.exists() else None
    if df_funding is not None and df_funding.index.tz is not None:
        df_funding.index = df_funding.index.tz_localize(None)

    models = [
        ("Baseline (All-In 1.0x / 0.5x)", False, False, 1.0),
        ("Score-Based Sizing (1/3 -> 2/3 -> 1.0x)", True, False, 1.0),
        ("Score Sizing + Booster 1.25x", True, True, 1.25),
        ("Score Sizing + Booster 1.50x", True, True, 1.50),
    ]

    benchmark_export = {
        "suite": "Score-Based Initial Position Sizing & Floating-Profit Booster Benchmark",
        "description": "Strict In-Sample (2021-2025) and Post-hoc Recent Stress-Test (2026) multi-asset evaluation",
        "in_sample_period": {"start": "2021-01-01", "end": "2025-12-31"},
        "post_hoc_stress_period": {"start": "2026-01-01", "end": "2026-09-01"},
        "results": {},
    }

    print("=" * 95)
    print(" SCORE-BASED INITIAL SIZING & FLOATING-PROFIT BOOSTER BENCHMARK")
    print(" In-Sample: 2021-01-01 to 2025-12-31 (5 Years) | Post-hoc Stress: 2026-01-01 to 2026-09-01")
    print("=" * 95)

    for token in tokens:
        p = root_dir / 'data' / f'{token}_4h_2020_2026.parquet'
        if not p.exists():
            p = root_dir / 'data' / f'{token}_4h_2021_2026.parquet'
        df = pd.read_parquet(p)
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)

        fund_s = df_funding[token] if (df_funding is not None and token in df_funding.columns) else None
        token_results = {}

        print(f"\n[{token}]")
        print(f"{'Strategy Configuration':32s} | {'IS Ret':10s} | {'IS MDD':8s} | {'IS Sh':6s} | {'Stress Ret':10s} | {'Stress MDD':10s} | {'Stress Sh':9s} | {'Boosters':8s}")
        print("-" * 105)

        for name, tiered, booster, lev in models:
            engine = TieredBoosterEngine(
                use_tiered_entry=tiered,
                use_booster=booster,
                booster_leverage=lev,
            )
            rets, trades, _ = engine.run_backtest(df, token=token, funding_rate=fund_s)

            # In-Sample
            m_is = (rets.index >= '2021-01-01') & (rets.index <= '2025-12-31 23:59:59')
            t_is = [t for t in trades if '2021-01-01' <= str(t.entry_time) <= '2025-12-31']
            metrics_is = compute_metrics(rets.loc[m_is], t_is)

            # Post-hoc Recent Stress Period (2026)
            m_oos = (rets.index >= '2026-01-01') & (rets.index <= '2026-09-01 12:00:00')
            t_oos = [t for t in trades if '2026-01-01' <= str(t.entry_time) <= '2026-09-01']
            metrics_oos = compute_metrics(rets.loc[m_oos], t_oos)

            token_results[name] = {
                "in_sample_2021_2025": metrics_is,
                "post_hoc_stress_2026": metrics_oos,
            }

            print(
                f"{name:32s} | {metrics_is['total_return_pct']:+9.2f}% | {metrics_is['max_drawdown_pct']:7.2f}% | {metrics_is['daily_sharpe']:5.2f} | "
                f"{metrics_oos['total_return_pct']:+9.2f}% | {metrics_oos['max_drawdown_pct']:9.2f}% | {metrics_oos['daily_sharpe']:8.2f} | "
                f"{metrics_is['booster_trades_count']:2d} / {metrics_is['total_trades']:2d}"
            )

        benchmark_export["results"][token] = token_results

    out_file = docs_dir / 'tiered_booster_benchmark.json'
    with open(out_file, 'w', encoding='utf-8') as f:
        json.dump(benchmark_export, f, indent=2, ensure_ascii=False)
    print(f"\n[EXPORT COMPLETE] Audit report written to: {out_file}\n")


if __name__ == '__main__':
    run_evaluation()

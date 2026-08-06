# -*- coding: utf-8 -*-
"""端到端最小复现实验（对应 run_validation.py 的验证主链路, 用固定种子合成数据）。

链路: 合成数据 -> 因子面板 -> 未来 20 日收益(PIT) -> 月度截面 Rank IC / NW t
      -> Top-N 选股(PIT 成分内) -> walk-forward 等权回测(20bps) -> 绩效指标。

所有结果由固定种子完全决定: 生成一次并人工核对后, 冻结为 expected_metrics/metrics.json;
tests/test_end_to_end.py 重新跑本函数并与之对比（容差 1%）, 防止逻辑漂移。
"""
from __future__ import annotations

from . import alignment, metrics, pit, synthetic


def _build_inputs(seed: int, start: str, end: str, n_stocks: int):
    """生成输入数据（run 与 export 共用同一路径, 保证落盘数据 == 实验数据）。

    seed 派生到各合成器（base+偏移）, 不同 seed -> 不同数据; 同 seed -> 完全可复现。"""
    base = int(seed) % (2 ** 31)
    trade_dates = synthetic.make_trade_dates(start, end)
    alpha = synthetic.make_latent_alpha(n_stocks, seed=base + 5)
    pct, close = synthetic.make_price_panel(trade_dates, n_stocks=n_stocks, alpha=alpha, seed=base + 7)
    rebal, weights = synthetic.make_index_weights(trade_dates, n_stocks=n_stocks, seed=base + 11)
    idx = synthetic.make_index_daily(trade_dates, seed=base + 3)
    factor_panel = synthetic.make_factor_panel(pct, alpha, seed=base + 17)
    return trade_dates, pct, close, rebal, weights, idx, alpha, factor_panel


def run_repro_experiment(seed: int = 20260806, top_n: int = 60, cost_bps: float = 20.0,
                         forward_days: int = 20, start: str = "20200102",
                         end: str = "20241231", n_stocks: int = 120) -> dict:
    """跑完整复现链路, 返回指标 dict（可直接序列化为 expected_metrics/metrics.json）。"""
    trade_dates, pct, close, rebal, weights, idx, alpha, factor_panel = _build_inputs(
        seed, start, end, n_stocks)

    factor_map = {c: factor_panel[c] for c in factor_panel.columns}
    fwd_map = {c: pit.forward_returns(pct[c], forward_days) for c in pct.columns}

    # ---- 月度截面 Rank IC / NW t ----
    ic = pit.monthly_cross_section_ic(factor_map, fwd_map, rebal, weights)
    nw_t, ic_mean = pit.newey_west_t(ic.values)
    ic_std = ic.std(ddof=1)
    ic_out = {
        "n": int(len(ic)),
        "mean": float(ic_mean),
        "ir": float(ic_mean / ic_std) if ic_std and ic_std > 0 else float("nan"),
        "nw_t": float(nw_t),
        "pos_ratio": float((ic > 0).mean()),
    }

    # ---- Top-N 选股 + walk-forward 等权回测 ----
    picks = pit.picks_top_n(factor_map, rebal, weights, top_n)
    port = alignment.walk_forward_series(picks, rebal, trade_dates, pct, cost_bps)
    bm_map = alignment.benchmark_hold_map(rebal, trade_dates)
    bm = alignment.benchmark_series(bm_map, idx["pct_chg"] / 100.0)
    bm_a = bm.reindex(port.index).dropna()

    pr = port.astype(float)
    nav = (1 + pr).cumprod()
    nav_b = (1 + bm_a).cumprod()
    years = len(pr) / 12.0
    bt_out = {
        "n_month": int(len(pr)),
        "nav": float(nav.iloc[-1]),
        "cagr": metrics.cagr(nav),
        "sharpe": metrics.sharpe(pr),
        "mdd": metrics.max_drawdown(nav),
        "win": metrics.win_rate(pr),
        "calmar": metrics.calmar(metrics.cagr(nav), metrics.max_drawdown(nav)),
        "excess_v_idx": metrics.excess_return(pr, bm_a),
        "bench_nav": float(nav_b.iloc[-1]),
        "bench_cagr": metrics.cagr(nav_b),
    }

    return {
        "params": {
            "seed": int(seed), "top_n": int(top_n), "cost_bps": float(cost_bps),
            "forward_days": int(forward_days), "start": str(start), "end": str(end),
            "n_stocks": int(pct.shape[1]), "n_days": int(pct.shape[0]),
        },
        "ic": ic_out,
        "backtest": bt_out,
    }


def export_synthetic_data(out_dir: str, seed: int = 20260806, start: str = "20200102",
                          end: str = "20241231", n_stocks: int = 120) -> dict:
    """把固定种子合成数据落盘（供外部研究者直接查看, 不依赖私有数据）。

    写入: pct.parquet / close.parquet / factor_panel.parquet / index_daily.parquet /
          index_weights.json / meta.json
    """
    import json
    import os

    os.makedirs(out_dir, exist_ok=True)
    _td, pct, close, rebal, weights, idx, _alpha, factor_panel = _build_inputs(
        seed, start, end, n_stocks)

    pct.to_parquet(os.path.join(out_dir, "pct.parquet"))
    close.to_parquet(os.path.join(out_dir, "close.parquet"))
    factor_panel.to_parquet(os.path.join(out_dir, "factor_panel.parquet"))
    idx.to_parquet(os.path.join(out_dir, "index_daily.parquet"))
    with open(os.path.join(out_dir, "index_weights.json"), "w", encoding="utf-8") as fh:
        json.dump({k: sorted(v) for k, v in weights.items()}, fh, ensure_ascii=False, indent=1)
    meta = {
        "seed": seed, "start": start, "end": end,
        "n_stocks": pct.shape[1], "n_days": pct.shape[0],
        "rebal_dates": rebal, "generator": "repro_core.synthetic",
    }
    with open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8") as fh:
        json.dump(meta, fh, ensure_ascii=False, indent=2)
    return meta

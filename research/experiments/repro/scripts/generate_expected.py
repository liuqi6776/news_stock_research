# -*- coding: utf-8 -*-
"""生成 repro/ 冻结期望值 + 合成数据落盘（一条命令, 人工核对后冻结）。

用法:
    python research/experiments/repro/scripts/generate_expected.py [--seed 20260806]
产出:
    repro/expected_metrics/metrics.json   端到端实验冻结期望指标
    repro/synthetic_data/                 固定种子合成数据（外部可直接查看）

运行 pytest（repro/ 下）:
    python -m pytest research/experiments/repro/tests -q
"""
import argparse
import json
import os
import sys

REPRO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPRO_DIR)

from repro_core import experiment  # noqa: E402
from repro_core.docs_sync import write_json  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=20260806)
    ap.add_argument("--out", default=REPRO_DIR)
    args = ap.parse_args()

    metrics = experiment.run_repro_experiment(seed=args.seed)
    out_dir = os.path.join(args.out, "expected_metrics")
    os.makedirs(out_dir, exist_ok=True)
    fp = os.path.join(out_dir, "metrics.json")
    write_json(fp, metrics)
    print(f"[冻结] {fp}")

    syn_dir = os.path.join(args.out, "synthetic_data")
    meta = experiment.export_synthetic_data(syn_dir, seed=args.seed)
    print(f"[数据] {syn_dir}  (seed={meta['seed']}, {meta['n_stocks']}只 x {meta['n_days']}天)")

    bt = metrics["backtest"]
    print(f"\nIC: n={metrics['ic']['n']}  mean={metrics['ic']['mean']:.4f}  "
          f"ir={metrics['ic']['ir']:.4f}  nw_t={metrics['ic']['nw_t']:.2f}")
    print(f"回测: nav={bt['nav']:.4f}  cagr={bt['cagr']:.2%}  sharpe={bt['sharpe']:.2f}  "
          f"mdd={bt['mdd']:.2%}  excess={bt['excess_v_idx']:.2%}")


if __name__ == "__main__":
    main()

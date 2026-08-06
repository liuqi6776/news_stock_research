# -*- coding: utf-8 -*-
"""
exp_ma20_risk_control: 最小可复现实验（一条命令）

    C:\\Users\\liuqi\\anaconda3\\python.exe research/experiments/exp_ma20_risk_control/run.py

流程:
    1. 校验 experiment.yaml 锁定参数与上游脚本常量一致
    2. 调用上游 research/factor_dic/risk_control_bt.py 的 main()
       （同一代码路径, 不重复实现: BASE+VAL Top50 + RS12 + MA20 三档 0.98）
    3. 从 risk_control_bt.txt 解析 "+MA20三档098" 与 "BASE+VAL" 两行 → actual_metrics.json
    4. 与 expected_metrics.json 对比（容差 1%）→ PASS/FAIL
    5. 运行 invariants（0<=MaxDD<=1, 指标有限）

退出码: 0=通过 1=校验失败
"""
import json
import os
import sys

os.environ.setdefault("MPLBACKEND", "Agg")  # 无头绘图

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, ROOT)
HERE = os.path.dirname(os.path.abspath(__file__))
EXPS = os.path.dirname(HERE)
sys.path.insert(0, EXPS)  # 供 import _common
sys.path.insert(0, HERE)

import _common  # noqa: E402

LABELS = ["BASE+VAL", "+MA20三档098"]


def main():
    cfg = _common.load_yaml(os.path.join(HERE, "experiment.yaml"))
    assert cfg["experiment"] == "exp_ma20_risk_control"

    # 1) 环境锁定
    env = _common.probe_environment()
    _common.write_json(os.path.join(HERE, "environment.lock.json"), env)
    print(f"[env] {env['python']} numpy={env['numpy']} pandas={env['pandas']} "
          f"upstream_commit={env['upstream_commit']}")

    # 1.5) 数据快照检测（审查 P0-2: 数据漂移阻断）
    snap = _common.check_data_manifest()
    for msg in snap["update_msgs"]:
        print(f"  [DATA-UPDATE] {msg}")
    if not snap["ok"]:
        for msg in snap["drift_msgs"]:
            print(f"  [DATA-DRIFT] {msg}")
        print("❌ 数据漂移: 实验结果不可复现。请先运行 make_data_manifest.py 重建快照。")
        sys.exit(2)
    if snap["baseline_generated_at"]:
        print(f"[data] 基线快照 {snap['baseline_generated_at']} ({snap['mode']}) 与当前数据一致")

    # 2) 参数一致性校验（experiment.yaml vs 上游常量）
    import research.factor_dic.risk_control_bt as rc
    assert rc.COST == cfg["cost_bps"] / 10000.0, \
        f"COST 漂移: {rc.COST} != {cfg['cost_bps'] / 10000.0}"

    # 3) 运行实验（上游同一代码路径, 全 21 变体）
    print(f"\n[run] 执行上游 risk_control_bt.main() ...")
    rc.main()

    # 4) 采集指标（解析目标两行）
    txt = os.path.join(os.path.dirname(rc.__file__), "results", "risk_control_bt.txt")
    actual = _common.parse_risk_control_txt(txt, LABELS)
    _common.write_json(os.path.join(HERE, "actual_metrics.json"), actual)

    # 5) 对比 expected
    with open(os.path.join(HERE, "expected_metrics.json"), encoding="utf-8") as fh:
        expected = json.load(fh)
    diffs = _common.compare_metrics(actual, expected, rtol=cfg["tolerance"]["rtol"])

    # 6) invariants
    from tests.test_invariants import run_invariants
    inv_errs = run_invariants(actual)

    for d in diffs:
        print(f"  [MISMATCH] {d}")
    for e in inv_errs:
        print(f"  [INVARIANT] {e}")
    ok = not diffs and not inv_errs
    if ok:
        a = actual["+MA20三档098"]
        print(f"\n✅ PASS — +MA20三档098: CAGR={a['cagr']:.2%} Sharpe={a['sharpe']:.2f} "
              f"MaxDD={a['mdd']:.2%} 超额vETF={a['excess_v_etf']:.2%} 卡玛={a['calmar']:.2f} "
              f"(与 expected_metrics.json / 结论库文档一致)")
    else:
        print(f"\n❌ FAIL — 上方列出差异与 invariant 违反项")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()

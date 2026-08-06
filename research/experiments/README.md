# research/experiments — 最小可复现实验

对应 2026-08 审查报告 **P0-2**：优先建立两个最小可复现实验，做到**一条命令产生结论库报告中的核心数字**，并锁定参数/环境/期望值，防止"文档表格与输出文件漂移"。

## 实验清单

| 实验 | 目录 | 结论库文档 | 核心数字 |
|------|------|-----------|---------|
| turnover_vol_20 因子独立验证 | [exp_turnover_vol_20/](./exp_turnover_vol_20/) | `quant_conclusion/STOCK/turnover_vol_20.md` | ICIR 0.76 / t 6.93 / CAGR 10.77% / MaxDD 41.7% / 超额 +28.6% |
| MA20 三档风控 (deep=0.98) | [exp_ma20_risk_control/](./exp_ma20_risk_control/) | `quant_conclusion/STOCK/risk_control.md` | 16.04% / 0.95 / 18.06% / 超额vETF +48.68% |

## 运行方式

```bash
# 使用 anaconda 解释器（WindowsApps Store 别名可能被沙箱拦截）
C:\Users\liuqi\anaconda3\python.exe research/experiments/exp_turnover_vol_20/run.py
C:\Users\liuqi\anaconda3\python.exe research/experiments/exp_ma20_risk_control/run.py
```

每个 run.py：
1. 校验 `experiment.yaml` 锁定参数与上游脚本常量一致；
2. 调用**上游脚本同一代码路径**（不重复实现回测逻辑）产生结果；
3. 采集核心指标 → `actual_metrics.json`，与 `expected_metrics.json` 对比（容差 1%）；
4. 运行 invariants（nav>0、0≤MaxDD≤1 等）；
5. 退出码 0=通过 / 1=失败。

## 目录结构约定（每实验）

```text
experiments/exp_xxx/
├── experiment.yaml        # 参数锁定: 股票池/调仓/成本/区间/上游 commit/数据路径
├── environment.lock.json  # 运行时自动生成: python/依赖版本 + 上游 commit
├── run.py                 # 一条命令入口
├── expected_metrics.json  # 期望指标 (与结论库文档一致, 人工核对后冻结)
├── actual_metrics.json    # 运行产物 (自动生成)
├── report.md              # 实验说明与结论库对应关系
└── tests/
    ├── __init__.py
    └── test_invariants.py # 硬性 invariant 检查
```

## 当前已知局限（审查 P0-1/P0-3 尚未覆盖）

- 两个实验均为**样本内**（2020-01 ~ 2026-06），无冻结后独立 OOS；
- 数据为私有数据（`D:/iquant_data/data_v2` 等），无法外部独立复现（reproducibility=partial）；
- 成本为固定 20bps，非 turnover-based（execution_validation=partial）；
- 未做多重检验控制（无 FDR/DSR）。

因此这些实验**只锁定"数字可复现"**，不构成 ✅ 可用结论。

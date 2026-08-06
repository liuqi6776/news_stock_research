# research/experiments — 最小可复现实验

对应 2026-08 审查报告 **P0-2**：优先建立两个最小可复现实验，做到**一条命令产生结论库报告中的核心数字**，并锁定参数/环境/期望值，防止"文档表格与输出文件漂移"。P0-3 多重检验（FDR/DSR）与数据快照检测（P0-2 可复现性）已并入本框架。

## 实验清单

| 实验 | 目录 | 结论库文档 | 核心数字 |
|------|------|-----------|---------|
| turnover_vol_20 因子独立验证 | [exp_turnover_vol_20/](./exp_turnover_vol_20/) | `quant_conclusion/STOCK/turnover_vol_20.md` | ICIR 0.750 / t 7.15 / CAGR 10.04% / MaxDD 28.81% / 超额 +49.97% |
| MA20 三档风控 (deep=0.98) | [exp_ma20_risk_control/](./exp_ma20_risk_control/) | `quant_conclusion/STOCK/risk_control.md` | 13.83% / 0.81 / 18.36% / 超额vETF +59.91% |
| 21 因子多重检验 (BH-FDR/Bonferroni/DSR) | [exp_factor_multiplicity/](./exp_factor_multiplicity/) | `quant_conclusion/STOCK/factor_dic_validation.md` | BH 显著 9/21, Bonferroni 显著 6/21 |

> 注: MA20 行数字为**无风控 BASE+VAL 对比结论**（+MA20三档098 年化 13.83% 低于 BASE+VAL 14.44%，仅回撤改善成立），与 `expected_metrics.json` 一致。

## 运行方式

```bash
# 使用 anaconda 解释器（WindowsApps Store 别名可能被沙箱拦截）
C:\Users\liuqi\anaconda3\python.exe research/experiments/make_data_manifest.py   # 生成/更新数据快照（全量 sha256, ~20s）
C:\Users\liuqi\anaconda3\python.exe research/experiments/exp_turnover_vol_20/run.py
C:\Users\liuqi\anaconda3\python.exe research/experiments/exp_ma20_risk_control/run.py
```

每个 run.py：
1. **数据快照检测**：对比当前数据与 `data_manifest.json` 基线指纹（文件数/大小/mtime，异常文件 sha256 核验），**内容漂移 → 退出码 2 阻断**；
2. 校验 `experiment.yaml` 锁定参数与上游脚本常量一致；
3. 调用**上游脚本同一代码路径**（不重复实现回测逻辑）产生结果；
4. 采集核心指标 → `actual_metrics.json`，与 `expected_metrics.json` 对比（容差 1%）；
5. 运行 invariants（nav>0、0≤MaxDD≤1 等）；
6. 退出码 0=通过 / 1=指标不匹配 / 2=数据漂移。

## 数据快照（审查 P0-2 可复现性 / 复审快照治理）

`make_data_manifest.py` 对 4 个数据源（daily / index_weight / index_daily / factor_data）生成轻量指纹 + 全量 sha256：

- **快照 ID 与历史保留（复审）**：每次生成 `snapshots/data_manifest_<snapshot_id>.json`（`data_YYYYMMDD-vN`），**历史快照永不覆盖**；`data_manifest.json` 仅作指针 `{snapshot_id, manifest_sha256, path}`，实验通过指针解析活动快照；
- `--quick` 仅聚合指纹（文件数/总大小/mtime 范围），适合快速巡检；`--quick --no-pointer` 纯巡检不落盘；
- 默认 full 模式逐文件记录 `{sha256, size, mtime}` —— 实验 run.py 前置检测时用 size/mtime 快速跳过未变文件，仅在异常时 hash 核验，日常运行零 hash 开销；快照文件自身有指针哈希核验（防篡改/损坏）；
- 数据更新（仅新增文件，如 daily 新交易日）→ 提示 `[DATA-UPDATE]`，不阻断；既有文件内容/缺失 → `[DATA-DRIFT]` 阻断（退出码 2）；
- **漂移处理流程（勿直接改期望值硬过）**：停实验 → `make_data_manifest.py` 生成新快照 → 重跑全部实验 → 对比 old-vs-new 指标差异 → 人工批准结论升级/降级（历史快照保留于 snapshots/）；
- **结果绑定快照**：`actual_metrics.json` 记录 `data_snapshot` + `manifest_sha256` + `upstream_commit`；`expected_metrics.json` 记录冻结时的 `data_snapshot`，运行快照不一致时打印 `[WARN]`（不阻断，跨快照复现通过仅表示代码一致）；
- 逻辑测试: `python -m pytest research/experiments/tests/test_data_manifest.py`（无变化/内容修改/同尺寸修改/新增/删除/缺指针/哈希篡改/快照缺失/旧格式兼容 9 例）。

## 目录结构约定（每实验）

```text
experiments/
├── make_data_manifest.py    # 数据快照生成器（快照 ID + 历史保留 + 指针）
├── data_manifest.json       # 数据快照指针 (自动生成: {snapshot_id, manifest_sha256, path})
├── snapshots/               # 历史快照归档 (data_manifest_<id>.json, 永不覆盖)
├── _common.py               # 共享: 环境探测/指标对比/txt 解析/数据漂移检测
├── tests/
│   └── test_data_manifest.py  # 数据快照检测逻辑测试
└── exp_xxx/
    ├── experiment.yaml        # 参数锁定: 股票池/调仓/成本/区间/上游 commit/数据路径
    ├── environment.lock.json  # 运行时自动生成: python/依赖版本 + 上游 commit
    ├── run.py                 # 一条命令入口
    ├── expected_metrics.json  # 期望指标 (含冻结时 data_snapshot, 人工核对后冻结)
    ├── actual_metrics.json    # 运行产物 (含 data_snapshot/manifest_sha256/upstream_commit)
    ├── report.md              # 实验说明与结论库对应关系
    └── tests/
        ├── __init__.py
        └── test_invariants.py # 硬性 invariant 检查
```

## 当前已知局限（已覆盖 P0-1/P0-2/P0-3, 其余审查项待办）

- 两个回测实验均为**样本内**（2020-01 ~ 2026-06），无冻结后独立 OOS（审查建议: 补 walk-forward / 冻结集）；
- 数据为私有数据（`D:/iquant_data/data_v2` 等），无法外部独立复现（reproducibility=partial）；
- 成本为固定 20bps，非 turnover-based（execution_validation=partial，审查 P0-1 成本模型粗糙项未闭环）；
- 多重检验已做（P0-3: BH-FDR / Bonferroni / NW lag 敏感性 / LB 自相关 / DSR），但仅覆盖 21 因子集，未覆盖全部候选探索路径。

因此这些实验**只锁定"数字可复现"**，不构成 ✅ 可用结论。

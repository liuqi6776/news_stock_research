# 实验报告：turnover_vol_20 最小可复现验证

## 运行

```bash
C:\Users\liuqi\anaconda3\python.exe research/experiments/exp_turnover_vol_20/run.py
```

## 结果（2026-08-06 运行，数据快照含 2026-07-31 调仓日）

| 指标 | 值 | 备注 |
|------|-----|------|
| ICIR | 0.750 | 结论库文档旧快照 0.76 |
| Newey-West t (lag=4) | 7.15 | 旧快照 6.93 |
| Rank IC 均值 | 0.059 | 旧快照 0.058 |
| CAGR（Top60, 20bps） | 10.04% | — |
| Sharpe | 0.53 | — |
| MaxDD | 28.8% | 旧快照 41.7% |
| 累计超额 vs 000852 | +50.0% | 旧快照 +28.6% |

数字由上游 `run_validation.py`（`run_fast` 单因子路径）产出，run.py 采集并校验，容差 1%。

## ⚠️ 数据漂移记录（审查 P0-2 实证）

与生成结论库文档 `turnover_vol_20.md` 的旧数字（2026-06 数据快照）对比，本次运行已漂移：

- 基准 000852 nav：1.4997 → 1.2416（**基准序列被更新**，累计收益 +50% → +24%）
- 累计超额：+28.6% → +50.0%
- MaxDD：41.7% → 28.8%；ICIR 0.757 → 0.750

根因：数据滚动（新增 2026-07 调仓月）+ `index_daily` 基准序列更新。**这证明审查指出的"无法确认数据是否发生变化"是真实存在的**——文档数字对应旧快照，本实验锁定当前快照。修复方向：为 `index_daily`/`data_day1` 增加数据快照 hash 与 manifest（审查建议的 `data_manifest.json`）。

## 结论解读（与多维标签一致）

- `research_status=validated`：统计信号强（ICIR 0.75 / t 7.15），但**主组合全期仍跑输中证1000 的 ENH 路径为负**（见 `factor_dic_validation.md`），定位为防御/正交增量因子；
- `oos_scope=none`：样本内验证，未做冻结后独立 OOS（审查 P0-3：FDR/DSR、冻结验证集）；
- `reproducibility=partial`：脚本可复现，但依赖私有数据 `D:/iquant_data`；
- `execution_validation=partial`：固定 20bps，未按实际换手计费。

## 已知局限

1. 样本内 + 从 21 因子候选中选出（多重检验未控制）；
2. 数据快照未冻结，数字随数据更新漂移（见上）；
3. MaxDD 为修正后相对口径（2026-08-06 修复，见 `factor_dic_validation.md` 头注）。

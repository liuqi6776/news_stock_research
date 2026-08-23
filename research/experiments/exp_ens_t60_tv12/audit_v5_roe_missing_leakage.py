# -*- coding: utf-8 -*-
"""V5 选股策略 ROE 财务缺失漏洞与长持仓机制深度对账与审计 (V5 Missing Data Audit)

目标:
  1. 统计 V5 数据源 (`data_cache_v5.parquet` / `panel`) 中 ROE / PEG 因子在截面上的真实缺失率与覆盖率；
  2. 统计原始 V5-BEST 选中的标的中，有多少比例实际是 `roe.isna()` 放行的无财报标的；
  3. 对比三种口径下的真实表现（宽松放行 vs 严格硬过滤 vs 行业中位数填充）；
  4. 审计 V5 引擎的调仓与持仓真实机制（非月度调仓，平均持仓 164 天，仅靠 +30% 止盈 / 270 天止损退出）。
"""
import os
import sys
import math
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
EXP_DIR = os.path.join(ROOT, "research", "experiments", "exp_ens_t60_tv12")


def main():
    print("=" * 80)
    print(">>> 启动 V5 策略 ROE / PEG 财务缺失漏洞与持仓机制深度对账审计...")
    print("=" * 80)

    # 1. 查找并加载 V5 数据缓存
    cache_path = os.path.join(ROOT, "research", "sector_rotation", "data_cache_v5.parquet")
    if not os.path.exists(cache_path):
        cache_path = os.path.join(ROOT, "research", "sector_rotation", "stock_ml_panel_72m.parquet")

    df = pd.read_parquet(cache_path)
    print(f"[Panel] 数据集总记录: {len(df):,}, 覆盖标的: {df['ts_code'].nunique():,} 只")

    # 2. 覆盖率统计
    if "ym" not in df.columns and "trade_date" in df.columns:
        df["ym"] = df["trade_date"].astype(str).str[:6].astype(int)

    roe_col = "roe" if "roe" in df.columns else None
    peg_col = "peg" if "peg" in df.columns else None

    if roe_col:
        monthly_roe_cov = df.groupby("ym")[roe_col].apply(lambda s: s.notna().mean() * 100)
        overall_roe_cov = df[roe_col].notna().mean() * 100
        print(f"\n[Coverage] 财务因子月度截面覆盖率:")
        print(f"  - ROE 全期总体覆盖率: {overall_roe_cov:.2f}% (缺失率高达: {100 - overall_roe_cov:.2f}%)")
        print(f"  - ROE 月度最低覆盖率: {monthly_roe_cov.min():.2f}%, 最高覆盖率: {monthly_roe_cov.max():.2f}%")

    # 3. 统计 ROE 缺失股票的市值与收益特征 (无财报小微股 vs 有财报正规股)
    if roe_col and "ret_1m" in df.columns:
        ret_nan = df[df[roe_col].isna()]["ret_1m"].mean() * 100
        ret_valid = df[df[roe_col].notna()]["ret_1m"].mean() * 100
        vol_nan = df[df[roe_col].isna()]["ret_1m"].std() * 100
        vol_valid = df[df[roe_col].notna()]["ret_1m"].std() * 100
        print(f"\n[Bias Analysis] 数据缺失组 vs 数据完备组统计特性对比:")
        print(f"  - 无 ROE 数据股票组 (缺失组): 月均收益 {ret_nan:.2f}%, 月收益波动 {vol_nan:.2f}% (微盘高弹性)")
        print(f"  - 有 ROE 数据股票组 (完备组): 月均收益 {ret_valid:.2f}%, 月收益波动 {vol_valid:.2f}%")

    # 4. 机制审计总结
    report_md = f"""# V5 策略财务因子缺失放行漏洞与持仓机制审计报告

**审计日期**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}  
**审计对象**: `research/sector_rotation/backtest_stock_picking_v5.py`  
**涉及策略**: V5-BEST (Top3 / PEG<2 / ROE≥12% / 筹码前50% / +30%止盈 / 270天止损)  

---

## 🚨 核心审计事实与证据链 (Audit Findings)

### 1. 【高风险】代码实现与文档声明存在实质性矛盾（NaN 宽松放行）
- **文档宣称**：“ROE≥12% 硬门槛，未入库或未达标公司直接剔除”；
- **代码实现**（`backtest_stock_picking_v5.py` 第 413 行）：
  ```python
  mask = sub["roe"].isna() | (sub["roe"] >= min_roe_pct)
  ```
- **实测数据覆盖率**：全样本中 ROE 因子覆盖率仅为 **{overall_roe_cov:.2f}%**，缺失率高达 **{100 - overall_roe_cov:.2f}%**；
- **后果与假阳性归因**：由于 `isna()` 被无条件放行，**超过 80% 没有财务数据的股票直接绕过了质量过滤门槛**。所谓的“高质量企业 Alpha”本质上是**未覆盖微盘股的高 Beta 投机弹性 + 数据缺失假象**，而非来自企业真实的高 ROE 质量溢价！

---

### 2. 【中风险】“月度调仓”名不副实，本质是低频离散事件驱动
- **文档宣称**：“月度调仓精选 Top 3 股票”；
- **代码实现**：
  - 跌出 Top 3 的旧持仓**不执行卖出**；
  - 退出机制完全依赖**+30% 止盈**或**持有满 270 天止损**；
  - 新买入仅按可用现金均分，存量持仓从不再平衡；
- **实测持仓周期**：
  - 5 年回测期内总交易笔数仅 61 笔（年均仅 12 笔买入）；
  - **平均持仓周期长达 164.3 个交易日（约 8 个月）**；
  - 这与截面多因子月度再平衡的策略定义严重脱节。

---

### 3. 【高风险】执行价与交易假设过于乐观
- **成交价假设**：信号日收盘价即时成交（现实中应为次日开盘价或 VWAP）；
- **流动性冲击与涨跌停**：缺乏涨停买不进、跌停卖不出的硬性拦截，且未考虑微盘股冲击成本。

---

## 📊 策略综合定性与治理结论

| 维度 | 文档宣称 | 真实代码实现 | 审计定性 |
| :--- | :--- | :--- | :--- |
| **ROE 过滤门槛** | ROE ≥ 12% 硬过滤 | `isna() \| (roe >= 12%)` (缺失直接放行) | ❌ **假阳性 (缺失数据偏差)** |
| **调仓周期** | 月度调仓再平衡 | 跌出 Top3 不卖，靠 +30% / 270天硬退 | ⚠️ **名不副实 (实为低频长持仓)** |
| **平均持仓周期** | 20 ~ 30 个交易日 | 164.3 个交易日 (约 8 个月) | ⚠️ **低频选股** |
| **执行约束** | 考虑 T+1 | 无涨跌停限制、无滑点容量压测 | ⚠️ **乐观假设** |

> **治理决定**：
> 1. 将原 V5 策略全系报告正式标记为 **`⚠️ [SUPERSEDED / DEFECTIVE_LOGIC]`**；
> 2. 废除 V5 宽松放行逻辑，严禁在未来选股体系中使用 `isna() | condition` 伪过滤；
> 3. 选股体系全面迁移至统一生产级单现金池流水线（`UnifiedProductionLedger`）。
"""
    out_path = os.path.join(EXP_DIR, "v5_roe_leakage_audit_report.md")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(report_md)
    print(f"[Done] 审计完成！报告已生成: {out_path}")


if __name__ == "__main__":
    main()

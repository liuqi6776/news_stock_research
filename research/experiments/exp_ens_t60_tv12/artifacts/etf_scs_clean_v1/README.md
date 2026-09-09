# etf_scs_clean_v1: 规范基线策略重测报告 / Clean Baseline Strategy Audit Report

## 1. 策略概述 / Overview
- **标的 / Asset**: 512100.SH (中证1000 ETF)
- **防守腿 / Defensive Leg**: 100% 纯现金 (年化 2.0% 结息，严禁配置国债/黄金)
- **回测区间 / Period**: 2023-01-03 至 2026-08-06 (870 真实交易日，严格终止于实际数据末端，零 ffill 填补)
- **交易费用 / Transaction Costs**: 买卖双边 2 bps 滑点，3 bps 手续费 (最低 5 元)，10% ADV 参与率上限
- **执行变体 / Execution Variants**:
  - **B0 (Direct Target)**: 目标仓位直连 SCS，无平滑
  - **B1 (Deadband Smoothing)**: 8% 阈值宽带 + 0.5 调整系数 + 0.0 硬清仓通道

## 2. 核心表现指标 / Performance Metrics

| 方案 / Variant | 年化收益 (CAGR) | 夏普比率 (Sharpe) | 年化波动 (Vol) | 最大回撤 (MaxDD) | 卡玛比率 (Calmar) | 交易笔数 (Trades) | 总佣金 (Commission) |
|---|---|---|---|---|---|---|---|
| **512100.SH Buy & Hold** | 5.6% | 0.26 | 24.91% | -38.36% | 0.15 | 1 | ¥659.80 |
| **etf_scs_clean_v1 (B0 直投)** | 10.84% | 0.74 | 12.14% | -9.97% | 1.09 | 709 | ¥86466.23 |
| **etf_scs_clean_v1 (B1 平滑)** | 10.31% | 0.75 | 11.23% | -10.44% | 0.99 | 512 | ¥50143.12 |

## 3. 分年度收益率 / Annual Returns

| 年份 / Year | 512100.SH B&H | B0 直投 | B1 平滑 |
|---|---|---|---|
| **2023** | -5.28% | 4.48% | 5.45% |
| **2024** | 1.94% | 21.41% | 16.71% |
| **2025** | 27.23% | 6.61% | 9.54% |
| **2026 (YTD)** | -0.97% | 7.06% | 5.57% |

## 4. 审计结论 / Audit Conclusion
- 严禁声称旧版 15.06% / 1.06 Sharpe 成立，该数字来源于多资产混入与开盘价前向填补；
- 当前 B0 与 B1 的上述指标为唯一经过 Ledger v2.4 严格撮合审计认定的可靠基线数据；
- 后续主动选股策略必须且只能以此纯 ETF+SCS 基线计算增量 Alpha (Delta Alpha = Strategy - ETF+SCS)。

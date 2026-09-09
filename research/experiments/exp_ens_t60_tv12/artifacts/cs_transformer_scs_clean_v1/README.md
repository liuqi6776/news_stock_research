# cs_transformer_scs_clean_v1: 主动层增量评测报告 / Incremental Alpha Evaluation Report

## 1. 策略概述 / Overview
- **选股模型 / Stock Model**: CS-Transformer 月末优选前 40 只个股 (去除 ST、前瞻拥挤度与行业风控约束)
- **防守腿 / Defensive Leg**: 100% 纯现金 (年化 2.0% 结息，剔除多资产混入)
- **择时引擎 / Timing**: 与 `etf_scs_clean_v1` 严格同源的 SCS 情绪周期择时
- **微观账本 / Micro Ledger**: Ledger v2.4 (10 bps 股票佣金，10% ADV 限额，T+1，T0 初始基准)
- **回测区间 / Period**: 2023-01-03 至 2026-08-06 (870 交易日)

## 2. 核心增量指标对账 / Incremental Alpha Metrics (vs. etf_scs_clean_v1)

| 策略方案 / Strategy | 年化收益 (CAGR) | 夏普比率 (Sharpe) | 年化波动 (Vol) | 最大回撤 (MaxDD) | 交易笔数 (Trades) | 总交易佣金 (Commission) |
|---|---|---|---|---|---|---|
| **ETF+SCS 基准 (B1)** | 10.31% | 0.75 | 11.23% | -10.44% | 512 | ¥50,143.12 |
| **CS-Transformer (B1)** | 16.68% | 1.26 | 11.17% | -8.89% | 17297 | ¥192916.16 |
| **增量贡献 ($\Delta lpha$)** | **+6.37%** | **+0.51** | **+-0.06%** | **1.55%** | +16785 | +¥142773.04 |

- **跟踪误差 (Tracking Error)**: 7.27%
- **信息比率 (Information Ratio, IR)**: 0.77

## 3. 分年度增量收益率 / Annual Return Attribution (B1)

| 年份 / Year | ETF+SCS (B1) | CS-Transformer (B1) | 增量 Alpha ($\Delta lpha$) |
|---|---|---|---|
| **2023** | 5.45% | 2.35% | -3.1% |
| **2024** | 16.71% | 30.59% | +13.88% |
| **2025** | 9.54% | 24.95% | +15.41% |
| **2026 (YTD)** | 5.57% | 4.27% | -1.3% |

## 4. 审计结论 / Audit Conclusion
- 严禁宣传旧版 22.15% CAGR / 1.56 Sharpe，该数字混入了国债/黄金牛市并低估了实际换仓磨损；
- 在纯现金防守腿和 Ledger v2.4 审计约束下，CS-Transformer 实际取得上述表现；
- 配对增量评价证实：主动选股提供了扎实的超额选股能力，但需要承担一定的跟踪误差与换手佣金。

# cs_transformer_scs_clean_v1: 主动层增量评测报告 / Incremental Alpha Evaluation Report

## 1. 策略概述 / Overview
- **选股模型 / Stock Model**: CS-Transformer 月末优选前 40 只个股 (去除 ST、前瞻拥挤度与行业风控约束)
- **防守腿 / Defensive Leg**: 100% 纯现金 (年化 2.0% 结息，剔除多资产混入)
- **择时引擎 / Timing**: 与 `etf_scs_clean_v1` 严格同源单进程调用的 SCS 情绪周期择时
- **微观账本 / Micro Ledger**: UnifiedProductionLedger v2.5 (10 bps 股票佣金，10% ADV 限额，次日缺口重试，T+1，T0 初始基准)
- **回测区间 / Period**: 2023-01-03 至 2026-08-06 (870 交易日)

## 2. 核心增量指标对账 / Incremental Alpha Metrics (vs. etf_scs_clean_v1)

| 策略方案 / Strategy | 年化收益 (CAGR) | 夏普比率 (Sharpe) | 年化波动 (Vol) | 最大回撤 (MaxDD) | 交易笔数 (Trades) | 总交易佣金 (Commission) | 总摩擦成本 (Friction) |
|---|---|---|---|---|---|---|---|
| **ETF+SCS 基准 (B1)** | 10.31% | 0.75 | 11.23% | -10.44% | 512 | ¥50143.12 | ¥83571.83 |
| **CS-Transformer (B1)** | 16.68% | 1.26 | 11.17% | -8.89% | 17297 | ¥192916.16 | ¥192916.16 |
| **增量贡献 ($\Delta Alpha$)** | **+6.37%** | **+0.51** | **-0.06%** | **+1.55%** | +16785 | +142773.04 | +109344.33 |

- **跟踪误差 (Tracking Error)**: 7.28%
- **信息比率 (Information Ratio, IR)**: 0.77
- **配对日超额 Newey-West HAC 检验**: Alpha(年化)=5.61%, t-stat=1.14, p-value=0.2546 (单尾 p=0.1273)
- **时间块 Bootstrap 95% 置信区间**: 年化超额 [-2.35%, 18.3%], 夏普差值 [-0.15, 1.56]

## 3. 分年度增量收益率 / Annual Return Attribution (B1)

| 年份 / Year | ETF+SCS (B1) | CS-Transformer (B1) | 增量 Alpha ($\Delta Alpha$) |
|---|---|---|---|
| **2023** | 5.45% | 2.35% | -3.1% |
| **2024** | 16.71% | 30.59% | +13.88% |
| **2025** | 9.54% | 24.95% | +15.41% |
| **2026 (YTD)** | 5.57% | 4.27% | -1.3% |

## 4. 股票摩擦压力测试 / Stock Friction Stress Test (B1)

| 摩擦档位 / Friction Level | 佣金 (Comm) | 滑点 (Slippage) | CAGR | Sharpe | MaxDD | 总摩擦金额 (Friction) | Delta CAGR | Delta Sharpe |
|---|---|---|---|---|---|---|---|---|
| **低摩擦 (10 bps)** | 10 bps | 0 bps | 16.68% | 1.26 | -8.89% | ¥192,916.16 | +6.37% | +0.51 |
| **中摩擦 (20 bps)** | 10 bps | 10 bps | 14.3% | 1.08 | -9.93% | ¥370,621.03 | +3.99% | +0.33 |
| **高摩擦 (50 bps)** | 10 bps | 40 bps | 8.33% | 0.6 | -12.96% | ¥833,907.15 | -1.98% | -0.15 |

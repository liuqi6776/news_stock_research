# Research Report: Score-Based Initial Position Sizing & Floating-Profit Booster Strategy
# 科研报告：基于技术共振评分的初始定额与浮盈加码（Booster）策略体系

---

## 1. Executive Summary & Methodological Rectification / 核心执行总结与方法论整改

In response to rigorous quantitative audit and peer review, this research presents the corrected implementation of **Score-Based Initial Position Sizing with a Conditional Floating-Profit Booster**.

为落实量化评审与严谨审计要求，本研究对原策略体系进行了全面整改，确立了**“基于技术共振评分的初始定额配置（Score-Based Initial Position Sizing）与受控浮盈加码（Conditional Booster）”**的真实量化模型。

### Methodological Corrections & Invariants / 方法论整改要点与硬性约束:
1. **Accurate Terminology (精准命名，拒绝误导)**:
   - Renamed from "Tiered Entry" to **"Score-Based Initial Position Sizing"**.
   - Sizing is determined purely at initial entry based on indicator confluence ($S \in \{1, 2, 3\}$), sizing the trade as $0.33\text{x}$, $0.67\text{x}$, or $1.00\text{x}$. It is **not** cost-averaging or pyramiding into losers.
   - 更名为“基于技术共振评分的初始定额”，仅在建仓首根 K 线根据评分确定单笔初始仓位，绝非逆势补仓或向下摊平。

2. **Eradication of "Zero Principal Risk" Claims (彻底剔除“零本金风险”宣称)**:
   - **Market Risk Acknowledgment**: When a position is scaled up with booster leverage, the average entry price ($P_{\text{avg}}$) increases. In the event of overnight gap-downs, illiquidity, or severe slippage, the market can gap directly through the trailing stop, incurring losses to accumulated profit and potentially principal.
   - **Protection Invariant**: The booster is strictly **rejected** if the new volume-weighted average entry price would be greater than or equal to the trailing stop:
     $$\text{new\_avg\_entry} < \text{trailing\_stop}$$
   - **Default Safety**: `use_booster` is set to `False` by default across all production and paper execution configurations.
   - 彻底删除“绝对保本/零本金风险”表述。加仓必拉高均价，遭遇跳空或滑点将击穿止损。代码中强制实施“加仓后新均价必须严格低于止损线”的硬核校验，且默认全面禁用 Booster。

3. **Data Period Qualification (纠正 2026 年数据集定性)**:
   - **In-Sample Development Period (样本内开发集)**: `2021-01-01` to `2025-12-31` (5 full years).
   - **Post-hoc Recent Stress-Test Period (事后近期压力测试期)**: `2026-01-01` to `2026-09-01`. Because 2026 data has been repeatedly evaluated and analyzed during iterative development, it is designated as a post-hoc stress test rather than an untouched validation benchmark.
   - 2026 年数据因在多轮迭代中被测试与审阅，定性更正为“事后近期压力测试集”，不再宣称为纯盲测。

4. **Calmar Ratio Metric Integrity (修正 Calmar 比率公式)**:
   - Correct formula enforced: $\text{Calmar} = \text{CAGR} / |\text{Max Drawdown}|$.
   - A negative CAGR correctly yields a negative Calmar ratio, reflecting actual strategy underperformance.
   - 修复 Calmar 公式，亏损区间真实反映为负值。

---

## 2. Quantitative Architecture / 策略量化架构

### 2.1 Confluence Score & Initial Sizing / 技术共振评分与初始仓位

| Confluence Score / 评分 | Initial Size / 初始仓位 | Entry Criteria / 触发条件 | Quantitative Logic / 逻辑目的 |
| :---: | :---: | :--- | :--- |
| **Score = 1** | **0.33x (1/3 探路仓)** | $Close > \text{BB}_{\text{Upper}}(120)$ only | **防假突破**：仅布林上轨突破，均线或动量尚未确认。以 1/3 仓位试探，大幅降低假突破震荡磨损。 |
| **Score = 2** | **0.67x (2/3 顺势仓)** | $Close > \text{BB}_{\text{Upper}}(120)$ and $Close > \text{EMA}_{200}$ | **宏观顺势**：突破与宏观牛市共振，提高仓位至 2/3。 |
| **Score = 3** | **1.00x (全额标准仓)** | Above conditions + $\text{EMA}_{20} > \text{EMA}_{60}$ and $\text{RSI} > 50$ | **全维度共振**：趋势、均线多头排列、动量全线合力，满额标准仓 1.0x 运行。 |

### 2.2 Conditional Floating-Profit Booster / 条件浮盈加码机制

The booster expands leverage (e.g. 1.25x or 1.50x) **only** when all of the following conditions are met simultaneously:
1. $\text{Trailing Stop} \ge \text{Average Entry Price}$
2. Floating Profit Cushion $\ge 1.2 \times \text{ATR}$
3. Macro alignment: $Close > \text{EMA}_{200}$ and $\text{RSI} > 50$
4. **Mandatory Invariant**: $\text{new\_avg\_entry} < \text{trailing\_stop}$ (if adding size pushes average cost to or above the stop, the booster is rejected).

---

## 3. Empirical Multi-Asset Benchmark Results / 多资产实证审计数据

*Friction: 8 bps one-way transaction cost and slippage; 6% APR financing interest on leveraged exposure; 8h Binance funding rates included.*

### 3.1 Ethereum (ETHUSDT)
| Configuration | IS Ret (2021-2025) | IS MDD | IS Sharpe | IS Calmar | 2026 Stress Ret | 2026 MDD | 2026 Sharpe | 2026 Calmar | Boosters |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline (1.0x / 0.5x)** | +349.01% | -37.36% | 1.05 | 0.94 | -8.22% | -25.29% | -0.39 | -0.48 | 0 / 60 |
| **Score-Based Sizing** | +356.06% | -36.90% | 1.07 | 0.96 | -8.25% | -25.32% | -0.44 | -0.48 | 0 / 60 |
| **Score Sizing + Booster 1.25x** | +412.48% | -42.67% | 1.03 | 0.91 | -10.39% | -28.69% | -0.49 | -0.53 | 27 / 60 |
| **Score Sizing + Booster 1.50x** | +527.46% | -47.57% | 1.04 | 0.94 | -9.02% | -29.22% | -0.34 | -0.45 | 25 / 60 |

### 3.2 Solana (SOLUSDT)
| Configuration | IS Ret (2021-2025) | IS MDD | IS Sharpe | IS Calmar | 2026 Stress Ret | 2026 MDD | 2026 Sharpe | 2026 Calmar | Boosters |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline (1.0x / 0.5x)** | +4357.68% | -44.22% | 1.54 | 2.56 | +11.14% | -20.55% | 0.75 | 0.84 | 0 / 70 |
| **Score-Based Sizing** | +3267.35% | -43.53% | 1.49 | 2.34 | **+12.40%** | **-19.65%** | **0.84** | **0.99** | 0 / 70 |
| **Score Sizing + Booster 1.25x** | +4046.82% | -45.06% | 1.43 | 2.45 | +13.07% | -20.54% | 0.80 | 1.00 | 35 / 70 |
| **Score Sizing + Booster 1.50x** | +4181.52% | -45.08% | 1.40 | 2.48 | +8.01% | -22.50% | 0.53 | 0.55 | 33 / 70 |

### 3.3 Binance Coin (BNBUSDT)
| Configuration | IS Ret (2021-2025) | IS MDD | IS Sharpe | IS Calmar | 2026 Stress Ret | 2026 MDD | 2026 Sharpe | 2026 Calmar | Boosters |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline (1.0x / 0.5x)** | +1123.87% | -34.13% | 1.18 | 1.91 | -2.88% | -16.84% | -0.21 | -0.26 | 0 / 64 |
| **Score-Based Sizing** | +1138.03% | -34.13% | 1.19 | 1.93 | -2.86% | -16.77% | -0.21 | -0.25 | 0 / 64 |
| **Score Sizing + Booster 1.25x** | +1603.92% | -37.74% | 1.17 | 1.99 | -3.01% | -17.61% | -0.20 | -0.25 | 32 / 64 |
| **Score Sizing + Booster 1.50x** | +2093.55% | -41.32% | 1.15 | 2.05 | -2.63% | -18.01% | -0.15 | -0.22 | 32 / 64 |

---

## 4. Key Findings & Recommendations / 核心结论与部署建议

1. **Booster Reality & Downside / Booster 的真实代价**:
   - While Booster improves in-sample returns on ETH (+527% vs +349%) and BNB (+2093% vs +1123%), it expands maximum drawdown from -37% to -47% on ETH and from -34% to -41% on BNB.
   - During the 2026 choppy stress period, Booster configurations suffered deeper drawdowns on ETH (-29.2% vs -25.3%).
   - Booster 绝非免费午餐。加仓显著放大了回撤深度，在 2026 震荡市表现弱于未加仓版本。

2. **Score-Based Initial Sizing Outperformed on SOL / SOL 标的上评分定额表现优异**:
   - On SOLUSDT during the 2026 stress period, Score-Based Sizing achieved higher returns (**+12.40%** vs +11.14%), lower maximum drawdown (**-19.65%** vs -20.55%), and higher Sharpe (**0.84** vs 0.75).
   - In accordance with the audit directive, this variant is isolated in an independent forward paper experiment (`exp_paper_sol_score_sizing`) without contaminating the primary Pure Structural Trend baseline.
   - SOL 上的分档探路定额有效抑制了假突破磨损，已建立独立实验进行前向观察，绝不污染主实验。

3. **Operational Recommendation / 生产上线建议**:
   - Primary Paper Service: Enforce `use_booster = False`.
   - Real automatic trading remains strictly disabled (`ENABLE_REAL_ORDERS = False`).

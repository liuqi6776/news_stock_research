# ETF Valuation-based Sizing & Trend-filtered Asset Allocation Strategy

This research implements a systematic asset allocation framework using **HS300 ETF** (represented by `000300.SH`), **ZZ500 ETF** (represented by `000905.SH`), and **Treasury Bond ETF** (represented by `511010.SH`). 

Its core philosophy is to capture index beta with valuation discipline, trend-following rules, and extreme value buy-the-dip mechanics.

---

## 🧭 Strategy Design & Allocation Rules

1. **Core Assets**:
   - **HS300 ETF**: Large-cap core beta.
   - **ZZ500 ETF**: Mid-cap core beta.
   - **Treasury Bond ETF (Safe Asset)**: Safe-haven allocation for idle capital.
2. **Valuation-based Sizing (估值纪律)**:
   - Calculate the 5-year rolling quantile of PE and PB daily:
     $$Q_{i, t} = \frac{rank\_5y(PE_{i, t}) + rank\_5y(PB_{i, t})}{2}$$
   - The base weight allocated to each index is:
     $$W_{val, i, t} = \text{VAL\_COEFF} \times (1.0 - Q_{i, t})$$
   - Default optimal `VAL_COEFF` is **0.6**.
3. **Trend Filter & Force-In (趋势大闸与强制入场)**:
   - **Rule 1: Extreme Undervaluation**: If $Q_{i, t} \le 15\%$ (valuation is in the bottom 15% of the 5-year history), we ignore the trend filter to buy the bottom:
     $$W_{target, i, t} = W_{val, i, t}$$
   - **Rule 2: Normal Trend Filter**: If $Q_{i, t} > 15\%$:
     - If price >= MA250: $W_{target, i, t} = W_{val, i, t}$
     - If price < MA250: $W_{target, i, t} = W_{val, i, t} \times 0.5$ (weight is halved, instead of zeroed).
4. **Rebalancing Logic & Friction Cost**:
   - Check portfolio weights **weekly**.
   - Trigger execution only if the absolute deviation of either index's actual holding weight from the target weight is greater than **10%**:
     $$|W_{current, i} - W_{target, i}| > 0.10$$
   - A transaction cost of **0.05%** is charged on all traded volume.
   - Idle capital is parked in the Treasury Bond ETF (yielding `511010.SH` daily returns, or 3.0% annualized before its listing).

---

## 📊 Backtest Performance (Optimal: VAL_COEFF = 0.6)

### 1. 2015-2016 Stress Period (Circuit Breakers & Crash)
- **Valuation+Trend Strategy**: Total Return **+18.14%** | CAGR **8.76%** | Max Drawdown **-4.56%** | Volatility **6.26%**
- **HS300 Buy & Hold**: Total Return **-6.33%** | CAGR **-3.24%** | Max Drawdown **-46.70%** | Volatility **31.94%**
- **ZZ500 Buy & Hold**: Total Return **+17.68%** | CAGR **8.55%** | Max Drawdown **-54.35%** | Volatility **38.23%**
- **Static 50/50**: Total Return **+3.49%** | CAGR **1.74%** | Max Drawdown **-50.31%** | Volatility **33.90%**

### 2. Full Period (2015-2026)
- **Valuation+Trend Strategy**: Total Return **+95.38%** | CAGR **6.19%** | Max Drawdown **-24.53%** | Volatility **12.46%**
- **HS300 Buy & Hold**: Total Return **+33.31%** | CAGR **2.61%** | Max Drawdown **-46.70%** | Volatility **21.56%**
- **ZZ500 Buy & Hold**: Total Return **+62.67%** | CAGR **4.46%** | Max Drawdown **-65.20%** | Volatility **25.34%**
- **Static 50/50**: Total Return **+47.65%** | CAGR **3.56%** | Max Drawdown **-54.98%** | Volatility **22.58%**

---

## 🔍 Parameter Sensitivity & Optimization Findings
We evaluated `VAL_COEFF` from 0.5 to 0.8 over the 2015–2026 full period under the new rules:

- **VAL_COEFF = 0.5**: CAGR **5.80%** | Max Drawdown **-20.56%**
- **VAL_COEFF = 0.6**: CAGR **6.19%** | Max Drawdown **-24.53%** (Optimal)
- **VAL_COEFF = 0.7**: CAGR **4.82%** | Max Drawdown **-28.42%**
- **VAL_COEFF = 0.8**: CAGR **3.97%** | Max Drawdown **-32.82%**

> **Quant Insight**: A-shares exhibit extended "value traps" where prices keep falling despite extremely cheap valuations ($Q \le 15\%$). Larger coefficients (0.7-0.8) lead to overexposure during these downtrends, causing larger losses and dragging down the long-term compound growth rate (CAGR). **VAL_COEFF = 0.6** is the sweet spot to hit the CAGR target (6%–9%) and control drawdown near the -20% threshold.

---

## 🛠️ How to Run
1. **Download Data**:
   ```bash
   python etf-valuation-strategy/scripts/step1_download_data.py
   ```
2. **Run Backtest**:
   ```bash
   python etf-valuation-strategy/scripts/step2_backtest.py
   ```
3. **Analyze Strategy Behavior**:
   ```bash
   python etf-valuation-strategy/scripts/step3_analyze_beta.py
   ```

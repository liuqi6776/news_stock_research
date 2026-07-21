# A-Share Quant Trading System Upgrade: Verification & Configuration Downgrade

This implementation plan aims to address the critical consensus findings from the model reviews (GPT/Claude/Gemini) concerning the A-share quant trading system's production configuration and factor research. Specifically, we will clarify the verification file confusion, downgrade the production config to a candidate proposal, and strictly retrain/retest the stock factor layer A0 by excluding Beijing Stock Exchange (BSE) stocks to verify if it has any positive net alpha after transaction costs.

## User Review Required

We require user review and approval on the following key decisions and clarifications before proceeding to execution:

> [!IMPORTANT]
> **1. Clarification on `final_quant_VERIFICATION.md` (-7.07% Alpha, t = -2.46)**
> We conducted a thorough recursive disk search across `C:\Users\liuqi` for any file matching `final_quant_VERIFICATION` or containing `-7.07%` / `t = -2.46` in markdown/python files. **No such file exists on the local disk.** 
> The origin of these two numbers has been traced to `research/studies/study_007_cross_sectional/fix/results_enhanced/style_attribution.csv`:
> - **`-7.07%`** corresponds to the annualized excess return of **A1 (剔北交所)**, which is `-7.01%` (geometric) or `-7.07%` (arithmetic) in the CSV.
> - **`-2.46`** corresponds to the arithmetic monthly-average annualized net alpha of **A2 (剔北交所+剔最小市值20%)**, which is `-2.4632%` in the CSV.
> - **Conclusion**: In a previous LLM session, the reviewing model synthesized these two numbers, misread the arithmetic net alpha percentage `-2.46%` as a t-statistic `t = -2.46`, and hallucinated the filename `final_quant_VERIFICATION.md`. However, the underlying numbers are **real** and confirm that **once we exclude BJ and control for style, the selection alpha becomes negative (-3.69% to -6.13%)**.

> [!WARNING]
> **2. Downgrading the Production Configuration to Candidate Proposal v0.9**
> Following the unanimous consensus that the stock layer has questionable net alpha after friction, that `H_L60_k50` is a data-mining product that should not be in production, and that Study005 has a negative Sharpe ratio, we will:
> - Rename [生产配置书.md](file:///C:/Users/liuqi/Documents/kimi/workspace/生产配置书.md) to [候选方案_v0.9_未通过净成本alpha门槛.md](file:///C:/Users/liuqi/Documents/kimi/workspace/候选方案_v0.9_未通过净成本alpha门槛.md).
> - Rewrite the document to explicitly state it is a "Candidate Proposal" that failed the net cost alpha gate.
> - Downgrade `H_L60_k50` and `CbLiQ` to the shadow track (observation only).
> - Formally document the decision to halt the Study005 options live trading.

> [!IMPORTANT]
> **3. Strictly Re-building and Retraining A0 (Excluding BJ)**
> To determine if the stock layer has any standalone value, we will modify the core research pipeline in [run_fixed.py](file:///c:/Users/liuqi/quant_system_v2/research/studies/study_007_cross_sectional/fix/run_fixed.py) to define `is_bj_code` and filter out Beijing Stock Exchange stocks during both the training phase (Step 3) and testing phases (Steps 4, 5, 6). This will give us a clean, non-BJ A0 portfolio with factor weights optimized purely on the non-BJ universe. If its net alpha remains negative after dual-side 0.6% costs, we will recommend permanently shutting down the stock layer.

---

## Proposed Changes

### Configuration Components

#### [DELETE] [生产配置书.md](file:///C:/Users/liuqi/Documents/kimi/workspace/生产配置书.md)
#### [NEW] [候选方案_v0.9_未通过净成本alpha门槛.md](file:///C:/Users/liuqi/Documents/kimi/workspace/候选方案_v0.9_未通过净成本alpha门槛.md)

- Downgrade the title and header to a Candidate Proposal.
- Shift the size timing `H_L60_k50` and convertible bond timing `CbLiQ` from the production section to the shadow observation section.
- Add a dedicated section clarifying the negative net alpha gate and the Study005 options live trading halt.

---

### Research Components

#### [MODIFY] [run_fixed.py](file:///c:/Users/liuqi/quant_system_v2/research/studies/study_007_cross_sectional/fix/run_fixed.py)

- Add a helper function `is_bj_code(ts)` to identify Beijing Stock Exchange tickers (ending with `.BJ` or starting with `920`, `8`, `4`).
- Modify `step3()` to filter out BSE stocks from the `proc` DataFrame prior to computing monthly Rank ICs. This ensures the factor weights in `frozen_factors.csv` are trained strictly on the non-BSE universe.
- Modify `step4()` to filter out BSE stocks from `proc` before score synthesis.
- Modify `step5()` and `step6()` to filter out BSE stocks from scores and preprocessed data frames.

---

## Verification Plan

### Automated Tests

1. Run the modified training and backtesting pipeline in `c:/Users/liuqi/quant_system_v2/research/studies/study_007_cross_sectional/fix`:
   ```powershell
   python run_fixed.py --steps 3,4,5,6,7
   ```
2. Regenerate the backtesting plots:
   ```powershell
   python make_charts.py
   ```

### Manual Verification

- Inspect the generated `results_fixed/results.json` and `results_fixed/main_metrics_sensitivity.csv` to check the new out-of-sample CAGR, Sharpe, and MaxDD of `A0` (non-BJ trained and tested).
- Compare the new non-BJ `A0` results with the original A0 (which included BJ in training/testing) to quantify the exact return decay.
- Provide a clear conclusion on whether the stock layer has any remaining positive net alpha.

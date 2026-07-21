# Task List: A-Share Quant Trading System Upgrade

- [x] Downgrade Production Configuration
  - [x] Rename and update [生产配置书.md](file:///C:/Users/liuqi/Documents/kimi/workspace/生产配置书.md) to [候选方案_v0.9_未通过净成本alpha门槛.md](file:///C:/Users/liuqi/Documents/kimi/workspace/候选方案_v0.9_未通过净成本alpha门槛.md)
- [x] Strictly Exclude Beijing Stock Exchange (BSE) in Research Pipeline
  - [x] Modify [run_fixed.py](file:///c:/Users/liuqi/quant_system_v2/research/studies/study_007_cross_sectional/fix/run_fixed.py) to exclude BSE in training and testing
- [x] Execute Pipeline Retraining & Testing
  - [x] Run [run_fixed.py](file:///c:/Users/liuqi/quant_system_v2/research/studies/study_007_cross_sectional/fix/run_fixed.py) with steps all (1 to 7)
  - [x] Run [make_charts.py](file:///c:/Users/liuqi/quant_system_v2/research/studies/study_007_cross_sectional/fix/make_charts.py) to regenerate plots
- [x] Verify and Summarize Results
  - [x] Analyze results in `results_fixed/`
  - [x] Create `walkthrough.md` with the final findings
- [x] Route B Feasibility Pre-Research
  - [x] Implement A0 hedging backtest script ([backtest_hedged.py](file:///c:/Users/liuqi/quant_system_v2/research/studies/study_007_cross_sectional/fix/backtest_hedged.py))
  - [x] Compute hedged portfolio performance metrics (Spot vs IM Futures)
  - [x] Run shorting cost sensitivity analysis ([sensitivity_analysis.py](file:///c:/Users/liuqi/quant_system_v2/research/studies/study_007_cross_sectional/fix/sensitivity_analysis.py))
  - [x] Update [walkthrough.md](file:///C:/Users/liuqi/.gemini/antigravity/brain/7605f628-18c5-44bf-9009-df1901a187cf/walkthrough.md) and sync all code/plots to GitHub


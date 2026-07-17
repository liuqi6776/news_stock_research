# Study 004: A-Share Daily-Level Quantitative Strategy with T+1 Constraint

## Overview

This study implements and evaluates a daily-level stock selection strategy for the Chinese A-share market using XGBoost classification with monthly walk-forward retraining. The key contribution is a rigorous evaluation under the **T+1 trading constraint** (cannot sell on the purchase day), which is a fundamental rule of the A-share market that is often ignored in backtesting.

**Key Finding**: Stop-loss mechanisms that appear highly effective in unconstrained backtests become completely counterproductive under T+1 constraints, because next-day gap-down openings bypass the stop-loss price.

## Best Configuration (T+1 Constrained)

| Parameter | Value |
|-----------|-------|
| Threshold | 0.64 |
| Max Positions | 10 |
| Stop Loss | 0% (none) |
| Take Profit | 5% |

### Performance

| Metric | Optimization (2022-2024) | Test (2025-2026) |
|--------|--------------------------|-------------------|
| CAGR | 37.0% | 8.3% |
| Sharpe Ratio | 2.39 | 1.04 |
| Max Drawdown | -6.2% | -6.5% |
| # Trades | 690 | 233 |

## Critical Findings

### 1. T+1 Constraint Destroys Stop-Loss Effectiveness

| Stop Loss | Without T+1 (CAGR) | With T+1 (CAGR) |
|-----------|--------------------|--------------------|
| None | +1.6% | -44.6% |
| -3% | +248.1% | -97.8% |
| -5% | +126.9% | -93.7% |
| -7% | +69.1% | -72.1% |

Without T+1 constraint, a -3% stop-loss appears to boost CAGR from 1.6% to 248%. This is an **illusion** — in reality, you cannot execute stop-loss on the purchase day. When T+1 is enforced, stop-loss becomes catastrophic because:

- **Gap-down bypass**: Next day's opening price may already be below the stop-loss level
- **Worse execution price**: You sell at the (lower) opening price instead of the stop-loss price
- **Asymmetric damage**: Stop-loss triggers on losers (selling at worse price) but cannot protect winners

### 2. Take-Profit 5% is the Key Parameter

Under T+1 constraints, a 5% take-profit (without stop-loss) is the only parameter combination that produces positive test returns. The mechanism:

- Cuts short profit givebacks (common in 1-day holding patterns)
- Preserves most positive returns (5% is a reasonable daily gain)
- Avoids the stop-loss trap entirely

### 3. Strategy Edge is Weak

Even with the best parameters, test-period CAGR is only ~8% with Sharpe ~1.0. The strategy barely outperforms risk-free rate after transaction costs. This suggests:

- XGBoost classification for 1-day return prediction has limited alpha
- The model's probability calibration is unstable across months
- More effective features or different model architectures are needed

## Pipeline Steps

```
Step 1: Build Features (2-4 hours)
    ├── Price features (pct_chg, amplitude, body_size, shadows, gap, etc.)
    ├── Volume features (vol_ratio, vol_amount)
    ├── Momentum features (5d, 10d, 20d, 60d)
    ├── Volatility features (5d, 10d, 20d)
    ├── Technical indicators (RSI, KDJ, BB, ATR, MACD)
    ├── Fundamental features (PE, PB, circ_mv, turnover_rate)
    ├── News features (stock impact, market impact)
    ├── Ranking features (THS hot rank)
    ├── Money flow features (net_mf_amount, buy_lg_amount)
    └── THS news features (new_gs, new_bs, new_gi)
    → Output: data/all_features_v2.parquet

Step 2: Monthly Walk-Forward Prediction (5-6 hours)
    ├── For each month from 2022-01 to 2026-03:
    │   ├── Train XGBoost on all data up to previous month
    │   ├── Predict probabilities for current month
    │   └── Store predictions with actual returns
    → Output: predictions/predictions_1d_wf_monthly.parquet

Step 3: T+1 Constrained Grid Search (~5 minutes)
    ├── Load T+1 OHLC data (next day open/high/low/close)
    ├── For each (threshold, max_pos, stop_loss, take_profit):
    │   ├── Select stocks above threshold, rank by probability
    │   ├── Apply T+1 stop-loss/take-profit logic
    │   └── Compute CAGR, Sharpe, MaxDD for opt and test periods
    → Output: results/wf_monthly_grid_t1_full.csv

Step 4: Analyze Results
    ├── Compare T+1 vs no-T+1 results
    ├── Identify best parameter combinations
    └── Generate conclusions
```

## T+1 Stop-Loss Logic (Key Implementation Detail)

```
Purchase Day (T):
  - Buy at entry_price
  - CANNOT sell (T+1 rule)

Sell Day (T+1):
  - If next_open <= entry_price * (1 + stop_loss):
      → Gap-down below stop-loss, sell at next_open (actual loss > stop_loss)
  - Elif next_open >= entry_price * (1 + take_profit):
      → Gap-up above take-profit, sell at next_open
  - Else:
      - If next_low <= stop_loss_price:
          → Stop-loss triggered, sell at stop_loss_price
      - Elif next_high >= take_profit_price:
          → Take-profit triggered, sell at take_profit_price
      - Else:
          → No trigger, sell at next_close (exit_price_1d)

  - If both SL and TP could trigger on same day:
      → Conservative: assume SL takes priority
```

## Data Requirements

### Raw Data Sources

| Source | Files | Description |
|--------|-------|-------------|
| Daily prices | `D:\iquant_data\data_v2\prices\{date}.parquet` | OHLCV + pre_close |
| Fundamentals | `D:\iquant_data\data_v2\other\{date}.parquet` | PE, PB, circ_mv, turnover |
| News | `D:\iquant_data\data_v2\news\analysis_{date}.json` | Stock/market impact |
| Rankings | `D:\iquant_data\data_v2\rank\{date}.parquet` | THS hot rankings |
| Money flow | `D:\Users\liuqi\iquant_data\data_v2\moneyflow\{date}.parquet` | Net/buy/sell amounts |
| THS news | `D:\iquant_data\data_v2\ths_news\{date}.parquet` | Good/bad/neutral news |

### Data Schema (all_features_v2.parquet)

| Column | Type | Description |
|--------|------|-------------|
| trade_date | str | Trading date (YYYYMMDD) |
| ts_code | str | Stock code (e.g., 000001.SZ) |
| open/high/low/close | float | OHLC prices |
| entry_price | float | Execution price (T+2 open) |
| exit_price_1d | float | Exit price (T+3 close) |
| return_1d | float | 1-day return = exit_price_1d / entry_price - 1 |
| pct_chg, amplitude, body_size, ... | float | 54 feature columns |

### Prediction Target

```
Label = 1 if return_1d > 1.5%, else 0
return_1d = (T+3 close - T+2 open) / T+2 open
```

The 2-day lag (T → T+2 open) accounts for signal generation and order execution timing.

## Feature List (54 features)

| Category | Features |
|----------|----------|
| Price pattern | pct_chg, amplitude, body_size, upper_shadow, lower_shadow, is_yang, gap, close_to_high, close_to_low |
| Volume | vol_ratio, vol_amount |
| Momentum | mom_5d, mom_10d, mom_20d, mom_60d |
| Volatility | vol_5d, vol_10d, vol_20d |
| Technical | rsi_14, kdj_k, bb_position, atr_14, macd |
| Fundamental | pe, pb, log_pe, log_pb, circ_mv, log_circ_mv, turnover_rate, volume_ratio |
| News | news_stock_impact, news_market_impact, news_has_mention |
| Ranking | ths_hot, ths_hot_rank |
| Money flow | net_mf_amount, net_mf_vol, buy_lg_amount, sell_lg_amount, net_mf_amount_norm |
| THS news | new_gs, new_bs, new_gi |

## Model Configuration

```python
XGBClassifier(
    n_estimators=100,
    max_depth=5,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1,
    eval_metric='logloss'
)
```

## File Structure

```
study_004_release/
├── README.md                          # This file
├── scripts/
│   ├── step1_build_features.py        # Feature construction from raw data
│   ├── step2_walkforward_predict.py   # Monthly walk-forward prediction
│   ├── step3_grid_search_t1.py        # T+1 constrained grid search
│   └── step4_analyze_results.py       # Result analysis and visualization
├── results/
│   └── wf_monthly_grid_t1_full.csv    # Full grid search results (1125 combinations)
└── conclusions/
    └── summary.json                   # Structured conclusions and best parameters
```

## How to Reproduce

```bash
# Step 1: Build features (requires raw data access)
python scripts/step1_build_features.py

# Step 2: Monthly walk-forward prediction (5-6 hours)
python scripts/step2_walkforward_predict.py

# Step 3: T+1 constrained grid search (~5 minutes)
python scripts/step3_grid_search_t1.py

# Step 4: Analyze results
python scripts/step4_analyze_results.py
```

## Dependencies

```
pandas >= 1.3
numpy >= 1.20
xgboost >= 1.5
scikit-learn >= 1.0
tqdm >= 4.60
```

## Caveats and Limitations

1. **Test period is short**: Only 15 months (2025-01 to 2026-03), statistical significance is limited
2. **Transaction cost model is simplified**: Fixed 0.3% per trade, actual costs vary by broker and position size
3. **No slippage model beyond transaction cost**: Real execution may have additional market impact
4. **Entry price assumption**: Uses T+2 open price as entry, which assumes signal generation on T and order submission on T+1
5. **Take-profit execution**: Assumes exact take-profit price execution, which may not be achievable in practice
6. **Model probability calibration**: XGBoost probabilities are not well-calibrated across different months, leading to unstable selection counts
7. **Survivorship bias**: Uses current stock universe, delisted stocks may not be included
8. **Feature staleness**: Some features (fundamentals, news) may have reporting delays

## Review Checklist for AI Auditors

- [ ] Is the T+1 stop-loss logic correctly implemented? (See `step3_grid_search_t1.py`, `compute_realized_return()`)
- [ ] Is the walk-forward prediction truly out-of-sample? (Each month trained only on prior data)
- [ ] Is there any data leakage in feature construction? (Features use only data available at time T)
- [ ] Is the entry/exit price definition correct? (entry = T+2 open, exit = T+3 close)
- [ ] Are transaction costs properly applied? (0.3% per trade, deducted from each trade return)
- [ ] Is the grid search fair? (Optimization on 2022-2024, test on 2025-2026, no overlap)
- [ ] Is the Sharpe ratio calculation correct? (Annualized from daily returns: mean/std * sqrt(252))
- [ ] Are the conclusions supported by the data? (Test period is short, results should be interpreted cautiously)

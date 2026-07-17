import pandas as pd
import numpy as np
import json

pred = pd.read_parquet('predictions/predictions_1d_open_wf_monthly.parquet')
pred['ds'] = pred['trade_date'].astype(str)

above = pred[pred.prob >= 0.55].copy()
above['rank'] = above.groupby('ds')['prob'].rank(ascending=False, method='first')
selected = above[above['rank'] <= 3].copy()
valid = selected.dropna(subset=['actual_return'])

print('Distribution of actual returns for selected stocks (th=0.55, pos=3):')
print(valid.actual_return.describe())
print()
print('Percentiles:')
for p in [1, 5, 10, 25, 50, 75, 90, 95, 99]:
    val = valid.actual_return.quantile(p/100)
    print(f'  {p}th: {val:.2%}')

# Check for extreme returns that might indicate data issues
extreme = valid[valid.actual_return.abs() > 0.2]
print(f'\nExtreme returns (>20%): {len(extreme)} out of {len(valid)}')
if len(extreme) > 0:
    print(extreme[['ds', 'ts_code', 'prob', 'actual_return']].head(20).to_string())

# Check: are there 涨停板 (limit up) stocks being selected?
limit_up = valid[valid.actual_return > 0.09]
print(f'\nReturns > 9% (possible limit-up gap): {len(limit_up)} out of {len(valid)}')

# The real question: is the model just picking momentum stocks?
# If the model selects stocks that already went up today (T),
# and they continue to go up on T+1 and T+2, that's momentum
# But it could also be that the model is picking stocks with
# positive news/catalysts that continue to play out

# Let me check: what's the avg pct_chg (T-day return) for selected stocks?
df = pd.read_parquet('data/all_features_v2.parquet')
df['ds'] = df['trade_date'].astype(str)

# Merge selected with features
merged = selected.merge(df[['ds', 'ts_code', 'pct_chg', 'mom_5d', 'turnover_rate']], 
                        on=['ds', 'ts_code'], how='left')
valid_merged = merged.dropna(subset=['actual_return', 'pct_chg'])

print(f'\nSelected stocks T-day characteristics:')
print(f'  avg pct_chg (T-day return): {valid_merged.pct_chg.mean():.2%}')
print(f'  avg mom_5d: {valid_merged.mom_5d.mean():.2%}')
print(f'  avg turnover_rate: {valid_merged.turnover_rate.mean():.2%}')

# Compare with all stocks
all_valid = df.dropna(subset=['pct_chg', 'return_1d_open'])
print(f'\nAll stocks T-day characteristics:')
print(f'  avg pct_chg: {all_valid.pct_chg.mean():.2%}')
print(f'  avg mom_5d: {all_valid.mom_5d.mean():.2%}')
print(f'  avg turnover_rate: {all_valid.turnover_rate.mean():.2%}')

# If selected stocks have much higher pct_chg, the model might be 
# just picking today's winners (momentum), which may or may not continue

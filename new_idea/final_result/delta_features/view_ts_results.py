import pandas as pd
r = pd.read_csv(r'c:\Users\liuqi\quant_system_v2\new_idea\final_result\delta_features\ts_feature_ranking.csv')
print(f'Total features: {len(r)}')
print(f'Selected: {r["selected"].sum()}')
print()
sel = r[r['selected']]
print('SELECTED FEATURES:')
for _, row in sel.iterrows():
    print(f'  {row["feature"]:<45} imp={row["importance"]:.4f}  corr={row["corr_with_target"]:.4f}')
print()
print('TOP 10 DROPPED (by importance):')
dropped = r[~r['selected']]
for _, row in dropped.head(10).iterrows():
    print(f'  {row["feature"]:<45} imp={row["importance"]:.4f}  corr={row["corr_with_target"]:.4f}')

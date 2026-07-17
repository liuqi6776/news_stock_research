import pandas as pd
f = pd.read_parquet(r'c:\Users\liuqi\quant_system_v2\new_idea\final_result\delta_features\ts_panel_labeled.parquet')
print(f'Rows: {len(f)}')
print(f'Date range: {f["date"].min()} - {f["date"].max()}')
print(f'Stocks: {f["ts_code"].nunique()}')
print(f'Label non-null: {f["label"].notna().sum()}')
labeled = f[f['label'].notna()]
print(f'Pos rate: {labeled["label"].mean():.3f}')
print(f'Columns: {len(f.columns)}')

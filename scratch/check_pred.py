import pandas as pd
import os

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
pred_path = os.path.join(ROOT_DIR, "research", "study_005_1d_advanced", "predictions", "predictions_005_options_wf.parquet")

if os.path.exists(pred_path):
    df = pd.read_parquet(pred_path)
    df['trade_date'] = df['trade_date'].astype(str)
    print(f"File exists. Rows: {len(df)}")
    print(f"Min date: {df['trade_date'].min()}")
    print(f"Max date: {df['trade_date'].max()}")
    print(f"Unique dates count: {df['trade_date'].nunique()}")
    print("\nRecent 5 dates:")
    print(sorted(df['trade_date'].unique())[-5:])
else:
    print(f"File does not exist at {pred_path}")

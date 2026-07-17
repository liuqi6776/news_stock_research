import pandas as pd
import os

PRED_FILE_BASE = 'research/study_005_1d_advanced/predictions/predictions_005_wf.parquet'
PRED_FILE_OPT = 'research/study_005_1d_advanced/predictions/predictions_005_options_wf.parquet'

print("Base columns:")
if os.path.exists(PRED_FILE_BASE):
    df_base = pd.read_parquet(PRED_FILE_BASE)
    print(df_base.columns.tolist())
    print("Base shape:", df_base.shape)
    print("Base stats:")
    print(df_base[['prob_up', 'prob_crash']].describe())

print("\nOpt columns:")
if os.path.exists(PRED_FILE_OPT):
    df_opt = pd.read_parquet(PRED_FILE_OPT)
    print(df_opt.columns.tolist())
    print("Opt shape:", df_opt.shape)
    print("Opt stats:")
    print(df_opt[['prob_up', 'prob_crash']].describe())

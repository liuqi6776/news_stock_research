import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pandas as pd
import numpy as np
import joblib

THIS_DIR = os.path.dirname(os.path.abspath(__file__))

model = joblib.load(os.path.join(THIS_DIR, 'best_model_v2.joblib'))

feature_cache_dir = os.path.join(THIS_DIR, 'feature_cache')
feat_files = sorted([f for f in os.listdir(feature_cache_dir) if f.endswith('.parquet')])
print(f"Found {len(feat_files)} files, first: {feat_files[0]}")

features = pd.read_parquet(os.path.join(feature_cache_dir, feat_files[0]))
print(f"Features shape: {features.shape}")
print(f"Features columns: {list(features.columns)}")

feats = list(model.feature_names_in_) if hasattr(model, 'feature_names_in_') else []
print(f"Model features ({len(feats)}): {feats[:5]}...")

# 检查features中是否有这些列
for f in feats:
    if f not in features.columns:
        print(f"  MISSING: {f}")
    else:
        print(f"  OK: {f}, non-null: {features[f].notna().sum()}/{len(features)}")

# 尝试取有数据的行
X = features[feats]
print(f"\nX shape before fillna: {X.shape}")
print(f"X null count per col:\n{X.isnull().sum()}")

X_filled = X.fillna(0)
print(f"\nX shape after fillna: {X_filled.shape}")
print(f"X_filled sample:\n{X_filled.head(3)}")

if len(X_filled) > 0:
    proba = model.predict_proba(X_filled)
    print(f"\nProba shape: {proba.shape}")
    print(f"Proba min: {proba[:, 1].min():.4f}, max: {proba[:, 1].max():.4f}")

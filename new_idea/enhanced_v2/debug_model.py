import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pandas as pd
import numpy as np
import joblib

THIS_DIR = os.path.dirname(os.path.abspath(__file__))

model = joblib.load(os.path.join(THIS_DIR, 'best_model_v2.joblib'))

print(f"Model type: {type(model)}")
print(f"Model classes: {model.classes_ if hasattr(model, 'classes_') else 'N/A'}")

feature_cache_dir = os.path.join(THIS_DIR, 'feature_cache')
feat_files = sorted([f for f in os.listdir(feature_cache_dir) if f.endswith('.parquet')])
features = pd.read_parquet(os.path.join(feature_cache_dir, feat_files[0]))

feats = list(model.feature_names_in_) if hasattr(model, 'feature_names_in_') else []
X = features[feats].fillna(0)

print(f"\nX shape: {X.shape}")
print(f"X sample:\n{X.head(2)}")

try:
    proba = model.predict_proba(X)
    print(f"\nProba shape: {proba.shape}")
    print(f"Proba sample:\n{proba[:5]}")
except Exception as e:
    print(f"predict_proba error: {e}")
    try:
        pred = model.predict(X)
        print(f"\nPredict output shape: {pred.shape}")
        print(f"Predict sample: {pred[:10]}")
        print(f"Unique values: {np.unique(pred)}")
    except Exception as e2:
        print(f"predict error: {e2}")

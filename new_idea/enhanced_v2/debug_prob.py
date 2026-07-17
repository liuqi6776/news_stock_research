import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pandas as pd
import numpy as np
import joblib

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
feature_cache_dir = os.path.join(THIS_DIR, 'feature_cache')

model = joblib.load(os.path.join(THIS_DIR, 'best_model_v2.joblib'))
feats = list(model.feature_names_in_) if hasattr(model, 'feature_names_in_') else []

feat_files = sorted([f for f in os.listdir(feature_cache_dir) if f.endswith('.parquet')])

# 找一个有数据的文件
for f in feat_files:
    df = pd.read_parquet(os.path.join(feature_cache_dir, f))
    if len(df) > 0:
        print(f"Using {f}, shape={df.shape}")
        
        missing = [col for col in feats if col not in df.columns]
        for col in missing:
            df[col] = 0.0
        
        X = df[feats].fillna(0)
        probs = model.predict_proba(X)[:, 1]
        
        print(f"Probabilities: min={probs.min():.4f}, max={probs.max():.4f}, mean={probs.mean():.4f}")
        print(f">= 0.4: {(probs >= 0.4).sum()}")
        print(f">= 0.5: {(probs >= 0.5).sum()}")
        print(f">= 0.6: {(probs >= 0.6).sum()}")
        print(f">= 0.3: {(probs >= 0.3).sum()}")
        print(f">= 0.2: {(probs >= 0.2).sum()}")
        
        # 打印top5
        df['prob'] = probs
        print("\nTop 5 by probability:")
        print(df[['ts_code', 'prob']].sort_values('prob', ascending=False).head())
        break

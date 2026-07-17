import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pandas as pd
import numpy as np
import joblib

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PRICE_DIR = r'D:\iquant_data\data_v2\data_day1'
feature_cache_dir = os.path.join(THIS_DIR, 'feature_cache')

model = joblib.load(os.path.join(THIS_DIR, 'best_model_v2.joblib'))
feats = list(model.feature_names_in_) if hasattr(model, 'feature_names_in_') else []

feat_files = sorted([f for f in os.listdir(feature_cache_dir) if f.endswith('.parquet')])
all_dates = [f.replace('.parquet', '').replace('feat_', '') for f in feat_files]

# 找一个有数据的日期
test_date = None
for d in all_dates:
    df = pd.read_parquet(os.path.join(feature_cache_dir, f"feat_{d}.parquet"))
    if len(df) > 0:
        test_date = d
        break

print(f"Testing with date: {test_date}")

# 读取特征
features = pd.read_parquet(os.path.join(feature_cache_dir, f"feat_{test_date}.parquet"))
print(f"Features shape: {features.shape}")

missing = [col for col in feats if col not in features.columns]
for col in missing:
    features[col] = 0.0

X = features[feats].fillna(0)
features['prob'] = model.predict_proba(X)[:, 1]

candidates = features[features['prob'] >= 0.4].sort_values('prob', ascending=False)
print(f"Candidates >= 0.4: {len(candidates)}")

if len(candidates) > 0:
    pick = candidates.iloc[0]
    ts_code = pick['ts_code']
    print(f"Top pick: {ts_code}, prob={pick['prob']:.4f}")
    
    # 检查价格文件
    curr_idx = all_dates.index(test_date)
    d_t1 = all_dates[curr_idx + 1]
    d_t2 = all_dates[curr_idx + 2]
    
    p_t0 = os.path.join(PRICE_DIR, f"{test_date}.parquet")
    p_t1 = os.path.join(PRICE_DIR, f"{d_t1}.parquet")
    p_t2 = os.path.join(PRICE_DIR, f"{d_t2}.parquet")
    
    print(f"\nPrice files:")
    print(f"  T  ({test_date}): exists={os.path.exists(p_t0)}")
    print(f"  T+1({d_t1}): exists={os.path.exists(p_t1)}")
    print(f"  T+2({d_t2}): exists={os.path.exists(p_t2)}")
    
    if os.path.exists(p_t1) and os.path.exists(p_t2):
        price_t1 = pd.read_parquet(p_t1, columns=['ts_code', 'open', 'low', 'pre_close'])
        price_t2 = pd.read_parquet(p_t2, columns=['ts_code', 'close', 'low', 'pre_close'])
        
        t1_row = price_t1[price_t1['ts_code'] == ts_code]
        t2_row = price_t2[price_t2['ts_code'] == ts_code]
        
        print(f"\nT+1 row found: {not t1_row.empty}")
        print(f"T+2 row found: {not t2_row.empty}")
        
        if not t1_row.empty:
            print(f"T+1 open: {t1_row['open'].values[0]}")
        if not t2_row.empty:
            print(f"T+2 close: {t2_row['close'].values[0]}")

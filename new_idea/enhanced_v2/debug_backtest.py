import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pandas as pd
import numpy as np
import joblib

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
feature_cache_dir = os.path.join(THIS_DIR, 'feature_cache')

model = joblib.load(os.path.join(THIS_DIR, 'best_model_v2.joblib'))

BASE_FEATS = ['hot_rank_pct', 'chip_concentration', 'winner_rate',
              'news_market_impact', 'news_stock_impact']
CHAN_FEATS = ['chan_bi_count', 'chan_zhongshu_count', 'chan_zhongshu_width',
              'chan_macd_divergence', 'chan_bi_direction', 'chan_leave_zhongshu']
LYNCH_FEATS = ['lynch_peg', 'lynch_peg_rank', 'lynch_quality_score',
               'lynch_growth_value', 'lynch_roe_stability', 'lynch_earnings_momentum']
QUANT_FEATS = ['qf_mom_1d', 'qf_mom_3d', 'qf_mom_5d', 'qf_mom_10d', 'qf_mom_20d',
               'qf_reversal_1d', 'qf_reversal_3d', 'qf_realized_vol', 'qf_atr_pct',
               'qf_rsi_14', 'qf_bb_position', 'qf_ma_cross_5_10', 'qf_ma_cross_10_20',
               'qf_vol_ratio_5_20', 'qf_pv_corr']
ENHANCED_FEATS = BASE_FEATS + CHAN_FEATS + LYNCH_FEATS + QUANT_FEATS

feats = list(model.feature_names_in_) if hasattr(model, 'feature_names_in_') else ENHANCED_FEATS

# 读取第一个feature cache文件
feat_files = sorted([f for f in os.listdir(feature_cache_dir) if f.endswith('.parquet')])
print(f"Found {len(feat_files)} feature cache files")

feat_file = os.path.join(feature_cache_dir, feat_files[0])
features = pd.read_parquet(feat_file)

print(f"\nFeature cache columns: {list(features.columns)}")
print(f"Model features ({len(feats)}): {feats}")

missing = [f for f in feats if f not in features.columns]
print(f"\nMissing features: {missing}")

# 填充缺失特征
for f in missing:
    features[f] = 0.0

X = features[feats].fillna(0)
probs = model.predict_proba(X)[:, 1]

print(f"\nProbability stats:")
print(f"  Min: {probs.min():.4f}")
print(f"  Max: {probs.max():.4f}")
print(f"  Mean: {probs.mean():.4f}")
print(f"  Median: {np.median(probs):.4f}")
print(f"  > 0.4: {(probs >= 0.4).sum()}")
print(f"  > 0.5: {(probs >= 0.5).sum()}")
print(f"  > 0.6: {(probs >= 0.6).sum()}")

features['prob'] = probs
print(f"\nTop 5 predictions:")
print(features[['ts_code'] + feats[:3] + ['prob']].sort_values('prob', ascending=False).head())

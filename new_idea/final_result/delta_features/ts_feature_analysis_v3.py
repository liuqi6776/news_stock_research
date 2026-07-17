"""
Feature Analysis v3 - Load only needed columns to save memory.
"""
import os, sys
import pandas as pd
import numpy as np
import joblib
import xgboost as xgb
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
TEST_START = '20230101'

def main():
    print("=" * 90, flush=True)
    print("  TS Feature Analysis v3 - Column-Optimized", flush=True)
    print("=" * 90, flush=True)

    labeled_path = os.path.join(THIS_DIR, 'ts_panel_labeled.parquet')

    # Get column names from parquet metadata
    import pyarrow.parquet as pq
    schema = pq.read_schema(labeled_path)
    all_cols = schema.names
    print(f"  Total columns: {len(all_cols)}", flush=True)
    
    # Identify feature columns (exclude raw data and metadata)
    exclude = {'ts_code', 'date', 'open', 'high', 'low', 'close', 'pct_chg', 'vol', 'amount',
               'pre_close', 'circ_mv', 'pe', 'pb', 'turnover_rate', 'volume_ratio',
               'cost_50pct', 'weight_avg', 'cost_15pct', 'cost_85pct',
               'chip_concentration', 'winner_rate', 'hot_rank_pct', 'label', 'label_ret',
               'news_market_impact', 'news_stock_impact'}
    feature_cols = [c for c in all_cols if c not in exclude]
    print(f"  Feature columns: {len(feature_cols)}", flush=True)

    # Load only needed columns: ts_code, date, label, label_ret, and features
    needed_cols = ['ts_code', 'date', 'label', 'label_ret'] + feature_cols
    print(f"  Loading {len(needed_cols)} columns...", flush=True)
    
    df = pd.read_parquet(labeled_path, columns=needed_cols)
    test_df = df[df['date'] >= int(TEST_START)].copy()
    del df
    print(f"  Test period: {len(test_df)} rows, pos_rate={test_df['label'].mean():.3f}", flush=True)

    # Filter valid features
    valid_feats = [f for f in feature_cols if test_df[f].notna().sum() > 10000]
    print(f"  Valid features (>10k non-null): {len(valid_feats)}", flush=True)

    # Correlation
    print("\n  --- Correlation with label_ret ---", flush=True)
    corr = test_df[valid_feats + ['label_ret']].corr()['label_ret'].drop('label_ret').abs().sort_values(ascending=False)
    print(f"  Top 25 features by |correlation|:", flush=True)
    for feat, c in corr.head(25).items():
        print(f"    {feat:<50} corr={c:.4f}", flush=True)

    # XGBoost importance (sample for speed)
    print("\n  --- Feature Importance (XGBoost) ---", flush=True)
    if len(test_df) > 500000:
        sample = test_df.sample(500000, random_state=42)
    else:
        sample = test_df

    X = sample[valid_feats].fillna(0)
    y = sample['label'].astype(int)

    model = xgb.XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.08,
                               subsample=0.8, colsample_bytree=0.7, min_child_weight=5,
                               eval_metric='logloss', verbosity=0, random_state=42, n_jobs=-1)
    model.fit(X, y)

    imp = pd.Series(model.feature_importances_, index=valid_feats).sort_values(ascending=False)
    print(f"  Top 30 features by importance:", flush=True)
    for feat, score in imp.head(30).items():
        print(f"    {feat:<50} imp={score:.4f}", flush=True)

    # Correlation filter
    print("\n  --- Correlation Filter (threshold=0.8) ---", flush=True)
    top_feats = imp.head(50).index.tolist()
    feat_corr = sample[top_feats].corr().abs()
    upper = feat_corr.where(np.triu(np.ones(feat_corr.shape), k=1).astype(bool))
    to_drop = [col for col in upper.columns if any(upper[col] > 0.8)]
    print(f"  Features to drop (corr>0.8): {len(to_drop)}", flush=True)
    for f in to_drop[:15]:
        correlated_with = upper[f][upper[f] > 0.8].index.tolist()
        print(f"    {f:<50} correlated with: {correlated_with[:3]}", flush=True)

    selected = [f for f in top_feats if f not in to_drop]
    print(f"  Selected features (from top 50): {len(selected)}", flush=True)
    for f in selected:
        print(f"    {f:<50} imp={imp[f]:.4f}  corr={corr.get(f, 0):.4f}", flush=True)

    # Save
    imp_df = pd.DataFrame({
        'feature': imp.index,
        'importance': imp.values,
        'corr_with_target': [corr.get(f, 0) for f in imp.index],
        'selected': [f in selected for f in imp.index]
    })
    imp_df.to_csv(os.path.join(THIS_DIR, 'ts_feature_ranking.csv'), index=False)
    pd.DataFrame({'feature': selected}).to_csv(os.path.join(THIS_DIR, 'ts_selected_features.csv'), index=False)
    joblib.dump(model, os.path.join(THIS_DIR, 'ts_analysis_model.joblib'))

    # Plot
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    fig, ax = plt.subplots(figsize=(14, 10))
    top20 = imp.head(20).sort_values()
    colors = ['green' if f in selected else 'red' for f in top20.index]
    ax.barh(range(len(top20)), top20.values, color=colors)
    ax.set_yticks(range(len(top20)))
    ax.set_yticklabels(top20.index, fontsize=8)
    ax.set_xlabel('Feature Importance')
    ax.set_title('Time-Series Feature Importance (Green=Selected, Red=Dropped by Corr Filter)')
    plt.tight_layout()
    plt.savefig(os.path.join(THIS_DIR, 'ts_feature_importance.png'), dpi=150, bbox_inches='tight')
    print(f"\nChart saved", flush=True)
    print(f"\nAll results saved to {THIS_DIR}", flush=True)

if __name__ == "__main__":
    main()

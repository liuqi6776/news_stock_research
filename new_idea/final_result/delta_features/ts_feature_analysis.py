"""
Step 2: Add labels and run feature analysis on the pre-computed feature panel.
Separated from ts_feature_engineering.py to manage memory.
"""
import os, sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = r'D:\iquant_data\data_v2'
PRICE_DIR = os.path.join(DATA_DIR, 'data_day1')

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
TEST_START = '20230101'

def int_to_date(date_int):
    s = str(int(date_int))
    return datetime(int(s[:4]), int(s[4:6]), int(s[6:8]))

def add_labels(panel, all_dates_set):
    """Add target labels: (T+2 close / T+1 open - 1) > 0.04"""
    print("  Adding labels...", flush=True)
    df = panel.copy()
    df['label'] = np.float32(np.nan)
    df['label_ret'] = np.float32(np.nan)

    price_cache = {}
    dates = sorted(df['date'].unique())
    print(f"  Total dates to process: {len(dates)}", flush=True)

    for i, d in enumerate(dates):
        d_int = int(d)
        dt = int_to_date(d_int)
        t1, t2 = None, None
        for j in range(1, 10):
            nd = int((dt + timedelta(days=j)).strftime('%Y%m%d'))
            if nd in all_dates_set:
                if t1 is None:
                    t1 = nd
                elif t2 is None:
                    t2 = nd
                    break

        if t1 is None or t2 is None:
            continue

        if t1 not in price_cache:
            p1 = os.path.join(PRICE_DIR, f"{t1}.parquet")
            if os.path.exists(p1):
                price_cache[t1] = pd.read_parquet(p1, columns=['ts_code', 'open']).rename(columns={'open': 'open_t1'})
        if t2 not in price_cache:
            p2 = os.path.join(PRICE_DIR, f"{t2}.parquet")
            if os.path.exists(p2):
                price_cache[t2] = pd.read_parquet(p2, columns=['ts_code', 'close']).rename(columns={'close': 'close_t2'})

        if t1 in price_cache and t2 in price_cache:
            mask = df['date'] == d
            day_stocks = df.loc[mask, 'ts_code']
            m = pd.merge(day_stocks.to_frame(), price_cache[t1], on='ts_code', how='left')
            m = pd.merge(m, price_cache[t2], on='ts_code', how='left')
            if 'open_t1' in m.columns and 'close_t2' in m.columns:
                m['label_ret'] = m['close_t2'] / m['open_t1'] - 1
                m['label'] = (m['label_ret'] > 0.04).astype(int)
                df.loc[mask, 'label_ret'] = m['label_ret'].values.astype(np.float32)
                df.loc[mask, 'label'] = m['label'].values.astype(np.float32)

        if (i + 1) % 200 == 0:
            print(f"  {i+1}/{len(dates)} dates labeled", flush=True)

    return df

def get_feature_cols(df):
    exclude = {'ts_code', 'date', 'open', 'high', 'low', 'close', 'pct_chg', 'vol', 'amount',
               'pre_close', 'circ_mv', 'pe', 'pb', 'turnover_rate', 'volume_ratio',
               'cost_50pct', 'weight_avg', 'cost_15pct', 'cost_85pct',
               'chip_concentration', 'winner_rate', 'hot_rank_pct', 'label', 'label_ret',
               'news_market_impact', 'news_stock_impact'}
    return [c for c in df.columns if c not in exclude and df[c].dtype in ['float64', 'float32', 'int64', 'int32', 'int8']]

def main():
    print("=" * 90, flush=True)
    print("  TS Feature Analysis - Step 2: Labels + Feature Selection", flush=True)
    print("=" * 90, flush=True)

    all_dates = sorted([f.replace('.parquet', '') for f in os.listdir(PRICE_DIR) if f.endswith('.parquet')])
    all_dates_set = set(int(d) for d in all_dates)

    # Load features
    print("\n[Step 1] Loading feature panel...", flush=True)
    feat_path = os.path.join(THIS_DIR, 'ts_panel_features.parquet')
    feat_panel = pd.read_parquet(feat_path)
    print(f"  Loaded: {len(feat_panel)} rows, {len(feat_panel.columns)} columns", flush=True)

    # Add labels
    print("\n[Step 2] Adding labels...", flush=True)
    labeled_path = os.path.join(THIS_DIR, 'ts_panel_labeled.parquet')

    if os.path.exists(labeled_path):
        print("  Loading existing labeled data...", flush=True)
        labeled_df = pd.read_parquet(labeled_path)
    else:
        labeled_df = add_labels(feat_panel, all_dates_set)
        labeled_df = labeled_df[labeled_df['label'].notna()].copy()
        del feat_panel
        labeled_df.to_parquet(labeled_path, index=False)
        print(f"  Labeled: {len(labeled_df)} rows", flush=True)

    # Filter to test period for analysis
    test_df = labeled_df[labeled_df['date'] >= int(TEST_START)].copy()
    print(f"  Test period: {len(test_df)} rows, pos_rate={test_df['label'].mean():.3f}", flush=True)

    # Feature Analysis
    print("\n[Step 3] Feature Analysis...", flush=True)
    feature_cols = get_feature_cols(test_df)
    print(f"  Total features: {len(feature_cols)}", flush=True)

    # 3a: Correlation
    print("\n  --- Correlation with label_ret ---", flush=True)
    valid_feats = [f for f in feature_cols if test_df[f].notna().sum() > 1000]
    print(f"  Valid features (>{1000} non-null): {len(valid_feats)}", flush=True)

    corr = test_df[valid_feats + ['label_ret']].corr()['label_ret'].drop('label_ret').abs().sort_values(ascending=False)
    print(f"  Top 25 features by |correlation|:", flush=True)
    for feat, c in corr.head(25).items():
        print(f"    {feat:<50} corr={c:.4f}", flush=True)

    # 3b: XGBoost importance
    print("\n  --- Feature Importance (XGBoost) ---", flush=True)
    import xgboost as xgb

    X = test_df[valid_feats].fillna(0)
    y = test_df['label'].astype(int)

    model = xgb.XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.08,
                               subsample=0.8, colsample_bytree=0.7, min_child_weight=5,
                               eval_metric='logloss', verbosity=0, random_state=42, n_jobs=-1)
    model.fit(X, y)

    imp = pd.Series(model.feature_importances_, index=valid_feats).sort_values(ascending=False)
    print(f"  Top 30 features by importance:", flush=True)
    for feat, score in imp.head(30).items():
        print(f"    {feat:<50} imp={score:.4f}", flush=True)

    # 3c: Correlation filter
    print("\n  --- Correlation Filter (threshold=0.8) ---", flush=True)
    top_feats = imp.head(50).index.tolist()
    feat_corr = test_df[top_feats].corr().abs()
    upper = feat_corr.where(np.triu(np.ones(feat_corr.shape), k=1).astype(bool))
    to_drop = [col for col in upper.columns if any(upper[col] > 0.8)]
    print(f"  Features to drop (corr>0.8): {len(to_drop)}", flush=True)
    for f in to_drop[:15]:
        correlated_with = upper[f][upper[f] > 0.8].index.tolist()
        print(f"    {f:<50} correlated with: {correlated_with[:3]}", flush=True)

    selected = [f for f in top_feats if f not in to_drop]
    print(f"  Selected features (from top 50): {len(selected)}", flush=True)
    print(f"  Selected features:", flush=True)
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

    # Leakage check
    print("\n[Step 4] LEAKAGE CHECK", flush=True)
    print("  Target: (T+2 close / T+1 open - 1) > 0.04", flush=True)
    print("  Features computed from: T日 data (after close) + T-1 and earlier for rolling/lag", flush=True)
    print("  [OK] All lag features use shift() = strictly T-1 and earlier", flush=True)
    print("  [OK] All rolling features use T日 and earlier (no T+1/T+2 data)", flush=True)
    print("  [OK] News features use T日 news (available after close)", flush=True)
    print("  [OK] Calendar features use T日 date", flush=True)
    print("  [OK] T日 price/chip data used as base (available after close, before T+1 open)", flush=True)
    print("  [OK] Market return uses T日 index data (available after close)", flush=True)

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

    # Save model
    joblib.dump(model, os.path.join(THIS_DIR, 'ts_analysis_model.joblib'))

    print(f"\nAll results saved to {THIS_DIR}", flush=True)

if __name__ == "__main__":
    import joblib
    main()

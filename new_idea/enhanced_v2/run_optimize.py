"""
Parameter optimization: find best prob_thresh, top_n, and feature subset
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_optimized import build_training_dataset_fast, train_xgboost, backtest_model
from train_and_backtest import BASE_FEATS, CHAN_FEATS, LYNCH_FEATS, QUANT_FEATS, ENHANCED_FEATS, THIS_DIR
import pandas as pd
import numpy as np
import joblib

df = build_training_dataset_fast()
if df is None:
    sys.exit(1)

print("\n" + "=" * 90)
print("PARAMETER OPTIMIZATION")
print("=" * 90)

# Step 1: Train Enhanced model with all features
print("\n[1] Training Enhanced model with all 32 features...")
enh_model, enh_fi = train_xgboost(df, ENHANCED_FEATS, "Enhanced")

# Step 2: Try different prob_thresh and top_n
print("\n[2] Grid search: prob_thresh x top_n...")
best_sharpe = -999
best_params = None

for prob_thresh in [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6]:
    for top_n in [1, 2, 3]:
        trades, equity, stats = backtest_model(
            enh_model, ENHANCED_FEATS, top_n=top_n, prob_thresh=prob_thresh)
        if stats and stats['n_trades'] >= 10:
            sharpe = stats['sharpe']
            print("  prob=%.2f top_n=%d: Sharpe=%.2f, Total=%.2f%%, WR=%.2f%%, Trades=%d, MaxDD=%.2f%%" % (
                prob_thresh, top_n, sharpe, stats['total_ret']*100,
                stats['win_rate']*100, stats['n_trades'], stats['max_dd']*100))
            if sharpe > best_sharpe:
                best_sharpe = sharpe
                best_params = (prob_thresh, top_n, stats)

print("\nBest params: prob_thresh=%.2f, top_n=%d" % (best_params[0], best_params[1]))
print("  Sharpe=%.2f, Total=%.2f%%, WR=%.2f%%, MaxDD=%.2f%%" % (
    best_params[2]['sharpe'], best_params[2]['total_ret']*100,
    best_params[2]['win_rate']*100, best_params[2]['max_dd']*100))

# Step 3: Feature subset selection - try removing low-importance features
print("\n[3] Feature subset optimization...")
top_feats_15 = enh_fi.head(15)['feature'].tolist()
top_feats_20 = enh_fi.head(20)['feature'].tolist()
top_feats_25 = enh_fi.head(25)['feature'].tolist()

prob_t = best_params[0]
top_n = best_params[1]

for name, feats in [("All_32", ENHANCED_FEATS), ("Top25", top_feats_25),
                     ("Top20", top_feats_20), ("Top15", top_feats_15)]:
    model, fi = train_xgboost(df, feats, name)
    trades, equity, stats = backtest_model(model, feats, top_n=top_n, prob_thresh=prob_t)
    if stats:
        print("  %-8s (%2d feats): Sharpe=%.2f, Total=%.2f%%, WR=%.2f%%, Trades=%d, MaxDD=%.2f%%" % (
            name, len(feats), stats['sharpe'], stats['total_ret']*100,
            stats['win_rate']*100, stats['n_trades'], stats['max_dd']*100))

# Step 4: Try different XGBoost hyperparameters
print("\n[4] XGBoost hyperparameter tuning...")
from xgboost import XGBClassifier

X = df[ENHANCED_FEATS].fillna(0).replace([np.inf, -np.inf], 0)
y = df['label']
pos_count = y.sum()
neg_count = len(y) - pos_count
scale_pos = neg_count / (pos_count + 1)

configs = [
    ("Default", dict(n_estimators=300, max_depth=6, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8)),
    ("Deep", dict(n_estimators=500, max_depth=8, learning_rate=0.03, subsample=0.7, colsample_bytree=0.7)),
    ("Shallow", dict(n_estimators=300, max_depth=4, learning_rate=0.1, subsample=0.9, colsample_bytree=0.9)),
    ("Regularized", dict(n_estimators=400, max_depth=5, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, reg_alpha=1.0, reg_lambda=2.0)),
]

for name, params in configs:
    model = XGBClassifier(
        scale_pos_weight=scale_pos,
        eval_metric='logloss',
        use_label_encoder=False,
        random_state=42,
        n_jobs=-1,
        **params
    )
    model.fit(X, y)
    trades, equity, stats = backtest_model(model, ENHANCED_FEATS, top_n=top_n, prob_thresh=prob_t)
    if stats:
        print("  %-12s: Sharpe=%.2f, Total=%.2f%%, WR=%.2f%%, Trades=%d, MaxDD=%.2f%%" % (
            name, stats['sharpe'], stats['total_ret']*100,
            stats['win_rate']*100, stats['n_trades'], stats['max_dd']*100))

# Step 5: Final best model
print("\n[5] Training final best model...")
final_model = XGBClassifier(
    n_estimators=500,
    max_depth=8,
    learning_rate=0.03,
    subsample=0.7,
    colsample_bytree=0.7,
    scale_pos_weight=scale_pos,
    eval_metric='logloss',
    use_label_encoder=False,
    random_state=42,
    n_jobs=-1,
)
final_model.fit(X, y)

trades, equity, stats = backtest_model(final_model, ENHANCED_FEATS, top_n=top_n, prob_thresh=prob_t)
if stats:
    print("\nFINAL MODEL: Sharpe=%.2f, Total=%.2f%%, WR=%.2f%%, Trades=%d, MaxDD=%.2f%%" % (
        stats['sharpe'], stats['total_ret']*100,
        stats['win_rate']*100, stats['n_trades'], stats['max_dd']*100))

    joblib.dump(final_model, os.path.join(THIS_DIR, "best_model_v2.joblib"))
    joblib.dump(ENHANCED_FEATS, os.path.join(THIS_DIR, "best_feats_v2.joblib"))
    pd.DataFrame(trades).to_csv(os.path.join(THIS_DIR, "trades_final_best.csv"), index=False)
    pd.DataFrame({"equity": equity}).to_csv(os.path.join(THIS_DIR, "equity_final_best.csv"), index=False)
    print("Saved best_model_v2.joblib")

    fi = pd.DataFrame({'feature': ENHANCED_FEATS, 'importance': final_model.feature_importances_})
    fi = fi.sort_values('importance', ascending=False)
    fi.to_csv(os.path.join(THIS_DIR, "feature_importance_final.csv"), index=False)

    print("\nFinal Feature Importance (Top 20):")
    for _, row in fi.head(20).iterrows():
        feat = row['feature']
        if feat in CHAN_FEATS: cat = 'CHAN'
        elif feat in LYNCH_FEATS: cat = 'LYNCH'
        elif feat in QUANT_FEATS: cat = 'QUANT'
        else: cat = 'BASE'
        print("  %-25s [%s]: %.4f" % (feat, cat, row['importance']))

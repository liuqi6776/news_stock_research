import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from train_and_backtest import (
    build_training_dataset, train_xgboost, backtest,
    BASE_FEATS, CHAN_FEATS, LYNCH_FEATS, QUANT_FEATS, ENHANCED_FEATS,
    THIS_DIR
)
import joblib

print("Step 1: Building training dataset...")
df = build_training_dataset()
if df is None:
    print("FAILED: No training data")
    sys.exit(1)

print("\nStep 2: Training Baseline Model (5 features)...")
base_model, base_fi = train_xgboost(df, BASE_FEATS, "Baseline")

print("\nStep 3: Training Enhanced Model (all features)...")
enh_model, enh_fi = train_xgboost(df, ENHANCED_FEATS, "Enhanced")

print("\nStep 4: Training Chan-only Model (5 + Chan)...")
chan_feats = BASE_FEATS + CHAN_FEATS
chan_model, chan_fi = train_xgboost(df, chan_feats, "Chan")

print("\nStep 5: Training Quant-only Model (5 + Quant)...")
quant_feats = BASE_FEATS + QUANT_FEATS
quant_model, quant_fi = train_xgboost(df, quant_feats, "Quant")

print("\nStep 6: Backtesting Baseline...")
base_trades, base_eq, base_stats = backtest(base_model, BASE_FEATS, top_n=1, prob_thresh=0.4)

print("\nStep 7: Backtesting Enhanced...")
enh_trades, enh_eq, enh_stats = backtest(enh_model, ENHANCED_FEATS, top_n=1, prob_thresh=0.4)

print("\nStep 8: Backtesting Chan-only...")
chan_trades, chan_eq, chan_stats = backtest(chan_model, chan_feats, top_n=1, prob_thresh=0.4)

print("\nStep 9: Backtesting Quant-only...")
quant_trades, quant_eq, quant_stats = backtest(quant_model, quant_feats, top_n=1, prob_thresh=0.4)

print("\n\n" + "=" * 60)
print("COMPARISON SUMMARY")
print("=" * 60)

results = {}
for name, stats in [("Baseline_5feat", base_stats), ("Enhanced_All", enh_stats),
                     ("Chan_Only", chan_stats), ("Quant_Only", quant_stats)]:
    if stats:
        results[name] = stats
        print(f"  {name:20s}: Sharpe={stats['sharpe']:.2f}, Total={stats['total_ret']:.2%}, "
              f"WR={stats['win_rate']:.2%}, Trades={stats['n_trades']}, MaxDD={stats['max_dd']:.2%}")

best_name = max(results, key=lambda k: results[k]["sharpe"]) if results else None
if best_name:
    print(f"\nBest model by Sharpe: {best_name} (Sharpe={results[best_name]['sharpe']:.2f})")

    model_map = {
        "Baseline_5feat": (base_model, BASE_FEATS),
        "Enhanced_All": (enh_model, ENHANCED_FEATS),
        "Chan_Only": (chan_model, chan_feats),
        "Quant_Only": (quant_model, quant_feats),
    }
    best_model, best_feats = model_map[best_name]
    joblib.dump(best_model, os.path.join(THIS_DIR, "best_model.joblib"))
    joblib.dump(best_feats, os.path.join(THIS_DIR, "best_feats.joblib"))
    print(f"Best model saved: best_model.joblib, features: {best_feats}")

enh_fi.to_csv(os.path.join(THIS_DIR, "feature_importance_enhanced.csv"), index=False)

import pandas as pd
summary = pd.DataFrame(results).T
summary.to_csv(os.path.join(THIS_DIR, "comparison_summary.csv"))

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_optimized import build_training_dataset_fast, train_xgboost, backtest_model
from train_and_backtest import BASE_FEATS, QUANT_FEATS, CHAN_FEATS, LYNCH_FEATS, ENHANCED_FEATS, THIS_DIR
import pandas as pd
import numpy as np
import joblib

df = build_training_dataset_fast()
if df is None:
    sys.exit(1)

quant_feats = BASE_FEATS + QUANT_FEATS
print("\nTraining Quant-only Model (%d features)..." % len(quant_feats))
quant_model, quant_fi = train_xgboost(df, quant_feats, "Quant")

print("\nBacktesting Quant-only...")
trades, equity, stats = backtest_model(quant_model, quant_feats, top_n=1, prob_thresh=0.4)

if stats:
    print("Quant_Only: Sharpe=%.2f, Total=%.2f%%, WR=%.2f%%, Trades=%d, MaxDD=%.2f%%" % (
        stats['sharpe'], stats['total_ret']*100, stats['win_rate']*100,
        stats['n_trades'], stats['max_dd']*100))
    pd.DataFrame(trades).to_csv(os.path.join(THIS_DIR, "trades_Quant_Only.csv"), index=False)
    pd.DataFrame({"equity": equity}).to_csv(os.path.join(THIS_DIR, "equity_Quant_Only.csv"), index=False)

print("\n" + "=" * 80)
print("FULL COMPARISON (ALL 5 MODELS)")
print("=" * 80)

models = ['Baseline_5feat', 'Enhanced_All', 'Chan_Only', 'Lynch_Only', 'Quant_Only']
print("%-20s  %8s  %10s  %8s  %8s  %6s  %8s" % (
    'Model', 'Sharpe', 'TotalRet', 'WinRate', 'AvgRet', 'Trades', 'MaxDD'))
print("-" * 80)

for m in models:
    eq_file = os.path.join(THIS_DIR, "equity_%s.csv" % m)
    tr_file = os.path.join(THIS_DIR, "trades_%s.csv" % m)
    try:
        eq = pd.read_csv(eq_file)
        tr = pd.read_csv(tr_file)
        eq_arr = eq["equity"].values
        rets = np.diff(eq_arr) / eq_arr[:-1]
        total_ret = eq_arr[-1] / eq_arr[0] - 1
        sharpe = np.mean(rets) / (np.std(rets) + 1e-8) * np.sqrt(252)
        max_dd = np.max(1 - eq_arr / np.maximum.accumulate(eq_arr))
        win_rate = (tr["ret"] > 0).mean()
        avg_ret = tr["ret"].mean()
        n_trades = len(tr)
        print("%-20s  %8.2f  %9.2f%%  %7.2f%%  %7.2f%%  %6d  %7.2f%%" % (
            m, sharpe, total_ret*100, win_rate*100, avg_ret*100, n_trades, max_dd*100))
    except Exception as e:
        print("%-20s  Error: %s" % (m, e))

# Feature importance for Enhanced model
print("\nTraining Enhanced model for feature importance...")
enh_model, enh_fi = train_xgboost(df, ENHANCED_FEATS, "Enhanced_FI")

print("\nTop 20 Feature Importance (Enhanced Model):")
for _, row in enh_fi.head(20).iterrows():
    feat = row['feature']
    category = 'BASE'
    if feat in CHAN_FEATS:
        category = 'CHAN'
    elif feat in LYNCH_FEATS:
        category = 'LYNCH'
    elif feat in QUANT_FEATS:
        category = 'QUANT'
    print("  %-25s [%s]: %.4f" % (feat, category, row['importance']))

# Category contribution
cat_imp = {}
for _, row in enh_fi.iterrows():
    feat = row['feature']
    if feat in CHAN_FEATS:
        cat = 'CHAN'
    elif feat in LYNCH_FEATS:
        cat = 'LYNCH'
    elif feat in QUANT_FEATS:
        cat = 'QUANT'
    else:
        cat = 'BASE'
    cat_imp[cat] = cat_imp.get(cat, 0) + row['importance']

print("\nFeature Category Contribution:")
for cat, imp in sorted(cat_imp.items(), key=lambda x: -x[1]):
    print("  %-8s: %.4f (%.1f%%)" % (cat, imp, imp*100))

# Save best model
best_name = 'Enhanced_All'
joblib.dump(enh_model, os.path.join(THIS_DIR, "best_model.joblib"))
joblib.dump(ENHANCED_FEATS, os.path.join(THIS_DIR, "best_feats.joblib"))
enh_fi.to_csv(os.path.join(THIS_DIR, "feature_importance_enhanced.csv"), index=False)
print("\nSaved best_model.joblib and feature_importance_enhanced.csv")

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_and_backtest import build_training_dataset, train_xgboost, backtest, BASE_FEATS, QUANT_FEATS
import joblib
import pandas as pd
import numpy as np

print("Building training dataset...")
df = build_training_dataset()
if df is None:
    sys.exit(1)

quant_feats = BASE_FEATS + QUANT_FEATS
print("Training Quant-only Model...")
quant_model, quant_fi = train_xgboost(df, quant_feats, "Quant")

print("Backtesting Quant-only...")
trades, equity, stats = backtest(quant_model, quant_feats, top_n=1, prob_thresh=0.4)

if stats:
    s = stats
    print("Quant_Only: Sharpe=%.2f, Total=%.2f%%, WR=%.2f%%, Trades=%d, MaxDD=%.2f%%" % (
        s["sharpe"], s["total_ret"]*100, s["win_rate"]*100, s["n_trades"], s["max_dd"]*100))
    pd.DataFrame(trades).to_csv("trades_Quant_Only.csv", index=False)
    pd.DataFrame({"equity": equity}).to_csv("equity_Quant_Only.csv", index=False)

# Full comparison
print("\n" + "=" * 70)
print("FULL COMPARISON SUMMARY")
print("=" * 70)

models = ["Baseline_5feat", "Enhanced_All", "Chan_Only", "Lynch_Only", "Quant_Only"]
for m in models:
    eq_file = "equity_%s.csv" % m
    tr_file = "trades_%s.csv" % m
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
        print("  %-20s: Sharpe=%6.2f, Total=%8.2f%%, WR=%5.2f%%, AvgRet=%5.2f%%, Trades=%3d, MaxDD=%5.2f%%" % (
            m, sharpe, total_ret*100, win_rate*100, avg_ret*100, n_trades, max_dd*100))
    except Exception as e:
        print("  %-20s: Error - %s" % (m, e))

# Feature importance for Enhanced model
print("\nTop 15 Feature Importance (Enhanced Model):")
enh_fi = quant_fi.sort_values("importance", ascending=False).head(15)
for _, row in enh_fi.iterrows():
    print("  %-25s: %.4f" % (row["feature"], row["importance"]))

# Save best model
best_sharpe = -999
best_name = None
for m in models:
    try:
        eq = pd.read_csv("equity_%s.csv" % m)
        eq_arr = eq["equity"].values
        rets = np.diff(eq_arr) / eq_arr[:-1]
        sharpe = np.mean(rets) / (np.std(rets) + 1e-8) * np.sqrt(252)
        if sharpe > best_sharpe:
            best_sharpe = sharpe
            best_name = m
    except:
        pass

print("\nBest model by Sharpe: %s (Sharpe=%.2f)" % (best_name, best_sharpe))

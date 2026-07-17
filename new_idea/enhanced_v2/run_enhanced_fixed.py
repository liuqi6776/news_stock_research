import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_and_backtest import build_training_dataset, train_xgboost, backtest, ENHANCED_FEATS
import pandas as pd
import numpy as np

print("=" * 60)
print("重新回测 Enhanced_All (修正A股交易规则)")
print("=" * 60)

print("\nStep 1: Building training dataset...")
df = build_training_dataset()
if df is None:
    print("FAILED: No training data")
    sys.exit(1)

print("\nStep 2: Training Enhanced Model (all features)...")
enh_model, enh_fi = train_xgboost(df, ENHANCED_FEATS, "Enhanced")

print("\nStep 3: Backtesting Enhanced (修正版)...")
trades, equity, stats = backtest(enh_model, ENHANCED_FEATS, top_n=1, prob_thresh=0.4)

if stats:
    print("\n" + "=" * 60)
    print("修正后 Enhanced_All 结果")
    print("=" * 60)
    print(f"  Trades: {stats['n_trades']}")
    print(f"  Win Rate: {stats['win_rate']:.2%}")
    print(f"  Avg Return: {stats['avg_ret']:.2%}")
    print(f"  Total Return: {stats['total_ret']:.2%}")
    print(f"  Sharpe: {stats['sharpe']:.2f}")
    print(f"  Max Drawdown: {stats['max_dd']:.2%}")
    print(f"  Final Equity: {stats['final_equity']:,.0f}")

    # 保存结果
    trades_df = pd.DataFrame(trades)
    trades_df.to_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "trades_Enhanced_All_Fixed.csv"), index=False)
    pd.DataFrame({"equity": equity}).to_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "equity_Enhanced_All_Fixed.csv"), index=False)
    print("\n结果已保存至 trades_Enhanced_All_Fixed.csv 和 equity_Enhanced_All_Fixed.csv")

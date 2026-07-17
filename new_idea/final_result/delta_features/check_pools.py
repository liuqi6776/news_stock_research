import pandas as pd
import numpy as np

eq = pd.read_csv(r'c:\Users\liuqi\quant_system_v2\new_idea\final_result\doubao\equity.csv')
print(f'Doubao equity: {len(eq)} rows')
total_ret = eq['nav'].iloc[-1] / eq['nav'].iloc[0] - 1
years = len(eq) / 252.0
ann_ret = (1 + total_ret) ** (1 / years) - 1
df_ret = eq['nav'].pct_change()
mdd = ((eq['nav'] - eq['nav'].cummax()) / eq['nav'].cummax()).min()
vol = df_ret.std() * np.sqrt(252)
sharpe = ann_ret / vol if vol > 0 else 0
print(f'Total return: {total_ret:.2%}, Ann: {ann_ret:.2%}, Sharpe: {sharpe:.2f}, MDD: {mdd:.2%}')

trades = pd.read_csv(r'c:\Users\liuqi\quant_system_v2\new_idea\final_result\doubao\trades.csv')
print(f'Trades: {len(trades)} rows')
print(f'Prob stats: mean={trades["prob"].mean():.4f}, std={trades["prob"].std():.4f}')
print(f'Prob > 0.4: {(trades["prob"] > 0.4).sum()}')
print(f'Prob > 0.5: {(trades["prob"] > 0.5).sum()}')
print(f'Trades cols: {list(trades.columns)}')

# Check v8 pool
v8 = pd.read_csv(r'c:\Users\liuqi\quant_system_v2\new_idea\final_result\delta_features\pool_ts_v8.csv')
print(f'\nV8 pool: {len(v8)} rows')
print(f'base_prob stats: mean={v8["base_prob"].mean():.4f}, std={v8["base_prob"].std():.4f}')
print(f'base_prob > 0.4: {(v8["base_prob"] > 0.4).sum()}')
print(f'base_prob > 0.5: {(v8["base_prob"] > 0.5).sum()}')

# Check enhanced pool
import os
enhanced_path = r'c:\Users\liuqi\quant_system_v2\new_idea\final_result\enhanced\pool_base_trades.csv'
if os.path.exists(enhanced_path):
    enh = pd.read_csv(enhanced_path)
    print(f'\nEnhanced pool: {len(enh)} rows')
    print(f'Prob stats: mean={enh["prob"].mean():.4f}, std={enh["prob"].std():.4f}')
    print(f'Prob > 0.4: {(enh["prob"] > 0.4).sum()}')
    print(f'Prob > 0.5: {(enh["prob"] > 0.5).sum()}')

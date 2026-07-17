import pandas as pd
import numpy as np

csv_path = r"C:\Users\liuqi\quant_system_v2\results_duobao\real_t1_existing_model_equity.csv"
df = pd.read_csv(csv_path)

initial_cap = 100000.0
final_cap = df['nav'].iloc[-1]

total_ret = final_cap / initial_cap - 1
years = len(df) / 252.0
ann_ret = (1 + total_ret) ** (1 / years) - 1 if years > 0 else 0
df_ret = df['nav'].pct_change()
mdd = ((df['nav'] - df['nav'].cummax()) / df['nav'].cummax()).min()
vol = df_ret.std() * np.sqrt(252)
sharpe = ann_ret / vol if vol > 0 else 0

print("="*80)
print("REAL T+1 STRATEGY - 最新回测结果")
print("="*80)
print(f"Initial Capital:  ¥{initial_cap:,.2f}")
print(f"Final Capital:    ¥{final_cap:,.2f}")
print(f"Total Return:     {total_ret:+.2%}")
print(f"Annual Return:    {ann_ret:+.2%}")
print(f"Max Drawdown:     {mdd:.2%}")
print(f"Sharpe Ratio:     {sharpe:.2f}")
print(f"Number of Trades: {len(df)}")
print(f"Start Date:       {df['date'].iloc[0]}")
print(f"End Date:         {df['date'].iloc[-1]}")
print("="*80 + "\n")

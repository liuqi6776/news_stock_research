import pandas as pd
import matplotlib.pyplot as plt
import os

# Create artifacts directory if not exists
artifact_dir = r"C:\Users\liuqi\.gemini\antigravity\brain\02be0ea3-cfcb-4a7a-b56a-03b516b44ee9"
os.makedirs(artifact_dir, exist_ok=True)

# Load data
csv_path = r"C:\Users\liuqi\quant_system_v2\backtesting\ptrade_equity.csv"
if not os.path.exists(csv_path):
    print(f"Error: {csv_path} not found.")
    exit(1)

df = pd.read_csv(csv_path)
df['date'] = pd.to_datetime(df['date'].astype(str))
df.set_index('date', inplace=True)

# Plot
plt.figure(figsize=(12, 6))
df['assets'].plot(label='Total Assets')
plt.title('PTrade Logic Backtest Equity Curve (2023-Now)')
plt.xlabel('Date')
plt.ylabel('Assets (RMB)')
plt.grid(True)
plt.legend()

# Save locally to artifacts
output_path = os.path.join(artifact_dir, "ptrade_equity_curve_2023_2026.png")
plt.savefig(output_path)
print(f"Plot saved to: {output_path}")

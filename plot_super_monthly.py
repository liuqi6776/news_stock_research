
import pandas as pd
import matplotlib.pyplot as plt
import os

OUT_DIR = r'C:\Users\liuqi\quant_system_v2'
csv_path = os.path.join(OUT_DIR, 'super_monthly_equity.csv')
eq_df = pd.read_parquet(csv_path) if csv_path.endswith('parquet') else pd.read_csv(csv_path)

plt.figure(figsize=(12, 7))
plt.plot(pd.to_datetime(eq_df['date']), eq_df['nav'], label='Super-Monthly Strategy', color='#1f77b4', linewidth=2)
plt.axhline(y=100000, color='gray', linestyle='--', alpha=0.5)
plt.title('Super-Monthly Strategy Performance (2023-2025)\nStrict T+1 | Concentrated Alpha', fontsize=14)
plt.xlabel('Date')
plt.ylabel('Net Asset Value (NAV)')
plt.grid(True, which='both', linestyle='--', alpha=0.5)
plt.legend()

save_path = os.path.join(OUT_DIR, 'super_monthly_performance.png')
plt.savefig(save_path, dpi=300, bbox_inches='tight')
print(f"Plot saved to {save_path}")

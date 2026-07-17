
import pandas as pd
import matplotlib.pyplot as plt
import os

OUT_DIR = r'C:\Users\liuqi\quant_system_v2'
paths = {
    'Super-Monthly': os.path.join(OUT_DIR, 'super_monthly_equity.csv'),
    'Super-Weekly': os.path.join(OUT_DIR, 'super_weekly_equity.csv')
}

plt.figure(figsize=(12, 7))
for label, path in paths.items():
    df = pd.read_csv(path)
    plt.plot(pd.to_datetime(df['date']), df['nav'], label=label, linewidth=2)

plt.axhline(y=100000, color='gray', linestyle='--', alpha=0.5)
plt.title('Performance Comparison: Super-Monthly vs Super-Weekly (2023-2025)\nConcentrated Monster Gene Alpha', fontsize=14)
plt.xlabel('Date')
plt.ylabel('NAV')
plt.grid(True, alpha=0.3)
plt.legend()

save_path = os.path.join(OUT_DIR, 'freq_comparison_v7.png')
plt.savefig(save_path, dpi=300)
print(f"Chart saved to {save_path}")

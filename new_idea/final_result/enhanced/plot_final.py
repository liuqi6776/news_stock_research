import pandas as pd, os, numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

THIS_DIR = os.path.dirname(os.path.abspath(__file__))

top_schemes = [
    'S4_Top1_P04',
    'S4_Top1_P04_TP20',
    'S4_Top1_P04_TP18',
    'S4_Top2_P04_TP18',
    'S4_Top2_P04',
    'S0_Baseline',
    'S2_TP20',
    'S3_ProbWeighted',
]

doubao_eq_path = os.path.join(os.path.dirname(THIS_DIR), 'doubao', 'equity.csv')

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

fig, axes = plt.subplots(2, 1, figsize=(18, 14))

# Plot 1: Top schemes vs doubao_result
if os.path.exists(doubao_eq_path):
    doubao_eq = pd.read_csv(doubao_eq_path)
    doubao_eq['date'] = pd.to_datetime(doubao_eq['date'])
    doubao_norm = doubao_eq['nav'] / doubao_eq['nav'].iloc[0]
    axes[0].plot(doubao_eq['date'], doubao_norm, label='doubao_result (ref): S=5.49 R=+2866%', linewidth=2.5, color='black', linestyle='--')

colors = plt.cm.tab10(np.linspace(0, 1, len(top_schemes)))
for i, sname in enumerate(top_schemes):
    eq_path = os.path.join(THIS_DIR, f'equity_{sname}.csv')
    if not os.path.exists(eq_path):
        continue
    eq = pd.read_csv(eq_path)
    eq['date'] = pd.to_datetime(eq['date'])
    eq_norm = eq['nav'] / eq['nav'].iloc[0]
    final_ret = eq['nav'].iloc[-1] / 100000.0 - 1
    years = len(eq) / 252.0
    ann_ret = (1 + final_ret) ** (1 / years) - 1 if years > 0 else 0
    df_ret = eq['nav'].pct_change()
    vol = df_ret.std() * np.sqrt(252)
    sharpe = ann_ret / vol if vol > 0 else 0
    label = f"{sname}: S={sharpe:.2f} R={final_ret:+.0%}"
    axes[0].plot(eq['date'], eq_norm, label=label, linewidth=1.5, color=colors[i])

axes[0].set_title('Enhanced Strategies vs doubao_result Baseline', fontsize=14, fontweight='bold')
axes[0].set_ylabel('NAV (normalized)')
axes[0].legend(fontsize=8, loc='upper left')
axes[0].grid(True, alpha=0.3)
axes[0].axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)

# Plot 2: Top 3 only with detailed NAV
top3 = ['S4_Top1_P04', 'S4_Top2_P04_TP18', 'S0_Baseline']
if os.path.exists(doubao_eq_path):
    doubao_eq = pd.read_csv(doubao_eq_path)
    doubao_eq['date'] = pd.to_datetime(doubao_eq['date'])
    axes[1].plot(doubao_eq['date'], doubao_eq['nav'], label='doubao_result', linewidth=2.5, color='black', linestyle='--')

for i, sname in enumerate(top3):
    eq_path = os.path.join(THIS_DIR, f'equity_{sname}.csv')
    if not os.path.exists(eq_path):
        continue
    eq = pd.read_csv(eq_path)
    eq['date'] = pd.to_datetime(eq['date'])
    axes[1].plot(eq['date'], eq['nav'], label=sname, linewidth=2)

axes[1].set_title('Top 3 Enhanced vs doubao_result (Absolute NAV)', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Date')
axes[1].set_ylabel('NAV (¥)')
axes[1].legend(fontsize=10)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(THIS_DIR, 'enhanced_final_comparison.png'), dpi=150, bbox_inches='tight')
print('Chart saved', flush=True)

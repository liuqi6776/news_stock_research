import pandas as pd, os, numpy as np

THIS_DIR = r'c:\Users\liuqi\quant_system_v2\new_idea\final_result\delta_features'

results = {}
for f in os.listdir(THIS_DIR):
    if f.startswith('equity_v3_') and f.endswith('.csv'):
        sname = f.replace('equity_v3_','').replace('.csv','')
        eq = pd.read_csv(os.path.join(THIS_DIR, f))
        if len(eq) == 0:
            continue
        final = eq['nav'].iloc[-1]
        total_ret = final / 100000.0 - 1
        years = len(eq) / 252.0
        ann_ret = (1 + total_ret) ** (1 / years) - 1 if years > 0 else 0
        dr = eq['nav'].pct_change()
        mdd = ((eq['nav'] - eq['nav'].cummax()) / eq['nav'].cummax()).min()
        vol = dr.std() * np.sqrt(252)
        sharpe = ann_ret / vol if vol > 0 else 0
        calmar = ann_ret / abs(mdd) if mdd != 0 else 0
        results[sname] = {'total': total_ret, 'sharpe': sharpe, 'mdd': mdd, 'calmar': calmar, 'trades': len(eq), 'win_rate': (dr > 0).mean()}

sorted_r = sorted(results.items(), key=lambda x: x[1]['sharpe'], reverse=True)
print(f"{'Rank':>4} {'Scheme':<35} {'Total':>10} {'Sharpe':>8} {'MDD':>10} {'Calmar':>8} {'WinRate':>8}")
print('-' * 90)
for rank, (sname, s) in enumerate(sorted_r, 1):
    print(f"{rank:>4} {sname:<35} {s['total']:>9.2%} {s['sharpe']:>7.2f} {s['mdd']:>9.2%} {s['calmar']:>7.2f} {s['win_rate']:>7.2%}")

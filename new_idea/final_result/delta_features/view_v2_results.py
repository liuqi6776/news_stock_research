import pandas as pd, os, numpy as np

THIS_DIR = os.path.dirname(os.path.abspath(__file__))

results = {}
for f in os.listdir(THIS_DIR):
    if f.startswith('equity_') and f.endswith('.csv'):
        sname = f.replace('equity_','').replace('.csv','')
        eq = pd.read_csv(os.path.join(THIS_DIR, f))
        if len(eq) == 0:
            continue
        initial = 100000.0
        final = eq['nav'].iloc[-1]
        total_ret = final / initial - 1
        years = len(eq) / 252.0
        ann_ret = (1 + total_ret) ** (1 / years) - 1 if years > 0 else 0
        df_ret = eq['nav'].pct_change()
        mdd = ((eq['nav'] - eq['nav'].cummax()) / eq['nav'].cummax()).min()
        vol = df_ret.std() * np.sqrt(252)
        sharpe = ann_ret / vol if vol > 0 else 0
        calmar = ann_ret / abs(mdd) if mdd != 0 else 0
        win_rate = (df_ret > 0).mean()

        trades_f = f.replace('equity_','trades_')
        trades_path = os.path.join(THIS_DIR, trades_f)
        n_trades = len(pd.read_csv(trades_path)) if os.path.exists(trades_path) else 0

        results[sname] = {
            'total': total_ret, 'sharpe': sharpe, 'mdd': mdd,
            'calmar': calmar, 'trades': n_trades, 'win_rate': win_rate
        }

sorted_by_sharpe = sorted(results.items(), key=lambda x: x[1]['sharpe'], reverse=True)

print(f"{'Rank':>4} {'Scheme':<40} {'Total':>10} {'Sharpe':>8} {'MDD':>10} {'Calmar':>8} {'Trades':>8} {'WinRate':>8}")
print('-' * 100)
for rank, (sname, s) in enumerate(sorted_by_sharpe, 1):
    print(f"{rank:>4} {sname:<40} {s['total']:>9.2%} {s['sharpe']:>7.2f} {s['mdd']:>9.2%} {s['calmar']:>7.2f} {s['trades']:>7d} {s['win_rate']:>7.2%}")

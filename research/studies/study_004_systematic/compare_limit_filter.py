import json

old = json.load(open('results/1d5d_grid_results.json'))
new = json.load(open('results/1d5d_grid_limit_filter_results.json'))

print("=" * 100)
print("COMPARISON: WITHOUT vs WITH Limit Up/Down Filter")
print("=" * 100)

combos_to_compare = [
    ("1d_open: th=0.55 pos=3 no-sl no-tp", "1d th=0.55 pos=3"),
    ("1d_open: th=0.5 pos=3 no-sl no-tp", "1d th=0.5 pos=3"),
    ("1d_open: th=0.6 pos=3 no-sl no-tp", "1d th=0.6 pos=3"),
    ("1d_open: th=0.55 pos=5 no-sl no-tp", "1d th=0.55 pos=5"),
    ("5d_open: th=0.5 pos=3 no-sl no-tp", "5d th=0.5 pos=3"),
    ("5d_open: th=0.55 pos=3 no-sl no-tp", "5d th=0.55 pos=3"),
    ("5d_open: th=0.5 pos=5 no-sl no-tp", "5d th=0.5 pos=5"),
]

for label, short_name in combos_to_compare:
    print(f"\n{'='*80}")
    print(f"  {short_name}")
    print(f"{'='*80}")
    for period_name in ['test_2025_2026', 'full_2022_2026']:
        key_old = f"{label} | {period_name}"
        key_new = f"{label} | {period_name}"
        o = old.get(key_old, {})
        n = new.get(key_new, {})
        if not o or not n:
            continue
        cagr_old = o.get('cagr', 0)
        cagr_new = n.get('cagr', 0)
        sharpe_old = o.get('sharpe', 0)
        sharpe_new = n.get('sharpe', 0)
        maxdd_old = o.get('max_dd', 0)
        maxdd_new = n.get('max_dd', 0)
        wr_old = o.get('win_rate_days', 0)
        wr_new = n.get('win_rate_days', 0)
        print(f"  {period_name}:")
        print(f"    CAGR:   {cagr_old:.1%} -> {cagr_new:.1%}  (diff: {cagr_new - cagr_old:.1%})")
        print(f"    Sharpe: {sharpe_old:.2f} -> {sharpe_new:.2f}  (diff: {sharpe_new - sharpe_old:.2f})")
        print(f"    MaxDD:  {maxdd_old:.1%} -> {maxdd_new:.1%}")
        print(f"    WinRate:{wr_old:.1%} -> {wr_new:.1%}")

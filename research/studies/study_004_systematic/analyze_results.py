import json

results = json.load(open('results/1d5d_grid_results.json'))

print("=== Test Period (2025-2026) ===")
print()
print("1d_open strategies:")
for key, val in results.items():
    if "1d_open" in key and "test_2025" in key:
        label = val["label"]
        cagr = val["cagr"]
        sharpe = val["sharpe"]
        max_dd = val["max_dd"]
        wr = val["win_rate_days"]
        print(f"  {label}: CAGR={cagr:.1%}, Sharpe={sharpe:.2f}, MaxDD={max_dd:.1%}, WR={wr:.1%}")

print()
print("5d_open strategies:")
for key, val in results.items():
    if "5d_open" in key and "test_2025" in key:
        label = val["label"]
        cagr = val["cagr"]
        sharpe = val["sharpe"]
        max_dd = val["max_dd"]
        wr = val["win_rate_days"]
        print(f"  {label}: CAGR={cagr:.1%}, Sharpe={sharpe:.2f}, MaxDD={max_dd:.1%}, WR={wr:.1%}")

print()
print("=== Full Period (2022-2026) ===")
print()
print("1d_open strategies:")
for key, val in results.items():
    if "1d_open" in key and "full_2022" in key:
        label = val["label"]
        cagr = val["cagr"]
        sharpe = val["sharpe"]
        max_dd = val["max_dd"]
        print(f"  {label}: CAGR={cagr:.1%}, Sharpe={sharpe:.2f}, MaxDD={max_dd:.1%}")

print()
print("5d_open strategies:")
for key, val in results.items():
    if "5d_open" in key and "full_2022" in key:
        label = val["label"]
        cagr = val["cagr"]
        sharpe = val["sharpe"]
        max_dd = val["max_dd"]
        print(f"  {label}: CAGR={cagr:.1%}, Sharpe={sharpe:.2f}, MaxDD={max_dd:.1%}")

print()
print("=== Opt Period (2022-2024) ===")
print()
print("1d_open strategies:")
for key, val in results.items():
    if "1d_open" in key and "opt_2022" in key:
        label = val["label"]
        cagr = val["cagr"]
        sharpe = val["sharpe"]
        max_dd = val["max_dd"]
        print(f"  {label}: CAGR={cagr:.1%}, Sharpe={sharpe:.2f}, MaxDD={max_dd:.1%}")

print()
print("5d_open strategies:")
for key, val in results.items():
    if "5d_open" in key and "opt_2022" in key:
        label = val["label"]
        cagr = val["cagr"]
        sharpe = val["sharpe"]
        max_dd = val["max_dd"]
        print(f"  {label}: CAGR={cagr:.1%}, Sharpe={sharpe:.2f}, MaxDD={max_dd:.1%}")

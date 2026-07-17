import re
with open(r"C:\Users\liuqi\iquant\quant_trading_system\all_script_V2_run_week.py", "r", encoding="utf-8", errors="ignore") as f:
    lines = f.readlines()

pat = re.compile(r'[a-f0-9]{55,56}')
for i, line in enumerate(lines, 1):
    m = pat.findall(line)
    if m:
        print(f"Line {i}: {line.strip()} (found {m})")

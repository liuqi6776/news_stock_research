with open(r"C:\Users\liuqi\iquant\quant_trading_system\Try.ipynb", "r", encoding="utf-8") as f:
    for i, line in enumerate(f, 1):
        if "b214" in line or "7e47" in line or "f716" in line or "ce16" in line:
            print(f"Line {i}: {line.strip()}")

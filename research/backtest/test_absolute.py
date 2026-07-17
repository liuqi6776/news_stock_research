import os

# Write to absolute path
output_path = r"c:\Users\liuqi\quant_system_v2\research\backtest\absolute_test.txt"
with open(output_path, "w") as f:
    f.write("This is a test\n")
    f.write(f"Current directory: {os.getcwd()}\n")

print(f"File written to: {output_path}")

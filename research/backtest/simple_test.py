import os

output_path = r"c:\Users\liuqi\quant_system_v2\research\backtest\simple_test_output.txt"
with open(output_path, "w") as f:
    f.write("Python is working!\n")
    f.write(f"Current directory: {os.getcwd()}\n")

print(f"Test output written to: {output_path}")

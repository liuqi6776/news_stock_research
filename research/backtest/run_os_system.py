import os

# 使用os.system运行命令
ret = os.system(r'python -c "print(\'Hello from os.system\')" > c:\Users\liuqi\quant_system_v2\research\backtest\os_system_output.txt')

with open(r"c:\Users\liuqi\quant_system_v2\research\backtest\os_system_result.txt", "w") as f:
    f.write(f"Return code: {ret}\n")

print("os.system completed")

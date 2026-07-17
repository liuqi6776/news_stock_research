import sys
import os

# 读取并执行回测代码
with open(r"c:\Users\liuqi\quant_system_v2\research\backtest\direct_backtest.py", "r", encoding="utf-8") as f:
    code = f.read()

# 执行代码
exec(code)

# 调用main函数
main()

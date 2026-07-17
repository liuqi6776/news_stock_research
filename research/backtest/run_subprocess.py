import subprocess
import sys

# 运行Python命令并捕获输出
result = subprocess.run(
    [sys.executable, "-c", "print('Hello from subprocess')"],
    capture_output=True,
    text=True
)

# 写入文件
with open(r"c:\Users\liuqi\quant_system_v2\research\backtest\subprocess_output.txt", "w") as f:
    f.write("STDOUT:\n")
    f.write(result.stdout)
    f.write("\nSTDERR:\n")
    f.write(result.stderr)
    f.write(f"\nExit code: {result.returncode}\n")

print("Subprocess completed")

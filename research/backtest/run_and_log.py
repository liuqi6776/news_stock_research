import subprocess
import sys

# Run the backtest and capture output
result = subprocess.run(
    [sys.executable, "walk_forward_v6_stream.py"],
    capture_output=True,
    text=True,
    cwd=r"c:\Users\liuqi\quant_system_v2\research\backtest"
)

# Write output to file
with open("v6_backtest_output.txt", "w", encoding="utf-8") as f:
    f.write("=== STDOUT ===\n")
    f.write(result.stdout)
    f.write("\n=== STDERR ===\n")
    f.write(result.stderr)
    f.write(f"\n=== EXIT CODE ===\n")
    f.write(str(result.returncode))

print(f"Backtest completed with exit code: {result.returncode}")
print(f"Output saved to v6_backtest_output.txt")

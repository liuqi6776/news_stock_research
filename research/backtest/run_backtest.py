import subprocess
import sys
import time

print("Starting backtest...")
process = subprocess.Popen(
    [sys.executable, "walk_forward_v6_stream.py"],
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True,
    bufsize=1
)

with open("backtest_live.log", "w", buffering=1) as f:
    for line in process.stdout:
        print(line, end='')
        f.write(line)
        f.flush()

process.wait()
print(f"\nProcess finished with exit code: {process.returncode}")

"""
V18 with real-time progress output to progress_v18.log
Redirects tqdm to file for monitoring.
"""
import os
import sys
import time
import io

# Capture all output
class Tee:
    def __init__(self, *files):
        self.files = files
    def write(self, data):
        for f in self.files:
            try:
                f.write(data)
            except:
                pass
    def flush(self):
        for f in self.files:
            try:
                f.flush()
            except:
                pass

PROGRESS_LOG = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'progress_v18.log')

# Clear previous log
with open(PROGRESS_LOG, 'w', encoding='utf-8') as f:
    f.write(f"V18 Progress Log - {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write("=" * 60 + "\n\n")

# Tee stdout/stderr to both console and log file
log_f = open(PROGRESS_LOG, 'a', encoding='utf-8')
sys.stdout = Tee(sys.__stdout__, log_f)
sys.stderr = Tee(sys.__stderr__, log_f)

# Now run V18 via subprocess (avoids BOM and import issues)
import subprocess
result = subprocess.run(
    [sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'run_super_weekly_v18.py')],
    cwd=os.path.dirname(os.path.abspath(__file__)),
    timeout=7200,
)
print(f"\nV18 Exit code: {result.returncode}")
log_f.close()

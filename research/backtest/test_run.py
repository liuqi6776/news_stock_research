import subprocess
import sys

# Test with a simple command
result = subprocess.run(
    [sys.executable, "-c", "print('Hello from subprocess')"],
    capture_output=True,
    text=True
)

with open("test_subprocess.txt", "w") as f:
    f.write("STDOUT:\n")
    f.write(result.stdout)
    f.write("\nSTDERR:\n")
    f.write(result.stderr)
    f.write(f"\nExit code: {result.returncode}\n")

print("Test completed")

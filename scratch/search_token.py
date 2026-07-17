import os
import re

search_dirs = [
    r"c:\Users\liuqi\quant_system_v2",
    r"C:\Users\liuqi\iquant\quant_trading_system"
]

# Tushare tokens are usually 56-char hex strings.
# Let's search for patterns like:
# 1. Any string that matches 'TUSHARE_TOKEN' or 'TOKEN ='
# 2. Any 56-character hex strings.
token_pat = re.compile(r'\b[a-f0-9]{56}\b')
token_var_pat = re.compile(r'tushare_token|token\s*=\s*[\'"][a-f0-9]{55,56}[\'"]', re.IGNORECASE)

print("Starting search...")

for base_dir in search_dirs:
    if not os.path.exists(base_dir):
        print(f"Directory not found: {base_dir}")
        continue
    print(f"\nSearching in: {base_dir}")
    for root, dirs, files in os.walk(base_dir):
        # Skip directories like .git, __pycache__, etc.
        dirs[:] = [d for d in dirs if d not in ('.git', '__pycache__', '.ipynb_checkpoints', '.openclaw', '.opencode', '.trae', 'scratch')]
        for file in files:
            if file.endswith(('.py', '.ipynb', '.env', '.example', '.md', '.txt', '.sh', '.bat')):
                filepath = os.path.join(root, file)
                try:
                    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                    
                    # Search by pattern
                    matches_token = token_pat.findall(content)
                    matches_var = token_var_pat.findall(content)
                    
                    if matches_token or matches_var:
                        print(f"File: {filepath}")
                        for i, line in enumerate(content.splitlines(), 1):
                            if token_pat.search(line) or token_var_pat.search(line) or 'tushare' in line.lower() and 'token' in line.lower():
                                # Mask token in output for security
                                masked_line = line
                                for m in token_pat.findall(line):
                                    masked_line = masked_line.replace(m, m[:4] + "****" + m[-4:])
                                print(f"  Line {i}: {masked_line.strip()}")
                except Exception as e:
                    print(f"Error reading {filepath}: {e}")

print("Search complete.")

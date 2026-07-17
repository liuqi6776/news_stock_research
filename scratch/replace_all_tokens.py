import os
import re
import json

new_token = "421ff94dd31be789aa7f95e61ad6fad5bcefa250a0b2c4d298224aa"

old_tokens = [
    "b214450029cfe30aa5909cfebf28c105027b2bb74228f50c1d65d14b",
    "8e64f95870a9a87f14b761c15526bf63142e0e56d8369c1a674ebc1f",
    "16a3d17ffb6f121fc62d9b9c1eea13934c96acabf53420661696d858",
    "7e47770802c6109df34270f907604f323a6f1d2df0c1a1796d110cf3",
    "7e47e25017560965ed54bca2ecfd24d8e4b482d7471694cf1dc40cf3",
    "f7161598d1d7b7542858c62daf359b46a633c85a8fa2e1875b371f99",
    "ce16de0bda429ad67e65607fbfebc2ad3712a13878de66e0022cb284",
    "21ac6ac27ce237f8f39ffc245fc44039df36a8f2e79bb586f0ed5d2d",
    "b97645ec0240cfe7acd56dd7bec6107b2bb6d848c872fdad4b91b7fb",
    "2caf2fa019c27819e33985af40a980a5ae0aa5524160105a0ab97332",
    "36c45ffd9aefc22ec2eafcd3c4f680c49ca81eb4f5573a0e6ca0c9bb",
    "901fcc23a544fd43539298a00cf43304dc0782428dab64e56511b4c4"
]

search_dirs = [
    r"c:\Users\liuqi\quant_system_v2",
    r"C:\Users\liuqi\iquant\quant_trading_system"
]

print("Starting global tushare token replacement...")

for base_dir in search_dirs:
    if not os.path.exists(base_dir):
        print(f"Directory not found: {base_dir}")
        continue
    
    print(f"\nProcessing directory: {base_dir}")
    for root, dirs, files in os.walk(base_dir):
        # Skip hidden/unnecessary directories
        dirs[:] = [d for d in dirs if d not in ('.git', '__pycache__', '.ipynb_checkpoints', '.openclaw', '.opencode', '.trae', 'scratch')]
        for file in files:
            filepath = os.path.join(root, file)
            
            # 1. Handle Jupyter Notebooks
            if file.endswith('.ipynb'):
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        nb = json.load(f)
                    
                    modified = [False]
                    
                    def replace_in_obj(obj):
                        if isinstance(obj, dict):
                            for k, v in obj.items():
                                if isinstance(v, (dict, list)):
                                    replace_in_obj(v)
                                elif isinstance(v, str):
                                    new_v = v
                                    for old in old_tokens:
                                        if old in new_v:
                                            new_v = new_v.replace(old, new_token)
                                            modified[0] = True
                                    obj[k] = new_v
                        elif isinstance(obj, list):
                            for i in range(len(obj)):
                                if isinstance(obj[i], (dict, list)):
                                    replace_in_obj(obj[i])
                                elif isinstance(obj[i], str):
                                    new_val = obj[i]
                                    for old in old_tokens:
                                        if old in new_val:
                                            new_val = new_val.replace(old, new_token)
                                            modified[0] = True
                                    obj[i] = new_val

                    replace_in_obj(nb)
                    
                    if modified[0]:
                        with open(filepath, 'w', encoding='utf-8') as f:
                            json.dump(nb, f, ensure_ascii=False, indent=1)
                        print(f"  [IPYNB] Updated tokens in: {filepath}")
                except Exception as e:
                    print(f"  [ERROR] Jupyter notebook {filepath}: {e}")
            
            # 2. Handle standard code files and config files
            elif file.endswith(('.py', '.env', '.example', '.sh', '.bat', '.md', '.txt')):
                try:
                    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                    
                    modified = False
                    new_content = content
                    for old in old_tokens:
                        if old in new_content:
                            new_content = new_content.replace(old, new_token)
                            modified = True
                    
                    if modified:
                        with open(filepath, 'w', encoding='utf-8') as f:
                            f.write(new_content)
                        print(f"  [TEXT] Updated tokens in: {filepath}")
                except Exception as e:
                    print(f"  [ERROR] Text file {filepath}: {e}")

print("\nGlobal token replacement complete.")

import os
import re
import json

new_token = "421ff94dd31be789aa7f95e61ad6fad5bcefa250a0b2c4d298224aa"

# List of old tokens we want to replace
old_tokens = [
    "b214450029cfe30aa5909cfebf28c105027b2bb74228f50c1d65d14b",
    "f7169bb8e2e92c4cd7a2333068e82ef6f9d3ec8d8396c21e54911f99",
    "ce167576579c8824be4c16d56d11f974cbbe3cf8ff0771cb8d9fb284",
    "7e47e25017560965ed54bca2ecfd24d8e4b482d7471694cf1dc40cf3",
    "8e64f95870a9a87f14b761c15526bf63142e0e56d8369c1a674ebc1f",
    "16a3d17ffb6f121fc62d9b9c1eea13934c96acabf53420661696d858"
]

search_dirs = [
    r"c:\Users\liuqi\quant_system_v2",
    r"C:\Users\liuqi\iquant\quant_trading_system"
]

print("Starting Jupyter Notebook token replacement...")

for base_dir in search_dirs:
    if not os.path.exists(base_dir):
        print(f"Directory not found: {base_dir}")
        continue
    
    for root, dirs, files in os.walk(base_dir):
        # Skip hidden/unnecessary directories
        dirs[:] = [d for d in dirs if d not in ('.git', '__pycache__', '.ipynb_checkpoints', '.openclaw', '.opencode', '.trae', 'scratch')]
        for file in files:
            if file.endswith('.ipynb'):
                filepath = os.path.join(root, file)
                try:
                    # Read as JSON to preserve structure and avoid regex corrupting JSON
                    with open(filepath, 'r', encoding='utf-8') as f:
                        nb = json.load(f)
                    
                    modified = [False]
                    
                    # Recursive function to find and replace strings in the JSON structure
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
                        print(f"Updated tokens in: {filepath}")
                    else:
                        print(f"No tokens to update in: {filepath}")
                        
                except Exception as e:
                    print(f"Error processing {filepath}: {e}")

print("Replacement complete.")

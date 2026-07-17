import os

files_to_clean = [
    r"c:\Users\liuqi\quant_system_v2\final_method\1_analyze_news.py",
    r"c:\Users\liuqi\quant_system_v2\jiayo-analysis\analyzer.py",
    r"c:\Users\liuqi\quant_system_v2\jiayo-analysis\debug_batch.py",
    r"c:\Users\liuqi\quant_system_v2\jiayo-analysis\test_analyze.py",
    r"c:\Users\liuqi\quant_system_v2\new_idea\final_result\doubao\doubao_use\1_analyze_news.py",
    r"c:\Users\liuqi\quant_system_v2\results_duobao\news_tools\analyzer.py",
    r"c:\Users\liuqi\quant_system_v2\results_duobao\toolstock\1_analyze_news.py"
]

target_key = "7c406ccb126c48e28758c255b9aede76.nTXKzG8O0EKO9YE9"

for filepath in files_to_clean:
    if os.path.exists(filepath):
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read()
            if target_key in content:
                print(f"Cleaning leaked key in: {filepath}")
                new_content = content.replace(target_key, "")
                with open(filepath, "w", encoding="utf-8") as f:
                    f.write(new_content)
            else:
                print(f"Key not found in: {filepath}")
        except Exception as e:
            print(f"Failed to clean {filepath}: {e}")
    else:
        print(f"File not found: {filepath}")

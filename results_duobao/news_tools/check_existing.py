import os
import json
import re

OUT_DIR = r"D:\iquant_data\data_v2\news_major1"
DATA_DIR = r"C:\Users\liuqi\clowspace\quant_system_v2\jiayo-analysis\data"


def extract_date_from_html(html):
    patterns = [
        r'(\d{4})-(\d{2})-(\d{2})',
        r'(\d{4})/(\d{2})/(\d{2})',
        r'(\d{4})[年-](\d{1,2})[月-](\d{1,2})',
    ]

    for pattern in patterns:
        match = re.search(pattern, html)
        if match:
            year = match.group(1)
            month = match.group(2).zfill(2)
            day = match.group(3).zfill(2)
            return f"{year}-{month}-{day}"

    return None


json_files = [f for f in os.listdir(OUT_DIR) if f.endswith('.json')]

print("=" * 100)
print("检查 news_major1 文件名与内容日期是否匹配")
print("=" * 100)

mismatch_count = 0
mismatch_files = []

for json_file in sorted(json_files):
    match = re.search(r'analysis_(\d{4}-\d{2}-\d{2})\.json', json_file)
    if not match:
        continue

    filename_date = match.group(1)

    try:
        with open(os.path.join(OUT_DIR, json_file), "r", encoding="utf-8") as f:
            data = json.load(f)

        article_date = data.get("article_date", "")
        article_title = data.get("article_title", "N/A")

        html_file = data.get("html_file", "")
        html_path = os.path.join(DATA_DIR, html_file) if html_file else None

        actual_date = ""
        if html_path and os.path.exists(html_path):
            with open(html_path, "r", encoding="utf-8") as f:
                html = f.read()
            actual_date = extract_date_from_html(html)

        if filename_date != article_date or (actual_date and actual_date != filename_date):
            mismatch_count += 1
            mismatch_files.append({
                "file": json_file,
                "filename_date": filename_date,
                "article_date": article_date,
                "actual_date": actual_date,
                "title": article_title
            })

            print(f"\n[ERROR] 不匹配: {json_file}")
            print(f"   文件名日期: {filename_date}")
            print(f"   article_date: {article_date}")
            if actual_date:
                print(f"   实际 HTML 日期: {actual_date}")
            print(f"   标题: {article_title[:60]}")

    except Exception as e:
        print(f"\n[WARN] 错误: {json_file} - {e}")

print("\n" + "=" * 100)
print(f"总文件数: {len(json_files)}")
print(f"不匹配: {mismatch_count}")

if mismatch_files:
    print("\n不匹配文件列表:")
    for f in mismatch_files:
        print(f"  {f['file']}")

print("=" * 100)

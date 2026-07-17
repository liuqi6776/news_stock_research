import os, re, json, sys
from datetime import datetime

DATA_DIR = r'C:\Users\liuqi\clowspace\quant_system_v2\jiayo-analysis\data'
OUT_DIR = r'D:\iquant_data\data_v2\news_major1'
API_KEY = ''

from zhipuai import ZhipuAI
client = ZhipuAI(api_key=API_KEY)

def extract_title(html):
    match = re.search(r'<title[^>]*>([^<]+)</title>', html, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return None

def extract_article_content(html):
    pattern = r'<div[^>]*class="text-box text-justify fsDetail"[^>]*>(.*?)</div>'
    match = re.search(pattern, html, re.DOTALL)
    if match:
        content = re.sub(r'<[^>]+>', '', match.group(1))
        content = re.sub(r'\s+', ' ', content).strip()
        return content

    for pattern in [
        r'<div[^>]*class="[^"]*content[^"]*"[^>]*>(.*?)</div>',
        r'<div[^>]*class="[^"]*article[^"]*"[^>]*>(.*?)</div>',
    ]:
        match = re.search(pattern, html, re.DOTALL)
        if match:
            content = re.sub(r'<[^>]+>', '', match.group(1))
            content = re.sub(r'\s+', ' ', content).strip()
            if len(content) > 500:
                return content

    return None

def extract_date_from_html(html):
    patterns = [
        r'<div[^>]*class="date[^"]*"[^>]*>.*?(\d{4})-(\d{2})-(\d{2})',
        r'(\d{4})年(\d{1,2})月(\d{1,2})日',
    ]
    for pattern in patterns:
        match = re.search(pattern, html, re.DOTALL)
        if match:
            if len(match.groups()) == 3:
                return f"{match.group(1)}-{match.group(2).zfill(2)}-{match.group(3).zfill(2)}"
    return None

html_files = [f for f in os.listdir(DATA_DIR) if f.endswith('.html')]
print(f"Found {len(html_files)} HTML files")

for i, html_file in enumerate(html_files[:3]):
    print(f"\n[{i+1}] Processing: {html_file}")

    with open(os.path.join(DATA_DIR, html_file), "r", encoding="utf-8") as f:
        html = f.read()

    print(f"  HTML size: {len(html)} bytes")

    title = extract_title(html)
    print(f"  Title: {title}")

    content = extract_article_content(html)
    print(f"  Content length: {len(content) if content else 0}")

    article_date = extract_date_from_html(html)
    print(f"  Date: {article_date}")

    out_file = os.path.join(OUT_DIR, f"analysis_{article_date}.json")
    print(f"  Output: {out_file}")
    print(f"  Exists: {os.path.exists(out_file)}")

    if not os.path.exists(out_file):
        if content and len(content) > 100:
            print(f"  Would analyze this file")
        else:
            print(f"  Content too short, would skip")

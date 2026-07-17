import os
import re
from datetime import datetime

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


html_files = [f for f in os.listdir(DATA_DIR) if f.endswith('.html')][:5]

print("=" * 80)
print("Testing date extraction:")
print("=" * 80)

for html_file in html_files:
    print(f"\nFile: {html_file}")

    with open(os.path.join(DATA_DIR, html_file), "r", encoding="utf-8") as f:
        html = f.read()

    date = extract_date_from_html(html)

    title_match = re.search(r'<title[^>]*>([^<]+)</title>', html, re.IGNORECASE)
    title = title_match.group(1).strip() if title_match else "N/A"

    print(f"  Title: {title[:60]}")
    print(f"  Extracted date: {date}")

print("\n" + "=" * 80)

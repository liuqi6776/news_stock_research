import os
import re

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


html_files = [f for f in os.listdir(DATA_DIR) if f.endswith('.html')]

print(f"Total HTML files: {len(html_files)}")

dates = []
for html_file in html_files:
    with open(os.path.join(DATA_DIR, html_file), "r", encoding="utf-8") as f:
        html = f.read()
    date = extract_date_from_html(html)
    if date:
        dates.append(date)

unique_dates = sorted(list(set(dates)))

print(f"Unique dates: {len(unique_dates)}")
print(f"First date: {unique_dates[0]}")
print(f"Last date:  {unique_dates[-1]}")

print("\nLast 20 dates:")
for d in unique_dates[-20:]:
    print(f"  {d}")

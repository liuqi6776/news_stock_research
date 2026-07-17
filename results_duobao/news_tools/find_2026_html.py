import os
import re

DATA_DIR = r"C:\Users\liuqi\clowspace\quant_system_v2\jiayo-analysis\data"

html_files = [f for f in os.listdir(DATA_DIR) if f.endswith('.html')]

dates_2026 = []

for f in html_files:
    filepath = os.path.join(DATA_DIR, f)
    with open(filepath, "r", encoding="utf-8") as fobj:
        html = fobj.read()
    
    patterns = [
        r'(\d{4})-(\d{2})-(\d{2})',
        r'(\d{4})/(\d{2})/(\d{2})',
        r'(\d{4})[年-](\d{1,2})[月-](\d{1,2})',
    ]
    
    date = None
    for pattern in patterns:
        match = re.search(pattern, html)
        if match:
            year = match.group(1)
            month = match.group(2).zfill(2)
            day = match.group(3).zfill(2)
            date = f"{year}-{month}-{day}"
            break
    
    if date and date.startswith("2026"):
        dates_2026.append((date, f))

dates_2026.sort(reverse=True)

print(f"Found {len(dates_2026)} files from 2026:")
for d, f in dates_2026:
    print(f"  {d} - {f}")

print(f"\nLatest date: {dates_2026[0][0] if dates_2026 else 'None'}")

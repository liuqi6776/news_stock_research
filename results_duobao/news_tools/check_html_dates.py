import os
import re
from datetime import datetime

DATA_DIR = r"C:\Users\liuqi\quant_system_v2\jiayo-analysis\data"

def extract_date_from_html(html_content):
    patterns = [
        r'(\d{4})-(\d{2})-(\d{2})',
        r'(\d{4})/(\d{2})/(\d{2})',
        r'(\d{4})[年-](\d{1,2})[月-](\d{1,2})',
    ]

    for pattern in patterns:
        match = re.search(pattern, html_content)
        if match:
            year = match.group(1)
            month = match.group(2).zfill(2)
            day = match.group(3).zfill(2)
            return f"{year}-{month}-{day}"

    return None


def check_dates():
    html_files = [f for f in os.listdir(DATA_DIR) if f.endswith('.html')]
    print(f"共有 {len(html_files)} 个 HTML 文件\n")
    
    dates_found = set()
    missing_dates = [
        "2026-03-27", "2026-03-28", "2026-03-29", "2026-03-30", "2026-03-31",
        "2026-04-01", "2026-04-02", "2026-04-03", "2026-04-06", "2026-04-07"
    ]
    
    for html_file in html_files:
        filepath = os.path.join(DATA_DIR, html_file)
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                html_content = f.read()
            
            date = extract_date_from_html(html_content)
            if date:
                dates_found.add(date)
                if date in missing_dates:
                    print(f"✓ 找到日期 {date}: {html_file}")
        except Exception as e:
            continue
    
    print(f"\n找到的日期数量: {len(dates_found)}")
    print(f"需要的日期: {missing_dates}")
    
    found_missing = [d for d in missing_dates if d in dates_found]
    print(f"已找到的缺失日期: {found_missing}")
    
    still_missing = [d for d in missing_dates if d not in dates_found]
    print(f"仍然缺失的日期: {still_missing}")


if __name__ == "__main__":
    check_dates()

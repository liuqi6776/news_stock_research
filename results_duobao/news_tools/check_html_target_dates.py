import os
import re

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

def check_latest_html_dates():
    html_files = [f for f in os.listdir(DATA_DIR) if f.endswith('.html')]
    print(f"共有 {len(html_files)} 个 HTML 文件\n")

    target_start = "2026-03-17"
    target_end = "2026-04-07"
    
    dates_found = set()
    
    for html_file in html_files:
        filepath = os.path.join(DATA_DIR, html_file)
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                html_content = f.read()
            date = extract_date_from_html(html_content)
            if date and date >= target_start and date <= target_end:
                dates_found.add(date)
                print(f"✓ 找到日期 {date}: {html_file}")
        except Exception as e:
            print(f"Error reading {html_file}: {e}")
    
    print(f"\n找到的目标日期: {sorted(dates_found)}")
    print(f"共 {len(dates_found)} 个日期")

if __name__ == "__main__":
    check_latest_html_dates()

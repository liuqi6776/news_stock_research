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


def check_dates():
    html_files = [f for f in os.listdir(DATA_DIR) if f.endswith('.html')]
    print(f"共有 {len(html_files)} 个 HTML 文件\n")
    
    dates_found = set()
    latest_date = None
    
    for html_file in html_files:
        filepath = os.path.join(DATA_DIR, html_file)
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                html_content = f.read()
            
            date = extract_date_from_html(html_content)
            if date:
                dates_found.add(date)
                if not latest_date or date > latest_date:
                    latest_date = date
        except Exception as e:
            continue
    
    sorted_dates = sorted(list(dates_found), reverse=True)
    print(f"找到的日期（最新10个）:")
    for i, date in enumerate(sorted_dates[:10]):
        print(f"  {date}")
    
    print(f"\n最新日期: {latest_date}")
    print(f"\n所有找到的日期数量: {len(dates_found)}")


if __name__ == "__main__":
    check_dates()

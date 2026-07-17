import os
import re
from datetime import datetime, timedelta

NEWS_MAJOR1_DIR = r"C:\Users\liuqi\quant_system_v2\news_major1"

def get_existing_dates():
    existing_files = [f for f in os.listdir(NEWS_MAJOR1_DIR) if f.startswith('analysis_') and f.endswith('.json')]
    existing_dates = set()
    for f in existing_files:
        match = re.search(r'analysis_(\d{4}-\d{2}-\d{2})\.json', f)
        if match:
            existing_dates.add(match.group(1))
    return sorted(existing_dates)

def date_range(start_date, end_date):
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")
    dates = []
    current = start
    while current <= end:
        dates.append(current.strftime("%Y-%m-%d"))
        current += timedelta(days=1)
    return dates

if __name__ == "__main__":
    existing_dates = get_existing_dates()
    print(f"已有的日期数量: {len(existing_dates)}")
    print(f"最早日期: {existing_dates[0] if existing_dates else 'N/A'}")
    print(f"最新日期: {existing_dates[-1] if existing_dates else 'N/A'}")
    
    print("\n=== 检查 2024-05-05 到 2026-04-07 的日期 ===")
    target_start = "2024-05-05"
    target_end = "2026-04-07"
    target_dates = date_range(target_start, target_end)
    
    missing_dates = [d for d in target_dates if d not in existing_dates]
    print(f"目标日期范围: {target_start} 到 {target_end}")
    print(f"目标日期总数: {len(target_dates)}")
    print(f"已有的日期数: {len([d for d in target_dates if d in existing_dates])}")
    print(f"缺少的日期数: {len(missing_dates)}")
    
    if missing_dates:
        print(f"\n缺少的日期 (前20个):")
        for d in missing_dates[:20]:
            print(f"  - {d}")
        if len(missing_dates) > 20:
            print(f"  ... 还有 {len(missing_dates) - 20} 个")

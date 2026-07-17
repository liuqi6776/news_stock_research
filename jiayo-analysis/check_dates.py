import os, re

DATA_DIR = r'C:\Users\liuqi\clowspace\quant_system_v2\jiayo-analysis\data'
files = [f for f in os.listdir(DATA_DIR) if f.endswith('.html')][:20]

dates = {}
for f in files:
    with open(os.path.join(DATA_DIR, f), 'r', encoding='utf-8') as file:
        html = file.read()
    date_match = re.search(r'<div[^>]*class="date[^"]*"[^>]*>.*?(\d{4})-(\d{2})-(\d{2})', html, re.DOTALL)
    date = date_match.group(1) + '-' + date_match.group(2) + '-' + date_match.group(3) if date_match else 'NOT_FOUND'
    if date not in dates:
        dates[date] = []
    dates[date].append(f)

print('Found dates:')
for date, files in sorted(dates.items()):
    print(f'  {date}: {len(files)} files')
    for f in files[:3]:
        print(f'    - {f}')

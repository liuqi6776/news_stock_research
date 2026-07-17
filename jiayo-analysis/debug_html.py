import os, re

DATA_DIR = r'C:\Users\liuqi\clowspace\quant_system_v2\jiayo-analysis\data'
html_file = '10ltnvryy2k.html'

with open(os.path.join(DATA_DIR, html_file), 'r', encoding='utf-8') as f:
    html = f.read()

print(f'HTML size: {len(html)} bytes')

title_match = re.search(r'<title[^>]*>([^<]+)</title>', html, re.IGNORECASE)
title = title_match.group(1).strip() if title_match else None
print(f'Title: {title}')

patterns = [
    r'<div[^>]*class="text-box text-justify fsDetail"[^>]*>(.*?)</div>',
    r'<div[^>]*class="[^"]*content[^"]*"[^>]*>(.*?)</div>',
    r'<div[^>]*class="[^"]*article[^"]*"[^>]*>(.*?)</div>',
]

content = None
for pattern in patterns:
    content_match = re.search(pattern, html, re.DOTALL)
    if content_match:
        content = re.sub(r'<[^>]+>', '', content_match.group(1))
        content = re.sub(r'\s+', ' ', content).strip()
        print(f'Content found with pattern: {pattern[:50]}...')
        print(f'Content length: {len(content)}')
        break

if not content:
    print('No content found with any pattern')
    body_match = re.search(r'<body[^>]*>(.*?)</body>', html, re.DOTALL)
    if body_match:
        print(f'Body length: {len(body_match.group(1))}')

date_match = re.search(r'<div[^>]*class="date[^"]*"[^>]*>.*?(\d{4})-(\d{2})-(\d{2})', html, re.DOTALL)
if date_match:
    article_date = date_match.group(1) + '-' + date_match.group(2) + '-' + date_match.group(3)
    print(f'Date: {article_date}')
else:
    print('Date not found')

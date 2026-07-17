import os
import re
import json
from playwright.sync_api import sync_playwright

EDGE_PATH = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"

COOKIES = [
    {"name": "SESSION", "value": "ZTg5MmNlYTMtYzcxNS00Y2YyLTgxMGUtOGZkNTQ3ZWIyYjRi", "domain": ".jiuyangongshe.com", "path": "/"},
    {"name": "Hm_lvt_58aa18061df7855800f2a1b32d6da7f4", "value": "1774348680", "domain": ".jiuyangongshe.com", "path": "/"},
    {"name": "Hm_lpvt_58aa18061df7855800f2a1b32d6da7f4", "value": "1774539273", "domain": ".jiuyangongshe.com", "path": "/"},
]

USER_ID = "4df747be1bf143a998171ef03559b517"
BASE_URL = f"https://www.jiuyangongshe.com/u/{USER_ID}"

print(f"Testing: {BASE_URL}")

with sync_playwright() as p:
    browser = p.chromium.launch(headless=True, executable_path=EDGE_PATH)
    context = browser.new_context()
    context.add_cookies(COOKIES)
    page = context.new_page()
    page.goto(BASE_URL, wait_until="domcontentloaded", timeout=30000)
    page.wait_for_timeout(8000)
    html = page.content()
    browser.close()

print(f"HTML length: {len(html)}")

pattern = r'href="(/a/[a-zA-Z0-9]+)"'
matches = re.findall(pattern, html)
print(f"Found {len(matches)} article links")

articles = []
seen = set()
for match in matches:
    article_id = match.replace("/a/", "")
    if article_id not in seen:
        seen.add(article_id)
        articles.append({
            "id": article_id,
            "url": f"https://www.jiuyangongshe.com{match}"
        })

for a in articles[:10]:
    print(f"  {a['id']}")

with open("jiayo-analysis/list_id/test_articles.json", "w", encoding="utf-8") as f:
    json.dump(articles, f, ensure_ascii=False, indent=2)

print(f"\nSaved {len(articles)} articles")

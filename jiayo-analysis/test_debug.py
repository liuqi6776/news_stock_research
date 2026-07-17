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

with open("jiayo-analysis/list_id/debug_html.txt", "w", encoding="utf-8") as f:
    f.write(html)
print("Saved HTML to debug_html.txt")

print("\nSearching for article links...")

patterns = [
    r'href="(/a/[a-f0-9]+)"',
    r'href="(/a/\w+)"',
    r'data-id="([a-f0-9]+)"',
    r'/a/\w{10,}',
]

for pat in patterns:
    matches = re.findall(pat, html)
    if matches:
        print(f"Pattern '{pat}': {len(matches)} matches")
        print(f"  Sample: {matches[:5]}")

title_match = re.search(r'<title>(.*?)</title>', html)
if title_match:
    print(f"\nPage title: {title_match.group(1)}")

body_sample = html[5000:10000] if len(html) > 10000 else html
print(f"\nBody sample (chars 5000-10000):\n{body_sample[:2000]}")

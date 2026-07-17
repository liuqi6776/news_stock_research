import os
import re
import json
import sys
import time
from playwright.sync_api import sync_playwright
from datetime import datetime

BASE_DIR = r"C:\Users\liuqi\quant_system_v2\jiayo-analysis"
LIST_DIR = os.path.join(BASE_DIR, "list_id")
DATA_DIR = os.path.join(BASE_DIR, "data")

USER_ID = "4df747be1bf143a998171ef03559b517"
BASE_URL = f"https://www.jiuyangongshe.com/u/{USER_ID}"
MAX_PAGES = 60

EDGE_PATH = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"

COOKIES = [
    {"name": "SESSION", "value": "ZTg5MmNlYTMtYzcxNS00Y2YyLTgxMGUtOGZkNTQ3ZWIyYjRi", "domain": ".jiuyangongshe.com", "path": "/"},
    {"name": "Hm_lvt_58aa18061df7855800f2a1b32d6da7f4", "value": "1774348680", "domain": ".jiuyangongshe.com", "path": "/"},
    {"name": "Hm_lpvt_58aa18061df7855800f2a1b32d6da7f4", "value": "1774539273", "domain": ".jiuyangongshe.com", "path": "/"},
    {"name": "Hm_lvt_2d6d056d37910563cdaa290ee2981080", "value": "1774348680", "domain": ".jiuyangongshe.com", "path": "/"},
    {"name": "Hm_lpvt_2d6d056d37910563cdaa290ee2981080", "value": "1774539273", "domain": ".jiuyangongshe.com", "path": "/"},
]

os.makedirs(LIST_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)


def scrape_article_links_from_page(page_num):
    if page_num == 1:
        url = BASE_URL
    else:
        url = f"{BASE_URL}/page/{page_num}"

    print(f"Scraping page {page_num}: {url}")

    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True, executable_path=EDGE_PATH)
            context = browser.new_context()
            context.add_cookies(COOKIES)
            page = context.new_page()
            page.goto(url, wait_until="domcontentloaded", timeout=30000)
            page.wait_for_timeout(5000)
            html = page.content()
            browser.close()

        pattern = r'href="(/a/[a-zA-Z0-9]+)"'
        matches = re.findall(pattern, html)

        article_links = []
        seen = set()
        for match in matches:
            article_id = match.replace("/a/", "")
            if article_id not in seen:
                seen.add(article_id)
                article_links.append({
                    "id": article_id,
                    "url": f"https://www.jiuyangongshe.com{match}"
                })

        return article_links

    except Exception as e:
        print(f"  Error: {e}")
        return []


def fetch_article_html_with_playwright(article_url, article_id):
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True, executable_path=EDGE_PATH)
            context = browser.new_context()
            context.add_cookies(COOKIES)
            page = context.new_page()
            page.goto(article_url, wait_until="domcontentloaded", timeout=30000)
            page.wait_for_timeout(5000)
            html = page.content()
            browser.close()
        return html
    except Exception as e:
        print(f"  Playwright error for {article_id}: {e}")
        return None


def save_article_html(article_id, html):
    filename = f"{article_id}.html"
    filepath = os.path.join(DATA_DIR, filename)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(html)
    return filepath


def scrape_page1():
    list_file = os.path.join(LIST_DIR, "article_list.json")

    if os.path.exists(list_file):
        print(f"文章列表已存在: {list_file}，跳过下载")
        with open(list_file, "r", encoding="utf-8") as f:
            all_articles = json.load(f)
        print(f"已加载 {len(all_articles)} 篇文章")
        return

    print(f"下载第1页: {BASE_URL}")
    articles = scrape_article_links_from_page(1)

    if not articles:
        print("未找到文章")
        return

    print(f"找到 {len(articles)} 篇文章")
    all_articles = articles

    with open(list_file, "w", encoding="utf-8") as f:
        json.dump(all_articles, f, ensure_ascii=False, indent=2)

    print(f"已保存到 {list_file}")


def fetch_html():
    list_file = os.path.join(LIST_DIR, "article_list.json")

    if not os.path.exists(list_file):
        print(f"文章列表不存在: {list_file}，请先运行 --step1")
        return

    with open(list_file, "r", encoding="utf-8") as f:
        all_articles = json.load(f)
    print(f"加载了 {len(all_articles)} 篇文章")

    success_count = 0
    skip_count = 0

    for i, article in enumerate(all_articles):
        article_id = article["id"]
        article_url = article["url"]
        html_file = os.path.join(DATA_DIR, f"{article_id}.html")

        if os.path.exists(html_file) and os.path.getsize(html_file) > 50000:
            skip_count += 1
            print(f"[{i+1}/{len(all_articles)}] 跳过(已存在): {article_id}")
            continue

        print(f"[{i+1}/{len(all_articles)}] 下载: {article_url}")
        html = fetch_article_html_with_playwright(article_url, article_id)

        if html and len(html) > 50000:
            save_article_html(article_id, html)
            success_count += 1
            print(f"  已保存 ({len(html)} bytes)")
        else:
            print(f"  失败或内容过小 ({len(html) if html else 0} bytes)")

        time.sleep(2)

    print(f"\n=== 完成 ===")
    print(f"成功: {success_count}, 跳过: {skip_count}, 总计: {len(all_articles)}")


def main():
    if len(sys.argv) > 1:
        arg = sys.argv[1]
        if arg == "--step1":
            scrape_page1()
            return
        elif arg == "--step2":
            fetch_html()
            return

    list_file = os.path.join(LIST_DIR, "article_list.json")

    if os.path.exists(list_file):
        print(f"Loading existing article list from {list_file}")
        with open(list_file, "r", encoding="utf-8") as f:
            all_articles = json.load(f)
        print(f"Loaded {len(all_articles)} articles")
    else:
        all_articles = []
        for page in range(1, MAX_PAGES + 1):
            print(f"\n=== Page {page}/{MAX_PAGES} ===")
            articles = scrape_article_links_from_page(page)

            if not articles:
                print(f"No more articles found at page {page}")
                break

            print(f"Found {len(articles)} articles on page {page}")
            all_articles.extend(articles)

            with open(list_file, "w", encoding="utf-8") as f:
                json.dump(all_articles, f, ensure_ascii=False, indent=2)

            time.sleep(2)

        print(f"\nTotal articles collected: {len(all_articles)}")

    print(f"\n=== Fetching article HTMLs ===")
    success_count = 0
    skip_count = 0

    for i, article in enumerate(all_articles):
        article_id = article["id"]
        article_url = article["url"]
        html_file = os.path.join(DATA_DIR, f"{article_id}.html")

        if os.path.exists(html_file) and os.path.getsize(html_file) > 50000:
            skip_count += 1
            print(f"[{i+1}/{len(all_articles)}] Skip (exists): {article_id}")
            continue

        print(f"[{i+1}/{len(all_articles)}] Fetching: {article_url}")
        html = fetch_article_html_with_playwright(article_url, article_id)

        if html and len(html) > 50000:
            save_article_html(article_id, html)
            success_count += 1
            print(f"  Saved ({len(html)} bytes)")
        else:
            print(f"  Failed or too small ({len(html) if html else 0} bytes)")

        time.sleep(2)

    print(f"\n=== Done ===")
    print(f"Success: {success_count}, Skipped: {skip_count}, Total: {len(all_articles)}")


if __name__ == "__main__":
    main()

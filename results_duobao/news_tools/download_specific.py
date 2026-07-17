import os
import json
import time
from playwright.sync_api import sync_playwright

BASE_DIR = r"C:\Users\liuqi\clowspace\quant_system_v2\jiayo-analysis"
LIST_DIR = os.path.join(BASE_DIR, "list_id")
DATA_DIR = os.path.join(BASE_DIR, "data")

os.makedirs(DATA_DIR, exist_ok=True)

COOKIES = [
    {"name": "SESSION", "value": "ZTg5MmNlYTMtYzcxNS00Y2YyLTgxMGUtOGZkNTQ3ZWIyYjRi", "domain": ".jiuyangongshe.com", "path": "/"},
    {"name": "Hm_lvt_58aa18061df7855800f2a1b32d6da7f4", "value": "1774348680", "domain": ".jiuyangongshe.com", "path": "/"},
    {"name": "Hm_lpvt_58aa18061df7855800f2a1b32d6da7f4", "value": "1775558360", "domain": ".jiuyangongshe.com", "path": "/"},
    {"name": "Hm_lvt_2d6d056d37910563cdaa290ee298108", "value": "1774348680", "domain": ".jiuyangongshe.com", "path": "/"},
    {"name": "Hm_lpvt_2d6d056d37910563cdaa290ee298108", "value": "1775558360", "domain": ".jiuyangongshe.com", "path": "/"},
    {"name": "HMACCOUNT", "value": "7BD4309FFF55449E", "domain": ".jiuyangongshe.com", "path": "/"},
    {"name": "admin", "value": "%7B%22user_id%22%3A%22ccca1ae12799429ca48774b0d10f79e7%22%2C%22country_code%22%3A%22%2B86%22%2C%22phone%22%3A%2213259770650%22%2C%22nickname%22%3A%22%E6%97%A0%E5%90%8D%E5%B0%8F%E9%9F%AD06500909%22%2C%22avatar%22%3A%22https%3A%2F%2Fjiucaigongshe.oss-cn-beijing.aliyuncs.com%2Favatar_default.png%22%2C%22gender%22%3A0%2C%22profile%22%3A%22%E8%BF%99%E4%B8%AA%E4%BA%BA%E5%BE%88%E6%87%92%EF%BC%8C%E4%BB%80%E4%B9%88%E9%83%BD%E6%B2%A1%E6%9C%89%E7%95%99%E4%B8%8B%22%2C%22open_id%22%3Anull%2C%22pc_open_id%22%3Anull%2C%22union_id%22%3Anull%2C%22city%22%3Anull%2C%22area%22%3Anull%2C%22follow_count%22%3A8%2C%22fans_count%22%3A0%2C%22like_count%22%3A0%2C%22posts%22%3A0%2C%22energy%22%3A100%2C%22integral%22%3A10%2C%22integral_grade%22%3A10%2C%22balance%22%3A0%2C%22interaction%22%3A0%2C%22verify%22%3A0%2C%22msg_vibrate%22%3A0%2C%22faction%22%3A0%2C%22faction_id%22%3A%22%22%2C%22investment_style%22%3A%22%22%2C%22investment_style_id%22%3A%22%22%2C%22status%22%3A0%2C%22reward_read_day%22%3A5%2C%22reward_read_time%22%3A%222026-03-27%2023%3A59%3A59%22%2C%22no_read_limit_time%22%3A%222025-09-22%2023%3A59%3A59%22%2C%22change_nickname_limit_time%22%3Anull%2C%22change_info_limit_time%22%3Anull%2C%22medal_count%22%3A0%2C%22withdraw_review%22%3A0%2C%22newest_article_tool_time%22%3Anull%2C%22create_time%22%3A%222025-09-09%2008%3A15%3A00%22%2C%22low_quality%22%3A0%2C%22style_str%22%3Anull%2C%22has_pwd%22%3A1%2C%22newest_article_tool%22%3A0%2C%22user_no%22%3A%22ccca1ae12799429ca48774b0d10f79e7%22%2C%22sessionToken%22%3A%22ZTg5MmNlYTMtYzcxNS00Y2YyLTgxMGUtOGZkNTQ3ZWIyYjRi%22%7D", "domain": ".jiuyangongshe.com", "path": "/"},
    {"name": "time", "value": "1", "domain": ".jiuyangongshe.com", "path": "/"},
]


def download_article(article_url, article_id):
    html_file = os.path.join(DATA_DIR, f"{article_id}.html")
    
    if os.path.exists(html_file) and os.path.getsize(html_file) > 50000:
        print(f"  跳过(已存在): {article_id}")
        return True
    
    try:
        print(f"  正在启动浏览器...")
        with sync_playwright() as p:
            print(f"  正在启动 Chromium...")
            browser = p.chromium.launch(headless=False)
            context = browser.new_context(
                user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/146.0.0.0 Safari/537.36 Edg/146.0.0.0"
            )
            context.add_cookies(COOKIES)
            page = context.new_page()
            
            print(f"  正在访问: {article_url}")
            page.goto(article_url, wait_until="domcontentloaded", timeout=60000)
            print(f"  等待页面加载...")
            page.wait_for_timeout(10000)
            
            print(f"  获取页面内容...")
            html = page.content()
            browser.close()
        
        print(f"  保存文件...")
        with open(html_file, "w", encoding="utf-8") as f:
            f.write(html)
        
        print(f"  ✓ 保存成功: {article_id} ({len(html)} bytes)")
        return True
        
    except Exception as e:
        print(f"  ✗ 错误: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    list_file = os.path.join(LIST_DIR, "article_list.json")
    with open(list_file, "r", encoding="utf-8") as f:
        articles = json.load(f)
    
    print("=== 需要下载的新文章 ===")
    new_articles = []
    for article in articles:
        article_date = article["date"]
        if article_date >= "2026-03-27":
            html_file = os.path.join(DATA_DIR, f"{article['id']}.html")
            if not os.path.exists(html_file) or os.path.getsize(html_file) < 50000:
                new_articles.append(article)
                print(f"  {article['date']}: {article['title']}")
    
    print(f"\n共 {len(new_articles)} 篇新文章需要下载\n")
    
    success = 0
    fail = 0
    
    for i, article in enumerate(new_articles):
        article_id = article["id"]
        article_url = article["url"]
        print(f"\n[{i+1}/{len(new_articles)}] 下载: {article['title']}")
        print(f"  URL: {article_url}")
        
        if download_article(article_url, article_id):
            success += 1
        else:
            fail += 1
        
        print(f"\n  等待 3 秒...")
        time.sleep(3)
    
    print(f"\n=== 完成 ===")
    print(f"成功: {success}, 失败: {fail}")


if __name__ == "__main__":
    main()

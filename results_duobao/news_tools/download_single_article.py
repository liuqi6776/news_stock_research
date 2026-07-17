import os
import time
from playwright.sync_api import sync_playwright

DATA_DIR = r"C:\Users\liuqi\quant_system_v2\jiayo-analysis\data"
os.makedirs(DATA_DIR, exist_ok=True)

url = "https://www.jiuyangongshe.com/a/2038b261aup"
article_id = "2038b261aup"
article_file = os.path.join(DATA_DIR, f"{article_id}.html")

COOKIES = [
    {"name": "SESSION", "value": "ZTg5MmNlYTMtYzcxNS00Y2YyLTgxMGUtOGZkNTQ3ZWIyYjRi", "domain": ".jiuyangongshe.com", "path": "/"},
    {"name": "Hm_lvt_58aa18061df7855800f2a1b32d6da7f4", "value": "1774348680", "domain": ".jiuyangongshe.com", "path": "/"},
    {"name": "Hm_lpvt_58aa18061df7855800f2a1b32d6da7f4", "value": "1775558360", "domain": ".jiuyangongshe.com", "path": "/"},
    {"name": "Hm_lvt_2d6d056d37910563cdaa290ee2981080", "value": "1774348680", "domain": ".jiuyangongshe.com", "path": "/"},
    {"name": "Hm_lpvt_2d6d056d37910563cdaa290ee2981080", "value": "1775558360", "domain": ".jiuyangongshe.com", "path": "/"},
    {"name": "HMACCOUNT", "value": "7BD4309FFF55449E", "domain": ".jiuyangongshe.com", "path": "/"},
    {"name": "admin", "value": "%7B%22user_id%22%3A%22ccca1ae12799429ca48774b0d10f79e7%22%2C%22country_code%22%3A%22%2B86%22%2C%22phone%22%3A%2213259770650%22%2C%22nickname%22%3A%22%E6%97%A0%E5%90%8D%E5%B0%8F%E9%9F%AD06500909%22%2C%22avatar%22%3A%22https%3A%2F%2Fjiucaigongshe.oss-cn-beijing.aliyuncs.com%2Favatar_default.png%22%2C%22gender%22%3A0%2C%22profile%22%3A%22%E8%BF%99%E4%B8%AA%E4%BA%BA%E5%BE%88%E6%87%92%EF%BC%8C%E4%BB%80%E4%B9%88%E9%83%BD%E6%B2%A1%E6%9C%89%E7%95%99%E4%B8%8B%22%2C%22open_id%22%3Anull%2C%22pc_open_id%22%3Anull%2C%22union_id%22%3Anull%2C%22city%22%3Anull%2C%22area%22%3Anull%2C%22follow_count%22%3A8%2C%22fans_count%22%3A0%2C%22like_count%22%3A0%2C%22posts%22%3A0%2C%22energy%22%3A100%2C%22integral%22%3A10%2C%22integral_grade%22%3A10%2C%22balance%22%3A0%2C%22interaction%22%3A0%2C%22verify%22%3A0%2C%22msg_vibrate%22%3A0%2C%22faction%22%3A0%2C%22faction_id%22%3A%22%22%2C%22investment_style%22%3A%22%22%2C%22investment_style_id%22%3A%22%22%2C%22status%22%3A0%2C%22reward_read_day%22%3A5%2C%22reward_read_time%22%3A%222026-03-27%2023%3A59%3A59%22%2C%22no_read_limit_time%22%3A%222025-09-22%2023%3A59%3A59%22%2C%22change_nickname_limit_time%22%3Anull%2C%22change_info_limit_time%22%3Anull%2C%22medal_count%22%3A0%2C%22withdraw_review%22%3A0%2C%22newest_article_tool_time%22%3Anull%2C%22create_time%22%3A%222025-09-09%2008%3A15%3A00%22%2C%22low_quality%22%3A0%2C%22style_str%22%3Anull%2C%22has_pwd%22%3A1%2C%22newest_article_tool%22%3A0%2C%22user_no%22%3A%22ccca1ae12799429ca48774b0d10f79e7%22%2C%22sessionToken%22%3A%22ZTg5MmNlYTMtYzcxNS00Y2YyLTgxMGUtOGZkNTQ3ZWIyYjRi%22%7D", "domain": ".jiuyangongshe.com", "path": "/"},
    {"name": "time", "value": "1", "domain": ".jiuyangongshe.com", "path": "/"},
]

def main():
    print(f"正在下载: {url}")
    
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)
        context = browser.new_context()
        context.add_cookies(COOKIES)
        page = context.new_page()
        
        try:
            page.goto(url, wait_until="domcontentloaded", timeout=60000)
            print("页面加载完成，等待内容...")
            page.wait_for_timeout(10000)
            
            html = page.content()
            
            with open(article_file, "w", encoding="utf-8") as f:
                f.write(html)
            
            print(f"✓ 保存成功: {article_file} ({len(html)} bytes)")
            
        except Exception as e:
            print(f"✗ 下载失败: {e}")
        finally:
            browser.close()

if __name__ == "__main__":
    main()

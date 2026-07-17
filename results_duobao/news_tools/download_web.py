import requests
import json
import time
import os

BASE_DIR = r"C:\Users\liuqi\clowspace\quant_system_v2\jiayo-analysis"
LIST_DIR = os.path.join(BASE_DIR, "list_id")
DATA_DIR = os.path.join(BASE_DIR, "data")

current_time = int(time.time())

headers = {
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7",
    "Accept-Encoding": "gzip, deflate, br, zstd",
    "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
    "Connection": "keep-alive",
    "Cookie": "Hm_lvt_2d6d056d37910563cdaa290ee2981080=1774348680; Hm_lvt_58aa18061df7855800f2a1b32d6da7f4=1774348680; HMACCOUNT=7BD4309FFF55449E; admin=%7B%22user_id%22%3A%22ccca1ae12799429ca48774b0d10f79e7%22%2C%22country_code%22%3A%22%2B86%22%2C%22phone%22%3A%2213259770650%22%2C%22nickname%22%3A%22%E6%97%A0%E5%90%8D%E5%B0%8F%E9%9F%AD06500909%22%2C%22avatar%22%3A%22https%3A%2F%2Fjiucaigongshe.oss-cn-beijing.aliyuncs.com%2Favatar_default.png%22%2C%22gender%22%3A0%2C%22profile%22%3A%22%E8%BF%99%E4%B8%AA%E4%BA%BA%E5%BE%88%E6%87%92%EF%BC%8C%E4%BB%80%E4%B9%88%E9%83%BD%E6%B2%A1%E6%9C%89%E7%95%99%E4%B8%8B%22%2C%22open_id%22%3Anull%2C%22pc_open_id%22%3Anull%2C%22union_id%22%3Anull%2C%22city%22%3Anull%2C%22area%22%3Anull%2C%22follow_count%22%3A8%2C%22fans_count%22%3A0%2C%22like_count%22%3A0%2C%22posts%22%3A0%2C%22energy%22%3A100%2C%22integral%22%3A10%2C%22integral_grade%22%3A10%2C%22balance%22%3A0%2C%22interaction%22%3A0%2C%22verify%22%3A0%2C%22msg_vibrate%22%3A0%2C%22faction%22%3A0%2C%22faction_id%22%3A%22%22%2C%22investment_style%22%3A%22%22%2C%22investment_style_id%22%3A%22%22%2C%22status%22%3A0%2C%22reward_read_day%22%3A5%2C%22reward_read_time%22%3A%222026-03-27%2023%3A59%3A59%22%2C%22no_read_limit_time%22%3A%222025-09-22%2023%3A59%3A59%22%2C%22change_nickname_limit_time%22%3Anull%2C%22change_info_limit_time%22%3Anull%2C%22medal_count%22%3A0%2C%22withdraw_review%22%3A0%2C%22newest_article_tool_time%22%3Anull%2C%22create_time%22%3A%222025-09-09%2008%3A15%3A00%22%2C%22low_quality%22%3A0%2C%22style_str%22%3Anull%2C%22has_pwd%22%3A1%2C%22newest_article_tool%22%3A0%2C%22user_no%22%3A%22ccca1ae12799429ca48774b0d10f79e7%22%2C%22sessionToken%22%3A%22ZTg5MmNlYTMtYzcxNS00Y2YyLTgxMGUtOGZkNTQ3ZWIyYjRi%22%7D; SESSION=ZTg5MmNlYTMtYzcxNS00Y2YyLTgxMGUtOGZkNTQ3ZWIyYjRi; time=1; Hm_lpvt_2d6d056d37910563cdaa290ee2981080=1775558360; Hm_lpvt_58aa18061df7855800f2a1b32d6da7f4=1775558360",
    "Host": "www.jiuyangongshe.com",
    "Referer": "https://www.jiuyangongshe.com/",
    "Sec-Ch-Ua": '"Chromium";v="146", "Not-A.Brand";v="24", "Microsoft Edge";v="146"',
    "Sec-Ch-Ua-Mobile": "?0",
    "Sec-Ch-Ua-Platform": '"Windows"',
    "Sec-Fetch-Dest": "document",
    "Sec-Fetch-Mode": "navigate",
    "Sec-Fetch-Site": "same-origin",
    "Sec-Fetch-User": "?1",
    "Upgrade-Insecure-Requests": "1",
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/146.0.0.0 Safari/537.36 Edg/146.0.0.0",
}


def get_article_html(article_id):
    article_file = os.path.join(DATA_DIR, f"{article_id}.html")
    if os.path.exists(article_file) and os.path.getsize(article_file) > 50000:
        return article_file, True
    
    url = f"https://www.jiuyangongshe.com/a/{article_id}"
    
    print(f"  正在下载: {url}")
    
    try:
        response = requests.get(url, headers=headers, timeout=30)
        if response.status_code != 200:
            print(f"  Error: HTTP {response.status_code}")
            return None, False
        
        html = response.text
        
        with open(article_file, "w", encoding="utf-8") as f:
            f.write(html)
        
        print(f"  保存成功: {len(html)} bytes")
        return article_file, False
        
    except Exception as e:
        print(f"  Error: {e}")
        return None, False


def main():
    list_file = os.path.join(LIST_DIR, "article_list.json")
    with open(list_file, "r", encoding="utf-8") as f:
        articles = json.load(f)
    
    print("=== 筛选 2026-03-27 之后的新文章 ===")
    
    new_articles = []
    for article in articles:
        article_date = article["date"]
        if article_date >= "2026-03-27":
            new_articles.append(article)
    
    print(f"找到 {len(new_articles)} 篇新文章:")
    for a in new_articles:
        print(f"  {a['date']}: {a['title']}")
    
    if not new_articles:
        print("\n没有需要下载的新文章！")
        return
    
    print("\n=== 开始下载 ===")
    
    success = 0
    skipped = 0
    failed = 0
    
    for i, article in enumerate(new_articles):
        print(f"\n[{i+1}/{len(new_articles)}] {article['date']} - {article['title']}")
        
        result, exists = get_article_html(article["id"])
        
        if result:
            if exists:
                skipped += 1
            else:
                success += 1
        else:
            failed += 1
        
        time.sleep(1)
    
    print(f"\n=== 下载完成 ===")
    print(f"成功: {success}")
    print(f"跳过: {skipped}")
    print(f"失败: {failed}")


if __name__ == "__main__":
    main()

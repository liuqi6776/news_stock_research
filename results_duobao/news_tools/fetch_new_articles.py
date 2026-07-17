import requests
import json
import time
import os
import re
from datetime import datetime

BASE_DIR = r"C:\Users\liuqi\clowspace\quant_system_v2\jiayo-analysis"
LIST_DIR = os.path.join(BASE_DIR, "list_id")
DATA_DIR = os.path.join(BASE_DIR, "data")

os.makedirs(LIST_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

# API 配置
list_url = "https://app.jiuyangongshe.com/jystock-app/api/v1/product/article/list"
article_url = "https://app.jiuyangongshe.com/jystock-app/api/v2/article/detail"

current_time = int(time.time())

headers = {
    "Accept": "application/json, text/plain, */*",
    "Accept-Encoding": "gzip, deflate, br, zstd",
    "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
    "Connection": "keep-alive",
    "Content-Type": "application/json",
    "Cookie": "Hm_lvt_2d6d056d37910563cdaa290ee2981080=1774348680; Hm_lvt_58aa18061df7855800f2a1b32d6da7f4=1774348680; HMACCOUNT=7BD4309FFF55449E; admin=%7B%22user_id%22%3A%22ccca1ae12799429ca48774b0d10f79e7%22%2C%22country_code%22%3A%22%2B86%22%2C%22phone%22%3A%2213259770650%22%2C%22nickname%22%3A%22%E6%97%A0%E5%90%8D%E5%B0%8F%E9%9F%AD06500909%22%2C%22avatar%22%3A%22https%3A%2F%2Fjiucaigongshe.oss-cn-beijing.aliyuncs.com%2Favatar_default.png%22%2C%22gender%22%3A0%2C%22profile%22%3A%22%E8%BF%99%E4%B8%AA%E4%BA%BA%E5%BE%88%E6%87%92%EF%BC%8C%E4%BB%80%E4%B9%88%E9%83%BD%E6%B2%A1%E6%9C%89%E7%95%99%E4%B8%8B%22%2C%22open_id%22%3Anull%2C%22pc_open_id%22%3Anull%2C%22union_id%22%3Anull%2C%22city%22%3Anull%2C%22area%22%3Anull%2C%22follow_count%22%3A8%2C%22fans_count%22%3A0%2C%22like_count%22%3A0%2C%22posts%22%3A0%2C%22energy%22%3A100%2C%22integral%22%3A10%2C%22integral_grade%22%3A10%2C%22balance%22%3A0%2C%22interaction%22%3A0%2C%22verify%22%3A0%2C%22msg_vibrate%22%3A0%2C%22faction%22%3A0%2C%22faction_id%22%3A%22%22%2C%22investment_style%22%3A%22%22%2C%22investment_style_id%22%3A%22%22%2C%22status%22%3A0%2C%22reward_read_day%22%3A5%2C%22reward_read_time%22%3A%222026-03-27%2023%3A59%3A59%22%2C%22no_read_limit_time%22%3A%222025-09-22%2023%3A59%3A59%22%2C%22change_nickname_limit_time%22%3Anull%2C%22change_info_limit_time%22%3Anull%2C%22medal_count%22%3A0%2C%22withdraw_review%22%3A0%2C%22newest_article_tool_time%22%3Anull%2C%22create_time%22%3A%222025-09-09%2008%3A15%3A00%22%2C%22low_quality%22%3A0%2C%22style_str%22%3Anull%2C%22has_pwd%22%3A1%2C%22newest_article_tool%22%3A0%2C%22user_no%22%3A%22ccca1ae12799429ca48774b0d10f79e7%22%2C%22sessionToken%22%3A%22ZTg5MmNlYTMtYzcxNS00Y2YyLTgxMGUtOGZkNTQ3ZWIyYjRi%22%7D; SESSION=ZTg5MmNlYTMtYzcxNS00Y2YyLTgxMGUtOGZkNTQ3ZWIyYjRi; time=1; Hm_lpvt_2d6d056d37910563cdaa290ee2981080=1775558360; Hm_lpvt_58aa18061df7855800f2a1b32d6da7f4=1775558360",
    "Host": "app.jiuyangongshe.com",
    "Origin": "https://www.jiuyangongshe.com",
    "Page-Time": str(current_time),
    "Platform": "3",
    "Referer": "https://www.jiuyangongshe.com/",
    "Sec-Ch-Ua": '"Chromium";v="146", "Not-A.Brand";v="24", "Microsoft Edge";v="146"',
    "Sec-Ch-Ua-Mobile": "?1",
    "Sec-Ch-Ua-Platform": "Android",
    "Sec-Fetch-Dest": "empty",
    "Sec-Fetch-Mode": "cors",
    "Sec-Fetch-Site": "same-site",
    "Timestamp": str(int(current_time * 1000)),
    "Token": "ZTg5MmNlYTMtYzcxNS00Y2YyLTgxMGUtOGZkNTQ3ZWIyYjRi",
    "User-Agent": "Mozilla/5.0 (Linux; Android 6.0; Nexus 5 Build/MRA58N) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/146.0.0.0 Mobile Safari/537.36 Edg/146.0.0.0",
}


def get_article_list():
    print("=== 获取文章列表 ===")
    
    all_articles = []
    page = 1
    
    while True:
        data = {
            "product_id": "1",
            "start": page,
            "end": page,
            "limit": 20,
            "offset": 0
        }
        
        try:
            response = requests.post(list_url, json=data, headers=headers, timeout=30)
            if response.status_code != 200:
                print(f"Page {page}: HTTP {response.status_code}")
                break
            
            j = response.json()
            if not j.get("data") or not j["data"].get("result"):
                print(f"Page {page}: No more data")
                break
            
            items = j["data"]["result"]
            print(f"Page {page}: {len(items)} articles, latest: {items[0]['create_time']}")
            
            for item in items:
                article_id = item["product_article_id"]
                create_time = item["create_time"]
                article_date = create_time.split(" ")[0]
                all_articles.append({
                    "id": article_id,
                    "url": f"https://www.jiuyangongshe.com/a/{article_id}",
                    "title": item.get("title", ""),
                    "date": article_date,
                    "create_time": create_time
                })
            
            if len(items) < 20:
                break
            
            page += 1
            time.sleep(0.5)
            
        except Exception as e:
            print(f"Error on page {page}: {e}")
            break
    
    print(f"\nTotal articles: {len(all_articles)}")
    
    list_file = os.path.join(LIST_DIR, "article_list.json")
    with open(list_file, "w", encoding="utf-8") as f:
        json.dump(all_articles, f, ensure_ascii=False, indent=2)
    
    print(f"Saved to: {list_file}")
    return all_articles


def get_existing_html_dates():
    existing_dates = set()
    
    html_files = [f for f in os.listdir(DATA_DIR) if f.endswith('.html')]
    
    for f in html_files:
        filepath = os.path.join(DATA_DIR, f)
        try:
            with open(filepath, "r", encoding="utf-8") as fobj:
                html = fobj.read()
            
            patterns = [
                r'(\d{4})-(\d{2})-(\d{2})',
                r'(\d{4})/(\d{2})/(\d{2})',
                r'(\d{4})[年-](\d{1,2})[月-](\d{1,2})',
            ]
            
            for pattern in patterns:
                match = re.search(pattern, html)
                if match:
                    year = match.group(1)
                    month = match.group(2).zfill(2)
                    day = match.group(3).zfill(2)
                    date = f"{year}-{month}-{day}"
                    existing_dates.add(date)
                    break
        except Exception as e:
            continue
    
    print(f"已存在 {len(existing_dates)} 个日期的 HTML 文件")
    return existing_dates


def get_article_html(article_id):
    article_file = os.path.join(DATA_DIR, f"{article_id}.html")
    if os.path.exists(article_file) and os.path.getsize(article_file) > 10000:
        return article_file
    
    data = {"product_article_id": article_id}
    
    try:
        response = requests.post(article_url, json=data, headers=headers, timeout=30)
        if response.status_code != 200:
            print(f"  Error: HTTP {response.status_code}")
            return None
        
        j = response.json()
        if not j.get("data") or not j["data"].get("content"):
            print(f"  Error: No content")
            return None
        
        content = j["data"]["content"]
        title = j["data"].get("title", "")
        
        html = f"""<!DOCTYPE html>
<html>
<head>
<title>{title}</title>
<meta charset="utf-8">
</head>
<body>
<div class="article-content">{content}</div>
</body>
</html>"""
        
        with open(article_file, "w", encoding="utf-8") as f:
            f.write(html)
        
        print(f"  保存成功: {article_file} ({len(html)} bytes)")
        return article_file
        
    except Exception as e:
        print(f"  Error: {e}")
        return None


def main():
    articles = get_article_list()
    
    existing_dates = get_existing_html_dates()
    
    print("\n=== 筛选新文章（2026-03-27之后） ===")
    
    new_articles = []
    for article in articles:
        article_date = article["date"]
        if article_date >= "2026-03-27" and article_date not in existing_dates:
            new_articles.append(article)
    
    print(f"需要下载的新文章: {len(new_articles)} 篇")
    for a in new_articles:
        print(f"  {a['date']}: {a['title']}")
    
    if not new_articles:
        print("没有需要下载的新文章！")
        return
    
    print("\n=== 下载新文章 HTML ===")
    
    success = 0
    skip = 0
    fail = 0
    
    for i, article in enumerate(new_articles):
        article_id = article["id"]
        print(f"[{i+1}/{len(new_articles)}] {article['date']}: {article['title'][:60]}...")
        
        result = get_article_html(article_id)
        if result:
            success += 1
        else:
            html_file = os.path.join(DATA_DIR, f"{article_id}.html")
            if os.path.exists(html_file):
                skip += 1
            else:
                fail += 1
        
        time.sleep(0.5)
    
    print(f"\n=== 完成 ===")
    print(f"成功: {success}, 跳过: {skip}, 失败: {fail}")


if __name__ == "__main__":
    main()

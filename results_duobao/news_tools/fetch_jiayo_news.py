import requests
import json
import time

url = "https://app.jiuyangongshe.com/jystock-app/api/v1/product/article/list"

cookies = {
    "SESSION": "ZTg5MmNlYTMtYzcxNS00Y2YyLTgxMGUtOGZkNTQ3ZWIyYjRi",
    "Hm_lvt_58aa18061df7855800f2a1b32d6da7f4": "1774348680",
    "Hm_lpvt_58aa18061df7855800f2a1b32d6da7f4": "1774357450",
}

headers = {
    "Accept": "application/json, text/plain, */*",
    "Accept-Encoding": "gzip, deflate, br, zstd",
    "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8,en-GB;q=0.7,en-US;q=0.6",
    "Connection": "keep-alive",
    "Content-Type": "application/json",
    "Cookie": "SESSION=ZTg5MmNlYTMtYzcxNS00Y2YyLTgxMGUtOGZkNTQ3ZWIyYjRi; Hm_lvt_58aa18061df7855800f2a1b32d6da7f4=1774348680; Hm_lpvt_58aa18061df7855800f2a1b32d6da7f4=1774357450",
    "Host": "app.jiuyangongshe.com",
    "Origin": "https://www.jiuyangongshe.com",
    "Page-Time": "1774357475",
    "Platform": "3",
    "Referer": "https://www.jiuyangongshe.com/",
    "Sec-Ch-Ua": '"Chromium";v="146", "Not-A.Brand";v="24", "Microsoft Edge";v="146"',
    "Sec-Ch-Ua-Mobile": "?1",
    "Sec-Ch-Ua-Platform": "Android",
    "Sec-Fetch-Dest": "empty",
    "Sec-Fetch-Mode": "cors",
    "Sec-Fetch-Site": "same-site",
    "Timestamp": "1774357631489",
    "Token": "4178dace335d2349a3dde9322a072c80",
    "User-Agent": "Mozilla/5.0 (Linux; Android 6.0; Nexus 5 Build/MRA58N) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/146.0.0.0 Mobile Safari/537.36 Edg/146.0.0.0",
}

data = {
    "product_id": "1",
    "start": "2023-11-06",
    "end": "2023-11-27",
    "limit": 50,
    "offset": 0
}

print("Calling API with correct headers...")
response = requests.post(url, json=data, headers=headers, cookies=cookies)
print(f"Status: {response.status_code}")
print(f"Response: {response.text[:3000]}")

if response.status_code == 200:
    try:
        j = response.json()
        if j.get("data") and j["data"].get("list"):
            print(f"\nFound {len(j['data']['list'])} articles")
            for item in j["data"]["list"][:5]:
                print(f"  {item['create_time']}: {item['title']}")
                print(f"    ID: {item['product_article_id']}")
    except:
        pass

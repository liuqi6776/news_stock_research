import os
import re
import json
import sys
from zhipuai import ZhipuAI
from datetime import datetime

DATA_DIR = r"C:\Users\liuqi\quant_system_v2\jiayo-analysis\data"
OUT_DIR = r"D:\iquant_data\data_v2\news_major1"

os.makedirs(OUT_DIR, exist_ok=True)

API_KEY = ""
client = ZhipuAI(api_key=API_KEY)


def extract_article_content(html):
    content = None

    content_match = re.search(r'<div[^>]*class="[^"]*content[^"]*"[^>]*>(.*?)</div>', html, re.DOTALL)
    if content_match:
        content = re.sub(r'<[^>]+>', '', content_match.group(1))
        content = re.sub(r'\s+', ' ', content).strip()

    if not content or len(content) < 500:
        content_match = re.search(r'<div[^>]*class="[^"]*article-content[^"]*"[^>]*>(.*?)</div>', html, re.DOTALL)
        if content_match:
            content = re.sub(r'<[^>]+>', '', content_match.group(1))
            content = re.sub(r'\s+', ' ', content).strip()

    if not content or len(content) < 500:
        body_match = re.search(r'<body[^>]*>(.*?)</body>', html, re.DOTALL)
        if body_match:
            body = body_match.group(1)
            body = re.sub(r'<script[^>]*>.*?</script>', '', body, flags=re.DOTALL)
            body = re.sub(r'<style[^>]*>.*?</style>', '', body, flags=re.DOTALL)
            body = re.sub(r'<[^>]+>', ' ', body)
            body = re.sub(r'\s+', ' ', body).strip()
            if len(body) > 1000:
                content = body

    return content


def extract_title(html):
    title_match = re.search(r'<title[^>]*>([^<]+)</title>', html, re.IGNORECASE)
    if title_match:
        title = title_match.group(1).strip()
        title = re.sub(r'\s*[-_|]\s*九研公社\s*$', '', title)
        title = re.sub(r'\s*[-_|]\s*韭研公社\s*$', '', title)
        return title.strip()
    return None


def extract_date_from_filename(filename):
    match = re.search(r'(\d{4}-\d{2}-\d{2})', filename)
    if match:
        return match.group(1)
    return None


def extract_date_from_html(html):
    patterns = [
        r'(\d{4})[年-](\d{1,2})[月-](\d{1,2})',
        r'(\d{4})-(\d{2})-(\d{2})',
        r'(\d{4})/(\d{2})/(\d{2})',
    ]

    for pattern in patterns:
        match = re.search(pattern, html)
        if match:
            year = match.group(1)
            month = match.group(2).zfill(2)
            day = match.group(3).zfill(2)
            return f"{year}-{month}-{day}"

    return None


def analyze_article(title, content):
    prompt = f"""你是一个专业的A股市场分析师。请分析以下文章内容，评估其对市场、板块和个股的影响。

文章标题：{title}

文章内容：
{content[:4000]}

请以JSON格式返回分析结果，格式如下：
{{
    "market_impact": 利好利空程度(-5到+5，负数表示利空，正数表示利好，0表示中性),
    "market_analysis": "对大盘影响的简要说明（50字以内）",
    "sectors": [
        {{
            "sector_name": "板块名称",
            "impact": 利好利空程度(-5到+5),
            "analysis": "简要说明（30字以内）"
        }}
    ],
    "stocks": [
        {{
            "stock_name": "股票名称",
            "stock_code": "股票代码",
            "impact": 利好利空程度(-5到+5),
            "analysis": "简要说明（30字以内）"
        }}
    ]
}}

只返回JSON，不要其他内容。"""

    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="glm-4-flash",
                messages=[
                    {"role": "user", "content": prompt}
                ],
                max_tokens=4096,
                temperature=0.1
            )

            result_text = response.choices[0].message.content
            result_text = result_text.strip()
            if result_text.startswith("```json"):
                result_text = result_text[7:]
            if result_text.endswith("```"):
                result_text = result_text[:-3]
            result_text = result_text.strip()

            return json.loads(result_text)
        except json.JSONDecodeError as e:
            print(f"  JSON error: {e}")
            if attempt < max_retries - 1:
                continue
        except Exception as e:
            print(f"  Attempt {attempt + 1} error: {e}")
            if attempt < max_retries - 1:
                continue

    return None


def analyze_html_file(html_file):
    try:
        with open(html_file, "r", encoding="utf-8") as f:
            html = f.read()
    except Exception as e:
        print(f"  Read error: {e}")
        return None

    if len(html) < 50000:
        print(f"  HTML too small: {len(html)} bytes")
        return None

    title = extract_title(html)
    content = extract_article_content(html)

    print(f"  Title: {title[:50] if title else 'N/A'}...")
    print(f"  Content length: {len(content) if content else 0}")

    if not content or len(content) < 100:
        print("  No content extracted")
        return None

    result = analyze_article(title or "Unknown Title", content)

    if result:
        result["article_title"] = title
        result["html_file"] = os.path.basename(html_file)

    return result


def main():
    html_files = [f for f in os.listdir(DATA_DIR) if f.endswith('.html')]
    print(f"Found {len(html_files)} HTML files in {DATA_DIR}")

    success_count = 0
    skip_count = 0
    error_count = 0

    for i, html_file in enumerate(html_files):
        print(f"\n[{i+1}/{len(html_files)}] Processing: {html_file}")

        article_date = extract_date_from_filename(html_file)
        if not article_date:
            with open(os.path.join(DATA_DIR, html_file), "r", encoding="utf-8") as f:
                html_content = f.read()
            article_date = extract_date_from_html(html_content)
        if not article_date:
            article_date = datetime.now().strftime("%Y-%m-%d")

        out_file = os.path.join(OUT_DIR, f"analysis_{article_date}.json")

        if os.path.exists(out_file):
            skip_count += 1
            print(f"  Skip (exists): {out_file}")
            continue

        html_path = os.path.join(DATA_DIR, html_file)
        result = analyze_html_file(html_path)

        if result:
            result["article_date"] = article_date

            with open(out_file, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)

            print(f"  Saved to: {out_file}")
            print(f"  Market impact: {result.get('market_impact', 'N/A')}")
            success_count += 1
        else:
            error_count += 1
            print(f"  Analysis failed")

    print(f"\n=== Done ===")
    print(f"Success: {success_count}, Skipped: {skip_count}, Failed: {error_count}, Total: {len(html_files)}")


def main_latest():
    html_files = [f for f in os.listdir(DATA_DIR) if f.endswith('.html')]
    print(f"Found {len(html_files)} HTML files in {DATA_DIR}")

    existing_json = [f for f in os.listdir(OUT_DIR) if f.startswith('analysis_') and f.endswith('.json')]
    existing_dates = set()
    for f in existing_json:
        match = re.search(r'analysis_(\d{4}-\d{2}-\d{2})\.json', f)
        if match:
            existing_dates.add(match.group(1))
    print(f"Found {len(existing_dates)} existing analysis dates: {sorted(existing_dates)[:5]}...")

    success_count = 0
    skip_count = 0
    error_count = 0
    new_count = 0

    for i, html_file in enumerate(html_files):
        article_date = extract_date_from_filename(html_file)
        if not article_date:
            with open(os.path.join(DATA_DIR, html_file), "r", encoding="utf-8") as f:
                html_content = f.read()
            article_date = extract_date_from_html(html_content)
        if not article_date:
            article_date = datetime.now().strftime("%Y-%m-%d")

        out_file = os.path.join(OUT_DIR, f"analysis_{article_date}.json")

        if os.path.exists(out_file):
            skip_count += 1
            print(f"[{i+1}/{len(html_files)}] 跳过(已分析): {html_file} -> {article_date}")
            continue

        new_count += 1
        print(f"[{i+1}/{len(html_files)}] 分析: {html_file} -> {article_date}")

        html_path = os.path.join(DATA_DIR, html_file)
        result = analyze_html_file(html_path)

        if result:
            result["article_date"] = article_date

            with open(out_file, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)

            print(f"  已保存: {out_file}")
            print(f"  Market impact: {result.get('market_impact', 'N/A')}")
            success_count += 1
        else:
            error_count += 1
            print(f"  分析失败")

    print(f"\n=== 完成 ===")
    print(f"成功: {success_count}, 跳过: {skip_count}, 失败: {error_count}, 新分析: {new_count}, 总HTML: {len(html_files)}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--latest":
        main_latest()
    else:
        main()

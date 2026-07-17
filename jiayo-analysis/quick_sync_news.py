"""
quick_sync_news.py
一键自动同步最新韭研公社新闻文章。
流程：
1. 抓取第一页的文章链接 (获取最新15篇文章)。
2. 比对本地已存在的 HTML 文件，只下载全新的文章网页并保存至 data/ 目录。
3. 调用 GLM-4-Flash 分析新的网页内容，自动识别日期，生成 analysis_YYYY-MM-DD.json 写入 D:\iquant_data\data_v2\news_major1\。
"""
import os
import re
import json
import time
import sys
from datetime import datetime
from playwright.sync_api import sync_playwright

# Add current directory to path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from scraper import scrape_article_links_from_page, fetch_article_html_with_playwright, save_article_html
from analyzer import extract_title, extract_article_content, extract_date_from_html, analyze_article

DATA_DIR = os.path.join(SCRIPT_DIR, "data")
OUT_DIR = r"D:\iquant_data\data_v2\news_major1"

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)

def sync_latest_news():
    print("==================================================")
    print("   Starting Automated News Scrape & Analysis     ")
    print("==================================================")
    
    # Step 1: Scrape Page 1 Links
    print("[INFO] Scraping page 1 links from Jiayo Gongshe...")
    articles = scrape_article_links_from_page(1)
    
    if not articles:
        print("[ERROR] Failed to fetch article links from page 1.")
        return
        
    print(f"[INFO] Scraped {len(articles)} articles from homepage.")
    
    # Step 2: Download HTML for new articles
    new_html_files = []
    
    for idx, article in enumerate(articles):
        article_id = article["id"]
        article_url = article["url"]
        html_file = os.path.join(DATA_DIR, f"{article_id}.html")
        
        # If HTML already exists and is valid size, skip
        if os.path.exists(html_file) and os.path.getsize(html_file) > 10000:
            continue
            
        print(f"  [DOWNLOAD] Fetching new article [{idx+1}]: {article_url} ...")
        html = fetch_article_html_with_playwright(article_url, article_id)
        
        if html and len(html) > 10000:
            filepath = save_article_html(article_id, html)
            new_html_files.append((article_id, filepath))
            print(f"  [SUCCESS] Saved {len(html)} bytes to {filepath}")
            time.sleep(2) # Anti-crawling pause
        else:
            print(f"  [ERROR] Failed to fetch HTML for {article_id}")
            
    # Check if there are existing HTML files that haven't been analyzed yet
    # to be extremely comprehensive!
    all_htmls = [f for f in os.listdir(DATA_DIR) if f.endswith('.html')]
    pending_html_paths = []
    
    # Find which local HTML files do not have an analysis JSON yet
    existing_jsons = os.listdir(OUT_DIR)
    existing_article_ids = set()
    for f in existing_jsons:
        if f.startswith('analysis_') and f.endswith('.json'):
            try:
                # Read JSON to see which HTML file generated it
                with open(os.path.join(OUT_DIR, f), 'r', encoding='utf-8') as jf:
                    data = json.load(jf)
                    html_file = data.get("html_file")
                    if html_file:
                        existing_article_ids.add(html_file.replace('.html', ''))
            except:
                pass
                
    for hf in all_htmls:
        art_id = hf.replace('.html', '')
        if art_id not in existing_article_ids:
            hpath = os.path.join(DATA_DIR, hf)
            pending_html_paths.append((art_id, hpath))
            
    print(f"[INFO] Found {len(pending_html_paths)} HTML files pending AI analysis.")
    
    if not pending_html_paths:
        print("[SUCCESS] No new news articles to analyze. Everything is up-to-date!")
        return
        
    # Step 3: Run AI Analysis on pending HTML files
    analyzed_count = 0
    
    for idx, (art_id, hpath) in enumerate(pending_html_paths):
        print(f"\n[AI ANALYSIS] [{idx+1}/{len(pending_html_paths)}] Analyzing: {art_id}.html ...")
        try:
            with open(hpath, "r", encoding="utf-8") as f:
                html_content = f.read()
                
            # Extract basic properties
            title = extract_title(html_content)
            content = extract_article_content(html_content)
            
            # Extract date from HTML content
            article_date = extract_date_from_html(html_content)
            
            # Fuzzy date fallback: try extracting from title (e.g. 2026.5.29)
            if not article_date and title:
                date_match = re.search(r'(\d{4})[./-](\d{1,2})[./-](\d{1,2})', title)
                if date_match:
                    year = date_match.group(1)
                    month = date_match.group(2).zfill(2)
                    day = date_match.group(3).zfill(2)
                    article_date = f"{year}-{month}-{day}"
                    
            if not article_date:
                # Default fallback
                article_date = datetime.now().strftime("%Y-%m-%d")
                
            out_file = os.path.join(OUT_DIR, f"analysis_{article_date}.json")
            
            # Skip if date-specific analysis already exists
            if os.path.exists(out_file):
                print(f"  [SKIP] Analysis file for {article_date} already exists.")
                continue
                
            print(f"  [NLP] Title: {title[:50]}...")
            print(f"  [NLP] Date: {article_date} | Content size: {len(content) if content else 0} chars")
            
            if not content or len(content) < 100:
                print("  [ERROR] Content too short or empty. Skipping.")
                continue
                
            # Call Zhipu AI
            result = analyze_article(title or "Market News", content)
            
            if result:
                result["article_date"] = article_date
                result["article_title"] = title
                result["html_file"] = f"{art_id}.html"
                
                with open(out_file, "w", encoding="utf-8") as f:
                    json.dump(result, f, ensure_ascii=False, indent=2)
                    
                print(f"  [SUCCESS] Structured analysis saved to: {out_file}")
                print(f"  [SUCCESS] Market Impact score: {result.get('market_impact', 0)}")
                analyzed_count += 1
            else:
                print("  [ERROR] AI analysis returned None.")
                
        except Exception as e:
            print(f"  [ERROR] Failed to process {art_id}: {e}")
            
    print(f"\n==================================================")
    print(f"   News sync finished. Analyzed and saved {analyzed_count} new entries.")
    print("==================================================")

if __name__ == "__main__":
    sync_latest_news()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
用LLM API（GPT-4o-mini）从原始新闻中提取股票代码
修复实体映射，提升新闻→股票的覆盖率

使用方法:
    export OPENAI_API_KEY="sk-..."
    export OPENAI_API_BASE="https://api.openai.com/v1"
    python extract_stocks_with_llm.py

成本估算:
    132,998条新闻 ÷ 10条/批 = 13,300次API调用
    每次~1,000 tokens input + 200 tokens output
    GPT-4o-mini: $0.15/1M input + $0.60/1M output
    总成本: ~$2.5-4
"""

import os
import sys
import json
import time
import pandas as pd
import requests
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

# ============ 配置 ============
API_KEY = os.getenv("OPENAI_API_KEY", "")
API_BASE = os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1")
MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
BATCH_SIZE = 10  # 每批处理条数
MAX_WORKERS = 5  # 并发数（控制速率）
DELAY = 0.3  # 每批间隔（秒）
MAX_RETRIES = 3  # 失败重试次数

INPUT_PATH = "D:/iquant_data/data_v2/news_raw_data"
OUTPUT_PATH = "C:/Users/liuqi/quant_system_v2/news_llm_extracted_stocks.csv"
LOG_PATH = "C:/Users/liuqi/quant_system_v2/news_llm_extraction_log.txt"

# ============ LLM Prompt ============
SYSTEM_PROMPT = """你是一位专业的金融新闻实体提取专家。你的任务是从财经新闻中提取涉及的股票代码（A股/港股/美股）。

## 提取规则：
1. 只提取**具体公司**的股票代码，不提取概念/板块/行业
2. 如果新闻中提到的是概念（如"人工智能"、"半导体"、"新能源"），不要输出代码，输出 "CONCEPT"
3. 如果提到的是外国公司（如英伟达、特斯拉、苹果），输出美股代码（如 NVDA.US）
4. 如果提到的是港股公司（如腾讯、美团），输出港股代码（如 0700.HK）
5. 如果新闻中没有提到具体公司，输出 "NONE"
6. 如果公司未上市，输出 "UNLISTED"
7. 优先提取**标题**和**首段**中提到的公司，如果全文搜索可能误匹配
8. 一条新闻可能涉及多个公司，请全部列出

## 输出格式（JSON）：
{
    "stocks": [
        {"code": "600519.SH", "name": "贵州茅台", "confidence": 0.95, "reason": "标题明确提到"},
        {"code": "CONCEPT", "name": "人工智能", "confidence": 0.3, "reason": "概念词，无具体公司"}
    ]
}
"""


def call_llm_api(news_batch):
    """调用LLM API分析新闻批次"""
    if not API_KEY:
        raise ValueError("OPENAI_API_KEY not set!")
    
    news_texts = []
    for i, news in enumerate(news_batch):
        news_texts.append(f"[{i}] 标题: {news['title'][:80]}\n    内容: {news['content'][:150]}")
    
    user_prompt = f"请从以下{len(news_batch)}条新闻中提取涉及的股票代码：\n\n" + "\n\n".join(news_texts)
    
    for attempt in range(MAX_RETRIES):
        try:
            response = requests.post(
                f"{API_BASE}/chat/completions",
                headers={
                    "Authorization": f"Bearer {API_KEY}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": MODEL,
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt}
                    ],
                    "temperature": 0.1,
                    "max_tokens": 2000,
                    "response_format": {"type": "json_object"}
                },
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                content = result['choices'][0]['message']['content']
                return json.loads(content)
            else:
                print(f"  API Error {response.status_code}: {response.text[:100]}")
                if attempt < MAX_RETRIES - 1:
                    time.sleep(2)
                continue
                    
        except Exception as e:
            print(f"  Request Error: {e}")
            if attempt < MAX_RETRIES - 1:
                time.sleep(2)
            continue
    
    return None


def process_all_news():
    """处理所有新闻"""
    import duckdb
    
    # 读取新闻
    print("Reading raw news...")
    con = duckdb.connect()
    df = con.execute(f"SELECT datetime, title, content FROM read_parquet('{INPUT_PATH}/*.parquet')").df()
    con.close()
    
    df['datetime'] = pd.to_datetime(df['datetime'])
    df['news_date'] = df['datetime'].dt.strftime('%Y%m%d')
    
    print(f"Total news: {len(df)}")
    print(f"Date range: {df['news_date'].min()} to {df['news_date'].max()}")
    
    # 分批处理
    all_results = []
    total_batches = (len(df) + BATCH_SIZE - 1) // BATCH_SIZE
    
    # 检查是否有已处理的部分（断点续传）
    if os.path.exists(OUTPUT_PATH):
        existing = pd.read_csv(OUTPUT_PATH)
        start_idx = len(existing)
        print(f"Resuming from batch {start_idx // BATCH_SIZE + 1} (already processed {start_idx})")
        all_results = existing.to_dict('records')
    else:
        start_idx = 0
    
    log_file = open(LOG_PATH, 'a', encoding='utf-8')
    log_file.write(f"\n{'='*60}\n")
    log_file.write(f"Start: {datetime.now()}\n")
    log_file.write(f"Total news: {len(df)}\n")
    log_file.write(f"API: {API_BASE}, Model: {MODEL}\n")
    log_file.write(f"{'='*60}\n")
    
    for batch_idx in range(start_idx // BATCH_SIZE, total_batches):
        start = batch_idx * BATCH_SIZE
        end = min(start + BATCH_SIZE, len(df))
        batch = df.iloc[start:end].to_dict('records')
        
        print(f"\n[{batch_idx + 1}/{total_batches}] Processing {start}-{end} ({len(batch)} items)")
        
        result = call_llm_api(batch)
        
        if result and 'stocks' in result:
            # 解析结果，为每条新闻分配提取的股票
            stocks_list = result.get('stocks', [])
            
            # 简化：每批返回一个列表，我们假设每个元素对应一条新闻
            # 如果返回的是嵌套结构，需要适配
            for i, news in enumerate(batch):
                extracted = []
                if i < len(stocks_list):
                    s = stocks_list[i]
                    if isinstance(s, dict):
                        extracted.append(s)
                    elif isinstance(s, list):
                        extracted = s
                
                all_results.append({
                    'news_date': news['news_date'],
                    'datetime': news['datetime'],
                    'title': news['title'][:100],
                    'extracted_stocks': json.dumps(extracted, ensure_ascii=False),
                    'has_stock': 1 if any(e.get('code', '') and not e.get('code', '').startswith('CONCEPT') and not e.get('code', '').startswith('NONE') and not e.get('code', '').startswith('UNLISTED') for e in extracted) else 0
                })
        else:
            # 失败，填充默认值
            for i, news in enumerate(batch):
                all_results.append({
                    'news_date': news['news_date'],
                    'datetime': news['datetime'],
                    'title': news['title'][:100],
                    'extracted_stocks': '[]',
                    'has_stock': 0
                })
            log_file.write(f"Batch {batch_idx + 1} FAILED\n")
        
        # 每50批保存一次
        if (batch_idx + 1) % 50 == 0:
            pd.DataFrame(all_results).to_csv(OUTPUT_PATH, index=False, encoding='utf-8-sig')
            print(f"  Saved checkpoint: {len(all_results)} records")
            log_file.write(f"Checkpoint: {batch_idx + 1} batches, {len(all_results)} records\n")
        
        time.sleep(DELAY)
    
    # 最终保存
    pd.DataFrame(all_results).to_csv(OUTPUT_PATH, index=False, encoding='utf-8-sig')
    
    log_file.write(f"\nDone: {datetime.now()}, Total: {len(all_results)}\n")
    log_file.close()
    
    print(f"\nDone! Results saved to: {OUTPUT_PATH}")
    print(f"Total processed: {len(all_results)}")
    
    # 统计
    results_df = pd.DataFrame(all_results)
    print(f"Has stock: {results_df['has_stock'].sum()} / {len(results_df)} ({results_df['has_stock'].mean():.2%})")


if __name__ == "__main__":
    if not API_KEY:
        print("ERROR: OPENAI_API_KEY not set!")
        print("Please set environment variable: export OPENAI_API_KEY='sk-...'")
        sys.exit(1)
    
    process_all_news()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
方案A：调用LLM API批量处理新闻情感分析
基于K2.6分析经验设计的专业Prompt

支持：OpenAI GPT-4 / Claude / 本地LLM
"""

import os
import pandas as pd
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

# ============ 配置 ============
API_KEY = os.getenv("OPENAI_API_KEY", "your-api-key-here")
API_BASE = os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1")
MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")  # 或 "claude-3-sonnet-20240229"
BATCH_SIZE = 10  # 每批处理条数
MAX_WORKERS = 5  # 并发数
DELAY = 0.5  # API调用间隔（秒）

# ============ K2.6设计的高质量Prompt ============
SYSTEM_PROMPT = """你是一位专业的金融新闻情感分析师。请对每条新闻进行精细化分析。

## 分析维度：
1. **情感分数** (-3到+3):
   - +3: 极强的利好（如业绩超预期200%+、重大并购重组、技术革命性突破）
   - +2: 强利好（如业绩超预期50%+、获得大额订单、政策支持）
   - +1: 一般利好（如业绩小幅增长、正常业务扩展）
   - 0: 中性（如人事变动、行业动态、无关新闻）
   - -1: 一般利空（如业绩小幅下滑、竞争加剧）
   - -2: 强利空（如业绩暴雷、大股东减持、违规调查）
   - -3: 极强利空（如退市、破产、重大安全事故、实控人失联）

2. **关键判断规则（K2.6经验）**：
   - "辟谣"类新闻：通常是**利空**（市场已price in谣言，辟谣只是确认没有额外利好）
   - "澄清公告"：如果澄清的是市场热点概念，通常是**强利空**（证伪）
   - "高管减持"：比"大股东减持"更负面（-2 vs -1.5）
   - "ST股复牌"：通常是**利空**（停牌核查后复牌往往补跌）
   - "业绩超预期"：需要看超预期的幅度，+50%以上才算强利好
   - "AI/大模型相关"：如果公司本身没有核心技术，蹭热点=中性偏空
   - "政策文件"：看是"支持"还是"限制"，以及具体力度
   - "涨价"：如果是产品涨价=利好，如果是原材料涨价=利空

3. **事件类型**：业绩/并购重组/技术突破/政策/监管/人事/市场动态/宏观/其他

4. **涉及板块**：AI/半导体/新能源/医药/金融/地产/消费/军工/汽车/其他

## 输出格式（JSON）：
{
    "sentiment": 分数,
    "event_type": "事件类型",
    "sector": "板块",
    "reason": "一句话分析理由",
    "stocks": ["可能涉及的股票名称"]
}
"""

USER_PROMPT_TEMPLATE = """请分析以下{count}条金融新闻：

{news_text}

请对每条新闻输出JSON格式分析。"""


def call_llm_api(news_batch):
    """调用LLM API分析新闻批次"""
    import requests
    
    # 构建新闻文本
    news_texts = []
    for i, news in enumerate(news_batch):
        news_texts.append(f"[{i+1}] 标题: {news['title']}
    内容: {news['content'][:200]}")
    
    user_prompt = USER_PROMPT_TEMPLATE.format(
        count=len(news_batch),
        news_text="

".join(news_texts)
    )
    
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
                "temperature": 0.2,  # 低温度保证一致性
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
            print(f"API Error: {response.status_code} - {response.text}")
            return None
            
    except Exception as e:
        print(f"Request Error: {e}")
        return None


def process_news_file(input_path, output_path):
    """处理新闻文件并保存结果"""
    import duckdb
    
    # 读取新闻
    con = duckdb.connect()
    df = con.execute(f"SELECT datetime, title, content FROM read_parquet('{input_path}')").df()
    con.close()
    
    df['datetime'] = pd.to_datetime(df['datetime'])
    df['news_date'] = df['datetime'].dt.strftime('%Y%m%d')
    
    print(f"Total news to process: {len(df)}")
    
    # 分批处理
    all_results = []
    total_batches = (len(df) + BATCH_SIZE - 1) // BATCH_SIZE
    
    for batch_idx in range(total_batches):
        start = batch_idx * BATCH_SIZE
        end = min(start + BATCH_SIZE, len(df))
        batch = df.iloc[start:end].to_dict('records')
        
        print(f"Processing batch {batch_idx+1}/{total_batches} ({start}-{end})")
        
        result = call_llm_api(batch)
        if result:
            all_results.append(result)
        else:
            # 失败时填充默认值
            for news in batch:
                all_results.append({
                    "sentiment": 0,
                    "event_type": "未知",
                    "sector": "其他",
                    "reason": "API调用失败",
                    "stocks": []
                })
        
        time.sleep(DELAY)
    
    # 保存结果
    result_df = pd.DataFrame(all_results)
    result_df['datetime'] = df['datetime']
    result_df['title'] = df['title']
    result_df.to_csv(output_path, index=False, encoding='utf-8-sig')
    
    print(f"Results saved to: {output_path}")
    return result_df


if __name__ == "__main__":
    # 使用示例
    INPUT_PATH = "D:/iquant_data/data_v2/news_raw_data/*.parquet"
    OUTPUT_PATH = "C:/Users/liuqi/quant_system_v2/news_llm_api_results.csv"
    
    process_news_file(INPUT_PATH, OUTPUT_PATH)

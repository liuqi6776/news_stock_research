import os, re, json
from zhipuai import ZhipuAI

DATA_DIR = r'C:\Users\liuqi\clowspace\quant_system_v2\jiayo-analysis\data'
html_file = '1h5dmzjgl6g.html'

API_KEY = ''
client = ZhipuAI(api_key=API_KEY)

with open(os.path.join(DATA_DIR, html_file), 'r', encoding='utf-8') as f:
    html = f.read()

print('HTML size:', len(html), 'bytes')

title_match = re.search(r'<title[^>]*>([^<]+)</title>', html, re.IGNORECASE)
title = title_match.group(1).strip() if title_match else None
print('Title:', title)

content = None
pattern = r'<div[^>]*class="text-box text-justify fsDetail"[^>]*>(.*?)</div>'
content_match = re.search(pattern, html, re.DOTALL)
if content_match:
    content = re.sub(r'<[^>]+>', '', content_match.group(1))
    content = re.sub(r'\s+', ' ', content).strip()
    print('Content found, length:', len(content))

date_match = re.search(r'<div[^>]*class="date[^"]*"[^>]*>.*?(\d{4})-(\d{2})-(\d{2})', html, re.DOTALL)
if date_match:
    article_date = date_match.group(1) + '-' + date_match.group(2) + '-' + date_match.group(3)
    print('Date:', article_date)
else:
    article_date = None

print('Output file: analysis_' + str(article_date or 'unknown') + '.json')

if content and len(content) > 100:
    print('\nAnalyzing with AI...')
    prompt = """你是一个专业的A股市场分析师。请分析以下文章内容，评估其对市场、板块和个股的影响。

文章标题：""" + str(title) + """

文章内容：
""" + content[:4000] + """

请以JSON格式返回分析结果，格式如下：
{
    "market_impact": 利好利空程度(-5到+5，负数表示利空，正数表示利好，0表示中性),
    "market_analysis": "对大盘影响的简要说明（50字以内）",
    "sectors": [
        {
            "sector_name": "板块名称",
            "impact": 利好利空程度(-5到+5),
            "analysis": "简要说明（30字以内）"
        }
    ],
    "stocks": [
        {
            "stock_name": "股票名称",
            "stock_code": "股票代码",
            "impact": 利好利空程度(-5到+5),
            "analysis": "简要说明（30字以内）"
        }
    ]
}

只返回JSON，不要其他内容。"""

    response = client.chat.completions.create(
        model='glm-4-flash',
        messages=[{'role': 'user', 'content': prompt}],
        max_tokens=4096,
        temperature=0.1
    )
    result = response.choices[0].message.content.strip()
    if result.startswith('```json'):
        result = result[7:]
    if result.endswith('```'):
        result = result[:-3]

    print('\n=== JSON Result ===')
    parsed = json.loads(result)
    print(json.dumps(parsed, ensure_ascii=False, indent=2))
else:
    print('No content to analyze')

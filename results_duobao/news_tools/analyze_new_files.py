import os
import re
import json
import time
from zhipuai import ZhipuAI

NEW_DIR = r"C:\Users\liuqi\quant_system_v2\jiayo-analysis\data\new"
OUTPUT_DIR = r"C:\Users\liuqi\quant_system_v2\news_major1"

os.makedirs(OUTPUT_DIR, exist_ok=True)

API_KEY = "a8f0b0b8010c40dd830c8f9c2d981575.JkFd7dLd4kf6u3xK"
client = ZhipuAI(api_key=API_KEY)

def extract_date_from_content(content):
    date_match = re.search(r"(\d{4})-(\d{2})-(\d{2})", content)
    if date_match:
        return f"{date_match.group(1)}-{date_match.group(2)}-{date_match.group(3)}"
    
    title_match = re.search(r"(\d{4})[.年](\d{1,2})[.月](\d{1,2})[日]?", content)
    if title_match:
        return f"{title_match.group(1)}-{int(title_match.group(2)):02d}-{int(title_match.group(3)):02d}"
    
    return None

def extract_title(content):
    title_match = re.search(r"(\d{1,2}月\d{1,2}日盘前纪要|盘前纪要)", content)
    if title_match:
        return title_match.group(1)
    return "盘前纪要"

def analyze_with_ai(content, article_date):
    prompt = f"""请分析以下盘前纪要内容，生成 JSON 格式的市场影响分析。

日期：{article_date}

内容：
{content[:8000]}

请严格按照以下 JSON 格式输出，不要添加任何其他文字：
{{
  "market_impact": 0,
  "market_analysis": "市场影响分析",
  "sectors": [
    {{
      "sector_name": "板块名称",
      "impact": 0,
      "analysis": "板块分析"
    }}
  ],
  "stocks": [
    {{
      "stock_name": "股票名称",
      "stock_code": "股票代码",
      "impact": 0,
      "analysis": "股票分析"
    }}
  ],
  "article_title": "文章标题",
  "html_file": "文件名",
  "article_date": "{article_date}"
}}

注意：
- market_impact: -1=利空, 0=中性, 1=利好
- sector impact: -1=利空, 0=中性, 1=利好
- stock impact: -1=利空, 0=中性, 1=利好
- 只分析有明确信息的板块和股票，不要编造
- stock_code 如果没有可以留空或填 "000000"
"""

    try:
        response = client.chat.completions.create(
            model="glm-4",
            messages=[
                {"role": "system", "content": "你是一个专业的股票市场分析师，擅长从新闻中提取市场影响信息。"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=4000
        )
        
        result_text = response.choices[0].message.content.strip()
        
        json_match = re.search(r'\{[\s\S]*\}', result_text)
        if json_match:
            result_text = json_match.group(0)
        
        result = json.loads(result_text)
        return result
    except Exception as e:
        print(f"  AI 分析错误: {e}")
        return None

def process_file(filepath):
    print(f"\n处理文件: {os.path.basename(filepath)}")
    
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()
        
        article_date = extract_date_from_content(content)
        if not article_date:
            print(f"  无法提取日期，跳过")
            return False
        
        print(f"  日期: {article_date}")
        
        output_file = os.path.join(OUTPUT_DIR, f"analysis_{article_date}.json")
        if os.path.exists(output_file):
            print(f"  文件已存在，跳过: {output_file}")
            return True
        
        title = extract_title(content)
        print(f"  标题: {title}")
        
        print("  正在 AI 分析...")
        analysis_result = analyze_with_ai(content, article_date)
        
        if analysis_result:
            analysis_result["article_title"] = title
            analysis_result["html_file"] = os.path.basename(filepath)
            analysis_result["article_date"] = article_date
            
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(analysis_result, f, ensure_ascii=False, indent=2)
            
            print(f"  ✓ 保存成功: {output_file}")
            return True
        else:
            print(f"  ✗ 分析失败")
            return False
            
    except Exception as e:
        print(f"  ✗ 处理错误: {e}")
        return False

def main():
    files = [f for f in os.listdir(NEW_DIR) if f.endswith(".html")]
    print(f"找到 {len(files)} 个 HTML 文件\n")
    
    success = 0
    skip = 0
    fail = 0
    
    for filename in files:
        filepath = os.path.join(NEW_DIR, filename)
        result = process_file(filepath)
        
        if result:
            if os.path.exists(os.path.join(OUTPUT_DIR, f"analysis_{extract_date_from_content(open(filepath, 'r', encoding='utf-8').read())}.json")):
                skip += 1
            else:
                success += 1
        else:
            fail += 1
        
        time.sleep(2)
    
    print(f"\n=== 完成 ===")
    print(f"成功: {success}, 跳过: {skip}, 失败: {fail}")

if __name__ == "__main__":
    main()

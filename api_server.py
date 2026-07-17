"""
api_server.py — 本地量化引擎 API 服务
连接云端 GitHub Pages Dashboard 与本地计算环境

启动方式:
    python api_server.py
    # 或配合 ngrok 公网穿透:
    ngrok http 8000
"""
import os, json, sys, subprocess, logging, re
from datetime import datetime
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

import pandas as pd
from fastapi import FastAPI, HTTPException, Request, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ───────────────────────────── 路径配置 ─────────────────────────────
BASE_DIR        = Path(__file__).parent.resolve()
JIAYO_DIR       = BASE_DIR / "jiayo-analysis"
NEWS_DIR        = Path("D:/iquant_data/data_v2/news_major1")

PRED_FILE_004 = BASE_DIR / "research/study_004_1d_release/predictions/predictions_1d_open_wf_monthly.parquet"
PRED_FILE_005 = BASE_DIR / "research/study_005_1d_advanced/predictions/predictions_005_options_wf.parquet"

# ───────────────────────────── 参数 ─────────────────────────────────
MAX_POSITIONS   = 3
STOP_LOSS       = -0.05
TAKE_PROFIT     = 0.05

# ───────────────────────────── App ──────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(title="Quant Local Engine API", version="1.1.0")

def sync_options_data():
    try:
        from data.update_options_data import main as sync_options
        logger.info("Auto-syncing options PCR data...")
        sync_options()
        logger.info("Options PCR data synced successfully.")
    except Exception as e:
        logger.error(f"Failed to auto-sync options PCR data: {e}")

@app.on_event("startup")
def startup_event():
    logger.info("API server started. Running initial options data sync...")
    sync_options_data()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://liuqi6776.github.io",
        "https://percolate-zipfile-corned.ngrok-free.dev",
        "http://localhost:3000", "http://127.0.0.1:3000",
        "http://localhost:5500", "http://127.0.0.1:5500",
        "null",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────── 数据模型 ───────────────────────────────────
class PipelineRequest(BaseModel):
    target_date: str          # YYYY-MM-DD
    news_json: Optional[str] = None  # 已经解析好的 JSON 字符串
    strategy: str = "conservative" # conservative 或 aggressive

class ParseHtmlRequest(BaseModel):
    target_date: str          # YYYY-MM-DD
    raw_html: Optional[str] = None    # HTML 原文
    raw_markdown: Optional[str] = None  # Markdown 原文（二选一）
    save_to_disk: bool = True  # 是否保存到 D:/iquant_data/data_v2/news_major1/


# ─────────────────────── 工具函数 ───────────────────────────────────
def normalize_date(date_str: str) -> str:
    return date_str.replace("-", "")


def get_public_url() -> str:
    try:
        import urllib.request
        resp = urllib.request.urlopen("http://127.0.0.1:4040/api/tunnels", timeout=2)
        tunnels = json.loads(resp.read().decode()).get("tunnels", [])
        if tunnels:
            return tunnels[0].get("public_url", "http://127.0.0.1:8000")
    except Exception:
        pass
    return "http://127.0.0.1:8000"


def get_signals_for_date(date_norm: str, strategy: str = "conservative") -> dict:
    is_conservative = (strategy == "conservative")
    pred_file = PRED_FILE_005 if is_conservative else PRED_FILE_004
    
    if not pred_file.exists():
        raise FileNotFoundError(f"预测文件不存在: {pred_file}")
        
    df = pd.read_parquet(pred_file)
    df["trade_date"] = df["trade_date"].astype(str)
    today = df[df["trade_date"] == date_norm].copy()
    
    if today.empty:
        return {"signals": [], "is_fallback": False, "no_data": True}
        
    if is_conservative:
        above = today[(today["prob_up"] >= 0.50) & (today["prob_crash"] <= 0.45)].copy()
        above = above.sort_values("prob_up", ascending=False)
        ind_counts = {}
        candidates = []
        for _, row in above.iterrows():
            ind = row.get("industry", "Unknown")
            if ind_counts.get(ind, 0) >= 2:
                continue
            candidates.append(row)
            ind_counts[ind] = ind_counts.get(ind, 0) + 1
            if len(candidates) >= MAX_POSITIONS:
                break
        candidates = pd.DataFrame(candidates) if candidates else pd.DataFrame(columns=today.columns)
        fallback_col = "prob_up"
    else:
        above = today[today["prob"] >= 0.50]
        if len(above) >= MAX_POSITIONS:
            candidates = above.nlargest(MAX_POSITIONS, "prob").reset_index(drop=True)
        else:
            candidates = above.reset_index(drop=True)
        fallback_col = "prob"

    is_fallback = len(candidates) == 0
    if is_fallback:
        top5 = today.nlargest(5, fallback_col)
        candidates = top5.reset_index(drop=True)
        
    results = []
    for _, row in candidates.iterrows():
        ep = row["entry_price"] if pd.notna(row["entry_price"]) else None
        
        prob_val = float(row["prob_up"]) if is_conservative else float(row["prob"])
        extra_info = ""
        if is_conservative:
            extra_info = f" | 防守指标: 暴跌概率 {float(row['prob_crash']):.1%} | 板块: {row.get('industry', 'Unknown')}"
            
        results.append({
            "ts_code":     row["ts_code"],
            "prob":        round(prob_val, 4),
            "entry_price": round(float(ep), 2) if ep else None,
            "action":      f"T+1 以开盘价买入{extra_info}",
            "stop_loss":   f"{STOP_LOSS*100:.0f}%",
            "take_profit": f"+6% 固定止盈" if is_conservative else f"+{TAKE_PROFIT*100:.0f}%",
        })

    # Extra: Rules Text & Top 10 Before Filtering
    rules_text = (
        "1. 筛选基础: 上涨概率 prob_up >= 50% \n"
        "2. 风控防御: 暴跌防守概率 prob_crash <= 45% \n"
        "3. 板块限额: 同一行业板块最多推荐 2 只股票 (分散行业风险) \n"
        "4. 仓位控制: 每日最多买入推荐前 3 只股票 (Max Positions = 3)"
    ) if is_conservative else (
        "1. 筛选基础: 预测概率 prob >= 50% \n"
        "2. 仓位控制: 每日最多买入推荐前 3 只股票 (Max Positions = 3)"
    )

    score_col = "prob_up" if is_conservative else "prob"
    top10_raw = today.nlargest(10, score_col)
    top10_results = []
    for idx, (_, row) in enumerate(top10_raw.iterrows(), 1):
        ep = row["entry_price"] if pd.notna(row["entry_price"]) else None
        prob_val = float(row[score_col])
        extra = f"行业: {row.get('industry', 'Unknown')}"
        if is_conservative:
            extra += f" | 暴跌概率: {float(row['prob_crash']):.1%}"
            
        top10_results.append({
            "rank":        idx,
            "ts_code":     row["ts_code"],
            "prob":        round(prob_val, 4),
            "entry_price": round(float(ep), 2) if ep else None,
            "info":        extra,
        })

    return {
        "signals": results, 
        "is_fallback": is_fallback,
        "rules": rules_text,
        "top10_raw": top10_results
    }


def extract_content_from_html(html: str) -> tuple[str, str]:
    """从 HTML 中提取 title 和纯文本 content"""
    title_m = re.search(r'<title[^>]*>([^<]+)</title>', html, re.IGNORECASE)
    title = title_m.group(1).strip() if title_m else "Unknown"
    title = re.sub(r'\s*[-_|]\s*韭研公社\s*$', '', title)

    # 优先从 text-box 抽取正文
    body_m = re.search(r'<div[^>]*class="[^"]*text-box[^"]*"[^>]*>(.*?)</div>\s*<div[^>]*class="[^"]*reward', html, re.DOTALL)
    if body_m:
        body = body_m.group(1)
    else:
        body_m2 = re.search(r'<body[^>]*>(.*?)</body>', html, re.DOTALL)
        body = body_m2.group(1) if body_m2 else html

    # strip scripts/styles
    body = re.sub(r'<script[^>]*>.*?</script>', '', body, flags=re.DOTALL)
    body = re.sub(r'<style[^>]*>.*?</style>', '', body, flags=re.DOTALL)
    body = re.sub(r'<[^>]+>', ' ', body)
    body = re.sub(r'&nbsp;', ' ', body)
    body = re.sub(r'&amp;', '&', body)
    body = re.sub(r'\s+', ' ', body).strip()

    return title, body


def analyze_with_zhipu(title: str, content: str) -> Optional[dict]:
    """调用 ZhipuAI GLM 分析文章，复用 jiayo-analysis/analyzer.py 的逻辑"""
    try:
        from zhipuai import ZhipuAI
        API_KEY = os.getenv("ZHIPU_API_KEY")
        if not API_KEY:
            logger.error("ZHIPU_API_KEY is not set in environment!")
            return None
        client = ZhipuAI(api_key=API_KEY)
    except ImportError:
        raise HTTPException(status_code=500, detail="zhipuai 未安装，请运行: pip install zhipuai")

    prompt = f"""你是一个专业的A股市场分析师。请分析以下文章内容，评估其对市场、板块和个股的影响。

文章标题：{title}

文章内容：
{content[:5000]}

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
            "stock_code": "股票代码（如有）",
            "impact": 利好利空程度(-5到+5),
            "analysis": "简要说明（30字以内）"
        }}
    ]
}}

只返回JSON，不要其他内容。"""

    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model="glm-4-flash",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=4096,
                temperature=0.1,
            )
            text = resp.choices[0].message.content.strip()
            if text.startswith("```json"):
                text = text[7:]
            if text.endswith("```"):
                text = text[:-3]
            return json.loads(text.strip())
        except json.JSONDecodeError as e:
            logger.warning(f"JSON decode error attempt {attempt+1}: {e}")
        except Exception as e:
            logger.warning(f"ZhipuAI error attempt {attempt+1}: {e}")
    return None


def get_data_status() -> dict:
    status = {}
    for name, path in [("predictions_aggressive", PRED_FILE_004), ("predictions_conservative", PRED_FILE_005)]:
        if path.exists():
            df = pd.read_parquet(path, columns=["trade_date"])
            dates = df["trade_date"].astype(str)
            status[name] = {"exists": True, "rows": len(df), "date_min": dates.min(), "date_max": dates.max()}
        else:
            status[name] = {"exists": False}
    return status


# ─────────────────────── 接口定义 ───────────────────────────────────
@app.get("/health")
def health_check():
    """连通性测试"""
    return {
        "status": "ok",
        "msg": "Local Quant Engine is Online ✓",
        "endpoint": get_public_url(),
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "data_status": get_data_status(),
    }


@app.get("/api/v1/signals")
def get_signals(date: str, strategy: str = "conservative"):
    """获取指定日期的交易信号。date 格式: 20260509 或 2026-05-09，strategy: conservative 或 aggressive"""
    date_norm = normalize_date(date)
    try:
        result = get_signals_for_date(date_norm, strategy)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    sigs = result["signals"]
    is_fb = result["is_fallback"]
    no_data = result.get("no_data", False)
    resp = {
        "status": "ok", 
        "date": date_norm, 
        "strategy": strategy, 
        "signals": sigs, 
        "signal_count": len(sigs),
        "rules": result.get("rules", ""),
        "top10_raw": result.get("top10_raw", [])
    }
    if no_data:
        resp["warning"] = f"日期 {date_norm} 无预测数据"
    elif is_fb:
        resp["warning"] = "无股票满足筛选条件，以下为prob最高的5只股票（仅供参考，非正式信号）"
    return resp


@app.post("/api/v1/parse/html")
async def parse_html_to_json(payload: ParseHtmlRequest):
    """
    将 raw HTML 或 Markdown 原文解析成 news_major1 JSON 格式。
    流程: 提取正文 → 调用 ZhipuAI GLM 分析 → 返回 JSON（可选保存到本地）
    """
    date_fmt = payload.target_date  # YYYY-MM-DD

    if not payload.raw_html and not payload.raw_markdown:
        raise HTTPException(status_code=400, detail="raw_html 或 raw_markdown 必须提供其中一个")

    # 1. 提取正文
    if payload.raw_html:
        title, content = extract_content_from_html(payload.raw_html)
    else:
        # Markdown: 直接去掉 markdown 标记符作为 content
        md = payload.raw_markdown
        title_m = re.search(r'^#+ (.+)$', md, re.MULTILINE)
        title = title_m.group(1).strip() if title_m else "盘前纪要"
        content = re.sub(r'#+ ', '', md)
        content = re.sub(r'\*+', '', content)
        content = re.sub(r'\s+', ' ', content).strip()

    logger.info(f"Parsing article: '{title}', content length={len(content)}")

    if len(content) < 100:
        raise HTTPException(status_code=422, detail=f"内容提取失败或过短（{len(content)}字），请检查输入格式")

    # 2. ZhipuAI 分析
    result = analyze_with_zhipu(title, content)
    if not result:
        raise HTTPException(status_code=502, detail="ZhipuAI 分析失败（3次重试均未能返回有效JSON）")

    result["article_title"] = title
    result["article_date"] = date_fmt

    # 3. 保存到本地
    saved_path = None
    if payload.save_to_disk:
        try:
            NEWS_DIR.mkdir(parents=True, exist_ok=True)
            out_path = NEWS_DIR / f"analysis_{date_fmt}.json"
            out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
            saved_path = str(out_path)
            logger.info(f"Saved analysis to: {saved_path}")
        except Exception as e:
            logger.warning(f"Save failed: {e}")

    return {
        "status": "ok",
        "date": date_fmt,
        "title": title,
        "content_length": len(content),
        "saved_to": saved_path,
        "analysis": result,
    }


@app.post("/api/v1/pipeline/run")
async def run_pipeline(payload: PipelineRequest):
    """
    主流程接口:
    1. 保存 news JSON 到本地（如果提供）
    2. 读取当日预测信号并返回
    """
    date_raw  = payload.target_date
    date_norm = normalize_date(date_raw)
    date_fmt  = f"{date_raw[:4]}-{date_raw[5:7]}-{date_raw[8:10]}"
    
    strategy = payload.strategy

    steps = []

    if payload.news_json:
        try:
            NEWS_DIR.mkdir(parents=True, exist_ok=True)
            news_path = NEWS_DIR / f"analysis_{date_fmt}.json"
            news_path.write_text(payload.news_json, encoding="utf-8")
            steps.append(f"✓ 新闻JSON已保存: {news_path}")
        except Exception as e:
            steps.append(f"⚠ 新闻保存失败: {e}")
    else:
        steps.append("ℹ 本次未附带新闻JSON")

    try:
        result = get_signals_for_date(date_norm, strategy)
        sigs = result["signals"]
        is_fb = result["is_fallback"]
        no_data = result.get("no_data", False)
        step_msg = f"✓ 信号读取完成: {len(sigs)} 个候选标的"
        if no_data:
            step_msg += f" ⚠ 日期 {date_norm} 无预测数据"
        elif is_fb:
            step_msg += " ⚠ 无满足条件股票，显示prob最高5只"
        steps.append(step_msg)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=f"预测文件缺失: {e}")

    resp = {
        "status": "success",
        "date": date_norm,
        "steps": steps,
        "signal_count": len(sigs),
        "results": sigs,
        "rules": result.get("rules", ""),
        "top10_raw": result.get("top10_raw", [])
    }
    if no_data:
        resp["warning"] = f"日期 {date_norm} 无预测数据"
    elif is_fb:
        resp["warning"] = "无股票满足筛选条件，以下为prob最高的5只股票（仅供参考，非正式信号）"
    return resp


def run_retrain_task():
    logger.info("Background options model retrain task started...")
    try:
        from run_retrain_with_options import run_retrain
        success = run_retrain()
        if success:
            logger.info("Background options model retrain completed successfully!")
        else:
            logger.error("Background options model retrain failed.")
    except Exception as e:
        logger.error(f"Error during background options model retrain: {e}")

@app.post("/api/v1/retrain/trigger")
async def trigger_retrain(background_tasks: BackgroundTasks):
    """后台触发量化模型重训（异步，串联期权同步、特征构建与滚动重训）"""
    background_tasks.add_task(run_retrain_task)
    return {
        "status": "started",
        "msg": "Option-enhanced model retraining has been triggered in the background. It will sync PCR data, rebuild features, and retrain XGBoost models."
    }


# ─────────────────────── 启动入口 ───────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    public_url = get_public_url()
    print("=" * 60)
    print(f"  Quant Local Engine")
    print(f"  Endpoint: {public_url}")
    print(f"  Docs:     {public_url}/docs")
    print("=" * 60)
    uvicorn.run("api_server:app", host="0.0.0.0", port=8000, reload=False, log_level="info")

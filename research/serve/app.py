# -*- coding: utf-8 -*-
"""量化策略每日信号与实时监控服务 (Quant Strategy Daily Signal Service)

启动命令:
    python research/serve/app.py   (默认本地运行: http://127.0.0.1:8000)

API 接口列表:
    GET  /               - 交互式策略监控仪表盘主页
    GET  /api/today      - 今日最新复合量化信号 (包含宏观S123、Top40选股、V8避险、IM对冲)
    GET  /api/history    - 历史调仓信号时间轴
    GET  /api/picks      - 指定日期的持仓与配置明细 (?date=YYYY-MM-DD)
    GET  /api/performance - 全历史策略回测指标与阶段对冲数据
    POST /api/run_signal - 触发后台重新生成最新信号
"""
import os
import sys
import glob
import json
import asyncio
from concurrent.futures import ThreadPoolExecutor

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SERVE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SERVE_DIR, "data", "composite")
DAILY_DIR = os.path.join(SERVE_DIR, "data", "daily")
ASSETS_DIR = os.path.join(SERVE_DIR, "assets")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(DAILY_DIR, exist_ok=True)
os.makedirs(ASSETS_DIR, exist_ok=True)

if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
if SERVE_DIR not in sys.path:
    sys.path.insert(0, SERVE_DIR)

from composite_signal_generator import generate_composite_signal  # noqa: E402

app = FastAPI(
    title="A-Share Composite Quant Strategy Service",
    description="全市场ENS选股+S123宏观择时+净值回撤降档+IM期货低基差对冲",
    version="2.2.0"
)
app.mount("/assets", StaticFiles(directory=ASSETS_DIR), name="assets")

executor = ThreadPoolExecutor(max_workers=2)
is_generating = False


def _load_daily_signals():
    files = sorted(glob.glob(os.path.join(DATA_DIR, "*.json")))
    out = []
    for fp in files:
        try:
            with open(fp, "r", encoding="utf-8") as fh:
                d = json.load(fh)
            d["file_date"] = os.path.splitext(os.path.basename(fp))[0]
            out.append(d)
        except Exception:
            continue
    out.sort(key=lambda x: x["file_date"], reverse=True)
    return out


@app.get("/", response_class=HTMLResponse)
def index():
    html_path = os.path.join(SERVE_DIR, "templates", "index.html")
    if not os.path.exists(html_path):
        raise HTTPException(404, "Dashboard template not found")
    with open(html_path, "r", encoding="utf-8") as fh:
        return fh.read()


@app.get("/api/today")
def api_today():
    daily = _load_daily_signals()
    if not daily:
        # 若暂无历史文件，现场生成一期
        try:
            sig = generate_composite_signal()
            return JSONResponse(sig)
        except Exception as e:
            raise HTTPException(500, f"信号生成失败: {str(e)}")
    return JSONResponse(daily[0])


@app.get("/api/history")
def api_history(limit: int = 90):
    daily = _load_daily_signals()
    return JSONResponse(daily[:limit])


@app.get("/api/picks")
def api_picks(date: str):
    fp = os.path.join(DATA_DIR, f"{date}.json")
    if not os.path.exists(fp):
        raise HTTPException(404, f"未找到 {date} 调仓记录")
    with open(fp, "r", encoding="utf-8") as fh:
        return JSONResponse(json.load(fh))


@app.get("/api/performance")
def api_performance():
    return JSONResponse({
        "strategy_name": "全市场ENS+三档S123+净值降档(-10%×0.5)+IM对冲(β=0.5)",
        "backtest_range": "2019-06-03 ~ 2026-08-17 (1,748 交易日)",
        "core_stock_strategy": {
            "cagr": "11.80%",
            "sharpe": 0.77,
            "maxdd_daily": "-25.48%",
            "maxdd_monthly": "-19.02%",
            "calmar": 0.46,
            "monthly_win_rate": "57.0%",
            "total_return": "+123.9%"
        },
        "im_hedged_strategy_beta_05": {
            "cagr": "11.78%",
            "sharpe": 0.94,
            "maxdd_daily": "-10.90%",
            "maxdd_monthly": "-8.01%",
            "calmar": 1.08,
            "evaluation_period": "2023 ~ 2026 (严格样本外 OOS)"
        },
        "benchmarks": {
            "csi_1000": {"cagr": "5.11%", "maxdd_daily": "-46.71%", "sharpe": 0.13},
            "csi_300": {"cagr": "3.48%", "maxdd_daily": "-45.60%", "sharpe": 0.08}
        },
        "annual_breakdown": {
            "2019": {"strategy": "+1.5%", "csi_1000": "+5.9%", "csi_300": "+12.8%"},
            "2020": {"strategy": "+19.3%", "csi_1000": "+17.1%", "csi_300": "+25.5%"},
            "2021": {"strategy": "+0.3%", "csi_1000": "+17.8%", "csi_300": "-6.2%"},
            "2022": {"strategy": "+17.5%", "csi_1000": "-21.3%", "csi_300": "-21.3%"},
            "2023": {"strategy": "+6.2%", "csi_1000": "-8.4%", "csi_300": "-11.7%"},
            "2024": {"strategy": "+12.6%", "csi_1000": "+1.8%", "csi_300": "+16.2%"},
            "2025": {"strategy": "+31.7%", "csi_1000": "+31.0%", "csi_300": "+21.2%"},
            "2026": {"strategy": "-2.4%", "csi_1000": "-2.9%", "csi_300": "-1.4%"}
        }
    })


@app.post("/api/run_signal")
async def api_run_signal(background_tasks: BackgroundTasks):
    global is_generating
    if is_generating:
        return JSONResponse({"status": "busy", "message": "信号计算任务正在后台执行中，请稍候刷新"})
    
    def _task():
        global is_generating
        is_generating = True
        try:
            generate_composite_signal()
        finally:
            is_generating = False

    background_tasks.add_task(_task)
    return JSONResponse({"status": "started", "message": "后台信号重算任务已启动，大约需要 1-2 分钟完成"})


@app.get("/api/chart")
def api_chart():
    chart_path = os.path.join(ASSETS_DIR, "integrated_composite_nav.png")
    if not os.path.exists(chart_path):
        raise HTTPException(404, "Chart asset not found")
    return FileResponse(chart_path, media_type="image/png")


if __name__ == "__main__":
    import uvicorn
    print("=" * 70)
    print(">>> 启动量化策略服务 (http://127.0.0.1:8000)...")
    print("=" * 70)
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=False)

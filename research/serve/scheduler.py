# -*- coding: utf-8 -*-
"""量化策略每日自动化调度与常驻服务 (Quant Strategy Daily Scheduler & Service)

功能:
  1. 启动 FastAPI Web 仪表盘服务 (http://127.0.0.1:8000)
  2. 每日早晨 08:00 AM 自动触发信号生成引擎，在开盘前 (9:30) 准备就绪今日操作建议
  3. 支持 Windows 计划任务与开机常驻

启动:
    python research/serve/scheduler.py
"""
import os
import sys
import time
import datetime
import threading

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SERVE_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
if SERVE_DIR not in sys.path:
    sys.path.insert(0, SERVE_DIR)

from composite_signal_generator import generate_composite_signal  # noqa: E402
import uvicorn  # noqa: E402


def run_daily_job_at_8am():
    """每日 08:00 AM 定时执行信号生成循环"""
    print("\n[调度器] 每日 08:00 AM 定时任务调度器已激活...")
    last_run_date = None
    
    while True:
        now = datetime.datetime.now()
        today_str = now.strftime("%Y-%m-%d")
        
        # 检查是否到达早晨 08:00 (08:00 - 08:05 窗口内且今日尚未执行)
        if now.hour == 8 and now.minute >= 0 and last_run_date != today_str:
            print(f"\n⏰ [{now.strftime('%Y-%m-%d %H:%M:%S')}] 触发早晨 08:00 AM 每日信号自动生成...")
            try:
                generate_composite_signal()
                last_run_date = today_str
                print(f"✅ [{datetime.datetime.now().strftime('%H:%M:%S')}] 今日信号已就绪，可在仪表盘查看！\n")
            except Exception as e:
                print(f"❌ [{datetime.datetime.now().strftime('%H:%M:%S')}] 信号生成异常: {e}\n")
        
        # 每 30 秒轮询一次时间
        time.sleep(30)


def start_service():
    print("=" * 80)
    print("  🚀 启动 A股综合复合量化策略服务 (含每日 08:00 AM 自动定时调度)")
    print("  🌐 Web 仪表盘地址: http://127.0.0.1:8000")
    print("  ⏰ 定时任务: 每日早晨 08:00 AM 自动刷新最新信号 (开盘前准备就绪)")
    print("=" * 80)

    # 1. 启动后台定时任务线程
    scheduler_thread = threading.Thread(target=run_daily_job_at_8am, daemon=True)
    scheduler_thread.start()

    # 2. 启动 FastAPI Web 监控服务
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=False, log_level="info")


if __name__ == "__main__":
    start_service()

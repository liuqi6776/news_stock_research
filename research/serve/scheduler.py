# -*- coding: utf-8 -*-
"""量化策略每日自动化调度与常驻服务 (Quant Strategy Daily Scheduler & Service)

功能:
  1. 启动 FastAPI Web 仪表盘服务 (http://127.0.0.1:8000)
  2. 每日早晨 07:00 AM 自动触发信号生成引擎，并推送邮件晨报给用户
  3. 支持 Windows 计划任务与后台常驻

启动命令:
    python research/serve/scheduler.py                 # 常驻服务 (07:00 自动推送 + 启动 Web)
    python research/serve/scheduler.py --send-now       # 立即计算并发送一封测试邮件
"""
import os
import sys
import time
import datetime
import threading
import argparse

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SERVE_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
if SERVE_DIR not in sys.path:
    sys.path.insert(0, SERVE_DIR)

from composite_signal_generator import generate_composite_signal  # noqa: E402
from notify import send_email_html, build_signal_email_html  # noqa: E402
import uvicorn  # noqa: E402


def execute_daily_routine():
    """执行每日信号刷新与邮件推送核心流程"""
    now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"\n[调度] [{now_str}] 正在执行每日早晨 07:00 AM 信号计算与邮件推送...")
    try:
        sig = generate_composite_signal()
        sig_date = sig.get("signal_date", datetime.datetime.now().strftime("%Y-%m-%d"))
        
        # 构建并发送 HTML 邮件
        email_body = build_signal_email_html(sig)
        subject = f"【量化策略晨报】{sig_date} 调仓决策与持仓清单"
        email_ok = send_email_html(subject, email_body)
        
        if email_ok:
            print(f"[成功] [{datetime.datetime.now().strftime('%H:%M:%S')}] 邮件推送成功！今日信号已发布至 Web 仪表盘。\n")
        else:
            print(f"[提示] [{datetime.datetime.now().strftime('%H:%M:%S')}] 信号已生成，但邮件发送失败 (请检查 SMTP 配置)。\n")
        return True
    except Exception as e:
        print(f"[错误] [{datetime.datetime.now().strftime('%H:%M:%S')}] 每日任务执行异常: {e}\n")
        return False


def run_daily_job_at_7am():
    """每日 07:00 AM 定时执行信号生成与邮件推送循环"""
    print("[调度器] 每日早晨 07:00 AM 定时任务调度器已就绪...")
    last_run_date = None
    
    while True:
        now = datetime.datetime.now()
        today_str = now.strftime("%Y-%m-%d")
        
        # 检查是否到达早晨 07:00 (07:00 - 07:05 窗口内且今日尚未执行)
        if now.hour == 7 and now.minute >= 0 and last_run_date != today_str:
            success = execute_daily_routine()
            if success:
                last_run_date = today_str
        
        # 每 30 秒轮询一次时间
        time.sleep(30)


def start_service():
    print("=" * 80)
    print("  启动 A股综合复合量化策略服务")
    print("  Web 仪表盘地址: http://127.0.0.1:8000")
    print("  定时任务: 每日早晨 07:00 AM 自动刷新信号并发送 Email 晨报")
    print("=" * 80)

    # 1. 启动后台定时任务线程
    scheduler_thread = threading.Thread(target=run_daily_job_at_7am, daemon=True)
    scheduler_thread.start()

    # 2. 启动 FastAPI Web 监控服务 (主线程阻塞运行)
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=False, log_level="info")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="量化策略每日调度器与服务")
    parser.add_argument("--send-now", action="store_true", help="立即运行一次信号生成并发送邮件")
    args = parser.parse_args()

    if args.send_now:
        print(">>> 立即执行一次信号计算并推送邮件...")
        execute_daily_routine()
    else:
        start_service()

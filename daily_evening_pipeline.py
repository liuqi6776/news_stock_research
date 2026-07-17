"""
daily_evening_pipeline.py
每日晚间 A股量化数据同步主程序。
运行时间建议：交易日 21:00 - 23:00

流程：
1. 自动同步/增量获取最新的期权 PCR 与成交量数据（调用 update_options_data.py）。
2. 获取今天交易日（T日）的行情、基本指标、同花顺热股榜、筹码分布等 Tushare 数据（调用 fetch_latest_data.py）。
3. 打印出完整的数据就绪状态，为次日早晨的 8:00 量化晨报做好充足的前置数据准备。
"""
import os
import sys
import subprocess
from datetime import datetime

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

def is_today_trading_day():
    """判断今天是否为A股交易日"""
    now = datetime.now()
    if now.weekday() in (5, 6):
        return False
    import akshare as ak
    try:
        df_cal = ak.tool_trade_date_hist_sina()
        cal_dates = set(df_cal['trade_date'].astype(str).str.replace('-', '').tolist())
        today_str = now.strftime('%Y%m%d')
        return today_str in cal_dates
    except Exception as e:
        print(f"[WARNING] 无法获取交易日历: {e}. 默认工作日为交易日。")
        return True

def run_evening_sync():
    print(f"=== 开启每日晚间数据自动同步管线: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===")
    
    if not is_today_trading_day():
        print("[INFO] 今天是非交易日，跳过晚间行情与期权数据同步。")
        return

    today_str = datetime.now().strftime('%Y%m%d')
    print(f"[INFO] 今日交易日确定为: {today_str}")

    # 1. 同步期权 PCR 数据
    print("\n--- 步骤 1: 增量更新期权 PCR 数据 (Akshare) ---")
    try:
        from data.update_options_data import main as sync_options
        sync_options()
        print("[SUCCESS] 期权 PCR 数据更新成功！")
    except Exception as e:
        print(f"[ERROR] 期权 PCR 数据更新失败: {e}")

    # 2. 调用 fetch_latest_data.py 下载今日 A股日线/基本面/热榜/筹码数据
    print(f"\n--- 步骤 2: 获取今日 A股行情与热榜数据 (Tushare) for {today_str} ---")
    try:
        # 由于是晚上运行，直接传入今天日期
        subprocess.run([sys.executable, "fetch_latest_data.py", today_str], check=True)
        print("[SUCCESS] 今日 A股 Tushare 数据拉取完成！")
    except Exception as e:
        print(f"[ERROR] 获取今日 A股 Tushare 数据失败: {e}")

    print(f"\n=== 每日晚间数据同步管线执行完毕: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===")

if __name__ == "__main__":
    run_evening_sync()

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
场内基金全量历史日线数据采集工具 (2014-01-01 至今, >10年)
存储到 D:\iquant_data\data_v2\fund1\{YYYYMMDD}.parquet

数据源: Tushare fund_daily (场内基金每日行情)
字段: ts_code, trade_date, open, high, low, close, pre_close, change, pct_chg, vol, amount
"""

import os
import sys
import time
import argparse
import logging
from datetime import datetime

import pandas as pd
import tushare as ts
from dotenv import load_dotenv

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENV_PATH = os.path.join(ROOT, ".env")

DATA_DIR = r"D:\iquant_data\data_v2\fund1"
FUND_BASIC_PATH = os.path.join(DATA_DIR, "fund_basic_E.parquet")
DEFAULT_START = "20140101"

# Tushare 限频控制: 每次请求间隔(秒), 失败重试
SLEEP_SEC = 0.35
MAX_RETRY = 8
RETRY_BASE_SLEEP = 5


def load_token():
    if os.path.exists(ENV_PATH):
        load_dotenv(ENV_PATH)
    token = os.getenv("TUSHARE_TOKEN")
    if not token or token.strip() in ("", "your_tushare_token_here"):
        raise SystemExit("TUSHARE_TOKEN 未配置，请检查 .env 文件")
    return token.strip()


def get_trading_dates(pro, start, end):
    """通过交易日历接口获取交易日"""
    df = pro.trade_cal(exchange='SSE', start_date=start, end_date=end, is_open='1')
    if df is None or df.empty:
        raise SystemExit(f"无法获取交易日历: {start}-{end}")
    return sorted(df["cal_date"].tolist())


def fetch_with_retry(pro, method, **kwargs):
    """带重试的 API 调用，处理限频"""
    func = getattr(pro, method)
    for attempt in range(1, MAX_RETRY + 1):
        try:
            return func(**kwargs)
        except Exception as e:
            if attempt >= MAX_RETRY:
                logger.error("%s(%s) 重试 %d 次仍失败: %s", method, kwargs, MAX_RETRY, e)
                raise
            wait = RETRY_BASE_SLEEP * attempt
            logger.warning("%s 调用失败(%d/%d): %s，%.0fs 后重试",
                           method, attempt, MAX_RETRY, str(e)[:120], wait)
            time.sleep(wait)
    return None


def fetch_fund_basic(pro):
    """获取全部场内基金基本信息 (上市+退市, 便于字段映射)"""
    frames = []
    for status in ("L", "D"):
        df = fetch_with_retry(pro, "fund_basic",
                              market="E", status=status,
                              fields="ts_code,name,management,found_date,due_date,"
                                     "list_date,delist_date,issue_date,list_status")
        if df is not None and not df.empty:
            frames.append(df)
            logger.info("fund_basic status=%s: %d 只", status, len(df))
        time.sleep(SLEEP_SEC)
    if frames:
        basic = pd.concat(frames, ignore_index=True).drop_duplicates("ts_code")
        basic.to_parquet(FUND_BASIC_PATH, index=False)
        logger.info("场内基金基本信息已保存: %d 只 -> %s", len(basic), FUND_BASIC_PATH)
    else:
        logger.warning("fund_basic 未获取到数据")


def fetch_fund_daily(pro, dates):
    """按交易日逐日获取场内基金日线并保存 parquet"""
    os.makedirs(DATA_DIR, exist_ok=True)
    existing = {f[:-8] for f in os.listdir(DATA_DIR) if f.endswith(".parquet")}
    todo = [d for d in dates if d not in existing]
    logger.info("计划 %d 个交易日，已存在 %d 个，待下载 %d 个",
                len(dates), len(dates) - len(todo), len(todo))

    failed = []
    for i, d in enumerate(todo, 1):
        try:
            df = fetch_with_retry(pro, "fund_daily", trade_date=d)
            if df is None:
                failed.append(d)
                continue
            if df.empty:
                # 空数据也写一个空 parquet 占位，避免重复请求
                df = pd.DataFrame(columns=["ts_code", "trade_date", "open", "high",
                                           "low", "close", "pre_close", "change",
                                           "pct_chg", "vol", "amount"])
            df.to_parquet(os.path.join(DATA_DIR, f"{d}.parquet"), index=False)
            if i % 50 == 0 or i == len(todo):
                logger.info("进度 %d/%d (%.1f%%)", i, len(todo), i / len(todo) * 100)
        except Exception as e:
            logger.error("日期 %s 下载失败: %s", d, e)
            failed.append(d)
        time.sleep(SLEEP_SEC)

    if failed:
        logger.warning("以下 %d 个交易日失败: %s", len(failed), failed)
    else:
        logger.info("全部完成，无失败日期")
    return failed


def main():
    ap = argparse.ArgumentParser(description="场内基金日线数据下载")
    ap.add_argument("--start", default=DEFAULT_START, help="开始日期 YYYYMMDD")
    ap.add_argument("--end", default=None, help="结束日期 YYYYMMDD，默认今天")
    args = ap.parse_args()

    end = args.end or datetime.now().strftime("%Y%m%d")
    logger.info("场内基金数据下载: %s - %s", args.start, end)

    pro = ts.pro_api(load_token())
    dates = get_trading_dates(pro, args.start, end)
    logger.info("交易日数量: %d", len(dates))

    fetch_fund_basic(pro)
    failed = fetch_fund_daily(pro, dates)

    if failed:
        # 失败日期写入日志，便于补跑
        with open(os.path.join(DATA_DIR, "_failed_dates.txt"), "w") as f:
            f.write("\n".join(failed))
        sys.exit(1)


if __name__ == "__main__":
    main()

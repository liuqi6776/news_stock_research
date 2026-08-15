#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
场外基金（开放式基金）净值全量采集工具
=======================================
存储到 D:\\iquant_data\\data_v2\\fund2

数据源:
  - 基金列表:  akshare fund_name_em (27449 只, 含类型)
  - 历史净值:  efinance fund.get_quote_history (日期/单位净值/累计净值/涨跌幅, 一次全量)

存储结构:
  fund2/
  ├── fund_basic_O.parquet        # 场外基金基础信息(代码/名称/类型)
  ├── nav/{code}.parquet          # 每只基金一个文件: 日期,单位净值,累计净值,涨跌幅
  ├── _progress.json              # 断点续传进度
  └── _failed.txt                 # 抓取失败清单

用法:
  python fetch_fund2.py                     # 全量(默认跳过货币基金)
  python fetch_fund2.py --limit 20          # 只抓前20只(测试)
  python fetch_fund2.py --workers 8         # 并发数(默认8)
  python fetch_fund2.py --no-skip-money     # 不跳过货币基金
  python fetch_fund2.py --start 20200101    # 只保留该日期之后(按净值日期过滤)

注意:
  - 运行环境: base python C:\\Users\\liuqi\\anaconda3\\python.exe
  - 货币基金单位净值恒为1(万份收益口径), 默认跳过, 无量化回测意义.
"""

import os
import json
import time
import argparse
import logging
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import requests
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("fetch_fund2")

DATA_DIR = r"D:\iquant_data\data_v2\fund2"
NAV_DIR = os.path.join(DATA_DIR, "nav")
BASIC_PATH = os.path.join(DATA_DIR, "fund_basic_O.parquet")
PROGRESS_PATH = os.path.join(DATA_DIR, "_progress.json")
FAILED_PATH = os.path.join(DATA_DIR, "_failed.txt")

# 每只基金最大重试次数
MAX_RETRY = 3
RETRY_SLEEP = 3

_print_lock = threading.Lock()
_done_count = 0

# ---------------------------------------------------------------------------
# 天天基金移动接口（efinance 底层同款, 但加超时防止并发挂起）
# ---------------------------------------------------------------------------
FUND_API_URL = "https://fundmobapi.eastmoney.com/FundMNewApi/FundMNHisNetList"
FUND_HEADERS = {
    "User-Agent": "EMProjJijin/6.2.8 (iPhone; iOS 13.6; Scale/2.00)",
    "GTOKEN": "98B423068C1F4DEF9842F82ADF08C5db",
    "clientInfo": "ttjj-iPhone10,1-iOS-iOS13.6",
    "Content-Type": "application/x-www-form-urlencoded",
    "Host": "fundmobapi.eastmoney.com",
    "Referer": "https://mpservice.com/516939c37bdb4ba2b1138c50cf69a2e1/release/pages/FundHistoryNetWorth",
}
FUND_REQ_DATA = {
    "IsShareNet": "true", "MobileKey": "1", "appType": "ttjj",
    "appVersion": "6.2.8", "cToken": "1", "deviceid": "1",
    "pageIndex": "1", "pageSize": "40000", "plat": "Iphone",
    "product": "EFund", "serverVersion": "6.2.8", "uToken": "1",
    "userId": "1", "version": "6.2.8",
}
_HTTP_TIMEOUT = (5, 20)  # (连接超时, 读取超时)
# 不用共享 Session: 长跑中连接池坏连接复用会导致请求挂起, 每请求新建连接最稳

# 限频保护: 每请求成功后小睡 + 连续失败全局退避
_REQUEST_SLEEP = 0.15
_CONSECUTIVE_FAIL_LIMIT = 20
_CONSECUTIVE_FAIL_BACKOFF = 30
_consecutive_fail = 0
_fail_lock = threading.Lock()


def load_fund_list(force=False):
    """获取场外基金列表, 缓存到 fund_basic_O.parquet"""
    if os.path.exists(BASIC_PATH) and not force:
        logger.info("从缓存加载基金列表: %s", BASIC_PATH)
        return pd.read_parquet(BASIC_PATH)
    import akshare as ak
    df = ak.fund_name_em()
    df.columns = ["code", "pinyin", "name", "fund_type", "pinyin_full"]
    df.to_parquet(BASIC_PATH, index=False)
    logger.info("基金列表已保存: %d 只 -> %s", len(df), BASIC_PATH)
    return df


def load_progress():
    if os.path.exists(PROGRESS_PATH):
        with open(PROGRESS_PATH, "r") as f:
            return json.load(f)
    return {"done": {}, "failed": {}}


def save_progress(progress):
    """原子写进度: 先写临时文件再替换, 避免并发读取读到半写内容"""
    tmp = PROGRESS_PATH + ".tmp"
    with open(tmp, "w") as f:
        json.dump(progress, f)
    os.replace(tmp, PROGRESS_PATH)


def fetch_one(code, start=None, retries=MAX_RETRY):
    """抓取单只基金历史净值, 返回标准 DataFrame 或 None"""
    global _consecutive_fail
    data = dict(FUND_REQ_DATA, FCODE=code)
    for attempt in range(1, retries + 1):
        try:
            resp = requests.post(FUND_API_URL, headers=FUND_HEADERS,
                                 data=data, timeout=_HTTP_TIMEOUT, verify=False)
            js = resp.json()
            rows = []
            for st in (js.get("Datas") or []):
                rows.append({
                    "date": st.get("FSRQ"),
                    "unit_nav": st.get("DWJZ"),
                    "acc_nav": st.get("LJJZ"),
                    "pct_chg": st.get("JZZZL"),
                })
            if not rows:
                return None
            df = pd.DataFrame(rows)
            df["date"] = pd.to_datetime(df["date"])
            for col in ("unit_nav", "acc_nav", "pct_chg"):
                df[col] = pd.to_numeric(df[col], errors="coerce")
            df = df.sort_values("date").reset_index(drop=True)
            if start:
                df = df[df["date"] >= pd.to_datetime(start)]
            with _fail_lock:
                _consecutive_fail = 0
            return df
        except Exception as e:
            if attempt >= retries:
                with _fail_lock:
                    _consecutive_fail += 1
                    n = _consecutive_fail
                if n >= _CONSECUTIVE_FAIL_LIMIT:
                    logger.warning("连续 %d 次失败, 暂停 %ds 后继续", n, _CONSECUTIVE_FAIL_BACKOFF)
                    time.sleep(_CONSECUTIVE_FAIL_BACKOFF)
                    with _fail_lock:
                        _consecutive_fail = 0
                logger.warning("基金 %s 抓取失败: %s", code, str(e)[:100])
                return None
            time.sleep(RETRY_SLEEP * attempt)
    return None


def process_fund(code, start, progress, skip_money):
    """处理单只基金: 抓取 + 保存 + 更新进度"""
    global _done_count
    if code in progress["done"] or code in progress["failed"]:
        return 0
    df = fetch_one(code, start)
    if df is None or df.empty:
        with _print_lock:
            progress["failed"][code] = "no_data"
        return 0
    # 货币基金检测: 净值全为1
    if skip_money and (df["unit_nav"] == 1.0).mean() > 0.95 and df["unit_nav"].nunique() <= 2:
        with _print_lock:
            progress["failed"][code] = "money_fund"
        return 0
    df.to_parquet(os.path.join(NAV_DIR, f"{code}.parquet"), index=False)
    time.sleep(_REQUEST_SLEEP)  # 限速, 降低被限频概率
    with _print_lock:
        progress["done"][code] = int(time.time())
        _done_count += 1
        if _done_count % 50 == 0:
            logger.info("已抓取 %d 只", _done_count)
    return 1


def main():
    ap = argparse.ArgumentParser(description="场外基金净值采集")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=None, help="只抓前 N 只(测试用)")
    ap.add_argument("--start", default=None, help="仅保留该日期(YYYYMMDD)之后的净值")
    ap.add_argument("--skip-money", action="store_true", default=True, help="跳过货币基金(默认)")
    ap.add_argument("--no-skip-money", dest="skip_money", action="store_false")
    ap.add_argument("--force-list", action="store_true", help="强制重新抓取基金列表")
    args = ap.parse_args()

    os.makedirs(NAV_DIR, exist_ok=True)
    basic = load_fund_list(force=args.force_list)
    logger.info("基金列表: %d 只", len(basic))

    progress = load_progress()
    todo = basic[~basic["code"].isin(set(progress["done"]) | set(progress["failed"]))].copy()
    logger.info("已完成 %d, 失败 %d, 待抓取 %d",
                len(progress["done"]), len(progress["failed"]), len(todo))

    if args.limit:
        todo = todo.head(args.limit)
        logger.info("测试模式: 只抓 %d 只", len(todo))

    if todo.empty:
        logger.info("没有待抓取基金, 全部完成")
        return

    t0 = time.time()
    codes = todo["code"].tolist()
    ok = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(process_fund, c, args.start, progress, args.skip_money): c for c in codes}
        for i, fut in enumerate(as_completed(futs), 1):
            ok += fut.result()
            if i % 200 == 0:
                save_progress(progress)
                elapsed = time.time() - t0
                speed = ok / elapsed
                remain = (len(codes) - i) / speed if speed > 0 else 0
                logger.info("进度 %d/%d, 成功 %d, 耗时 %.0fs, 预计剩余 %.0fs",
                            i, len(codes), ok, elapsed, remain)

    save_progress(progress)
    with open(FAILED_PATH, "w") as f:
        f.write("\n".join(f"{c}:{r}" for c, r in sorted(progress["failed"].items())))
    logger.info("全部完成: 成功 %d, 失败 %d, 总耗时 %.0fs",
                len(progress["done"]), len(progress["failed"]), time.time() - t0)


if __name__ == "__main__":
    main()

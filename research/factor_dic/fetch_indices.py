# -*- coding: utf-8 -*-
"""抓取大小盘/基准ETF指数日线(000300/000905/932000/000016 + 512100/510300 ETF), 供市场状态研究使用
保存到 research/chip_momentum/data/index_daily/ (与 000852.SH 同目录同 schema)"""
import os
import sys
import time

import pandas as pd
import tushare as ts

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config.settings import settings

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "chip_momentum", "data", "index_daily")
START, END = "20180101", "20260806"

INDICES = {"000300.SH": "沪深300", "000905.SH": "中证500", "000016.SH": "上证50", "932000.CSI": "中证2000", "000852.SH": "中证1000"}
ETFS = {"512100.SH": "中证1000ETF", "510300.SH": "沪深300ETF"}


def norm(df):
    df = df.copy()
    df["trade_date"] = df["trade_date"].astype(str)
    return df.sort_values("trade_date", ascending=False).reset_index(drop=True)


def main():
    pro = ts.pro_api(settings.TUSHARE_TOKEN)
    for code, name in INDICES.items():
        fp = os.path.join(OUT_DIR, f"{code}.parquet")
        try:
            df = pro.index_daily(ts_code=code, start_date=START, end_date=END)
        except Exception as e:
            print(f"[fail] {code} {name}: {e}")
            continue
        if df is None or df.empty:
            print(f"[empty] {code} {name}")
            continue
        df = norm(df)
        df.to_parquet(fp)
        print(f"[ok] {code} {name} {len(df)} 行 -> {fp}")
        time.sleep(0.4)
    for code, name in ETFS.items():
        fp = os.path.join(OUT_DIR, f"{code}.parquet")
        try:
            df = pro.fund_daily(ts_code=code, start_date=START, end_date=END)
        except Exception as e:
            print(f"[fail] {code} {name}: {e}")
            continue
        if df is None or df.empty:
            print(f"[empty] {code} {name}")
            continue
        df = norm(df)
        df.to_parquet(fp)
        print(f"[ok] {code} {name} {len(df)} 行 -> {fp}")
        time.sleep(0.4)


if __name__ == "__main__":
    main()

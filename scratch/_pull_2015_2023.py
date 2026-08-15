# -*- coding: utf-8 -*-
"""用 tushare 补齐 2015-2023 数据（断点续传）:
1. moneyflow:        2015-2019 每日资金流 -> moneyflow1/YYYYMMDD.parquet
2. fina_indicator_vip: 2015-2022 季度财务 -> fundamental1/fina_indicator_cache.parquet

用法:
  python _pull_2015_2023.py moneyflow
  python _pull_2015_2023.py fina
  python _pull_2015_2023.py all
"""
import os, sys, time, glob
import pandas as pd
import tushare as ts

TOKEN = "421ff94dd31be789aa7f95e61ad6fad5bcefa250a0b2c4d298224aa5"
ts.set_token(TOKEN)
pro = ts.pro_api()

DATA = r"D:/iquant_data/data_v2"
MF_DIR = os.path.join(DATA, "moneyflow1")
FIN_PATH = os.path.join(DATA, "fundamental1", "fina_indicator_cache.parquet")

FIN_COLS = ["ts_code", "ann_date", "end_date", "roe", "roe_dt", "or_yoy", "netprofit_yoy",
            "netprofit_margin", "grossprofit_margin", "eps", "dt_eps", "current_ratio",
            "quick_ratio", "debt_to_assets"]


def pull_moneyflow():
    existing = set()
    for f in os.listdir(MF_DIR):
        if f.endswith(".parquet") and os.path.getsize(os.path.join(MF_DIR, f)) > 1024:
            existing.add(f[:8])
    trade_days = sorted(os.path.splitext(os.path.basename(f))[0]
                        for f in glob.glob(os.path.join(DATA, "data_day1", "*.parquet")))
    trade_days = [d for d in trade_days if d < "20200101"]
    todo = [d for d in trade_days if d not in existing]
    print(f"资金流: 交易日 {len(trade_days)}, 已存在 {len(trade_days)-len(todo)}, 需拉取 {len(todo)}", flush=True)
    fail = 0
    for i, d in enumerate(todo):
        ok = False
        for attempt in range(5):
            try:
                df = pro.moneyflow(trade_date=d)
                if df is not None and len(df):
                    df.to_parquet(os.path.join(MF_DIR, f"{d}.parquet"), index=False)
                    ok = True
                else:
                    time.sleep(1.5)
                break
            except Exception as e:
                time.sleep(2 ** attempt)
        if not ok:
            fail += 1
            print(f"  [FAIL] {d} 连续5次失败", flush=True)
        if (i + 1) % 100 == 0:
            print(f"  进度 {i+1}/{len(todo)} (失败{fail})", flush=True)
        time.sleep(0.15)
    print(f"资金流完成: 失败 {fail} 个", flush=True)


def pull_fina():
    fin_old = pd.read_parquet(FIN_PATH)
    covered = set(fin_old["end_date"].astype(str).str[:8])
    periods = []
    for y in range(2015, 2023):
        for md in ["0331", "0630", "0930", "1231"]:
            p = f"{y}{md}"
            if p not in covered:
                periods.append(p)
    print(f"财务: 已覆盖 {len(covered)} 期, 需拉取 {len(periods)} 期", flush=True)
    parts = [fin_old]
    for i, p in enumerate(periods):
        df = None
        for attempt in range(5):
            try:
                df = pro.fina_indicator_vip(period=p)
                if df is not None and len(df):
                    break
                df = None
                time.sleep(1.5)
            except Exception as e:
                time.sleep(2 ** attempt)
        if df is None or not len(df):
            print(f"  [FAIL] period={p} 连续5次失败", flush=True)
            continue
        df = df[[c for c in FIN_COLS if c in df.columns]].copy()
        parts.append(df)
        print(f"  进度 {i+1}/{len(periods)}: {p} -> {len(df)} 行", flush=True)
        time.sleep(0.3)
    merged = pd.concat(parts, ignore_index=True)
    merged = merged.drop_duplicates(subset=["ts_code", "end_date"], keep="last")
    merged = merged.sort_values(["ts_code", "ann_date"]).reset_index(drop=True)
    merged.to_parquet(FIN_PATH, index=False)
    print(f"财务完成: 合并后 {len(merged):,} 行 -> {FIN_PATH}", flush=True)


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "all"
    if mode in ("moneyflow", "all"):
        pull_moneyflow()
    if mode in ("fina", "all"):
        pull_fina()

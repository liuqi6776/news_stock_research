# -*- coding: utf-8 -*-
"""直接抽查: 标签=1且未止盈的票, 对比 T日基准价 vs 买入价 vs 未来最高价"""
import os, glob
import numpy as np
import pandas as pd

ROOT = r"c:\Users\liuqi\quant_system_v2"
DATA = r"D:/iquant_data/data_v2"
OUT_DIR = os.path.join(ROOT, "research", "sector_rotation", "results")

df_pred = pd.read_csv(os.path.join(OUT_DIR, "universe_safe_hit30_oos_pred.csv"))
df_pred["trade_date_dt"] = pd.to_datetime(df_pred["trade_date"].astype(str), format="%Y%m%d")
codes_need = set(df_pred["ts_code"].unique())

px_files = sorted(glob.glob(os.path.join(DATA, "data_day1", "*.parquet")))
parts = []
for f in px_files:
    if os.path.getsize(f) <= 1024: continue
    d = os.path.basename(f)[:8]
    if d < "20221201": continue
    df = pd.read_parquet(f, columns=["ts_code", "trade_date", "close"])
    df = df[df["ts_code"].isin(codes_need)]
    if len(df): parts.append(df)
px = pd.concat(parts, ignore_index=True)
px["trade_date"] = pd.to_datetime(px["trade_date"].astype(str), format="%Y%m%d")
px = px.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)
CL_MAP = {}
for code, gdf in px.groupby("ts_code"):
    CL_MAP[code] = (gdf["trade_date"].tolist(), gdf["close"].values)
ALL_DAYS = sorted(px["trade_date"].unique())
DAY2I = {d:i for i,d in enumerate(ALL_DAYS)}

def get_close(code, dt):
    if code not in CL_MAP: return None
    dates, closes = CL_MAP[code]
    lo, hi = 0, len(dates)-1
    while lo <= hi:
        m = (lo + hi) // 2
        if dates[m] == dt: return closes[m]
        if dates[m] < dt: lo = m + 1
        else: hi = m - 1
    return None

# 抽样: 标签=1, prob最高的几只票
df_pred["rank"] = df_pred.groupby("trade_date")["prob"].rank(ascending=False)
samples = df_pred[(df_pred["rank"] <= 20) & (df_pred["safe_hit30"] == 1)].head(8)
print(f"{'trade_date':<10} {'ts_code':<12} {'prob':<6} {'T_close':<10} {'T+1_close':<10} {'偏移%':<8} {'标签fwd_max':<10} {'买后180d实际max':<10}")
for _, r in samples.iterrows():
    m_int = r["trade_date"]; code = r["ts_code"]
    m_dt = pd.to_datetime(str(int(m_int)), format="%Y%m%d")
    t_close = get_close(code, m_dt)
    buy_day = None
    for td in ALL_DAYS:
        if td >= m_dt: buy_day = td; break
    bp = get_close(code, buy_day)
    # 买入后180天 max
    dates, closes = CL_MAP[code]
    seg_dates = [d for d in dates if buy_day <= d <= buy_day + pd.Timedelta(days=260)]
    seg_closes = np.array([get_close(code, d) for d in seg_dates], dtype=float)
    if t_close is None or bp is None or bp <= 0 or len(seg_closes) < 10:
        continue
    offset = bp / t_close - 1
    max_ret = seg_closes.max() / bp - 1
    print(f"{int(m_int):<10} {code:<12} {r['prob']:<6.3f} {t_close:<10.2f} {bp:<10.2f} "
          f"{offset:<8.1%} {r['fwd100_maxret']:<10.1%} {max_ret:<10.1%}")

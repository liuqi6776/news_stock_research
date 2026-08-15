# -*- coding: utf-8 -*-
"""
诊断回测: 为什么标签77%命中率, 回测只有35%止盈?
逐笔买入记录: 标签safe_hit30 vs 实际持仓期最大涨幅 vs 结果(止盈/时间止损)
定位: 买入价偏移 / 时间窗口 / 数据缺失
"""
import os, glob, time
import numpy as np
import pandas as pd

ROOT = r"c:\Users\liuqi\quant_system_v2"
DATA = r"D:/iquant_data/data_v2"
OUT_DIR = os.path.join(ROOT, "research", "sector_rotation", "results")
t0 = time.time()

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

# ---- 全局Top20 (每行业≤3) 每月候选 ----
TOP_GLOBAL, MAXK = 20, 3
RB_PICKS = {}
for m_int in sorted(df_pred["trade_date"].unique()):
    m_dt = pd.to_datetime(str(m_int), format="%Y%m%d")
    subset = df_pred[df_pred["trade_date"] == m_int].sort_values("prob", ascending=False)
    picks = subset.groupby("industry", sort=False).head(MAXK).head(TOP_GLOBAL)
    pick_list = [(r["ts_code"], r["industry"], r["prob"]) for _, r in picks.iterrows()]
    for td in ALL_DAYS:
        if td >= m_dt:
            RB_PICKS[td] = pick_list
            break

# ---- 诊断: 对每笔买入, 计算实际持仓期表现 ----
rows = []
for m_int in sorted(df_pred["trade_date"].unique()):
    m_dt = pd.to_datetime(str(m_int), format="%Y%m%d")
    subset = df_pred[df_pred["trade_date"] == m_int].sort_values("prob", ascending=False)
    picks = subset.groupby("industry", sort=False).head(MAXK).head(TOP_GLOBAL)
    # 找买入日
    buy_day = None
    for td in ALL_DAYS:
        if td >= m_dt:
            buy_day = td; break
    if buy_day is None: continue
    bi = DAY2I[buy_day]
    for _, r in picks.iterrows():
        code = r["ts_code"]; ind = r["industry"]
        bp = get_close(code, buy_day)
        if bp is None or bp <= 0: continue
        # 未来180天内的最大/最小相对买入价的收益
        dates, closes = CL_MAP[code]
        seg_dates = [d for d in dates if buy_day <= d <= buy_day + pd.Timedelta(days=260)]
        if len(seg_dates) < 10: continue
        seg_closes = np.array([get_close(code, d) for d in seg_dates], dtype=float)
        rets = seg_closes / bp - 1
        max_ret = rets.max(); min_ret = rets.min()
        days_to_max = int(np.argmax(rets))
        days_to_tp = np.argmax(rets >= 0.30) if (rets >= 0.30).any() else np.nan
        rows.append({
            "trade_date": m_int, "ts_code": code, "industry": ind,
            "prob": r["prob"], "label_safe_hit30": r["safe_hit30"],
            "buy_price": bp, "max_ret": max_ret, "min_ret": min_ret,
            "days_to_tp": days_to_tp, "hit_tp_180d": 0 if np.isnan(days_to_tp) else 1,
        })

diag = pd.DataFrame(rows)
print(f"诊断买入记录: {len(diag)}笔")

# 交叉表: 标签 vs 实际180天止盈
print("\n=== 标签safe_hit30 vs 实际180天内涨30% ===")
if len(diag) and diag["label_safe_hit30"].notna().any():
    ctab = pd.crosstab(diag["label_safe_hit30"].astype(int),
                       diag["hit_tp_180d"], margins=True)
    print(ctab)
    hit_lab = diag[diag["label_safe_hit30"] == 1]
    miss_lab = diag[diag["label_safe_hit30"] == 0]
    if len(hit_lab):
        print(f"\n标签=1的票: {len(hit_lab)}笔, 其中实际180天止盈 {hit_lab['hit_tp_180d'].mean():.0%}")
        print(f"  标签=1但未止盈的: 平均max_ret={hit_lab[hit_lab.hit_tp_180d==0]['max_ret'].mean():.1%}, "
              f"平均min_ret={hit_lab[hit_lab.hit_tp_180d==0]['min_ret'].mean():.1%}")
    if len(miss_lab):
        print(f"\n标签=0的票: {len(miss_lab)}笔, 其中实际180天止盈 {miss_lab['hit_tp_180d'].mean():.0%}")

# 买入价 vs 标签基准价偏移
print("\n=== 买入价 vs 标签基准价 ===")
diag2 = diag.copy()
# 标签基准价 = 调仓日收盘 (df_pred里的fwd100_maxret是按调仓日收盘算的)
print(f"  平均最大涨幅: {diag['max_ret'].mean():.1%} | 平均最大跌幅: {diag['min_ret'].mean():.1%}")

# 时间分布: days_to_tp
valid_tp = diag[diag["hit_tp_180d"] == 1]
print(f"\n=== 实际止盈所需天数分布 (共{len(valid_tp)}笔) ===")
if len(valid_tp):
    print(valid_tp["days_to_tp"].describe().round(1))
    print(f"  100天内止盈: {(valid_tp['days_to_tp']<=100).mean():.0%}")

# 保存诊断
diag.to_csv(os.path.join(OUT_DIR, "universe_diag_per_trade.csv"), index=False, encoding="utf-8-sig")
print(f"\n耗时 {time.time()-t0:.0f}s")

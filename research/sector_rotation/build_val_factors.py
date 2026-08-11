# -*- coding: utf-8 -*-
"""构建估值正交因子并 merge 到大面板 (stock_ml_panel_ortho_72m.parquet)

数据源: other_day1  (完整历史 2015-2026, 无前瞻偏差)
  - pe, pb           估值
  - circ_mv          流通市值
  - turnover_rate    换手率
  - volume_ratio     量比

新增因子 (横截面标准化):
  - pe_rank, pb_rank           估值截面分位 (低PE=便宜)
  - pe_pct_3y, pb_pct_3y       3年分位 (自身历史位置)
  - ln_circ_mv                 市值对数
  - pe_ep_rank                 EP率分位 (1/PE)
  - turnover_rate_rank         换手率截面分位
  - volume_ratio_rank          量比截面分位
"""
import glob, os, time, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

t0 = time.time()
ROOT = r"c:\Users\liuqi\quant_system_v2"
DATA = r"D:/iquant_data/data_v2"
IN = os.path.join(ROOT, "research", "sector_rotation", "stock_ml_panel_ortho_72m.parquet")
OUT = os.path.join(ROOT, "research", "sector_rotation", "stock_ml_panel_ortho2_72m.parquet")

panel = pd.read_parquet(IN)
print(f"[面板] {len(panel):,} 行, {panel['trade_date'].nunique()} 月末", flush=True)

# ---- other_day1 估值 ----
fs = sorted(glob.glob(os.path.join(DATA, "other_day1", "*.parquet")))
parts = []
for f in fs:
    if os.path.getsize(f) <= 1024: continue
    parts.append(pd.read_parquet(f, columns=["ts_code","trade_date","close","turnover_rate",
                                              "volume_ratio","pe","pb","circ_mv"]))
od = pd.concat(parts, ignore_index=True)
od["trade_date"] = od["trade_date"].astype(int)
od = od.replace([np.inf, -np.inf], np.nan)
# 月末快照: 每(股票,月)取最后一天
od["month"] = od["trade_date"] // 100
od = od.sort_values(["ts_code","trade_date"]).drop_duplicates(subset=["ts_code","month"], keep="last")
od = od.drop(columns=["month","close"])
print(f"[估值源] {len(od):,} 行, {od['trade_date'].min()}~{od['trade_date'].max()}", flush=True)

# 基础因子
od["pe_ep"] = 1.0 / od["pe"]
od["ln_circ_mv"] = np.log(od["circ_mv"] + 1e-6)

# 截面分位 (月末横截面)
def _rank_pct(s):
    return s.rank(pct=True)
gg = od.groupby("trade_date")
od["pe_rank"]     = gg["pe_ep"].transform(_rank_pct)   # 高EP=低PE=便宜
od["pb_rank"]     = gg["pb"].transform(lambda s: 1 - s.rank(pct=True))  # 低PB=便宜
od["ln_mv_rank"]  = gg["ln_circ_mv"].transform(_rank_pct)
od["turn_rank"]   = gg["turnover_rate"].transform(_rank_pct)
od["volratio_rank"] = gg["volume_ratio"].transform(_rank_pct)

# 3年自身分位 (月度数据: 36个月窗口, min_periods=12)
od = od.sort_values(["ts_code","trade_date"])
sg = od.groupby("ts_code")
od["pe_pct_3y"] = sg["pe"].transform(lambda s: s.rolling(36, min_periods=12).rank(pct=True))
od["pb_pct_3y"] = sg["pb"].transform(lambda s: s.rolling(36, min_periods=12).rank(pct=True))
od["turn_pct_3y"] = sg["turnover_rate"].transform(lambda s: s.rolling(36, min_periods=12).rank(pct=True))

VAL_FEATS = ["pe_ep","ln_circ_mv","pe_rank","pb_rank","ln_mv_rank",
             "turn_rank","volratio_rank","pe_pct_3y","pb_pct_3y","turn_pct_3y"]
od_keep = od[["ts_code","trade_date"] + VAL_FEATS]
print(f"[估值因子] {len(VAL_FEATS)} 个: {VAL_FEATS}, {time.time()-t0:.0f}s", flush=True)

# ---- merge ----
panel = panel.merge(od_keep, on=["ts_code","trade_date"], how="left")
for f in VAL_FEATS:
    print(f"  {f:>14} cov={panel[f].notna().mean():.0%}", flush=True)

panel.to_parquet(OUT, index=False)
print(f"[保存] {OUT}, {len(panel):,} 行, {time.time()-t0:.0f}s")

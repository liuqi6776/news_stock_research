# -*- coding: utf-8 -*-
"""
STEP1: 将 stock_ml_panel_72m.parquet 改造成
       Target = fwd_100_hit30 (未来100个交易日内是否有一次累计>=30%涨幅)
       Feature = C8因子 + is_undervalued_sector (行业PE分位<30%标记)

输出: research/sector_rotation/stock_ml_panel_hit30.parquet
"""
import os
import glob
import time
import numpy as np
import pandas as pd

ROOT = r"c:\Users\liuqi\quant_system_v2"
DATA = r"D:/iquant_data/data_v2"
OLD_PANEL = os.path.join(ROOT, "research", "sector_rotation", "stock_ml_panel_72m.parquet")
PE_CSV = os.path.join(ROOT, "research", "sector_rotation", "results", "industry_pe.csv")
INDUSTRY_PARQ = os.path.join(DATA, "industry1", "industry.parquet")
OUT = os.path.join(ROOT, "research", "sector_rotation", "stock_ml_panel_hit30.parquet")

t0 = time.time()
# ---------- 1. 读旧ML面板 ----------
panel = pd.read_parquet(OLD_PANEL)
print(f"[1] 旧ML面板: {len(panel):,}行, 调仓日{panel['trade_date'].nunique()}月, 股票{panel['ts_code'].nunique()}只")
print(f"    时间范围: {panel['trade_date'].min()}~{panel['trade_date'].max()}")
print(f"    列: {list(panel.columns)}")

# ---------- 2. 行业名映射对齐 (旧panel里的industry列 映射到 industry_pe.csv 的列名) ----------
pe_df = pd.read_csv(PE_CSV, index_col=0); pe_df.index = pe_df.index.astype(str)
pe_pct = pe_df.rolling(60, min_periods=12).rank(pct=True)
print(f"[2] industry_pe.csv: {pe_df.shape[0]}月 × {pe_df.shape[1]}个行业")

# 旧panel里industry列的取值 → 匹配pe列名
inds_in_panel = sorted(panel['industry'].dropna().unique())
inds_in_pe = list(pe_df.columns)
overlap = [i for i in inds_in_panel if i in set(inds_in_pe)]
print(f"    panel有industry={len(inds_in_panel)}种, PE有行业{len(inds_in_pe)}种, 交集={len(overlap)}")
# 未匹配的用近似关键字找找 (银行/白酒/证券/医药/电子 这类大类)
unmatched = [i for i in inds_in_panel if i not in set(inds_in_pe)]
if unmatched:
    print(f"    未匹配{len(unmatched)}: {unmatched[:20]}")

# ---------- 3. 低估板块标记 is_undervalued_sector + pe_pct_rank ----------
# pe_pct.index是月末YYYYMMDD(int): 如20150130
panel['dt'] = pd.to_datetime(panel['trade_date'].astype(str), format='%Y%m%d')
# 把行业pe_pct拉平成 (yyyymm, industry) -> 分位值 (yyyymm=202001)
records = []
for ym_int in pe_pct.index:
    ym_str = str(int(ym_int))
    yyyymm = int(ym_str[:4] + ym_str[4:6])  # 202001
    for col in pe_pct.columns:
        records.append((yyyymm, col, pe_pct.loc[ym_int, col]))
pe_map_df = pd.DataFrame(records, columns=['yyyymm', 'industry', 'pe_pct_val'])
panel['yyyymm'] = panel['dt'].dt.year * 100 + panel['dt'].dt.month
panel = panel.merge(pe_map_df, on=['yyyymm', 'industry'], how='left')
panel['is_undervalued_sector'] = (panel['pe_pct_val'] < 0.30).astype(float)
# 非稳定板块但低估的也保留;额外有"稳定板块池"标记(白酒/银行/消费等)
STABLE_KW = ['白酒','银行','证券','保险','医药','医疗','中药','食品','饮料','消费','家电','汽车',
             '煤炭','钢铁','有色','电力','石化','石油','化工','建筑','地产','通信','电子','半导体']
panel['is_stable_sector'] = panel['industry'].fillna('').apply(
    lambda s: 1.0 if any(k in s for k in STABLE_KW) else 0.0)
panel['is_undervalued_and_stable'] = ((panel['is_undervalued_sector']==1)&(panel['is_stable_sector']==1)).astype(float)
print(f"[3] 低估板块标记完成")
print(f"    低估标记=1占比: {panel['is_undervalued_sector'].mean():.1%}")
print(f"    稳定板块=1占比: {panel['is_stable_sector'].mean():.1%}")
print(f"    低估+稳定=1占比: {panel['is_undervalued_and_stable'].mean():.1%}")

# ---------- 4. 构建 Target: fwd_100_hit30 ----------
# 用data_day1/*.parquet 每只股票日频收盘, 对每个调仓日T, 看T+1~T+100交易日内最高收盘 / T日收盘 - 1 >= 30%
px_files = sorted(glob.glob(os.path.join(DATA, "data_day1", "*.parquet")))
print(f"[4] 读取日频收盘构造hit30标签, 行情文件={len(px_files)}个")
# 只取panel里出现过的股票
codes_need = set(panel['ts_code'].unique())
print(f"    需要股票数={len(codes_need)}")
parts = []
for f in px_files:
    if os.path.getsize(f) <= 1024: continue
    d = os.path.basename(f)[:8]
    if d < "20190601": continue
    df = pd.read_parquet(f, columns=["ts_code", "trade_date", "close"])
    df = df[df["ts_code"].isin(codes_need)]
    if len(df):
        parts.append(df)
px = pd.concat(parts, ignore_index=True)
px["trade_date"] = pd.to_datetime(px["trade_date"].astype(str), format="%Y%m%d")
px = px.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)
print(f"    行情={len(px):,}行, {px['ts_code'].nunique()}只, {px['trade_date'].min().date()}~{px['trade_date'].max().date()}")

# 对每只股票, rolling forward 100天内最高收盘 / T日收盘 - 1
# 向量化: rolling(100).max().shift(-100) = 未来i+1..i+100的最大值 (过去窗口shift到未来)
print(f"    向量化计算fwd100_maxret (rolling100.max + shift(-100))...")
px = px.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)
g = px.groupby("ts_code", sort=False)
# count >= 10 表示未来至少10个交易日有数据
cnt = g["close"].transform(lambda s: s.rolling(100, min_periods=10).count().shift(-100))
fmax = g["close"].transform(lambda s: s.rolling(100, min_periods=10).max().shift(-100))
px["fwd100_maxret"] = np.where(cnt >= 10, fmax / px["close"] - 1, np.nan)
px["fwd_100_hit30"] = np.where(px["fwd100_maxret"] >= 0.30, 1.0, 0.0)
px.loc[px["fwd100_maxret"].isna(), "fwd_100_hit30"] = np.nan
hit_df = px[["ts_code","trade_date","fwd_100_hit30","fwd100_maxret"]].copy()
print(f"    完成日频hit30标签计算, hit率={hit_df['fwd_100_hit30'].mean():.1%}")

# merge到panel
panel["trade_date_dt"] = pd.to_datetime(panel["trade_date"].astype(str), format="%Y%m%d")
panel = panel.merge(hit_df[["ts_code","trade_date","fwd_100_hit30","fwd100_maxret"]],
                    left_on=["ts_code","trade_date_dt"],
                    right_on=["ts_code","trade_date"], how="left", suffixes=("","_px"))
print(f"[5] merge后 panel={len(panel):,}行, hit30非空={panel['fwd_100_hit30'].notna().sum():,} 占比={panel['fwd_100_hit30'].notna().mean():.1%}")
print(f"    fwd_100_hit30正样本率(非空里): {panel['fwd_100_hit30'].mean():.1%}")

# ---------- 5. 清理脏列 + 保存 ----------
drop_cols = [c for c in ['dt','yyyymm','trade_date_dt','trade_date_px','fwd_20'] if c in panel.columns]
panel = panel.drop(columns=drop_cols)
panel.to_parquet(OUT, index=False)
print(f"[6] 保存: {OUT}")
print(f"    耗时{time.time()-t0:.0f}s | 最终面板={panel.shape}")
print(f"    最终列: {list(panel.columns)}")

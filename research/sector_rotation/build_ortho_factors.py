# -*- coding: utf-8 -*-
"""构建正交信息源因子并 merge 到大面板

数据源:
  1. industry1/industry.parquet  — 行业分类 (静态)
  2. data_day1/*.parquet        — 日行情 (算行业动量/拥挤度)
  3. moneyflow1/*.parquet       — 个股大单资金流 (净流入/占比)
  4. ths_news1/*.parquet        — 个股新闻情绪 (利好new_gs / 利空new_bs)

新增因子 (~11 个):
  行业因子 (3):  ind_mom_20     行业20日动量
                 ind_mf_20      行业20日资金流占比
                 ind_crowd_20   行业20日拥挤度 (收益离散度)
  资金流因子 (4): net_mf_ratio_5/20  个股5/20日净流入/成交额
                  lg_net_ratio_5/20  大单+超大单 5/20日净流入/成交额
  新闻情绪 (4):  news_gs_bs_5/20     5/20日(利好-利空)
                  news_total_5/20    5/20日新闻总数

输出: stock_ml_panel_ortho_72m.parquet
"""
import os, glob, time, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

t0 = time.time()
ROOT = r"c:\Users\liuqi\quant_system_v2"
DATA = r"D:/iquant_data/data_v2"
PANEL_LARGE = os.path.join(ROOT, "research", "sector_rotation", "stock_ml_panel_large_72m.parquet")
OUT = os.path.join(ROOT, "research", "sector_rotation", "stock_ml_panel_ortho_72m.parquet")

# ---------- 1. 加载大面板 ----------
panel = pd.read_parquet(PANEL_LARGE)
month_last = sorted(panel["trade_date"].unique())
print(f"[大面板] {len(panel):,} 行, {len(month_last)} 月末快照, {time.time()-t0:.0f}s")

# ---------- 2. 行业分类 ----------
ind = pd.read_parquet(os.path.join(DATA, "industry1", "industry.parquet"))
ind_map = ind[["ts_code", "industry"]].drop_duplicates()
print(f"[行业] {ind_map['industry'].nunique()} 个行业, {len(ind_map)} 只, {time.time()-t0:.0f}s")

# ---------- 3. 日行情 → 行业动量/拥挤度 ----------
# 复用 build_large_factor_panel 的行情读取逻辑
px_files = sorted(glob.glob(os.path.join(DATA, "data_day1", "*.parquet")))
parts = []
for f in px_files:
    if os.path.getsize(f) <= 1024: continue
    d = os.path.basename(f)[:8]
    if d < "20190601": continue
    df = pd.read_parquet(f, columns=["ts_code","trade_date","pct_chg"])
    parts.append(df)
px = pd.concat(parts, ignore_index=True)
px["trade_date"] = px["trade_date"].astype(int)
px = px.merge(ind_map, on="ts_code", how="left")
px["r"] = px["pct_chg"] / 100.0
# 行业日收益 (等权)
ind_daily = px.groupby(["trade_date","industry"])["r"].agg(["mean","std"]).rename(
    columns={"mean":"ind_ret","std":"ind_std"}).reset_index()
# 行业20日动量
ind_daily = ind_daily.sort_values(["industry","trade_date"])
ind_daily["ind_mom_20"] = ind_daily.groupby("industry")["ind_ret"].transform(
    lambda s: (1+s).rolling(20).apply(np.prod, raw=True) - 1)
# 行业拥挤度: 20日成分收益离散度均值 (越大越拥挤)
ind_daily["ind_crowd_20"] = ind_daily.groupby("industry")["ind_std"].transform(
    lambda s: s.rolling(20).mean())
print(f"[行业日] {len(ind_daily):,} 行, {time.time()-t0:.0f}s")

# ---------- 4. 个股资金流 (moneyflow1) ----------
mf_files = sorted(glob.glob(os.path.join(DATA, "moneyflow1", "*.parquet")))
mf_parts = []
for f in mf_files:
    if os.path.getsize(f) <= 1024: continue
    mf_parts.append(pd.read_parquet(f))
mf = pd.concat(mf_parts, ignore_index=True)
mf["trade_date"] = mf["trade_date"].astype(int)
mf = mf[["trade_date","ts_code","buy_lg_amount","sell_lg_amount",
         "buy_elg_amount","sell_elg_amount","net_mf_amount"]]
# 成交额 = 大中小单买卖之和 (近似用 net_mf 口径的成交额)
mf["amount"] = (mf["buy_lg_amount"]+mf["sell_lg_amount"]
                + mf["buy_elg_amount"]+mf["sell_elg_amount"])
mf["lg_net"] = (mf["buy_lg_amount"]-mf["sell_lg_amount"]
                + mf["buy_elg_amount"]-mf["sell_elg_amount"])
mf = mf.sort_values(["ts_code","trade_date"])
g = mf.groupby("ts_code")
# 5/20日聚合
for w in (5, 20):
    mf[f"net_{w}"]  = g["net_mf_amount"].transform(lambda s,w=w: s.rolling(w).sum())
    mf[f"amt_{w}"]  = g["amount"].transform(lambda s,w=w: s.rolling(w).sum())
    mf[f"lgnet_{w}"] = g["lg_net"].transform(lambda s,w=w: s.rolling(w).sum())
    mf[f"net_mf_ratio_{w}"] = mf[f"net_{w}"] / (mf[f"amt_{w}"] + 1e-6)
    mf[f"lg_net_ratio_{w}"] = mf[f"lgnet_{w}"] / (mf[f"amt_{w}"] + 1e-6)

# 行业资金流: 行业内个股净流入/成交额之和 (在裁剪列之前算, 需要 net_mf_amount/amount)
mf_ind = mf.merge(ind_map, on="ts_code", how="left")
mf_ind_agg = mf_ind.groupby(["trade_date","industry"], as_index=False)[
    ["net_mf_amount","amount"]].sum()
mf_ind_agg = mf_ind_agg.sort_values(["industry","trade_date"])
for w in (5, 20):
    mf_ind_agg[f"ind_nm_{w}"] = mf_ind_agg.groupby("industry")["net_mf_amount"].transform(
        lambda s,w=w: s.rolling(w).sum())
    mf_ind_agg[f"ind_amt_{w}"] = mf_ind_agg.groupby("industry")["amount"].transform(
        lambda s,w=w: s.rolling(w).sum())
mf_ind_agg["ind_mf_20"] = mf_ind_agg["ind_nm_20"] / (mf_ind_agg["ind_amt_20"] + 1e-6)
mf_ind_keep = ["trade_date","industry","ind_mf_20"]
# 裁剪个股列 (在行业聚合算完之后)
mf_keep = [c for c in mf.columns if c in ("trade_date","ts_code") or
           c.startswith("net_mf_ratio_") or c.startswith("lg_net_ratio_")]
mf = mf[mf_keep].dropna(subset=[c for c in mf_keep if c.startswith(("net_mf","lg_net"))])
print(f"[资金流] {len(mf):,} 行个股, {time.time()-t0:.0f}s")

# ---------- 5. 新闻情绪 (ths_news1) ----------
news_files = sorted(glob.glob(os.path.join(DATA, "ths_news1", "*.parquet")))
news_parts = []
for f in news_files:
    if os.path.getsize(f) <= 1024: continue
    news_parts.append(pd.read_parquet(f, columns=["ts_code","datetime","new_gs","new_bs"]))
news = pd.concat(news_parts, ignore_index=True)
news["trade_date"] = news["datetime"].astype(int)
news["news_gs_bs"] = news["new_gs"] - news["new_bs"]
news["news_total"] = news["new_gs"] + news["new_bs"]
news = news.sort_values(["ts_code","trade_date"])
ng = news.groupby("ts_code")
for w in (5, 20):
    news[f"news_gs_bs_{w}"] = ng["news_gs_bs"].transform(lambda s,w=w: s.rolling(w).sum())
    news[f"news_total_{w}"] = ng["news_total"].transform(lambda s,w=w: s.rolling(w).sum())
news_keep = ["trade_date","ts_code"] + [c for c in news.columns if c.startswith("news_")]
news = news[news_keep].drop_duplicates(subset=["trade_date","ts_code"])
print(f"[新闻] {len(news):,} 行, {time.time()-t0:.0f}s")

# ---------- 6. Merge 到大面板 ----------
# 大面板已含 industry 列, 直接用它
panel = panel.merge(ind_daily[["trade_date","industry","ind_mom_20","ind_crowd_20"]],
                    on=["trade_date","industry"], how="left")
panel = panel.merge(mf_ind_agg[mf_ind_keep],
                    on=["trade_date","industry"], how="left")
panel = panel.merge(mf[["trade_date","ts_code"] + [c for c in mf.columns if c.startswith(("net_mf_ratio_","lg_net_ratio_"))]],
                    on=["trade_date","ts_code"], how="left")
panel = panel.merge(news[["trade_date","ts_code"] + [c for c in news.columns if c.startswith("news_")]],
                    on=["trade_date","ts_code"], how="left")

ORTHO_FEATS = ["ind_mom_20","ind_crowd_20","ind_mf_20",
               "net_mf_ratio_5","net_mf_ratio_20","lg_net_ratio_5","lg_net_ratio_20",
               "news_gs_bs_5","news_gs_bs_20","news_total_5","news_total_20"]
ORTHO_FEATS = [c for c in ORTHO_FEATS if c in panel.columns]
print(f"[正交因子] {len(ORTHO_FEATS)} 个: {ORTHO_FEATS}")

panel.to_parquet(OUT, index=False)
print(f"[保存] {OUT}, {len(panel):,} 行, {time.time()-t0:.0f}s")
# 打印正交因子覆盖率和单因子IC
for f in ORTHO_FEATS:
    cov = panel[f].notna().mean()
    ics = []
    for dt, gg in panel.groupby("trade_date"):
        if len(gg) < 50: continue
        ic = gg[f].rank().corr(gg["fwd_20"].rank())
        if np.isfinite(ic): ics.append(ic)
    s = pd.Series(ics)
    print(f"  {f:>18} cov={cov:.0%} IC={s.mean():+.4f} ICIR={s.mean()/(s.std(ddof=1)+1e-9)*np.sqrt(12):+.2f}")
print(f"[总耗时] {time.time()-t0:.0f}s")

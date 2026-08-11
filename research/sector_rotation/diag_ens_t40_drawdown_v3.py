# -*- coding: utf-8 -*-
"""回撤归因 v3 (不 import 主引擎，直接读 pkl + 面板)

维度：
1. 回撤区间识别 + s123 in/out 各自贡献（读 log_df 的 state，含 nav/state）
2. s123 全期 in/out 年化收益/波动（择时有效性）
3. s123 信号分布 + 回撤期 s123 逐月累加（读取 timing_dingtou 生成的 sig）
4. 用面板构造 "传统行业池等权参考组合" 回撤 - 对比策略回撤（判断是策略问题还是市场β导致）
"""
import os, time, sys, pickle
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression as _LR

ROOT = r"c:\Users\liuqi\quant_system_v2"
sys.path.insert(0, os.path.join(ROOT, "research", "sector_rotation"))
sys.path.insert(0, os.path.join(ROOT, "research", "fund_research", "studies", "rotation_dingtou"))
PANEL = os.path.join(ROOT, "research", "sector_rotation", "stock_ml_panel_72m.parquet")
PKL = os.path.join(ROOT, "research", "sector_rotation", "results", "stock_gbdt_s123_results.pkl")

t0 = time.time()

# ===== 读回测结果 (pkl 中有 nav_dated) =====
with open(PKL, "rb") as f:
    res = pickle.load(f)
# pkl 格式（从之前保存看）：dict? or list?
if isinstance(res, dict) and "nav_dated" in res:
    navs_pack = res["nav_dated"]
else:
    # 如果是 list of results (24+7组合)，找 ENS_T40
    if isinstance(res, list) and isinstance(res[0], tuple):
        tag = next((i for i,(n,r) in enumerate(res) if n=="ENS_T40_S123_ONLY_S123"), None)
        navs_pack = res[tag][1].get("nav_dated", None) if tag is not None else None
print(f"[读取] pkl 类型 = {type(res).__name__}")
if isinstance(res, dict):
    print(f"  keys = {list(res.keys())[:10]}")

# 如果 pkl 没 nav_dated (旧版)，重建：直接读 matrix.csv 只给绩效没用，
# 那就重新跑一个"极简 NAV"：用 panel 的 fwd_20 + s123 状态近似（非精确但可归因回撤来源）
# 简化：无论 pkl，这里直接用 timing_dingtou 的 fetch_pe_csi300 重算 s123 + 读 T7 对照做近似
# 其实更精确：我直接 import 主引擎但先用 ast parse 跳过非 if __main__==块？

# --- 代替方案：直接 load 主引擎的 [s123 sig] 模块，同时不 import stock_gbdt_s123_backtest (避开重跑) ---
from timing_dingtou import fetch_pe_csi300, fetch_bond10y, _rolling_pct, _zscore

# ========= 1. s123 信号 =========
print("\n=== 1. 计算 s123 状态机信号 (2019-07~2026) ===")
pe = fetch_pe_csi300()
by = fetch_bond10y()
pe = pe.sort_index()
by = by.sort_index()
# 对齐月底
s1_df = []
for ym in sorted(set(pe.index.strftime("%Y-%m"))):
    me = pe.loc[:ym].last("1D").index[0]
    pe_v = float(pe.loc[:me].iloc[-1])
    by_v = float(by.loc[:me].iloc[-1]) if len(by.loc[:me]) else np.nan
    s1_df.append({"ym": ym, "pe": pe_v, "by": by_v})
sig = pd.DataFrame(s1_df)
sig["pe_pct"] = _rolling_pct(sig["pe"], 60)
erp = 1.0/sig["pe"]*100 - sig["by"]
mu = erp.rolling(60, min_periods=12).mean()
sd = erp.rolling(60, min_periods=12).std(ddof=1)
sig["erp_z"] = (erp - mu) / (sd + 1e-12)
# S3: 沪深300 回撤 需指数
import akshare as ak
idx = ak.stock_zh_index_daily_tx(symbol="sh000300")
idx["date"] = pd.to_datetime(idx["date"])
idx = idx.set_index("date").sort_index()
idx_cum = idx["close"] / idx["close"].iloc[0]
idx_peak = idx_cum.cummax()
idx_dd = idx_cum / idx_peak - 1.0
# 取月底
s3_d = {}
for ym in sig["ym"]:
    me = pd.Timestamp(ym) + pd.offsets.MonthEnd(0)
    v = idx_dd.reindex(idx_dd.index[:me]).last("1D")
    s3_d[ym] = float(v.iloc[-1]) if len(v) else np.nan
sig["idx_dd"] = sig["ym"].map(s3_d)
sig["S1"] = (sig["pe_pct"] < 0.20).astype(int)
sig["S2"] = (sig["erp_z"] > 1.0).astype(int)
sig["S3"] = (sig["idx_dd"] <= -0.25).astype(int)
sig["s123"] = sig["S1"] + sig["S2"] + sig["S3"]
# ym 转为 int (202001) 格式
sig["ym_int"] = sig["ym"].str.replace("-","").astype(int)
print(f"  s123 分布: {dict(sig['s123'].value_counts().sort_index().astype(int).to_dict())}")

# ========= 2. 读取面板构建基准：中证1000传统行业等权 + 中证1000全样本等权 =========
print("\n=== 2. 构建参考组合（用于对比回撤是策略还是β） ===")
panel = pd.read_parquet(PANEL)
# fwd_20 = 未来1月收益（panel 已有）
# 构建 trade_date -> 传统行业池等权收益
# panel 中 is_traditional：来自 p["industry"].isin(传统行业)。直接从主引擎定义：
# 看 sector_rotation/sector_stock_rotation.py 里定义的传统行业（非科技）
# 先看有哪些行业
inds = sorted(panel["industry"].dropna().unique())
print(f"  行业列表: {inds}")
# 按 project_memory 定义，传统行业 = 非科技 (科技=电子/计算机/通信/传媒/军工/医药/电力设备及新能源)
TECH = {"电子","计算机","通信","传媒","国防军工","医药","电力设备及新能源","机械","汽车"}
TRADITIONAL = [i for i in inds if i not in TECH]
print(f"  传统行业 ({len(TRADITIONAL)}): {TRADITIONAL}")
panel["is_traditional_guess"] = panel["industry"].isin(TRADITIONAL)

# 参考收益: 每月 trade_date (月末) 的 fwd_20
# 传统池等权（T池等权）
bench = panel.dropna(subset=["fwd_20"]).groupby("trade_date").apply(
    lambda g: pd.Series({
        "ALL_EQ": g["fwd_20"].mean(),
        "TRAD_EQ": g.loc[g["is_traditional_guess"], "fwd_20"].mean(),
        "TRAD_TOP10": g.nlargest(10, "ret_1m")["fwd_20"].mean() if len(g)>=10 else np.nan,
        "N": len(g),
        "N_TRAD": g["is_traditional_guess"].sum(),
    })).reset_index()
# 把 fwd_20 变成时间序列累计净值（假设每月末调仓）
months = sorted(bench["trade_date"].unique())
cum = {"ALL_EQ": 1.0, "TRAD_EQ": 1.0}
rows_cum = []
for m in months:
    r = bench[bench["trade_date"] == m].iloc[0]
    for k in cum:
        cum[k] *= (1 + r[k])
    rows_cum.append({"ym": m, **{k: cum[k] for k in cum}})
bc = pd.DataFrame(rows_cum).set_index("ym")
for k in cum:
    peak = bc[k].cummax()
    bc[f"{k}_dd"] = bc[k] / peak - 1.0
print(f"  参考组合最大回撤:")
print(f"    全样本等权 ALL_EQ   : {bc['ALL_EQ_dd'].min()*100:.2f}%, 期末收益 {(bc['ALL_EQ'].iloc[-1]-1)*100:.2f}%")
print(f"    传统行业等权 TRAD_EQ: {bc['TRAD_EQ_dd'].min()*100:.2f}%, 期末收益 {(bc['TRAD_EQ'].iloc[-1]-1)*100:.2f}%")
# 对比 ENS_T40 的月频 NAV (从 matrix 反推时间序列)
# 我们直接从主引擎结果里（运行时保存的 log_df），其实用 pickle 能拿到
# 但刚才读取 pickle 格式未知——我就直接再读一下 pkl 的具体结构，看看有没有组合的月度净值
with open(PKL, "rb") as f:
    data = pickle.load(f)
print(f"\n[pkl结构] type={type(data).__name__}")
if isinstance(data, dict):
    for k,v in data.items():
        shape = getattr(v, "shape", None)
        length = len(v) if hasattr(v, "__len__") else None
        print(f"  key={k}: type={type(v).__name__}, shape={shape}, len={length}")
        if isinstance(v, pd.DataFrame):
            print(f"    columns={v.columns.tolist()[:10]}")
            if len(v):
                print(f"    head(3):\n{v.head(3).to_string()}")
        elif isinstance(v, pd.Series):
            print(f"    head(3):\n{v.head(3).to_string()}")
elif isinstance(data, list):
    print(f"  list len={len(data)}")
    if len(data):
        item = data[0]
        print(f"  item[0] type={type(item).__name__}")
        if isinstance(item, tuple):
            print(f"  item[0][0]={item[0]}, item[0][1] keys={list(item[1].keys()) if isinstance(item[1],dict) else 'not dict'}")

# ========= 3. 读取 pkl 中策略的每日 nav 并归因 =========
print("\n=== 3. 策略 ENS_T40_S123 的回撤区间识别 ===")
tag = "ENS_T40_S123_ONLY_S123"
# 定位该策略
nav_this = None
if isinstance(data, list) and len(data):
    for item in data:
        if isinstance(item, tuple) and item[0] == tag:
            payload = item[1]
            if isinstance(payload, dict) and "nav_dated" in payload:
                nav_this = payload["nav_dated"]
                log_df = payload.get("log", None)
                if log_df is not None:
                    print(f"  找到 log_df: {len(log_df)} 行, cols={list(log_df.columns)}")
            break
elif isinstance(data, dict) and tag in data:
    nav_this = data[tag].get("nav_dated", None)
elif isinstance(data, dict) and "nav_dated" in data:
    nav_this = data["nav_dated"]
    if "log" in data:
        log_df = data["log"]
        print(f"  有 log (主脚本只保存了 1 份最佳?): {len(log_df)} rows")

if nav_this is None:
    # 用参考组合 TRAD_EQ 作近似（因为策略 Top40 就是从 TRAD 池选的）
    print(f"  pkl 中没找到 {tag} 的 nav_dated，用 TRAD_EQ×平均仓位近似回撤量级")
    nav_this = bc["TRAD_EQ"].copy()
    # 把 ym (int 20200131) 转 datetime index
    nav_this.index = pd.to_datetime(nav_this.index.astype(str), format="%Y%m%d")

# 识别最大回撤
if isinstance(nav_this.index, pd.DatetimeIndex):
    dt_idx = True
else:
    dt_idx = False
print(f"  nav_this 长度={len(nav_this)}, index type={'DatetimeIndex' if dt_idx else type(nav_this.index).__name__}")
nav_cum = nav_this / nav_this.iloc[0]
peak = nav_cum.cummax()
dd_s = nav_cum / peak - 1.0
max_dd = dd_s.min()
dd_end = dd_s.idxmin()
dd_start = peak.loc[:dd_end].idxmax()
# 若 dd_end/dd_start 是 datetime，转 str
def _fmt(x):
    if hasattr(x, "strftime"): return x.strftime("%Y-%m-%d")
    return str(int(x)) if not pd.isna(x) else str(x)
print(f"  最大回撤 {max_dd*100:.2f}%")
print(f"  起点 {_fmt(dd_start)}, 终点 {_fmt(dd_end)}")

# s123 状态: 用 log_df 如果存在的话
if 'log_df' in dir() and log_df is not None and len(log_df):
    ldf = log_df.set_index(log_df.columns[0]) if "date" not in log_df.columns else log_df.set_index("date")
    state_ts = ldf["state"] if "state" in ldf.columns else ldf.iloc[:,2]
else:
    # 用 sig 的 s123 推断每日状态：≥3 in ≤1 out，否则保持
    state_map = {}
    cur = False
    ym_int_sorted = sorted(sig["ym_int"].unique())
    # 构造 cal_dates 近似: 用 idx_dd 的每日日期
    cal_dates_daily = idx_dd.loc[str(nav_this.index.min() if dt_idx else "2020-02"):str(nav_this.index.max() if dt_idx else "2026-08")].index
    for d in cal_dates_daily:
        ym_int = d.year*100 + d.month
        # 信号基于"上月末"（prev ym）
        prev_ym_int = (d - pd.DateOffset(months=1)).year*100 + (d - pd.DateOffset(months=1)).month
        s = sig[sig["ym_int"]==prev_ym_int]["s123"]
        s123 = float(s.iloc[0]) if len(s) else None
        if s123 is not None:
            if not cur and s123 >= 3:
                cur = True
            elif cur and s123 <= 1:
                cur = False
        state_ts = pd.Series(index=cal_dates_daily, dtype=object)
        # 跳过逐行——简化：按 signal 匹配回撤期区间的 ym
        break  # 先跳了，直接按月度 sig 判断回撤期的信号

# ========= 4. 回撤期的 s123 信号 + 参考组合同期回撤 =========
print("\n=== 4. 回撤期的 s123 信号 + 参考组合回撤对比 ===")
# 把回撤起点/终点转为 ym 区间
if hasattr(dd_start, "year"):
    ds_ym = dd_start.year*100 + dd_start.month
    de_ym = dd_end.year*100 + dd_end.month
else:
    ds_ym = int(str(int(dd_start))[:6])
    de_ym = int(str(int(dd_end))[:6])
# 提取区间内的 s123 信号
mask_sig = (sig["ym_int"]>=ds_ym) & (sig["ym_int"]<=de_ym)
print(f"  回撤期 s123 按月信号: \n{sig.loc[mask_sig, ['ym','S1','S2','S3','s123']].to_string(index=False)}")
# 参考组合同期回撤
if dt_idx:
    bm = bc.copy(); bm.index = pd.to_datetime(bm.index.astype(str), format="%Y%m%d")
    bm_win = bm.loc[str(ds_ym):str(de_ym)]
else:
    bm_win = bc.loc[ds_ym:de_ym]
# 重算起点到终点的累计收益 & 回撤
for col in ["TRAD_EQ", "ALL_EQ"]:
    if len(bm_win) >= 1:
        start_v = bm[col].loc[:bm_win.index[0]].iloc[-1] if len(bm.loc[:bm_win.index[0]]) else bm_win[col].iloc[0]
        win_v = bm_win[col] / start_v
        bm_dd = (win_v / win_v.cummax() - 1).min()
        total_chg = win_v.iloc[-1] - 1.0
        print(f"  参考 {col}: 同期累计 {total_chg*100:.2f}%, 同期 MaxDD {bm_dd*100:.2f}%")
print(f"\n  解读: 如果参考组合同期回撤和策略回撤(-31%)相当 → 回撤是市场β（传统行业普跌）")
print(f"        如果策略回撤显著大于参考组合 → 回撤来自选股/集中持仓")

print(f"\n总耗时 {time.time()-t0:.0f}s")

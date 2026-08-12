# -*- coding: utf-8 -*-
"""v4-A3 前视修复版 + 对照

背景: v4-A3 baseline (年化35.5%/619万) 是硬编码数字, 源码已不可得。
      v5/v6 已确认同款前视 (当月月末数据选股 → 当月月初买入)。
本脚本:
  1. 用 v6 修复引擎在 v4-A3 参数下近似重建 v4-A3 原版逻辑 (无ST/次新/筹码/软加分增强)
  2. 跑【原版】(前视) 和【修复版】(信号错位1月) 两个版本, 量化 v4 前视幅度
  3. 重算 512100 正确基线 (修正硬编码 -60% 的错误数字)
"""
import os, glob, time, calendar, re
import numpy as np
import pandas as pd

ROOT = r"c:\Users\liuqi\quant_system_v2"
DATA = r"D:\iquant_data\data_v2"
PE_CSV = os.path.join(ROOT, "research", "sector_rotation", "results", "industry_pe.csv")
OUT_DIR = os.path.join(ROOT, "research", "sector_rotation", "results")
ML_PANEL = os.path.join(ROOT, "research", "sector_rotation", "stock_ml_panel_72m.parquet")
IND_PARQ = os.path.join(DATA, "industry1", "industry.parquet")
OTHER_DAY_DIR = os.path.join(DATA, "other_day1")

t00 = time.time()
BUY_FEE, SELL_FEE = 0.0010, 0.0015
INIT, TP, MAX_HOLD, PE_PCT_THR = 1_000_000, 0.30, 270, 0.30

# ---- 行业映射 (与 v6 一致) ----
BLACKLIST_SECTORS = {"银行", "证券保险", "地产", "公用事业", "基建"}
PREFERRED_SECTORS = {"煤炭","石油石化","钢铁","有色金属","化工","电力","新能源",
    "半导体芯片","电子","汽车","机械","建材","家电","农业","医药医疗","白酒消费",
    "计算机软件","通信","军工","交通运输","环保","纺织服装","传媒","人工智能"}
SECT_MAP = {
    "医药医疗": ["中成药","化学制药","生物制药","医疗保健","医药商业"],
    "白酒消费": ["白酒","食品","乳制品","啤酒","红黄酒","软饮料"],
    "银行": ["银行"], "证券保险": ["证券","保险"],
    "地产": ["全国地产","区域地产","园区开发","房产服务"],
    "煤炭": ["煤炭开采","焦炭加工"], "钢铁": ["普钢","特种钢","钢加工"],
    "有色金属": ["黄金","铜","铝","铅锌","小金属"],
    "石油石化": ["石油加工","石油开采"],
    "化工": ["化工原料","化工机械","化纤","农药化肥","塑料","日用化工","染料涂料","橡胶"],
    "电力": ["火力发电","水力发电","新型电力"], "公用事业": ["供气供热","水务"],
    "新能源": ["电气设备"], "半导体芯片": ["半导体","元器件","电器仪表"],
    "电子": ["IT设备","元器件"], "计算机软件": ["软件服务","互联网"],
    "通信": ["通信设备","电信运营"], "传媒": ["影视音像","出版业","广告包装"],
    "军工": ["航空","船舶"], "汽车": ["汽车整车","汽车配件","汽车服务","摩托车"],
    "家电": ["家用电器"], "建材": ["水泥","玻璃","其他建材"],
    "机械": ["专用机械","工程机械","机床制造","机械基件","轻工机械","纺织机械","农用机械"],
    "农业": ["种植业","饲料","渔业","农业综合"], "纺织服装": ["纺织","服饰"],
    "交通运输": ["机场","港口","空运","水运","仓储物流","公共交通","路桥"],
    "环保": ["环境保护"], "基建": ["建筑工程","路桥"],
    "人工智能": ["软件服务","互联网","元器件"],
}
IND2SECT = {}
for sect, inds in SECT_MAP.items():
    for i in inds: IND2SECT.setdefault(i, []).append(sect)

print("[1] 基础映射...", flush=True)
ind_df = pd.read_parquet(IND_PARQ)
ind_df = ind_df[["ts_code","name","industry","list_date"]].copy()
ind_df["sectors"] = ind_df["industry"].map(lambda s: IND2SECT.get(s, []))
def has_black(sects): return any(s in BLACKLIST_SECTORS for s in sects)
ind_df["black"] = ind_df["sectors"].apply(has_black)
ind_df = ind_df[(ind_df["sectors"].apply(len) > 0) & (~ind_df["black"])].copy()
code_info = ind_df.set_index("ts_code")[["name","industry","sectors"]].to_dict("index")
print(f"    可映射股票: {len(code_info)}只", flush=True)

# ---- ML面板 ----
print("[2] ML面板...", flush=True)
ml = pd.read_parquet(ML_PANEL)
ml["dt"] = pd.to_datetime(ml["trade_date"].astype(str), format="%Y%m%d")
ml["ym"] = ml["dt"].dt.year * 100 + ml["dt"].dt.month
print(f"    {len(ml):,}行 {ml['ts_code'].nunique()}只 {ml['ym'].nunique()}月", flush=True)

# ---- 估值快照 ----
print("[3] 估值快照...", flush=True)
other_files = sorted(glob.glob(os.path.join(OTHER_DAY_DIR, "*.parquet")))
date_to_file = {}
for f in other_files:
    try: date_to_file[pd.Timestamp(os.path.basename(f)[:8])] = f
    except: pass
yms_unique = sorted(ml["ym"].unique())
val_parts = []
for ym_int in yms_unique:
    y, m = ym_int // 100, ym_int % 100
    last_day = calendar.monthrange(y, m)[1]
    found = None
    for d in range(last_day, 0, -1):
        cand = pd.Timestamp(year=y, month=m, day=d)
        if cand in date_to_file: found = date_to_file[cand]; break
    if found is None: continue
    df = pd.read_parquet(found)
    df["ym"] = ym_int
    keep = [c for c in ["ts_code","ym","pe","circ_mv"] if c in df.columns]
    val_parts.append(df[keep].copy())
val_df = pd.concat(val_parts, ignore_index=True).drop_duplicates(subset=["ts_code","ym"], keep="last")
ml2 = ml.merge(val_df, on=["ts_code","ym"], how="left")
print(f"    估值快照 {len(val_df):,}行 × {val_df['ym'].nunique()}月", flush=True)

# ---- v4 评分 (同 v5/v6 的 z-score 加权, v4 也是这套) ----
def build_scores_v4(df):
    d = df.copy()
    for col in ["pe","roe","netprofit_yoy","chip_conc_20"]:
        if col in d.columns:
            d[col] = d.groupby("ym")[col].transform(lambda x: x.fillna(x.median()))
    if "pb" in d.columns: d["pb"] = d["pb"].clip(0.3, 20)
    if "netprofit_yoy" in d.columns:
        d["yoy_c"] = d["netprofit_yoy"].clip(lower=5, upper=100)
        d["peg"] = d["pe"] / d["yoy_c"].where(d["yoy_c"] > 0, np.nan)
        d["peg"] = d.groupby("ym")["peg"].transform(lambda x: x.fillna(x.median()))
        d["peg"] = d["peg"].clip(0, 10)
    cols_sign = [("peg",-1),("chip_conc_20",+1),("roe",+1),("pe",-1),("netprofit_yoy",+1)]
    for col, sign in cols_sign:
        col_z = f"z_{col}"
        if col not in d.columns: d[col_z] = 0.0; continue
        g = d.groupby("ym")[col]; mu, sd = g.transform("mean"), g.transform("std")
        d[col_z] = (sign * (d[col] - mu) / (sd + 1e-9)).clip(-3, 3)
    d["score"] = (d["z_peg"].fillna(0)*0.40 + d["z_chip_conc_20"].fillna(0)*0.25 +
                  d["z_roe"].fillna(0)*0.15 + d["z_pe"].fillna(0)*0.10 + d["z_netprofit_yoy"].fillna(0)*0.10)
    return d
ml_scored = build_scores_v4(ml2)
print(f"    v4评分完成", flush=True)

# ---- 板块信号 ----
print("[4] 板块信号...", flush=True)
pe_df = pd.read_csv(PE_CSV, index_col=0); pe_df.index = pe_df.index.astype(str)
pe_pct = pe_df.rolling(48, min_periods=12).rank(pct=True)
sect_pct = {}
for sect, inds in SECT_MAP.items():
    avail = [i for i in inds if i in pe_pct.columns]
    if avail: sect_pct[sect] = pe_pct[avail].median(axis=1)
sect_pct_df = pd.DataFrame(sect_pct)
sect_pct_df.index = pd.to_datetime(sect_pct_df.index, format="%Y%m%d")
sect_signal = {}
for idx in sect_pct_df.index:
    ym_int = idx.year * 100 + idx.month
    row = {c: sect_pct_df.loc[idx, c] for c in sect_pct_df.columns
           if pd.notna(sect_pct_df.loc[idx, c]) and c not in BLACKLIST_SECTORS}
    sect_signal[ym_int] = row
print(f"    {len(sect_signal)}个月", flush=True)

# ---- 收盘面板 ----
print("[5] 收盘面板...", flush=True)
pool_codes = set(code_info.keys()) & set(ml["ts_code"].unique())
close_dfs = []
for f in sorted(other_files):
    try: d = pd.Timestamp(os.path.basename(f)[:8])
    except: continue
    if d < pd.Timestamp("2019-12-01") or d > ml["dt"].max(): continue
    df = pd.read_parquet(f, columns=["ts_code","close"])
    df = df[df["ts_code"].isin(pool_codes)].copy()
    if len(df) == 0: continue
    df["dt"] = d
    close_dfs.append(df)
px = pd.concat(close_dfs, ignore_index=True)
close_panel = {}
for code, g in px.groupby("ts_code"):
    s = g.sort_values("dt").set_index("dt")["close"]
    close_panel[code] = s[~s.index.duplicated()].astype(float)
all_dates_set = set()
for s in close_panel.values(): all_dates_set.update(s.index)
all_dates = sorted(all_dates_set)
del close_dfs, px
print(f"    {len(close_panel)}只 {len(all_dates)}天", flush=True)

# ---- v4-A3 策略 (近似重建, shift控制前视) ----
def run_v4a3(shift=True):
    """shift=True: 修复版 (T月数据 → T+1月执行); shift=False: 原版前视"""
    def _next_month(ym_int):
        y, m = ym_int // 100, ym_int % 100
        return (y+1)*100 + 1 if m == 12 else y*100 + m + 1
    ym_to_dt = {}
    for ym_dt in sorted(ml_scored["dt"].unique()):
        ym_int = ym_dt.year * 100 + ym_dt.month
        ym_to_dt[ym_int] = ym_dt
    monthly_picks = {}
    # v4-A3 参数: Top3 + PE≤80 + 50-1500亿 + 白名单×1.2
    top_k, max_same_sector = 3, 2
    max_pe, pw = 80, 1.2
    min_circ, max_circ = 50 * 10_000, 1500 * 10_000
    for ym_int in yms_unique:
        k = ym_int if not shift else _next_month(ym_int)
        if ym_int not in sect_signal: monthly_picks[k] = []; continue
        sig = sect_signal[ym_int]
        undv = [s for s, v in sig.items() if v < PE_PCT_THR]
        if not undv: monthly_picks[k] = []; continue
        if ym_int not in ym_to_dt: monthly_picks[k] = []; continue
        dt_real = ym_to_dt[ym_int]
        sub = ml_scored[ml_scored["dt"] == dt_real].copy()
        if len(sub) == 0: monthly_picks[k] = []; continue
        sub["sects"] = sub["ts_code"].map(lambda c: code_info.get(c, {}).get("sectors", []))
        sub["in_undv"] = sub["sects"].apply(lambda lst: any(s in undv for s in lst))
        sub = sub[sub["in_undv"]].copy()
        if len(sub) == 0: monthly_picks[k] = []; continue
        # v4 硬过滤: 仅 PE(0,80] + 市值50-1500亿
        if "pe" in sub.columns:
            sub = sub[sub["pe"].isna() | ((sub["pe"] > 0) & (sub["pe"] <= max_pe))]
        if "circ_mv" in sub.columns:
            sub = sub[sub["circ_mv"].isna() | ((sub["circ_mv"] >= min_circ) & (sub["circ_mv"] <= max_circ))]
        if len(sub) == 0: monthly_picks[k] = []; continue
        # 白名单加权
        def is_pref(sects): return any(s in PREFERRED_SECTORS for s in sects)
        sub["_pref"] = sub["sects"].apply(is_pref).astype(int)
        sub["score_adj"] = sub["score"] * (1.0 + sub["_pref"] * (pw - 1.0))
        sub = sub.sort_values("score_adj", ascending=False)
        chosen, sec_count = [], {}
        def first_sect(c):
            sects = code_info.get(c, {}).get("sectors", [])
            for s in sects:
                if s in undv: return s
            return sects[0] if sects else "_X_"
        for _, row in sub.iterrows():
            c = row["ts_code"]; s = first_sect(c)
            if sec_count.get(s, 0) >= max_same_sector and len(chosen) >= top_k - 1: continue
            chosen.append(c); sec_count[s] = sec_count.get(s, 0) + 1
            if len(chosen) >= top_k: break
        monthly_picks[k] = chosen
    # 执行 (成交: 当日收盘价, 同 v6)
    cash, holdings, nav_series, trades = float(INIT), {}, [], []
    for di, day in enumerate(all_dates):
        is_monthly_start = (di == 0) or (all_dates[di-1].month != day.month)
        if is_monthly_start:
            ym_int = day.year * 100 + day.month
            target = monthly_picks.get(ym_int, [])
            new_codes = [c for c in target if c not in holdings
                         and c in close_panel and day in close_panel[c].index]
            if new_codes and cash > 10000:
                per = cash / len(new_codes)
                for c in new_codes:
                    p = close_panel[c].loc[day]
                    if p <= 0: continue
                    qty = int(per * (1 - BUY_FEE) / p / 100) * 100
                    if qty < 100: continue
                    cost = qty * p * (1 + BUY_FEE)
                    if cost > cash: continue
                    cash -= cost
                    holdings[c] = {"buy_price": p, "qty": qty, "buy_di": di}
                    trades.append((day, "BUY", c, cost, np.nan, 0))
        for c in list(holdings.keys()):
            if c not in close_panel or day not in close_panel[c].index: continue
            p = close_panel[c].loc[day]
            ret = p / holdings[c]["buy_price"] - 1
            held = di - holdings[c]["buy_di"]
            if ret >= TP or held >= MAX_HOLD:
                proceeds = p * holdings[c]["qty"] * (1 - SELL_FEE)
                cash += proceeds
                trades.append((day, "TP" if ret >= TP else "T270", c, proceeds, ret, held))
                del holdings[c]
        total = cash
        for c, h in holdings.items():
            s = close_panel[c]
            up_to = s[s.index <= day]
            if len(up_to): total += up_to.iloc[-1] * h["qty"]
        nav_series.append((day, total))
    nav_s = pd.Series(dict(nav_series)).sort_index()
    nav_s = nav_s[nav_s.index >= pd.Timestamp("2020-01-01")]
    return nav_s, trades

def metrics(nav_s, trades, label):
    tr = nav_s.iloc[-1] / INIT - 1
    yrs = (nav_s.index[-1] - nav_s.index[0]).days / 365.25
    ann = (1 + tr) ** (1/yrs) - 1 if yrs > 0 else np.nan
    mdd = ((nav_s - nav_s.cummax()) / nav_s.cummax()).min()
    ret = nav_s.pct_change().dropna()
    shp = ret.mean() / (ret.std(ddof=1) + 1e-12) * np.sqrt(252)
    buys = sum(1 for t in trades if t[1] == "BUY")
    tps = sum(1 for t in trades if t[1] == "TP")
    tls = sum(1 for t in trades if t[1] == "T270")
    hds = [t[5] for t in trades if t[1] != "BUY"]
    return {"配置": label, "年化": ann, "累计": tr, "回撤": mdd, "夏普": shp,
            "买入": buys, "止盈": tps, "止损": tls, "止盈率": tps/(buys+1e-9),
            "平均持仓天": np.mean(hds) if hds else 0, "期末(万)": nav_s.iloc[-1]/1e4}

# ---- 512100 正确基线 (2020-2025) ----
print("\n[6] 512100 正确基线 (2020-2025)...", flush=True)
def load_512100():
    parts = []
    for f in sorted(glob.glob(os.path.join(DATA, "data_day1", "*.parquet"))):
        if os.path.getsize(f) <= 1024: continue
        d = os.path.basename(f)[:8]
        if d < "20191201" or d > "20251231": continue
        df = pd.read_parquet(f, columns=["ts_code","trade_date","close"])
        df = df[df["ts_code"] == "512100.SH"]
        if len(df): parts.append(df)
    if not parts: return None
    s = pd.concat(parts, ignore_index=True)
    s["trade_date"] = pd.to_datetime(s["trade_date"].astype(str), format="%Y%m%d")
    s = s.sort_values("trade_date").drop_duplicates("trade_date").set_index("trade_date")["close"].astype(float)
    return s

s512 = load_512100()
if s512 is not None:
    s512 = s512[s512.index >= pd.Timestamp("2020-01-01")]
    tr512 = s512.iloc[-1] / s512.iloc[0] - 1
    yrs = len(s512) / 242
    mdd512 = (s512 / s512.cummax() - 1).min()
    shp512 = s512.pct_change().dropna().mean() / (s512.pct_change().dropna().std()+1e-12) * np.sqrt(252)
    print(f"    512100: 年化{(1+tr512)**(1/yrs)-1:.2%} 累计{tr512:.2%} MaxDD{mdd512:.2%} Sharpe{shp512:.2f}")
else:
    print("    512100 日线不可得, 跳过")

# ---- 跑对照 ----
print("\n[7] 回测...", flush=True)
rows = []
for shift, lb in [(False, "v4-A3近似【原版前视】"), (True, "v4-A3近似【修复版】")]:
    nv, trs = run_v4a3(shift=shift)
    m = metrics(nv, trs, lb)
    rows.append(m)
    print(f"  {lb}: 年化{m['年化']:.1%} 累计{m['累计']:.1%} 回撤{m['回撤']:.1%} 夏普{m['夏普']:.2f} "
          f"买入{m['买入']} 止盈{m['止盈']} 止盈率{m['止盈率']:.0%} 期末{m['期末(万)']:.0f}万", flush=True)

# 硬编码原始baseline + 512100
rows.append({"配置": "v4-A3【硬编码原始】", "年化": 0.355, "累计": 5.19, "回撤": -0.295,
             "夏普": 1.41, "买入": 144, "止盈": 90, "止损": 44, "止盈率": 0.625,
             "平均持仓天": 132.0, "期末(万)": 619.3})
if s512 is not None:
    rows.append({"配置": "512100持有【正确日频】", "年化": (1+tr512)**(1/yrs)-1, "累计": tr512,
                 "回撤": mdd512, "夏普": shp512, "买入": 0, "止盈": 0, "止损": 0,
                 "止盈率": 0.0, "平均持仓天": 0.0, "期末(万)": s512.iloc[-1]/1e4})

rdf = pd.DataFrame(rows)
print("\n" + "="*120)
print(rdf.to_string(index=False, float_format=lambda x: f"{x:.2f}" if not isinstance(x,str) else x))
print("="*120)
rdf.to_csv(os.path.join(OUT_DIR, "v4a3_fixed_compare.csv"), index=False, encoding="utf-8-sig")

# 图
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
rcParams["axes.unicode_minus"] = False
fig, ax = plt.subplots(figsize=(14, 7))
nav_orig, nav_fix = None, None
for shift, lb in [(False, "v4-A3 原版(前视)"), (True, "v4-A3 修复版")]:
    nv, _ = run_v4a3(shift=shift)
    s = nv / nv.iloc[0]
    ax.plot(s.index, s.values, label=f"{lb} (期末{nv.iloc[-1]/1e4:.0f}万)")
if s512 is not None:
    s5 = s512 / s512.iloc[0]
    ax.plot(s5.index, s5.values, label=f"512100持有 (期末{s512.iloc[-1]/1e4:.0f}万)", ls="--")
ax.axhline(6.19, color="orange", ls="--", lw=1, label="硬编码 v4-A3 baseline 619万 (年化35.5%)")
ax.set_title("v4-A3 前视修复对比 (2020-2025)")
ax.legend(fontsize=9); ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "v4a3_fixed_compare.png"), dpi=130)
print(f"\n[完成] {time.time()-t00:.0f}s, 图: v4a3_fixed_compare.png")

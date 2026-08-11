# -*- coding: utf-8 -*-
"""C8 + mf_abs_ratio_20 (第一个变种) 回测, 2021-2026收益曲线

因子: mf_abs_ratio_20 = 20日累计大单净流入 / 20日累计|大单净流入|
       (方向一致性, ∈[-1,1])
模型: GBDT d3 (与C8 baseline一致, 因为MLP高维不稳定)
组合: T40/T60 × s123 × TV12/TV18/无
"""
import os, sys, time, warnings, pickle, glob
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.linear_model import LinearRegression as _LR
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = r"c:\Users\liuqi\quant_system_v2"
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "research", "fund_research", "studies", "rotation_dingtou"))
sys.path.insert(0, os.path.join(ROOT, "research", "sector_rotation"))

from timing_dingtou import fetch_pe_csi300, fetch_bond10y, _rolling_pct, _zscore
from etf_optimize_backtest2 import load_hv_daily, load_industry_daily, load_index_ret

OUT_DIR = os.path.join(ROOT, "research", "sector_rotation", "results")
os.makedirs(OUT_DIR, exist_ok=True)
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False
t0 = time.time()
COST = 20 / 10000.0
SQRT_242 = np.sqrt(242.0)

C8_COLS = ["ivol","ret_1m","momentum_20","volatility_20","alpha_006","alpha_012",
           "enh4_score","vwap_20_resid","float_pnl_20_resid","chip_shift_5_resid"]
MF_IMPROVED = "mf_abs_ratio_20"  # 第一个变种

# === 1. 面板 + 因子 ===
print("[1] 准备面板...", flush=True)
IMPROVED = os.path.join(ROOT, "research/sector_rotation/stock_mf_improved_72m.parquet")
if os.path.exists(IMPROVED):
    mfi = pd.read_parquet(IMPROVED)  # 列: ts_code, trade_date(=month YYYYMM), mf_abs_ratio_20...
    mfi = mfi.rename(columns={"trade_date": "month"})
    mfi["month"] = mfi["month"].astype("int64")
else:
    raise FileNotFoundError(IMPROVED)

panel_orig = pd.read_parquet(os.path.join(ROOT, "research/sector_rotation/stock_ml_panel_72m.parquet"))
panel = panel_orig.copy()
panel["month"] = panel["trade_date"].astype("int64") // 100
panel = panel.merge(mfi[["ts_code","month", MF_IMPROVED]], on=["ts_code","month"], how="left")

def winsorize(s, lo=0.01, hi=0.99):
    if s.notna().sum() < 5: return s
    a, b = s.quantile([lo, hi]); return s.clip(a, b)

PRICE_COLS = ["ret_1m","ivol","momentum_5","momentum_10","momentum_20","momentum_60",
              "volatility_5","volatility_10","volatility_20","alpha_006","alpha_009","alpha_012","alpha_023"]
FIN_COLS = ["roe","or_yoy","netprofit_yoy"]
CHIP_COLS = ["vwap_20","float_pnl_20","chip_shift_5"]
CHIP_BASE = ["ivol","ret_1m","momentum_20","volatility_20","alpha_006","alpha_012"]
CHIP_RESID = ["vwap_20_resid","float_pnl_20_resid","chip_shift_5_resid"]

for c in PRICE_COLS + FIN_COLS + CHIP_COLS:
    panel[c] = panel.groupby("trade_date")[c].transform(lambda s: winsorize(s))
    panel[c] = panel.groupby("trade_date")[c].transform(lambda s: (s - s.mean())/(s.std(ddof=1)+1e-12))
panel[FIN_COLS] = panel[FIN_COLS].fillna(-99.0)
gg = panel.groupby("trade_date")
panel["enh4_score"] = (-0.40*gg["ivol"].rank(pct=True) - 0.35*gg["ret_1m"].rank(pct=True)
                       + 0.15*gg["roe"].rank(pct=True) + 0.05*gg["or_yoy"].rank(pct=True)
                       + 0.05*gg["netprofit_yoy"].rank(pct=True))
for c in CHIP_COLS: panel[f"{c}_resid"] = np.nan
for dt, grp in panel.groupby("trade_date"):
    if len(grp) < 50: continue
    Xb = grp[CHIP_BASE].values
    for c in CHIP_COLS:
        y = grp[c].values
        mask = np.isfinite(y) & np.all(np.isfinite(Xb), axis=1)
        if mask.sum() < 50: continue
        lr = _LR().fit(Xb[mask], y[mask])
        panel.loc[grp.index[mask], f"{c}_resid"] = -(y - lr.predict(Xb))
for c in CHIP_RESID:
    panel[c] = panel.groupby("trade_date")[c].transform(lambda s: winsorize(s))
    panel[c] = panel.groupby("trade_date")[c].transform(lambda s: (s - s.mean())/(s.std(ddof=1)+1e-12))
# mf_abs_ratio_20 截面标准化
panel[MF_IMPROVED] = panel.groupby("trade_date")[MF_IMPROVED].transform(lambda s: winsorize(s))
panel[MF_IMPROVED] = panel.groupby("trade_date")[MF_IMPROVED].transform(
    lambda s: (s - s.mean())/(s.std(ddof=1)+1e-12))
panel[MF_IMPROVED] = panel[MF_IMPROVED].fillna(0.0)
panel["fwd_20"] = panel.groupby("trade_date")["fwd_20"].transform(lambda s: winsorize(s))
panel = panel.dropna(subset=C8_COLS + ["fwd_20"])

oos_months = sorted([m for m in panel["trade_date"].unique() if m >= 20230101])
print(f"    {len(panel):,} 行, {MF_IMPROVED} cov={panel[MF_IMPROVED].ne(0).mean():.0%}, OOS {len(oos_months)}月, {time.time()-t0:.0f}s", flush=True)

# === 2. WFO打分 ===
CACHE = os.path.join(OUT_DIR, "_mf_var1_scores_cache.pkl")
score_c8 = {}
score_c8mf = {}
if os.path.exists(CACHE):
    with open(CACHE, "rb") as f:
        d_ = pickle.load(f)
    score_c8, score_c8mf, score_enh4 = d_["c8"], d_["c8mf"], d_["enh4"]
    print(f"    加载打分缓存, {time.time()-t0:.0f}s", flush=True)
else:
    score_enh4 = {d: (-0.40*g["ivol"].rank(pct=True) - 0.35*g["ret_1m"].rank(pct=True)
                      + 0.15*g["roe"].rank(pct=True) + 0.05*g["or_yoy"].rank(pct=True)
                      + 0.05*g["netprofit_yoy"].rank(pct=True))
                 for d, g in panel.groupby("trade_date")}
    for i, m in enumerate(oos_months):
        tr = panel[panel["trade_date"] < m].sort_values("trade_date")
        val_months = sorted(tr["trade_date"].unique())[-3:]
        vm = tr["trade_date"].isin(val_months).values
        om = panel[panel["trade_date"] == m]
        for name, feats, sdict in [("C8", C8_COLS, score_c8),
                                    ("C8+mf1", C8_COLS + [MF_IMPROVED], score_c8mf)]:
            X, y = tr[feats].values, tr["fwd_20"].values
            Xtr, ytr, Xv, yv = X[~vm], y[~vm], X[vm], y[vm]
            mdl = lgb.LGBMRegressor(n_estimators=500, learning_rate=0.05, num_leaves=7,
                                    max_depth=3, min_child_samples=80, reg_lambda=2.0, reg_alpha=0.1,
                                    subsample=0.9, colsample_bytree=0.9, random_state=42, verbose=-1)
            mdl.fit(Xtr, ytr, eval_set=[(Xv, yv)], callbacks=[lgb.early_stopping(50, verbose=False)])
            sdict[m] = pd.Series(mdl.predict(om[feats].values), index=om["ts_code"])
        if (i+1) % 9 == 0: print(f"    WFO {i+1}/{len(oos_months)}, {time.time()-t0:.0f}s", flush=True)
    for d in sorted(panel["trade_date"].unique()):
        if d not in score_c8: score_c8[d] = score_enh4[d]
        if d not in score_c8mf: score_c8mf[d] = score_enh4[d]
    with open(CACHE, "wb") as f:
        pickle.dump({"c8": score_c8, "c8mf": score_c8mf, "enh4": score_enh4}, f)

score_ens = {}
for d in sorted(panel["trade_date"].unique()):
    a, b = score_c8[d], score_c8mf[d]
    common = a.index.intersection(b.index)
    score_ens[d] = 0.5 * a[common].rank(pct=True) + 0.5 * b[common].rank(pct=True)

def mic(scores, months=oos_months):
    ics=[]
    for m in months:
        s = scores.get(m)
        if s is None: continue
        g = panel[panel["trade_date"] == m].set_index("ts_code")
        common = s.index.intersection(g.index)
        if len(common)<50: continue
        ic = s[common].rank().corr(g.loc[common,"fwd_20"].rank())
        if np.isfinite(ic): ics.append(ic)
    s_ = pd.Series(ics)
    return s_.mean(), s_.mean()/(s_.std()+1e-12)*np.sqrt(12), s_.gt(0).mean()

for name, sc in [("C8",score_c8),("C8+mf1",score_c8mf),("ENS",score_ens)]:
    ic, icir, pos = mic(sc)
    print(f"  WFO IC {name}: {ic:+.4f} ICIR={icir:+.2f} 正率={pos*100:.0f}%", flush=True)

# === 3. 回测引擎 ===
print("\n[3] 回测引擎...", flush=True)
im = pd.read_parquet(os.path.join(ROOT, "research", "studies", "study_008_enhancements",
                                  "data", "industry_map.parquet"))
ind_map = dict(zip(im["ts_code"], im["industry"]))
iw = pd.concat([pd.read_parquet(f) for f in glob.glob(
    os.path.join(r"D:/iquant_data/data_v2/index_weight", "*.parquet"))], ignore_index=True)
iw = iw[iw["index_code"] == "000852.SH"]
iw["iw_date"] = iw["trade_date"].astype(int)
iw_dates = sorted(iw["iw_date"].unique())
iw_by_date = {d: set(g["con_code"]) for d, g in iw.groupby("iw_date")}

panel_codes = set(panel["ts_code"].unique())
px_parts = []
for f in sorted(glob.glob(os.path.join(r"D:/iquant_data/data_v2/data_day1", "*.parquet"))):
    if os.path.getsize(f) <= 1024: continue
    d = os.path.basename(f)[:8]
    if d < "20190601": continue
    df = pd.read_parquet(f, columns=["ts_code","trade_date","open","high","low","close","pct_chg","pre_close"])
    df = df[df["ts_code"].isin(panel_codes)]
    if len(df): px_parts.append(df)
px = pd.concat(px_parts, ignore_index=True)
px["trade_date"] = px["trade_date"].astype(int)
px["r"] = px["pct_chg"] / 100.0
px = px.sort_values(["ts_code","trade_date"])
ret_w = px.pivot_table(index="trade_date", columns="ts_code", values="r", aggfunc="last")
close_w = px.pivot_table(index="trade_date", columns="ts_code", values="close", aggfunc="last").ffill()
open_w = px.pivot_table(index="trade_date", columns="ts_code", values="open", aggfunc="last")
preclose_w = px.pivot_table(index="trade_date", columns="ts_code", values="pre_close", aggfunc="last")
cal_dates = sorted(ret_w.index)
trad_set = set(panel[panel["is_traditional"]]["ts_code"])

pe = fetch_pe_csi300(); bond = fetch_bond10y()
close_ix = pe["close"]; dd_ix = close_ix / close_ix.cummax() - 1.0
erp = 1.0 / pe["pe_ttm"] - bond["y10"].reindex(pe.index).ffill()
month_keys = sorted(set(d // 100 for d in cal_dates))
sig_rows = []
for ym in month_keys:
    d = pd.Timestamp(f"{ym}01") + pd.offsets.MonthEnd(0)
    s1 = 1 if _rolling_pct(pe["pe_ttm"], d) < 0.20 else 0
    s2 = 1 if _zscore(erp, d) > 1.0 else 0
    s3 = 1 if float(dd_ix.asof(d)) <= -0.25 else 0
    sig_rows.append({"ym": ym, "s123": s1 + s2 + s3})
sig_df = pd.DataFrame(sig_rows).set_index("ym")
v8 = load_hv_daily()
all_dates = sorted(set().union(*[set(s.index) for s in v8.values()]))
v8_df = pd.DataFrame(index=all_dates)
for code, s in v8.items(): v8_df[code] = s.reindex(all_dates)
v8_daily = (v8_df * pd.Series({"511990.SH": 1/3, "511260.SH": 1/3, "518880.SH": 1/3})).sum(axis=1).fillna(0)
v8_daily.index = v8_daily.index.astype(int)
v8_daily = v8_daily.reindex(cal_dates).fillna(0)

TOP_N = {"T40": 40, "T60": 60}
MAX_IND = {"T40": 4, "T60": 4}
SCORE_MAP = {"C8GBDT": score_c8, "C8MF1": score_c8mf, "ENS": score_ens}
month_last_map = {d // 100: d for d in sorted(panel["trade_date"].unique())}

def latest_members(rebal_d):
    for d in reversed(iw_dates):
        if d <= rebal_d: return iw_by_date[d]
    return set()

def select_with_limit(scores, max_per_ind, top_n):
    scores = scores.dropna()
    sorted_codes = scores.sort_values(ascending=False)
    selected, ind_count = [], {}
    for code in sorted_codes.index:
        ind = ind_map.get(code, "其他")
        if ind_count.get(ind, 0) < max_per_ind:
            selected.append(code); ind_count[ind] = ind_count.get(ind, 0) + 1
        if len(selected) >= top_n: break
    return selected

def build_vol_signal(vol_lookback):
    ix_ret = load_index_ret("000852.SH")
    ix_ret.index = ix_ret.index.astype(int)
    ix_ret = ix_ret.reindex(cal_dates).ffill().fillna(0.0)
    return (ix_ret.rolling(vol_lookback).std() * SQRT_242).shift(1)

def run_bt(score_src, top_tag, timing=True, tgt_vol=None, floor_w=0.4, vol_lookback=20):
    top_n = TOP_N[top_tag]; max_ind = MAX_IND[top_tag]
    scores = SCORE_MAP[score_src]
    vol_sig = build_vol_signal(vol_lookback) if tgt_vol is not None else None
    rebals = [min(d for d in cal_dates if d // 100 == ym) for ym in sorted(set(d // 100 for d in cal_dates))]
    sig_map = sig_df["s123"].to_dict()
    state_in = False; positions = {}; cash = 0.0; reserve = 1.0e6
    navs = []; prev_s123 = None; logs = []
    for i, d in enumerate(cal_dates):
        ym = d // 100
        if d == rebals[0]: prev_s123 = sig_map.get(ym, 0)
        if i > 0 and cal_dates[i-1] // 100 != ym:
            prev_s123 = sig_map.get(cal_dates[i-1] // 100, 0)
        target_state = False
        if timing:
            if prev_s123 is None: target_state = False
            elif not state_in and prev_s123 >= 3: target_state = True
            elif state_in and prev_s123 <= 1: target_state = False
            else: target_state = state_in
        else: target_state = True
        reserve *= (1 + v8_daily.at[d])
        if d in rebals:
            snap = month_last_map.get(d // 100 - 1)
            pool = scores.get(snap)
            if pool is not None:
                pool = pool[pool.index.isin(latest_members(d)) & pool.index.isin(trad_set)]
            if target_state and not state_in and pool is not None and len(pool):
                sel = select_with_limit(pool, max_ind, top_n)
                equity = cash + reserve
                w = 1.0 if tgt_vol is None else float(np.clip(tgt_vol/(vol_sig.get(d,np.nan) or 0.01), floor_w, 1.0))
                sb = equity * w; reserve = equity * (1-w); cash = sb; positions = {}
                alloc = sb / len(sel) if sel else 0
                for c in sel:
                    o = open_w.at[d, c]
                    if np.isnan(o) or o <= 0: continue
                    plim = preclose_w.at[d, c] * (0.9 if c[:3] in ("300","688") else 0.95) if not np.isnan(preclose_w.at[d, c]) else 0
                    if plim and o <= plim: continue
                    sh = int(alloc / (o * 1.001) // 100 * 100)
                    if sh > 0 and cash >= sh * o * 1.001: cash -= sh * o * 1.001; positions[c] = positions.get(c, 0) + sh
                if positions: state_in = True
            elif not target_state and state_in:
                for c, sh in positions.items():
                    o = open_w.at[d, c]
                    if not np.isnan(o) and o > 0: cash += sh * o * 0.999
                positions = {}; reserve += cash; cash = 0.0; state_in = False
            elif target_state and state_in and pool is not None and len(pool):
                sel = select_with_limit(pool, max_ind, top_n)
                equity = cash + reserve + sum((sh * close_w.at[d, c] if not np.isnan(close_w.at[d, c]) else 0) for c, sh in positions.items())
                w = 1.0 if tgt_vol is None else float(np.clip(tgt_vol/(vol_sig.get(d,np.nan) or 0.01), floor_w, 1.0))
                ts = equity * w
                for c in list(positions):
                    if c not in sel:
                        o = open_w.at[d, c]
                        if not np.isnan(o) and o > 0: cash += positions[c] * o * 0.999
                        del positions[c]
                cv = sum((positions.get(c,0)*close_w.at[d,c] if not np.isnan(close_w.at[d,c]) else 0) for c in positions)
                deficit = ts - cv
                if deficit > 0: avail = min(reserve, deficit); reserve -= avail; cash += avail
                alloc = ts / len(sel) if sel else 0
                for c in sel:
                    o = open_w.at[d, c]
                    if np.isnan(o) or o <= 0: continue
                    have = positions.get(c, 0) * close_w.at[d, c]
                    diff = alloc - have
                    if diff > 100:
                        plim = preclose_w.at[d, c] * (0.9 if c[:3] in ("300","688") else 0.95) if not np.isnan(preclose_w.at[d, c]) else 0
                        if plim and o <= plim: continue
                        sh = int(diff / (o * 1.001) // 100 * 100)
                        if sh > 0 and cash >= sh * o * 1.001: cash -= sh * o * 1.001; positions[c] = positions.get(c, 0) + sh
                    elif diff < -100:
                        sh = int(-diff / (o * 0.999) // 100 * 100); sh = min(sh, positions.get(c, 0))
                        if sh > 0: cash += sh * o * 0.999; positions[c] -= sh
                        if positions.get(c, 0) <= 0: del positions[c]
        reserve += cash; cash = 0.0
        pos_val = sum((sh * close_w.at[d, c] if not np.isnan(close_w.at[d, c]) else 0) for c, sh in positions.items())
        nav = cash + reserve + pos_val
        navs.append(nav)
        logs.append({"date": d, "nav": nav, "n_pos": len(positions), "state": state_in})
    nav_s = pd.Series(navs)
    tot = nav_s.iloc[-1] / nav_s.iloc[0] - 1.0; yrs = len(nav_s) / 242.0
    ann = (1 + tot) ** (1 / yrs) - 1 if yrs > 0 else 0
    dd = ((nav_s - nav_s.cummax()) / nav_s.cummax()).min()
    sharpe = (nav_s.pct_change().fillna(0).mean() / (nav_s.pct_change().fillna(0).std() + 1e-8)) * SQRT_242
    return {"ann": ann, "maxdd": dd, "sharpe": sharpe, "calmar": ann / (-dd + 1e-9),
            "nav": nav_s, "log": pd.DataFrame(logs)}

# === 4. 跑矩阵 ===
print("[4] 回测矩阵...", flush=True)
CONFIGS = [
    ("C8MF1_T40_S123",        "C8MF1", "T40", True, None, 0.4, 20),
    ("C8MF1_T60_S123",        "C8MF1", "T60", True, None, 0.4, 20),
    ("C8MF1_T60_S123_TV12",   "C8MF1", "T60", True, 0.12, 0.5, 20),
    ("C8MF1_T60_S123_TV18",   "C8MF1", "T60", True, 0.18, 0.4, 20),
    ("ENS_T40_S123",          "ENS",   "T40", True, None, 0.4, 20),
    ("ENS_T60_S123",          "ENS",   "T60", True, None, 0.4, 20),
    ("ENS_T60_S123_TV12",     "ENS",   "T60", True, 0.12, 0.5, 20),
    ("ENS_T60_S123_TV18",     "ENS",   "T60", True, 0.18, 0.4, 20),
    ("C8GBDT_T40_S123",       "C8GBDT","T40", True, None, 0.4, 20),
    ("C8GBDT_T60_S123",       "C8GBDT","T60", True, None, 0.4, 20),
]
results = {}
for tag, src, tt, tm, tv, fl, lb in CONFIGS:
    r = run_bt(src, tt, tm, tv, fl, lb)
    results[tag] = r
    print(f"  {tag:<24} CAGR={r['ann']:>7.2%} MaxDD={r['maxdd']:>7.2%} Calmar={r['calmar']:>5.2f} Sharpe={r['sharpe']:>5.2f}", flush=True)

# T7 对照 (从已有pkl加载)
print("\n[5] T7 ETF对照...", flush=True)
EXIST_PKL = os.path.join(ROOT, "research/sector_rotation/results/stock_gbdt_s123_results.pkl")
if os.path.exists(EXIST_PKL):
    with open(EXIST_PKL, "rb") as f:
        ep = pickle.load(f)
    t7_nav_raw = ep["t7"]["nav"]  # 月度 Series, index是字符串'YYYYMM'
    t7_nav = pd.Series({int(k): float(v) for k, v in t7_nav_raw.items()}).sort_index()
    print(f"  T7_ETF对照: 加载已有 (月度数据 {len(t7_nav)} 月)")
else:
    t7_nav = pd.Series({202101:1.0, 202608:1.0}, dtype=float)
    print(f"  T7_ETF对照: 无已有pkl, 跳过")

# === 5. 收益曲线 (2021~2026) + 表 ===
print(f"\n{'='*80}", flush=True)
print(f"{'版本':<28} {'CAGR':>8} {'MaxDD':>8} {'Calmar':>7} {'Sharpe':>7}")
print("-"*80)
rows = []
for tag, r in results.items():
    rows.append({"版本":tag,"CAGR":r["ann"],"MaxDD":r["maxdd"],"Calmar":r["calmar"],"Sharpe":r["sharpe"]})
    print(f"{tag:<28} {r['ann']:>7.2%} {r['maxdd']:>7.2%} {r['calmar']:>6.2f} {r['sharpe']:>6.2f}")
# T7 stats
if len(t7_nav) > 2:
    nv = t7_nav.astype(float)
    tot = nv.iloc[-1] / nv.iloc[0] - 1; yrs = len(nv) / 12.0
    st7_cagr = (1+tot)**(1/yrs)-1
    st7_maxdd = (nv / nv.cummax() - 1).min()
    st7_sharpe = (nv.pct_change().dropna().mean() / (nv.pct_change().dropna().std()+1e-8)) * np.sqrt(12)
    print(f"{'T7_ETF对照':<28} {st7_cagr:>7.2%} {st7_maxdd:>7.2%} {st7_cagr/(-st7_maxdd+1e-9):>6.2f} {st7_sharpe:>6.2f}")
    rows.append({"版本":"T7_ETF对照","CAGR":st7_cagr,"MaxDD":st7_maxdd,"Calmar":st7_cagr/(-st7_maxdd+1e-9),"Sharpe":st7_sharpe})
pd.DataFrame(rows).to_csv(os.path.join(OUT_DIR, "mf_var1_backtest_matrix.csv"), index=False, encoding="utf-8-sig")

# 图
KEYS = [
    ("ENS_T60_S123_TV12",         "全局最优 ENS_T60_TV12",   "crimson"),
    ("C8MF1_T60_S123",            "C8+mf1 T60 s123",        "steelblue"),
    ("C8MF1_T60_S123_TV18",       "C8+mf1 T60 s123 +TV18",  "darkorange"),
    ("ENS_T40_S123",              "ENS(C8+C8MF1) T40 s123","purple"),
    ("C8GBDT_T60_S123",           "C8 GBDT baseline",      "slategray"),
]
fig, axes = plt.subplots(3, 1, figsize=(14, 15))
# 2021+ 切片
cut_mask = {d: i for i, d in enumerate(cal_dates) if int(d) >= 20210101}
cut_i0 = min(cut_mask.values()) if cut_mask else 0

ax = axes[0]
for tag, lb, co in KEYS:
    nv = results[tag]["nav"].iloc[cut_i0:].reset_index(drop=True)
    ax.plot(nv.index, nv / nv.iloc[0], label=lb, color=co, lw=1.3)
# T7 (月度对齐)
t7m = t7_nav.astype(float)
t7_idx = sorted(t7m.index)
if t7_idx and str(t7_idx[0]).isdigit():
    t7_idx2 = [int(i) for i in t7_idx if int(i) >= 202101]
    t7_v2 = [float(t7m.loc[i]) for i in t7_idx2]
    if len(t7_v2) > 1:
        first_d = [d for d in cal_dates if int(d) // 100 == t7_idx2[0]]
        x0 = cal_dates.index(first_d[0]) - cut_i0 if first_d else 0
        x_t7 = np.linspace(x0, len(results[KEYS[0][0]]["nav"])-cut_i0-1, len(t7_v2))
        ax.plot(x_t7, np.array(t7_v2) / t7_v2[0], label="T7 ETF对照",
                color="darkgreen", lw=1.8, ls="--")
xticks = [i for i, d in enumerate(cal_dates[cut_i0:]) if str(d)[-2:] == "01" and str(d)[4:6] in ("01","07")]
ax.set_xticks(xticks); ax.set_xticklabels([str(cal_dates[cut_i0+t])[:6] for t in xticks], rotation=45, fontsize=8)
ax.set_title("C8+mf_abs_ratio_20 收益曲线对比 (2021-2026, 日频)")
ax.legend(fontsize=8); ax.grid(alpha=0.3); ax.set_ylabel("NAV")

ax = axes[1]
for tag, lb, co in KEYS:
    nv = results[tag]["nav"].iloc[cut_i0:].reset_index(drop=True)
    dd = nv / nv.cummax() - 1
    ax.plot(nv.index, dd, label=lb, color=co, lw=1.3)
ax.set_xticks(xticks); ax.set_xticklabels([str(cal_dates[cut_i0+t])[:6] for t in xticks], rotation=45, fontsize=8)
ax.set_title("回撤对比"); ax.legend(fontsize=8); ax.grid(alpha=0.3); ax.set_ylabel("Drawdown")

# 年度收益
def yearly(nav):
    out = {}
    for i, v in enumerate(nav):
        d = cal_dates[cut_i0 + i]
        y = int(d) // 10000
        out.setdefault(y, [nav.iloc[0]] if i == 0 else []).append(v)
    rtn = {}
    for y in sorted(out):
        if len(out[y]) >= 2: rtn[str(y)] = out[y][-1] / out[y][0] - 1
    return rtn
yrs_d = {}
for tag, lb, co in KEYS: yrs_d[lb] = yearly(results[tag]["nav"].iloc[cut_i0:])
yr_d2 = {}
for ym, v in t7_nav.astype(float).items():
    y = str(int(ym))[:4]
    if not yr_d2: yr_d2.setdefault("first", [1.0])
    yr_d2.setdefault(y, []).append(float(v))
yr_d2t7 = {}
first_seen = None
for y, vs in sorted(yr_d2.items()):
    if y == "first": continue
    if first_seen is None: first_seen = vs[0]
    if len(vs) >= 2 and y >= "2021": yr_d2t7[y] = vs[-1] / vs[0] - 1
if yr_d2t7: yrs_d["T7 ETF对照"] = yr_d2t7
for k in list(yrs_d.keys()):
    yrs_d[k] = {y: v for y, v in yrs_d[k].items() if y >= "2021"}
if yrs_d:
    pd.DataFrame(yrs_d).sort_index().plot(kind="bar", ax=axes[2])
    axes[2].set_title("年度收益对比"); axes[2].legend(fontsize=7, ncol=2); axes[2].grid(alpha=0.3, axis="y")

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "mf_var1_nav_curve_2021_2026.png"), dpi=120)

# 保存 pkl
for tag, r in results.items():
    log = r["log"]
    r["nav_dated"] = pd.Series(log["nav"].values, index=log["date"].values).sort_index()
    r["log"] = None
with open(os.path.join(OUT_DIR, "mf_var1_results.pkl"), "wb") as f:
    pickle.dump({"results": results, "t7_nav": t7_nav}, f)
print(f"\n[完成] {time.time()-t0:.0f}s, 图: {OUT_DIR}/mf_var1_nav_curve_2021_2026.png")

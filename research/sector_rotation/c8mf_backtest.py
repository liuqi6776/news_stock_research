# -*- coding: utf-8 -*-
"""大单净流入因子 IC + C8+MF GBDT 回测

步骤:
1. 大单净流入4因子单独IC
2. C8+MF (14因子) GBDT WFO IC
3. C8+MF GBDT 完整回测 (与 ENS_T40/T60 同引擎)
"""
import os, sys, time, warnings, pickle, glob
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression as _LR

ROOT = r"c:\Users\liuqi\quant_system_v2"
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "research", "fund_research", "studies", "rotation_dingtou"))
sys.path.insert(0, os.path.join(ROOT, "research", "sector_rotation"))

from timing_dingtou import fetch_pe_csi300, fetch_bond10y, _rolling_pct, _zscore
from etf_optimize_backtest2 import load_hv_daily, INDUSTRY_ETFS, load_industry_daily, load_index_ret

OUT_DIR = os.path.join(ROOT, "research", "sector_rotation", "results")
os.makedirs(OUT_DIR, exist_ok=True)
t0 = time.time()
COST = 20 / 10000.0
SQRT_242 = np.sqrt(242.0)

C8_COLS = ["ivol","ret_1m","momentum_20","volatility_20","alpha_006","alpha_012",
           "enh4_score","vwap_20_resid","float_pnl_20_resid","chip_shift_5_resid"]
MF_COLS = ["net_mf_ratio_5","lg_net_ratio_5","net_mf_ratio_20","lg_net_ratio_20"]
C8MF_COLS = C8_COLS + MF_COLS  # 14因子

# === 1. 加载面板 ===
print("[1] 加载面板...", flush=True)
panel_ortho = pd.read_parquet(os.path.join(ROOT, "research/sector_rotation/stock_ml_panel_ortho2_72m.parquet"))
panel_orig = pd.read_parquet(os.path.join(ROOT, "research/sector_rotation/stock_ml_panel_72m.parquet"))
panel = panel_orig.merge(
    panel_ortho[["trade_date","ts_code"] + MF_COLS],
    on=["trade_date","ts_code"], how="left")
for c in MF_COLS:
    panel[c] = panel[c].fillna(0.0)

# === 2. 特征准备 (与 stock_gbdt_s123_backtest.py 一致) ===
PRICE_COLS = ["ret_1m","ivol","momentum_5","momentum_10","momentum_20","momentum_60",
              "volatility_5","volatility_10","volatility_20",
              "alpha_006","alpha_009","alpha_012","alpha_023"]
FIN_COLS = ["roe","or_yoy","netprofit_yoy"]
CHIP_COLS = ["vwap_20","float_pnl_20","chip_shift_5"]
CHIP_BASE = ["ivol","ret_1m","momentum_20","volatility_20","alpha_006","alpha_012"]
CHIP_RESID_COLS = ["vwap_20_resid","float_pnl_20_resid","chip_shift_5_resid"]

def winsorize(s, lo=0.01, hi=0.99):
    if s.notna().sum() < 5: return s
    a, b = s.quantile([lo, hi]); return s.clip(a, b)

def prep_feats(df):
    df = df.copy()
    df["has_fin"] = df["roe"].notna().astype(int)
    for c in PRICE_COLS + FIN_COLS + CHIP_COLS:
        df[c] = df.groupby("trade_date")[c].transform(lambda s: winsorize(s))
    for c in PRICE_COLS + FIN_COLS + CHIP_COLS:
        df[c] = df.groupby("trade_date")[c].transform(
            lambda s: (s - s.mean())/(s.std(ddof=1)+1e-12))
    df[FIN_COLS] = df[FIN_COLS].fillna(-99.0)
    gg = df.groupby("trade_date")
    df["enh4_score"] = (-0.40*gg["ivol"].rank(pct=True) - 0.35*gg["ret_1m"].rank(pct=True)
                        + 0.15*gg["roe"].rank(pct=True) + 0.05*gg["or_yoy"].rank(pct=True)
                        + 0.05*gg["netprofit_yoy"].rank(pct=True))
    for c in CHIP_COLS: df[f"{c}_resid"] = np.nan
    for dt, grp in df.groupby("trade_date"):
        if len(grp) < 50: continue
        Xb = grp[CHIP_BASE].values
        for c in CHIP_COLS:
            y = grp[c].values
            mask = np.isfinite(y) & np.all(np.isfinite(Xb), axis=1)
            if mask.sum() < 50: continue
            lr = _LR().fit(Xb[mask], y[mask])
            df.loc[grp.index[mask], f"{c}_resid"] = -(y - lr.predict(Xb))
    for c in CHIP_RESID_COLS:
        df[c] = df.groupby("trade_date")[c].transform(lambda s: winsorize(s))
        df[c] = df.groupby("trade_date")[c].transform(
            lambda s: (s - s.mean())/(s.std(ddof=1)+1e-12))
    # 大单净流入因子: 截面zscore
    for c in MF_COLS:
        df[c] = df.groupby("trade_date")[c].transform(lambda s: winsorize(s))
        df[c] = df.groupby("trade_date")[c].transform(
            lambda s: (s - s.mean())/(s.std(ddof=1)+1e-12))
        df[c] = df[c].fillna(0.0)
    df["fwd_20"] = df.groupby("trade_date")["fwd_20"].transform(lambda s: winsorize(s))
    return df

p = prep_feats(panel)
oos_months = sorted([m for m in p["trade_date"].unique() if m >= 20230101])
print(f"    面板: {len(p):,} 行, OOS {len(oos_months)} 月, {time.time()-t0:.0f}s", flush=True)

# === 3. 大单净流入单因子 IC ===
print("\n[2] 大单净流入单因子 IC (2023+ OOS)...", flush=True)
for c in MF_COLS:
    ics = []
    for m in oos_months:
        g = p[p["trade_date"] == m]
        if len(g) < 50: continue
        ic = g[c].rank().corr(g["fwd_20"].rank())
        if np.isfinite(ic): ics.append(ic)
    s = pd.Series(ics)
    print(f"  {c:>20}: IC={s.mean():+.4f} ICIR={s.mean()/(s.std()+1e-12)*np.sqrt(12):+.2f} 正率={s.gt(0).mean()*100:.0f}%", flush=True)

# === 4. C8+MF GBDT WFO IC ===
print("\n[3] C8+MF (14因子) GBDT WFO IC...", flush=True)
score_c8mf = {}
score_c8gbdt = {}
for i, m in enumerate(oos_months):
    tr = p[p["trade_date"] < m].sort_values("trade_date")
    val_months = sorted(tr["trade_date"].unique())[-3:]
    vm = tr["trade_date"].isin(val_months).values
    om = p[p["trade_date"] == m]
    for name, feats, sdict in [("C8", C8_COLS, score_c8gbdt), ("C8MF", C8MF_COLS, score_c8mf)]:
        feats_r = [c for c in feats if c in tr.columns]
        X, y = tr[feats_r].values, tr["fwd_20"].values
        Xtr, ytr, Xv, yv = X[~vm], y[~vm], X[vm], y[vm]
        mdl = lgb.LGBMRegressor(n_estimators=500, learning_rate=0.05, num_leaves=7,
                                max_depth=3, min_child_samples=80, reg_lambda=2.0, reg_alpha=0.1,
                                subsample=0.9, colsample_bytree=0.9, random_state=42, verbose=-1)
        mdl.fit(Xtr, ytr, eval_set=[(Xv, yv)], callbacks=[lgb.early_stopping(50, verbose=False)])
        sdict[m] = pd.Series(mdl.predict(om[feats_r].values), index=om["ts_code"])
    if (i+1) % 9 == 0: print(f"    WFO {i+1}/{len(oos_months)}, {time.time()-t0:.0f}s", flush=True)

# IC 统计
def mic(scores, panel):
    ics = []
    for m in oos_months:
        g = panel[panel["trade_date"] == m]
        s = scores.get(m)
        if s is None: continue
        common = s.index.intersection(g["ts_code"])
        if len(common) < 50: continue
        ic = s[common].rank().corr(g.set_index("ts_code").loc[common, "fwd_20"].rank())
        if np.isfinite(ic): ics.append(ic)
    s_ = pd.Series(ics)
    return s_.mean(), s_.mean()/(s_.std()+1e-12)*np.sqrt(12), s_.gt(0).mean()

for name, sc in [("C8 GBDT", score_c8gbdt), ("C8+MF GBDT", score_c8mf)]:
    ic, icir, pos = mic(sc, p)
    print(f"  {name:>16}: IC={ic:+.4f} ICIR={icir:+.2f} 正率={pos*100:.0f}%", flush=True)

# 2023前用ENH4填充
score_enh4 = {}
for d, g in p.groupby("trade_date"):
    score_enh4[d] = (-0.40*g["ivol"].rank(pct=True) - 0.35*g["ret_1m"].rank(pct=True)
                     + 0.15*g["roe"].rank(pct=True) + 0.05*g["or_yoy"].rank(pct=True)
                     + 0.05*g["netprofit_yoy"].rank(pct=True))
for d in sorted(p["trade_date"].unique()):
    if d not in score_c8gbdt: score_c8gbdt[d] = score_enh4[d]
    if d not in score_c8mf: score_c8mf[d] = score_enh4[d]

# ENS_C8MF: C8 GBDT 秩 + C8+MF GBDT 秩
score_ens_c8mf = {}
for d in sorted(p["trade_date"].unique()):
    g, m_ = score_c8gbdt[d], score_c8mf[d]
    common = g.index.intersection(m_.index)
    score_ens_c8mf[d] = 0.5 * g[common].rank(pct=True) + 0.5 * m_[common].rank(pct=True)

# 缓存
with open(os.path.join(OUT_DIR, "_c8mf_scores_cache.pkl"), "wb") as f:
    pickle.dump({"c8gbdt": score_c8gbdt, "c8mf": score_c8mf,
                 "ens_c8mf": score_ens_c8mf, "enh4": score_enh4}, f)
print(f"\n    打分完成+缓存, {time.time()-t0:.0f}s", flush=True)

# === 5. 回测引擎 (复用) ===
print("\n[4] 回测...", flush=True)
im = pd.read_parquet(os.path.join(ROOT, "research", "studies", "study_008_enhancements",
                                  "data", "industry_map.parquet"))
ind_map = dict(zip(im["ts_code"], im["industry"]))
iw = pd.concat([pd.read_parquet(f) for f in glob.glob(
    os.path.join(r"D:/iquant_data/data_v2/index_weight", "*.parquet"))], ignore_index=True)
iw = iw[iw["index_code"] == "000852.SH"]
iw["iw_date"] = iw["trade_date"].astype(int)
iw_dates = sorted(iw["iw_date"].unique())
iw_by_date = {d: set(g["con_code"]) for d, g in iw.groupby("iw_date")}

panel_codes = set(p["ts_code"].unique())
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

# s123
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

# V8
v8 = load_hv_daily()
all_dates = sorted(set().union(*[set(s.index) for s in v8.values()]))
v8_df = pd.DataFrame(index=all_dates)
for code, s in v8.items(): v8_df[code] = s.reindex(all_dates)
v8_daily = (v8_df * pd.Series({"511990.SH": 1/3, "511260.SH": 1/3, "518880.SH": 1/3})).sum(axis=1).fillna(0)
v8_daily.index = v8_daily.index.astype(int)
v8_daily = v8_daily.reindex(cal_dates).fillna(0)

TOP_N = {"T40": 40, "T60": 60}
MAX_IND = {"T40": 4, "T60": 4}

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

month_last_map = {d // 100: d for d in sorted(p["trade_date"].unique())}
trad_set = set(p[p["is_traditional"]]["ts_code"])

SCORE_MAP = {"ENH4": score_enh4, "C8GBDT": score_c8gbdt,
             "C8MF": score_c8mf, "ENS_C8MF": score_ens_c8mf}

def run_backtest(score_src, top_tag, timing=True, tgt_vol=None, floor_w=0.4, vol_lookback=20):
    top_n = TOP_N[top_tag]; max_ind = MAX_IND[top_tag]
    scores = SCORE_MAP[score_src]
    vol_sig = build_vol_signal(vol_lookback) if tgt_vol is not None else None
    rebals = []
    for ym in sorted(set(d // 100 for d in cal_dates)):
        rebals.append(min(d for d in cal_dates if d // 100 == ym))
    sig_map = sig_df["s123"].to_dict()
    state_in = False; positions = {}; cash = 0.0; reserve = 1.0e6
    navs = []; prev_s123 = None
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
            pool = scores.get(snap) if snap else None
            if pool is not None:
                pool = pool[pool.index.isin(latest_members(d)) & pool.index.isin(trad_set)]
            if target_state and not state_in and pool is not None and len(pool):
                sel = select_with_limit(pool, max_ind, top_n)
                equity = cash + reserve; w = 1.0 if tgt_vol is None else float(np.clip(tgt_vol/(vol_sig.get(d,np.nan) or 0.01), floor_w, 1.0))
                stock_budget = equity * w; reserve = equity * (1-w); cash = stock_budget; positions = {}
                alloc = stock_budget / len(sel) if sel else 0
                for c in sel:
                    o = open_w.at[d, c]
                    if np.isnan(o) or o <= 0: continue
                    plim = preclose_w.at[d, c] * (0.9 if c[:3] in ("300","688") else 0.95) if not np.isnan(preclose_w.at[d, c]) else 0
                    if plim and o <= plim: continue
                    sh = int(alloc / (o * 1.001) // 100 * 100)
                    if sh > 0 and cash >= sh * o * 1.001:
                        cash -= sh * o * 1.001; positions[c] = positions.get(c, 0) + sh
                if positions: state_in = True
            elif not target_state and state_in:
                for c, sh in positions.items():
                    o = open_w.at[d, c]
                    if not np.isnan(o) and o > 0: cash += sh * o * 0.999
                positions = {}; reserve += cash; cash = 0.0; state_in = False
            elif target_state and state_in and pool is not None and len(pool):
                sel = select_with_limit(pool, max_ind, top_n)
                equity = cash + reserve + sum(sh * close_w.at[d, c] if not np.isnan(close_w.at[d, c]) else 0 for c, sh in positions.items())
                w = 1.0 if tgt_vol is None else float(np.clip(tgt_vol/(vol_sig.get(d,np.nan) or 0.01), floor_w, 1.0))
                target_stock = equity * w
                for c in list(positions):
                    if c not in sel:
                        o = open_w.at[d, c]
                        if not np.isnan(o) and o > 0: cash += positions[c] * o * 0.999
                        del positions[c]
                cur_val = sum(positions.get(c,0)*close_w.at[d,c] if not np.isnan(close_w.at[d,c]) else 0 for c in positions)
                deficit = target_stock - cur_val
                if deficit > 0: avail = min(reserve, deficit); reserve -= avail; cash += avail
                alloc = target_stock / len(sel) if sel else 0
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
        pos_val = sum(sh * close_w.at[d, c] if not np.isnan(close_w.at[d, c]) else 0 for c, sh in positions.items())
        navs.append(cash + reserve + pos_val)
    nav_s = pd.Series(navs)
    tot = nav_s.iloc[-1] / nav_s.iloc[0] - 1.0; yrs = len(nav_s) / 242.0
    ann = (1 + tot) ** (1 / yrs) - 1 if yrs > 0 else 0
    dd = ((nav_s - nav_s.cummax()) / nav_s.cummax()).min()
    sharpe = (nav_s.pct_change().fillna(0).mean() / (nav_s.pct_change().fillna(0).std() + 1e-8)) * SQRT_242
    return {"ann": ann, "maxdd": dd, "sharpe": sharpe, "calmar": ann / (-dd + 1e-9), "nav": nav_s}

# === 6. 跑矩阵 ===
CONFIGS = [
    ("C8GBDT_T40_S123",       "C8GBDT", "T40", True, None, 0.4, 20),
    ("C8MF_T40_S123",         "C8MF",   "T40", True, None, 0.4, 20),
    ("ENS_C8MF_T40_S123",     "ENS_C8MF","T40",True, None, 0.4, 20),
    ("C8GBDT_T60_S123",       "C8GBDT", "T60", True, None, 0.4, 20),
    ("C8MF_T60_S123",         "C8MF",   "T60", True, None, 0.4, 20),
    ("ENS_C8MF_T60_S123",     "ENS_C8MF","T60",True, None, 0.4, 20),
    ("C8MF_T60_S123_TV12",    "C8MF",   "T60", True, 0.12, 0.5, 20),
    ("ENS_C8MF_T60_S123_TV12","ENS_C8MF","T60",True, 0.12, 0.5, 20),
    ("C8MF_T60_S123_TV18",    "C8MF",   "T60", True, 0.18, 0.4, 20),
    ("ENS_C8MF_T60_S123_TV18","ENS_C8MF","T60",True, 0.18, 0.4, 20),
]
results = {}
for tag, src, tt, tm, tv, fl, lb in CONFIGS:
    r = run_backtest(src, tt, tm, tv, fl, lb)
    results[tag] = r
    print(f"  {tag:<32} CAGR={r['ann']:>7.2%} MaxDD={r['maxdd']:>7.2%} Calmar={r['calmar']:>5.2f} Sharpe={r['sharpe']:>5.2f}", flush=True)

# === 7. 汇总 ===
print(f"\n{'='*90}", flush=True)
print(f"{'版本':<36} {'CAGR':>8} {'MaxDD':>8} {'Calmar':>7} {'Sharpe':>7}")
print("-"*90)
rows = []
for tag, r in results.items():
    rows.append({"版本": tag, "CAGR": r["ann"], "MaxDD": r["maxdd"], "Calmar": r["calmar"], "Sharpe": r["sharpe"]})
    print(f"{tag:<36} {r['ann']:>7.2%} {r['maxdd']:>7.2%} {r['calmar']:>6.2f} {r['sharpe']:>6.2f}")
print(f"{'='*90}", flush=True)
pd.DataFrame(rows).to_csv(os.path.join(OUT_DIR, "c8mf_backtest_matrix.csv"), index=False, encoding="utf-8-sig")
print(f"\n[完成] {time.time()-t0:.0f}s, 结果: {OUT_DIR}", flush=True)

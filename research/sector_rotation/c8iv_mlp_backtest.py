# -*- coding: utf-8 -*-
"""C8+IND+VAL MLP 完整回测 (23因子, 与 stock_gbdt_s123_backtest.py 同引擎)

新增打分源: 'C8MLP'(C8 10因子) vs 'C8IVMLP'(C8+IND+VAL 23因子)
组合: ×T40/T60 ×S123_ONLY/IND_MA5 ×S123/ALWAYS → 16 个 + TV 变体 + T7对照

直接复用引擎, 仅替换打分源.
"""
import os, sys, time, warnings
warnings.filterwarnings("ignore")
import joblib, glob, pickle
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression as _LR
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = r"c:\Users\liuqi\quant_system_v2"
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "research", "fund_research", "studies", "rotation_dingtou"))
sys.path.insert(0, os.path.join(ROOT, "research", "sector_rotation"))

from timing_dingtou import fetch_pe_csi300, fetch_bond10y, _rolling_pct, _zscore
from etf_optimize_backtest2 import load_hv_daily, INDUSTRY_ETFS, load_industry_daily, load_index_ret

OUT_DIR = os.path.join(ROOT, "research", "sector_rotation", "results")
os.makedirs(OUT_DIR, exist_ok=True)
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

t0 = time.time()
COST = 20 / 10000.0
SQRT_242 = np.sqrt(242.0)
TOP_N_CHOICES = {"T40": 40, "T60": 60}
MAX_PER_IND = {"T40": 4, "T60": 4}

C8_COLS = ["ivol", "ret_1m", "momentum_20", "volatility_20", "alpha_006", "alpha_012",
           "enh4_score", "vwap_20_resid", "float_pnl_20_resid", "chip_shift_5_resid"]
ORTHO_IND = ["ind_mom_20", "ind_crowd_20", "ind_mf_20"]
VAL_FEATS = ["pe_ep", "ln_circ_mv", "pe_rank", "pb_rank", "ln_mv_rank",
             "turn_rank", "volratio_rank", "pe_pct_3y", "pb_pct_3y", "turn_pct_3y"]
C8IV_COLS = C8_COLS + ORTHO_IND + VAL_FEATS  # 23 因子

# === 1. 加载面板 (ortho2 含 IND+VAL) ===
print("[1] 加载面板与行情...", flush=True)
panel_ortho = pd.read_parquet(os.path.join(ROOT, "research/sector_rotation/stock_ml_panel_ortho2_72m.parquet"))
panel_orig = pd.read_parquet(os.path.join(ROOT, "research/sector_rotation/stock_ml_panel_72m.parquet"))
# 保留完整字段 (panel_orig 有 is_traditional, 行业限制所需)
panel = panel_orig.merge(
    panel_ortho[["trade_date","ts_code"] + ORTHO_IND + VAL_FEATS],
    on=["trade_date","ts_code"], how="left")
# 正交因子 NaN → 0 (中性)
for c in ORTHO_IND + VAL_FEATS:
    if c in panel.columns:
        panel[c] = panel[c].fillna(0.0)

# 行业映射 & 中证1000 成分
im = pd.read_parquet(os.path.join(ROOT, "research", "studies", "study_008_enhancements",
                                  "data", "industry_map.parquet"))
ind_map = dict(zip(im["ts_code"], im["industry"]))

iw_files = os.path.join(r"D:/iquant_data/data_v2/index_weight", "*.parquet")
iw = pd.concat([pd.read_parquet(f) for f in glob.glob(iw_files)], ignore_index=True)
iw = iw[iw["index_code"] == "000852.SH"]
iw["iw_date"] = iw["trade_date"].astype(int)
iw_dates = sorted(iw["iw_date"].unique())
iw_by_date = {d: set(g["con_code"]) for d, g in iw.groupby("iw_date")}

# 日频行情
panel_codes = set(panel["ts_code"].unique())
px_parts = []
px_dir = r"D:/iquant_data/data_v2/data_day1"
for f in sorted(glob.glob(os.path.join(px_dir, "*.parquet"))):
    if os.path.getsize(f) <= 1024: continue
    d = os.path.basename(f)[:8]
    if d < "20190601": continue
    df = pd.read_parquet(f, columns=["ts_code", "trade_date", "open", "high", "low",
                                     "close", "pct_chg", "vol", "pre_close", "amount"])
    df = df[df["ts_code"].isin(panel_codes)]
    if len(df): px_parts.append(df)
px = pd.concat(px_parts, ignore_index=True)
px["trade_date"] = px["trade_date"].astype(int)
px["r"] = px["pct_chg"] / 100.0
print(f"    日频: {len(px):,} 行, {px['ts_code'].nunique()} 只, {time.time()-t0:.0f}s")

px = px.sort_values(["ts_code", "trade_date"])
ret_w = px.pivot_table(index="trade_date", columns="ts_code", values="r", aggfunc="last")
close_w = px.pivot_table(index="trade_date", columns="ts_code", values="close", aggfunc="last")
open_w = px.pivot_table(index="trade_date", columns="ts_code", values="open", aggfunc="last")
preclose_w = px.pivot_table(index="trade_date", columns="ts_code", values="pre_close", aggfunc="last")
close_w = close_w.ffill()
cal_dates = sorted(ret_w.index)

# === 2. s123 信号 + V8 ===
print("[2] s123 + V8...", flush=True)
pe = fetch_pe_csi300()
bond = fetch_bond10y()
close_ix = pe["close"]
dd_ix = close_ix / close_ix.cummax() - 1.0
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

# === 3. 打分生成器 ===
print("[3] 打分生成...", flush=True)
PRICE_COLS = ["ret_1m", "ivol", "momentum_5", "momentum_10", "momentum_20", "momentum_60",
              "volatility_5", "volatility_10", "volatility_20",
              "alpha_006", "alpha_009", "alpha_012", "alpha_023"]
FIN_COLS = ["roe", "or_yoy", "netprofit_yoy"]
CHIP_COLS = ["vwap_20", "float_pnl_20", "chip_shift_5"]
CHIP_BASE = ["ivol", "ret_1m", "momentum_20", "volatility_20", "alpha_006", "alpha_012"]
CHIP_RESID_COLS = ["vwap_20_resid", "float_pnl_20_resid", "chip_shift_5_resid"]

def winsorize(s, lo=0.01, hi=0.99):
    if s.notna().sum() < 5: return s
    a, b = s.quantile([lo, hi]); return s.clip(a, b)

def prep_feats_all(df):
    """准备 C8+IND+VAL 所需全部字段 (含 ortho 因子 zscore + C8 残差筹码)"""
    df = df.copy()
    df["has_fin"] = df["roe"].notna().astype(int)
    # 基础价量/财务/筹码
    for c in PRICE_COLS + FIN_COLS + CHIP_COLS:
        df[c] = df.groupby("trade_date")[c].transform(lambda s: winsorize(s))
    for c in PRICE_COLS + FIN_COLS + CHIP_COLS:
        df[c] = df.groupby("trade_date")[c].transform(
            lambda s: (s - s.mean()) / (s.std(ddof=1) + 1e-12))
    df[FIN_COLS] = df[FIN_COLS].fillna(-99.0)
    # ENH4
    gg = df.groupby("trade_date")
    df["enh4_score"] = (-0.40 * gg["ivol"].rank(pct=True) - 0.35 * gg["ret_1m"].rank(pct=True)
                        + 0.15 * gg["roe"].rank(pct=True) + 0.05 * gg["or_yoy"].rank(pct=True)
                        + 0.05 * gg["netprofit_yoy"].rank(pct=True))
    # 筹码残差
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
            lambda s: (s - s.mean()) / (s.std(ddof=1) + 1e-12))
    # 正交 IND/VAL 因子: 已经是 ortho2 面板构造好的截面zscore/分位, 再做 winsorize+尾修
    for c in ORTHO_IND + VAL_FEATS:
        if c in df.columns:
            df[c] = df.groupby("trade_date")[c].transform(lambda s: winsorize(s))
            df[c] = df.groupby("trade_date")[c].transform(
                lambda s: (s - s.mean()) / (s.std(ddof=1) + 1e-12))
            df[c] = df[c].fillna(0.0)
    return df

# fwd_20 与 winsorize
p = panel.copy()
p["fwd_20"] = p.groupby("trade_date")["fwd_20"].transform(lambda s: winsorize(s))

# 3a. ENH4 打分 (2023前 填充用)
p0 = prep_feats_all(p)
score_enh4 = {d: (-0.40 * g["ivol"].rank(pct=True) - 0.35 * g["ret_1m"].rank(pct=True)
                  + 0.15 * g["roe"].rank(pct=True) + 0.05 * g["or_yoy"].rank(pct=True)
                  + 0.05 * g["netprofit_yoy"].rank(pct=True))
              for d, g in p0.groupby("trade_date")}

# 3b. WFO 滚动重训 MLP (C8 10因子 vs C8+IND+VAL 23因子)
oos_months = [d for d in sorted(p["trade_date"].unique()) if d >= 20230101]
score_c8mlp = {}
score_c8ivmlp = {}

for i, m in enumerate(oos_months):
    tr_all = prep_feats_all(p[p["trade_date"] < m]).sort_values("trade_date")
    val_months = sorted(tr_all["trade_date"].unique())[-3:]
    vm = tr_all["trade_date"].isin(val_months).values
    om = prep_feats_all(p[p["trade_date"] == m])

    for name, feats, sdict in [("C8", C8_COLS, score_c8mlp),
                                ("C8IV", C8IV_COLS, score_c8ivmlp)]:
        feats_real = [c for c in feats if c in tr_all.columns]
        X, y = tr_all[feats_real].values, tr_all["fwd_20"].values
        Xtr, ytr, Xv, yv = X[~vm], y[~vm], X[vm], y[vm]
        sc = StandardScaler()
        Xtr_s = sc.fit_transform(Xtr); Xv_s = sc.transform(Xv)
        mdl = MLPRegressor((256,128,64,32,16), activation="relu", solver="adam", alpha=0.5,
            batch_size=512, learning_rate_init=0.001, max_iter=200,
            early_stopping=True, n_iter_no_change=12, validation_fraction=0.1, random_state=42)
        mdl.fit(Xtr_s, ytr)
        sdict[m] = pd.Series(mdl.predict(sc.transform(om[feats_real].values)), index=om["ts_code"])
    if (i + 1) % 6 == 0:
        print(f"    MLP 重训 {i+1}/{len(oos_months)}, {time.time()-t0:.0f}s", flush=True)

# 2023前 用 ENH4 填充
for d in sorted(p["trade_date"].unique()):
    if d not in score_c8mlp: score_c8mlp[d] = score_enh4[d]
    if d not in score_c8ivmlp: score_c8ivmlp[d] = score_enh4[d]

print(f"    C8 MLP / C8IV MLP 打分完成: {len(oos_months)} 月, {time.time()-t0:.0f}s", flush=True)

# 3c. GBDT 打分 (baseline, 方便对比)
SCORE_CACHE = os.path.join(OUT_DIR, "_c8iv_scores_cache.pkl")
import pickle as _pkl
if os.path.exists(SCORE_CACHE):
    with open(SCORE_CACHE, "rb") as _f:
        _c = _pkl.load(_f)
    score_c8mlp = _c["c8mlp"]
    score_c8ivmlp = _c["c8ivmlp"]
    score_enh4 = _c["enh4"]
    score_gbdt = _c["gbdt"]
    print(f"    加载打分缓存: c8mlp/c8ivmlp/enh4/gbdt, {time.time()-t0:.0f}s", flush=True)
else:
    score_gbdt = {}
    for i, m in enumerate(oos_months):
        tr_all = prep_feats_all(p[p["trade_date"] < m]).sort_values("trade_date")
        val_months = sorted(tr_all["trade_date"].unique())[-3:]
        vm = tr_all["trade_date"].isin(val_months).values
        feats = [c for c in C8_COLS if c in tr_all.columns]
        X, y = tr_all[feats].values, tr_all["fwd_20"].values
        Xtr, ytr, Xv, yv = X[~vm], y[~vm], X[vm], y[vm]
        mdl = lgb.LGBMRegressor(n_estimators=500, learning_rate=0.05, num_leaves=7,
                                max_depth=3, min_child_samples=80, reg_lambda=2.0, reg_alpha=0.1,
                                subsample=0.9, colsample_bytree=0.9, random_state=42, verbose=-1)
        mdl.fit(Xtr, ytr, eval_set=[(Xv, yv)], callbacks=[lgb.early_stopping(50, verbose=False)])
        om = prep_feats_all(p[p["trade_date"] == m])
        score_gbdt[m] = pd.Series(mdl.predict(om[feats].values), index=om["ts_code"])
    for d in sorted(p["trade_date"].unique()):
        if d not in score_gbdt: score_gbdt[d] = score_enh4[d]
    with open(SCORE_CACHE, "wb") as _f:
        _pkl.dump({"c8mlp": score_c8mlp, "c8ivmlp": score_c8ivmlp,
                   "enh4": score_enh4, "gbdt": score_gbdt}, _f)
    print(f"    GBDT 打分完成 + 缓存, {time.time()-t0:.0f}s", flush=True)

# ENS: GBDT 秩 + C8IVMLP 秩 (新混合, 看是否比 GBDT+ENH4 更优)
score_ens2 = {}
for d in sorted(p["trade_date"].unique()):
    g, m_ = score_gbdt[d], score_c8ivmlp[d]
    common = g.index.intersection(m_.index)
    score_ens2[d] = 0.5 * g[common].rank(pct=True) + 0.5 * m_[common].rank(pct=True)

# === 4. 选股工具 ===
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
            selected.append(code)
            ind_count[ind] = ind_count.get(ind, 0) + 1
        if len(selected) >= top_n: break
    return selected

# === 5. 回测引擎 (与 stock_gbdt_s123_backtest.py 完全一致) ===
v8_nav_full = (1 + v8_daily).cumprod()
def build_vol_signal(vol_lookback):
    ix_ret = load_index_ret("000852.SH")
    ix_ret.index = ix_ret.index.astype(int)
    ix_ret = ix_ret.reindex(cal_dates).ffill().fillna(0.0)
    return (ix_ret.rolling(vol_lookback).std() * SQRT_242).shift(1)

def run_backtest(score_src, top_tag, sell_mode, timing, tgt_vol=None,
                 floor_w=0.4, vol_lookback=20):
    top_n = TOP_N_CHOICES[top_tag]
    max_ind = MAX_PER_IND[top_tag]
    scores = {"ENH": score_enh4, "GBDT": score_gbdt,
              "C8MLP": score_c8mlp, "C8IV": score_c8ivmlp,
              "ENS2": score_ens2}[score_src]
    vol_sig = build_vol_signal(vol_lookback) if tgt_vol is not None else None
    rebals = []
    for ym in sorted(set(d // 100 for d in cal_dates)):
        rebals.append(min(d for d in cal_dates if d // 100 == ym))
    month_last_map = {d // 100: d for d in sorted(p["trade_date"].unique())}

    def rebal_scores(d):
        prev_ym = d // 100 - 1
        snap = month_last_map.get(prev_ym)
        if snap is None: return None
        pool = scores.get(snap)
        if pool is None: return None
        trad_codes = set(p0.loc[(p0["trade_date"] == snap) & (p0["is_traditional"]), "ts_code"])
        members = latest_members(d)
        return pool[pool.index.isin(members) & pool.index.isin(trad_codes)]

    def tgt_w(d):
        if tgt_vol is None: return 1.0
        v = vol_sig.get(d, np.nan)
        if not np.isfinite(v) or v <= 0: return 1.0
        return float(np.clip(tgt_vol / v, floor_w, 1.0))

    sig_map = sig_df["s123"].to_dict()
    state_in = False
    positions = {}
    cash = 0.0
    reserve = 1.0e6
    navs = []; portfolio_log = []
    prev_s123 = None
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
        else:
            target_state = True

        reserve *= (1 + v8_daily.at[d])
        if d in rebals:
            if target_state and not state_in:
                pool = rebal_scores(d)
                if pool is not None:
                    sel = select_with_limit(pool, max_ind, top_n)
                    equity = cash + reserve
                    w = tgt_w(d)
                    stock_budget = equity * w
                    reserve = equity * (1 - w)
                    cash = stock_budget
                    positions = {}
                    alloc = stock_budget / len(sel) if len(sel) else 0
                    for c in sel:
                        o = open_w.at[d, c]
                        if np.isnan(o) or o <= 0: continue
                        plim = preclose_w.at[d, c] * (0.9 if c[:3] in ("300", "688") else 0.95) if not np.isnan(preclose_w.at[d, c]) else 0
                        if not np.isnan(plim) and o <= plim: continue
                        sh = int(alloc / (o * 1.001) // 100 * 100)
                        if sh > 0 and cash >= sh * o * 1.001:
                            cash -= sh * o * 1.001
                            positions[c] = positions.get(c, 0) + sh
                    if len(positions) > 0: state_in = True
            elif not target_state and state_in:
                for c, sh in positions.items():
                    o = open_w.at[d, c]
                    if not np.isnan(o) and o > 0: cash += sh * o * 0.999
                positions = {}; reserve += cash; cash = 0.0; state_in = False
            elif target_state and state_in:
                pool = rebal_scores(d)
                if pool is not None:
                    sel = select_with_limit(pool, max_ind, top_n)
                    equity = cash + reserve + sum(
                        sh * close_w.at[d, c] if not np.isnan(close_w.at[d, c]) else 0
                        for c, sh in positions.items())
                    w = tgt_w(d)
                    target_stock = equity * w
                    for c in list(positions):
                        if c not in sel:
                            o = open_w.at[d, c]
                            if not np.isnan(o) and o > 0:
                                cash += positions[c] * o * 0.999
                            del positions[c]
                    cur_val = sum((positions.get(c, 0) * close_w.at[d, c]
                                   if not np.isnan(close_w.at[d, c]) else 0)
                                  for c in positions)
                    deficit = target_stock - cur_val
                    if deficit > 0:
                        avail = min(reserve, deficit)
                        reserve -= avail; cash += avail
                    alloc = target_stock / len(sel) if len(sel) else 0
                    for c in sel:
                        o = open_w.at[d, c]
                        if np.isnan(o) or o <= 0: continue
                        have = positions.get(c, 0) * close_w.at[d, c]
                        diff = alloc - have
                        if diff > 100:
                            plim = preclose_w.at[d, c] * (0.9 if c[:3] in ("300", "688") else 0.95) if not np.isnan(preclose_w.at[d, c]) else 0
                            if not np.isnan(plim) and o <= plim: continue
                            sh = int(diff / (o * 1.001) // 100 * 100)
                            if sh > 0 and cash >= sh * o * 1.001:
                                cash -= sh * o * 1.001
                                positions[c] = positions.get(c, 0) + sh
                        elif diff < -100:
                            sh = int(-diff / (o * 0.999) // 100 * 100)
                            sh = min(sh, positions.get(c, 0))
                            if sh > 0:
                                cash += sh * o * 0.999
                                positions[c] -= sh
                                if positions[c] <= 0: del positions[c]

        if state_in and sell_mode == "IND_MA5":
            prev_d = cal_dates[i-1] if i > 0 else None
            for c in list(positions):
                if prev_d is None: continue
                close_t1 = close_w.at[prev_d, c]
                if np.isnan(close_t1): continue
                hist = close_w[c].loc[:prev_d]
                ma5 = hist.tail(5).mean()
                if not np.isnan(ma5) and close_t1 < ma5:
                    o = open_w.at[d, c]
                    if not np.isnan(o) and o > 0:
                        cash += positions[c] * o * 0.999
                        del positions[c]
        reserve += cash; cash = 0.0
        pos_val = sum((sh * close_w.at[d, c] if not np.isnan(close_w.at[d, c]) else 0)
                      for c, sh in positions.items())
        nav = cash + reserve + pos_val
        navs.append(nav)
        portfolio_log.append({"date": d, "nav": nav, "n_pos": len(positions), "state": state_in})
    nav_s = pd.Series(navs)
    tot = nav_s.iloc[-1] / nav_s.iloc[0] - 1.0
    yrs = len(nav_s) / 242.0
    ann = (1 + tot) ** (1 / yrs) - 1 if yrs > 0 else 0
    dd = ((nav_s - nav_s.cummax()) / nav_s.cummax()).min()
    sharpe = (nav_s.pct_change().fillna(0).mean() / (nav_s.pct_change().fillna(0).std() + 1e-8)) * SQRT_242
    log_df = pd.DataFrame(portfolio_log)
    return {"ann": ann, "maxdd": dd, "sharpe": sharpe, "calmar": ann / (-dd + 1e-9),
            "nav": nav_s, "log": log_df}

# === 6. 跑矩阵 ===
print("[4] 回测矩阵...", flush=True)
SCORES = ["GBDT", "C8MLP", "C8IV", "ENS2"]  # baseline + 3个候选
configs = []
for s in SCORES:
    for t in ("T60",):  # T60 更稳, 重点
        for sm in ("S123_ONLY",):
            for timing in (True,):
                configs.append((f"{s}_{t}_{sm}_S123", s, t, sm, True))

# TV 变体 (用户风险偏好匹配)
TVS = [
    ("C8IV_T60_S123_TV12", "C8IV", "T60", "S123_ONLY", True, 0.12, 0.5, 20),
    ("C8IV_T60_S123_TV15", "C8IV", "T60", "S123_ONLY", True, 0.15, 0.5, 20),
    ("C8IV_T60_S123_TV18", "C8IV", "T60", "S123_ONLY", True, 0.18, 0.4, 20),
    ("ENS2_T60_S123_TV18", "ENS2", "T60", "S123_ONLY", True, 0.18, 0.4, 20),
    ("C8MLP_T60_S123_TV18", "C8MLP", "T60", "S123_ONLY", True, 0.18, 0.4, 20),
]
results = {}
for tag, s, t, sm, timing in configs:
    r = run_backtest(s, t, sm, timing)
    results[tag] = r
    print(f"  {tag:<32} CAGR={r['ann']:>7.2%} MaxDD={r['maxdd']:>7.2%} "
          f"Calmar={r['calmar']:>5.2f} Sharpe={r['sharpe']:>5.2f}", flush=True)

for tag, s, t, sm, timing, tv, fl, lb in TVS:
    r = run_backtest(s, t, sm, timing, tgt_vol=tv, floor_w=fl, vol_lookback=lb)
    results[tag] = r
    print(f"  {tag:<32} CAGR={r['ann']:>7.2%} MaxDD={r['maxdd']:>7.2%} "
          f"Calmar={r['calmar']:>5.2f} Sharpe={r['sharpe']:>5.2f}", flush=True)

# T7 对照
print("\n[5] T7 ETF 对照...", flush=True)
from sector_rotation_traditional import build_signals4, run_graded, TRADITIONAL_ETFS, W_MAP
from etf_optimize_backtest2 import build_series, hv_monthly_ret, monthly_from_daily, calc_stats
panel_etf = load_industry_daily()
trad_codes = [c for _, c in TRADITIONAL_ETFS]
trad_panel = {c: s for c, s in panel_etf.items() if c in set(trad_codes)}
ew_trad_daily = build_series(trad_panel)
monthly_nav = {}
for code, s in panel_etf.items():
    nav_s = (1 + s).cumprod()
    monthly_nav[code] = nav_s.groupby(s.index.str[:6]).last()
nav_panel = pd.DataFrame(monthly_nav).sort_index()
hv = load_hv_daily()
v8_m = hv_monthly_ret(hv)
plain_trad_m = monthly_from_daily(ew_trad_daily)
sig4 = build_signals4(list(nav_panel.index), nav_panel, trad_codes)
t7 = run_graded(nav_panel, sig4, plain_trad_m, v8_m, use_v8=True, mode="strict",
                entry_sig=3, exit_sig=1, sig_col="s123")
st7 = calc_stats(t7)
print(f"  T7(ETF): CAGR={st7['CAGR']:.2%} MaxDD={st7['MaxDD']:.2%} Calmar={st7['Calmar']:.2f} Sharpe={st7['Sharpe']:.2f}")

# === 7. 汇总 + 图 ===
print(f"\n{'='*100}")
print(f"{'版本':<36} {'CAGR':>8} {'MaxDD':>8} {'Calmar':>7} {'Sharpe':>7}")
print("-" * 100)
rows = []
for tag, r in results.items():
    rows.append({"版本": tag, "CAGR": r["ann"], "MaxDD": r["maxdd"],
                 "Calmar": r["calmar"], "Sharpe": r["sharpe"]})
    print(f"{tag:<36} {r['ann']:>7.2%} {r['maxdd']:>7.2%} {r['calmar']:>6.2f} {r['sharpe']:>6.2f}")
print(f"{'T7_ETF对照':<36} {st7['CAGR']:>7.2%} {st7['MaxDD']:>7.2%} {st7['Calmar']:>6.2f} {st7['Sharpe']:>6.2f}")
rows.append({"版本": "T7_ETF对照", "CAGR": st7["CAGR"], "MaxDD": st7["MaxDD"],
             "Calmar": st7["Calmar"], "Sharpe": st7["Sharpe"]})
pd.DataFrame(rows).to_csv(os.path.join(OUT_DIR, "c8iv_mlp_backtest_matrix.csv"),
                          index=False, encoding="utf-8-sig")

# 图
KEY = [
    ("GBDT_T60_S123_ONLY_S123", "GBDT(C8) T60 s123", "steelblue"),
    ("C8MLP_T60_S123_ONLY_S123", "C8_MLP T60 s123", "darkorange"),
    ("C8IV_T60_S123_ONLY_S123", "C8+IND+VAL_MLP T60 s123", "crimson"),
    ("C8IV_T60_S123_TV18", "C8IV T60 s123 +TV18", "purple"),
    ("ENS2_T60_S123_TV18", "ENS2(GBDT+C8IV) T60 +TV18", "teal"),
]
t7_nav = t7["nav"]; t7_nav.index = [str(i) for i in t7_nav.index]

fig, axes = plt.subplots(3, 1, figsize=(14, 16))
ax = axes[0]
for tag, lb, co in KEY:
    nv = results[tag]["nav"]
    ax.plot(nv.index.astype(str), nv / nv.iloc[0], label=lb, color=co, lw=1.3)
ax.plot(t7_nav.index, t7_nav / t7_nav.iloc[0], label="ETF原版 T7", color="darkgreen", lw=1.8)
ax.set_title("C8+IND+VAL_MLP vs GBDT vs C8MLP vs T7 (净值 2020~2025)")
ax.legend(fontsize=8); ax.grid(alpha=0.3); ax.set_ylabel("NAV")
for t in ax.get_xticklabels(): t.set_visible(False); t.set_visible(False)
ax.set_xticks(ax.get_xticks()[::40])

ax = axes[1]
for tag, lb, co in KEY:
    nv = results[tag]["nav"]
    dd = nv / nv.cummax() - 1
    ax.plot(nv.index.astype(str), dd, label=lb, color=co, lw=1.3)
t7_dd = t7_nav / t7_nav.cummax() - 1
ax.plot(t7_nav.index, t7_dd, label="ETF原版 T7", color="darkgreen", lw=1.8)
ax.set_title("回撤对比")
ax.legend(fontsize=8); ax.grid(alpha=0.3); ax.set_ylabel("Drawdown")
ax.set_xticks(ax.get_xticks()[::40])

ax = axes[2]
def yearly(nav, is_monthly=False):
    idx = [str(i) for i in nav.index]
    out = {}
    for y in sorted(set(i[:4] for i in idx)):
        s = nav[[i.startswith(y) for i in idx]]
        if len(s) >= 2: out[y] = s.iloc[-1] / s.iloc[0] - 1
    return out
yrs_d = {}
for tag, lb, co in KEY: yrs_d[lb] = yearly(results[tag]["nav"])
yrs_d["ETF原版 T7"] = yearly(t7_nav)
pd.DataFrame(yrs_d).reindex(sorted(yrs_d["ETF原版 T7"])).plot(kind="bar", ax=ax)
ax.set_title("年度收益对比"); ax.legend(fontsize=7, ncol=3); ax.grid(alpha=0.3, axis="y")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "c8iv_vs_gbdt_vs_t7.png"), dpi=120)

# 保存 pkl
for tag, r in results.items():
    log = r["log"]
    r["nav_dated"] = pd.Series(log["nav"].values, index=log["date"].values).sort_index()
    r["log"] = None
with open(os.path.join(OUT_DIR, "c8iv_mlp_results.pkl"), "wb") as f:
    pickle.dump({"results": results, "t7": t7}, f)

print(f"\n[完成] {time.time()-t0:.0f}s, 结果: {OUT_DIR}")

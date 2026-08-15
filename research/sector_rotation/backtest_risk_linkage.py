# -*- coding: utf-8 -*-
"""组合/风险层联动优化 (linkage optimization)

背景: 基线 UNDERVAL_T60_V8VT6_NL = CAGR 9.71%, MaxDD -17.73%, Sharpe 1.05, Calmar 0.55
单杠杆试探(熔断/集中度/波动率目标/增强/阶梯/门槛放宽)均未超过基线 Calmar, 需"联动"而非孤立调参。

本脚本在 run_backtest 中新增三类联动机制 (均相对基线 NL 口径: UNDERVAL, T60, max_ind=99, V8VT6):
  A. DD-adaptive vol target (回撤自适应波动率目标)
     回撤越深, 目标波动率 tgt_vol 与暴露下限 vol_floor 同步平滑下调:
         m = dd_vol_mult(dd) ∈ [ratio, 1.0]
         tgt_vol_eff = vol_tgt * m ; vol_floor_eff = vol_floor * m
     区别于 dd_scale 的二元档位, 这是通过波动率通道的平滑回撤控制。
  B. Hysteresis dd_scale (滞回熔断)
     熔断乘数记忆"最差回撤", 回撤修复过缓冲区(默认3%)才释放仓位, 消除 dd_scale 的 whipsaw。
  C. Vol-regime × s123 (波动率状态 × 进场门槛)
     组合实现波动率高时收紧进场(需 s123=3), 低时放宽(s123=2), 让择时门槛随风险状态联动。

复用 backtest_undervalued_sector_stock.py 的数据加载与 GBDT 打分缓存(不重训)。
"""
import os
import sys
import time
import glob
import numpy as np
import pandas as pd
import lightgbm as lgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = r"c:\Users\liuqi\quant_system_v2"
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "research", "fund_research", "studies", "rotation_dingtou"))
sys.path.insert(0, os.path.join(ROOT, "research", "sector_rotation"))

from timing_dingtou import fetch_pe_csi300, fetch_bond10y, _rolling_pct, _zscore  # noqa: E402
from etf_optimize_backtest2 import load_hv_daily  # noqa: E402

OUT_DIR = os.path.join(ROOT, "research", "sector_rotation", "results")
os.makedirs(OUT_DIR, exist_ok=True)
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

t0 = time.time()
COST = 20 / 10000.0
SQRT_242 = np.sqrt(242.0)

PE_WINDOW = 48
PE_LOW = 0.30

PRICE_COLS = ["ret_1m", "ivol", "momentum_5", "momentum_10", "momentum_20", "momentum_60",
              "volatility_5", "volatility_10", "volatility_20",
              "alpha_006", "alpha_009", "alpha_012", "alpha_023"]
FIN_COLS = ["roe", "or_yoy", "netprofit_yoy"]
FEAT_COLS = PRICE_COLS + FIN_COLS + ["has_fin"]
CHIP_COLS = ["vwap_20", "float_pnl_20", "chip_shift_5"]
CHIP_BASE = ["ivol", "ret_1m", "momentum_20", "volatility_20", "alpha_006", "alpha_012"]
CHIP_RESID_COLS = ["vwap_20_resid", "float_pnl_20_resid", "chip_shift_5_resid"]
GBDT_FEATS = ["ivol", "ret_1m", "momentum_20", "volatility_20", "alpha_006", "alpha_012",
              "enh4_score", "vwap_20_resid", "float_pnl_20_resid", "chip_shift_5_resid"]

VALUE_TRAP = {
    "银行", "证券", "多元金融", "保险",
    "全国地产", "区域地产", "园区开发", "房产服务",
    "供气供热", "水务", "火力发电", "水力发电", "新型电力",
    "建筑工程", "装修装饰",
    "港口", "路桥", "公路", "机场", "铁路", "水运", "空运", "公共交通", "仓储物流",
}
GROWTH_MFG = {
    "汽车整车", "汽车配件", "汽车服务", "摩托车",
    "专用机械", "工程机械", "机床制造", "机械基件", "轻工机械", "纺织机械", "农用机械",
    "IT设备", "软件服务", "互联网", "半导体", "元器件", "电器仪表", "通信设备", "电信运营",
    "电气设备",
}

# ============ 1. 数据加载 ============
print("[1] 加载面板与行情...", flush=True)
panel = pd.read_parquet(os.path.join(ROOT, "research", "sector_rotation", "stock_ml_panel_72m.parquet"))
code_to_ind = panel.groupby("ts_code")["industry"].last().to_dict()
code_to_trad = panel.groupby("ts_code")["is_traditional"].last().to_dict()

iw_files = os.path.join(r"D:/iquant_data/data_v2/index_weight", "*.parquet")
iw = pd.concat([pd.read_parquet(f) for f in glob.glob(iw_files)], ignore_index=True)
iw = iw[iw["index_code"] == "000852.SH"]
iw["iw_date"] = iw["trade_date"].astype(int)
iw_dates = sorted(iw["iw_date"].unique())
iw_by_date = {d: set(g["con_code"]) for d, g in iw.groupby("iw_date")}

panel_codes = set(panel["ts_code"].unique())
px_parts = []
px_dir = r"D:/iquant_data/data_v2/data_day1"
for f in sorted(glob.glob(os.path.join(px_dir, "*.parquet"))):
    if os.path.getsize(f) <= 1024:
        continue
    d = os.path.basename(f)[:8]
    if d < "20190601":
        continue
    df = pd.read_parquet(f, columns=["ts_code", "trade_date", "open", "high", "low",
                                     "close", "pct_chg", "vol", "pre_close", "amount"])
    df = df[df["ts_code"].isin(panel_codes)]
    if len(df):
        px_parts.append(df)
px = pd.concat(px_parts, ignore_index=True)
px["trade_date"] = px["trade_date"].astype(int)
px["r"] = px["pct_chg"] / 100.0
px = px.sort_values(["ts_code", "trade_date"])
ret_w = px.pivot_table(index="trade_date", columns="ts_code", values="r", aggfunc="last")
close_w = px.pivot_table(index="trade_date", columns="ts_code", values="close", aggfunc="last")
open_w = px.pivot_table(index="trade_date", columns="ts_code", values="open", aggfunc="last")
preclose_w = px.pivot_table(index="trade_date", columns="ts_code", values="pre_close", aggfunc="last")
close_w = close_w.ffill()
print(f"    日频面板 {len(px):,} 行, 宽表 {ret_w.shape}, 耗时{time.time()-t0:.0f}s")

# ============ 2. s123 信号 + V8 ============
print("[2] s123 信号 + V8...", flush=True)
pe = fetch_pe_csi300()
bond = fetch_bond10y()
close_ix = pe["close"]
dd_ix = close_ix / close_ix.cummax() - 1.0
erp = 1.0 / pe["pe_ttm"] - bond["y10"].reindex(pe.index).ffill()
cal_dates = sorted(ret_w.index)
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
for code, s in v8.items():
    v8_df[code] = s.reindex(all_dates)
v8_daily = (v8_df * pd.Series({"511990.SH": 1/3, "511260.SH": 1/3, "518880.SH": 1/3})).sum(axis=1).fillna(0)
v8_daily.index = v8_daily.index.astype(int)
v8_daily = v8_daily.reindex(cal_dates).fillna(0)

# ============ 3. 行业 PE 分位 ============
print("[3] 行业 PE 分位...", flush=True)
pe_df = pd.read_csv(os.path.join(OUT_DIR, "industry_pe.csv"), index_col=0)
pe_df.index = pe_df.index.astype(int)
pe_pct = pe_df.rolling(PE_WINDOW, min_periods=24).rank(pct=True)
pe_month_last = {d // 100: d for d in sorted(pe_pct.index)}

# ============ 4. GBDT 打分 (读缓存) + ENH4 ============
print("[4] 打分生成 (读缓存)...", flush=True)

def winsorize(s, lo=0.01, hi=0.99):
    if s.notna().sum() < 5:
        return s
    a, b = s.quantile([lo, hi])
    return s.clip(a, b)

p = panel.copy()
for c in PRICE_COLS + FIN_COLS:
    p[c] = p.groupby("trade_date")[c].transform(lambda s: winsorize(s))
p["has_fin"] = p["roe"].notna().astype(int)
for c in PRICE_COLS + FIN_COLS:
    p[c] = p.groupby("trade_date")[c].transform(lambda s: (s - s.mean()) / (s.std(ddof=1) + 1e-12))
p[FIN_COLS] = p[FIN_COLS].fillna(-99.0)
p["enh4_score"] = (-0.40 * p["ivol"].rank(pct=True) - 0.35 * p["ret_1m"].rank(pct=True)
                   + 0.15 * p["roe"].rank(pct=True) + 0.05 * p["or_yoy"].rank(pct=True)
                   + 0.05 * p["netprofit_yoy"].rank(pct=True))
score_enh4 = {d: g.set_index("ts_code")["enh4_score"] for d, g in p.groupby("trade_date")}

GBDT_CACHE = os.path.join(OUT_DIR, "score_gbdt_cache.parquet")
_sg = pd.read_parquet(GBDT_CACHE)
score_gbdt = {int(m): g.set_index("ts_code")["score"] for m, g in _sg.groupby("month")}
for d in sorted(panel["trade_date"].unique()):
    if d not in score_gbdt:
        score_gbdt[d] = score_enh4[d]
print(f"    GBDT 打分完成 {len(score_gbdt)} 月, 耗时{time.time()-t0:.0f}s")

# ============ 5. 选股 + 板块过滤 ============
def latest_members(rebal_d):
    for d in reversed(iw_dates):
        if d <= rebal_d:
            return iw_by_date[d]
    return set()

def select_with_limit(scores, max_per_ind, top_n):
    scores = scores.dropna()
    sorted_codes = scores.sort_values(ascending=False)
    selected, ind_count = [], {}
    for code in sorted_codes.index:
        ind = code_to_ind.get(code, "其他")
        if ind_count.get(ind, 0) < max_per_ind:
            selected.append(code)
            ind_count[ind] = ind_count.get(ind, 0) + 1
        if len(selected) >= top_n:
            break
    return selected

def apply_sector_filter(pool, filter_name, sector_pct_snap):
    if pool is None or len(pool) == 0:
        return pool
    if filter_name == "ALL":
        return pool
    inds = np.array([code_to_ind.get(c, "其他") for c in pool.index])
    if filter_name == "TRAD":
        trad = np.array([code_to_trad.get(c, False) for c in pool.index])
        return pool[trad]
    if filter_name == "UNDERVAL":
        pct = np.array([sector_pct_snap.get(i, np.nan) for i in inds])
        return pool[pct < PE_LOW]
    if filter_name == "NOTRAP":
        trap = np.array([i in VALUE_TRAP for i in inds])
        return pool[~trap]
    if filter_name == "UNDERVAL_NOTRAP":
        pct = np.array([sector_pct_snap.get(i, np.nan) for i in inds])
        trap = np.array([i in VALUE_TRAP for i in inds])
        return pool[(pct < PE_LOW) & (~trap)]
    if filter_name == "UNDERVAL_GROWTH":
        pct = np.array([sector_pct_snap.get(i, np.nan) for i in inds])
        grow = np.array([i in GROWTH_MFG for i in inds])
        return pool[(pct < PE_LOW) & grow]
    return pool

# ============ 6. 日频回测引擎 (联动版) ============
MM_ANN = 0.02
MM_DAILY = (1 + MM_ANN) ** (1 / 242) - 1

def compute_tvol_w(port_ret_hist, tgt_vol, floor_w, cap=1.0):
    ret = np.asarray(port_ret_hist, dtype=float)
    vol_d = ret.std()
    if not np.isfinite(vol_d) or vol_d <= 0:
        return 1.0
    w = tgt_vol / (vol_d * SQRT_242)
    return float(np.clip(w, floor_w, cap))

DD_BREAKS = [(-0.05, 1.0), (-0.10, 0.75), (-0.15, 0.50), (-0.20, 0.25), (-np.inf, 0.0)]

def dd_scale(dd):
    for thr, w in DD_BREAKS:
        if dd >= thr:
            return w
    return 0.0

def dd_vol_mult(dd, dd_cut=-0.20, ratio=0.4):
    """回撤 -> 风险预算乘数 (平滑): dd=0 -> 1.0, dd<=dd_cut -> ratio, 线性过渡"""
    if dd >= 0:
        return 1.0
    if dd <= dd_cut:
        return float(ratio)
    return float(1.0 + (1.0 - ratio) * (dd / dd_cut))

def dd_scale_hys(dd, worst_dd, buf=0.03):
    """滞回熔断: 乘数记忆最差回撤, 修复过 buf 才释放"""
    if dd < worst_dd:
        worst_dd = dd
    elif dd >= worst_dd + buf:
        worst_dd = dd
    return dd_scale(worst_dd), worst_dd

def compute_weights(codes, d, mode, lookback=60):
    """组合内个股权重: equal(等权) / invvol(逆波动率) / invvar(逆方差)
    波动率用 d 之前(不含 d)最近 lookback 日收益计算, 避免使用调仓日当天(未来)信息。"""
    n = len(codes)
    if n == 0:
        return {}
    if mode == "equal" or n == 1:
        return {c: 1.0 / n for c in codes}
    hist = ret_w.loc[ret_w.index < d, list(codes)].tail(lookback)
    vol = hist.std()
    if vol.notna().sum() >= 2:
        lo, hi = vol.quantile([0.1, 0.9])
        if hi > lo:
            vol = vol.clip(lo, hi)
    med = vol.median() if vol.notna().any() else 0.0
    if not np.isfinite(med) or med <= 0:
        med = 0.01
    vol = vol.fillna(med).clip(lower=1e-4)
    if mode == "invvol":
        score = 1.0 / vol
    elif mode == "invvar":
        score = 1.0 / (vol ** 2)
    else:
        score = pd.Series(1.0, index=codes)
    score = score / score.sum()
    return score.to_dict()

def run_backtest(filter_name="UNDERVAL", top_n=60, max_ind=99,
                 vol_tgt=0.06, vol_floor=0.5, vol_lookback=60, hedge="v8",
                 dd_break=False, dd_hysteresis=False,
                 dd_vol=False, dd_cut=-0.20, dd_vol_ratio=0.4,
                 vol_regime=False, vol_regime_hi=0.08, entry_th=3, exit_th=1,
                 weighting="equal", weight_lookback=60):
    rebals = []
    for ym in sorted(set(d // 100 for d in cal_dates)):
        rebals.append(min(d for d in cal_dates if d // 100 == ym))
    month_last_map = {d // 100: d for d in sorted(panel["trade_date"].unique())}
    sig_map = sig_df["s123"].to_dict()
    hedge_daily = v8_daily if hedge == "v8" else pd.Series(MM_DAILY, index=cal_dates)

    def pool_at(rebal_d):
        y = rebal_d // 10000
        m = (rebal_d // 100) % 100
        prev_ym = (y - 1) * 100 + 12 if m == 1 else y * 100 + (m - 1)
        snap = month_last_map.get(prev_ym)
        if snap is None:
            return None
        pool = score_gbdt.get(snap)
        if pool is None:
            return None
        members = latest_members(rebal_d)
        pool = pool[pool.index.isin(members)]
        pe_snap = pe_month_last.get(prev_ym)
        sector_pct_snap = pe_pct.loc[pe_snap] if pe_snap is not None else pd.Series(dtype=float)
        return apply_sector_filter(pool, filter_name, sector_pct_snap)

    state_in = False
    positions = {}
    cash = 0.0
    reserve = 1.0e6
    navs = []
    prev_s123 = None
    w_risk = 1.0
    port_ret_hist = []
    prev_nav = None
    nav_peak = 1.0e6
    worst_dd = 0.0
    in_days = 0
    expo_sum = 0.0
    for i, d in enumerate(cal_dates):
        ym = d // 100
        if d == rebals[0]:
            prev_s123 = sig_map.get(ym, 0)
        if i > 0 and cal_dates[i-1] // 100 != ym:
            prev_s123 = sig_map.get(cal_dates[i-1] // 100, 0)

        # 波动率状态 × 进场门槛 (联动 C)
        eth_now = entry_th
        if vol_regime and len(port_ret_hist) >= 20:
            vol_ann_now = float(np.std(port_ret_hist[-vol_lookback:]) * SQRT_242)
            eth_now = 3 if vol_ann_now > vol_regime_hi else 2

        if prev_s123 is None:
            target_state = False
            w_s123 = 0.0
        else:
            if not state_in and prev_s123 >= eth_now:
                target_state = True
            elif state_in and prev_s123 <= exit_th:
                target_state = False
            else:
                target_state = state_in
            w_s123 = 1.0 if target_state else 0.0

        reserve *= (1 + hedge_daily.at[d])

        if d in rebals:
            dd_now = (navs[-1] / nav_peak - 1.0) if navs else 0.0
            # 联动 B: 滞回熔断
            if dd_break and navs:
                if dd_hysteresis:
                    dd_mult, worst_dd = dd_scale_hys(dd_now, worst_dd)
                else:
                    dd_mult = dd_scale(dd_now)
            else:
                dd_mult = 1.0
            # 联动 A: 回撤自适应波动率目标
            if dd_vol:
                m = dd_vol_mult(dd_now, dd_cut, dd_vol_ratio)
                tgt_vol_eff = vol_tgt * m
                vol_floor_eff = vol_floor * m
            else:
                tgt_vol_eff = vol_tgt
                vol_floor_eff = vol_floor

            if target_state and not state_in:
                pool = pool_at(d)
                if pool is not None and len(pool) > 0:
                    sel = select_with_limit(pool, max_ind, top_n)
                    wts = compute_weights(sel, d, weighting, weight_lookback)
                    equity = cash + reserve
                    if vol_tgt is not None and len(port_ret_hist) >= 20:
                        w_risk = compute_tvol_w(port_ret_hist[-vol_lookback:], tgt_vol_eff, vol_floor_eff)
                    else:
                        w_risk = 1.0
                    stock_budget = equity * w_risk * w_s123 * dd_mult
                    reserve = equity - stock_budget
                    cash = stock_budget
                    positions = {}
                    for c in sel:
                        alloc = stock_budget * wts.get(c, 1.0 / len(sel))
                        o = open_w.at[d, c]
                        if np.isnan(o) or o <= 0:
                            continue
                        plim = preclose_w.at[d, c] * (0.8 if c[:3] in ("300", "688") else 0.9) if not np.isnan(preclose_w.at[d, c]) else 0
                        if not np.isnan(plim) and o <= plim:
                            continue
                        sh = int(alloc / (o * 1.001) // 100 * 100)
                        if sh > 0 and cash >= sh * o * 1.001:
                            cash -= sh * o * 1.001
                            positions[c] = positions.get(c, 0) + sh
                    if len(positions) > 0:
                        state_in = True
            elif not target_state and state_in:
                for c, sh in positions.items():
                    o = open_w.at[d, c]
                    if not np.isnan(o) and o > 0:
                        cash += sh * o * 0.999
                positions = {}
                reserve += cash
                cash = 0.0
                state_in = False
                w_risk = 0.0
            elif target_state and state_in:
                pool = pool_at(d)
                if pool is not None and len(pool) > 0:
                    sel = select_with_limit(pool, max_ind, top_n)
                    wts = compute_weights(sel, d, weighting, weight_lookback)
                    equity = cash + reserve + sum(sh * close_w.at[d, c] if not np.isnan(close_w.at[d, c]) else 0
                                                  for c, sh in positions.items())
                    if vol_tgt is not None and len(port_ret_hist) >= 20:
                        w_risk = compute_tvol_w(port_ret_hist[-vol_lookback:], tgt_vol_eff, vol_floor_eff)
                    else:
                        w_risk = 1.0
                    target_stock = equity * w_risk * w_s123 * dd_mult
                    for c in list(positions):
                        if c not in sel:
                            o = open_w.at[d, c]
                            if not np.isnan(o) and o > 0:
                                cash += positions[c] * o * 0.999
                            del positions[c]
                    for c in sel:
                        alloc = target_stock * wts.get(c, 1.0 / len(sel))
                        o = open_w.at[d, c]
                        if np.isnan(o) or o <= 0:
                            continue
                        have = positions.get(c, 0) * close_w.at[d, c]
                        diff = alloc - have
                        if diff > 100:
                            plim = preclose_w.at[d, c] * (0.8 if c[:3] in ("300", "688") else 0.9) if not np.isnan(preclose_w.at[d, c]) else 0
                            if not np.isnan(plim) and o <= plim:
                                continue
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
                                if positions[c] <= 0:
                                    del positions[c]

        reserve += cash
        cash = 0.0
        pos_val = sum(sh * close_w.at[d, c] if not np.isnan(close_w.at[d, c]) else 0
                      for c, sh in positions.items())
        nav = cash + reserve + pos_val
        navs.append(nav)
        nav_peak = max(nav_peak, nav)
        if state_in:
            in_days += 1
            expo_sum += w_risk * w_s123
        if prev_nav is not None and prev_nav > 0:
            port_ret_hist.append(nav / prev_nav - 1.0)
        prev_nav = nav

    nav_s = pd.Series(navs, index=cal_dates)
    tot = nav_s.iloc[-1] / nav_s.iloc[0] - 1.0
    yrs = len(nav_s) / 242.0
    ann = (1 + tot) ** (1 / yrs) - 1 if yrs > 0 else 0
    dd = ((nav_s - nav_s.cummax()) / nav_s.cummax()).min()
    sharpe = (nav_s.pct_change().fillna(0).mean() / (nav_s.pct_change().fillna(0).std() + 1e-8)) * SQRT_242
    return {"ann": ann, "maxdd": dd, "sharpe": sharpe, "calmar": ann / (-dd + 1e-9),
            "nav": nav_s, "final": nav_s.iloc[-1] / nav_s.iloc[0],
            "occupancy": in_days / len(cal_dates),
            "avg_expo": expo_sum / len(cal_dates)}

# ============ 7. 联动矩阵 ============
print("\n[5] 联动矩阵 (UNDERVAL_T60_NL 口径)...", flush=True)
BASE = dict(filter_name="UNDERVAL", top_n=60, max_ind=99,
            vol_tgt=0.06, vol_floor=0.5, hedge="v8")
VARIANTS = [
    ("NL",           dict()),                                           # 等权基线
    ("NL_IV60",      dict(weighting="invvol", weight_lookback=60)),     # 逆波动率 60日
    ("NL_IV120",     dict(weighting="invvol", weight_lookback=120)),    # 逆波动率 120日
    ("NL_IVAR60",    dict(weighting="invvar", weight_lookback=60)),     # 逆方差 60日
    ("NL_IVAR120",   dict(weighting="invvar", weight_lookback=120)),    # 逆方差 120日
]
results = {}
for tag, kw in VARIANTS:
    cfg = dict(BASE)
    cfg.update(kw)
    res = run_backtest(**cfg)
    results[tag] = res
    print(f"  {tag:<16} CAGR={res['ann']:>7.2%} MaxDD={res['maxdd']:>7.2%} "
          f"Sharpe={res['sharpe']:>5.2f} Calmar={res['calmar']:>5.2f} Final={res['final']:>5.2f} "
          f"占用={res['occupancy']:>6.1%} 敞口={res['avg_expo']:>5.2f}", flush=True)

# ============ 8. 汇总 ============
rows = []
for tag, res in results.items():
    rows.append({"变体": tag, "CAGR": res["ann"], "MaxDD": res["maxdd"], "Sharpe": res["sharpe"],
                 "Calmar": res["calmar"], "Final": res["final"],
                 "占用率": res["occupancy"], "平均敞口": res["avg_expo"]})
df = pd.DataFrame(rows)
df.to_csv(os.path.join(OUT_DIR, "risk_weighting_matrix.csv"), index=False, encoding="utf-8-sig")

print("\n=== 按 Calmar 排序 ===")
print(df.sort_values("Calmar", ascending=False).round(4).to_string(index=False))

# 联动 vs 基线 NAV/回撤 对比图
fig, axes = plt.subplots(2, 1, figsize=(14, 10))
for tag in ["NL", "NL_IV60", "NL_IV120", "NL_IVAR60"]:
    nav = results[tag]["nav"]
    axes[0].plot(nav.index.astype(str), nav / nav.iloc[0], label=tag, lw=1.3)
    axes[1].plot(nav.index.astype(str), nav / nav.cummax() - 1, label=tag, lw=1.3)
axes[0].set_title("组合内加权方式对比 (UNDERVAL_T60_NL, V8VT6)")
axes[0].legend(fontsize=9); axes[0].grid(alpha=0.3); axes[0].set_ylabel("NAV")
axes[1].set_title("回撤对比"); axes[1].legend(fontsize=9); axes[1].grid(alpha=0.3); axes[1].set_ylabel("Drawdown")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "risk_weighting_curve.png"), dpi=120)
print(f"[图] risk_weighting_curve.png")

print(f"\n[完成] 总耗时 {time.time()-t0:.0f}s")

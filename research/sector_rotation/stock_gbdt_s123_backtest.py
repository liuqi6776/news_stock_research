# -*- coding: utf-8 -*-
"""B4+B5: 股票 GBDT/ENH4 + s123 择时 + 个股MA5卖 日频回测引擎 (Plan B)

四层日频 state machine (无前视):
  L1 s123 状态机: 月末算 S1(PE分位<20%)/S2(ERP>μ+1σ)/S3(回撤≤-25%),
                  s123>=3 进 / <=1 出, 下月生效
  L2 股票池: 中证1000当月成分 ∩ is_traditional ∩ 月成交额>1亿 ∩ 上市≥60日
  L3 打分选股: ENH4(线性) 或 GBDT(滚动重训), 行业限制 select_with_limit
  L4 持仓/卖出: 组合级 s123 进出 + 个股级 close<MA5 卖出(可选) → 资金进 V8

版本矩阵 16 组合:
  打分 ENH4/GBDT × 持仓 T40/T60 × 卖出 S123_ONLY/IND_MA5 × 择时 S123/ALWAYS
对照: T7 (ETF s123 3进/1出+V8), MA5+20缓冲 (ETF)

产出: research/sector_rotation/results/stock_gbdt_s123_matrix.csv + 图
"""
import os
import sys
import time
import joblib
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
from etf_optimize_backtest2 import load_hv_daily, INDUSTRY_ETFS, load_industry_daily, load_index_ret  # noqa: E402

OUT_DIR = os.path.join(ROOT, "research", "sector_rotation", "results")
os.makedirs(OUT_DIR, exist_ok=True)
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

t0 = time.time()
COST = 20 / 10000.0  # 双边20bps (买卖各10bps)
SQRT_242 = np.sqrt(242.0)
TOP_N_CHOICES = {"T40": 40, "T60": 60}
MAX_PER_IND = {"T40": 4, "T60": 4}

PRICE_COLS = ["ret_1m", "ivol", "momentum_5", "momentum_10", "momentum_20", "momentum_60",
              "volatility_5", "volatility_10", "volatility_20",
              "alpha_006", "alpha_009", "alpha_012", "alpha_023"]
FIN_COLS = ["roe", "or_yoy", "netprofit_yoy"]
FEAT_COLS = PRICE_COLS + FIN_COLS + ["has_fin"]
CHIP_COLS = ["vwap_20", "float_pnl_20", "chip_shift_5"]  # 筹码因子(面板中已有)
CHIP_BASE = ["ivol", "ret_1m", "momentum_20", "volatility_20", "alpha_006", "alpha_012"]
CHIP_RESID_COLS = ["vwap_20_resid", "float_pnl_20_resid", "chip_shift_5_resid"]

# ============ 1. 数据加载 ============
print("[1] 加载面板与行情...", flush=True)
panel = pd.read_parquet(os.path.join(ROOT, "research", "sector_rotation", "stock_ml_panel_72m.parquet"))

# 行业映射
im = pd.read_parquet(os.path.join(ROOT, "research", "studies", "study_008_enhancements",
                                  "data", "industry_map.parquet"))
ind_map = dict(zip(im["ts_code"], im["industry"]))

# 中证1000 成分历史 (月度快照 → 调仓日成分)
iw_files = os.path.join(r"D:/iquant_data/data_v2/index_weight", "*.parquet")
import glob
iw = pd.concat([pd.read_parquet(f) for f in glob.glob(iw_files)], ignore_index=True)
iw = iw[iw["index_code"] == "000852.SH"]
iw["iw_date"] = iw["trade_date"].astype(int)
iw_dates = sorted(iw["iw_date"].unique())
iw_by_date = {d: set(g["con_code"]) for d, g in iw.groupby("iw_date")}

# 日频行情 (仅面板股票, 2019-06 起)
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
print(f"    日频面板: {len(px):,} 行, {px['ts_code'].nunique()} 只, 耗时{time.time()-t0:.0f}s")

# 日频收益宽表 (trade_date × ts_code) + close 宽表
px = px.sort_values(["ts_code", "trade_date"])
ret_w = px.pivot_table(index="trade_date", columns="ts_code", values="r", aggfunc="last")
close_w = px.pivot_table(index="trade_date", columns="ts_code", values="close", aggfunc="last")
open_w = px.pivot_table(index="trade_date", columns="ts_code", values="open", aggfunc="last")
preclose_w = px.pivot_table(index="trade_date", columns="ts_code", values="pre_close", aggfunc="last")
close_w = close_w.ffill()  # 停牌日按停牌价估值
print(f"    宽表: {ret_w.shape}, 耗时{time.time()-t0:.0f}s")

# ============ 2. s123 信号 ============
print("[2] s123 信号...", flush=True)
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
print(f"    s123 信号: 触发>=3 占比 {(sig_df['s123']>=3).mean():.1%}, <=1 占比 {(sig_df['s123']<=1).mean():.1%}")

# V8 避险日收益
v8 = load_hv_daily()
all_dates = sorted(set().union(*[set(s.index) for s in v8.values()]))
v8_df = pd.DataFrame(index=all_dates)
for code, s in v8.items():
    v8_df[code] = s.reindex(all_dates)
v8_daily = (v8_df * pd.Series({"511990.SH": 1/3, "511260.SH": 1/3, "518880.SH": 1/3})).sum(axis=1).fillna(0)
v8_daily.index = v8_daily.index.astype(int)
v8_daily = v8_daily.reindex(cal_dates).fillna(0)

# ============ 3. 打分生成器 ============
print("[3] 打分生成...", flush=True)
def winsorize(s, lo=0.01, hi=0.99):
    if s.notna().sum() < 5:
        return s
    a, b = s.quantile([lo, hi])
    return s.clip(a, b)

# ENH4 打分 (每月截面)
p = panel.copy()
for c in PRICE_COLS + FIN_COLS:
    p[c] = p.groupby("trade_date")[c].transform(lambda s: winsorize(s))
p["has_fin"] = p["roe"].notna().astype(int)
for c in PRICE_COLS + FIN_COLS:
    p[c] = p.groupby("trade_date")[c].transform(
        lambda s: (s - s.mean()) / (s.std(ddof=1) + 1e-12))
p[FIN_COLS] = p[FIN_COLS].fillna(-99.0)

# ENH4 打分 (截面 rank 加权, 与 diag_gbdt_features 一致)
p["enh4_score"] = (-0.40 * p["ivol"].rank(pct=True) - 0.35 * p["ret_1m"].rank(pct=True)
                   + 0.15 * p["roe"].rank(pct=True) + 0.05 * p["or_yoy"].rank(pct=True)
                   + 0.05 * p["netprofit_yoy"].rank(pct=True))
p["score_enh4"] = p["enh4_score"]
score_enh4 = {d: g.set_index("ts_code")["score_enh4"] for d, g in p.groupby("trade_date")}

# GBDT 特征集 C8: C7(核心价量6+ENH4) + 3残差筹码 (diag 最优, IC 0.0966 > C7 0.0936)
GBDT_FEATS = ["ivol", "ret_1m", "momentum_20", "volatility_20", "alpha_006", "alpha_012",
              "enh4_score", "vwap_20_resid", "float_pnl_20_resid", "chip_shift_5_resid"]

from sklearn.linear_model import LinearRegression as _LR

def prep_feats(df, feats):
    df = df.copy()
    df["has_fin"] = df["roe"].notna().astype(int)
    for c in PRICE_COLS + FIN_COLS + CHIP_COLS:
        df[c] = df.groupby("trade_date")[c].transform(lambda s: winsorize(s))
    for c in PRICE_COLS + FIN_COLS + CHIP_COLS:
        df[c] = df.groupby("trade_date")[c].transform(
            lambda s: (s - s.mean()) / (s.std(ddof=1) + 1e-12))
    df[FIN_COLS] = df[FIN_COLS].fillna(-99.0)
    df["enh4_score"] = (-0.40 * df["ivol"].rank(pct=True) - 0.35 * df["ret_1m"].rank(pct=True)
                        + 0.15 * df["roe"].rank(pct=True) + 0.05 * df["or_yoy"].rank(pct=True)
                        + 0.05 * df["netprofit_yoy"].rank(pct=True))
    # 筹码因子残差化 (逐月截面 OLS 对 C7 基础因子正交, 取负对齐方向)
    for c in CHIP_COLS:
        df[f"{c}_resid"] = np.nan
    for dt, grp in df.groupby("trade_date"):
        if len(grp) < 50:
            continue
        Xb = grp[CHIP_BASE].values
        for c in CHIP_COLS:
            y = grp[c].values
            mask = np.isfinite(y) & np.all(np.isfinite(Xb), axis=1)
            if mask.sum() < 50:
                continue
            lr = _LR(fit_intercept=True)
            lr.fit(Xb[mask], y[mask])
            resid = y - lr.predict(Xb)
            df.loc[grp.index[mask], f"{c}_resid"] = -resid
    for c in CHIP_RESID_COLS:
        df[c] = df.groupby("trade_date")[c].transform(lambda s: winsorize(s))
        df[c] = df.groupby("trade_date")[c].transform(
            lambda s: (s - s.mean()) / (s.std(ddof=1) + 1e-12))
    return df

# GBDT 滚动重训打分 (预测月 m 用 <= m-1 数据训练, C7 特征 + 早停)
oos_months = [d for d in sorted(panel["trade_date"].unique()) if d >= 20230101]
score_gbdt = {}
for i, m in enumerate(oos_months):
    tr = prep_feats(panel[panel["trade_date"] < m], GBDT_FEATS).sort_values("trade_date")
    X, y = tr[GBDT_FEATS].values, tr["fwd_20"].values
    val_months = sorted(tr["trade_date"].unique())[-3:]
    vm = tr["trade_date"].isin(val_months).values
    mdl = lgb.LGBMRegressor(n_estimators=500, learning_rate=0.05, num_leaves=7,
                            max_depth=3, min_child_samples=80, reg_lambda=2.0, reg_alpha=0.1,
                            subsample=0.9, colsample_bytree=0.9, random_state=42, verbose=-1)
    mdl.fit(X[~vm], y[~vm], eval_set=[(X[vm], y[vm])],
            callbacks=[lgb.early_stopping(50, verbose=False)])
    om = prep_feats(panel[panel["trade_date"] == m], GBDT_FEATS)
    score_gbdt[m] = pd.Series(mdl.predict(om[GBDT_FEATS]), index=om["ts_code"])
    if (i + 1) % 12 == 0:
        print(f"    GBDT 重训 {i+1}/{len(oos_months)}, 耗时{time.time()-t0:.0f}s", flush=True)
print(f"    GBDT 打分完成: {len(score_gbdt)} 月, 耗时{time.time()-t0:.0f}s")

# 2023 之前调仓月用 ENH4 打分填充 GBDT (无模型)
for d in sorted(panel["trade_date"].unique()):
    if d not in score_gbdt:
        score_gbdt[d] = score_enh4[d]

# ENS 混合打分: 0.5*ENH4秩 + 0.5*GBDT秩 (逐月截面)
score_ens = {}
for d in sorted(panel["trade_date"].unique()):
    e, g = score_enh4[d], score_gbdt[d]
    common = e.index.intersection(g.index)
    s = 0.5 * e[common].rank(pct=True) + 0.5 * g[common].rank(pct=True)
    score_ens[d] = s

# ============ 4. 选股工具 ============
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
        ind = ind_map.get(code, "其他")
        if ind_count.get(ind, 0) < max_per_ind:
            selected.append(code)
            ind_count[ind] = ind_count.get(ind, 0) + 1
        if len(selected) >= top_n:
            break
    return selected

# ============ 5. 日频回测引擎 ============
# V8 避险: 真实跟踪 reserve 市值 (每日按 v8_daily 增值), 修复原引擎出场期 NAV 不记 V8 的 bug
# target volatility 层: 中证1000(000852) 指数近 vol_lookback 日年化波动率 (T-1 信号),
#   调仓日 w = clip(tgt_vol / vol_t1, floor_w, 1.0), (1-w) 仓位放 V8
v8_nav_full = (1 + v8_daily).cumprod()  # int index

def build_vol_signal(vol_lookback):
    ix_ret = load_index_ret("000852.SH")  # str yyyymmdd index
    ix_ret.index = ix_ret.index.astype(int)
    ix_ret = ix_ret.reindex(cal_dates).ffill().fillna(0.0)
    ix_vol = ix_ret.rolling(vol_lookback).std() * SQRT_242  # 年化波动率
    return ix_vol.shift(1)  # T-1: 只用 T 日之前收盘算出的波动率

def run_backtest(score_src, top_tag, sell_mode, timing, tgt_vol=None,
                 floor_w=0.4, vol_lookback=20, period=None):
    """score_src: 'ENH'/'GBDT'/'ENS'; top_tag: 'T40'/'T60'; sell_mode: 'S123_ONLY'/'IND_MA5';
       timing: True(s123) / False(ALWAYS); tgt_vol: None=无波动率层, 否则目标年化波动率"""
    top_n = TOP_N_CHOICES[top_tag]
    max_ind = MAX_PER_IND[top_tag]
    scores = {"ENH": score_enh4, "GBDT": score_gbdt, "ENS": score_ens}[score_src]
    vol_sig = build_vol_signal(vol_lookback) if tgt_vol is not None else None

    # 调仓日: 每月首个交易日; 打分用上月末收盘快照 (无前视)
    rebals = []
    for ym in sorted(set(d // 100 for d in cal_dates)):
        mf = min(d for d in cal_dates if d // 100 == ym)
        rebals.append(mf)
    month_last_map = {d // 100: d for d in sorted(panel["trade_date"].unique())}

    def rebal_scores(d):
        prev_ym = d // 100 - 1
        snap = month_last_map.get(prev_ym)
        if snap is None:
            return None
        pool = scores.get(snap)
        if pool is None:
            return None
        trad_codes = set(p.loc[(p["trade_date"] == snap) & (p["is_traditional"]), "ts_code"])
        members = latest_members(d)
        pool = pool[pool.index.isin(members) & pool.index.isin(trad_codes)]
        return pool

    def tgt_w(d):
        if tgt_vol is None:
            return 1.0
        v = vol_sig.get(d, np.nan)
        if not np.isfinite(v) or v <= 0:
            return 1.0
        return float(np.clip(tgt_vol / v, floor_w, 1.0))

    sig_map = sig_df["s123"].to_dict()

    state_in = False
    positions = {}  # code -> shares (股票)
    cash = 0.0
    reserve = 1.0e6  # V8 避险市值 (初始全仓 V8)
    navs = []
    portfolio_log = []

    # 上一月信号 (T-1 月末计算)
    prev_s123 = None
    for i, d in enumerate(cal_dates):
        ym = d // 100
        # --- 月末信号 → 下月状态 (在月初第一个交易日切换) ---
        if d == rebals[0]:
            prev_s123 = sig_map.get(ym, 0)
        if i > 0 and cal_dates[i-1] // 100 != ym:
            prev_s123 = sig_map.get(cal_dates[i-1] // 100, 0)

        target_state = False
        if timing:
            if prev_s123 is None:
                target_state = False
            else:
                if not state_in and prev_s123 >= 3:
                    target_state = True
                elif state_in and prev_s123 <= 1:
                    target_state = False
                else:
                    target_state = state_in
        else:
            target_state = True

        # --- V8 每日增值 (先于当日调仓) ---
        reserve *= (1 + v8_daily.at[d])

        # --- 调仓日: 组合级进出 + 月度再平衡 (含 target vol 缩放) ---
        if d in rebals:
            if target_state and not state_in:
                # 建仓: 上月末收盘打分, 本月首日开盘买入 (无前视)
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
                        if np.isnan(o) or o <= 0:
                            continue
                        plim = preclose_w.at[d, c] * (0.9 if c[:3] in ("300", "688") else 0.95) if not np.isnan(preclose_w.at[d, c]) else 0
                        if not np.isnan(plim) and o <= plim * 1.0:  # 跌停不买
                            continue
                        sh = int(alloc / (o * 1.001) // 100 * 100)
                        if sh > 0 and cash >= sh * o * 1.001:
                            cash -= sh * o * 1.001
                            positions[c] = positions.get(c, 0) + sh
                    if len(positions) > 0:  # 成分缺失(空 sel)时不锁定 in 状态, 下月重试
                        state_in = True
            elif not target_state and state_in:
                # 清仓 → 全部转 V8
                for c, sh in positions.items():
                    o = open_w.at[d, c]
                    if not np.isnan(o) and o > 0:
                        cash += sh * o * 0.999
                positions = {}
                reserve += cash
                cash = 0.0
                state_in = False
            elif target_state and state_in:
                # 月度再平衡到等权 TopN × 目标波动率仓位
                pool = rebal_scores(d)
                if pool is not None:
                    sel = select_with_limit(pool, max_ind, top_n)
                    equity = cash + reserve + sum(sh * close_w.at[d, c] if not np.isnan(close_w.at[d, c]) else 0
                                                  for c, sh in positions.items())
                    w = tgt_w(d)
                    target_stock = equity * w
                    # 卖出不在目标的 (按当日 open 结算)
                    for c in list(positions):
                        if c not in sel:
                            o = open_w.at[d, c]
                            if not np.isnan(o) and o > 0:
                                cash += positions[c] * o * 0.999
                            del positions[c]
                    # 资金缺口从 V8 reserve 补足 (加仓/回补买盘预算)
                    cur_val = sum(positions.get(c, 0) * close_w.at[d, c]
                                  if not np.isnan(close_w.at[d, c]) else 0 for c in positions)
                    deficit = target_stock - cur_val
                    if deficit > 0:
                        avail = min(reserve, deficit)
                        reserve -= avail
                        cash += avail
                    alloc = target_stock / len(sel) if len(sel) else 0
                    for c in sel:
                        o = open_w.at[d, c]
                        if np.isnan(o) or o <= 0:
                            continue
                        have = positions.get(c, 0) * close_w.at[d, c]
                        diff = alloc - have
                        if diff > 100:  # 买
                            plim = preclose_w.at[d, c] * (0.9 if c[:3] in ("300", "688") else 0.95) if not np.isnan(preclose_w.at[d, c]) else 0
                            if not np.isnan(plim) and o <= plim:
                                continue
                            sh = int(diff / (o * 1.001) // 100 * 100)
                            if sh > 0 and cash >= sh * o * 1.001:
                                cash -= sh * o * 1.001
                                positions[c] = positions.get(c, 0) + sh
                        elif diff < -100:  # 卖
                            sh = int(-diff / (o * 0.999) // 100 * 100)
                            sh = min(sh, positions.get(c, 0))
                            if sh > 0:
                                cash += sh * o * 0.999
                                positions[c] -= sh
                                if positions[c] <= 0:
                                    del positions[c]

        # --- 非调仓日: 个股 MA5 卖出 (T-1 收盘判断, T 日开盘卖出, 资金进 V8) ---
        if state_in and sell_mode == "IND_MA5":
            prev_d = cal_dates[i-1] if i > 0 else None
            for c in list(positions):
                if prev_d is None:
                    continue
                close_t1 = close_w.at[prev_d, c]
                if np.isnan(close_t1):
                    continue
                hist = close_w[c].loc[:prev_d]
                ma5 = hist.tail(5).mean()
                if not np.isnan(ma5) and close_t1 < ma5:
                    o = open_w.at[d, c]
                    if not np.isnan(o) and o > 0:
                        cash += positions[c] * o * 0.999
                        del positions[c]

        # --- 每日收尾: 现金扫入 V8 (不闲置) ---
        reserve += cash
        cash = 0.0

        # NAV
        pos_val = sum(sh * close_w.at[d, c] if not np.isnan(close_w.at[d, c]) else 0
                      for c, sh in positions.items())
        nav = cash + reserve + pos_val
        navs.append(nav)
        portfolio_log.append({"date": d, "nav": nav, "n_pos": len(positions), "state": state_in})

    nav_s = pd.Series(navs)
    if period:
        mask = pd.Series(cal_dates).astype(str).str[:4].astype(int) >= period
        nav_s = nav_s[mask.values]
    tot = nav_s.iloc[-1] / nav_s.iloc[0] - 1.0
    yrs = len(nav_s) / 242.0
    ann = (1 + tot) ** (1 / yrs) - 1 if yrs > 0 else 0
    dd = ((nav_s - nav_s.cummax()) / nav_s.cummax()).min()
    sharpe = (nav_s.pct_change().fillna(0).mean() / (nav_s.pct_change().fillna(0).std() + 1e-8)) * SQRT_242
    log_df = pd.DataFrame(portfolio_log)
    return {"ann": ann, "maxdd": dd, "sharpe": sharpe, "calmar": ann / (-dd + 1e-9),
            "nav": nav_s, "log": log_df}

# ============ 6. 跑 24 组合 + target volatility 变体 ============
print("[4] 回测矩阵...", flush=True)
configs = []
for score_src in ("ENH", "GBDT", "ENS"):
    for top_tag in ("T40", "T60"):
        for sell_mode in ("S123_ONLY", "IND_MA5"):
            for timing in (True, False):
                tag = f"{score_src}_{top_tag}_{sell_mode}_{'S123' if timing else 'ALWAYS'}"
                configs.append((tag, score_src, top_tag, sell_mode, timing))

# target volatility 变体: (tag, score_src, top_tag, sell_mode, timing, tgt_vol, floor_w, vol_lookback)
TV_CONFIGS = [
    ("ENS_T60_S123_TV12", "ENS", "T60", "S123_ONLY", True, 0.12, 0.4, 20),
    ("ENS_T60_S123_TV15", "ENS", "T60", "S123_ONLY", True, 0.15, 0.4, 20),
    ("ENS_T60_S123_TV18", "ENS", "T60", "S123_ONLY", True, 0.18, 0.4, 20),
    ("ENS_T60_S123_TV21", "ENS", "T60", "S123_ONLY", True, 0.21, 0.4, 20),
    ("ENS_T60_S123_TV18_L60", "ENS", "T60", "S123_ONLY", True, 0.18, 0.4, 60),
    ("ENH_T40_S123_TV18", "ENH", "T40", "S123_ONLY", True, 0.18, 0.4, 20),
]

results = {}
for tag, score_src, top_tag, sell_mode, timing in configs:
    res = run_backtest(score_src, top_tag, sell_mode, timing)
    results[tag] = res
    print(f"  {tag:<36} CAGR={res['ann']:>7.2%} MaxDD={res['maxdd']:>7.2%} "
          f"Calmar={res['calmar']:>5.2f} Sharpe={res['sharpe']:>5.2f}", flush=True)

print("[4b] target volatility 变体...", flush=True)
for tag, score_src, top_tag, sell_mode, timing, tgt, fl, lb in TV_CONFIGS:
    res = run_backtest(score_src, top_tag, sell_mode, timing, tgt_vol=tgt, floor_w=fl, vol_lookback=lb)
    results[tag] = res
    print(f"  {tag:<36} CAGR={res['ann']:>7.2%} MaxDD={res['maxdd']:>7.2%} "
          f"Calmar={res['calmar']:>5.2f} Sharpe={res['sharpe']:>5.2f}", flush=True)

# 对照: T7 (ETF s123 3进/1出 + V8)
print("\n[5] ETF 对照 (T7)...", flush=True)
from sector_rotation_traditional import build_signals4, run_graded, TRADITIONAL_ETFS, W_MAP  # noqa: E402
from etf_optimize_backtest2 import build_series, hv_monthly_ret, monthly_from_daily, calc_stats  # noqa: E402
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
print(f"  T7(ETF): CAGR={st7['CAGR']:.2%} MaxDD={st7['MaxDD']:.2%} Calmar={st7['Calmar']:.2f}")

# ============ 7. 汇总输出 ============
print("\n" + "=" * 120)
print(f"{'版本':<36} {'CAGR':>8} {'MaxDD':>8} {'Calmar':>7} {'Sharpe':>7}")
print("-" * 120)
rows = []
for tag, res in results.items():
    rows.append({"版本": tag, "CAGR": res["ann"], "MaxDD": res["maxdd"],
                 "Calmar": res["calmar"], "Sharpe": res["sharpe"]})
    print(f"{tag:<36} {res['ann']:>7.2%} {res['maxdd']:>7.2%} {res['calmar']:>6.2f} {res['sharpe']:>6.2f}")
print(f"{'T7_ETF对照':<36} {st7['CAGR']:>7.2%} {st7['MaxDD']:>7.2%} {st7['Calmar']:>6.2f} {st7['Sharpe']:>6.2f}")
rows.append({"版本": "T7_ETF对照", "CAGR": st7["CAGR"], "MaxDD": st7["MaxDD"],
             "Calmar": st7["Calmar"], "Sharpe": st7["Sharpe"]})
pd.DataFrame(rows).to_csv(os.path.join(OUT_DIR, "stock_gbdt_s123_matrix.csv"),
                          index=False, encoding="utf-8-sig")

# ============ 8. 对比图 (NAV + 回撤 + 年度) ============
KEY = [
    ("ENH_T40_S123_ONLY_S123", "ENH4 T40 s123", "steelblue"),
    ("GBDT_T40_S123_ONLY_S123", "GBDT(C7) T40 s123", "darkorange"),
    ("ENS_T60_S123_ONLY_S123", "ENS T60 s123", "purple"),
    ("ENS_T60_S123_TV18", "ENS T60 s123 +TV18", "crimson"),
]
t7_nav = t7["nav"]  # 月频 NAV (index=ym)
t7_nav.index = [str(i) for i in t7_nav.index]

fig, axes = plt.subplots(3, 1, figsize=(14, 16))
ax = axes[0]
for tag, label, color in KEY:
    nav = results[tag]["nav"]
    ax.plot(nav.index.astype(str), nav / nav.iloc[0], label=label, color=color, lw=1.3)
ax.plot(t7_nav.index, t7_nav / t7_nav.iloc[0], label="ETF原版 T7", color="darkgreen", lw=1.8)
ax.set_title("方案B: 股票 GBDT/ENH4 选股 vs ETF原版 T7 (净值, 2020-2025)")
ax.legend(fontsize=9); ax.grid(alpha=0.3); ax.set_ylabel("NAV")

ax = axes[1]
for tag, label, color in KEY:
    nav = results[tag]["nav"]
    dd = nav / nav.cummax() - 1
    ax.plot(nav.index.astype(str), dd, label=label, color=color, lw=1.3)
t7_dd = t7_nav / t7_nav.cummax() - 1
ax.plot(t7_nav.index, t7_dd, label="ETF原版 T7", color="darkgreen", lw=1.8)
ax.set_title("回撤对比 (股票 32-36% vs ETF 19%)")
ax.legend(fontsize=9); ax.grid(alpha=0.3); ax.set_ylabel("Drawdown")

ax = axes[2]
yrs = {}
def yearly(nav, is_monthly=False):
    idx = [str(i) for i in nav.index]
    out = {}
    for y in sorted(set(i[:4] for i in idx)):
        s = nav[[i.startswith(y) for i in idx]]
        out[y] = s.iloc[-1] / s.iloc[0] - 1
    return out
for tag, label, color in KEY:
    yrs[label] = yearly(results[tag]["nav"])
yrs["ETF原版 T7"] = yearly(t7_nav)
ydf = pd.DataFrame(yrs).reindex(sorted(yrs["ETF原版 T7"]))
ydf.plot(kind="bar", ax=ax)
ax.set_title("年度收益对比")
ax.legend(fontsize=8, ncol=5); ax.grid(alpha=0.3, axis="y")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "stock_vs_etf_compare.png"), dpi=120)
print(f"[图] 已保存: {os.path.join(OUT_DIR, 'stock_vs_etf_compare.png')}")

# ============ 9. 保存结果 (nav + log + T7) 供后续口径分析复用 ============
import pickle
# 为每个组合的 nav 附加日期 index (从 log 的 date 字段)
for tag, res in results.items():
    log = res["log"]
    res["nav_dated"] = pd.Series(log["nav"].values, index=log["date"].values).sort_index()
    res["log"] = None  # 释放内存
with open(os.path.join(OUT_DIR, "stock_gbdt_s123_results.pkl"), "wb") as f:
    pickle.dump({"results": results, "t7": t7}, f)
print(f"[存] 结果已保存: {os.path.join(OUT_DIR, 'stock_gbdt_s123_results.pkl')}")

print(f"\n[完成] 总耗时 {time.time()-t0:.0f}s")

# -*- coding: utf-8 -*-
"""低估板块 + 剔除价值陷阱 → 板块内 GBDT 选股 + s123 择时 回测

验证用户提出的"高收益(选股) + 低回撤(低估板块+板块类型限制)"三步骤方向:
  1. 先获取被低估的板块 (行业 PE 分位 < 30%)
  2. 只选某些类型板块 (剔除金融/地产/公用/基建/交运 价值陷阱)
  3. 板块内用 GBDT 打分选股 (TopN, 行业限数)
  4. s123 择时 (3进/1出) + V8 避险 → 控制回撤

核心: 在 stock_gbdt_s123_backtest.py 的 GBDT 打分引擎之上, 把 is_traditional
粗过滤替换为"低估 + 剔除价值陷阱"精细过滤, 用 A/B 对比验证其增量。

对比变体 (GBDT打分, s123 3进/1出, V8, T40/T60):
  ALL             无板块过滤 (中证1000成分全池)      <- 基线
  TRAD            is_traditional (排除6科技, 保留价值陷阱)  <- 现有T7口径
  UNDERVAL        PE分位<30% 低估板块
  NOTRAP          剔除价值陷阱板块
  UNDERVAL_NOTRAP 低估 + 剔除价值陷阱   <- 用户核心提案
  UNDERVAL_GROWTH 低估 + 仅成长/制造板块  <- 更严格版
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
from etf_optimize_backtest2 import load_hv_daily, load_index_ret  # noqa: E402

OUT_DIR = os.path.join(ROOT, "research", "sector_rotation", "results")
os.makedirs(OUT_DIR, exist_ok=True)
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

t0 = time.time()
COST = 20 / 10000.0          # 双边20bps (买卖各10bps)
SQRT_242 = np.sqrt(242.0)
TOP_N_CHOICES = {"T40": 40, "T60": 60}
MAX_PER_IND = {"T40": 4, "T60": 4}

# PE 分位参数 (记忆: 48月滚动最优, 低估阈值 30%)
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

# ============ 价值陷阱 / 成长制造 板块定义 (110细分行业口径, 对齐 panel.industry) ============
VALUE_TRAP = {
    # 金融
    "银行", "证券", "多元金融", "保险",
    # 地产
    "全国地产", "区域地产", "园区开发", "房产服务",
    # 公用
    "供气供热", "水务", "火力发电", "水力发电", "新型电力",
    # 基建/建筑
    "建筑工程", "装修装饰",
    # 交运
    "港口", "路桥", "公路", "机场", "铁路", "水运", "空运", "公共交通", "仓储物流",
}

GROWTH_MFG = {
    # 汽车
    "汽车整车", "汽车配件", "汽车服务", "摩托车",
    # 机械
    "专用机械", "工程机械", "机床制造", "机械基件", "轻工机械", "纺织机械", "农用机械",
    # IT/软件/电子
    "IT设备", "软件服务", "互联网", "半导体", "元器件", "电器仪表", "通信设备", "电信运营",
    # 新能源/电气
    "电气设备",
}

# ============ 1. 数据加载 ============
print("[1] 加载面板与行情...", flush=True)
panel = pd.read_parquet(os.path.join(ROOT, "research", "sector_rotation", "stock_ml_panel_72m.parquet"))

# 个股 → 行业 / is_traditional (静态, 取末值)
code_to_ind = panel.groupby("ts_code")["industry"].last().to_dict()
code_to_trad = panel.groupby("ts_code")["is_traditional"].last().to_dict()

# 中证1000 成分历史
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
print(f"    s123: >=3占比 {(sig_df['s123']>=3).mean():.1%}, <=1占比 {(sig_df['s123']<=1).mean():.1%}")
print(f"    s123 分布: {sig_df['s123'].value_counts().sort_index().to_dict()}")

v8 = load_hv_daily()
all_dates = sorted(set().union(*[set(s.index) for s in v8.values()]))
v8_df = pd.DataFrame(index=all_dates)
for code, s in v8.items():
    v8_df[code] = s.reindex(all_dates)
v8_daily = (v8_df * pd.Series({"511990.SH": 1/3, "511260.SH": 1/3, "518880.SH": 1/3})).sum(axis=1).fillna(0)
v8_daily.index = v8_daily.index.astype(int)
v8_daily = v8_daily.reindex(cal_dates).fillna(0)
v8_nav_full = (1 + v8_daily).cumprod()

# ============ 3. 行业 PE 分位 (48月滚动) ============
print("[3] 行业 PE 分位...", flush=True)
pe_df = pd.read_csv(os.path.join(OUT_DIR, "industry_pe.csv"), index_col=0)
pe_df.index = pe_df.index.astype(int)
pe_pct = pe_df.rolling(PE_WINDOW, min_periods=24).rank(pct=True)
# 月末 PE 分位日期 → 上一月快照
pe_month_last = {d // 100: d for d in sorted(pe_pct.index)}
print(f"    行业PE {pe_df.shape}, 分位有效 {pe_pct.dropna(how='all').index[0]} 起")

# ============ 4. GBDT / ENH4 打分 (滚动 WFO, 复用 stock_gbdt_s123 逻辑) ============
print("[4] 打分生成 (GBDT 滚动重训)...", flush=True)

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
    p[c] = p.groupby("trade_date")[c].transform(
        lambda s: (s - s.mean()) / (s.std(ddof=1) + 1e-12))
p[FIN_COLS] = p[FIN_COLS].fillna(-99.0)
p["enh4_score"] = (-0.40 * p["ivol"].rank(pct=True) - 0.35 * p["ret_1m"].rank(pct=True)
                   + 0.15 * p["roe"].rank(pct=True) + 0.05 * p["or_yoy"].rank(pct=True)
                   + 0.05 * p["netprofit_yoy"].rank(pct=True))
p["score_enh4"] = p["enh4_score"]
score_enh4 = {d: g.set_index("ts_code")["score_enh4"] for d, g in p.groupby("trade_date")}

from sklearn.linear_model import LinearRegression as _LR  # noqa: E402

def prep_feats(df):
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

GBDT_CACHE = os.path.join(OUT_DIR, "score_gbdt_cache.parquet")
oos_months = [d for d in sorted(panel["trade_date"].unique()) if d >= 20230101]
if os.path.exists(GBDT_CACHE):
    _sg = pd.read_parquet(GBDT_CACHE)
    score_gbdt = {int(m): g.set_index("ts_code")["score"] for m, g in _sg.groupby("month")}
else:
    score_gbdt = {}
    for i, m in enumerate(oos_months):
        tr = prep_feats(panel[panel["trade_date"] < m]).sort_values("trade_date")
        X, y = tr[GBDT_FEATS].values, tr["fwd_20"].values
        val_months = sorted(tr["trade_date"].unique())[-3:]
        vm = tr["trade_date"].isin(val_months).values
        mdl = lgb.LGBMRegressor(n_estimators=500, learning_rate=0.05, num_leaves=7,
                                max_depth=3, min_child_samples=80, reg_lambda=2.0, reg_alpha=0.1,
                                subsample=0.9, colsample_bytree=0.9, random_state=42, verbose=-1)
        mdl.fit(X[~vm], y[~vm], eval_set=[(X[vm], y[vm])],
                callbacks=[lgb.early_stopping(50, verbose=False)])
        om = prep_feats(panel[panel["trade_date"] == m])
        score_gbdt[m] = pd.Series(mdl.predict(om[GBDT_FEATS]), index=om["ts_code"])
        if (i + 1) % 12 == 0:
            print(f"    GBDT 重训 {i+1}/{len(oos_months)}, 耗时{time.time()-t0:.0f}s", flush=True)
    _rows = []
    for m, s in score_gbdt.items():
        _rows.append(pd.DataFrame({"month": [int(m)] * len(s), "ts_code": s.index, "score": s.values}))
    pd.concat(_rows, ignore_index=True).to_parquet(GBDT_CACHE)
    print(f"    GBDT 打分已缓存", flush=True)

# 2023 前用 ENH4 填充
for d in sorted(panel["trade_date"].unique()):
    if d not in score_gbdt:
        score_gbdt[d] = score_enh4[d]
print(f"    GBDT 打分完成 {len(score_gbdt)} 月, 耗时{time.time()-t0:.0f}s")

# 因子增强快照: 每个交易日的横截面动量/alpha rank (用于选股增强)
ENH_FACTORS = ["momentum_20", "alpha_012", "momentum_60"]
factor_snap = {}
for d, g in p.groupby("trade_date"):
    factor_snap[d] = g.set_index("ts_code")[ENH_FACTORS].rank(pct=True)
print(f"    因子增强快照 {len(factor_snap)} 日")

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
    """pool: Series(ts_code->score). 返回过滤后的 pool"""
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

# ============ 6. 日频回测引擎 (s123 3进/1出 + V8 / target-vol 避险) ============
MM_ANN = 0.02
MM_DAILY = (1 + MM_ANN) ** (1 / 242) - 1

def compute_tvol_w(port_ret_hist, tgt_vol, floor_w, cap=1.0):
    """组合近 lookback 日收益 -> 目标风险仓位 w = clip(tgt_vol/vol_ann, floor_w, cap)"""
    ret = np.asarray(port_ret_hist, dtype=float)
    vol_d = ret.std()
    if not np.isfinite(vol_d) or vol_d <= 0:
        return 1.0
    w = tgt_vol / (vol_d * SQRT_242)
    return float(np.clip(w, floor_w, cap))

# trailing drawdown 熔断档位: 组合回撤越深, 风险仓位越低
DD_BREAKS = [(-0.05, 1.0), (-0.10, 0.75), (-0.15, 0.50), (-0.20, 0.25), (-np.inf, 0.0)]

def dd_scale(dd):
    """组合当前回撤 -> 风险仓位乘数 (0-1)"""
    for thr, w in DD_BREAKS:
        if dd >= thr:
            return w
    return 0.0

def enhance_pool(pool, snap, enhance, mom_w, alpha_w):
    """在 GBDT 打分基础上叠加动量/alpha 因子增强 (横截面 rank)"""
    if not enhance or snap not in factor_snap:
        return pool
    f = factor_snap[snap].reindex(pool.index)
    gbdt_z = (pool - pool.mean()) / (pool.std(ddof=1) + 1e-12)
    out = gbdt_z.copy()
    if enhance in ("MOM", "MA"):
        out = out + mom_w * f["momentum_20"].fillna(0.5)
    if enhance == "MA":
        out = out + alpha_w * f["alpha_012"].fillna(0.5)
    return out

def run_backtest(filter_name, top_n, max_ind, vol_tgt=None, vol_floor=0.5,
                 vol_lookback=60, hedge="v8", enhance=None, mom_w=0.0, alpha_w=0.0,
                 dd_break=False, entry_th=3, exit_th=1, staged=False):
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
        # 板块 PE 分位快照 (上一月末)
        pe_snap = pe_month_last.get(prev_ym)
        sector_pct_snap = pe_pct.loc[pe_snap] if pe_snap is not None else pd.Series(dtype=float)
        pool = apply_sector_filter(pool, filter_name, sector_pct_snap)
        return enhance_pool(pool, snap, enhance, mom_w, alpha_w)

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
    in_days = 0
    expo_sum = 0.0
    for i, d in enumerate(cal_dates):
        ym = d // 100
        if d == rebals[0]:
            prev_s123 = sig_map.get(ym, 0)
        if i > 0 and cal_dates[i-1] // 100 != ym:
            prev_s123 = sig_map.get(cal_dates[i-1] // 100, 0)

        if prev_s123 is None:
            target_state = False
            w_s123 = 0.0
        else:
            if staged:
                # 阶梯仓位: s123=3满仓/2半仓/1观望/0清仓 (提高资金利用率)
                w_s123 = {3: 1.0, 2: 0.5, 1: 0.0, 0: 0.0}.get(prev_s123, 0.0)
                target_state = w_s123 > 0
            else:
                if not state_in and prev_s123 >= entry_th:
                    target_state = True
                elif state_in and prev_s123 <= exit_th:
                    target_state = False
                else:
                    target_state = state_in
                w_s123 = 1.0 if target_state else 0.0

        reserve *= (1 + hedge_daily.at[d])

        if d in rebals:
            if dd_break and navs:
                dd_now = navs[-1] / nav_peak - 1.0
                dd_mult = dd_scale(dd_now)
            else:
                dd_mult = 1.0
            if target_state and not state_in:
                pool = pool_at(d)
                if pool is not None and len(pool) > 0:
                    sel = select_with_limit(pool, max_ind, top_n)
                    equity = cash + reserve
                    if vol_tgt is not None and len(port_ret_hist) >= 20:
                        w_risk = compute_tvol_w(port_ret_hist[-vol_lookback:], vol_tgt, vol_floor)
                    else:
                        w_risk = 1.0
                    stock_budget = equity * w_risk * w_s123 * dd_mult
                    reserve = equity - stock_budget
                    cash = stock_budget
                    positions = {}
                    alloc = stock_budget / len(sel) if len(sel) else 0
                    for c in sel:
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
                    equity = cash + reserve + sum(sh * close_w.at[d, c] if not np.isnan(close_w.at[d, c]) else 0
                                                  for c, sh in positions.items())
                    if vol_tgt is not None and len(port_ret_hist) >= 20:
                        w_risk = compute_tvol_w(port_ret_hist[-vol_lookback:], vol_tgt, vol_floor)
                    else:
                        w_risk = 1.0
                    target_stock = equity * w_risk * w_s123 * dd_mult
                    for c in list(positions):
                        if c not in sel:
                            o = open_w.at[d, c]
                            if not np.isnan(o) and o > 0:
                                cash += positions[c] * o * 0.999
                            del positions[c]
                    alloc = target_stock / len(sel) if len(sel) else 0
                    for c in sel:
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

# ============ 7. 跑变体矩阵 ============
print("\n[5] 回测矩阵 (板块过滤 × TopN × 避险)...", flush=True)
FILTERS = ["ALL", "TRAD", "UNDERVAL", "NOTRAP", "UNDERVAL_NOTRAP", "UNDERVAL_GROWTH"]
HEDGES = [("", None, 0.5, "v8"), ("_MM", None, 0.5, "mm"),
          ("_VT7", 0.07, 0.5, "mm"), ("_VT6", 0.06, 0.5, "mm"),
          ("_V8VT7", 0.07, 0.5, "v8"), ("_V8VT6", 0.06, 0.5, "v8")]
results = {}
for fname in FILTERS:
    for top_tag, top_n in TOP_N_CHOICES.items():
        for htag, vt, vf, hg in HEDGES:
            is_core = fname in ("UNDERVAL_NOTRAP", "UNDERVAL")
            if not is_core and htag not in ("", "_VT7"):
                continue
            tag = f"{fname}_{top_tag}{htag}"
            res = run_backtest(fname, top_n, MAX_PER_IND[top_tag], vol_tgt=vt, vol_floor=vf, hedge=hg)
            results[tag] = res
            print(f"  {tag:<28} CAGR={res['ann']:>7.2%} MaxDD={res['maxdd']:>7.2%} "
                  f"Sharpe={res['sharpe']:>5.2f} Calmar={res['calmar']:>5.2f} Final={res['final']:>5.2f}", flush=True)

# ============ 7b. 强化探索矩阵 (收益: 因子增强+集中度; 回撤: trailing drawdown 熔断) ============
print("\n[6] 强化探索 (增强×集中度×熔断, 基于 V8VT6)...", flush=True)
# 每条: (tag后缀, enhance, mom_w, alpha_w, max_ind, dd_break)
ENH_TESTS = [
    ("",            None,  0.0, 0.0, 4,  False),   # 基线 V8VT6 (复用主矩阵)
    ("_MOM05",      "MOM", 0.5, 0.0, 4,  False),   # 动量增强 0.5
    ("_MOM10",      "MOM", 1.0, 0.0, 4,  False),   # 动量增强 1.0
    ("_MA",         "MA",  0.5, 0.3, 4,  False),   # 动量+alpha 双增强
    ("_MI8",        None,  0.0, 0.0, 8,  False),   # 集中度放宽 8
    ("_NL",         None,  0.0, 0.0, 99, False),   # 无行业限制
    ("_MOM05_MI8",  "MOM", 0.5, 0.0, 8,  False),   # 增强+集中度
    ("_DB",         None,  0.0, 0.0, 4,  True),    # 熔断(基线+熔断)
    ("_MI8_DB",     None,  0.0, 0.0, 8,  True),    # 集中度8+熔断
    ("_NL_DB",      None,  0.0, 0.0, 99, True),    # 无限制+熔断
    ("_MOM05_MI8_DB", "MOM", 0.5, 0.0, 8, True),   # 最优收益+熔断
]
for fname in ("UNDERVAL", "UNDERVAL_NOTRAP"):
    for top_tag, top_n in TOP_N_CHOICES.items():
        for tag, enh, mw, aw, mi, ddb in ENH_TESTS:
            if tag == "":
                continue
            full_tag = f"{fname}_{top_tag}_V8VT6{tag}"
            res = run_backtest(fname, top_n, mi, vol_tgt=0.06, vol_floor=0.5, hedge="v8",
                               enhance=enh, mom_w=mw, alpha_w=aw, dd_break=ddb)
            results[full_tag] = res
            print(f"  {full_tag:<32} CAGR={res['ann']:>7.2%} MaxDD={res['maxdd']:>7.2%} "
                  f"Sharpe={res['sharpe']:>5.2f} Calmar={res['calmar']:>5.2f} Final={res['final']:>5.2f}", flush=True)

# ============ 7c. 资金利用率探索矩阵 (收益层: 放宽 s123 择时门槛, 基于 V8VT6+NL) ============
print("\n[7] 资金利用率探索 (s123 择时门槛, 基于 V8VT6+NL)...", flush=True)
# 每条: (tag后缀, entry_th, exit_th, staged)
UTIL_TESTS = [
    ("",        3, 1, False),   # 基线 3进1出
    ("_E2",     2, 1, False),   # 放宽进场: 2进1出
    ("_X0",     3, 0, False),   # 放宽离场: 3进0出
    ("_E2X0",   2, 0, False),   # 双放宽: 2进0出
    ("_STAGE",  3, 1, True),    # 阶梯仓位: 3满/2半/1清
]
for fname in ("UNDERVAL",):
    for top_tag, top_n in TOP_N_CHOICES.items():
        for tag, eth, xth, stg in UTIL_TESTS:
            full_tag = f"{fname}_{top_tag}_V8VT6_NL{tag}"
            res = run_backtest(fname, top_n, 99, vol_tgt=0.06, vol_floor=0.5, hedge="v8",
                               entry_th=eth, exit_th=xth, staged=stg)
            results[full_tag] = res
            print(f"  {full_tag:<32} CAGR={res['ann']:>7.2%} MaxDD={res['maxdd']:>7.2%} "
                  f"Sharpe={res['sharpe']:>5.2f} Calmar={res['calmar']:>5.2f} "
                  f"占用={res['occupancy']:>6.1%} 敞口={res['avg_expo']:>5.2f}", flush=True)

# ============ 8. 汇总输出 ============
print("\n" + "=" * 100)
print(f"{'变体':<30} {'CAGR':>8} {'MaxDD':>8} {'Sharpe':>7} {'Calmar':>7} {'Final':>7}")
print("-" * 100)
rows = []
for tag, res in results.items():
    rows.append({"变体": tag, "CAGR": res["ann"], "MaxDD": res["maxdd"],
                 "Sharpe": res["sharpe"], "Calmar": res["calmar"], "Final": res["final"]})
    print(f"{tag:<30} {res['ann']:>7.2%} {res['maxdd']:>7.2%} {res['sharpe']:>6.2f} {res['calmar']:>6.2f} {res['final']:>6.2f}")
pd.DataFrame(rows).to_csv(os.path.join(OUT_DIR, "undervalued_sector_stock_matrix.csv"),
                          index=False, encoding="utf-8-sig")

# ============ 9. 对比图 ============
def yearly(nav):
    idx = [str(i) for i in nav.index]
    out = {}
    for y in sorted(set(i[:4] for i in idx)):
        s = nav[[i.startswith(y) for i in idx]]
        out[y] = s.iloc[-1] / s.iloc[0] - 1
    return out

# 主图: T40 各过滤变体 NAV + 回撤
key_t40 = [
    ("ALL_T40", "ALL 无过滤", "gray"),
    ("TRAD_T40", "TRAD 传统(现有)", "steelblue"),
    ("UNDERVAL_T40", "UNDERVAL 低估", "darkorange"),
    ("NOTRAP_T40", "NOTRAP 剔除价值陷阱", "green"),
    ("UNDERVAL_NOTRAP_T40", "UNDERVAL+NOTRAP 核心", "crimson"),
    ("UNDERVAL_GROWTH_T40", "UNDERVAL+GROWTH 严格", "purple"),
]
fig, axes = plt.subplots(2, 1, figsize=(14, 10))
for tag, label, color in key_t40:
    nav = results[tag]["nav"]
    axes[0].plot(nav.index.astype(str), nav / nav.iloc[0], label=f"{label}", color=color, lw=1.4)
axes[0].set_title("低估+剔除价值陷阱板块 × GBDT选股 (T40, s123 3进/1出, 2020-2025)")
axes[0].legend(fontsize=9); axes[0].grid(alpha=0.3); axes[0].set_ylabel("NAV")
for tag, label, color in key_t40:
    nav = results[tag]["nav"]
    dd = nav / nav.cummax() - 1
    axes[1].plot(nav.index.astype(str), dd, label=f"{label}", color=color, lw=1.4)
axes[1].set_title("回撤对比")
axes[1].legend(fontsize=9); axes[1].grid(alpha=0.3); axes[1].set_ylabel("Drawdown")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "undervalued_sector_stock_curve.png"), dpi=120)
print(f"[图] undervalued_sector_stock_curve.png")

# 年度收益
fig, ax = plt.subplots(figsize=(14, 6))
yrs = {}
for tag, label, color in key_t40:
    yrs[label] = yearly(results[tag]["nav"])
ydf = pd.DataFrame(yrs).reindex(sorted(yrs[key_t40[0][1]]))
ydf.plot(kind="bar", ax=ax)
ax.set_title("年度收益对比 (T40)")
ax.legend(fontsize=8, ncol=3); ax.grid(alpha=0.3, axis="y")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "undervalued_sector_stock_yearly.png"), dpi=120)
print(f"[图] undervalued_sector_stock_yearly.png")

# target-vol 替换 V8 对比图 (核心 filter 的避险变体)
key_hedge = [
    ("UNDERVAL_NOTRAP_T40", "UNDERVAL+NOTRAP V8避险", "crimson"),
    ("UNDERVAL_NOTRAP_T40_MM", "UNDERVAL+NOTRAP 货基避险", "darkorange"),
    ("UNDERVAL_NOTRAP_T40_VT7", "UNDERVAL+NOTRAP VolTarget7%", "darkgreen"),
    ("UNDERVAL_NOTRAP_T40_VT6", "UNDERVAL+NOTRAP VolTarget6%", "navy"),
]
fig, axes = plt.subplots(2, 1, figsize=(14, 10))
for tag, label, color in key_hedge:
    nav = results[tag]["nav"]
    axes[0].plot(nav.index.astype(str), nav / nav.iloc[0], label=label, color=color, lw=1.4)
axes[0].set_title("target-vol 替换 V8 对比 (UNDERVAL+NOTRAP T40, s123 3进/1出)")
axes[0].legend(fontsize=9); axes[0].grid(alpha=0.3); axes[0].set_ylabel("NAV")
for tag, label, color in key_hedge:
    nav = results[tag]["nav"]
    dd = nav / nav.cummax() - 1
    axes[1].plot(nav.index.astype(str), dd, label=label, color=color, lw=1.4)
axes[1].set_title("回撤对比 (target-vol vs V8)")
axes[1].legend(fontsize=9); axes[1].grid(alpha=0.3); axes[1].set_ylabel("Drawdown")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "undervalued_sector_stock_targetvol.png"), dpi=120)
print(f"[图] undervalued_sector_stock_targetvol.png")

# 结论
rows_sorted = pd.DataFrame(rows).sort_values("Calmar", ascending=False)
best = rows_sorted.iloc[0]
print("\n=== 按 Calmar 排序 Top5 ===")
print(rows_sorted.head(5).round(4).to_string(index=False))
print(f"\n[完成] 总耗时 {time.time()-t0:.0f}s")

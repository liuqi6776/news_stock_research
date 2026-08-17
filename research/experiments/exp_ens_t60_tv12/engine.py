# -*- coding: utf-8 -*-
"""ENS_T60_TV12 参数族轻量回测引擎（PBO/CSCV + purged CV 实验用）

从 stock_gbdt_s123_backtest.py 提取核心逻辑并函数化，仅保留冻结版配置的搜索空间：
  - 打分: ENH4 / GBDT / ENS（0.5×ENH4秩 + 0.5×GBDT秩）
  - 持仓: T40 / T60（行业<=4）
  - target volatility: tgt_vol × floor_w × vol_lookback
固定为冻结版其余维度: s123 择时 + S123_ONLY 卖出 + V8 避险 + 双边 20bps。

共享计算（panel / s123 / V8 / 打分）只做一次, 每个配置复用 → 216 配置可并行/顺序快速回测。
用法: 本模块只负责引擎, 不直接运行; 由 pbo_cscv.py / 冒烟测试调用。
"""
import glob
import os
import sys

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.linear_model import LinearRegression as _LR

ROOT = r"c:\Users\liuqi\quant_system_v2"
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "research", "fund_research", "studies", "rotation_dingtou"))
sys.path.insert(0, os.path.join(ROOT, "research", "sector_rotation"))

from timing_dingtou import fetch_pe_csi300, fetch_bond10y, _rolling_pct, _zscore  # noqa: E402
from etf_optimize_backtest2 import load_hv_daily, load_index_ret  # noqa: E402
from industry_l1 import build_l1_map  # noqa: E402

SQRT_242 = np.sqrt(242.0)
TOP_N_CHOICES = {"T40": 40, "T60": 60}
MAX_PER_IND = {"T40": 4, "T60": 4}

PRICE_COLS = ["ret_1m", "ivol", "momentum_5", "momentum_10", "momentum_20", "momentum_60",
              "volatility_5", "volatility_10", "volatility_20",
              "alpha_006", "alpha_009", "alpha_012", "alpha_023"]
FIN_COLS = ["roe", "or_yoy", "netprofit_yoy"]
CHIP_COLS = ["vwap_20", "float_pnl_20", "chip_shift_5"]
CHIP_BASE = ["ivol", "ret_1m", "momentum_20", "volatility_20", "alpha_006", "alpha_012"]
CHIP_RESID_COLS = ["vwap_20_resid", "float_pnl_20_resid", "chip_shift_5_resid"]
GBDT_FEATS = ["ivol", "ret_1m", "momentum_20", "volatility_20", "alpha_006", "alpha_012",
              "enh4_score", "vwap_20_resid", "float_pnl_20_resid", "chip_shift_5_resid"]

PANEL_PATH = os.path.join(ROOT, "research", "sector_rotation", "stock_ml_panel_72m.parquet")
FULLMARKET_PANEL_PATH = os.path.join(ROOT, "research", "sector_rotation",
                                     "stock_ml_panel_fullmarket_72m.parquet")
IND_MAP_PATH = os.path.join(ROOT, "research", "studies", "study_008_enhancements",
                            "data", "industry_map.parquet")
PX_DIR = r"D:/iquant_data/data_v2/data_day1"
IW_GLOB = r"D:/iquant_data/data_v2/index_weight/*.parquet"


def winsorize(s, lo=0.01, hi=0.99):
    if s.notna().sum() < 5:
        return s
    a, b = s.quantile([lo, hi])
    return s.clip(a, b)


def prep_feats(df, feats):
    """逐字复刻 stock_gbdt_s123_backtest.prep_feats（C8: 价量+ENH4+3残差筹码）。"""
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


def init_shared(universe="csi1000"):
    """一次性加载全部共享数据, 返回 dict（打分字典 + 行情宽表 + s123 + V8）。

    universe: "csi1000"=中证1000成分(冻结基线) | "fullmarket"=全市场(5869只, 流动性/停牌已过滤)。
    """
    full = (universe == "fullmarket")
    panel = pd.read_parquet(FULLMARKET_PANEL_PATH if full else PANEL_PATH)

    if full:
        # 全市场面板已内嵌 industry(110通达信细分), 直接取最新快照去重
        latest_ind = panel.drop_duplicates("ts_code", keep="last")
        ind_map = dict(zip(latest_ind["ts_code"], latest_ind["industry"]))
    else:
        im = pd.read_parquet(IND_MAP_PATH)
        ind_map = dict(zip(im["ts_code"], im["industry"]))
    ind_l1_map = build_l1_map(ind_map)

    panel_codes = set(panel["ts_code"].unique())

    if full:
        def latest_members(rebal_d):
            return panel_codes
    else:
        iw = pd.concat([pd.read_parquet(f) for f in glob.glob(IW_GLOB)], ignore_index=True)
        iw = iw[iw["index_code"] == "000852.SH"]
        iw["iw_date"] = iw["trade_date"].astype(int)
        iw_dates = sorted(iw["iw_date"].unique())
        iw_by_date = {d: set(g["con_code"]) for d, g in iw.groupby("iw_date")}

        def latest_members(rebal_d):
            for d in reversed(iw_dates):
                if d <= rebal_d:
                    return iw_by_date[d]
            return set()
    px_parts = []
    for f in sorted(glob.glob(os.path.join(PX_DIR, "*.parquet"))):
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
    cal_dates = sorted(ret_w.index)

    # s123 信号（月末算, 下月生效）
    pe = fetch_pe_csi300()
    bond = fetch_bond10y()
    close_ix = pe["close"]
    dd_ix = close_ix / close_ix.cummax() - 1.0
    erp = 1.0 / pe["pe_ttm"] - bond["y10"].reindex(pe.index).ffill()
    sig_rows = []
    for ym in sorted(set(d // 100 for d in cal_dates)):
        d = pd.Timestamp(f"{ym}01") + pd.offsets.MonthEnd(0)
        s1 = 1 if _rolling_pct(pe["pe_ttm"], d) < 0.20 else 0
        s2 = 1 if _zscore(erp, d) > 1.0 else 0
        s3 = 1 if float(dd_ix.asof(d)) <= -0.25 else 0
        sig_rows.append({"ym": ym, "s123": s1 + s2 + s3})
    sig_df = pd.DataFrame(sig_rows).set_index("ym")

    # V8 避险日收益
    v8 = load_hv_daily()
    all_dates = sorted(set().union(*[set(s.index) for s in v8.values()]))
    v8_df = pd.DataFrame(index=all_dates)
    for code, s in v8.items():
        v8_df[code] = s.reindex(all_dates)
    v8_daily = (v8_df * pd.Series({"511990.SH": 1/3, "511260.SH": 1/3, "518880.SH": 1/3})).sum(axis=1).fillna(0)
    v8_daily.index = v8_daily.index.astype(int)
    v8_daily = v8_daily.reindex(cal_dates).fillna(0)

    # ---- 打分生成 ----
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
    score_enh4 = {d: g.set_index("ts_code")["enh4_score"] for d, g in p.groupby("trade_date")}

    # GBDT 滚动重训（2023+ 每月, 与回测口径一致）
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
    for d in sorted(panel["trade_date"].unique()):
        if d not in score_gbdt:
            score_gbdt[d] = score_enh4[d]

    score_ens = {}
    for d in sorted(panel["trade_date"].unique()):
        e, g = score_enh4[d], score_gbdt[d]
        common = e.index.intersection(g.index)
        score_ens[d] = 0.5 * e[common].rank(pct=True) + 0.5 * g[common].rank(pct=True)

    # 调仓日: 每月首个交易日; 打分用上月末收盘快照
    rebals = [min(d for d in cal_dates if d // 100 == ym)
              for ym in sorted(set(d // 100 for d in cal_dates))]
    month_last_map = {d // 100: d for d in sorted(panel["trade_date"].unique())}

    ma20_w_daily = build_ma20_w(cal_dates)

    return {
        "panel": panel, "ind_map": ind_map, "ind_l1_map": ind_l1_map,
        "latest_members": latest_members,
        "ret_w": ret_w, "close_w": close_w, "open_w": open_w, "preclose_w": preclose_w,
        "cal_dates": cal_dates, "sig_df": sig_df, "v8_daily": v8_daily,
        "scores": {"ENH": score_enh4, "GBDT": score_gbdt, "ENS": score_ens},
        "rebals": rebals, "month_last_map": month_last_map,
        "ma20_w_daily": ma20_w_daily,
    }


def build_vol_signal(shared, vol_lookback):
    ix_ret = load_index_ret("000852.SH")
    ix_ret.index = ix_ret.index.astype(int)
    cal = shared["cal_dates"]
    ix_ret = ix_ret.reindex(cal).ffill().fillna(0.0)
    ix_vol = ix_ret.rolling(vol_lookback).std() * SQRT_242
    return ix_vol.shift(1)


def build_ma20_w(cal_dates, deep=0.98, window=20):
    """中证1000(000852) MA20 三档日频仓位（T-1 信号、T 日生效）。

    NAV vs MA20（NAV 与 close 等价线性缩放）:
      close >= MA20          -> 1.0
      deep*MA20 <= close < MA20 -> 0.5
      close <  deep*MA20     -> 0.0
    与 risk_control 线 risk_control_bt.py 的 MA20 三档一致。
    """
    r = load_index_ret("000852.SH")
    r.index = r.index.astype(int)
    r = r.sort_index()
    nav = (1 + r).cumprod()
    ma = nav.rolling(window).mean()
    close_1 = nav.shift(1)
    ma_1 = ma.shift(1)
    w = pd.Series(1.0, index=nav.index)
    below = close_1 < ma_1
    deep_below = close_1 < ma_1 * deep
    w[below & ~deep_below] = 0.5
    w[deep_below] = 0.0
    return w.reindex(cal_dates).ffill().fillna(1.0)


def run_backtest(shared, score_src, top_tag, tgt_vol=None, floor_w=0.4, vol_lookback=20,
                 cap_ind_l1=None, log_holdings=False):
    """日频回测（s123 择时 + S123_ONLY 卖出 + V8 避险 + 双边20bps）。

    cap_ind_l1: 单申万一级行业 <= cap_ind_l1（占股票端权重, 等价于每级 <= int(top_n*cap) 只）,
                None=不约束（冻结版基线）。
    log_holdings: True 时返回 (nav_s, monthly, {调仓日: [持仓ts_code列表]}) 三元素, 否则两元素。
    """
    top_n = TOP_N_CHOICES[top_tag]
    max_ind = MAX_PER_IND[top_tag]
    max_per_ind_l1 = int(top_n * cap_ind_l1) if cap_ind_l1 is not None else None
    scores = shared["scores"][score_src]
    cal_dates = shared["cal_dates"]
    rebals = shared["rebals"]
    month_last_map = shared["month_last_map"]
    latest_members = shared["latest_members"]
    ind_map = shared["ind_map"]
    ind_l1_map = shared["ind_l1_map"]
    panel = shared["panel"]
    ret_w, close_w, open_w, preclose_w = shared["ret_w"], shared["close_w"], shared["open_w"], shared["preclose_w"]
    v8_daily = shared["v8_daily"]
    sig_map = shared["sig_df"]["s123"].to_dict()
    vol_sig = build_vol_signal(shared, vol_lookback) if tgt_vol is not None else None

    def rebal_scores(d):
        y = d // 10000
        m = (d // 100) % 100
        prev_ym = (y - 1) * 100 + 12 if m == 1 else y * 100 + (m - 1)
        snap = month_last_map.get(prev_ym)
        if snap is None:
            return None
        pool = scores.get(snap)
        if pool is None:
            return None
        trad_codes = set(panel.loc[(panel["trade_date"] == snap) & (panel["is_traditional"]), "ts_code"])
        members = latest_members(d)
        return pool[pool.index.isin(members) & pool.index.isin(trad_codes)]

    def tgt_w(d):
        if tgt_vol is None:
            return 1.0
        v = vol_sig.get(d, np.nan)
        if not np.isfinite(v) or v <= 0:
            return 1.0
        return float(np.clip(tgt_vol / v, floor_w, 1.0))

    def select_with_limit(scores_in):
        scores_in = scores_in.dropna()
        sorted_codes = scores_in.sort_values(ascending=False)
        selected, ind_count, l1_count = [], {}, {}
        for code in sorted_codes.index:
            ind = ind_map.get(code, "其他")
            if ind_count.get(ind, 0) >= max_ind:
                continue
            if max_per_ind_l1 is not None:
                l1 = ind_l1_map.get(code, "其他")
                if l1_count.get(l1, 0) >= max_per_ind_l1:
                    continue
            selected.append(code)
            ind_count[ind] = ind_count.get(ind, 0) + 1
            if max_per_ind_l1 is not None:
                l1 = ind_l1_map.get(code, "其他")
                l1_count[l1] = l1_count.get(l1, 0) + 1
            if len(selected) >= top_n:
                break
        return selected

    state_in = False
    positions = {}
    cash = 0.0
    reserve = 1.0e6
    navs = []
    holdings_log = {}
    prev_s123 = None
    for i, d in enumerate(cal_dates):
        ym = d // 100
        if d == rebals[0]:
            prev_s123 = sig_map.get(ym, 0)
        if i > 0 and cal_dates[i-1] // 100 != ym:
            prev_s123 = sig_map.get(cal_dates[i-1] // 100, 0)

        target_state = False
        if prev_s123 is None:
            target_state = False
        else:
            if not state_in and prev_s123 >= 3:
                target_state = True
            elif state_in and prev_s123 <= 1:
                target_state = False
            else:
                target_state = state_in

        reserve *= (1 + v8_daily.at[d])

        if d in rebals:
            if target_state and not state_in:
                pool = rebal_scores(d)
                if pool is not None:
                    sel = select_with_limit(pool)
                    if log_holdings:
                        holdings_log[d] = list(sel)
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
                        plim = preclose_w.at[d, c] * (0.8 if c[:3] in ("300", "688") else 0.9) if not np.isnan(preclose_w.at[d, c]) else 0
                        if not np.isnan(plim) and o <= plim * 1.0:
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
            elif target_state and state_in:
                pool = rebal_scores(d)
                if pool is not None:
                    sel = select_with_limit(pool)
                    if log_holdings:
                        holdings_log[d] = list(sel)
                    equity = cash + reserve + sum(sh * close_w.at[d, c] if not np.isnan(close_w.at[d, c]) else 0
                                                  for c, sh in positions.items())
                    w = tgt_w(d)
                    target_stock = equity * w
                    for c in list(positions):
                        if c not in sel:
                            o = open_w.at[d, c]
                            if not np.isnan(o) and o > 0:
                                cash += positions[c] * o * 0.999
                            del positions[c]
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

        pos_val = sum(sh * close_w.at[d, c] if not np.isnan(close_w.at[d, c]) else 0
                      for c, sh in positions.items())
        nav = cash + reserve + pos_val
        navs.append(nav)
        reserve += cash
        cash = 0.0

    nav_s = pd.Series(navs, index=pd.Index(cal_dates, name="trade_date"))
    # 月频收益（月末净值 pct_change）
    nav_m = nav_s.groupby((nav_s.index // 100).astype(str)).last()
    monthly = nav_m.pct_change().dropna()
    if log_holdings:
        return nav_s, monthly, holdings_log
    return nav_s, monthly


def run_backtest_tiered(shared, score_src, top_tag, tgt_vol=None, floor_w=0.4, vol_lookback=20,
                        cap_ind_l1=None, timing_mode="tiered", dd_degrade=None,
                        dd_degrade_scale=0.5, log_holdings=False):
    """权重梯度版回测（杠杆二）。

    timing_mode:
      - 'binary':    滞回二元开关（与 run_backtest 冻结基线一致: s123>=3 满仓 / <=1 清仓 / =2 维持）
      - 'tiered':    s123 三档梯度（>=3 → 1.0, ==2 → 0.5, <=1 → 0.0），无滞回
      - 'ma20':      纯 MA20 三档（中证1000 NAV vs MA20 → 1.0/0.5/0），无 s123 门槛, 始终在场
      - 's123_ma20': s123 三档 × MA20 三档 相乘合成（sw*mw ∈ {0,0.25,0.5,1.0}）
    dd_degrade: 组合自身回撤阈值（负值, 如 -0.10），用 T-1 收盘净值算回撤, 触发时仓位 × dd_degrade_scale（硬降档）。
    dd_degrade_scale: 触发降档后的仓位保留比例（0.5=减半, 0.3=减到三成）。
    """
    top_n = TOP_N_CHOICES[top_tag]
    max_ind = MAX_PER_IND[top_tag]
    max_per_ind_l1 = int(top_n * cap_ind_l1) if cap_ind_l1 is not None else None
    scores = shared["scores"][score_src]
    cal_dates = shared["cal_dates"]
    rebals = shared["rebals"]
    month_last_map = shared["month_last_map"]
    latest_members = shared["latest_members"]
    ind_map = shared["ind_map"]
    ind_l1_map = shared["ind_l1_map"]
    panel = shared["panel"]
    ret_w, close_w, open_w, preclose_w = shared["ret_w"], shared["close_w"], shared["open_w"], shared["preclose_w"]
    v8_daily = shared["v8_daily"]
    sig_map = shared["sig_df"]["s123"].to_dict()
    vol_sig = build_vol_signal(shared, vol_lookback) if tgt_vol is not None else None

    def rebal_scores(d):
        y = d // 10000
        m = (d // 100) % 100
        prev_ym = (y - 1) * 100 + 12 if m == 1 else y * 100 + (m - 1)
        snap = month_last_map.get(prev_ym)
        if snap is None:
            return None
        pool = scores.get(snap)
        if pool is None:
            return None
        trad_codes = set(panel.loc[(panel["trade_date"] == snap) & (panel["is_traditional"]), "ts_code"])
        members = latest_members(d)
        return pool[pool.index.isin(members) & pool.index.isin(trad_codes)]

    def tgt_w(d):
        if tgt_vol is None:
            return 1.0
        v = vol_sig.get(d, np.nan)
        if not np.isfinite(v) or v <= 0:
            return 1.0
        return float(np.clip(tgt_vol / v, floor_w, 1.0))

    def select_with_limit(scores_in):
        scores_in = scores_in.dropna()
        sorted_codes = scores_in.sort_values(ascending=False)
        selected, ind_count, l1_count = [], {}, {}
        for code in sorted_codes.index:
            ind = ind_map.get(code, "其他")
            if ind_count.get(ind, 0) >= max_ind:
                continue
            if max_per_ind_l1 is not None:
                l1 = ind_l1_map.get(code, "其他")
                if l1_count.get(l1, 0) >= max_per_ind_l1:
                    continue
            selected.append(code)
            ind_count[ind] = ind_count.get(ind, 0) + 1
            if max_per_ind_l1 is not None:
                l1 = ind_l1_map.get(code, "其他")
                l1_count[l1] = l1_count.get(l1, 0) + 1
            if len(selected) >= top_n:
                break
        return selected

    positions = {}
    cash = 0.0
    reserve = 1.0e6
    navs = []
    holdings_log = {}
    prev_s123 = None
    state_in = False
    last_nav = 1.0e6
    peak_nav = 1.0e6
    ma20_w_daily = shared["ma20_w_daily"]

    for i, d in enumerate(cal_dates):
        ym = d // 100
        if d == rebals[0]:
            prev_s123 = sig_map.get(ym, 0)
        if i > 0 and cal_dates[i-1] // 100 != ym:
            prev_s123 = sig_map.get(cal_dates[i-1] // 100, 0)

        # ---- 择时权重 ----
        if prev_s123 is None:
            tw = 0.0
        elif timing_mode == "binary":
            if not state_in and prev_s123 >= 3:
                state_in = True
            elif state_in and prev_s123 <= 1:
                state_in = False
            tw = 1.0 if state_in else 0.0
        elif timing_mode == "tiered":
            tw = 1.0 if prev_s123 >= 3 else (0.5 if prev_s123 == 2 else 0.0)
        elif timing_mode == "ma20":
            tw = float(ma20_w_daily.get(d, 1.0))
        elif timing_mode == "s123_ma20":
            sw = 1.0 if prev_s123 >= 3 else (0.5 if prev_s123 == 2 else 0.0)
            mw = float(ma20_w_daily.get(d, 1.0))
            tw = sw * mw
        else:
            tw = 0.0

        reserve *= (1 + v8_daily.at[d])

        if d in rebals:
            vw = tgt_w(d)
            w = tw * vw
            # 组合回撤硬降档（T-1 收盘净值）
            if dd_degrade is not None and last_nav > 0:
                dd = last_nav / peak_nav - 1.0
                if dd < dd_degrade:
                    w = w * dd_degrade_scale

            if w <= 1e-9:
                for c, sh in positions.items():
                    o = open_w.at[d, c]
                    if not np.isnan(o) and o > 0:
                        cash += sh * o * 0.999
                positions = {}
                reserve += cash
                cash = 0.0
                state_in = False
            else:
                pool = rebal_scores(d)
                if pool is not None and len(pool):
                    sel = select_with_limit(pool)
                    if log_holdings:
                        holdings_log[d] = list(sel)
                    equity = cash + reserve + sum(
                        sh * close_w.at[d, c] if not np.isnan(close_w.at[d, c]) else 0
                        for c, sh in positions.items())
                    target_stock = equity * w
                    for c in list(positions):
                        if c not in sel:
                            o = open_w.at[d, c]
                            if not np.isnan(o) and o > 0:
                                cash += positions[c] * o * 0.999
                            del positions[c]
                    cur_val = sum(
                        positions.get(c, 0) * close_w.at[d, c]
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
                        have = positions.get(c, 0) * (close_w.at[d, c] if not np.isnan(close_w.at[d, c]) else 0)
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
                    state_in = True

        pos_val = sum(sh * close_w.at[d, c] if not np.isnan(close_w.at[d, c]) else 0
                      for c, sh in positions.items())
        nav = cash + reserve + pos_val
        navs.append(nav)
        last_nav = nav
        peak_nav = max(peak_nav, nav)
        reserve += cash
        cash = 0.0

    nav_s = pd.Series(navs, index=pd.Index(cal_dates, name="trade_date"))
    nav_m = nav_s.groupby((nav_s.index // 100).astype(str)).last()
    monthly = nav_m.pct_change().dropna()
    if log_holdings:
        return nav_s, monthly, holdings_log
    return nav_s, monthly

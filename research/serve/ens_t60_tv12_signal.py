# -*- coding: utf-8 -*-
"""ENS_T60_TV12 前向信号生成器 (月末输出目标权重表)

仿照 serve/daily_signal.py 的结构, 拼装已有件 (不做新研究):
  选股: C8 GBDT 滚动推理 + ENS 混合打分 (0.5×ENH4秩 + 0.5×GBDT秩), 月末截面 Top60, 每行业<=4
  择时: s123 状态机 (>=3 进 / <=1 出), 逻辑与 stock_gbdt_s123_backtest.py 一致
  TV12: 目标波动层, 中证1000 近20日已实现波动缩放仓位 (tgt_vol=0.12, floor=0.4)

无前视约束 (与回测引擎 stock_gbdt_s123_backtest.py 同口径):
  - GBDT 只用 trade_date < 快照月 的数据训练; 推理只用快照月(月末收盘)特征
  - s123 用月末(MonthEnd)数据计算, 下月首个交易日生效 (状态机逐月推进)
  - TV12 波动率用 shift(1) (T-1 信号, 只用调仓日前收盘算出的已实现波动)
  - 成分股 = index_weight 中 trade_date<=调仓日 的最近一期快照 (数据滞后时用旧快照并标注)

输出: 月末目标权重表 (个股 Top60 等权 × TV12 仓位, 余量进 V8 避险 511990/511260/518880 等权)
落盘: research/serve/data/ens/YYYYMMDD.json (signal_date=月末快照日, 独立目录避免与 RS12 覆盖)

用法:
    python research/serve/ens_t60_tv12_signal.py                    # 最新快照信号
    python research/serve/ens_t60_tv12_signal.py --snap 20260731     # 指定快照(调试/补历史)
"""
import os
import sys
import glob
import json
import time
import argparse

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.linear_model import LinearRegression as _LR

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

SERVE_DIR = os.path.dirname(os.path.abspath(__file__))
ENS_DIR = os.path.join(SERVE_DIR, "data", "ens")

# ---- 数据路径 ----
DATA = r"D:/iquant_data/data_v2"
PANEL_PATH = os.path.join(ROOT, "research", "sector_rotation", "stock_ml_panel_fwd.parquet")
IND_MAP_PATH = os.path.join(ROOT, "research", "studies", "study_008_enhancements",
                            "data", "industry_map.parquet")
NAME_MAP_PATH = os.path.join(ROOT, "stock_name_map.parquet")
PE_CACHE = os.path.join(ROOT, "research", "fund_research", "cache", "pe_csi300.parquet")
BOND_CACHE = os.path.join(ROOT, "research", "fund_research", "cache", "bond10y.parquet")
IDX_DIR = os.path.join(ROOT, "research", "chip_momentum", "data", "index_daily")

# ---- 与回测引擎一致的常量 (stock_gbdt_s123_backtest.py) ----
SQRT_242 = np.sqrt(242.0)
TOP_N = 60
MAX_PER_IND = 4
TGT_VOL = 0.12          # ENS_T60_S123_TV12
FLOOR_W = 0.4
VOL_LOOKBACK = 20
PCT_WIN = 2400          # 近10年交易日 (s123 S1/S2 窗口)
PE_QUANT = 0.20
ERP_Z = 1.0
DD_THRESH = -0.25

PRICE_COLS = ["ret_1m", "ivol", "momentum_5", "momentum_10", "momentum_20", "momentum_60",
              "volatility_5", "volatility_10", "volatility_20",
              "alpha_006", "alpha_009", "alpha_012", "alpha_023"]
FIN_COLS = ["roe", "or_yoy", "netprofit_yoy"]
CHIP_COLS = ["vwap_20", "float_pnl_20", "chip_shift_5"]
CHIP_BASE = ["ivol", "ret_1m", "momentum_20", "volatility_20", "alpha_006", "alpha_012"]
CHIP_RESID_COLS = ["vwap_20_resid", "float_pnl_20_resid", "chip_shift_5_resid"]
GBDT_FEATS = ["ivol", "ret_1m", "momentum_20", "volatility_20", "alpha_006", "alpha_012",
              "enh4_score", "vwap_20_resid", "float_pnl_20_resid", "chip_shift_5_resid"]

# V8 避险组合 (与回测 HV_WEIGHTS 一致, 等权)
V8_CODES = ["511990.SH", "511260.SH", "518880.SH"]
V8_NAMES = {"511990.SH": "短债(511990)", "511260.SH": "信用债(511260)", "518880.SH": "黄金(518880)"}


# ============ 因子预处理 (逐字复刻 stock_gbdt_s123_backtest.py prep_feats) ============
def winsorize(s, lo=0.01, hi=0.99):
    if s.notna().sum() < 5:
        return s
    a, b = s.quantile([lo, hi])
    return s.clip(a, b)


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


# ============ s123 信号 (复刻 timing_dingtou 的 S1/S2/S3) ============
def _load_pe_bond():
    pe = pd.read_parquet(PE_CACHE)
    bond = pd.read_parquet(BOND_CACHE)
    pe.index = pd.to_datetime(pe.index)
    bond.index = pd.to_datetime(bond.index)
    return pe, bond


def _rolling_pct(s, d, win=PCT_WIN):
    sub = s[s.index <= d]
    if len(sub) < max(200, win // 4):
        return np.nan
    w = sub.iloc[-win:]
    return float((w < w.iloc[-1]).mean())


def _zscore(s, d, win=PCT_WIN):
    sub = s[s.index <= d]
    if len(sub) < max(200, win // 4):
        return np.nan
    w = sub.iloc[-win:]
    mu, sd = w.mean(), w.std()
    return float((w.iloc[-1] - mu) / sd) if sd > 0 else np.nan


def s123_at(pe, bond, ym):
    """月末(MonthEnd)计算某月 s123 信号, 返回 (n_sig, detail dict)."""
    d = pd.Timestamp(f"{ym}01") + pd.offsets.MonthEnd(0)
    close_ix = pe["close"]
    dd_ix = close_ix / close_ix.cummax() - 1.0
    erp = 1.0 / pe["pe_ttm"] - bond["y10"].reindex(pe.index).ffill()
    pe_pct = _rolling_pct(pe["pe_ttm"], d)
    erp_z = _zscore(erp, d)
    dd_pct = float(dd_ix.asof(d))
    s1 = 1 if pe_pct < PE_QUANT else 0
    s2 = 1 if erp_z > ERP_Z else 0
    s3 = 1 if dd_pct <= DD_THRESH else 0
    detail = {
        "pe_pct": None if pe_pct != pe_pct else round(pe_pct, 3),
        "erp_z": None if erp_z != erp_z else round(erp_z, 3),
        "dd_pct": round(dd_pct, 4),
        "s1": s1, "s2": s2, "s3": s3,
    }
    return s1 + s2 + s3, detail


# ============ 月份算术 + 状态机 ============
def prev_ym(ym):
    y, m = divmod(ym, 100)
    return (y - 1) * 100 + 12 if m == 1 else ym - 1


def ym_range(start_ym, end_ym):
    out = []
    y, m = divmod(start_ym, 100)
    ey, em = divmod(end_ym, 100)
    while (y, m) <= (ey, em):
        out.append(y * 100 + m)
        m += 1
        if m > 12:
            m = 1
            y += 1
    return out


def _apply(state_in, sig):
    if not state_in and sig >= 3:
        return True
    if state_in and sig <= 1:
        return False
    return state_in


# ============ TV12 目标波动层 (复刻 build_vol_signal + tgt_w) ============
def load_index_ret(code):
    fp = os.path.join(IDX_DIR, f"{code}.parquet")
    df = pd.read_parquet(fp)
    s = df.set_index("trade_date")["close"].pct_change().dropna()
    s.index = s.index.astype(str).str[:8]
    return s


def build_vol_signal(cal_dates, vol_lookback=VOL_LOOKBACK):
    ix_ret = load_index_ret("000852.SH")
    ix_ret.index = ix_ret.index.astype(int)
    ix_ret = ix_ret.reindex(cal_dates).ffill().fillna(0.0)
    ix_vol = ix_ret.rolling(vol_lookback).std() * SQRT_242
    return ix_vol.shift(1)   # T-1: 只用 T 日之前收盘算出的波动率


def tgt_w(d, vol_sig, tgt_vol=TGT_VOL, floor_w=FLOOR_W):
    v = vol_sig.get(d, np.nan)
    if not np.isfinite(v) or v <= 0:
        return 1.0
    return float(np.clip(tgt_vol / v, floor_w, 1.0))


# ============ 选股 (复刻 select_with_limit) ============
def select_with_limit(scores, ind_map, max_per_ind=MAX_PER_IND, top_n=TOP_N):
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


def load_trade_dates():
    return sorted(f[:8] for f in os.listdir(os.path.join(DATA, "data_day1"))
                  if f.endswith(".parquet"))


def main():
    ap = argparse.ArgumentParser(description="ENS_T60_TV12 前向信号生成器")
    ap.add_argument("--snap", type=int, default=None, help="指定月末快照 YYYYMMDD, 默认最新")
    args = ap.parse_args()

    t0 = time.time()

    # ---------- 1. 面板 + 行业映射 + 成分股 ----------
    panel = pd.read_parquet(PANEL_PATH)
    snap = args.snap or int(panel["trade_date"].max())
    snap_rows = panel[panel["trade_date"] == snap]
    if snap_rows.empty:
        print(f"[err] 面板无快照 {snap} (面板范围 {panel['trade_date'].min()}~{panel['trade_date'].max()})")
        sys.exit(1)
    print(f"[1] 面板 {panel['trade_date'].nunique()} 月, 快照 {snap} 共 {len(snap_rows)} 只, "
          f"耗时{time.time()-t0:.0f}s", flush=True)

    im = pd.read_parquet(IND_MAP_PATH)
    ind_map = dict(zip(im["ts_code"], im["industry"]))

    iw = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(os.path.join(DATA, "index_weight", "*.parquet")))],
                   ignore_index=True)
    iw = iw[iw["index_code"] == "000852.SH"]
    iw["iw_date"] = iw["trade_date"].astype(int)
    iw_dates = sorted(iw["iw_date"].unique())
    iw_by_date = {d: set(g["con_code"]) for d, g in iw.groupby("iw_date")}

    def latest_members(rebal_d):
        for d in reversed(iw_dates):
            if d <= rebal_d:
                return iw_by_date[d]
        return set()

    # ---------- 2. 交易日历 + 调仓日 ----------
    trade_dates = load_trade_dates()
    cal_dates = [int(d) for d in trade_dates]
    rebal_date = min((d for d in cal_dates if d > snap), default=snap)
    snap_month = snap // 100
    print(f"[2] 交易日历 {cal_dates[0]}~{cal_dates[-1]}, 调仓日 {rebal_date}", flush=True)

    # ---------- 3. s123 状态机 (月末算, 下月生效) ----------
    pe, bond = _load_pe_bond()
    months_needed = ym_range(201901, snap_month)
    sig_map = {}
    sig_detail = {}
    for ym in months_needed:
        n, det = s123_at(pe, bond, ym)
        sig_map[ym] = n
        sig_detail[ym] = det
    # 逐月推进状态 (每月初调仓用上月信号); 得到快照月末的状态 + 下月目标状态
    state_in = False
    for M in ym_range(201902, snap_month):
        state_in = _apply(state_in, sig_map.get(prev_ym(M), 0))
    sig_next = sig_map.get(snap_month, 0)
    target_state = _apply(state_in, sig_next)
    print(f"[3] s123: 上月信号 {sig_next} (S1/S2/S3={sig_detail[snap_month]['s1']}/"
          f"{sig_detail[snap_month]['s2']}/{sig_detail[snap_month]['s3']}), "
          f"当前状态 {'在' if state_in else '离'}场 -> 目标 {'进场' if target_state else '离场'}", flush=True)

    # ---------- 4. GBDT 滚动推理 (训练只用 trade_date < snap) ----------
    tr = prep_feats(panel[panel["trade_date"] < snap], GBDT_FEATS).sort_values("trade_date")
    X, y = tr[GBDT_FEATS].values, tr["fwd_20"].values
    val_months = sorted(tr["trade_date"].unique())[-3:]
    vm = tr["trade_date"].isin(val_months).values
    mdl = lgb.LGBMRegressor(n_estimators=500, learning_rate=0.05, num_leaves=7,
                            max_depth=3, min_child_samples=80, reg_lambda=2.0, reg_alpha=0.1,
                            subsample=0.9, colsample_bytree=0.9, random_state=42, verbose=-1)
    mdl.fit(X[~vm], y[~vm], eval_set=[(X[vm], y[vm])],
            callbacks=[lgb.early_stopping(50, verbose=False)])
    om = prep_feats(snap_rows, GBDT_FEATS)
    score_gbdt = pd.Series(mdl.predict(om[GBDT_FEATS].values), index=om["ts_code"].values)
    score_enh4 = pd.Series(om["enh4_score"].values, index=om["ts_code"].values)
    print(f"[4] GBDT 训练 {len(tr)} 行 (截止 {val_months[-1]}), 推理 {len(om)} 只, "
          f"耗时{time.time()-t0:.0f}s", flush=True)

    # ---------- 5. ENS 混合打分 + 选股 ----------
    common = score_enh4.index.intersection(score_gbdt.index)
    score_ens = 0.5 * score_enh4[common].rank(pct=True) + 0.5 * score_gbdt[common].rank(pct=True)

    trad_codes = set(snap_rows.loc[snap_rows["is_traditional"], "ts_code"])
    members = latest_members(rebal_date)
    pool = score_ens[score_ens.index.isin(members) & score_ens.index.isin(trad_codes)]
    sel = select_with_limit(pool, ind_map, MAX_PER_IND, TOP_N)
    print(f"[5] ENS 选股: 池 {len(pool)} 只 (传统{len(trad_codes)}∩成分{len(members)}), "
          f"入选 {len(sel)} 只", flush=True)

    # ---------- 6. TV12 目标波动仓位 ----------
    vol_sig = build_vol_signal(cal_dates, VOL_LOOKBACK)
    w = tgt_w(rebal_date, vol_sig)
    print(f"[6] TV12 目标波动仓位 w={w:.3f} (tgt_vol={TGT_VOL}, floor={FLOOR_W})", flush=True)

    # ---------- 7. 组装目标权重表 ----------
    name_map = {}
    if os.path.exists(NAME_MAP_PATH):
        try:
            nd = pd.read_parquet(NAME_MAP_PATH)
            name_map = dict(zip(nd["ts_code"].astype(str), nd["name"].astype(str)))
        except Exception:
            pass

    stock_w = w / len(sel) if len(sel) > 0 else 0.0
    if target_state and len(sel) > 0:
        action = (f"满仓 ENS_T60_TV12 组合 (Top{len(sel)} 等权 × TV12 仓位 {w:.2f}, "
                  f"余量 {1-w:.2f} 进 V8 避险)")
        position = "股票组合(TV12)"
        order_stocks = [(c, stock_w) for c in sel]
        reserve_w = 1.0 - w
    else:
        if target_state and len(sel) == 0:
            print("[warn] 目标进场但选股为空, 兜底转 V8 避险", flush=True)
        action = "s123 弱 (<=1) 或选股为空, 清仓转 V8 避险组合 (等权)"
        position = "V8避险组合"
        order_stocks = []
        reserve_w = 1.0

    picks_out = []
    for code in sel:
        picks_out.append({
            "code": code,
            "name": name_map.get(code, ""),
            "score": round(float(score_ens.get(code, np.nan)), 4),
            "target_weight": round(stock_w, 5),
        })

    order_out = []
    for code, wgt in order_stocks:
        order_out.append({"code": code, "name": name_map.get(code, ""),
                          "target_weight": round(float(wgt), 5)})
    if reserve_w > 1e-9:
        for code in V8_CODES:
            order_out.append({"code": code, "name": V8_NAMES[code],
                              "target_weight": round(reserve_w / len(V8_CODES), 5)})

    notes = []
    if iw_dates and iw_dates[-1] < rebal_date:
        notes.append(f"成分股清单截至 {iw_dates[-1]} (index_weight 最新一期, 用于调仓日 {rebal_date})")
    notes.append(f"GBDT 训练数据截至 {val_months[-1]} (trade_date < {snap}), 模型只用快照月之前训练的版本")
    notes.append(f"财务 PIT: 面板 ann_date 口径, 无前视")

    sig = {
        "strategy": "ENS_T60_TV12",
        "signal_date": str(snap),
        "execution_date": str(rebal_date),
        "as_of_date": str(snap),
        "rebalance_date": str(rebal_date),
        "s123_on": bool(target_state),
        "s123_value": int(sig_next),
        "s123_detail": sig_detail[snap_month],
        "prev_state_in": bool(state_in),
        "tgt_vol_w": round(w, 4),
        "action": action,
        "position": position,
        "picks": picks_out,
        "picks_count": len(picks_out),
        "order_picks": order_out,
        "order_picks_count": len(order_out),
        "data_notes": notes,
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    # ---------- 8. 落盘 ----------
    os.makedirs(ENS_DIR, exist_ok=True)
    fp = os.path.join(ENS_DIR, f"{snap}.json")
    with open(fp, "w", encoding="utf-8") as fh:
        json.dump(sig, fh, ensure_ascii=False, indent=2)
    print(f"[saved] {fp}")

    # ---------- 9. 打印摘要 ----------
    print("\n" + "=" * 78)
    print(f"ENS_T60_TV12 目标权重表  信号日 {sig['signal_date']}  执行日 {sig['execution_date']}")
    print("=" * 78)
    print(f"s123: 上月信号 {sig['s123_value']} | 上一状态 {'在场' if sig['prev_state_in'] else '离场'} "
          f"| 目标 {'进场持股' if sig['s123_on'] else '离场避险'}")
    print(f"TV12 仓位: {sig['tgt_vol_w']:.3f} (余量 {(1 - sig['tgt_vol_w']):.3f} 进 V8)")
    print(f"操作: {sig['action']}")
    print(f"持仓: {sig['position']} | 订单 {sig['order_picks_count']} 项 (个股 {sig['picks_count']})")
    print("\n目标权重 Top 10:")
    for i, p in enumerate(sig["order_picks"][:10], 1):
        print(f"  {i:>2}. {p['code']}  {p['name']:<14}  w={p['target_weight']:.4f}")
    if notes:
        print("\n数据时效提示:")
        for n in notes:
            print(f"  - {n}")
    print(f"\n总耗时 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()

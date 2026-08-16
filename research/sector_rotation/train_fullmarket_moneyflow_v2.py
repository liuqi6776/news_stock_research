# -*- coding: utf-8 -*-
"""
资金流因子深化 + 稳健性验证

1. 扩展资金流因子 (多窗口 5/10/20/60 + 大单/特大单拆分)
2. 模型级滚动WFO对比:
     V2 原4因子 (20日)
     V3 扩展因子 (12个)
3. 分年度收益拆解 (验证非单年暴涨带动)
"""
import os, glob, time, warnings, bisect, json
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import lightgbm as lgb

t0 = time.time()
ROOT = r"c:\Users\liuqi\quant_system_v2"
DATA = r"D:/iquant_data/data_v2"
OUT_DIR = os.path.join(ROOT, "research", "sector_rotation", "results")
PANEL = os.path.join(ROOT, "research", "sector_rotation", "stock_ml_panel_fullmarket_2015.parquet")

# ============ 1. 面板 + 质量 ============
panel = pd.read_parquet(PANEL)
fin = pd.read_parquet(os.path.join(DATA, "fundamental1", "fina_indicator_cache.parquet"))
QUAL_EXTRA = ["roe_dt", "netprofit_margin", "grossprofit_margin", "eps", "dt_eps",
              "current_ratio", "quick_ratio", "debt_to_assets"]
fin = fin[["ts_code", "ann_date"] + QUAL_EXTRA].dropna(subset=["ann_date"])
fin["ann_date"] = fin["ann_date"].astype(str).str.replace("-", "").str[:8].astype(int)
fin = fin.sort_values("ann_date")
panel = panel.sort_values("trade_date")
panel = pd.merge_asof(panel, fin, left_on="trade_date", right_on="ann_date",
                      by="ts_code", direction="backward")
panel["safe_hit30"] = ((panel["fwd100_maxret"] >= 0.30) &
                       (panel["fwd100_minret"] >= -0.20)).astype(float)
panel = panel.dropna(subset=["safe_hit30"])
print(f"[1] 面板+质量: {len(panel):,}行, {time.time()-t0:.0f}s")

# ============ 2. 扩展资金流因子 ============
print("[2] 构建扩展资金流因子...")
mf_files = sorted(glob.glob(os.path.join(DATA, "moneyflow1", "*.parquet")))
need_cols = ["ts_code", "trade_date", "buy_lg_amount", "sell_lg_amount",
             "buy_elg_amount", "sell_elg_amount", "buy_sm_amount", "buy_md_amount",
             "net_mf_amount"]
parts = []
for f in mf_files:
    if os.path.getsize(f) <= 1024:
        continue
    try:
        df = pd.read_parquet(f, columns=need_cols)
    except Exception:
        continue
    if len(df):
        parts.append(df)
mf = pd.concat(parts, ignore_index=True)
mf["trade_date"] = mf["trade_date"].astype(int)
mf = mf.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)

mf["main_buy"] = mf["buy_lg_amount"] + mf["buy_elg_amount"]
mf["main_sell"] = mf["sell_lg_amount"] + mf["sell_elg_amount"]
mf["main_net"] = mf["main_buy"] - mf["main_sell"]
mf["lg_net"] = mf["buy_lg_amount"] - mf["sell_lg_amount"]
mf["elg_net"] = mf["buy_elg_amount"] - mf["sell_elg_amount"]
mf["total_amt"] = (mf["buy_lg_amount"] + mf["sell_lg_amount"] +
                   mf["buy_elg_amount"] + mf["sell_elg_amount"] +
                   mf["buy_sm_amount"] + mf["buy_md_amount"])
g = mf.groupby("ts_code", sort=False)

# 多窗口净流入额 + 占比 + 买卖比
for w in (5, 10, 20, 60):
    mf[f"mf_main_net_{w}"] = g["main_net"].transform(lambda s, ww=w: s.rolling(ww).sum())
    mf[f"mf_main_ratio_{w}"] = (g["main_net"].transform(lambda s, ww=w: s.rolling(ww).sum()) /
                                (g["total_amt"].transform(lambda s, ww=w: s.rolling(ww).sum()) + 1e-9))
# 大单/特大单占比 (20日)
mf["mf_lg_ratio_20"] = (g["lg_net"].transform(lambda s: s.rolling(20).sum()) /
                        (g["total_amt"].transform(lambda s: s.rolling(20).sum()) + 1e-9))
mf["mf_elg_ratio_20"] = (g["elg_net"].transform(lambda s: s.rolling(20).sum()) /
                         (g["total_amt"].transform(lambda s: s.rolling(20).sum()) + 1e-9))
# 买卖比 (20/60日)
for w in (20, 60):
    mf[f"mf_main_bs_{w}"] = (g["main_buy"].transform(lambda s, ww=w: s.rolling(ww).sum()) /
                             (g["main_sell"].transform(lambda s, ww=w: s.rolling(ww).sum()) + 1e-9))
print(f"  资金流因子计算完成, {time.time()-t0:.0f}s")

V2_MF = ["mf_main_net_20", "mf_main_ratio_20", "mf_net_ratio_20", "mf_main_bs_20"]
# net_ratio_20 需要单独算 (总净流入/成交额)
mf["mf_net_ratio_20"] = (g["net_mf_amount"].transform(lambda s: s.rolling(20).sum()) /
                         (g["total_amt"].transform(lambda s: s.rolling(20).sum()) + 1e-9))
V3_MF = [c for c in [
    "mf_main_net_5", "mf_main_net_10", "mf_main_net_20", "mf_main_net_60",
    "mf_main_ratio_5", "mf_main_ratio_20", "mf_main_ratio_60",
    "mf_lg_ratio_20", "mf_elg_ratio_20",
    "mf_net_ratio_20", "mf_main_bs_20", "mf_main_bs_60"] if c in mf.columns]

month_last = sorted(panel["trade_date"].unique())
mf_snap = mf[mf["trade_date"].isin(month_last)][["ts_code", "trade_date"] + V3_MF].copy()
panel = panel.merge(mf_snap, on=["ts_code", "trade_date"], how="left")
print(f"  合并资金流: {len(V3_MF)}个因子, 非空率={panel['mf_main_net_20'].notna().mean():.1%}")

# ============ 3. 特征定义 + 标准化 ============
PRICE_COLS = ["ret_1m", "ivol", "momentum_5", "momentum_10", "momentum_20", "momentum_60",
              "volatility_5", "volatility_10", "volatility_20",
              "alpha_006", "alpha_009", "alpha_012", "alpha_023"]
CHIP_COLS = ["vwap_20", "float_pnl_20", "prof_pct_20", "chip_conc_20", "chip_shift_5", "pos_vol_20"]
QUAL_COLS = ["roe", "roe_dt", "or_yoy", "netprofit_yoy", "netprofit_margin",
             "grossprofit_margin", "eps", "dt_eps", "current_ratio", "quick_ratio",
             "debt_to_assets"]
BASE_QUAL = PRICE_COLS + CHIP_COLS + QUAL_COLS
V2_COLS = BASE_QUAL + V2_MF
V3_COLS = BASE_QUAL + V3_MF

def winsorize(s, lo=0.01, hi=0.99):
    if s.notna().sum() < 30:
        return s
    a, b = s.quantile([lo, hi])
    return s.clip(a, b)

for c in PRICE_COLS + CHIP_COLS + QUAL_COLS + V3_MF:
    panel[c] = panel.groupby("trade_date")[c].transform(lambda s: winsorize(s))
for c in PRICE_COLS + CHIP_COLS + QUAL_COLS + V3_MF:
    panel[c] = panel.groupby("trade_date")[c].transform(
        lambda s: (s - s.mean()) / (s.std(ddof=1) + 1e-12))
panel["has_fin"] = panel["roe"].notna().astype(int)
panel["has_mf"] = panel["mf_main_net_20"].notna().astype(int)
panel[QUAL_COLS] = panel[QUAL_COLS].fillna(-99.0)
panel[V3_MF] = panel[V3_MF].fillna(0.0)

months = sorted(panel["trade_date"].unique())
oos_months = [m for m in months if m >= 20180101]

# ============ 4. WFO ============
def run_wfo(feat_cols, tag):
    print(f"\n{'='*70}\n[WFO] {tag} 特征数={len(feat_cols)}\n{'='*70}")
    pred_list = []
    last_model = None
    for m in oos_months:
        tr = panel[panel["trade_date"] < m].sort_values("trade_date")
        X_tr = tr[feat_cols].values
        y_tr = tr["safe_hit30"].astype(int).values
        val_months = sorted(tr["trade_date"].unique())[-3:]
        val_mask = tr["trade_date"].isin(val_months).values
        X_fit, y_fit = X_tr[~val_mask], y_tr[~val_mask]
        X_val, y_val = X_tr[val_mask], y_tr[val_mask]
        pos = max(int(y_fit.sum()), 1); neg = len(y_fit) - pos
        model = lgb.LGBMClassifier(
            objective="binary", n_estimators=500, learning_rate=0.05,
            num_leaves=31, max_depth=6, min_child_samples=40,
            reg_alpha=0.2, reg_lambda=1.0, subsample=0.8, colsample_bytree=0.8,
            random_state=42, verbose=-1, scale_pos_weight=neg/pos)
        model.fit(X_fit, y_fit, eval_set=[(X_val, y_val)],
                  callbacks=[lgb.early_stopping(50, verbose=False)])
        last_model = model
        df_m = panel[panel["trade_date"] == m].copy()
        df_m["prob"] = model.predict_proba(df_m[feat_cols])[:, 1]
        pred_list.append(df_m[["trade_date", "ts_code", "industry", "prob",
                               "safe_hit30", "fwd100_maxret", "fwd100_minret"]])
    df_oos = pd.concat(pred_list, ignore_index=True)
    return df_oos, last_model

df_v2, _ = run_wfo(V2_COLS, "V2 资金流(4因子)")
df_v3, model_v3 = run_wfo(V3_COLS, "V3 资金流扩展(12因子)")

# ============ 5. 回测 (含分年度) ============
def backtest(df, tag, delist_discount=0.0):
    print(f"\n{'='*70}\n[回测] {tag}\n{'='*70}")
    codes_bt = set(df["ts_code"].unique())
    px_files = sorted(glob.glob(os.path.join(DATA, "data_day1", "*.parquet")))
    parts = []
    for f in px_files:
        if os.path.getsize(f) <= 1024:
            continue
        d = os.path.basename(f)[:8]
        if d < "20171201":
            continue
        ddf = pd.read_parquet(f, columns=["ts_code", "trade_date", "close"])
        ddf = ddf[ddf["ts_code"].isin(codes_bt)]
        if len(ddf):
            parts.append(ddf)
    px = pd.concat(parts, ignore_index=True)
    px["trade_date"] = pd.to_datetime(px["trade_date"].astype(str), format="%Y%m%d")
    px = px.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)
    CL_MAP = {}
    for code, gdf in px.groupby("ts_code"):
        CL_MAP[code] = (gdf["trade_date"].tolist(), gdf["close"].values)
    ALL_DAYS = sorted(px["trade_date"].unique())

    def get_close(code, dt):
        if code not in CL_MAP:
            return None
        dates, closes = CL_MAP[code]
        lo, hi = 0, len(dates) - 1
        while lo <= hi:
            m = (lo + hi) // 2
            if dates[m] == dt:
                return closes[m]
            if dates[m] < dt:
                lo = m + 1
            else:
                hi = m - 1
        return None

    def get_close_asof(code, dt):
        """最近 <= dt 的收盘价(停牌/缺价时用最后可得价结转); 无任何可得价返回 None"""
        if code not in CL_MAP:
            return None
        dates, closes = CL_MAP[code]
        i = bisect.bisect_right(dates, dt) - 1
        return closes[i] if i >= 0 else None

    # 每只股票的最后交易日/最后收盘价: 用于识别退市(数据终止)并强平
    LAST_DATE = {code: dates[-1] for code, (dates, _) in CL_MAP.items()}
    LAST_CLOSE = {code: closes[-1] for code, (_, closes) in CL_MAP.items()}

    TOP_GLOBAL, MAXK = 20, 3
    RB_PICKS = {}
    for m_int in sorted(df["trade_date"].unique()):
        m_dt = pd.to_datetime(str(m_int), format="%Y%m%d")
        sub = df[df["trade_date"] == m_int].sort_values("prob", ascending=False)
        picks = sub.groupby("industry", sort=False).head(MAXK).head(TOP_GLOBAL)
        plist = [(r["ts_code"], r["industry"], r["prob"]) for _, r in picks.iterrows()]
        for td in ALL_DAYS:
            if td > m_dt:  # 收盘生成信号, 下一交易日成交(与 freeze 口径一致)
                RB_PICKS[td] = plist
                break

    BUY_FEE, SELL_FEE, INIT = 0.0010, 0.0015, 1_000_000
    TP, MAX_HOLD = 0.30, 180
    START = pd.Timestamp("2018-01-01")
    cash, holdings = INIT, {}
    nav_series, trades, delist_log = [], [], []
    start_idx = next(i for i, d in enumerate(ALL_DAYS) if d >= START)

    for di in range(start_idx, len(ALL_DAYS)):
        day = ALL_DAYS[di]
        # 退市强平: 持仓股价格数据已终止(day 超过其最后交易日) → 按最后收盘价(打折)卖出
        for code in list(holdings.keys()):
            if code in LAST_DATE and day > LAST_DATE[code]:
                h = holdings[code]
                sp = LAST_CLOSE[code] * (1 - delist_discount)
                cash += sp * h["qty"] * (1 - SELL_FEE)
                delist_log.append((code, day, h["buy_price"], LAST_CLOSE[code], sp, h["qty"]))
                del holdings[code]
        if day in RB_PICKS:
            picks = RB_PICKS[day]
            held_inds = set(h["industry"] for h in holdings.values())
            new_picks = [(c, ind, p) for c, ind, p in picks
                         if ind not in held_inds and c not in holdings]
            if new_picks and cash > 10000:
                per = cash / len(new_picks)
                for code, ind, prob in new_picks:
                    bp = get_close(code, day)
                    if bp is None or bp <= 0:
                        continue
                    qty = int((per * (1 - BUY_FEE)) / (bp * 100)) * 100
                    if qty <= 0:
                        continue
                    cost = qty * bp * (1 + BUY_FEE)
                    cash -= cost
                    holdings[code] = {"buy_price": bp, "qty": qty,
                                      "buy_day_idx": di, "industry": ind}
                    trades.append((day, "BUY", code, cost, np.nan, 0, ind))
        for code in list(holdings.keys()):
            h = holdings[code]
            cnow = get_close_asof(code, day)
            if cnow is None:
                continue
            ret = cnow / h["buy_price"] - 1
            held = di - h["buy_day_idx"]
            if ret >= TP or held >= MAX_HOLD:
                proceeds = cnow * h["qty"] * (1 - SELL_FEE)
                cash += proceeds
                op = "TP" if ret >= TP else "T180"
                trades.append((day, op, code, proceeds, ret, held, h["industry"]))
                del holdings[code]
        total = cash
        for code, h in holdings.items():
            cnow = get_close_asof(code, day)
            if cnow is None:
                cnow = h["buy_price"]
            total += cnow * h["qty"]
        nav_series.append((day, total))

    nav = pd.Series(dict(nav_series)).sort_index()
    tr = nav.iloc[-1] / INIT - 1
    years = (nav.index[-1] - nav.index[0]).days / 365.25
    ann = (1 + tr) ** (1 / years) - 1 if years > 0 else np.nan
    peak = nav.cummax(); mdd = ((nav - peak) / peak).min()
    rets = nav.pct_change().dropna()
    sharpe = rets.mean() / rets.std(ddof=1) * np.sqrt(252) if len(rets) > 1 and rets.std(ddof=1) > 0 else np.nan
    buys = sum(1 for t in trades if t[1] == "BUY")
    tps = sum(1 for t in trades if t[1] == "TP")
    t180 = sum(1 for t in trades if t[1] == "T180")
    held_days = [t[5] for t in trades if t[1] != "BUY"]
    print(f"  累计 {tr:.1%} | 年化 {ann:.1%} | 回撤 {mdd:.1%} | 夏普 {sharpe:.2f}")
    print(f"  买入{buys}笔 | 止盈{tps}笔 | 止损{t180}笔 | 平均持仓{np.mean(held_days):.0f}天")
    # 分年度
    print("  --- 分年度收益 ---")
    yr = nav.resample("Y").last()
    prev = INIT
    for y, v in yr.items():
        ret = v / prev - 1
        print(f"    {y.year}: {ret:+.1%}  (期末{v/1e4:.0f}万)")
        prev = v
    if delist_log:
        uniq = len({e[0] for e in delist_log})
        loss = sum((e[2] - e[4]) * e[5] for e in delist_log)
        print(f"  退市强平 {len(delist_log)} 笔 / {uniq} 只, 相对买入价损失 {loss:,.0f} 元 (DELIST_DISCOUNT={delist_discount})")
    else:
        print(f"  无退市强平 (DELIST_DISCOUNT={delist_discount})")
    return nav, delist_log

nav_v2, delist_v2 = backtest(df_v2, "V2 资金流(4因子)")
nav_v3, delist_v3 = backtest(df_v3, "V3 资金流扩展(12因子) DELIST_DISCOUNT=0.0", delist_discount=0.0)
nav_v3_d30, delist_v3_d30 = backtest(df_v3, "V3 资金流扩展(12因子) DELIST_DISCOUNT=0.3", delist_discount=0.3)

# 特征重要性
fi = sorted(zip(V3_COLS, model_v3.feature_importances_), key=lambda x: -x[1])
print("\n--- V3 特征重要性 Top20 ---")
for f, imp in fi[:20]:
    print(f"  {f:<24} {imp}")

# 保存
df_v3.to_csv(os.path.join(OUT_DIR, "fullmarket_moneyflow_v3_2015_oos_pred.csv"), index=False, encoding="utf-8-sig")
nav_v3.to_csv(os.path.join(OUT_DIR, "fullmarket_moneyflow_v3_2015_nav.csv"), header=True)
nav_v3_d30.to_csv(os.path.join(OUT_DIR, "fullmarket_moneyflow_v3_2015_nav_delist30.csv"), header=True)
nav_v2.to_csv(os.path.join(OUT_DIR, "fullmarket_moneyflow_2015_nav.csv"), header=True)

# 退市强平明细落盘(下次重跑产出真实明细, 不再只留在 stdout)
def _delist_records(log):
    return [{"ts_code": e[0], "force_date": str(e[1].date()),
             "buy_price": round(float(e[2]), 4), "last_close": round(float(e[3]), 4),
             "force_price": round(float(e[4]), 4), "qty": int(e[5])} for e in log]

delist_summary = {
    "note": "退市强平明细(backtest 返回 delist_log 落盘); 相对买入价损失 = (buy_price - force_price) * qty",
    "v2_4factor": {"discount": 0.0, "count": len(delist_v2), "records": _delist_records(delist_v2)},
    "v3_12factor_d0": {"discount": 0.0, "count": len(delist_v3), "records": _delist_records(delist_v3)},
    "v3_12factor_d30": {"discount": 0.3, "count": len(delist_v3_d30), "records": _delist_records(delist_v3_d30)},
}
with open(os.path.join(OUT_DIR, "delist_log.json"), "w", encoding="utf-8") as f:
    json.dump(delist_summary, f, ensure_ascii=False, indent=2)
print(f"[saved] delist_log.json")

print(f"\n总耗时 {time.time()-t0:.0f}s")

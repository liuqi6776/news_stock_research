# -*- coding: utf-8 -*-
"""
筹码分布因子 (cyq1 真实获利盘/成本分布) 叠加验证

1. 从 cyq1 构建真实筹码因子:
     cyq_wr       获利盘比例 (winner_rate)
     cyq_wr_chg20 获利盘20日变化
     cyq_conc90    90%成本集中度 (cost_95-cost_5)/(cost_95+cost_5)
     cyq_conc70    70%成本集中度 (cost_85-cost_15)/(cost_85+cost_15)
     cyq_cost_skew 成本分布偏度 (weight_avg vs cost_50)
     cyq_wr_std20  获利盘20日波动
2. 模型级滚动WFO对比:
     V3 资金流扩展(12因子) [基线]
     V4 V3 + 筹码分布因子
3. 分年度收益拆解
"""
import os, glob, time, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import lightgbm as lgb

t0 = time.time()
ROOT = r"c:\Users\liuqi\quant_system_v2"
DATA = r"D:/iquant_data/data_v2"
OUT_DIR = os.path.join(ROOT, "research", "sector_rotation", "results")
PANEL = os.path.join(ROOT, "research", "sector_rotation", "stock_ml_panel_fullmarket_72m.parquet")
ENRICH = os.path.join(ROOT, "research", "sector_rotation", "panel_fullmarket_v4_enrich.parquet")

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

# ============ 2. 资金流因子 (V3 12因子) ============
print("[2] 构建资金流因子...")
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
for w in (5, 10, 20, 60):
    mf[f"mf_main_net_{w}"] = g["main_net"].transform(lambda s, ww=w: s.rolling(ww).sum())
    mf[f"mf_main_ratio_{w}"] = (g["main_net"].transform(lambda s, ww=w: s.rolling(ww).sum()) /
                                (g["total_amt"].transform(lambda s, ww=w: s.rolling(ww).sum()) + 1e-9))
mf["mf_lg_ratio_20"] = (g["lg_net"].transform(lambda s: s.rolling(20).sum()) /
                        (g["total_amt"].transform(lambda s: s.rolling(20).sum()) + 1e-9))
mf["mf_elg_ratio_20"] = (g["elg_net"].transform(lambda s: s.rolling(20).sum()) /
                         (g["total_amt"].transform(lambda s: s.rolling(20).sum()) + 1e-9))
for w in (20, 60):
    mf[f"mf_main_bs_{w}"] = (g["main_buy"].transform(lambda s, ww=w: s.rolling(ww).sum()) /
                             (g["main_sell"].transform(lambda s, ww=w: s.rolling(ww).sum()) + 1e-9))
mf["mf_net_ratio_20"] = (g["net_mf_amount"].transform(lambda s: s.rolling(20).sum()) /
                         (g["total_amt"].transform(lambda s: s.rolling(20).sum()) + 1e-9))
V3_MF = ["mf_main_net_5", "mf_main_net_10", "mf_main_net_20", "mf_main_net_60",
         "mf_main_ratio_5", "mf_main_ratio_20", "mf_main_ratio_60",
         "mf_lg_ratio_20", "mf_elg_ratio_20",
         "mf_net_ratio_20", "mf_main_bs_20", "mf_main_bs_60"]
print(f"  资金流完成, {time.time()-t0:.0f}s")

# ============ 3. 筹码分布因子 (cyq1) ============
print("[3] 构建筹码分布因子...")
cyq_files = sorted(glob.glob(os.path.join(DATA, "cyq1", "*.parquet")))
cyq_cols = ["ts_code", "trade_date", "cost_5pct", "cost_15pct", "cost_50pct",
            "cost_85pct", "cost_95pct", "weight_avg", "winner_rate"]
parts = []
for f in cyq_files:
    if os.path.getsize(f) <= 1024:
        continue
    try:
        df = pd.read_parquet(f, columns=cyq_cols)
    except Exception:
        continue
    if len(df):
        parts.append(df)
cyq = pd.concat(parts, ignore_index=True)
cyq["trade_date"] = cyq["trade_date"].astype(int)
cyq = cyq.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)
cyq["cyq_wr"] = cyq["winner_rate"] / 100.0
cyq["cyq_conc90"] = (cyq["cost_95pct"] - cyq["cost_5pct"]) / (cyq["cost_95pct"] + cyq["cost_5pct"] + 1e-9)
cyq["cyq_conc70"] = (cyq["cost_85pct"] - cyq["cost_15pct"]) / (cyq["cost_85pct"] + cyq["cost_15pct"] + 1e-9)
cyq["cyq_cost_skew"] = (cyq["weight_avg"] - cyq["cost_50pct"]) / (cyq["weight_avg"] + 1e-9)
gc = cyq.groupby("ts_code", sort=False)
cyq["cyq_wr_chg20"] = gc["cyq_wr"].transform(lambda s: s - s.shift(20))
cyq["cyq_wr_std20"] = gc["cyq_wr"].transform(lambda s: s.rolling(20).std())
CYQ_COLS = ["cyq_wr", "cyq_wr_chg20", "cyq_conc90", "cyq_conc70",
            "cyq_cost_skew", "cyq_wr_std20"]
print(f"  筹码因子完成, {time.time()-t0:.0f}s")

# ============ 4. 合并到面板 (月末快照) ============
month_last = sorted(panel["trade_date"].unique())
mf_snap = mf[mf["trade_date"].isin(month_last)][["ts_code", "trade_date"] + V3_MF].copy()
panel = panel.merge(mf_snap, on=["ts_code", "trade_date"], how="left")
cyq_snap = cyq[cyq["trade_date"].isin(month_last)][["ts_code", "trade_date"] + CYQ_COLS].copy()
panel = panel.merge(cyq_snap, on=["ts_code", "trade_date"], how="left")
print(f"  合并后: 资金流非空率={panel['mf_main_net_20'].notna().mean():.1%}, "
      f"筹码非空率={panel['cyq_wr'].notna().mean():.1%}")

# ============ 5. 特征定义 + 标准化 ============
PRICE_COLS = ["ret_1m", "ivol", "momentum_5", "momentum_10", "momentum_20", "momentum_60",
              "volatility_5", "volatility_10", "volatility_20",
              "alpha_006", "alpha_009", "alpha_012", "alpha_023"]
CHIP_COLS = ["vwap_20", "float_pnl_20", "prof_pct_20", "chip_conc_20", "chip_shift_5", "pos_vol_20"]
QUAL_COLS = ["roe", "roe_dt", "or_yoy", "netprofit_yoy", "netprofit_margin",
             "grossprofit_margin", "eps", "dt_eps", "current_ratio", "quick_ratio",
             "debt_to_assets"]
BASE_QUAL = PRICE_COLS + CHIP_COLS + QUAL_COLS
V3_COLS = BASE_QUAL + V3_MF
V4_COLS = V3_COLS + CYQ_COLS

def winsorize(s, lo=0.01, hi=0.99):
    if s.notna().sum() < 30:
        return s
    a, b = s.quantile([lo, hi])
    return s.clip(a, b)

ALL_FEATS = PRICE_COLS + CHIP_COLS + QUAL_COLS + V3_MF + CYQ_COLS
for c in ALL_FEATS:
    panel[c] = panel.groupby("trade_date")[c].transform(lambda s: winsorize(s))
for c in ALL_FEATS:
    panel[c] = panel.groupby("trade_date")[c].transform(
        lambda s: (s - s.mean()) / (s.std(ddof=1) + 1e-12))
panel["has_fin"] = panel["roe"].notna().astype(int)
panel["has_mf"] = panel["mf_main_net_20"].notna().astype(int)
panel["has_cyq"] = panel["cyq_wr"].notna().astype(int)
panel[QUAL_COLS] = panel[QUAL_COLS].fillna(-99.0)
panel[V3_MF] = panel[V3_MF].fillna(0.0)
panel[CYQ_COLS] = panel[CYQ_COLS].fillna(0.0)

# 保存富集面板 (供后续复用)
panel.to_parquet(ENRICH, index=False)
print(f"  富集面板已保存: {ENRICH}")

months = sorted(panel["trade_date"].unique())
oos_months = [m for m in months if m >= 20230101]

# ============ 6. WFO ============
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

df_v3, _ = run_wfo(V3_COLS, "V3 资金流(12因子) [基线]")
df_v4, model_v4 = run_wfo(V4_COLS, "V4 +筹码分布(6因子)")

# ============ 7. 回测 (含分年度) ============
def backtest(df, tag):
    print(f"\n{'='*70}\n[回测] {tag}\n{'='*70}")
    codes_bt = set(df["ts_code"].unique())
    px_files = sorted(glob.glob(os.path.join(DATA, "data_day1", "*.parquet")))
    parts = []
    for f in px_files:
        if os.path.getsize(f) <= 1024:
            continue
        d = os.path.basename(f)[:8]
        if d < "20221201":
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

    TOP_GLOBAL, MAXK = 20, 3
    RB_PICKS = {}
    for m_int in sorted(df["trade_date"].unique()):
        m_dt = pd.to_datetime(str(m_int), format="%Y%m%d")
        sub = df[df["trade_date"] == m_int].sort_values("prob", ascending=False)
        picks = sub.groupby("industry", sort=False).head(MAXK).head(TOP_GLOBAL)
        plist = [(r["ts_code"], r["industry"], r["prob"]) for _, r in picks.iterrows()]
        for td in ALL_DAYS:
            if td >= m_dt:
                RB_PICKS[td] = plist
                break

    BUY_FEE, SELL_FEE, INIT = 0.0010, 0.0015, 1_000_000
    TP, MAX_HOLD = 0.30, 180
    START = pd.Timestamp("2023-01-01")
    cash, holdings = INIT, {}
    nav_series, trades = [], []
    start_idx = next(i for i, d in enumerate(ALL_DAYS) if d >= START)

    for di in range(start_idx, len(ALL_DAYS)):
        day = ALL_DAYS[di]
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
            cnow = get_close(code, day)
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
            cnow = get_close(code, day) or h["buy_price"]
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
    print("  --- 分年度收益 ---")
    yr = nav.resample("Y").last()
    prev = INIT
    for y, v in yr.items():
        ret = v / prev - 1
        print(f"    {y.year}: {ret:+.1%}  (期末{v/1e4:.0f}万)")
        prev = v
    return nav

nav_v3 = backtest(df_v3, "V3 资金流(12因子) [基线]")
nav_v4 = backtest(df_v4, "V4 +筹码分布(6因子)")

# 特征重要性
fi = sorted(zip(V4_COLS, model_v4.feature_importances_), key=lambda x: -x[1])
print("\n--- V4 特征重要性 Top20 ---")
for f, imp in fi[:20]:
    print(f"  {f:<24} {imp}")

# 保存
df_v4.to_csv(os.path.join(OUT_DIR, "fullmarket_cyq_v4_oos_pred.csv"), index=False, encoding="utf-8-sig")
nav_v3.to_csv(os.path.join(OUT_DIR, "fullmarket_moneyflow_v3_nav.csv"), header=True)
nav_v4.to_csv(os.path.join(OUT_DIR, "fullmarket_cyq_v4_nav.csv"), header=True)
print(f"\n总耗时 {time.time()-t0:.0f}s")

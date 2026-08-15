# -*- coding: utf-8 -*-
"""
全市场个股精选 (中证1000 → 全市场 5869 只) + 质量分层

流程:
  1. 读全市场面板 stock_ml_panel_fullmarket_72m.parquet
  2. 补全质量因子 (roe_dt, netprofit_margin, grossprofit_margin, eps, dt_eps,
     current_ratio, quick_ratio, debt_to_assets), PIT 用 ann_date 对齐
  3. safe_hit30 标签 = 未来100日最高涨幅>=30% 且 最低回撤>=-20%
  4. 两个特征变体 Walk-Forward LGBM:
       V0 基线  = 价量 + 筹码 (无财务)
       V1 质量  = 价量 + 筹码 + 11质量因子 + has_fin
  5. OOS TopN 命中率诊断 + 质量分层诊断 (Top20内 高/低质量命中率)
  6. 回测: 全局Top20(每行业≤3), 30%止盈, 180天时间止损
对比中证1000基线 (现役 universe_safe_hit30)
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
os.makedirs(OUT_DIR, exist_ok=True)

# ============ 1. 读全市场面板 + 补全质量因子 ============
print("=" * 70)
print("[1] 读全市场面板 + 补全质量因子")
print("=" * 70)
panel = pd.read_parquet(PANEL)
print(f"  基础面板: {len(panel):,}行, 月={panel['trade_date'].nunique()}, "
      f"股票={panel['ts_code'].nunique()}, 范围={panel['trade_date'].min()}~{panel['trade_date'].max()}")

fin = pd.read_parquet(os.path.join(DATA, "fundamental1", "fina_indicator_cache.parquet"))
QUAL_EXTRA = ["roe_dt", "netprofit_margin", "grossprofit_margin", "eps", "dt_eps",
              "current_ratio", "quick_ratio", "debt_to_assets"]
fin = fin[["ts_code", "ann_date"] + QUAL_EXTRA].dropna(subset=["ann_date"])
fin["ann_date"] = fin["ann_date"].astype(str).str.replace("-", "").str[:8].astype(int)
fin = fin.sort_values("ann_date")
panel = panel.sort_values("trade_date")
panel = pd.merge_asof(panel, fin, left_on="trade_date", right_on="ann_date",
                      by="ts_code", direction="backward")
print(f"  补全质量因子后: {len(panel):,}行, roe_dt非空率={panel['roe_dt'].notna().mean():.1%}, "
      f"grossprofit_margin非空率={panel['grossprofit_margin'].notna().mean():.1%}")

# ============ 2. safe_hit30 标签 ============
panel["safe_hit30"] = ((panel["fwd100_maxret"] >= 0.30) &
                       (panel["fwd100_minret"] >= -0.20)).astype(float)
panel = panel.dropna(subset=["safe_hit30"])
print(f"  safe_hit30 正样本率: {panel['safe_hit30'].mean():.1%}")

# ============ 3. 特征定义 ============
PRICE_COLS = ["ret_1m", "ivol", "momentum_5", "momentum_10", "momentum_20", "momentum_60",
              "volatility_5", "volatility_10", "volatility_20",
              "alpha_006", "alpha_009", "alpha_012", "alpha_023"]
CHIP_COLS = ["vwap_20", "float_pnl_20", "prof_pct_20", "chip_conc_20", "chip_shift_5", "pos_vol_20"]
QUAL_COLS = ["roe", "roe_dt", "or_yoy", "netprofit_yoy", "netprofit_margin",
             "grossprofit_margin", "eps", "dt_eps", "current_ratio", "quick_ratio",
             "debt_to_assets"]

BASE_COLS = PRICE_COLS + CHIP_COLS
V1_COLS = BASE_COLS + QUAL_COLS

def winsorize(s, lo=0.01, hi=0.99):
    if s.notna().sum() < 30:
        return s
    a, b = s.quantile([lo, hi])
    return s.clip(a, b)

# 标准化 (横截面, 按 trade_date)
for c in BASE_COLS + QUAL_COLS:
    panel[c] = panel.groupby("trade_date")[c].transform(lambda s: winsorize(s))
for c in BASE_COLS + QUAL_COLS:
    panel[c] = panel.groupby("trade_date")[c].transform(
        lambda s: (s - s.mean()) / (s.std(ddof=1) + 1e-12))
panel["has_fin"] = panel["roe"].notna().astype(int)
panel[QUAL_COLS] = panel[QUAL_COLS].fillna(-99.0)

months = sorted(panel["trade_date"].unique())
oos_months = [m for m in months if m >= 20230101]
print(f"\n  训练 {months[0]}~{oos_months[0]-1} | 预测 OOS {oos_months[0]}~{oos_months[-1]} "
      f"({len(oos_months)}月)")

# ============ 4. Walk-Forward LGBM ============
def run_wfo(feat_cols, tag):
    print(f"\n{'='*70}\n[WFO] {tag} 特征数={len(feat_cols)}\n{'='*70}")
    pred_list = []
    last_model = None
    for i, m in enumerate(oos_months):
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
        keep = ["trade_date", "ts_code", "industry", "prob",
                "safe_hit30", "fwd100_maxret", "fwd100_minret"] + QUAL_COLS
        pred_list.append(df_m[keep])
    df_oos = pd.concat(pred_list, ignore_index=True)
    print(f"  [OOS] 预测 {len(df_oos):,}条, {df_oos['trade_date'].nunique()}月")
    return df_oos, last_model, feat_cols

df_v0, _, _ = run_wfo(BASE_COLS, "V0 基线(价量+筹码, 无财务)")
df_v1, model_v1, feat_v1 = run_wfo(V1_COLS, "V1 质量(价量+筹码+11质量因子)")

# ============ 5. TopN 命中率诊断 + 质量分层 ============
def topn_diag(df, tag):
    print(f"\n--- {tag} TopN safe_hit命中率 ---")
    for topN, maxK in [(10, 2), (20, 3), (30, 5)]:
        picks = (df.sort_values(["trade_date", "prob"], ascending=[True, False])
                  .groupby(["trade_date", "industry"], sort=False).head(maxK)
                  .groupby("trade_date", sort=False).head(topN))
        if len(picks) == 0:
            continue
        hr = picks["safe_hit30"].mean()
        am = picks["fwd100_maxret"].mean()
        print(f"   全局Top{topN}(每行业≤{maxK}): 命中={hr:.1%} 最大涨幅均值={am:.1%}")

topn_diag(df_v0, "V0基线")
topn_diag(df_v1, "V1质量")

# 质量分层诊断: 在 V1 每月 Top20 内, 按质量分(横截面)分高/低
print("\n--- 质量分层诊断: Top20内 高/低质量 命中率 ---")
qual_proxy = QUAL_COLS  # 用标准化后的质量因子均值作质量分
df_v1["qual_score"] = df_v1[qual_proxy].mean(axis=1)
picks20 = (df_v1.sort_values(["trade_date", "prob"], ascending=[True, False])
            .groupby(["trade_date", "industry"], sort=False).head(3)
            .groupby("trade_date", sort=False).head(20))
picks20 = picks20[picks20["roe"] > -99]  # 只留真正有财务数据的
if len(picks20) > 0:
    med = picks20.groupby("trade_date")["qual_score"].transform("median")
    hi = picks20[picks20["qual_score"] >= med]
    lo = picks20[picks20["qual_score"] < med]
    print(f"  高质量(上半) 命中={hi['safe_hit30'].mean():.1%} (n={len(hi)})")
    print(f"  低质量(下半) 命中={lo['safe_hit30'].mean():.1%} (n={len(lo)})")
    print(f"  高-低 命中率差 = {hi['safe_hit30'].mean()-lo['safe_hit30'].mean():+.1%}")

# 特征重要性
fi = sorted(zip(feat_v1, model_v1.feature_importances_), key=lambda x: -x[1])
print("\n--- V1质量 特征重要性 Top20 ---")
for f, imp in fi[:20]:
    print(f"  {f:<24} {imp}")

# ============ 6. 回测 ============
def backtest(df, tag):
    print(f"\n{'='*70}\n[回测] {tag} (全局Top20, 每行业≤3, 30%止盈, 180天止损)\n{'='*70}")
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
    print(f"  时间: {nav.index[0].date()} → {nav.index[-1].date()} ({years:.1f}年)")
    print(f"  期初 {INIT/1e4:.0f}万 → 期末 {nav.iloc[-1]/1e4:.1f}万")
    print(f"  累计 {tr:.1%} | 年化 {ann:.1%} | 回撤 {mdd:.1%} | 夏普 {sharpe:.2f}")
    print(f"  买入{buys}笔 | 30%止盈{tps}笔 | 180天止损{t180}笔 | 平均持仓{np.mean(held_days):.0f}天")
    return nav, trades

nav_v0, trd_v0 = backtest(df_v0, "V0 基线(全市场, 无财务)")
nav_v1, trd_v1 = backtest(df_v1, "V1 质量(全市场, 加质量因子)")

# ============ 保存 ============
df_v1.to_csv(os.path.join(OUT_DIR, "fullmarket_quality_oos_pred.csv"), index=False, encoding="utf-8-sig")
df_v0.to_csv(os.path.join(OUT_DIR, "fullmarket_base_oos_pred.csv"), index=False, encoding="utf-8-sig")
nav_v1.to_csv(os.path.join(OUT_DIR, "fullmarket_quality_nav.csv"), header=True)
nav_v0.to_csv(os.path.join(OUT_DIR, "fullmarket_base_nav.csv"), header=True)
pd.DataFrame(trd_v1, columns=["date", "op", "code", "amount", "ret", "days", "industry"]
            ).to_csv(os.path.join(OUT_DIR, "fullmarket_quality_trades.csv"), index=False, encoding="utf-8-sig")

print(f"\n总耗时 {time.time()-t0:.0f}s")

# -*- coding: utf-8 -*-
"""
新Target: safe_hit30 = (未来100天最大涨幅>=30%) AND (未来100天最大跌幅<=20%)
  → 过滤"先暴跌再涨回来"的假正例
模型: LGBMClassifier → predict_proba 概率排序
回测: 低换手 — 只买没持仓的低估行业Top6, 30%止盈才卖, 不设强平
"""
import os, glob, time
import numpy as np
import pandas as pd
import lightgbm as lgb

ROOT = r"c:\Users\liuqi\quant_system_v2"
DATA = r"D:/iquant_data/data_v2"
PANEL = os.path.join(ROOT, "research", "sector_rotation", "stock_ml_panel_hit30.parquet")
OUT_DIR = os.path.join(ROOT, "research", "sector_rotation", "results")
t0 = time.time()

# ===== Part 1: 构建新Target safe_hit30 =====
print("=" * 60)
print("Part 1: 构建新Target safe_hit30 (回撤<20% 且 涨幅>=30%)")
print("=" * 60)
panel = pd.read_parquet(PANEL)
codes_need = set(panel['ts_code'].unique())

px_files = sorted(glob.glob(os.path.join(DATA, "data_day1", "*.parquet")))
parts = []
for f in px_files:
    if os.path.getsize(f) <= 1024: continue
    d = os.path.basename(f)[:8]
    if d < "20190601": continue
    df = pd.read_parquet(f, columns=["ts_code", "trade_date", "close"])
    df = df[df["ts_code"].isin(codes_need)]
    if len(df): parts.append(df)
px = pd.concat(parts, ignore_index=True)
px["trade_date"] = pd.to_datetime(px["trade_date"].astype(str), format="%Y%m%d")
px = px.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)
print(f"  行情: {len(px):,}行, {px['ts_code'].nunique()}只")

# 向量化: 未来100天的最高/最低收盘 relative to T日
g = px.groupby("ts_code", sort=False)
cnt  = g["close"].transform(lambda s: s.shift(-1).rolling(100, min_periods=10).count())
fmax = g["close"].transform(lambda s: s.shift(-1).rolling(100, min_periods=10).max())
fmin = g["close"].transform(lambda s: s.shift(-1).rolling(100, min_periods=10).min())
px["fwd100_maxret"] = np.where(cnt >= 10, fmax / px["close"] - 1, np.nan)
px["fwd100_minret"] = np.where(cnt >= 10, fmin / px["close"] - 1, np.nan)

# 新Target: 回撤<20% 且 涨幅>=30%
mask_safe = (px["fwd100_maxret"] >= 0.30) & (px["fwd100_minret"] >= -0.20)
px["safe_hit30"] = np.where(mask_safe, 1.0, 0.0)
px.loc[px["fwd100_maxret"].isna(), "safe_hit30"] = np.nan

# 旧Target对比
old_hit = (px["fwd100_maxret"] >= 0.30).astype(float)
old_hit[px["fwd100_maxret"].isna()] = np.nan
print(f"  旧hit30正样本率: {old_hit.mean():.1%}")
print(f"  新safe_hit30正样本率: {px['safe_hit30'].mean():.1%}")
filtered = old_hit.sum() - px['safe_hit30'].sum()
print(f"  过滤掉{filtered:.0f}个先暴跌再涨回的假正例 (占旧正例的{filtered/old_hit.sum():.1%})")

# merge到面板
panel["td_dt"] = pd.to_datetime(panel["trade_date"].astype(str), format="%Y%m%d")
panel = panel.merge(
    px[["ts_code", "trade_date", "safe_hit30", "fwd100_minret"]],
    left_on=["ts_code", "td_dt"], right_on=["ts_code", "trade_date"],
    how="left", suffixes=("", "_px"))
for c in ["td_dt", "trade_date_px"]:
    if c in panel.columns: panel = panel.drop(columns=c)
panel = panel.dropna(subset=["safe_hit30"])

# ===== Part 2: 预处理 + Walk-Forward训练 LGBMClassifier =====
print("\n" + "=" * 60)
print("Part 2: Walk-Forward LGBMClassifier (输出概率)")
print("=" * 60)
PRICE_COLS = ["ret_1m", "ivol", "momentum_5", "momentum_10", "momentum_20", "momentum_60",
              "volatility_5", "volatility_10", "volatility_20",
              "alpha_006", "alpha_009", "alpha_012", "alpha_023"]
FIN_COLS = ["roe", "or_yoy", "netprofit_yoy"]
CHIP_COLS = ["vwap_20", "float_pnl_20", "prof_pct_20", "chip_conc_20", "chip_shift_5", "pos_vol_20"]
SECTOR_FEAT = ["pe_pct_val", "is_undervalued_sector", "is_stable_sector",
               "is_undervalued_and_stable", "is_traditional", "f_rev", "f_ivol"]
FEAT_COLS = PRICE_COLS + FIN_COLS + CHIP_COLS + SECTOR_FEAT

def winsorize(s, lo=0.01, hi=0.99):
    if s.notna().sum() < 30: return s
    a, b = s.quantile([lo, hi])
    return s.clip(a, b)

for c in PRICE_COLS + FIN_COLS + CHIP_COLS + ["pe_pct_val", "f_rev", "f_ivol"]:
    panel[c] = panel.groupby("trade_date")[c].transform(lambda s: winsorize(s))
panel["has_fin"] = panel["roe"].notna().astype(int)
FEAT_COLS.append("has_fin")
for c in PRICE_COLS + FIN_COLS + CHIP_COLS + ["pe_pct_val", "f_rev", "f_ivol"]:
    panel[c] = panel.groupby("trade_date")[c].transform(
        lambda s: (s - s.mean()) / (s.std(ddof=1) + 1e-12))
panel[FIN_COLS] = panel[FIN_COLS].fillna(-99.0)
panel = panel.dropna(subset=FEAT_COLS + ["safe_hit30"])

months = sorted(panel["trade_date"].unique())
oos_months = [m for m in months if m >= 20230101]
print(f"  训练 {months[0]}~{oos_months[0]-1}, 预测 {oos_months[0]}~{oos_months[-1]} ({len(oos_months)}月)")

pred_list = []
last_model = None
for i, m in enumerate(oos_months):
    train_panel = panel[panel["trade_date"] < m]
    df_tr = train_panel.sort_values("trade_date")
    X_tr = df_tr[FEAT_COLS].values
    y_tr = df_tr["safe_hit30"].astype(int).values
    # 早停验证: 最后3个训练月
    val_months = sorted(df_tr["trade_date"].unique())[-3:]
    val_mask = df_tr["trade_date"].isin(val_months).values
    X_fit, y_fit = X_tr[~val_mask], y_tr[~val_mask]
    X_val, y_val = X_tr[val_mask], y_tr[val_mask]

    pos = max(int(y_fit.sum()), 1)
    neg = len(y_fit) - pos
    spw = neg / pos

    model = lgb.LGBMClassifier(
        objective="binary", n_estimators=500, learning_rate=0.05,
        num_leaves=31, max_depth=6, min_child_samples=40,
        reg_alpha=0.2, reg_lambda=1.0, subsample=0.8, colsample_bytree=0.8,
        random_state=42, verbose=-1, scale_pos_weight=spw,
    )
    model.fit(X_fit, y_fit, eval_set=[(X_val, y_val)],
              callbacks=[lgb.early_stopping(50, verbose=False)])
    last_model = model

    # 预测: 只在低估池内打分
    df_m = panel[panel["trade_date"] == m].copy()
    df_m = df_m[df_m["is_undervalued_sector"] == 1].copy()
    if len(df_m) == 0: continue
    df_m["prob"] = model.predict_proba(df_m[FEAT_COLS])[:, 1]
    pred_list.append(df_m[["trade_date", "ts_code", "industry", "prob",
                           "safe_hit30", "fwd100_maxret", "fwd100_minret",
                           "is_undervalued_and_stable"]])
    if (i+1) % 6 == 0 or i == len(oos_months)-1:
        print(f"  WFO {i+1}/{len(oos_months)}: {m}, train月={df_tr['trade_date'].nunique()}, 候选{len(df_m)}只")

df_oos = pd.concat(pred_list, ignore_index=True)
print(f"\n[3] OOS预测: {len(df_oos)}条, {df_oos['trade_date'].nunique()}月")

# 诊断: TopN safe_hit命中率
print("\n--- OOS TopN safe_hit命中率 ---")
for topN in [1, 3, 5, 10, 20]:
    topn = df_oos.groupby("trade_date").head(topN)
    if len(topn) == 0: continue
    hr = topn["safe_hit30"].mean()
    avg_max = topn["fwd100_maxret"].mean()
    avg_min = topn["fwd100_minret"].mean()
    print(f"  Top{topN:<2} safe_hit率={hr:.1%} | 平均最大涨幅={avg_max:.1%} | 平均最大跌幅={avg_min:.1%}")

# ===== Part 3: 低换手回测 =====
print("\n" + "=" * 60)
print("Part 3: 低换手回测 (只买没持仓的行业, 30%止盈, 不设强平)")
print("=" * 60)
codes_bt = set(df_oos["ts_code"].unique())
px_bt = px[px["ts_code"].isin(codes_bt)].copy()
CL_MAP = {}
for code, gdf in px_bt.groupby("ts_code"):
    CL_MAP[code] = (gdf["trade_date"].tolist(), gdf["close"].values)
ALL_DAYS = sorted(px_bt["trade_date"].unique())

def get_close(code, dt):
    if code not in CL_MAP: return None
    dates, closes = CL_MAP[code]
    lo, hi = 0, len(dates)-1
    while lo <= hi:
        m = (lo + hi) // 2
        if dates[m] == dt: return closes[m]
        if dates[m] < dt: lo = m + 1
        else: hi = m - 1
    return None

# 调仓日 → 每个低估行业Top6
RB_PICKS = {}
for m_int in sorted(df_oos["trade_date"].unique()):
    m_dt = pd.to_datetime(str(m_int), format="%Y%m%d")
    subset = df_oos[df_oos["trade_date"] == m_int].copy()
    S_list = sorted(subset["industry"].dropna().unique())
    picks = []
    for ind in S_list:
        ind_df = subset[subset["industry"] == ind].sort_values("prob", ascending=False).head(6)
        for _, row in ind_df.iterrows():
            picks.append((row["ts_code"], row["industry"], row["prob"]))
    # 对齐到最近交易日
    for td in ALL_DAYS:
        if td >= m_dt:
            RB_PICKS[td] = picks
            break

# 日频回测
BUY_FEE = 0.0010; SELL_FEE = 0.0015; INIT = 1_000_000; TP = 0.30
cash = INIT
holdings = {}
nav_series = []; trades = []

for di, day in enumerate(ALL_DAYS):
    # 1. 调仓日: 只买当前没持仓行业的Top6
    if day in RB_PICKS:
        picks = RB_PICKS[day]
        held_inds = set(h['industry'] for h in holdings.values())
        new_picks = [(c, ind, p) for c, ind, p in picks
                     if ind not in held_inds and c not in holdings]
        if new_picks and cash > 10000:
            per = cash / len(new_picks)
            for code, ind, prob in new_picks:
                bp = get_close(code, day)
                if bp is None or bp <= 0: continue
                qty = int((per * (1 - BUY_FEE)) / (bp * 100)) * 100
                if qty <= 0: continue
                cost = qty * bp * (1 + BUY_FEE)
                cash -= cost
                holdings[code] = {"buy_price": bp, "qty": qty,
                                  "buy_day_idx": di, "industry": ind}
                trades.append((day, "BUY", code, cost, np.nan, 0, ind))

    # 2. 30%止盈 (不设强平)
    for code in list(holdings.keys()):
        h = holdings[code]
        cnow = get_close(code, day)
        if cnow is None: continue
        ret = cnow / h["buy_price"] - 1
        if ret >= TP:
            proceeds = cnow * h["qty"] * (1 - SELL_FEE)
            cash += proceeds
            trades.append((day, "TP", code, proceeds, ret,
                           di - h["buy_day_idx"], h["industry"]))
            del holdings[code]

    # 3. 净值
    total = cash
    for code, h in holdings.items():
        cnow = get_close(code, day) or h["buy_price"]
        total += cnow * h["qty"]
    nav_series.append(total)

nav_s = pd.Series(nav_series, index=ALL_DAYS, name="nav")
tr = nav_s.iloc[-1] / INIT - 1
years = (nav_s.index[-1] - nav_s.index[0]).days / 365.25
ann = (1 + tr) ** (1/years) - 1 if years > 0 else np.nan
peak = nav_s.cummax()
mdd = ((nav_s - peak) / peak).min()
rets = nav_s.pct_change().dropna()
sharpe = rets.mean() / rets.std(ddof=1) * np.sqrt(252) if len(rets) > 1 and rets.std(ddof=1) > 0 else np.nan
buys = sum(1 for t in trades if t[1]=="BUY")
tps = sum(1 for t in trades if t[1]=="TP")
held_days = [t[5] for t in trades if t[1] != "BUY"]

print(f"\n  时间: {nav_s.index[0].date()} → {nav_s.index[-1].date()} ({years:.1f}年)")
print(f"  期初: {INIT/1e4:.0f}万 → 期末: {nav_s.iloc[-1]/1e4:.1f}万")
print(f"  累计: {tr:.1%} | 年化: {ann:.1%}")
print(f"  回撤: {mdd:.1%} | 夏普: {sharpe:.2f}")
print(f"  买入{buys}笔 | 止盈{tps}笔 | 平均持仓{np.mean(held_days):.0f}天")
print(f"  (不设强平: 只有涨过30%才卖)")

# 特征重要性
if last_model:
    fi = sorted(zip(FEAT_COLS, last_model.feature_importances_), key=lambda x:-x[1])
    print("\n--- 特征重要性 Top10 ---")
    for f, imp in fi[:10]:
        print(f"  {f:<28} {imp}")

# 保存
df_oos.to_csv(os.path.join(OUT_DIR, "safe_hit30_oos_pred.csv"), index=False, encoding="utf-8-sig")
nav_s.to_csv(os.path.join(OUT_DIR, "safe_hit30_nav.csv"), header=True)
pd.DataFrame(trades, columns=["date","op","code","amount","ret","days","industry"]
            ).to_csv(os.path.join(OUT_DIR, "safe_hit30_trades.csv"), index=False, encoding="utf-8-sig")
print(f"\n耗时 {time.time()-t0:.0f}s")

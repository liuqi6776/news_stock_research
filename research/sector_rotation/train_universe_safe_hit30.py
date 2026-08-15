# -*- coding: utf-8 -*-
"""
板块之上的统一ML架构 (v4)

v1~v3 的问题: 硬性先把 is_undervalued_sector==1 过滤掉再板块内选股 → 两阶段硬约束
v4 改进: 训练在全市场所有板块上进行
  - 特征 = 个股因子 + 板块级特征(PE分位/板块动量/板块波动/低估标记) + 个股x板块交互
  - ML 自己学会: 什么板块+什么个股组合最可能 safe_hit30 (回撤<20% 且 涨幅>=30%)
  - 回测: 全市场打分, 每行业最多K只保持分散, 30%止盈低换手
"""
import os, glob, time
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import precision_recall_fscore_support

ROOT = r"c:\Users\liuqi\quant_system_v2"
DATA = r"D:/iquant_data/data_v2"
PANEL = os.path.join(ROOT, "research", "sector_rotation", "stock_ml_panel_hit30.parquet")
RET_CSV = os.path.join(ROOT, "research", "sector_rotation", "results", "industry_ret.csv")
PE_CSV = os.path.join(ROOT, "research", "sector_rotation", "results", "industry_pe.csv")
OUT_DIR = os.path.join(ROOT, "research", "sector_rotation", "results")
os.makedirs(OUT_DIR, exist_ok=True)
t0 = time.time()

# ================= Part 1: 读取 & 构建全市场面板 =================
print("=" * 60)
print("Part 1: 构建全市场面板 + 板块级特征")
print("=" * 60)
panel = pd.read_parquet(PANEL)
print(f"  基础面板: {len(panel):,}行, 月={panel['trade_date'].nunique()}, 股票={panel['ts_code'].nunique()}只")

# 板块月度特征 (industry_ret: 月度收益, industry_pe: 月度PE)
ret_df = pd.read_csv(RET_CSV, index_col=0); ret_df.index = ret_df.index.astype(str)
pe_df = pd.read_csv(PE_CSV, index_col=0); pe_df.index = pe_df.index.astype(str)
pe_pct = pe_df.rolling(60, min_periods=12).rank(pct=True)
# 板块动量/波动
sector_ret_3m  = ret_df.rolling(3,  min_periods=2).apply(lambda x: (1+x).prod()-1, raw=True)
sector_ret_12m = ret_df.rolling(12, min_periods=6).apply(lambda x: (1+x).prod()-1, raw=True)
sector_vol_12m = ret_df.rolling(12, min_periods=6).std(ddof=1)
sector_ret_1m  = ret_df
print(f"  板块月度特征: {ret_df.shape[1]}个行业 × {ret_df.shape[0]}月")

# 拉平成 (yyyymm, industry) -> 板块特征
recs = []
for ym_int in pe_pct.index:
    ym_str = str(int(ym_int)); yyyymm = int(ym_str[:4] + ym_str[4:6])
    for col in pe_pct.columns:
        if col in ret_df.columns and ym_int in ret_df.index:
            recs.append((yyyymm, col,
                         pe_pct.loc[ym_int, col],
                         sector_ret_1m.loc[ym_int, col],
                         sector_ret_3m.loc[ym_int, col],
                         sector_ret_12m.loc[ym_int, col],
                         sector_vol_12m.loc[ym_int, col]))
sect_df = pd.DataFrame(recs, columns=["yyyymm", "industry", "s_pe_pct",
                                      "s_ret_1m", "s_ret_3m", "s_ret_12m", "s_vol_12m"])
print(f"  板块特征表: {len(sect_df):,}条")

panel["dt"] = pd.to_datetime(panel["trade_date"].astype(str), format="%Y%m%d")
panel["yyyymm"] = panel["dt"].dt.year * 100 + panel["dt"].dt.month
panel = panel.merge(sect_df, on=["yyyymm", "industry"], how="left")
# 交互特征: 低估板块 × 个股动量
panel["x_undv_mom20"] = panel["is_undervalued_sector"] * panel["momentum_20"]
panel["x_undv_mom60"] = panel["is_undervalued_sector"] * panel["momentum_60"]
print(f"  合并后: {panel.shape}")

# ================= Part 2: safe_hit30 标签(回撤<20% 且 涨幅>=30%) =================
codes_need = set(panel["ts_code"].unique())
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
g = px.groupby("ts_code", sort=False)
cnt  = g["close"].transform(lambda s: s.rolling(100, min_periods=10).count().shift(-100))
fmax = g["close"].transform(lambda s: s.rolling(100, min_periods=10).max().shift(-100))
fmin = g["close"].transform(lambda s: s.rolling(100, min_periods=10).min().shift(-100))
px["fwd100_maxret"] = np.where(cnt >= 10, fmax / px["close"] - 1, np.nan)
px["fwd100_minret"] = np.where(cnt >= 10, fmin / px["close"] - 1, np.nan)
px["safe_hit30"] = np.where((px["fwd100_maxret"] >= 0.30) & (px["fwd100_minret"] >= -0.20), 1.0, 0.0)
px.loc[px["fwd100_maxret"].isna(), "safe_hit30"] = np.nan

panel = panel.merge(px[["ts_code", "trade_date", "safe_hit30", "fwd100_maxret", "fwd100_minret"]],
                    left_on=["ts_code", "dt"], right_on=["ts_code", "trade_date"],
                    how="left", suffixes=("", "_px"))
for c in ["dt", "trade_date_px"]:
    if c in panel.columns: panel = panel.drop(columns=c)
panel = panel.dropna(subset=["safe_hit30"])
print(f"  标签完成: safe_hit30正样本率={panel['safe_hit30'].mean():.1%} (全市场)")

# ================= Part 3: 特征工程 + Walk-Forward =================
PRICE_COLS = ["ret_1m", "ivol", "momentum_5", "momentum_10", "momentum_20", "momentum_60",
              "volatility_5", "volatility_10", "volatility_20",
              "alpha_006", "alpha_009", "alpha_012", "alpha_023"]
FIN_COLS = ["roe", "or_yoy", "netprofit_yoy"]
CHIP_COLS = ["vwap_20", "float_pnl_20", "prof_pct_20", "chip_conc_20", "chip_shift_5", "pos_vol_20"]
SECT_COLS = ["s_pe_pct", "s_ret_1m", "s_ret_3m", "s_ret_12m", "s_vol_12m",
             "is_undervalued_sector", "is_stable_sector", "is_undervalued_and_stable",
             "is_traditional", "f_rev", "f_ivol",
             "x_undv_mom20", "x_undv_mom60"]
FEAT_COLS = PRICE_COLS + FIN_COLS + CHIP_COLS + SECT_COLS

def winsorize(s, lo=0.01, hi=0.99):
    if s.notna().sum() < 30: return s
    a, b = s.quantile([lo, hi])
    return s.clip(a, b)

for c in PRICE_COLS + FIN_COLS + CHIP_COLS + ["s_pe_pct", "s_ret_1m", "s_ret_3m", "s_ret_12m", "s_vol_12m", "f_rev", "f_ivol"]:
    panel[c] = panel.groupby("trade_date")[c].transform(lambda s: winsorize(s))
panel["has_fin"] = panel["roe"].notna().astype(int)
FEAT_COLS.append("has_fin")
for c in PRICE_COLS + FIN_COLS + CHIP_COLS + ["s_pe_pct", "s_ret_1m", "s_ret_3m", "s_ret_12m", "s_vol_12m", "f_rev", "f_ivol"]:
    panel[c] = panel.groupby("trade_date")[c].transform(
        lambda s: (s - s.mean()) / (s.std(ddof=1) + 1e-12))
panel[FIN_COLS] = panel[FIN_COLS].fillna(-99.0)
panel = panel.dropna(subset=FEAT_COLS + ["safe_hit30"])

months = sorted(panel["trade_date"].unique())
oos_months = [m for m in months if m >= 20230101]
print(f"\n  训练 {months[0]}~{oos_months[0]-1} | 预测 {oos_months[0]}~{oos_months[-1]} ({len(oos_months)}月)")
print(f"  特征数: {len(FEAT_COLS)} (含板块级特征)")

# ================= Part 4: Walk-Forward LGBMClassifier =================
pred_list = []
last_model = None
for i, m in enumerate(oos_months):
    train_panel = panel[panel["trade_date"] < m].sort_values("trade_date")
    X_tr = train_panel[FEAT_COLS].values
    y_tr = train_panel["safe_hit30"].astype(int).values
    val_months = sorted(train_panel["trade_date"].unique())[-3:]
    val_mask = train_panel["trade_date"].isin(val_months).values
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
    df_m["prob"] = model.predict_proba(df_m[FEAT_COLS])[:, 1]
    pred_list.append(df_m[["trade_date", "ts_code", "industry", "prob",
                           "safe_hit30", "fwd100_maxret", "fwd100_minret",
                           "is_undervalued_sector", "s_pe_pct"]])
    if (i+1) % 6 == 0 or i == len(oos_months)-1:
        print(f"  WFO {i+1}/{len(oos_months)}: {m}, train月={train_panel['trade_date'].nunique()}, 全市场{len(df_m)}只")

df_oos = pd.concat(pred_list, ignore_index=True)
print(f"\n[OOS] 预测: {len(df_oos):,}条, {df_oos['trade_date'].nunique()}月 (全市场打分)")

# ---- 诊断: 全市场TopN + 每行业最多K只 ----
print("\n--- OOS 全市场打分 TopN safe_hit命中率 ---")
for topN, maxK in [(10, 2), (10, 3), (20, 3), (30, 5)]:
    picks = (df_oos.sort_values(["trade_date", "prob"], ascending=[True, False])
                  .groupby(["trade_date", "industry"], sort=False)
                  .head(maxK)
                  .groupby("trade_date", sort=False)
                  .head(topN))
    if len(picks) == 0: continue
    hr = picks["safe_hit30"].mean()
    avg_max = picks["fwd100_maxret"].mean()
    avg_min = picks["fwd100_minret"].mean()
    print(f"  全局Top{topN}(每行业≤{maxK}) safe_hit率={hr:.1%} | 最大涨幅均值={avg_max:.1%} | 最大跌幅均值={avg_min:.1%}")

# 特征重要性
fi = sorted(zip(FEAT_COLS, last_model.feature_importances_), key=lambda x: -x[1])
print("\n--- 特征重要性 Top15 (板块之上的统一模型) ---")
for f, imp in fi[:15]:
    print(f"  {f:<28} {imp}")

# ================= Part 5: 低换手回测 =================
print("\n" + "=" * 60)
print("Part 5: 低换手回测 (每行业≤K只, 30%止盈, 不设强平)")
print("=" * 60)
TOP_GLOBAL, MAXK = 20, 3
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

# 每月全市场打分 → 每行业TopK → 全局TopN
RB_PICKS = {}
for m_int in sorted(df_oos["trade_date"].unique()):
    m_dt = pd.to_datetime(str(m_int), format="%Y%m%d")
    subset = df_oos[df_oos["trade_date"] == m_int].sort_values("prob", ascending=False)
    picks = (subset.groupby("industry", sort=False).head(MAXK).head(TOP_GLOBAL))
    pick_list = [(r["ts_code"], r["industry"], r["prob"]) for _, r in picks.iterrows()]
    for td in ALL_DAYS:
        if td >= m_dt:
            RB_PICKS[td] = pick_list
            break

BUY_FEE = 0.0010; SELL_FEE = 0.0015; INIT = 1_000_000; TP = 0.30
cash = INIT
holdings = {}
nav_series = []; trades = []

for di, day in enumerate(ALL_DAYS):
    if day in RB_PICKS:
        picks = RB_PICKS[day]
        held_inds = set(h["industry"] for h in holdings.values())
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
print(f"  买入{buys}笔 | 止盈{tps}笔 | 止盈率={tps/(tps+max(buys-tps,1)):.1%} | 平均持仓{np.mean(held_days):.0f}天")
print(f"  参数: 全局Top{TOP_GLOBAL}, 每行业≤{MAXK}, 30%止盈, 无强平, 低换手")

df_oos.to_csv(os.path.join(OUT_DIR, "universe_safe_hit30_oos_pred.csv"), index=False, encoding="utf-8-sig")
nav_s.to_csv(os.path.join(OUT_DIR, "universe_safe_hit30_nav.csv"), header=True)
pd.DataFrame(trades, columns=["date","op","code","amount","ret","days","industry"]
            ).to_csv(os.path.join(OUT_DIR, "universe_safe_hit30_trades.csv"), index=False, encoding="utf-8-sig")
print(f"\n耗时 {time.time()-t0:.0f}s")

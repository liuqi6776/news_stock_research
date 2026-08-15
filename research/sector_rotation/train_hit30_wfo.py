# -*- coding: utf-8 -*-
"""
STEP2: Walk-Forward滚动训练 + 回测一体化

Target: fwd_100_hit30 (未来100交易日内有一次>=30%涨幅, 0/1)
模型: LGBMRanker (lambdarank), 因为最终要的是选股ranking
池子限制: 仅在 is_undervalued_sector==1 (低估板块) 内选股, 或 is_undervalued_and_stable==1

回测: 每月初调仓, 用模型打所有低估池内股票分数, 买TopN, 单票30%止盈 / 60天强平
"""
import os
import time
import joblib
import glob
import numpy as np
import pandas as pd
import lightgbm as lgb

ROOT = r"c:\Users\liuqi\quant_system_v2"
DATA = r"D:/iquant_data/data_v2"
PANEL = os.path.join(ROOT, "research", "sector_rotation", "stock_ml_panel_hit30.parquet")
OUT_DIR = os.path.join(ROOT, "research", "sector_rotation", "results")
os.makedirs(OUT_DIR, exist_ok=True)

t0 = time.time()
panel = pd.read_parquet(PANEL)
print(f"[1] 面板: {panel.shape} 调仓月={panel['trade_date'].nunique()}")

# ---------- 特征 ----------
PRICE_COLS = ["ret_1m", "ivol",
              "momentum_5", "momentum_10", "momentum_20", "momentum_60",
              "volatility_5", "volatility_10", "volatility_20",
              "alpha_006", "alpha_009", "alpha_012", "alpha_023"]
FIN_COLS = ["roe", "or_yoy", "netprofit_yoy"]
CHIP_COLS = ["vwap_20", "float_pnl_20", "prof_pct_20", "chip_conc_20", "chip_shift_5", "pos_vol_20"]
SECTOR_FEAT = ["pe_pct_val", "is_undervalued_sector", "is_stable_sector", "is_undervalued_and_stable",
               "is_traditional", "f_rev", "f_ivol"]
FEAT_COLS = PRICE_COLS + FIN_COLS + CHIP_COLS + SECTOR_FEAT
print(f"[2] 特征数={len(FEAT_COLS)}")

def winsorize(s, lo=0.01, hi=0.99):
    a, b = s.quantile([lo, hi]) if s.notna().sum() >= 30 else (s.min(), s.max())
    return s.clip(a, b)

# ---------- 预处理: 截面winsorize + zscore ----------
for c in PRICE_COLS + FIN_COLS + CHIP_COLS + ["pe_pct_val","f_rev","f_ivol"]:
    panel[c] = panel.groupby("trade_date")[c].transform(lambda s: winsorize(s))
# has_fin
panel["has_fin"] = panel["roe"].notna().astype(int)
FEAT_COLS.append("has_fin")
for c in PRICE_COLS + FIN_COLS + CHIP_COLS + ["pe_pct_val","f_rev","f_ivol"]:
    panel[c] = panel.groupby("trade_date")[c].transform(
        lambda s: (s - s.mean()) / (s.std(ddof=1) + 1e-12))
panel[FIN_COLS] = panel[FIN_COLS].fillna(-99.0)
panel = panel.dropna(subset=FEAT_COLS + ["fwd_100_hit30"])
print(f"    dropna后: {len(panel):,}, 正样本率={panel['fwd_100_hit30'].mean():.1%}")

# ---------- 训练切分 ----------
months = sorted(panel["trade_date"].unique())
oos_months = [m for m in months if m >= 20230101]
print(f"[3] WFO: 训练 {months[0]}~{oos_months[0]-1}, 预测 {oos_months[0]}~{oos_months[-1]} ({len(oos_months)}月)")

def make_rank_data(df, target_col="fwd_100_hit30", use_rank=True):
    df = df.sort_values("trade_date").copy()
    X = df[FEAT_COLS].values
    if use_rank:
        # Lambdarank: 把0/1标签按trade_date分组转成rank (0~4 or 0~1直接也行)
        # 实际上hit30是0/1标签, 用0/1直接rank (相当于把hit=1排到更高)
        y = df[target_col].astype(int).values
    else:
        y = df[target_col].values
    group = df.groupby("trade_date", sort=False).size().values
    return df, X, y, group

# ---------- Walk-Forward 预测 ----------
pred_list = []
last_model = None
for i, m in enumerate(oos_months):
    train_panel = panel[panel["trade_date"] < m]
    if train_panel.empty: continue
    df_tr, X_tr, y_tr, g_tr = make_rank_data(train_panel)
    # 早停: 最后3个训练月
    val_months = sorted(df_tr["trade_date"].unique())[-3:]
    val_mask = df_tr["trade_date"].isin(val_months).values
    X_fit, y_fit = X_tr[~val_mask], y_tr[~val_mask]
    X_val, y_val = X_tr[val_mask], y_tr[val_mask]
    g_fit = df_tr.loc[~val_mask].groupby("trade_date", sort=False).size().values
    g_val = df_tr.loc[val_mask].groupby("trade_date", sort=False).size().values

    model = lgb.LGBMRanker(
        objective="lambdarank",
        n_estimators=500, learning_rate=0.05, num_leaves=31, max_depth=6,
        min_child_samples=40, reg_alpha=0.2, reg_lambda=1.0,
        subsample=0.8, colsample_bytree=0.8, random_state=42, verbose=-1,
    )
    model.fit(X_fit, y_fit, group=g_fit,
              eval_set=[(X_val, y_val)], eval_group=[g_val],
              eval_metric="ndcg", eval_at=[5, 10],
              callbacks=[lgb.early_stopping(50, verbose=False)])
    last_model = model

    # 预测: 只在低估池内打分 & 买入
    df_m = panel[panel["trade_date"] == m].copy()
    df_m = df_m[df_m["is_undervalued_sector"] == 1].copy()
    if len(df_m) == 0:
        continue
    df_m["pred"] = model.predict(df_m[FEAT_COLS])
    df_m = df_m.sort_values("pred", ascending=False)
    pred_list.append(df_m[["trade_date","ts_code","industry","pred","fwd_100_hit30","fwd100_maxret",
                           "is_undervalued_and_stable","is_stable_sector"]])
    if (i+1) % 6 == 0 or i == len(oos_months)-1:
        print(f"  WFO {i+1}/{len(oos_months)}: {m}, train月={df_tr['trade_date'].nunique()}, 候选{len(df_m)}只")

df_oos = pd.concat(pred_list, ignore_index=True)
print(f"[4] OOS预测完成: {len(df_oos)}条, 覆盖{df_oos['trade_date'].nunique()}月")
# ---------- 诊断: TopN命中率 ----------
for topN in [1, 3, 5, 10, 20]:
    topn = df_oos.groupby("trade_date").head(topN)
    if len(topn):
        hr = topn["fwd_100_hit30"].mean()
        avg_maxret = topn["fwd100_maxret"].mean()
        print(f"    Top{topN:<2} 命中率={hr:.1%} | 未来100天平均最大涨幅={avg_maxret:.1%}")
# 稳定低估池内top命中率
stab = df_oos[df_oos["is_undervalued_and_stable"]==1].copy()
for topN in [1,3,5,10]:
    topn = stab.groupby("trade_date").head(topN)
    if len(topn):
        hr = topn["fwd_100_hit30"].mean()
        avg_maxret = topn["fwd100_maxret"].mean()
        print(f"    [稳定低估池] Top{topN:<2} 命中率={hr:.1%} | 未来100天平均最大涨幅={avg_maxret:.1%}")

# ---------- 回测: 月调仓, 买低估池内模型Top6, 单票30%止盈/60天强平 ----------
print(f"[5] 读取日频行情做回测 (止盈30% / 60天强平)...")
codes_need = set(df_oos["ts_code"].unique())
px_files = sorted(glob.glob(os.path.join(DATA, "data_day1", "*.parquet")))
parts = []
for f in px_files:
    if os.path.getsize(f) <= 1024: continue
    d = os.path.basename(f)[:8]
    if d < "20221201": continue
    df = pd.read_parquet(f, columns=["ts_code", "trade_date", "close"])
    df = df[df["ts_code"].isin(codes_need)]
    if len(df):
        parts.append(df)
px = pd.concat(parts, ignore_index=True)
px["trade_date"] = pd.to_datetime(px["trade_date"].astype(str), format="%Y%m%d")
px = px.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)
print(f"    行情={len(px):,}行")
# 调仓日序列 & 日期索引
all_tds = sorted(px["trade_date"].unique())
td2idx = {d:i for i,d in enumerate(all_tds)}
# 买Top6, 单票等权
BUY_FEE = 0.0010
SELL_FEE = 0.0015
INIT = 1_000_000
TOPN = 6
cash = INIT
holdings = {}  # code -> {buy_price, qty, value, buy_td_idx, buy_cost_amt}
nav_ts = []
trades = []
rebalance_mons = sorted(df_oos["trade_date"].unique())
for m in rebalance_mons:
    m_dt = pd.to_datetime(str(m), format="%Y%m%d")
    # 找T+1后的最近交易日 (开盘买入)
    idx0 = None
    for td in all_tds:
        if td >= m_dt:
            idx0 = td2idx[td]; break
    if idx0 is None: continue
    # --- 调仓前先止盈/强平 ---
    for code in list(holdings.keys()):
        h = holdings[code]
        # 到今天了, 看看之前这几天有没有hit30或到60天
        sold = False
        close_sel = px[(px["ts_code"]==code) & (px["trade_date"].isin(all_tds[h["buy_td_idx"]:idx0+1]))]
        if not close_sel.empty:
            sub = close_sel.sort_values("trade_date")["close"].values
            # 遍历每天判断
            for j, c in enumerate(sub):
                ret_here = c / h["buy_price"] - 1
                days_held = j
                hit_tp = ret_here >= 0.30
                hit_60 = days_held >= 60
                if hit_tp or hit_60:
                    sell_price = c
                    proceeds = (sell_price * h["qty"]) * (1 - SELL_FEE)
                    cash += proceeds
                    trades.append((all_tds[h["buy_td_idx"]+j],
                                   "TP" if hit_tp else "SL", code,
                                   proceeds, ret_here, days_held))
                    del holdings[code]
                    sold = True
                    break
    # --- 找本月候选TopN (仅低估, 未持仓) ---
    top_picks = df_oos[df_oos["trade_date"]==m].head(TOPN)["ts_code"].tolist()
    to_buy = [c for c in top_picks if c not in holdings]
    if to_buy and cash > 1000:
        per = cash / len(to_buy)
        for code in to_buy:
            # 买入: 调仓日当天开盘(近似close, 误差不大)
            buy_td = all_tds[idx0]
            df_one = px[(px["ts_code"]==code) & (px["trade_date"]==buy_td)]
            if df_one.empty: continue
            bp = df_one["close"].iloc[0]
            if bp <= 0: continue
            qty = (per * (1 - BUY_FEE)) // (bp * 100) * 100  # 100股整数手
            if qty <= 0: continue
            cost = qty * bp * (1 + BUY_FEE)
            cash -= cost
            holdings[code] = {"buy_price": bp, "qty": qty,
                              "buy_td_idx": idx0, "buy_cost_amt": cost}
            trades.append((buy_td, "BUY", code, cost, np.nan, 0))
    # --- 算净值 (用当前持有票最新close) ---
    total = cash
    for code, h in holdings.items():
        cl_now = None
        for ti in range(idx0, -1, -1):
            r = px.loc[(px["ts_code"]==code) & (px["trade_date"]==all_tds[ti]), "close"]
            if len(r): cl_now = r.iloc[0]; break
        if cl_now is not None:
            total += cl_now * h["qty"]
    nav_ts.append((m_dt, total))

# 最后再算一次剩余持仓净值
if nav_ts:
    last_m = max(m for m,_ in nav_ts)
    # 找所有持仓最后一天的收盘价
    rem_total = cash
    for code, h in list(holdings.items()):
        close_sel = px[px["ts_code"]==code].tail(1)["close"]
        if len(close_sel):
            rem_total += close_sel.iloc[0] * h["qty"]
    # 替换最后一条
    nav_ts[-1] = (last_m, rem_total)

nav_df = pd.DataFrame(nav_ts, columns=["date","value"]).set_index("date")["value"].sort_index()
tr = nav_df.iloc[-1] / INIT - 1
years = (nav_df.index[-1] - nav_df.index[0]).days / 365.25
ann = (1 + tr) ** (1/years) - 1 if years > 0 else np.nan
peak = nav_df.cummax()
mdd = ((nav_df - peak) / peak).min()
sharpe = np.nan
if len(nav_df) >= 3:
    rets = nav_df.pct_change().dropna()
    if len(rets) > 0 and rets.std(ddof=1) > 0:
        sharpe = rets.mean() / rets.std(ddof=1) * np.sqrt(12)
buys = sum(1 for t in trades if t[1]=="BUY")
tps = sum(1 for t in trades if t[1]=="TP")
sls = sum(1 for t in trades if t[1]=="SL")
print(f"\n========== [回测结果 OOS 2023-2026] ==========")
print(f"  时间: {nav_df.index[0].date()} → {nav_df.index[-1].date()} ({years:.1f}年)")
print(f"  期初: {INIT/1e4:.0f}万  → 期末: {nav_df.iloc[-1]/1e4:.1f}万")
print(f"  累计收益: {tr:.1%} | 年化: {ann:.1%}")
print(f"  最大回撤: {mdd:.1%} | 夏普(月): {sharpe:.2f}")
print(f"  买入{buys}笔 | 止盈{tps}笔 | 强平{sls}笔 | 胜率{tps/(tps+sls+1e-9):.1%}")

# 保存
df_oos.to_csv(os.path.join(OUT_DIR, "hit30_oos_predictions.csv"), index=False, encoding="utf-8-sig")
nav_df.to_csv(os.path.join(OUT_DIR, "hit30_stock_nav.csv"), header=True)
pd.DataFrame(trades, columns=["date","op","code","amount","ret","days"]
            ).to_csv(os.path.join(OUT_DIR, "hit30_stock_trades.csv"), index=False, encoding="utf-8-sig")
if last_model is not None:
    joblib.dump(last_model, os.path.join(ROOT, "artifacts", "stock_hit30_lgbmranker.joblib"))
    fi = sorted(zip(FEAT_COLS, last_model.feature_importances_), key=lambda x:-x[1])
    print("\n--- 特征重要性 Top10 ---")
    for f, imp in fi[:10]:
        print(f"  {f:<28} {imp}")
print(f"\n总耗时 {time.time()-t0:.0f}s")

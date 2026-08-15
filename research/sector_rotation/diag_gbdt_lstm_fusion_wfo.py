# -*- coding: utf-8 -*-
"""WFO 融合: GBDT(C8) + 时序 LSTM + 滚动线性 meta-learner

目标: 验证 GBDT 与 LSTM 融合是否提升 OOS 选股能力。

结构 (两阶段 WFO, 严格 PIT, 无未来泄漏):
  Phase 1: GBDT C8 与 时序 LSTM 各自 WFO 预测所有月份
  Phase 2: 滚动线性 meta-learner (岭回归) 学习两模型截面 rank 的融合权重

对比 (OOS 2023+): 纯 GBDT / 纯 LSTM / 等权融合 / 岭回归融合
指标: 月度 IC / ICIR / 正率 / Q1-Q5
"""
import os, time, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import lightgbm as lgb
import torch, torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.linear_model import LinearRegression, Ridge

ROOT = r"c:\Users\liuqi\quant_system_v2"
PANEL = os.path.join(ROOT, "research", "sector_rotation", "stock_ml_panel_72m.parquet")
OUT = os.path.join(ROOT, "research", "sector_rotation", "results", "gbdt_lstm_fusion_wfo.csv")
os.makedirs(os.path.dirname(OUT), exist_ok=True)

C7_COLS   = ["ivol","ret_1m","momentum_20","volatility_20","alpha_006","alpha_012","enh4_score"]
CHIP_COLS = ["vwap_20","float_pnl_20","chip_shift_5"]
CHIP_RESID= ["vwap_20_resid","float_pnl_20_resid","chip_shift_5_resid"]
BASE_COLS = ["ivol","ret_1m","momentum_20","volatility_20","alpha_006","alpha_012"]
C8_COLS   = C7_COLS + CHIP_RESID
T_SEQ = 12  # LSTM 时间步: 个股过去 12 个月

def zscore(s): return (s - s.mean()) / (s.std(ddof=1) + 1e-12)
def winsorize(s, lo=0.01, hi=0.99):
    if s.notna().sum() < 5: return s
    a, b = s.quantile([lo, hi]); return s.clip(a, b)

# ================= 预处理 (与 diag_model_compare_dl.py 一致) =================
t0 = time.time()
panel = pd.read_parquet(PANEL)
for c in ["roe","or_yoy"]:
    panel[c] = panel.groupby("trade_date")[c].transform(lambda s: winsorize(s))
panel["roe"] = panel["roe"].fillna(-99.0)
panel["or_yoy"] = panel["or_yoy"].fillna(-99.0)
g = panel.groupby("trade_date")
panel["enh4_score"] = (-0.40*g["ivol"].rank(pct=True) -0.35*g["ret_1m"].rank(pct=True)
                       +0.15*g["roe"].rank(pct=True) +0.05*g["or_yoy"].rank(pct=True))
ALL_RAW = list(set(C7_COLS + CHIP_COLS) - {"enh4_score"})
for c in ALL_RAW:
    panel[c] = panel.groupby("trade_date")[c].transform(lambda s: winsorize(s))
for c in ALL_RAW:
    panel[c] = panel.groupby("trade_date")[c].transform(zscore)
panel["fwd_20"] = panel.groupby("trade_date")["fwd_20"].transform(lambda s: winsorize(s))
panel = panel.dropna(subset=C7_COLS + CHIP_COLS + ["fwd_20"])
for c in CHIP_COLS:
    panel[f"{c}_resid"] = np.nan
for dt, grp in panel.groupby("trade_date"):
    if len(grp) < 50: continue
    Xb = grp[BASE_COLS].values
    for c in CHIP_COLS:
        y = grp[c].values
        mask = np.isfinite(y) & np.all(np.isfinite(Xb), axis=1)
        if mask.sum() < 50: continue
        lr = LinearRegression().fit(Xb[mask], y[mask])
        panel.loc[grp.index[mask], f"{c}_resid"] = -(y - lr.predict(Xb[mask]))
for c in CHIP_RESID:
    panel[c] = panel.groupby("trade_date")[c].transform(lambda s: winsorize(s))
    panel[c] = panel.groupby("trade_date")[c].transform(zscore)
panel = panel.dropna(subset=C8_COLS + ["fwd_20"])
months = sorted(panel["trade_date"].unique())
oos_months = [m for m in months if m >= 20230101]
print(f"[数据] {len(panel):,} 行, {len(months)} 月, OOS {len(oos_months)} 月, {time.time()-t0:.0f}s")

def monthly_ic(d, factor, ret="fwd_20"):
    out = {}
    for dt, gg in d.groupby("trade_date"):
        if len(gg) < 30: continue
        out[dt] = gg[factor].rank().corr(gg[ret].rank())
    return pd.Series(out).dropna()

# ================= 构造时序样本 (个股滑窗 [t-T+1 .. t] -> fwd_20[t]) =================
t1 = time.time()
seq_dates, seq_codes, seq_y = [], [], []
seq_X = []
for code, grp in panel.groupby("ts_code"):
    grp = grp.sort_values("trade_date")
    X = grp[C8_COLS].values.astype(np.float32)
    y = grp["fwd_20"].values.astype(np.float32)
    dates = grp["trade_date"].values
    if len(grp) < T_SEQ + 2:
        continue
    for i in range(T_SEQ, len(grp)):
        seq_dates.append(dates[i]); seq_codes.append(code)
        seq_X.append(X[i-T_SEQ:i]); seq_y.append(y[i])
seq_X = np.stack(seq_X).astype(np.float32)
seq_y = np.array(seq_y, dtype=np.float32)
seq_dates = np.array(seq_dates)
seq_codes = np.array(seq_codes)
print(f"[时序样本] {seq_X.shape[0]:,} 条 (T={T_SEQ}, K={len(C8_COLS)}), {time.time()-t1:.0f}s")

# ================= Phase 1a: GBDT C8 WFO 预测所有月份 =================
GBDT_PARAMS = dict(n_estimators=500, learning_rate=0.05, num_leaves=7,
                   max_depth=3, min_child_samples=80, reg_lambda=2.0, reg_alpha=0.1,
                   subsample=0.9, colsample_bytree=0.9, random_state=42, verbose=-1)

def wfo_gbdt_all(min_train=24):
    pred_list = []
    for m in months:
        tr = panel[panel["trade_date"] < m]
        if tr["trade_date"].nunique() < min_train:
            continue
        val_months = sorted(tr["trade_date"].unique())[-3:]
        vm = tr["trade_date"].isin(val_months).values
        X = tr[C8_COLS].values; y = tr["fwd_20"].values
        mdl = lgb.LGBMRegressor(**GBDT_PARAMS)
        mdl.fit(X[~vm], y[~vm], eval_set=[(X[vm], y[vm])],
                callbacks=[lgb.early_stopping(50, verbose=False)])
        t_m = panel[panel["trade_date"] == m]
        pred_list.append(pd.DataFrame({"trade_date": m, "ts_code": t_m["ts_code"].values,
                                       "gbdt": mdl.predict(t_m[C8_COLS].values)}))
    return pd.concat(pred_list, ignore_index=True)

t2 = time.time()
gbdt_df = wfo_gbdt_all()
gbdt_df = gbdt_df.merge(panel[["trade_date","ts_code","fwd_20"]], on=["trade_date","ts_code"], how="left")
print(f"[GBDT WFO] 覆盖 {gbdt_df['trade_date'].nunique()} 月, {time.time()-t2:.0f}s")

# ================= Phase 1b: 时序 LSTM WFO 预测所有月份 =================
DEV = torch.device("cpu")
K = len(C8_COLS)

class LSTM_M(nn.Module):
    def __init__(self, n_feat, hidden=32, n_layers=1, dropout=0.1):
        super().__init__()
        self.lstm = nn.LSTM(n_feat, hidden, n_layers, batch_first=True, dropout=dropout, bidirectional=True)
        self.head = nn.Sequential(nn.LayerNorm(hidden*2), nn.Linear(hidden*2, hidden),
                                  nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, 1))
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :]).squeeze(-1)

def train_lstm(Xtr, ytr, Xval, yval, epochs=10, lr=1e-3, wd=1e-2, seed=42):
    torch.manual_seed(seed); np.random.seed(seed)
    model = LSTM_M(K).to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    Xt = torch.from_numpy(Xtr).to(DEV); yt = torch.from_numpy(ytr).to(DEV)
    Xv = torch.from_numpy(Xval).to(DEV); yv = torch.from_numpy(yval).to(DEV)
    dl = DataLoader(TensorDataset(Xt, yt), batch_size=1024, shuffle=True)
    best = np.inf; best_sd = None; pat = 0
    for ep in range(epochs):
        model.train()
        for bx, by in dl:
            opt.zero_grad(); nn.functional.mse_loss(model(bx), by).backward(); opt.step()
        model.eval()
        with torch.no_grad():
            vl = nn.functional.mse_loss(model(Xv), yv).item()
        if vl < best:
            best = vl; best_sd = {k: v.cpu().clone() for k, v in model.state_dict().items()}; pat = 0
        else:
            pat += 1
            if pat >= 3: break
    if best_sd: model.load_state_dict(best_sd)
    model.eval()
    return model

def wfo_lstm_all(win=36):
    """rolling 36 月窗口 + 月度重训, 预测所有月份 (Phase 1)"""
    pred_list = []
    n_predict = 0
    for m in months:
        idx = months.index(m)
        if idx < win:  # 前 win 个月不足训练窗口
            continue
        start_date = months[idx - win]
        tr = (seq_dates >= start_date) & (seq_dates < m)
        if tr.sum() < 2000:
            continue
        # 训练窗口内最后 3 个月做验证早停
        tr_dates = seq_dates[tr]
        val_months = sorted(np.unique(tr_dates))[-3:]
        val_mask = np.isin(seq_dates, val_months) & tr
        train_mask = tr & ~val_mask
        model = train_lstm(seq_X[train_mask], seq_y[train_mask],
                           seq_X[val_mask], seq_y[val_mask])
        te_mask = (seq_dates == m)
        with torch.no_grad():
            p = model(torch.from_numpy(seq_X[te_mask]).to(DEV)).cpu().numpy().ravel()
        pred_list.append(pd.DataFrame({"trade_date": m, "ts_code": seq_codes[te_mask], "lstm": p}))
        n_predict += 1
        if n_predict % 12 == 0:
            print(f"  [LSTM WFO] 已预测 {n_predict} 月 (最新 {m}), {time.time()-t0:.0f}s")
    return pd.concat(pred_list, ignore_index=True)

t3 = time.time()
lstm_df = wfo_lstm_all(win=24)  # 24 月窗口让 LSTM 尽早覆盖, 保证 meta 层有足够历史
lstm_df = lstm_df.merge(panel[["trade_date","ts_code","fwd_20"]], on=["trade_date","ts_code"], how="left")
print(f"[LSTM WFO] 覆盖 {lstm_df['trade_date'].nunique()} 月, {time.time()-t3:.0f}s")

# ================= Phase 2: 对齐 + 截面 rank + meta-learner =================
meta_df = gbdt_df[["trade_date","ts_code","gbdt","fwd_20"]].merge(
    lstm_df[["trade_date","ts_code","lstm"]], on=["trade_date","ts_code"], how="inner")
meta_df = meta_df.dropna(subset=["gbdt","lstm","fwd_20"])
meta_df["gbdt_rank"] = meta_df.groupby("trade_date")["gbdt"].rank(pct=True)
meta_df["lstm_rank"] = meta_df.groupby("trade_date")["lstm"].rank(pct=True)
print(f"[对齐] 交集 {len(meta_df):,} 行, {meta_df['trade_date'].nunique()} 月 (其中 OOS {meta_df[meta_df['trade_date']>=20230101]['trade_date'].nunique()} 月)")

def ic_summary(ics):
    icir = ics.mean() / (ics.std(ddof=1) + 1e-12) * np.sqrt(12)
    return ics.mean(), icir, (ics > 0).mean()

def q1q5(df, pred_col):
    d = df.copy()
    d["q"] = d.groupby("trade_date")[pred_col].transform(
        lambda s: pd.qcut(s.rank(method="first"), 5, labels=["Q1","Q2","Q3","Q4","Q5"]))
    piv = d.groupby("q", observed=True)["fwd_20"].mean()
    return piv["Q1"] - piv["Q5"]

# 滚动线性 meta-learner (岭回归, expanding window)
def wfo_meta_ridge(oos_months, alpha=1.0, min_train=24):
    pred_list = []
    weights = []
    for m in oos_months:
        tr = meta_df[meta_df["trade_date"] < m].dropna()
        if tr["trade_date"].nunique() < min_train:
            continue
        X = tr[["gbdt_rank","lstm_rank"]].values
        y = tr["fwd_20"].values
        mdl = Ridge(alpha=alpha, fit_intercept=True).fit(X, y)
        weights.append((mdl.coef_[0], mdl.coef_[1]))
        t_m = meta_df[meta_df["trade_date"] == m].dropna()
        p = mdl.predict(t_m[["gbdt_rank","lstm_rank"]].values)
        pred_list.append(pd.DataFrame({"trade_date": m, "ts_code": t_m["ts_code"].values,
                                       "pred": p, "fwd_20": t_m["fwd_20"].values}))
    df = pd.concat(pred_list, ignore_index=True)
    w = np.array(weights)
    return df, w

t4 = time.time()
ridge_df, w_ridge = wfo_meta_ridge(oos_months, min_train=12)  # 低维(2特征)回归, 12月足够
w_g, w_l = w_ridge.mean(axis=0)
print(f"[Meta Ridge] 平均权重 gbdt={w_g:+.3f} lstm={w_l:+.3f} (OOS 逐月滚动学习), {time.time()-t4:.0f}s")

# ================= 汇总对比 =================
def build_row(label, df, pred_col):
    oos = df[df["trade_date"] >= 20230101]
    ics = monthly_ic(oos, pred_col)
    m, icir, pos = ic_summary(ics)
    return {"model": label, "ic": m, "icir": icir, "pos_rate": pos, "q1_q5": q1q5(oos, pred_col),
            "n_obs": len(oos)}

oos_meta = meta_df[meta_df["trade_date"] >= 20230101].copy()
oos_meta["ens_eq"] = 0.5 * oos_meta["gbdt_rank"] + 0.5 * oos_meta["lstm_rank"]

rows = [
    build_row("GBDT_only", oos_meta, "gbdt_rank"),
    build_row("LSTM_only", oos_meta, "lstm_rank"),
    build_row("Equal_Weight", oos_meta, "ens_eq"),
    build_row("Ridge_Meta", ridge_df, "pred"),
]

print("\n=== OOS 融合对比 (2023+) ===")
for r in rows:
    print(f"  {r['model']:>14} | IC={r['ic']:+.4f} ICIR={r['icir']:+.2f} "
          f"Q1-Q5={r['q1_q5']:+.2f} 正率={r['pos_rate']*100:.0f}% (n={r['n_obs']:,})")

pd.DataFrame(rows).to_csv(OUT, index=False, encoding="utf-8-sig")

# 关键结论: 岭回归融合 vs 纯 GBDT 的 IC 差
d_ic = rows[3]["ic"] - rows[0]["ic"]
d_icir = rows[3]["icir"] - rows[0]["icir"]
print(f"\n[结论] 岭回归融合 vs 纯 GBDT: IC 差={d_ic:+.4f}, ICIR 差={d_icir:+.2f}")
print(f"  (正=融合有提升; 学习权重 gbdt={w_g:+.3f} / lstm={w_l:+.3f})")
print(f"[保存] {OUT}  总耗时 {time.time()-t0:.0f}s")

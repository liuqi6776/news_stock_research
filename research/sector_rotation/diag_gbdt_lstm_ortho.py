# -*- coding: utf-8 -*-
"""诊断: GBDT(C8) vs 时序 LSTM 的互补性 (相关性 + 残差 IC)

回答核心问题: LSTM 是否有 GBDT 之外的信息?
- 若 |corr| > 0.8 且残差 IC ≈ 0 → 融合无意义
- 若 |corr| < 0.5 且残差 IC 显著 > 0 → 值得做 stacking

口径: 快速预判版 (单次训练 2020-2022 -> OOS 2023-2025), 用于判断互补性,
     不作为最终策略指标。若 LSTM 有希望再升级为完整 WFO。
"""
import os, time, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.linear_model import LinearRegression

ROOT = r"c:\Users\liuqi\quant_system_v2"
PANEL = os.path.join(ROOT, "research", "sector_rotation", "stock_ml_panel_72m.parquet")

C7_COLS = ["ivol", "ret_1m", "momentum_20", "volatility_20", "alpha_006", "alpha_012", "enh4_score"]
CHIP_COLS = ["vwap_20", "float_pnl_20", "chip_shift_5"]
CHIP_RESID = ["vwap_20_resid", "float_pnl_20_resid", "chip_shift_5_resid"]
BASE_COLS = ["ivol", "ret_1m", "momentum_20", "volatility_20", "alpha_006", "alpha_012"]
C8_COLS = C7_COLS + CHIP_RESID

T_SEQ = 12  # LSTM 时间步: 过去 12 个月

def zscore(s): return (s - s.mean()) / (s.std(ddof=1) + 1e-12)
def winsorize(s, lo=0.01, hi=0.99):
    if s.notna().sum() < 5: return s
    a, b = s.quantile([lo, hi]); return s.clip(a, b)

t0 = time.time()
panel = pd.read_parquet(PANEL)

# ---- 预处理 (与 diag_model_compare_dl.py 一致) ----
for c in ["roe", "or_yoy"]:
    panel[c] = panel.groupby("trade_date")[c].transform(lambda s: winsorize(s))
panel["roe"] = panel["roe"].fillna(-99.0)
panel["or_yoy"] = panel["or_yoy"].fillna(-99.0)
g = panel.groupby("trade_date")
panel["enh4_score"] = (-0.40 * g["ivol"].rank(pct=True)
                       -0.35 * g["ret_1m"].rank(pct=True)
                       +0.15 * g["roe"].rank(pct=True)
                       +0.05 * g["or_yoy"].rank(pct=True))
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
        lr = LinearRegression(fit_intercept=True).fit(Xb[mask], y[mask])
        panel.loc[grp.index[mask], f"{c}_resid"] = -(y - lr.predict(Xb))
for c in CHIP_RESID:
    panel[c] = panel.groupby("trade_date")[c].transform(lambda s: winsorize(s))
    panel[c] = panel.groupby("trade_date")[c].transform(zscore)
panel = panel.dropna(subset=C8_COLS + ["fwd_20"])
print(f"[数据] {len(panel):,} 行, {panel['trade_date'].nunique()} 月, 预处理 {time.time()-t0:.0f}s")

# ---- 构造时序样本 (每股票滑动窗口 [t-T+1 .. t] -> fwd_20[t]) ----
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
seq_X = np.stack(seq_X)   # [N, T, K]
seq_y = np.array(seq_y)
seq_dates = np.array(seq_dates)
seq_codes = np.array(seq_codes)
print(f"[时序样本] {seq_X.shape[0]:,} 条 (T={T_SEQ}, K={len(C8_COLS)}), 构造 {time.time()-t1:.0f}s")

# ---- 划分 train/test (单次: 2020-2022 训, 2023+ 测) ----
CUT = 20230101
tr_mask = panel["trade_date"] < CUT
te_mask = panel["trade_date"] >= CUT
seq_tr = seq_dates < CUT
seq_te = seq_dates >= CUT

# ========== GBDT C8 ==========
t2 = time.time()
Xtr = panel.loc[tr_mask, C8_COLS].values; ytr = panel.loc[tr_mask, "fwd_20"].values
Xte = panel.loc[te_mask, C8_COLS].values
mdl_gbdt = lgb.LGBMRegressor(n_estimators=500, learning_rate=0.05, num_leaves=7,
                             max_depth=3, min_child_samples=80, reg_lambda=2.0, reg_alpha=0.1,
                             subsample=0.9, colsample_bytree=0.9, random_state=42, verbose=-1)
mdl_gbdt.fit(Xtr, ytr)
gbdt_pred = mdl_gbdt.predict(Xte)
gbdt_df = pd.DataFrame({"trade_date": panel.loc[te_mask, "trade_date"].values,
                        "ts_code": panel.loc[te_mask, "ts_code"].values,
                        "gbdt": gbdt_pred,
                        "fwd_20": panel.loc[te_mask, "fwd_20"].values})
print(f"[GBDT] 训练 {Xtr.shape[0]:,} 行, 预测 {Xte.shape[0]:,} 行, {time.time()-t2:.0f}s")

# ========== 时序 LSTM ==========
t3 = time.time()
try:
    import torch, torch.nn as nn
    from torch.utils.data import TensorDataset, DataLoader
except Exception as e:
    print(f"[LSTM] torch 不可用: {e}")
    raise SystemExit(1)

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

def train_lstm(Xtr_, ytr_, epochs=12, lr=1e-3, wd=1e-2):
    model = LSTM_M(K, hidden=32).to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    Xt = torch.from_numpy(Xtr_).to(DEV); yt = torch.from_numpy(ytr_).to(DEV)
    # 用最后 10% 做验证早停
    n = len(Xt); nv = int(n * 0.1)
    idx = np.random.RandomState(42).permutation(n)
    Xv, yv = Xt[idx[:nv]], yt[idx[:nv]]
    Xt_, yt_ = Xt[idx[nv:]], yt[idx[nv:]]
    dl = DataLoader(TensorDataset(Xt_, yt_), batch_size=1024, shuffle=True)
    best = np.inf; best_sd = None; pat = 0
    for ep in range(epochs):
        model.train()
        for bx, by in dl:
            opt.zero_grad(); nn.functional.mse_loss(model(bx), by).backward(); opt.step()
        model.eval()
        with torch.no_grad(): vl = nn.functional.mse_loss(model(Xv), yv).item()
        if vl < best:
            best = vl; best_sd = {k: v.cpu().clone() for k, v in model.state_dict().items()}; pat = 0
        else:
            pat += 1
            if pat >= 3: break
    if best_sd: model.load_state_dict(best_sd)
    model.eval()
    return model

model = train_lstm(seq_X[seq_tr], seq_y[seq_tr])
Xte_3d = seq_X[seq_te]
with torch.no_grad():
    lstm_pred = model(torch.from_numpy(Xte_3d).to(DEV)).cpu().numpy().ravel()
lstm_df = pd.DataFrame({"trade_date": seq_dates[seq_te],
                        "ts_code": seq_codes[seq_te],
                        "lstm": lstm_pred})
lstm_df = lstm_df.merge(panel[["trade_date", "ts_code", "fwd_20"]], on=["trade_date", "ts_code"], how="left")
print(f"[LSTM] 训练 {seq_X[seq_tr].shape[0]:,} 条, 预测 {seq_X[seq_te].shape[0]:,} 条, {time.time()-t3:.0f}s")

# ========== 对齐 & 计算 ==========
def monthly_ic(df, pred_col):
    out = {}
    for dt, gg in df.groupby("trade_date"):
        if len(gg) < 30: continue
        out[dt] = gg[pred_col].rank().corr(gg["fwd_20"].rank())
    return pd.Series(out).dropna()

def ic_summary(ics):
    icir = ics.mean() / (ics.std(ddof=1) + 1e-12) * np.sqrt(12)
    return ics.mean(), icir, (ics > 0).mean()

merged = gbdt_df[["trade_date", "ts_code", "gbdt", "fwd_20"]].merge(
    lstm_df[["trade_date", "ts_code", "lstm"]], on=["trade_date", "ts_code"], how="inner")
print(f"\n[对齐] 交集 {len(merged):,} 行, {merged['trade_date'].nunique()} 月")

ics_gbdt = monthly_ic(merged, "gbdt")
ics_lstm = monthly_ic(merged, "lstm")
print(f"  GBDT IC={ics_gbdt.mean():+.4f} ICIR={ic_summary(ics_gbdt)[1]:+.2f} 正率={ic_summary(ics_gbdt)[2]*100:.0f}%")
print(f"  LSTM IC={ics_lstm.mean():+.4f} ICIR={ic_summary(ics_lstm)[1]:+.2f} 正率={ic_summary(ics_lstm)[2]*100:.0f}%")

# 相关性: 每月 gbdt 与 lstm 的 rank corr 均值
corrs = []
for dt, gg in merged.groupby("trade_date"):
    if len(gg) < 30: continue
    corrs.append(gg["gbdt"].rank().corr(gg["lstm"].rank()))
corr_avg = np.nanmean(corrs)
print(f"  月度预测相关性 (rank corr) 均值 = {corr_avg:+.3f}")

# 残差 IC: lstm 对 gbdt 截面正交后, 残差是否仍有 IC
resid_ics = {}
for dt, gg in merged.groupby("trade_date"):
    if len(gg) < 30: continue
    Xb = gg[["gbdt"]].values
    y = gg["lstm"].values
    lr = LinearRegression().fit(Xb, y)
    resid = y - lr.predict(Xb)
    gg = gg.assign(resid=resid)
    resid_ics[dt] = gg["resid"].rank().corr(gg["fwd_20"].rank())
resid_ics = pd.Series(resid_ics).dropna()
print(f"  LSTM 对 GBDT 正交后残差 IC = {resid_ics.mean():+.4f} "
      f"ICIR={ic_summary(resid_ics)[1]:+.2f} 正率={ic_summary(resid_ics)[2]*100:.0f}%")

print(f"\n[总耗时] {time.time()-t0:.0f}s")

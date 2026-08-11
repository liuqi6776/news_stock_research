# -*- coding: utf-8 -*-
"""全因子 DL 实验: merge 大面板 + C8精炼因子 → 全部因子(不去重不筛选)

模型:
  1. GBDT d10 (深树, 全因子)
  2. MLP 深层 5层 (256,128,64,32,16)
  3. FT-Transformer (d_model=128, n_layers=4)
  4. LSTM (把因子序列化为时序输入)
  5. C8 GBDT d3 (baseline 对照)
"""
import os, time, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

ROOT = r"c:\Users\liuqi\quant_system_v2"
PANEL_LARGE = os.path.join(ROOT, "research", "sector_rotation", "stock_ml_panel_large_72m.parquet")
PANEL_ORIG  = os.path.join(ROOT, "research", "sector_rotation", "stock_ml_panel_72m.parquet")
OUT_CSV = os.path.join(ROOT, "research", "sector_rotation", "results", "dl_full_features_ic.csv")

C7_COLS = ["ivol","ret_1m","momentum_20","volatility_20","alpha_006","alpha_012","enh4_score"]
CHIP_COLS = ["vwap_20","float_pnl_20","chip_shift_5"]
CHIP_RESID = ["vwap_20_resid","float_pnl_20_resid","chip_shift_5_resid"]
BASE_COLS = ["ivol","ret_1m","momentum_20","volatility_20","alpha_006","alpha_012"]
C8_COLS = C7_COLS + CHIP_RESID

def zscore(s): return (s - s.mean()) / (s.std(ddof=1) + 1e-12)
def winsorize(s, lo=0.01, hi=0.99):
    if s.notna().sum() < 5: return s
    a, b = s.quantile([lo, hi]); return s.clip(a, b)

t0 = time.time()

# === 1. 大面板 ===
large = pd.read_parquet(PANEL_LARGE)
exclude_large = {"trade_date","ts_code","open","high","low","close","pct_chg","vol",
                 "industry","is_traditional","fwd_20"}
large_feats = [c for c in large.columns if c not in exclude_large]
print(f"[大面板] {len(large):,} 行, {len(large_feats)} 因子")

# === 2. 原面板 → 计算 enh4_score + 筹码残差 ===
orig = pd.read_parquet(PANEL_ORIG)
for c in ["roe","or_yoy"]:
    orig[c] = orig.groupby("trade_date")[c].transform(lambda s: winsorize(s))
orig["roe"] = orig["roe"].fillna(-99.0)
orig["or_yoy"] = orig["or_yoy"].fillna(-99.0)
gg = orig.groupby("trade_date")
orig["enh4_score"] = (-0.40*gg["ivol"].rank(pct=True) -0.35*gg["ret_1m"].rank(pct=True)
                      +0.15*gg["roe"].rank(pct=True) +0.05*gg["or_yoy"].rank(pct=True))
ALL_RAW = list(set(C7_COLS + CHIP_COLS) - {"enh4_score"})
for c in ALL_RAW:
    orig[c] = orig.groupby("trade_date")[c].transform(lambda s: winsorize(s))
    orig[c] = orig.groupby("trade_date")[c].transform(zscore)
orig["fwd_20"] = orig.groupby("trade_date")["fwd_20"].transform(lambda s: winsorize(s))
orig = orig.dropna(subset=C7_COLS + CHIP_COLS + ["fwd_20"])
for c in CHIP_COLS: orig[f"{c}_resid"] = np.nan
for dt, grp in orig.groupby("trade_date"):
    if len(grp) < 50: continue
    Xb = grp[BASE_COLS].values
    for c in CHIP_COLS:
        y = grp[c].values
        mask = np.isfinite(y) & np.all(np.isfinite(Xb), axis=1)
        if mask.sum() < 50: continue
        lr = LinearRegression(fit_intercept=True)
        lr.fit(Xb[mask], y[mask])
        orig.loc[grp.index[mask], f"{c}_resid"] = -(y - lr.predict(Xb))
for c in CHIP_RESID:
    orig[c] = orig.groupby("trade_date")[c].transform(lambda s: winsorize(s))
    orig[c] = orig.groupby("trade_date")[c].transform(zscore)
orig_extra = orig[["trade_date","ts_code"] + C8_COLS].dropna(subset=C8_COLS)
print(f"[原面板] {len(orig_extra):,} 行, C8 {len(C8_COLS)} 因子 (含 enh4_score + 筹码残差)")

# === 3. Merge ===
# 大面板里有同名列(ivol, ret_1m 等), 只从 orig 加 enh4_score + 3个残差
extra_cols = ["enh4_score"] + CHIP_RESID
panel = large.merge(
    orig_extra[["trade_date","ts_code"] + extra_cols],
    on=["trade_date","ts_code"], how="inner")
print(f"[Merge] {len(panel):,} 行")

# 全因子列表
exclude_all = {"trade_date","ts_code","open","high","low","close","pct_chg","vol",
               "industry","is_traditional","fwd_20"}
all_feats = [c for c in panel.columns if c not in exclude_all]
print(f"[全因子] {len(all_feats)} 个 (不去重不筛选)")

# 截面标准化
for f in all_feats:
    panel[f] = panel.groupby("trade_date")[f].transform(lambda s: winsorize(s))
    panel[f] = panel.groupby("trade_date")[f].transform(zscore)
panel["fwd_20"] = panel.groupby("trade_date")["fwd_20"].transform(lambda s: winsorize(s))
panel = panel.dropna(subset=all_feats + ["fwd_20"])

months = sorted(panel["trade_date"].unique())
oos_months = [m for m in months if m >= 20230101]
print(f"[OOS] {len(oos_months)} 月, 总耗时 {time.time()-t0:.0f}s\n")

# === 4. WFO 框架 ===
def mic(df, pred_col="pred"):
    out = {}
    for dt, gg in df.groupby("trade_date"):
        if len(gg) < 50: continue
        out[dt] = gg[pred_col].rank().corr(gg["fwd_20"].rank())
    s = pd.Series(out).dropna()
    return s.mean(), s.std(ddof=1), (s>0).mean()

def wfo(feats, model_factory, need_scale=False, label="", is_lstm=False):
    t1 = time.time()
    preds = []
    for m_idx, m in enumerate(oos_months):
        tr = panel[panel["trade_date"] < m]
        if tr["trade_date"].nunique() < 24: continue
        vm = tr["trade_date"].isin(sorted(tr["trade_date"].unique())[-3:]).values
        Xa, ya = tr[feats].values, tr["fwd_20"].values
        Xtr, ytr, Xv, yv = Xa[~vm], ya[~vm], Xa[vm], ya[vm]
        sc = StandardScaler() if need_scale else None
        if need_scale:
            Xtr = sc.fit_transform(Xtr); Xv = sc.transform(Xv)
        if is_lstm:
            # LSTM: reshape to (N, seq_len=n_feat, 1)
            Xtr_3d = Xtr.reshape(Xtr.shape[0], len(feats), 1).astype(np.float32)
            Xv_3d  = Xv.reshape(Xv.shape[0], len(feats), 1).astype(np.float32)
            mdl = model_factory(Xtr_3d, ytr, Xv_3d, yv, n_feat=len(feats))
        else:
            mdl = model_factory(Xtr, ytr, Xv, yv)
        tt = panel[panel["trade_date"] == m]
        Xt = tt[feats].values
        if need_scale: Xt = sc.transform(Xt)
        if is_lstm:
            Xt_3d = Xt.reshape(Xt.shape[0], len(feats), 1).astype(np.float32)
            pred = mdl.predict(Xt_3d)
        else:
            pred = mdl.predict(Xt)
        preds.append(pd.DataFrame({"trade_date":m, "ts_code":tt["ts_code"],
                                   "pred":pred, "fwd_20":tt["fwd_20"].values}))
        if (m_idx+1) % 6 == 0:
            print(f"    {label} WFO {m_idx+1}/{len(oos_months)} 月, {time.time()-t1:.0f}s")
    df = pd.concat(preds, ignore_index=True)
    ic, std, pos = mic(df)
    icir = ic / (std + 1e-12) * np.sqrt(12)
    df["q"] = df.groupby("trade_date")["pred"].transform(
        lambda s: pd.qcut(s.rank(method="first"), 5, labels=["Q1","Q2","Q3","Q4","Q5"]))
    piv = df.groupby("q", observed=True)["fwd_20"].mean()
    return {"ic": ic, "icir": icir, "pos_rate": pos,
            "q1q5": piv["Q1"]-piv["Q5"], "sec": int(time.time()-t1), "nfeat": len(feats)}

# === 5. 模型工厂 ===
def gbdt_d10(Xtr, ytr, Xv, yv):
    m = lgb.LGBMRegressor(n_estimators=1000, learning_rate=0.02, num_leaves=63,
                          max_depth=10, min_child_samples=120, reg_lambda=5.0, reg_alpha=1.0,
                          subsample=0.7, colsample_bytree=0.5, random_state=42, verbose=-1)
    m.fit(Xtr, ytr, eval_set=[(Xv, yv)], callbacks=[lgb.early_stopping(50, verbose=False)])
    return m

def gbdt_d3(Xtr, ytr, Xv, yv):
    m = lgb.LGBMRegressor(n_estimators=500, learning_rate=0.05, num_leaves=7,
                          max_depth=3, min_child_samples=80, reg_lambda=2.0, reg_alpha=0.1,
                          subsample=0.9, colsample_bytree=0.9, random_state=42, verbose=-1)
    m.fit(Xtr, ytr, eval_set=[(Xv, yv)], callbacks=[lgb.early_stopping(50, verbose=False)])
    return m

def mlp_deep(Xtr, ytr, Xv, yv):
    m = MLPRegressor((256,128,64,32,16), activation="relu", solver="adam", alpha=0.5,
                     batch_size=512, learning_rate_init=0.001, max_iter=200,
                     early_stopping=True, n_iter_no_change=12, validation_fraction=0.1,
                     random_state=42)
    m.fit(Xtr, ytr)
    return m

# PyTorch models
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
DEV = torch.device("cpu")

class FTTransformer(nn.Module):
    def __init__(self, n_feat, d_model=128, n_heads=8, n_layers=4, dropout=0.2):
        super().__init__()
        self.proj = nn.Linear(1, d_model)
        self.cls = nn.Parameter(torch.zeros(1, 1, d_model))
        enc = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads,
            dim_feedforward=d_model*4, dropout=dropout, activation="gelu",
            batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(enc, num_layers=n_layers)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_model, 1))
    def forward(self, x):
        B = x.shape[0]
        z = self.proj(x.unsqueeze(-1))
        cls = self.cls.expand(B, -1, -1)
        z = torch.cat([cls, z], dim=1)
        z = self.enc(z)
        return self.head(z[:, 0, :]).squeeze(-1)

class LSTMModel(nn.Module):
    """LSTM: 把因子序列化为 (batch, n_feat, 1) 的时序输入"""
    def __init__(self, n_feat, hidden=128, n_layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=hidden, num_layers=n_layers,
                            batch_first=True, dropout=dropout, bidirectional=True)
        self.head = nn.Sequential(
            nn.LayerNorm(hidden*2),
            nn.Linear(hidden*2, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, 1))
    def forward(self, x):
        # x: (B, n_feat, 1)
        out, (hn, cn) = self.lstm(x)
        # 取最后一个 time step 的双向输出
        last = out[:, -1, :]  # (B, hidden*2)
        return self.head(last).squeeze(-1)

def _train_torch(model, Xtr, ytr, Xv, yv, epochs=80, lr=1e-3, wd=1e-2):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", patience=5, factor=0.5)
    yt = torch.from_numpy(ytr.astype(np.float32)).to(DEV)
    yve = torch.from_numpy(yv.astype(np.float32)).to(DEV)
    ds = TensorDataset(Xtr, yt)
    dl = DataLoader(ds, batch_size=512, shuffle=True)
    best_vl = np.inf; best_sd = None; pat = 0
    for ep in range(epochs):
        model.train()
        for bx, by in dl:
            opt.zero_grad()
            loss = F.mse_loss(model(bx), by)
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            vl = F.mse_loss(model(Xv), yve).item()
        if vl < best_vl:
            best_vl = vl
            best_sd = {k: v.detach().cpu().clone() for k,v in model.state_dict().items()}
            pat = 0
        else:
            pat += 1
            if pat >= 12: break
        sched.step(vl)
    if best_sd: model.load_state_dict(best_sd)
    model.eval()
    return model

def ftt_factory(n_feat):
    def _f(Xtr, ytr, Xv, yv):
        model = FTTransformer(n_feat, d_model=128, n_heads=8, n_layers=4, dropout=0.2).to(DEV)
        Xt = torch.from_numpy(Xtr.astype(np.float32)).to(DEV)
        Xve = torch.from_numpy(Xv.astype(np.float32)).to(DEV)
        return _train_torch(model, Xt, ytr, Xve, yv, epochs=80)
    return _f

def lstm_factory(Xtr_3d, ytr, Xv_3d, yv, n_feat):
    model = LSTMModel(n_feat, hidden=128, n_layers=2, dropout=0.2).to(DEV)
    Xt = torch.from_numpy(Xtr_3d).to(DEV)
    Xve = torch.from_numpy(Xv_3d).to(DEV)
    model = _train_torch(model, Xt, ytr, Xve, yv, epochs=60)
    class _P:
        def __init__(self, m): self.m = m
        def predict(self, X):
            with torch.no_grad():
                return self.m(torch.from_numpy(X).to(DEV)).cpu().numpy().ravel()
    return _P(model)

# === 6. 跑对比 ===
NF = len(all_feats)
print(f"=== WFO OOS IC 对比 (全因子 {NF} 个, 2023+ {len(oos_months)} 月) ===\n")

rows = []

# GBDT d10
print(f">> GBDT d10 (深树, {NF} 因子)...")
r = wfo(all_feats, gbdt_d10, need_scale=False, label="GBDT_d10")
print(f"  {'GBDT_d10':>22} | IC={r['ic']:+.4f} ICIR={r['icir']:+.2f} Q1-Q5={r['q1q5']:+.2f} 正率={r['pos_rate']*100:.0f}% {r['sec']}s\n")
rows.append({"label":"GBDT_d10", **r})

# MLP 深层
print(f">> MLP 深层5层 ({NF} 因子)...")
r = wfo(all_feats, mlp_deep, need_scale=True, label="MLP_deep5")
print(f"  {'MLP_deep5':>22} | IC={r['ic']:+.4f} ICIR={r['icir']:+.2f} Q1-Q5={r['q1q5']:+.2f} 正率={r['pos_rate']*100:.0f}% {r['sec']}s\n")
rows.append({"label":"MLP_deep5", **r})

# FT-Transformer
print(f">> FT-Transformer (d_model=128, 4层, {NF} 因子)...")
r = wfo(all_feats, ftt_factory(NF), need_scale=True, label="FTT_128_4L")
print(f"  {'FTT_128_4L':>22} | IC={r['ic']:+.4f} ICIR={r['icir']:+.2f} Q1-Q5={r['q1q5']:+.2f} 正率={r['pos_rate']*100:.0f}% {r['sec']}s\n")
rows.append({"label":"FTT_128_4L", **r})

# LSTM
print(f">> LSTM (hidden=128, 2层, {NF} 因子序列化)...")
r = wfo(all_feats, lstm_factory, need_scale=True, label="LSTM_128", is_lstm=True)
print(f"  {'LSTM_128':>22} | IC={r['ic']:+.4f} ICIR={r['icir']:+.2f} Q1-Q5={r['q1q5']:+.2f} 正率={r['pos_rate']*100:.0f}% {r['sec']}s\n")
rows.append({"label":"LSTM_128", **r})

# C8 baseline
print(f">> C8 GBDT d3 (baseline, 10因子)...")
r = wfo(C8_COLS, gbdt_d3, need_scale=False, label="C8_GBDT_d3")
print(f"  {'C8_GBDT_d3':>22} | IC={r['ic']:+.4f} ICIR={r['icir']:+.2f} Q1-Q5={r['q1q5']:+.2f} 正率={r['pos_rate']*100:.0f}% {r['sec']}s\n")
rows.append({"label":"C8_GBDT_d3", **r})

# === 汇总 ===
print(f"\n{'='*75}")
print(f"{'模型':>24} {'因子数':>5} {'IC':>8} {'ICIR':>7} {'Q1-Q5':>7} {'正率':>5}")
print(f"{'-'*75}")
for r in sorted(rows, key=lambda x: x["ic"], reverse=True):
    print(f"{r['label']:>24} {r['nfeat']:>5} {r['ic']:+.4f}  {r['icir']:+.2f}   {r['q1q5']:+.2f}   {r['pos_rate']*100:.0f}%")
print(f"{'='*75}")

best = max(rows, key=lambda x: x["ic"])
base = next(r for r in rows if r["label"]=="C8_GBDT_d3")
print(f"\n🏆 最优: {best['label']} IC={best['ic']:+.4f}")
print(f"   vs C8 baseline: ΔIC={best['ic']-base['ic']:+.4f}")
if best["ic"] - base["ic"] > 0.005:
    print(f"   ✅ DL 全因子 显著优于 C8 GBDT baseline!")
    print(f"   → 下一步: 用 {best['label']} 跑完整回测")
elif best["ic"] - base["ic"] > -0.003:
    print(f"   ≈ 持平 C8 baseline")
else:
    print(f"   ⚠️ 仍不如 C8 baseline (C8 精炼因子信息密度更高)")

pd.DataFrame(rows).to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
print(f"\n[保存] {OUT_CSV}")
print(f"[总耗时] {time.time()-t0:.0f}s")

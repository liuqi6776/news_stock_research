# -*- coding: utf-8 -*-
"""GPU 版: FTT + LSTM 全因子 (83因子), RTX 3060 Ti CUDA"""
import os, time, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

ROOT = r"c:\Users\liuqi\quant_system_v2"
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[GPU] {DEV} | {torch.cuda.get_device_name(0) if DEV.type=='cuda' else 'CPU'}")

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
large = pd.read_parquet(os.path.join(ROOT, "research/sector_rotation/stock_ml_panel_large_72m.parquet"))
excl = {"trade_date","ts_code","open","high","low","close","pct_chg","vol","industry","is_traditional","fwd_20"}

orig = pd.read_parquet(os.path.join(ROOT, "research/sector_rotation/stock_ml_panel_72m.parquet"))
for c in ["roe","or_yoy"]:
    orig[c] = orig.groupby("trade_date")[c].transform(lambda s: winsorize(s))
orig["roe"] = orig["roe"].fillna(-99.0); orig["or_yoy"] = orig["or_yoy"].fillna(-99.0)
gg = orig.groupby("trade_date")
orig["enh4_score"] = (-0.40*gg["ivol"].rank(pct=True) -0.35*gg["ret_1m"].rank(pct=True)
                      +0.15*gg["roe"].rank(pct=True) +0.05*gg["or_yoy"].rank(pct=True))
for c in ["ivol","ret_1m","momentum_20","volatility_20","alpha_006","alpha_012","vwap_20","float_pnl_20","chip_shift_5"]:
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
        lr = LinearRegression().fit(Xb[mask], y[mask])
        orig.loc[grp.index[mask], f"{c}_resid"] = -(y - lr.predict(Xb))
for c in CHIP_RESID:
    orig[c] = orig.groupby("trade_date")[c].transform(lambda s: winsorize(s))
    orig[c] = orig.groupby("trade_date")[c].transform(zscore)
orig_extra = orig[["trade_date","ts_code"] + C8_COLS].dropna(subset=C8_COLS)

panel = large.merge(orig_extra[["trade_date","ts_code","enh4_score"] + CHIP_RESID],
                    on=["trade_date","ts_code"], how="inner")
all_feats = [c for c in panel.columns if c not in excl]
for f in all_feats:
    panel[f] = panel.groupby("trade_date")[f].transform(lambda s: winsorize(s))
    panel[f] = panel.groupby("trade_date")[f].transform(zscore)
panel["fwd_20"] = panel.groupby("trade_date")["fwd_20"].transform(lambda s: winsorize(s))
panel = panel.dropna(subset=all_feats + ["fwd_20"])
oos_months = sorted([m for m in panel["trade_date"].unique() if m >= 20230101])
print(f"[数据] {len(panel):,} 行, {len(all_feats)} 因子, OOS {len(oos_months)} 月, {time.time()-t0:.0f}s")

def mic(df):
    out = {}
    for dt, gg in df.groupby("trade_date"):
        if len(gg) < 50: continue
        out[dt] = gg["pred"].rank().corr(gg["fwd_20"].rank())
    s = pd.Series(out).dropna()
    return s.mean(), s.std(ddof=1), (s>0).mean()

def wfo(feats, factory, need_scale=False, label="", is_3d=False):
    t1 = time.time(); preds = []
    for m_idx, m in enumerate(oos_months):
        tr = panel[panel["trade_date"] < m]
        if tr["trade_date"].nunique() < 24: continue
        vm = tr["trade_date"].isin(sorted(tr["trade_date"].unique())[-3:]).values
        Xa, ya = tr[feats].values, tr["fwd_20"].values
        Xtr, ytr, Xv, yv = Xa[~vm], ya[~vm], Xa[vm], ya[vm]
        sc = StandardScaler() if need_scale else None
        if need_scale: Xtr = sc.fit_transform(Xtr); Xv = sc.transform(Xv)
        if is_3d:
            Xtr_3d = Xtr.reshape(-1, len(feats), 1).astype(np.float32)
            Xv_3d  = Xv.reshape(-1, len(feats), 1).astype(np.float32)
            mdl = factory(Xtr_3d, ytr, Xv_3d, yv, n_feat=len(feats))
        else:
            mdl = factory(Xtr, ytr, Xv, yv)
        tt = panel[panel["trade_date"] == m]
        Xt = tt[feats].values
        if need_scale: Xt = sc.transform(Xt)
        if is_3d:
            Xt_3d = Xt.reshape(-1, len(feats), 1).astype(np.float32)
            pred = mdl.predict(Xt_3d)
        else:
            pred = mdl.predict(Xt)
        preds.append(pd.DataFrame({"trade_date":m,"pred":pred,"fwd_20":tt["fwd_20"].values}))
        if (m_idx+1) % 6 == 0: print(f"    {label} WFO {m_idx+1}/{len(oos_months)}, {time.time()-t1:.0f}s")
    df = pd.concat(preds, ignore_index=True)
    ic, std, pos = mic(df)
    icir = ic / (std + 1e-12) * np.sqrt(12)
    df["q"] = df.groupby("trade_date")["pred"].transform(
        lambda s: pd.qcut(s.rank(method="first"), 5, labels=["Q1","Q2","Q3","Q4","Q5"]))
    piv = df.groupby("q", observed=True)["fwd_20"].mean()
    return {"ic": ic, "icir": icir, "pos_rate": pos,
            "q1q5": piv["Q1"]-piv["Q5"], "sec": int(time.time()-t1), "nfeat": len(feats)}

# === 模型 ===
class FTTransformer(nn.Module):
    def __init__(self, n_feat, d_model=128, n_heads=8, n_layers=4, dropout=0.2):
        super().__init__()
        self.proj = nn.Linear(1, d_model)
        self.cls = nn.Parameter(torch.zeros(1, 1, d_model))
        enc = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads,
            dim_feedforward=d_model*4, dropout=dropout, activation="gelu",
            batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(enc, num_layers=n_layers)
        self.head = nn.Sequential(nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_model, 1))
    def forward(self, x):
        B = x.shape[0]
        z = self.proj(x.unsqueeze(-1))
        cls = self.cls.expand(B, -1, -1)
        z = torch.cat([cls, z], dim=1)
        return self.head(self.enc(z)[:, 0, :]).squeeze(-1)

class LSTMModel(nn.Module):
    def __init__(self, n_feat, hidden=128, n_layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(1, hidden, n_layers, batch_first=True, dropout=dropout, bidirectional=True)
        self.head = nn.Sequential(nn.LayerNorm(hidden*2),
            nn.Linear(hidden*2, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, 1))
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :]).squeeze(-1)

def _train_gpu(model, Xtr, ytr, Xv, yv, epochs=50, lr=1e-3, wd=1e-2, patience=8):
    model = model.to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=4, factor=0.5)
    yt = torch.from_numpy(ytr.astype(np.float32)).to(DEV)
    yve = torch.from_numpy(yv.astype(np.float32)).to(DEV)
    dl = DataLoader(TensorDataset(Xtr, yt), batch_size=8192, shuffle=True)
    scaler = torch.amp.GradScaler() if DEV.type == 'cuda' else None
    best_vl = np.inf; best_sd = None; pat = 0
    for ep in range(epochs):
        model.train()
        for bx, by in dl:
            opt.zero_grad()
            if scaler:
                with torch.amp.autocast('cuda'):
                    loss = F.mse_loss(model(bx), by)
                scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            else:
                F.mse_loss(model(bx), by).backward(); opt.step()
        model.eval()
        with torch.no_grad(): vl = F.mse_loss(model(Xv), yve).item()
        if vl < best_vl: best_vl = vl; best_sd = {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}; pat = 0
        else:
            pat += 1
            if pat >= patience: break
        sched.step(vl)
    if best_sd: model.load_state_dict(best_sd)
    model.eval()
    class _P:
        def __init__(self, m): self.m = m
        def predict(self, X):
            with torch.no_grad():
                return self.m(torch.from_numpy(X.astype(np.float32)).to(DEV)).cpu().numpy().ravel()
    return _P(model)

NF = len(all_feats)
# 已有结果 (CPU 跑的)
rows = [
    {"label": "GBDT_d10 (83因子)", "ic": 0.0270, "icir": 0.93, "pos_rate": 0.69, "q1q5": -0.00, "nfeat": 83, "sec": 15},
    {"label": "MLP_deep5 (83因子)", "ic": 0.0758, "icir": 1.95, "pos_rate": 0.69, "q1q5": -0.01, "nfeat": 83, "sec": 293},
]

# === FT-Transformer (GPU, AMP混合精度) ===
print(f"\n>> FT-Transformer (d_model=64, 4 heads, 2 layers, 50 epochs, GPU+AMP)...")
def ftt_factory(Xtr, ytr, Xv, yv):
    m = FTTransformer(NF, d_model=64, n_heads=4, n_layers=2, dropout=0.15)
    Xt = torch.from_numpy(Xtr.astype(np.float32)).to(DEV)
    Xve = torch.from_numpy(Xv.astype(np.float32)).to(DEV)
    return _train_gpu(m, Xt, ytr, Xve, yv, epochs=50, patience=8)
r = wfo(all_feats, ftt_factory, need_scale=True, label="FTT_64_2L")
print(f"  {'FTT_64_2L':>22} | IC={r['ic']:+.4f} ICIR={r['icir']:+.2f} Q1-Q5={r['q1q5']:+.2f} 正率={r['pos_rate']*100:.0f}% {r['sec']}s")
rows.append({"label": "FT-Transformer (83因子)", **r})

# === LSTM (GPU, AMP混合精度) ===
print(f"\n>> LSTM (hidden=128, 2层双向, 50 epochs, GPU+AMP)...")
def lstm_factory(Xtr_3d, ytr, Xv_3d, yv, n_feat):
    m = LSTMModel(n_feat, hidden=128, n_layers=2, dropout=0.2)
    Xt = torch.from_numpy(Xtr_3d).to(DEV)
    Xve = torch.from_numpy(Xv_3d).to(DEV)
    return _train_gpu(m, Xt, ytr, Xve, yv, epochs=50, patience=8)
r = wfo(all_feats, lstm_factory, need_scale=True, label="LSTM_128", is_3d=True)
print(f"  {'LSTM_128':>22} | IC={r['ic']:+.4f} ICIR={r['icir']:+.2f} Q1-Q5={r['q1q5']:+.2f} 正率={r['pos_rate']*100:.0f}% {r['sec']}s")
rows.append({"label": "LSTM_128 (83因子)", **r})

# === C8 baseline ===
print(f"\n>> C8 GBDT d3 (baseline, 10因子)...")
def gbdt_d3(Xtr, ytr, Xv, yv):
    m = lgb.LGBMRegressor(n_estimators=500, learning_rate=0.05, num_leaves=7,
        max_depth=3, min_child_samples=80, reg_lambda=2.0, reg_alpha=0.1,
        subsample=0.9, colsample_bytree=0.9, random_state=42, verbose=-1)
    m.fit(Xtr, ytr, eval_set=[(Xv, yv)], callbacks=[lgb.early_stopping(50, verbose=False)])
    return m
r = wfo(C8_COLS, gbdt_d3, need_scale=False, label="C8_GBDT_d3")
print(f"  {'C8_GBDT_d3':>22} | IC={r['ic']:+.4f} ICIR={r['icir']:+.2f} Q1-Q5={r['q1q5']:+.2f} 正率={r['pos_rate']*100:.0f}% {r['sec']}s")
rows.append({"label": "C8_GBDT_d3 (baseline)", **r})

# === 汇总 ===
print(f"\n{'='*80}")
print(f"{'模型':>30} {'因子数':>5} {'IC':>8} {'ICIR':>7} {'Q1-Q5':>7} {'正率':>5} {'耗时':>5}")
print(f"{'-'*80}")
for r in sorted(rows, key=lambda x: x["ic"], reverse=True):
    print(f"{r['label']:>30} {r['nfeat']:>5} {r['ic']:+.4f}  {r['icir']:+.2f}   {r['q1q5']:+.2f}   {r['pos_rate']*100:.0f}%  {r['sec']}s")
print(f"{'='*80}")

best = max(rows, key=lambda x: x["ic"])
base = next(r for r in rows if "baseline" in r["label"])
print(f"\n🏆 最优: {best['label']} IC={best['ic']:+.4f}")
print(f"   vs C8 baseline: ΔIC={best['ic']-base['ic']:+.4f}")
if best["ic"] - base["ic"] > 0.005:
    print(f"   ✅ DL 全因子 显著优于 C8 GBDT baseline!")
elif best["ic"] - base["ic"] > -0.003:
    print(f"   ≈ 持平 C8 baseline")
else:
    print(f"   ⚠️ 仍不如 C8 baseline")

OUT = os.path.join(ROOT, "research/sector_rotation/results/dl_full_features_ic.csv")
pd.DataFrame(rows).to_csv(OUT, index=False, encoding="utf-8-sig")
print(f"\n[保存] {OUT}")
print(f"[总耗时] {time.time()-t0:.0f}s")

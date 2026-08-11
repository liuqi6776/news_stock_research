# -*- coding: utf-8 -*-
"""C8精炼因子 + 正交因子 合并实验 (17因子, 低相关叠加)

假设: C8(0.0966) 是精炼高密度, 正交因子(行业/资金流) 是低相关新信息
叠加后 MLP 应能超过 C8 GBDT
"""
import os, time, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

ROOT = r"c:\Users\liuqi\quant_system_v2"
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[GPU] {DEV}")

ORTHO_FEATS = ["ind_mom_20","ind_crowd_20","ind_mf_20",
               "net_mf_ratio_5","net_mf_ratio_20","lg_net_ratio_5","lg_net_ratio_20"]
C8_COLS = ["ivol","ret_1m","momentum_20","volatility_20","alpha_006","alpha_012",
           "enh4_score","vwap_20_resid","float_pnl_20_resid","chip_shift_5_resid"]
C8_ORTHO = C8_COLS + ORTHO_FEATS  # 17 因子

def zscore(s): return (s - s.mean()) / (s.std(ddof=1) + 1e-12)
def winsorize(s, lo=0.01, hi=0.99):
    if s.notna().sum() < 5: return s
    a, b = s.quantile([lo, hi]); return s.clip(a, b)

t0 = time.time()

# === 构造 C8+正交 面板: 从 ortho 面板拿正交因子, 从 orig 面板构造 C8 ===
panel_ortho = pd.read_parquet(os.path.join(ROOT, "research/sector_rotation/stock_ml_panel_ortho_72m.parquet"))
ortho_part = panel_ortho[["trade_date","ts_code","fwd_20"] + ORTHO_FEATS].copy()

orig = pd.read_parquet(os.path.join(ROOT, "research/sector_rotation/stock_ml_panel_72m.parquet"))
for c in ["roe","or_yoy"]:
    orig[c] = orig.groupby("trade_date")[c].transform(lambda s: winsorize(s))
orig["roe"] = orig["roe"].fillna(-99.0); orig["or_yoy"] = orig["or_yoy"].fillna(-99.0)
gg = orig.groupby("trade_date")
orig["enh4_score"] = (-0.40*gg["ivol"].rank(pct=True) -0.35*gg["ret_1m"].rank(pct=True)
                      +0.15*gg["roe"].rank(pct=True) +0.05*gg["or_yoy"].rank(pct=True))
C7_RAW = ["ivol","ret_1m","momentum_20","volatility_20","alpha_006","alpha_012"]
CHIP_RAW = ["vwap_20","float_pnl_20","chip_shift_5"]
for c in C7_RAW + CHIP_RAW:
    orig[c] = orig.groupby("trade_date")[c].transform(lambda s: winsorize(s))
    orig[c] = orig.groupby("trade_date")[c].transform(zscore)
orig["fwd_20"] = orig.groupby("trade_date")["fwd_20"].transform(lambda s: winsorize(s))
orig = orig.dropna(subset=["ivol","ret_1m","momentum_20","volatility_20","alpha_006","alpha_012",
                           "enh4_score","vwap_20","float_pnl_20","chip_shift_5","fwd_20"])
for c in CHIP_RAW: orig[f"{c}_resid"] = np.nan
for dt, grp in orig.groupby("trade_date"):
    if len(grp) < 50: continue
    Xb = grp[C7_RAW].values
    for c in CHIP_RAW:
        y = grp[c].values
        mask = np.isfinite(y) & np.all(np.isfinite(Xb), axis=1)
        if mask.sum() < 50: continue
        lr = LinearRegression().fit(Xb[mask], y[mask])
        orig.loc[grp.index[mask], f"{c}_resid"] = -(y - lr.predict(Xb))
for c in ["vwap_20_resid","float_pnl_20_resid","chip_shift_5_resid"]:
    orig[c] = orig.groupby("trade_date")[c].transform(lambda s: winsorize(s))
    orig[c] = orig.groupby("trade_date")[c].transform(zscore)
c8_part = orig[["trade_date","ts_code"] + C8_COLS].dropna(subset=C8_COLS)

# 合并
panel = ortho_part.merge(c8_part, on=["trade_date","ts_code"], how="inner")
# 正交因子标准化
for f in ORTHO_FEATS:
    panel[f] = panel.groupby("trade_date")[f].transform(lambda s: winsorize(s))
    panel[f] = panel.groupby("trade_date")[f].transform(zscore)
panel = panel.dropna(subset=C8_ORTHO + ["fwd_20"])
oos_months = sorted([m for m in panel["trade_date"].unique() if m >= 20230101])
print(f"[面板] {len(panel):,} 行, {len(C8_ORTHO)} 因子 (C8 10 + 正交 7), OOS {len(oos_months)} 月, {time.time()-t0:.0f}s")

# === 检查正交因子与 C8 的相关性 (是否真正正交) ===
print(f"\n=== 正交因子 vs C8 截面相关性 (均值) ===")
corr_parts = []
for dt, gg in panel.groupby("trade_date"):
    if len(gg) < 50: continue
    corr_parts.append(gg[ORTHO_FEATS + C8_COLS].corr())
corr_avg = pd.concat(corr_parts).groupby(level=0).mean()
for f in ORTHO_FEATS:
    maxc = corr_avg.loc[f, C8_COLS].abs().max()
    print(f"  {f:>18} max|corr with C8| = {maxc:.3f}")

# === WFO ===
def mic(df):
    out = {}
    for dt, gg in df.groupby("trade_date"):
        if len(gg) < 50: continue
        out[dt] = gg["pred"].rank().corr(gg["fwd_20"].rank())
    s = pd.Series(out).dropna()
    return s.mean(), s.std(ddof=1), (s>0).mean()

def wfo(feats, factory, need_scale=False, label=""):
    t1 = time.time(); preds = []
    for m_idx, m in enumerate(oos_months):
        tr = panel[panel["trade_date"] < m]
        if tr["trade_date"].nunique() < 24: continue
        vm = tr["trade_date"].isin(sorted(tr["trade_date"].unique())[-3:]).values
        Xa, ya = tr[feats].values, tr["fwd_20"].values
        Xtr, ytr, Xv, yv = Xa[~vm], ya[~vm], Xa[vm], ya[vm]
        sc = StandardScaler() if need_scale else None
        if need_scale: Xtr = sc.fit_transform(Xtr); Xv = sc.transform(Xv)
        mdl = factory(Xtr, ytr, Xv, yv)
        tt = panel[panel["trade_date"] == m]
        Xt = tt[feats].values
        if need_scale: Xt = sc.transform(Xt)
        preds.append(pd.DataFrame({"trade_date":m,"pred":mdl.predict(Xt),
                                   "fwd_20":tt["fwd_20"].values}))
        if (m_idx+1) % 9 == 0: print(f"    {label} WFO {m_idx+1}/{len(oos_months)}, {time.time()-t1:.0f}s")
    df = pd.concat(preds, ignore_index=True)
    ic, std, pos = mic(df)
    icir = ic / (std + 1e-12) * np.sqrt(12)
    df["q"] = df.groupby("trade_date")["pred"].transform(
        lambda s: pd.qcut(s.rank(method="first"), 5, labels=["Q1","Q2","Q3","Q4","Q5"]))
    piv = df.groupby("q", observed=True)["fwd_20"].mean()
    return {"ic": ic, "icir": icir, "pos_rate": pos,
            "q1q5": piv["Q1"]-piv["Q5"], "sec": int(time.time()-t1), "nfeat": len(feats)}

def gbdt_d3(Xtr, ytr, Xv, yv):
    m = lgb.LGBMRegressor(n_estimators=500, learning_rate=0.05, num_leaves=7,
        max_depth=3, min_child_samples=80, reg_lambda=2.0, reg_alpha=0.1,
        subsample=0.9, colsample_bytree=0.9, random_state=42, verbose=-1)
    m.fit(Xtr, ytr, eval_set=[(Xv, yv)], callbacks=[lgb.early_stopping(50, verbose=False)])
    return m

def gbdt_d5(Xtr, ytr, Xv, yv):
    m = lgb.LGBMRegressor(n_estimators=800, learning_rate=0.03, num_leaves=15,
        max_depth=5, min_child_samples=90, reg_lambda=3.0, reg_alpha=0.5,
        subsample=0.8, colsample_bytree=0.7, random_state=42, verbose=-1)
    m.fit(Xtr, ytr, eval_set=[(Xv, yv)], callbacks=[lgb.early_stopping(50, verbose=False)])
    return m

def mlp_mid(Xtr, ytr, Xv, yv):
    m = MLPRegressor((128,64,32), activation="relu", solver="adam", alpha=0.3,
        batch_size=512, learning_rate_init=0.001, max_iter=200,
        early_stopping=True, n_iter_no_change=12, validation_fraction=0.1, random_state=42)
    m.fit(Xtr, ytr)
    return m

def mlp_deep(Xtr, ytr, Xv, yv):
    m = MLPRegressor((256,128,64,32,16), activation="relu", solver="adam", alpha=0.5,
        batch_size=512, learning_rate_init=0.001, max_iter=200,
        early_stopping=True, n_iter_no_change=12, validation_fraction=0.1, random_state=42)
    m.fit(Xtr, ytr)
    return m

class FTTransformer(nn.Module):
    def __init__(self, n_feat, d_model=64, n_heads=4, n_layers=2, dropout=0.15):
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

def _train_gpu(model, Xtr, ytr, Xv, yv, epochs=60, lr=1e-3, wd=1e-2, patience=10):
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

def ftt_factory(n_feat):
    def _f(Xtr, ytr, Xv, yv):
        m = FTTransformer(n_feat).to(DEV)
        Xt = torch.from_numpy(Xtr.astype(np.float32)).to(DEV)
        Xve = torch.from_numpy(Xv.astype(np.float32)).to(DEV)
        return _train_gpu(m, Xt, ytr, Xve, yv, epochs=60, patience=10)
    return _f

# === 跑 ===
rows = []
print(f"\n=== WFO OOS 对比 (C8+正交 {len(C8_ORTHO)} 因子, 2023+ {len(oos_months)} 月) ===")

for label, fac, sc in [
    ("GBDT_d3 (C8+ortho)", gbdt_d3, False),
    ("GBDT_d5 (C8+ortho)", gbdt_d5, False),
    ("MLP_mid (C8+ortho)", mlp_mid, True),
    ("MLP_deep (C8+ortho)", mlp_deep, True),
]:
    print(f"\n>> {label}...")
    r = wfo(C8_ORTHO, fac, need_scale=sc, label=label.split()[0])
    print(f"  {label:>22} | IC={r['ic']:+.4f} ICIR={r['icir']:+.2f} Q1-Q5={r['q1q5']:+.2f} 正率={r['pos_rate']*100:.0f}% {r['sec']}s")
    rows.append({"label": label, **r})

print(f"\n>> FTT (C8+ortho)...")
r = wfo(C8_ORTHO, ftt_factory(len(C8_ORTHO)), need_scale=True, label="FTT")
print(f"  {'FTT (C8+ortho)':>22} | IC={r['ic']:+.4f} ICIR={r['icir']:+.2f} Q1-Q5={r['q1q5']:+.2f} 正率={r['pos_rate']*100:.0f}% {r['sec']}s")
rows.append({"label":"FTT (C8+ortho)", **r})

# C8 only baselines
print(f"\n>> C8 GBDT d3 (纯C8 baseline)...")
r = wfo(C8_COLS, gbdt_d3, label="C8_baseline")
print(f"  {'C8_GBDT_d3 (baseline)':>22} | IC={r['ic']:+.4f} ICIR={r['icir']:+.2f} Q1-Q5={r['q1q5']:+.2f} 正率={r['pos_rate']*100:.0f}% {r['sec']}s")
rows.append({"label":"C8_GBDT_d3 (baseline)", **r})

print(f"\n>> C8 MLP (纯C8, 看正交是否有效)...")
r = wfo(C8_COLS, mlp_deep, need_scale=True, label="C8_MLP")
print(f"  {'C8_MLP_deep':>22} | IC={r['ic']:+.4f} ICIR={r['icir']:+.2f} Q1-Q5={r['q1q5']:+.2f} 正率={r['pos_rate']*100:.0f}% {r['sec']}s")
rows.append({"label":"C8_MLP_deep", **r})

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
if best["ic"] - base["ic"] > 0.003:
    print(f"   ✅ C8+正交 显著优于纯 C8 baseline!")
elif best["ic"] - base["ic"] > -0.003:
    print(f"   ≈ 持平 C8 baseline (正交无增量)")
else:
    print(f"   ⚠️ 反而不如纯 C8")

OUT = os.path.join(ROOT, "research/sector_rotation/results/c8_ortho_ic.csv")
pd.DataFrame(rows).to_csv(OUT, index=False, encoding="utf-8-sig")
print(f"\n[保存] {OUT}")
print(f"[总耗时] {time.time()-t0:.0f}s")

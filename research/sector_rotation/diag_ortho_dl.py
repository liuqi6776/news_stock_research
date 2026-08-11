# -*- coding: utf-8 -*-
"""全因子 + 正交因子 GPU WFO 对比: MLP / FTT / GBDT vs C8 baseline

面板: stock_ml_panel_ortho_72m.parquet (79大面板 + 11正交)
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
print(f"[GPU] {DEV} | {torch.cuda.get_device_name(0) if DEV.type=='cuda' else 'CPU'}")

# 正交因子 (排除 news 全0因子)
ORTHO_FEATS = ["ind_mom_20","ind_crowd_20","ind_mf_20",
               "net_mf_ratio_5","net_mf_ratio_20","lg_net_ratio_5","lg_net_ratio_20"]

C8_COLS = ["ivol","ret_1m","momentum_20","volatility_20","alpha_006","alpha_012",
           "enh4_score","vwap_20_resid","float_pnl_20_resid","chip_shift_5_resid"]

def zscore(s): return (s - s.mean()) / (s.std(ddof=1) + 1e-12)
def winsorize(s, lo=0.01, hi=0.99):
    if s.notna().sum() < 5: return s
    a, b = s.quantile([lo, hi]); return s.clip(a, b)

t0 = time.time()
panel = pd.read_parquet(os.path.join(ROOT, "research/sector_rotation/stock_ml_panel_ortho_72m.parquet"))
excl = {"trade_date","ts_code","open","high","low","close","pct_chg","vol",
        "industry","is_traditional","fwd_20","enh4_score","vwap_20_resid",
        "float_pnl_20_resid","chip_shift_5_resid"}
all_feats = [c for c in panel.columns if c not in excl]
all_feats = [c for c in all_feats if not c.startswith("news_")]  # news 数据2025-02后缺失, 排除
print(f"[面板] {len(panel):,} 行, {len(all_feats)} 因子 (含 {len(ORTHO_FEATS)} 正交)")

# 需要重算 enh4_score + 筹码残差 (面板里没有)
# C8 因子单独从 orig 面板构造
C7_RAW = ["ivol","ret_1m","momentum_20","volatility_20","alpha_006","alpha_012"]
CHIP_RAW = ["vwap_20","float_pnl_20","chip_shift_5"]

# --- 截面标准化 (全因子) ---
for f in all_feats:
    panel[f] = panel.groupby("trade_date")[f].transform(lambda s: winsorize(s))
    panel[f] = panel.groupby("trade_date")[f].transform(zscore)
panel["fwd_20"] = panel.groupby("trade_date")["fwd_20"].transform(lambda s: winsorize(s))
panel = panel.dropna(subset=all_feats + ["fwd_20"])
oos_months = sorted([m for m in panel["trade_date"].unique() if m >= 20230101])
print(f"[OOS] {len(oos_months)} 月, {time.time()-t0:.0f}s")

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
        if (m_idx+1) % 6 == 0: print(f"    {label} WFO {m_idx+1}/{len(oos_months)}, {time.time()-t1:.0f}s")
    df = pd.concat(preds, ignore_index=True)
    ic, std, pos = mic(df)
    icir = ic / (std + 1e-12) * np.sqrt(12)
    df["q"] = df.groupby("trade_date")["pred"].transform(
        lambda s: pd.qcut(s.rank(method="first"), 5, labels=["Q1","Q2","Q3","Q4","Q5"]))
    piv = df.groupby("q", observed=True)["fwd_20"].mean()
    return {"ic": ic, "icir": icir, "pos_rate": pos,
            "q1q5": piv["Q1"]-piv["Q5"], "sec": int(time.time()-t1), "nfeat": len(feats)}

# === 模型工厂 ===
def gbdt_d3(Xtr, ytr, Xv, yv):
    m = lgb.LGBMRegressor(n_estimators=500, learning_rate=0.05, num_leaves=7,
        max_depth=3, min_child_samples=80, reg_lambda=2.0, reg_alpha=0.1,
        subsample=0.9, colsample_bytree=0.9, random_state=42, verbose=-1)
    m.fit(Xtr, ytr, eval_set=[(Xv, yv)], callbacks=[lgb.early_stopping(50, verbose=False)])
    return m

def gbdt_d10(Xtr, ytr, Xv, yv):
    m = lgb.LGBMRegressor(n_estimators=1000, learning_rate=0.02, num_leaves=63,
        max_depth=10, min_child_samples=120, reg_lambda=5.0, reg_alpha=1.0,
        subsample=0.7, colsample_bytree=0.5, random_state=42, verbose=-1)
    m.fit(Xtr, ytr, eval_set=[(Xv, yv)], callbacks=[lgb.early_stopping(50, verbose=False)])
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

def ftt_factory(n_feat):
    def _f(Xtr, ytr, Xv, yv):
        m = FTTransformer(n_feat).to(DEV)
        Xt = torch.from_numpy(Xtr.astype(np.float32)).to(DEV)
        Xve = torch.from_numpy(Xv.astype(np.float32)).to(DEV)
        return _train_gpu(m, Xt, ytr, Xve, yv, epochs=50, patience=8)
    return _f

# === 构造 C8 baseline 面板 ===
orig = pd.read_parquet(os.path.join(ROOT, "research/sector_rotation/stock_ml_panel_72m.parquet"))
for c in ["roe","or_yoy"]:
    orig[c] = orig.groupby("trade_date")[c].transform(lambda s: winsorize(s))
orig["roe"] = orig["roe"].fillna(-99.0); orig["or_yoy"] = orig["or_yoy"].fillna(-99.0)
gg = orig.groupby("trade_date")
orig["enh4_score"] = (-0.40*gg["ivol"].rank(pct=True) -0.35*gg["ret_1m"].rank(pct=True)
                      +0.15*gg["roe"].rank(pct=True) +0.05*gg["or_yoy"].rank(pct=True))
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
orig = orig.dropna(subset=C8_COLS + ["fwd_20"])
oos_months_c8 = sorted([m for m in orig["trade_date"].unique() if m >= 20230101])

def wfo_c8(factory):
    t1 = time.time(); preds = []
    for m in oos_months_c8:
        tr = orig[orig["trade_date"] < m]
        if tr["trade_date"].nunique() < 24: continue
        vm = tr["trade_date"].isin(sorted(tr["trade_date"].unique())[-3:]).values
        Xa, ya = tr[C8_COLS].values, tr["fwd_20"].values
        Xtr, ytr, Xv, yv = Xa[~vm], ya[~vm], Xa[vm], ya[vm]
        mdl = factory(Xtr, ytr, Xv, yv)
        tt = orig[orig["trade_date"] == m]
        preds.append(pd.DataFrame({"trade_date":m,"pred":mdl.predict(tt[C8_COLS].values),
                                   "fwd_20":tt["fwd_20"].values}))
    df = pd.concat(preds, ignore_index=True)
    ic, std, pos = mic(df)
    icir = ic / (std + 1e-12) * np.sqrt(12)
    df["q"] = df.groupby("trade_date")["pred"].transform(
        lambda s: pd.qcut(s.rank(method="first"), 5, labels=["Q1","Q2","Q3","Q4","Q5"]))
    piv = df.groupby("q", observed=True)["fwd_20"].mean()
    return {"ic": ic, "icir": icir, "pos_rate": pos,
            "q1q5": piv["Q1"]-piv["Q5"], "sec": int(time.time()-t1), "nfeat": len(C8_COLS)}

# === 跑 ===
NF = len(all_feats)
rows = []

print(f"\n=== WFO OOS 对比 (正交+全因子 {NF} 个, 2023+ {len(oos_months)} 月) ===")

print(f"\n>> GBDT d3 (全因子 {NF})...")
r = wfo(all_feats, gbdt_d3, label="GBDT_d3")
print(f"  {'GBDT_d3':>22} | IC={r['ic']:+.4f} ICIR={r['icir']:+.2f} Q1-Q5={r['q1q5']:+.2f} 正率={r['pos_rate']*100:.0f}% {r['sec']}s")
rows.append({"label":"GBDT_d3 (all+ortho)", **r})

print(f"\n>> GBDT d10 (全因子 {NF})...")
r = wfo(all_feats, gbdt_d10, label="GBDT_d10")
print(f"  {'GBDT_d10':>22} | IC={r['ic']:+.4f} ICIR={r['icir']:+.2f} Q1-Q5={r['q1q5']:+.2f} 正率={r['pos_rate']*100:.0f}% {r['sec']}s")
rows.append({"label":"GBDT_d10 (all+ortho)", **r})

print(f"\n>> MLP 深层5层 (全因子 {NF})...")
r = wfo(all_feats, mlp_deep, need_scale=True, label="MLP_deep5")
print(f"  {'MLP_deep5':>22} | IC={r['ic']:+.4f} ICIR={r['icir']:+.2f} Q1-Q5={r['q1q5']:+.2f} 正率={r['pos_rate']*100:.0f}% {r['sec']}s")
rows.append({"label":"MLP_deep5 (all+ortho)", **r})

print(f"\n>> FT-Transformer (全因子 {NF}, GPU+AMP)...")
r = wfo(all_feats, ftt_factory(NF), need_scale=True, label="FTT_64_2L")
print(f"  {'FTT_64_2L':>22} | IC={r['ic']:+.4f} ICIR={r['icir']:+.2f} Q1-Q5={r['q1q5']:+.2f} 正率={r['pos_rate']*100:.0f}% {r['sec']}s")
rows.append({"label":"FTT_64_2L (all+ortho)", **r})

print(f"\n>> C8 GBDT d3 (baseline, 10因子)...")
r = wfo_c8(gbdt_d3)
print(f"  {'C8_GBDT_d3':>22} | IC={r['ic']:+.4f} ICIR={r['icir']:+.2f} Q1-Q5={r['q1q5']:+.2f} 正率={r['pos_rate']*100:.0f}% {r['sec']}s")
rows.append({"label":"C8_GBDT_d3 (baseline)", **r})

# 正交因子单因子 IC
print(f"\n=== 正交因子 IC 汇总 ===")
for f in ORTHO_FEATS:
    ics = []
    for dt, gg in panel.groupby("trade_date"):
        if len(gg) < 50: continue
        ic = gg[f].rank().corr(gg["fwd_20"].rank())
        if np.isfinite(ic): ics.append(ic)
    s = pd.Series(ics)
    print(f"  {f:>18} IC={s.mean():+.4f} ICIR={s.mean()/(s.std(ddof=1)+1e-9)*np.sqrt(12):+.2f} 正率={(s>0).mean()*100:.0f}%")

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

OUT = os.path.join(ROOT, "research/sector_rotation/results/ortho_dl_ic.csv")
pd.DataFrame(rows).to_csv(OUT, index=False, encoding="utf-8-sig")
print(f"\n[保存] {OUT}")
print(f"[总耗时] {time.time()-t0:.0f}s")

# -*- coding: utf-8 -*-
"""大因子集 IC 筛选 + GBDT vs MLP vs FT-Transformer WFO 对比

Phase 1: 逐月截面 IC 分析 → 筛选 |IC|>0.02
Phase 2: 高相关去冗余 |r|>0.7 → 保留 IC 更高的
Phase 3: WFO OOS IC 对比 (GBDT d3 / GBDT d7 / MLP / FT-Transformer)
"""
import os, time, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

ROOT = r"c:\Users\liuqi\quant_system_v2"
PANEL = os.path.join(ROOT, "research", "sector_rotation", "stock_ml_panel_large_72m.parquet")
OUT_CSV = os.path.join(ROOT, "research", "sector_rotation", "results", "dl_large_features_ic.csv")

# C8 baseline 对照
C8_COLS = ["ivol","ret_1m","momentum_20","volatility_20","alpha_006","alpha_012",
           "enh4_score","vwap_20_resid","float_pnl_20_resid","chip_shift_5_resid"]

t0 = time.time()
panel = pd.read_parquet(PANEL)
exclude = {"trade_date","ts_code","open","high","low","close","pct_chg","vol",
           "industry","is_traditional","fwd_20"}
all_feats = [c for c in panel.columns if c not in exclude]
panel = panel.dropna(subset=all_feats + ["fwd_20"])
months = sorted(panel["trade_date"].unique())
oos_months = [m for m in months if m >= 20230101]
print(f"[数据] {len(panel):,} 行, {len(all_feats)} 因子, {len(months)} 月, OOS {len(oos_months)} 月, {time.time()-t0:.0f}s")

# === Phase 1: 逐月截面 IC ===
def monthly_ic(factor):
    ics = []
    for dt, gg in panel.groupby("trade_date"):
        if len(gg) < 50: continue
        ic = gg[factor].rank().corr(gg["fwd_20"].rank())
        if np.isfinite(ic): ics.append(ic)
    s = pd.Series(ics)
    return s.mean(), s.std(ddof=1), (s>0).mean()

print(f"\n=== Phase 1: 逐月截面 IC 筛选 (|IC|>0.015) ===")
ic_results = []
for f in all_feats:
    mean_ic, std_ic, pos_rate = monthly_ic(f)
    ic_results.append({"factor": f, "ic": mean_ic, "std": std_ic,
                       "icir": mean_ic/(std_ic+1e-12)*np.sqrt(12), "pos_rate": pos_rate})
ic_df = pd.DataFrame(ic_results)
ic_df["abs_ic"] = ic_df["ic"].abs()
ic_df = ic_df.sort_values("abs_ic", ascending=False)
selected = ic_df[ic_df["abs_ic"] > 0.015].copy()
print(f"  筛选前: {len(all_feats)} → 筛选后: {len(selected)} (|IC|>0.015)")
print(f"  Top 20 IC:")
for _, r in selected.head(20).iterrows():
    print(f"    {r['factor']:>22} IC={r['ic']:+.4f} ICIR={r['icir']:+.2f} 正率={r['pos_rate']*100:.0f}%")

# === Phase 2: 高相关去冗余 ===
print(f"\n=== Phase 2: 去冗余 (|r|>0.7 保留 IC 更高的) ===")
sel_feats = selected["factor"].tolist()
# 计算因子间截面相关（取所有月末的平均相关矩阵）
corr_parts = []
for dt, gg in panel.groupby("trade_date"):
    if len(gg) < 50: continue
    c = gg[sel_feats].corr()
    corr_parts.append(c)
corr_avg = pd.concat(corr_parts).groupby(level=0).mean()
# 贪心去冗余
removed = set()
for i in range(len(sel_feats)):
    for j in range(i+1, len(sel_feats)):
        fi, fj = sel_feats[i], sel_feats[j]
        if fi in removed or fj in removed: continue
        r = corr_avg.loc[fi, fj] if fi in corr_avg.index and fj in corr_avg.columns else 0
        if abs(r) > 0.7:
            # 保留 IC 更高的
            ic_i = selected[selected["factor"]==fi]["abs_ic"].values[0]
            ic_j = selected[selected["factor"]==fj]["abs_ic"].values[0]
            drop = fj if ic_i >= ic_j else fi
            removed.add(drop)
final_feats = [f for f in sel_feats if f not in removed]
print(f"  去冗余: {len(sel_feats)} → {len(final_feats)} (移除 {len(removed)} 个)")
if removed:
    print(f"  移除: {sorted(removed)}")
print(f"  最终因子集 ({len(final_feats)} 个): {final_feats}")

# === Phase 3: WFO 模型对比 ===
def zscore(s): return (s - s.mean()) / (s.std(ddof=1) + 1e-12)
def winsorize(s, lo=0.01, hi=0.99):
    if s.notna().sum() < 5: return s
    a, b = s.quantile([lo, hi]); return s.clip(a, b)

# 截面标准化
for f in final_feats:
    panel[f] = panel.groupby("trade_date")[f].transform(lambda s: winsorize(s))
    panel[f] = panel.groupby("trade_date")[f].transform(zscore)
panel["fwd_20"] = panel.groupby("trade_date")["fwd_20"].transform(lambda s: winsorize(s))

def mic(df, pred_col="pred"):
    out = {}
    for dt, gg in df.groupby("trade_date"):
        if len(gg) < 50: continue
        out[dt] = gg[pred_col].rank().corr(gg["fwd_20"].rank())
    s = pd.Series(out).dropna()
    return s.mean(), s.std(ddof=1), (s>0).mean()

def wfo(feats, model_factory, need_scale=False, label=""):
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
        mdl = model_factory(Xtr, ytr, Xv, yv)
        tt = panel[panel["trade_date"] == m]
        Xt = tt[feats].values
        if need_scale: Xt = sc.transform(Xt)
        preds.append(pd.DataFrame({"trade_date":m, "ts_code":tt["ts_code"],
                                   "pred":mdl.predict(Xt), "fwd_20":tt["fwd_20"].values}))
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

# 模型工厂
def gbdt_d3(Xtr, ytr, Xv, yv):
    m = lgb.LGBMRegressor(n_estimators=500, learning_rate=0.05, num_leaves=7,
                          max_depth=3, min_child_samples=80, reg_lambda=2.0, reg_alpha=0.1,
                          subsample=0.9, colsample_bytree=0.9, random_state=42, verbose=-1)
    m.fit(Xtr, ytr, eval_set=[(Xv, yv)], callbacks=[lgb.early_stopping(50, verbose=False)])
    return m

def gbdt_d7(Xtr, ytr, Xv, yv):
    m = lgb.LGBMRegressor(n_estimators=800, learning_rate=0.03, num_leaves=31,
                          max_depth=7, min_child_samples=100, reg_lambda=3.0, reg_alpha=0.5,
                          subsample=0.8, colsample_bytree=0.6, random_state=42, verbose=-1)
    m.fit(Xtr, ytr, eval_set=[(Xv, yv)], callbacks=[lgb.early_stopping(50, verbose=False)])
    return m

def mlp_factory(hidden=(128,64,32), alpha=0.3):
    def _f(Xtr, ytr, Xv, yv):
        m = MLPRegressor(hidden, activation="relu", solver="adam", alpha=alpha,
                         batch_size=512, learning_rate_init=0.001, max_iter=150,
                         early_stopping=True, n_iter_no_change=10, validation_fraction=0.1,
                         random_state=42)
        m.fit(Xtr, ytr)
        return m
    return _f

# FT-Transformer (纯 PyTorch, 如果有 torch)
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import TensorDataset, DataLoader
    HAS_TORCH = True
except:
    HAS_TORCH = False

if HAS_TORCH:
    class FTTransformer(nn.Module):
        def __init__(self, n_feat, d_model=64, n_heads=4, n_layers=3, dropout=0.2):
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

    def ftt_factory(d_model=64, n_heads=4, n_layers=3, dropout=0.2, lr=1e-3, wd=1e-2, epochs=80):
        def _f(Xtr, ytr, Xv, yv):
            DEV = torch.device("cpu")
            nfeat = Xtr.shape[1]
            model = FTTransformer(nfeat, d_model, n_heads, n_layers, dropout).to(DEV)
            opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
            sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", patience=5, factor=0.5)
            Xt = torch.from_numpy(Xtr.astype(np.float32)).to(DEV)
            yt = torch.from_numpy(ytr.astype(np.float32)).to(DEV)
            Xve = torch.from_numpy(Xv.astype(np.float32)).to(DEV)
            yve = torch.from_numpy(yv.astype(np.float32)).to(DEV)
            ds = TensorDataset(Xt, yt)
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
                    vl = F.mse_loss(model(Xve), yve).item()
                if vl < best_vl:
                    best_vl = vl
                    best_sd = {k: v.detach().cpu().clone() for k,v in model.state_dict().items()}
                    pat = 0
                else:
                    pat += 1
                    if pat >= 10: break
                sched.step(vl)
            if best_sd: model.load_state_dict(best_sd)
            model.eval()
            class _P:
                def __init__(self, m): self.m = m
                def predict(self, X):
                    with torch.no_grad():
                        return self.m(torch.from_numpy(X.astype(np.float32)).to(DEV)).cpu().numpy().ravel()
            return _P(model)
        return _f

# === 跑对比 ===
print(f"\n=== Phase 3: WFO OOS IC 对比 (2023+, {len(oos_months)} 月) ===")
print(f"  特征集: {len(final_feats)} 因子")

configs = [
    ("GBDT_d3 (浅树)",   lambda *a: gbdt_d3(*a),    False),
    ("GBDT_d7 (深树)",   lambda *a: gbdt_d7(*a),    False),
    ("MLP_128_64_32",    mlp_factory((128,64,32), 0.3), True),
    ("MLP_256_128_64",   mlp_factory((256,128,64), 0.5), True),
]
if HAS_TORCH:
    configs.append(("FT-Transformer", ftt_factory(d_model=64, n_heads=4, n_layers=3, dropout=0.2, epochs=80), True))

rows = []
for label, fac, sc in configs:
    print(f"\n  >> {label} (n_feat={len(final_feats)})...")
    r = wfo(final_feats, fac, need_scale=sc, label=label)
    print(f"  {label:>20} | IC={r['ic']:+.4f} ICIR={r['icir']:+.2f} "
          f"Q1-Q5={r['q1q5']:+.2f} 正率={r['pos_rate']*100:.0f}% {r['sec']}s")
    rows.append({"label": label, **r})

# C8 baseline 对照
print(f"\n  >> C8 GBDT d3 (baseline)...")
# 需要残差化, 直接从原 panel 读
panel_orig = pd.read_parquet(os.path.join(ROOT, "research", "sector_rotation", "stock_ml_panel_72m.parquet"))
C7_COLS = ["ivol","ret_1m","momentum_20","volatility_20","alpha_006","alpha_012","enh4_score"]
CHIP_COLS = ["vwap_20","float_pnl_20","chip_shift_5"]
CHIP_RESID = ["vwap_20_resid","float_pnl_20_resid","chip_shift_5_resid"]
BASE_COLS = ["ivol","ret_1m","momentum_20","volatility_20","alpha_006","alpha_012"]
from sklearn.linear_model import LinearRegression
for c in ["roe","or_yoy"]:
    panel_orig[c] = panel_orig.groupby("trade_date")[c].transform(lambda s: winsorize(s))
panel_orig["roe"] = panel_orig["roe"].fillna(-99.0)
panel_orig["or_yoy"] = panel_orig["or_yoy"].fillna(-99.0)
gg = panel_orig.groupby("trade_date")
panel_orig["enh4_score"] = (-0.40*gg["ivol"].rank(pct=True) -0.35*gg["ret_1m"].rank(pct=True)
                            +0.15*gg["roe"].rank(pct=True) +0.05*gg["or_yoy"].rank(pct=True))
ALL_RAW = list(set(C7_COLS + CHIP_COLS) - {"enh4_score"})
for c in ALL_RAW:
    panel_orig[c] = panel_orig.groupby("trade_date")[c].transform(lambda s: winsorize(s))
    panel_orig[c] = panel_orig.groupby("trade_date")[c].transform(zscore)
panel_orig["fwd_20"] = panel_orig.groupby("trade_date")["fwd_20"].transform(lambda s: winsorize(s))
panel_orig = panel_orig.dropna(subset=C7_COLS + CHIP_COLS + ["fwd_20"])
for c in CHIP_COLS: panel_orig[f"{c}_resid"] = np.nan
for dt, grp in panel_orig.groupby("trade_date"):
    if len(grp) < 50: continue
    Xb = grp[BASE_COLS].values
    for c in CHIP_COLS:
        y = grp[c].values
        mask = np.isfinite(y) & np.all(np.isfinite(Xb), axis=1)
        if mask.sum() < 50: continue
        lr = LinearRegression(fit_intercept=True)
        lr.fit(Xb[mask], y[mask])
        panel_orig.loc[grp.index[mask], f"{c}_resid"] = -(y - lr.predict(Xb))
for c in CHIP_RESID:
    panel_orig[c] = panel_orig.groupby("trade_date")[c].transform(lambda s: winsorize(s))
    panel_orig[c] = panel_orig.groupby("trade_date")[c].transform(zscore)
panel_orig = panel_orig.dropna(subset=C8_COLS + ["fwd_20"])
oos_months_orig = sorted([m for m in panel_orig["trade_date"].unique() if m >= 20230101])

preds = []
for m in oos_months_orig:
    tr = panel_orig[panel_orig["trade_date"] < m]
    if tr["trade_date"].nunique() < 24: continue
    vm = tr["trade_date"].isin(sorted(tr["trade_date"].unique())[-3:]).values
    Xa, ya = tr[C8_COLS].values, tr["fwd_20"].values
    Xtr, ytr, Xv, yv = Xa[~vm], ya[~vm], Xa[vm], ya[vm]
    mdl = gbdt_d3(Xtr, ytr, Xv, yv)
    tt = panel_orig[panel_orig["trade_date"] == m]
    preds.append(pd.DataFrame({"trade_date":m, "pred":mdl.predict(tt[C8_COLS].values),
                               "fwd_20":tt["fwd_20"].values}))
df8 = pd.concat(preds, ignore_index=True)
ic8, std8, pos8 = mic(df8)
icir8 = ic8 / (std8 + 1e-12) * np.sqrt(12)
df8["q"] = df8.groupby("trade_date")["pred"].transform(
    lambda s: pd.qcut(s.rank(method="first"), 5, labels=["Q1","Q2","Q3","Q4","Q5"]))
piv8 = df8.groupby("q", observed=True)["fwd_20"].mean()
rows.append({"label": "C8_GBDT_d3 (baseline)", "ic": ic8, "icir": icir8,
             "pos_rate": pos8, "q1q5": piv8["Q1"]-piv8["Q5"], "sec": 0, "nfeat": len(C8_COLS)})
print(f"  {'C8_GBDT_d3 (baseline)':>20} | IC={ic8:+.4f} ICIR={icir8:+.2f} "
      f"Q1-Q5={piv8['Q1']-piv8['Q5']:+.2f} 正率={pos8*100:.0f}%")

# === 汇总 ===
print(f"\n{'='*70}")
print(f"{'模型':>24} {'因子数':>5} {'IC':>8} {'ICIR':>7} {'Q1-Q5':>7} {'正率':>5}")
print(f"{'-'*70}")
for r in sorted(rows, key=lambda x: x["ic"], reverse=True):
    print(f"{r['label']:>24} {r['nfeat']:>5} {r['ic']:+.4f}  {r['icir']:+.2f}   {r['q1q5']:+.2f}   {r['pos_rate']*100:.0f}%")
print(f"{'='*70}")

best = max(rows, key=lambda x: x["ic"])
base = next(r for r in rows if "baseline" in r["label"])
print(f"\n🏆 最优: {best['label']} IC={best['ic']:+.4f}")
print(f"   vs C8 baseline: ΔIC={best['ic']-base['ic']:+.4f}")
if best["ic"] - base["ic"] > 0.005:
    print(f"   ✅ 大特征集{'+DL' if 'FT' in best['label'] or 'MLP' in best['label'] else '+GBDT'} 显著优于 C8 baseline")
elif best["ic"] - base["ic"] > -0.003:
    print(f"   ≈ 持平 C8 baseline (Δ<0.005)")
else:
    print(f"   ⚠️ 不如 C8 baseline")

pd.DataFrame(rows).to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
print(f"\n[保存] {OUT_CSV}")
print(f"[总耗时] {time.time()-t0:.0f}s")

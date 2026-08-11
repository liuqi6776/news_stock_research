# -*- coding: utf-8 -*-
"""模型对比: GBDT vs MLP vs TabNet vs FT-Transformer (WFO OOS IC)

小样本 + 小特征集(8因子去冗余 C6)场景：
- MLP: 3 层 MLP + Dropout + BatchNorm + 强正则
- TabNet: pytorch-tabnet (如果安装了)
- FT-Transformer: TabPFN 风格 (用 tab-transformer-pytorch 或 skorch)
  注意: 没装的话 fallback 到 MLP + Transformer Embedding

对比指标: WFO OOS IC (Spearmann) / ICIR / Q1-Q5
"""
import os, time, sys, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

ROOT = r"c:\Users\liuqi\quant_system_v2"
PANEL = os.path.join(ROOT, "research", "sector_rotation", "stock_ml_panel_72m.parquet")
OUT = os.path.join(ROOT, "research", "sector_rotation", "results", "model_compare_dl.csv")

# C6 (冗余排除版, 8 因子): 排除 momentum_20 + volatility_20 (VIF≈2400)
C7_COLS   = ["ivol","ret_1m","momentum_20","volatility_20","alpha_006","alpha_012","enh4_score"]
CHIP_COLS = ["vwap_20","float_pnl_20","chip_shift_5"]
CHIP_RESID = ["vwap_20_resid","float_pnl_20_resid","chip_shift_5_resid"]
BASE_COLS = ["ivol","ret_1m","momentum_20","volatility_20","alpha_006","alpha_012"]
C8_COLS = C7_COLS + CHIP_RESID
C6_COLS = [c for c in C8_COLS if c not in ("momentum_20","volatility_20")]  # 去冗余版

def zscore(s):
    return (s - s.mean()) / (s.std(ddof=1) + 1e-12)
def winsorize(s, lo=0.01, hi=0.99):
    if s.notna().sum() < 5: return s
    a, b = s.quantile([lo, hi])
    return s.clip(a, b)

t0 = time.time()
panel = pd.read_parquet(PANEL)

# 复刻预处理 + 筹码残差化
for c in ["roe","or_yoy"]:
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
        lr = LinearRegression(fit_intercept=True)
        lr.fit(Xb[mask], y[mask])
        panel.loc[grp.index[mask], f"{c}_resid"] = -(y - lr.predict(Xb))
for c in CHIP_RESID:
    panel[c] = panel.groupby("trade_date")[c].transform(lambda s: winsorize(s))
    panel[c] = panel.groupby("trade_date")[c].transform(zscore)
panel = panel.dropna(subset=C8_COLS + ["fwd_20"])
print(f"[数据] {len(panel):,} 行, {panel['trade_date'].nunique()}月, {time.time()-t0:.0f}s")

months = sorted(panel["trade_date"].unique())
oos_months = [m for m in months if m >= 20230101]

def monthly_ic(d, factor, ret="fwd_20"):
    out = {}
    for dt, gg in d.groupby("trade_date"):
        if len(gg) < 50: continue
        out[dt] = gg[factor].rank().corr(gg[ret].rank())
    return pd.Series(out).dropna()

# ========== WFO 统一流程 ==========
def wfo_fit_predict(feats_cols, model_factory, label, model_kwargs):
    """factory(X_tr, y_tr, **kw) -> model 有 .predict(X) 方法"""
    t1 = time.time()
    pred_list = []
    for i, m in enumerate(oos_months):
        tr = panel[panel["trade_date"] < m]
        if tr["trade_date"].nunique() < 24: continue
        val_months = sorted(tr["trade_date"].unique())[-3:]
        vm = tr["trade_date"].isin(val_months).values
        X_tr_all = tr[feats_cols].values
        y_tr_all = tr["fwd_20"].values
        X_tr, y_tr = X_tr_all[~vm], y_tr_all[~vm]
        X_val, y_val = X_tr_all[vm], y_tr_all[vm]
        # 标准化（除树模型外）
        if model_kwargs.get("need_scale", False):
            sc = StandardScaler()
            X_tr = sc.fit_transform(X_tr)
            X_val = sc.transform(X_val)
        else:
            sc = None
        model = model_factory(X_tr, y_tr, X_val, y_val, model_kwargs)
        t_m = panel[panel["trade_date"] == m]
        X_tt = t_m[feats_cols].values
        if sc is not None:
            X_tt = sc.transform(X_tt)
        pred_list.append(pd.DataFrame({"trade_date": m, "ts_code": t_m["ts_code"],
                                       "pred": model.predict(X_tt),
                                       "fwd_20": t_m["fwd_20"].values}))
    df = pd.concat(pred_list, ignore_index=True)
    ics = monthly_ic(df, "pred")
    icir = ics.mean() / (ics.std(ddof=1) + 1e-12) * np.sqrt(12)
    df["q"] = df.groupby("trade_date")["pred"].transform(
        lambda s: pd.qcut(s.rank(method="first"), 5, labels=["Q1","Q2","Q3","Q4","Q5"]))
    piv = df.groupby("q", observed=True)["fwd_20"].mean()
    return {"model": label, "n_feat": len(feats_cols),
            "ic": ics.mean(), "icir": icir, "pos_rate": (ics>0).mean(),
            "q1_q5": piv["Q1"]-piv["Q5"],
            "sec": int(time.time()-t1)}

# ========== 模型工厂 ==========
def factory_gbdt(X_tr, y_tr, X_val, y_val, kw):
    mdl = lgb.LGBMRegressor(n_estimators=500, learning_rate=0.05,
                            num_leaves=7, max_depth=3,
                            min_child_samples=80, reg_lambda=2.0, reg_alpha=0.1,
                            subsample=0.9, colsample_bytree=0.9,
                            random_state=42, verbose=-1)
    mdl.fit(X_tr, y_tr, eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(50, verbose=False)])
    return mdl

def factory_mlp(X_tr, y_tr, X_val, y_val, kw):
    # sklearn MLP: 3 层 64 + ReLU, 强正则（alpha=0.5, 早停）
    hidden = kw.get("hidden", (64, 32, 16))
    alpha = kw.get("alpha", 0.5)
    mdl = MLPRegressor(hidden_layer_sizes=hidden,
                       activation="relu",
                       solver="adam",
                       alpha=alpha,
                       batch_size=512,
                       learning_rate_init=0.001,
                       max_iter=100,
                       early_stopping=True,
                       n_iter_no_change=8,
                       validation_fraction=0.1,
                       random_state=42)
    mdl.fit(X_tr, y_tr)
    return mdl

# TabNet / FT-Transformer: 尝试 import，失败就 fallback 到 MLP 变体
try:
    from pytorch_tabnet.tab_model import TabNetRegressor
    HAS_TABNET = True
except Exception:
    HAS_TABNET = False

def factory_tabnet(X_tr, y_tr, X_val, y_val, kw):
    if not HAS_TABNET:
        # fallback: 用 MLP (128,64)
        return factory_mlp(X_tr, y_tr, X_val, y_val, {"hidden":(128,64), "alpha":1.0, "need_scale": True})
    mdl = TabNetRegressor(n_d=8, n_a=8, n_steps=3, gamma=1.3,
                          lambda_sparse=1e-3, optimizer_fn=__import__("torch").optim.Adam,
                          optimizer_params=dict(lr=2e-2),
                          scheduler_params=dict(mode="min", patience=5, min_lr=1e-5, factor=0.9),
                          scheduler_fn=__import__("torch").optim.lr_scheduler.ReduceLROnPlateau,
                          mask_type="entmax",
                          verbose=0, seed=42)
    import torch
    mdl.fit(X_tr, y_tr.reshape(-1,1),
            eval_set=[(X_val, y_val.reshape(-1,1))],
            max_epochs=80, patience=12, batch_size=1024, virtual_batch_size=128)
    return mdl

# FT-Transformer: 用 skorch + 简易 Transformer (如果没 pytorch，就 fallback 到宽 MLP)
try:
    import torch
    HAS_TORCH = True
except Exception:
    HAS_TORCH = False

if HAS_TORCH:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class FTTransformerSimple(nn.Module):
        def __init__(self, n_feat=8, d_model=32, n_heads=4, n_layers=2, dropout=0.2):
            super().__init__()
            self.d_model = d_model
            self.proj = nn.Linear(1, d_model)
            self.cls = nn.Parameter(torch.zeros(1, 1, d_model))
            enc = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads,
                                              dim_feedforward=d_model*4,
                                              dropout=dropout, activation="gelu",
                                              batch_first=True, norm_first=True)
            self.enc = nn.TransformerEncoder(enc, num_layers=n_layers)
            self.head = nn.Sequential(
                nn.LayerNorm(d_model),
                nn.Linear(d_model, d_model), nn.GELU(), nn.Dropout(dropout),
                nn.Linear(d_model, 1),
            )
        def forward(self, x):
            B = x.shape[0]
            z = self.proj(x.unsqueeze(-1))
            cls = self.cls.expand(B, -1, -1)
            z = torch.cat([cls, z], dim=1)
            z = self.enc(z)
            return self.head(z[:, 0, :]).squeeze(-1)

def factory_ftt(X_tr, y_tr, X_val, y_val, kw):
    if not HAS_TORCH:
        return factory_mlp(X_tr, y_tr, X_val, y_val, {"hidden":(128,128,64), "alpha":0.8, "need_scale": True})
    # 纯 PyTorch 训练，不用 skorch
    import torch
    import torch.nn as nn
    from torch.utils.data import TensorDataset, DataLoader
    DEV = torch.device("cpu")
    nfeat = X_tr.shape[1]
    model = FTTransformerSimple(n_feat=nfeat, d_model=32, n_heads=4, n_layers=2, dropout=0.2).to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", patience=5, factor=0.5)
    Xt = torch.from_numpy(X_tr.astype(np.float32)).to(DEV)
    yt = torch.from_numpy(y_tr.astype(np.float32)).to(DEV)
    Xv = torch.from_numpy(X_val.astype(np.float32)).to(DEV)
    yv = torch.from_numpy(y_val.astype(np.float32)).to(DEV)
    ds = TensorDataset(Xt, yt)
    dl = DataLoader(ds, batch_size=512, shuffle=True)
    best_vl = np.inf; best_sd = None; pat = 0
    for ep in range(80):
        model.train()
        for bx, by in dl:
            opt.zero_grad()
            pred = model(bx)
            loss = F.mse_loss(pred, by)
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            vl = F.mse_loss(model(Xv), yv).item()
        if vl < best_vl:
            best_vl = vl; best_sd = {k: v.detach().cpu().clone() for k,v in model.state_dict().items()}; pat = 0
        else:
            pat += 1
            if pat >= 10: break
        sched.step(vl)
    if best_sd is not None:
        model.load_state_dict(best_sd)
    model.eval()
    class _P:
        def __init__(self, m): self.m = m
        def predict(self, X):
            with torch.no_grad():
                return self.m(torch.from_numpy(X.astype(np.float32)).to(DEV)).cpu().numpy().ravel()
    return _P(model)

# ========== 跑所有对比 ==========
print("\n=== 模型对比 (OOS WFO 2023+, C8 vs C6) ===")
configs = [
    # (模型名, factory, 特征集, kwargs)
    ("GBDT_C8_d3",  factory_gbdt,  C8_COLS, {}),
    ("GBDT_C6_d3",  factory_gbdt,  C6_COLS, {}),
    ("MLP_C8_small",factory_mlp,   C8_COLS, {"hidden":(32,16),    "alpha":0.5, "need_scale": True}),
    ("MLP_C6_big",  factory_mlp,   C6_COLS, {"hidden":(128,64,32),"alpha":0.3, "need_scale": True}),
    ("TabNet_C6",   factory_tabnet,C6_COLS, {"need_scale": True}),
    ("FTT_C6",      factory_ftt,   C6_COLS, {"need_scale": True}),
]

rows = []
for label, factory, feats, kw in configs:
    r = wfo_fit_predict(feats, factory, label, kw)
    print(f"  {r['model']:>14} n_feat={r['n_feat']} | IC={r['ic']:+.4f} ICIR={r['icir']:+.2f} "
          f"Q1-Q5={r['q1_q5']:+.2f} 正率={r['pos_rate']*100:.0f}% 耗时{r['sec']}s")
    rows.append(r)

pd.DataFrame(rows).to_csv(OUT, index=False, encoding="utf-8-sig")

# 优胜结论
best = max(rows, key=lambda x: x["ic"])
print(f"\n🏆 最优模型: {best['model']}, OOS IC={best['ic']:+.4f}")
print(f"   vs GBDT_C8 基准: IC差={best['ic']-rows[0]['ic']:+.4f} "
      f"({'正=DL更优' if best['ic']>rows[0]['ic'] else '负=GBDT仍优'})")
print(f"[保存] {OUT} 总耗时 {time.time()-t0:.0f}s")

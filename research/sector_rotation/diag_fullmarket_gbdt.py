# -*- coding: utf-8 -*-
"""
诊断: D盘全市场面板(5068只) 是否产出可用个股精选 alpha
全市场 vs 中证1000类子集(大盘1000) vs 小盘1000 的日频/月度 IC/ICIR 对比
标签: target_R (日频次日收益百分比近似)
"""
import sys, os, time, gc
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import lightgbm as lgb
import joblib
import pyarrow.parquet as pq
import pyarrow as pa

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

PARQUET = r"D:\iquant_data\new_end_dfC.parquet"
FEA_COLS = r"D:\iquant_data\fea_cols.joblib"
START_DATE = "2020-01-01"   # 样本期起点(训练起点)
TEST_START = "2023-01-01"   # 测试起点
TEST_END   = "2023-10-19"

t0 = time.time()
def log(msg):
    print(f"[{time.time()-t0:7.1f}s] {msg}", flush=True)

# ---- 1. 读数据(只读需要列, 排除全 null 的券商推荐列) ----
fea_cols = joblib.load(FEA_COLS)
BASE_COLS = ["date", "ts_code", "target_R", "circ_mv"]
sch = pq.ParquetFile(PARQUET).schema_arrow
fea_null = [c for c in fea_cols if pa.types.is_null(sch.field(c).type)]
fea_keep = [c for c in fea_cols
            if c not in BASE_COLS and not pa.types.is_null(sch.field(c).type)]
log(f"fea_cols 总数={len(fea_cols)}, 全 null(券商推荐)={len(fea_null)}, "
    f"有效特征={len(fea_keep)} (circ_mv 仅用于划分子集, 不作特征)")

need_cols = BASE_COLS + fea_keep
log(f"读入 parquet, 列数={len(need_cols)}, date>={START_DATE} ...")
df = pd.read_parquet(PARQUET, columns=need_cols,
                     filters=[("date", ">=", pd.Timestamp(START_DATE))])
log(f"读入完成: rows={len(df)}")

df = df.reset_index(drop=True)  # 丢弃 parquet 恢复的 DatetimeIndex(name='date'), 避免与列名歧义

df = df.dropna(subset=["target_R"])
log(f"去标签NaN后 rows={len(df)}, 股票数={df['ts_code'].nunique()}, "
    f"日期范围 {df['date'].min().date()} ~ {df['date'].max().date()}")

# ---- 2. 特征处理: 按日横截面 winsorize(1/99) + zscore ----
df = df.sort_values("date").reset_index(drop=True)
dates = df["date"].to_numpy()
X = df[fea_keep].to_numpy(dtype=np.float32)
y = df["target_R"].to_numpy(dtype=np.float64)
circ = df["circ_mv"].to_numpy(dtype=np.float64)
del df
gc.collect()

dkeys = dates.astype("datetime64[D]").astype("int64")
uniq, first, counts = np.unique(dkeys, return_index=True, return_counts=True)
n_days = len(uniq)
log(f"横截面处理: {n_days} 个交易日 x {X.shape[1]} 特征 (winsorize 1/99 + zscore)")

for i in range(n_days):
    lo = first[i]; cnt = counts[i]
    Xg = X[lo:lo+cnt]
    q01 = np.nanpercentile(Xg, 1, axis=0)
    q99 = np.nanpercentile(Xg, 99, axis=0)
    q01 = np.where(np.isnan(q01), -np.inf, q01)
    q99 = np.where(np.isnan(q99), np.inf, q99)
    Xg = np.clip(Xg, q01, q99)
    mu = np.nanmean(Xg, axis=0)
    sd = np.nanstd(Xg, axis=0)
    sd = np.where(sd < 1e-6, 1.0, sd)
    X[lo:lo+cnt] = (Xg - mu) / sd
    # 标签 winsorize
    yg = y[lo:lo+cnt]
    q01y = np.nanpercentile(yg, 1); q99y = np.nanpercentile(yg, 99)
    y[lo:lo+cnt] = np.clip(yg, q01y, q99y)

X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
log("特征处理完成")

# ---- 3. train/test 划分 ----
train_mask = dates < np.datetime64(TEST_START)
test_mask = ~train_mask
X_tr, y_tr = X[train_mask], y[train_mask]
X_te, y_te = X[test_mask], y[test_mask]
dates_te, circ_te = dates[test_mask], circ[test_mask]
log(f"训练样本={X_tr.shape[0]}, 测试样本={X_te.shape[0]} "
    f"(训练<{TEST_START}, 测试>={TEST_START})")

# ---- 4. 训练 LightGBM ----
feat_names = [f"col_{i}" for i in range(X_tr.shape[1])]
model = lgb.LGBMRegressor(
    n_estimators=500, learning_rate=0.05, num_leaves=7, max_depth=3,
    min_child_samples=80, reg_lambda=2.0, reg_alpha=0.1,
    subsample=0.9, colsample_bytree=0.9, random_state=42,
    n_jobs=-1, verbose=-1)
log("开始训练 LightGBM ...")
t_fit = time.time()
model.fit(X_tr, y_tr, feature_name=feat_names,
          callbacks=[lgb.log_evaluation(period=100)])
log(f"训练完成, 耗时 {time.time()-t_fit:.0f}s")

# ---- 5. 预测 ----
pred = model.predict(X_te)
log("预测完成")

# ---- 6. IC 评估 ----
def daily_rank_ic(pred, y, dates, min_n=50):
    d = pd.DataFrame({"date": dates, "pred": pred, "y": y})
    out = {}
    for dt, g in d.groupby("date"):
        if len(g) < min_n:
            continue
        out[dt] = g["pred"].rank().corr(g["y"].rank())
    return pd.Series(out).dropna()

def summarize(ic, name):
    ic = ic.dropna()
    mean = ic.mean(); std = ic.std()
    icir_d = mean / (std + 1e-12) * np.sqrt(252)
    pos_d = (ic > 0).mean()
    monthly = ic.groupby(ic.index.to_period("M")).mean()
    m_mean = monthly.mean(); m_std = monthly.std()
    icir_m = m_mean / (m_std + 1e-12) * np.sqrt(12)
    pos_m = (monthly > 0).mean()
    print(f"  [{name:<8}] n_days={len(ic):3d} 日频IC={mean:+.4f} ICIR={icir_d:+.3f} "
          f"正IC占比={pos_d:.1%} | 月度IC={m_mean:+.4f} 月度ICIR={icir_m:+.3f} "
          f"月度正占比={pos_m:.1%} (n_months={len(monthly)})")
    return dict(ic=mean, icir_d=icir_d, pos_d=pos_d, m_ic=m_mean, icir_m=icir_m, pos_m=pos_m)

test_df = pd.DataFrame({
    "date": np.asarray(dates_te).reshape(-1),
    "pred": np.asarray(pred).reshape(-1),
    "y": np.asarray(y_te).reshape(-1),
    "circ": np.asarray(circ_te).reshape(-1),
})
print("\n=== 测试期 OOS rank IC 对比 (2023-01 ~ 2023-10) ===")
summarize(daily_rank_ic(test_df["pred"], test_df["y"], test_df["date"]), "全市场")

large = test_df.groupby("date", group_keys=False).apply(lambda g: g.nlargest(1000, "circ"))
summarize(daily_rank_ic(large["pred"], large["y"], large["date"]), "大盘1000")

small = test_df.groupby("date", group_keys=False).apply(lambda g: g.nsmallest(1000, "circ"))
summarize(daily_rank_ic(small["pred"], small["y"], small["date"]), "小盘1000")

# ---- 7. feature importance ----
imp = model.feature_importances_
order = np.argsort(imp)[::-1]
name_map = {f"col_{i}": fea_keep[i] for i in range(len(fea_keep))}
print("\n=== feature importance top15 (split) ===")
for k in range(15):
    idx = order[k]
    print(f"  {k+1:2d}. {name_map[feat_names[idx]]:<40s} {imp[idx]:.4f}")

log(f"全部完成, 总耗时 {time.time()-t0:.0f}s")

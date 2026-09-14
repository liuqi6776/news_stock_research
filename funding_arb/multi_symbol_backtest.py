# -*- coding: utf-8 -*-
"""
多标的资金费率套利回测（币安 USDⓈ-M 永续）
策略：Delta中性收租优化版
  - 持现货1x，当刚结算费率 r_T >= theta_in 时做空1x合约对冲收租
  - 当 r_T <= theta_out（默认0，即实际转负）时平掉对冲腿（滞后阈值，避免来回打脸）
  - 反手/开平均用 maker 0.02%
参数网格: theta_in ∈ {0, 0.005%, 0.01%, 0.02%}
"""
import sys, time
from pathlib import Path
import requests
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(sys.executable).parent.parent.parent))
from daimon_runtime import setup_plot
setup_plot()
import matplotlib.pyplot as plt

BASE = "https://fapi.binance.com"
OUT = Path(__file__).parent
DAYS = 365 * 3
now_ms = int(time.time() * 1000)
start_ms = now_ms - DAYS * 86400_000

# ---------- 1. 选出成交量最高的 USDT 永续 ----------
for attempt in range(3):
    r = requests.get(f"{BASE}/fapi/v1/ticker/24hr", timeout=20)
    data = r.json()
    if isinstance(data, list):
        break
    print(f"ticker返回异常(第{attempt+1}次): {str(data)[:200]}")
    time.sleep(2)
tick = pd.DataFrame(data)
tick = tick[tick["symbol"].str.endswith("USDT")]
tick["quoteVolume"] = tick["quoteVolume"].astype(float)
# 排除稳定币/法币类和指数类
exclude = {"USDCUSDT", "BUSDUSDT", "TUSDUSDT", "FDUSDUSDT", "EURUSDT", "AEURUSDT", "XUSDUSDT", "BFUSDUSDT", "USD1USDT", "USDEUSDT", "USDPUSDT", "GYENUSDT", "BIDRUSDT", "IDRTUSDT", "NGNUSDT", "UAHUSDT", "ZARUSDT", "ARSUSDT", "BRLUSDT", "TRYUSDT", "RUBUSDT", "DAIUSDT"}
tick = tick[~tick["symbol"].isin(exclude)]
top = tick.nlargest(15, "quoteVolume")["symbol"].tolist()
print("TOP15 by 24h quote volume:", top)

# 检查币安是否有股票类永续
for attempt in range(6):
    exinfo = requests.get(f"{BASE}/fapi/v1/exchangeInfo", timeout=20).json()
    if "symbols" in exinfo:
        break
    print(f"exchangeInfo异常(第{attempt+1}次): {str(exinfo)[:120]}")
    time.sleep(20)
all_syms = [s["symbol"] for s in exinfo["symbols"]]
stock_like = [s for s in all_syms if any(k in s for k in ["TSLA", "AAPL", "NVDA", "MSFT", "AMZN", "GOOG", "META", "COIN", "MSTR", "SPY", "QQQ"])]
# 股票类永续（币安已上线）+ 加密热门，合并去重
STOCK_PERPS = ["TSLAUSDT", "NVDAUSDT", "AAPLUSDT", "MSFTUSDT", "AMZNUSDT",
               "GOOGLUSDT", "METAUSDT", "COINUSDT", "MSTRUSDT", "SPYUSDT", "QQQUSDT"]
print("币安合约中股票类标的:", [s for s in stock_like if s in STOCK_PERPS or s not in STOCK_PERPS])
symbols = list(dict.fromkeys(top + STOCK_PERPS))

# ---------- 2. 拉数据 ----------
def _get(path, params):
    for attempt in range(6):
        rr = requests.get(f"{BASE}{path}", params=params, timeout=20)
        data = rr.json()
        if isinstance(data, list):
            return data
        msg = str(data)[:120]
        wait = 20 if ("banned" in msg or "-1003" in msg or "418" in msg or "429" in msg) else 2
        print(f"  请求异常(第{attempt+1}次, 等{wait}s): {msg}")
        time.sleep(wait)
    raise RuntimeError(f"请求失败: {path} {params}")

def fetch_funding(symbol):
    rows, st = [], start_ms
    while True:
        data = _get("/fapi/v1/fundingRate", {"symbol": symbol, "startTime": st, "limit": 1000})
        if not data: break
        rows += data
        last = data[-1]["fundingTime"]
        if len(data) < 1000 or last >= now_ms - 8 * 3600_000: break
        st = last + 1
        time.sleep(0.4)
    if not rows: return None
    df = pd.DataFrame(rows)
    df["fundingTime"] = pd.to_datetime(df["fundingTime"], unit="ms")
    df["fundingRate"] = pd.to_numeric(df["fundingRate"], errors="coerce")
    return df.dropna(subset=["fundingRate"]).drop_duplicates("fundingTime").sort_values("fundingTime").reset_index(drop=True)

def fetch_klines(symbol):
    rows, st = [], start_ms
    while True:
        data = _get("/fapi/v1/klines", {"symbol": symbol, "interval": "8h",
                                        "startTime": st, "limit": 1500})
        if not data: break
        rows += data
        last = data[-1][0]
        if len(data) < 1500: break
        st = last + 1
        time.sleep(0.4)
    if not rows: return None
    df = pd.DataFrame(rows, columns=["openTime","open","high","low","close","vol",
                                     "closeTime","qv","trades","tbb","tbq","ig"])
    df["openTime"] = pd.to_datetime(df["openTime"], unit="ms")
    df["close"] = df["close"].astype(float)
    return df[["openTime","close"]].drop_duplicates("openTime").sort_values("openTime").reset_index(drop=True)

# ---------- 3. 回测 ----------
MAKER = 0.0002
def backtest(df, theta_in, theta_out=0.0):
    hedge = 0.0
    eq = 1.0; fund_sum = 0.0; fee_sum = 0.0; nswitch = 0
    eqs = [1.0]
    for _, row in df.iterrows():
        r_t = row["fundingRate"]
        target = -1.0 if r_t >= theta_in else (0.0 if r_t <= theta_out else hedge)
        # theta_out < r < theta_in 时保持原状态（滞后带）
        if target != hedge:
            fee = abs(target - hedge) * MAKER
            fee_sum += fee; eq -= fee; nswitch += 1
            hedge = target
        pnl_price = (1 + hedge) * row["ret_next"]
        pnl_fund = -hedge * row["r_next"]
        fund_sum += pnl_fund
        eq = eq * (1 + pnl_price) + pnl_fund * eq
        eqs.append(eq)
    eqs = pd.Series(eqs)
    rets = eqs.pct_change().dropna()
    n = len(df)
    yrs = n * 8 / 24 / 365
    total = eqs.iloc[-1] - 1
    ann = (1 + total) ** (1 / yrs) - 1 if yrs > 0 else np.nan
    dd = (eqs / eqs.cummax() - 1).min()
    sharpe = rets.mean() / (rets.std() + 1e-12) * np.sqrt(3 * 365)
    return dict(total=total, ann=ann, mdd=dd, sharpe=sharpe,
                fund=fund_sum, fees=fee_sum, nswitch=nswitch, eqs=eqs, yrs=yrs)

results = []
curves = {}
detail_rows = []
for sym in symbols:
    try:
        f_csv = OUT / f"{sym}_funding.csv"
        k_csv = OUT / f"{sym}_klines.csv"
        if f_csv.exists():
            f = pd.read_csv(f_csv, parse_dates=["fundingTime"])
        else:
            f = fetch_funding(sym)
            if f is not None: f.to_csv(f_csv, index=False)
        if k_csv.exists():
            k = pd.read_csv(k_csv, parse_dates=["openTime"])
        else:
            k = fetch_klines(sym)
            if k is not None: k.to_csv(k_csv, index=False)
        print(f"  {sym}: funding={0 if f is None else len(f)}, klines={0 if k is None else len(k)}")
        if f is None or k is None or len(f) < 60:
            print(f"{sym}: 数据不足，跳过")
            continue
        f["fundingTime"] = pd.to_datetime(f["fundingTime"]).dt.floor("8h")
        frate = f.groupby("fundingTime")["fundingRate"].sum().reset_index()  # 4h结算的标的聚合到8h
        df = k.merge(frate, left_on="openTime", right_on="fundingTime", how="left")
        df["fundingRate"] = df["fundingRate"].ffill()
        df = df.dropna(subset=["fundingRate"]).reset_index(drop=True)
        df["ret_next"] = df["close"].shift(-1) / df["close"] - 1
        df["r_next"] = df["fundingRate"].shift(-1)
        df = df.dropna().reset_index(drop=True)
        print(f"  {sym}: 对齐后={len(df)}")
        if len(df) < 60:
            print(f"{sym}: 对齐后数据不足，跳过")
            continue

        avg_rate = df["fundingRate"].mean()
        pos_ratio = (df["fundingRate"] > 0).mean()
        gross_carry = df.loc[df["r_next"] > 0, "r_next"].sum()  # 对冲时可捕获的正费率总量
        bh_total = df["close"].iloc[-1] / df["close"].iloc[0] - 1

        # 参数网格
        best = None
        for th in [0.0, 0.00005, 0.0001, 0.0002]:
            res = backtest(df, theta_in=th, theta_out=0.0)
            res["theta_in"] = th
            if best is None or res["sharpe"] > best["sharpe"]:
                best = res
        base = backtest(df, theta_in=0.0, theta_out=0.0)

        curves[sym] = (df["openTime"], best["eqs"])
        results.append(dict(标的=sym, 数据年数=round(best["yrs"], 1),
                            平均费率bp8h=round(avg_rate * 1e4, 2),
                            正费率占比=f"{pos_ratio*100:.0f}%",
                            最优阈值bp=round(best["theta_in"] * 1e4, 1),
                            年化=f"{best['ann']*100:.1f}%",
                            最大回撤=f"{best['mdd']*100:.1f}%",
                            夏普=round(best["sharpe"], 2),
                            切换次数=best["nswitch"],
                            费率收入pct=round(best["fund"]*100, 1),
                            手续费pct=round(best["fees"]*100, 1),
                            买入持有=f"{bh_total*100:.0f}%",
                            阈值0年化=f"{base['ann']*100:.1f}%"))
        print(f"{sym}: done, best theta={best['theta_in']*1e4:.1f}bp, ann={best['ann']*100:.1f}%, sharpe={best['sharpe']:.2f}")
    except Exception as e:
        print(f"{sym}: ERROR {e}")

res_df = pd.DataFrame(results).sort_values("夏普", ascending=False)
res_df.to_csv(OUT / "multi_symbol_results.csv", index=False)
print("\n", res_df.to_string(index=False))

# ---------- 4. 画图 ----------
fig, axes = plt.subplots(1, 2, figsize=(15, 6))
ax = axes[0]
plot_syms = res_df.head(8)["标的"].tolist()
for sym in plot_syms:
    t, eqs = curves[sym]
    tt = pd.concat([pd.Series([t.iloc[0]]), t]).iloc[:len(eqs)]
    ax.plot(tt, eqs.values, label=sym, lw=1.1)
ax.set_yscale("log")
ax.set_title("Delta中性费率套利净值（最优阈值，maker 0.02%）")
ax.legend(fontsize=8); ax.grid(alpha=0.3)

ax2 = axes[1]
rs = res_df.sort_values("年化", key=lambda s: s.str.rstrip('%').astype(float), ascending=True)
vals = rs["年化"].str.rstrip('%').astype(float)
colors = ["darkorange" if s in ("BTCUSDT", "ETHUSDT") else "steelblue" for s in rs["标的"]]
ax2.barh(rs["标的"], vals, color=colors)
ax2.set_title("各标的年化收益对比（%）")
ax2.grid(alpha=0.3, axis="x")
fig.tight_layout()
fig.savefig(OUT / "multi_symbol_backtest.png", bbox_inches="tight", dpi=130)
print("saved:", OUT / "multi_symbol_backtest.png")

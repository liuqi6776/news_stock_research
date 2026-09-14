# -*- coding: utf-8 -*-
"""
币安 BTCUSDT 永续合约资金费率策略回测
策略V1（朋友的做法）：方向跟随资金费率 —— 费率为正做空收租，为负做多收租，翻信号即平仓反手
策略V2（Delta中性）：持BTC现货 + 费率预期为正时做空1x合约对冲（为负时空仓仅持币）
基准：买入持有BTC
信号规则（无前视偏差）：在结算时刻T，用刚结算的费率 r_T 的符号作为下一区间 [T, T+8h] 的方向预测
"""
import sys, time, json
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

def fetch_funding():
    rows, st = [], start_ms
    while True:
        r = requests.get(f"{BASE}/fapi/v1/fundingRate",
                         params={"symbol": "BTCUSDT", "startTime": st, "limit": 1000},
                         timeout=20)
        r.raise_for_status()
        data = r.json()
        if not data:
            break
        rows += data
        last = data[-1]["fundingTime"]
        if len(data) < 1000 or last >= now_ms - 8 * 3600_000:
            break
        st = last + 1
        time.sleep(0.2)
    df = pd.DataFrame(rows)
    df["fundingTime"] = pd.to_datetime(df["fundingTime"], unit="ms")
    df["fundingRate"] = pd.to_numeric(df["fundingRate"], errors="coerce")
    df["markPrice"] = pd.to_numeric(df["markPrice"], errors="coerce")
    df = df.dropna(subset=["fundingRate"])
    return df.drop_duplicates("fundingTime").sort_values("fundingTime").reset_index(drop=True)

def fetch_klines_8h():
    rows, st = [], start_ms
    while True:
        r = requests.get(f"{BASE}/fapi/v1/klines",
                         params={"symbol": "BTCUSDT", "interval": "8h",
                                 "startTime": st, "limit": 1500}, timeout=20)
        r.raise_for_status()
        data = r.json()
        if not data:
            break
        rows += data
        last = data[-1][0]
        if len(data) < 1500:
            break
        st = last + 1
        time.sleep(0.2)
    df = pd.DataFrame(rows, columns=["openTime","open","high","low","close","vol",
                                     "closeTime","qv","trades","tbb","tbq","ig"])
    df["openTime"] = pd.to_datetime(df["openTime"], unit="ms")
    for c in ["open","close"]:
        df[c] = df[c].astype(float)
    return df[["openTime","open","close"]].drop_duplicates("openTime").sort_values("openTime").reset_index(drop=True)

fund = fetch_funding()
kl = fetch_klines_8h()
fund.to_csv(OUT / "btcusdt_funding_history.csv", index=False)
kl.to_csv(OUT / "btcusdt_klines_8h.csv", index=False)
print(f"funding records: {len(fund)}  {fund.fundingTime.iloc[0]} -> {fund.fundingTime.iloc[-1]}")
print(f"klines: {len(kl)}")

# ---- 对齐：资金费结算时刻 == 8h K线开盘时刻（00/08/16 UTC） ----
df = kl.merge(fund[["fundingTime","fundingRate"]], left_on="openTime", right_on="fundingTime", how="left")
df["fundingRate"] = df["fundingRate"].ffill()
df = df.dropna(subset=["fundingRate"]).reset_index(drop=True)

# 区间 i: [T_i, T_i+1]，在 T_i 时刻已知 r_i（刚结算），用 sign(r_i) 预测下一区间方向
# 该区间实际结算的费率是 r_{i+1}
df["ret_next"] = df["close"].shift(-1) / df["close"] - 1.0      # 区间价格收益
df["r_next"] = df["fundingRate"].shift(-1)                       # 区间末结算费率
df["signal"] = np.sign(df["fundingRate"])                        # T_i 时刻的方向预测
df = df.dropna().reset_index(drop=True)

TAKER = 0.0005  # 币安合约 taker 0.05%

def stats(eq, name):
    eq = pd.Series(eq)
    rets = eq.pct_change().dropna()
    total = eq.iloc[-1] / eq.iloc[0] - 1
    yrs = len(eq) * 8 / 24 / 365
    ann = (1 + total) ** (1 / yrs) - 1
    dd = (eq / eq.cummax() - 1).min()
    sharpe = rets.mean() / (rets.std() + 1e-12) * np.sqrt(3 * 365)
    return {"策略": name, "总收益": f"{total*100:.1f}%", "年化": f"{ann*100:.1f}%",
            "最大回撤": f"{dd*100:.1f}%", "夏普": f"{sharpe:.2f}"}

# ---- V1：朋友的策略 —— 方向跟随费率（纯合约1x，收租方向持仓） ----
pos = 0.0
eq1 = [1.0]; fund_income1 = 0.0; fees1 = 0.0; flips = 0
for _, row in df.iterrows():
    target = -row["signal"]  # 费率为正 -> 做空收租(-1)；为负 -> 做多(+1)
    if target != pos:
        fee = abs(target - pos) * TAKER
        fees1 += fee
        eq1[-1] -= fee
        if pos != 0: flips += 1
        pos = target
    pnl_price = pos * row["ret_next"]
    pnl_fund = -pos * row["r_next"]   # 多头支付正费率，空头收
    fund_income1 += pnl_fund
    eq1.append(eq1[-1] * (1 + pnl_price) + pnl_fund * eq1[-1])

# ---- V2：Delta中性 —— 持现货1x + 信号为正时做空1x合约；为负时空仓（仅持币） ----
hedge = 0.0
eq2 = [1.0]; fund_income2 = 0.0; fees2 = 0.0
for _, row in df.iterrows():
    target = -1.0 if row["signal"] > 0 else 0.0
    if target != hedge:
        fee = abs(target - hedge) * TAKER
        fees2 += fee
        eq2[-1] -= fee
        hedge = target
    pnl_price = (1 + hedge) * row["ret_next"]   # 现货1x + 对冲腿
    pnl_fund = -hedge * row["r_next"]
    fund_income2 += pnl_fund
    eq2.append(eq2[-1] * (1 + pnl_price) + pnl_fund * eq2[-1])

# ---- V3：朋友完整版 —— 始终持有BTC(抵押) + 合约跟随费率方向（正费率做空=对冲，负费率做多=2x多头） ----
leg = 0.0
eq4 = [1.0]; fund_income4 = 0.0; fees4 = 0.0; flips4 = 0
for _, row in df.iterrows():
    target = -row["signal"]  # 合约腿
    if target != leg:
        fee = abs(target - leg) * TAKER
        fees4 += fee
        eq4[-1] -= fee
        if leg != 0: flips4 += 1
        leg = target
    pnl_price = (1 + leg) * row["ret_next"]  # 现货1x + 合约腿
    pnl_fund = -leg * row["r_next"]
    fund_income4 += pnl_fund
    eq4.append(eq4[-1] * (1 + pnl_price) + pnl_fund * eq4[-1])

# ---- 基准：买入持有 ----
eq3 = (1 + df["ret_next"]).cumprod().values
eq3 = np.concatenate([[1.0], eq3])

# ---- 信号准确率：sign(r_i) 预测 sign(r_{i+1}) ----
acc = (np.sign(df["fundingRate"]) == np.sign(df["r_next"])).mean()
pos_ratio = (df["fundingRate"] > 0).mean()

print(f"\n信号方向准确率 sign(r_t)->sign(r_t+8h): {acc*100:.1f}%")
print(f"费率为正的时间占比: {pos_ratio*100:.1f}%")
print(f"V1 反手次数: {flips}  累计费率收入: {fund_income1*100:.2f}%  累计手续费: {fees1*100:.2f}%")
print(f"V2 累计费率收入: {fund_income2*100:.2f}%  累计手续费: {fees2*100:.2f}%")
print(f"V3 反手次数: {flips4}  累计费率收入: {fund_income4*100:.2f}%  累计手续费: {fees4*100:.2f}%")

res = pd.DataFrame([
    stats(eq1, "V1 纯合约方向跟随费率"),
    stats(eq2, "V2 Delta中性（持币+正费率时对冲）"),
    stats(eq4, "V3 朋友完整版（押BTC+合约跟随费率）"),
    stats(eq3, "基准：买入持有BTC"),
])
print("\n", res.to_string(index=False))
res.to_csv(OUT / "backtest_summary.csv", index=False)

# ---- 画图 ----
t = pd.concat([pd.Series([df["openTime"].iloc[0]]), df["openTime"]]).iloc[:len(eq1)]
fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True,
                         gridspec_kw={"height_ratios": [2, 1]})
ax = axes[0]
ax.plot(t, eq1, label="V1 纯合约方向跟随", lw=1.2)
ax.plot(t, eq2, label="V2 Delta中性", lw=1.2)
ax.plot(t, eq4, label="V3 朋友完整版（押币+跟随费率）", lw=1.4)
ax.plot(t, eq3, label="买入持有BTC", lw=1.2, alpha=0.7)
ax.set_yscale("log")
ax.set_ylabel("净值（对数）")
ax.set_title("BTCUSDT 永续资金费率策略回测（近3年，币安，taker 0.05%）")
ax.legend(); ax.grid(alpha=0.3)

ax2 = axes[1]
ax2.plot(df["openTime"], df["fundingRate"] * 100, lw=0.6, color="darkorange")
ax2.axhline(0, color="gray", lw=0.5)
ax2.set_ylabel("资金费率 %/8h")
ax2.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(OUT / "funding_backtest.png", bbox_inches="tight", dpi=130)
print("saved:", OUT / "funding_backtest.png")

# 额外统计：费率收入本身的年化（纯收租部分）
ann_fund = df["r_next"].abs().mean() * 3 * 365
print(f"\n近3年平均|费率|年化（收租毛收益上限）: {ann_fund*100:.1f}%")
print(f"近3年平均费率（多头付空头方向）: {df['fundingRate'].mean()*100:.4f}%/8h")

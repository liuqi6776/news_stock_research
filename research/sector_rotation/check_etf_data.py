# -*- coding: utf-8 -*-
"""检查 512100 ETF close vs pct_chg 不一致问题"""
import pandas as pd
import os

fp = "c:/Users/liuqi/quant_system_v2/research/chip_momentum/data/index_daily/512100.SH.parquet"
df = pd.read_parquet(fp)
df["trade_date"] = df["trade_date"].astype(str)
df = df.sort_values("trade_date").reset_index(drop=True)
df["ym"] = df["trade_date"].str[:6]

print("=== 512100 ETF 原始数据检查 ===")
print("Columns:", list(df.columns))
print("Shape:", df.shape)
print("Date range:", df["trade_date"].min(), "~", df["trade_date"].max())
print()

# 取首月末和末月末
me = df.groupby("ym").last().reset_index()
me = me[(me["ym"] >= "202003") & (me["ym"] <= "202608")]
print(f"首月末(202003): close={me['close'].iloc[0]:.4f}  pct_chg={me['pct_chg'].iloc[0]:.4f}")
print(f"末月末(202608): close={me['close'].iloc[-1]:.4f}  pct_chg={me['pct_chg'].iloc[-1]:.4f}")
print(f"NAV(close相除): {me['close'].iloc[-1] / me['close'].iloc[0]:.4f}")
print()

# 用 pct_chg 累乘
df2 = df[(df["ym"] >= "202003") & (df["ym"] <= "202608")].copy()
daily_ret = df2["pct_chg"].fillna(0) / 100.0
nav_pctchg = (1 + daily_ret).prod()
print(f"NAV(pct_chg累乘): {nav_pctchg:.4f}")
print()

# 检查 close 跳变
print("=== close 日跳变 >15% 的记录 ===")
df2["close_pct"] = df2["close"].pct_change()
jumps = df2[df2["close_pct"].abs() > 0.15]
if len(jumps) > 0:
    print(jumps[["trade_date", "close", "pre_close", "pct_chg", "close_pct"]].head(20).to_string())
else:
    print("无大幅跳变")

# 检查 close vs pre_close 的关系
print()
print("=== close vs pre_close 一致性检查 (前10条) ===")
df2["pct_from_close"] = (df2["close"] / df2["close"].shift(1) - 1) * 100
df2["pct_diff"] = df2["pct_chg"] - df2["pct_from_close"]
inconsistent = df2[df2["pct_diff"].abs() > 0.5]
print(f"pct_chg 与 close推算 不一致(差>0.5pp) 的天数: {len(inconsistent)} / {len(df2)}")
if len(inconsistent) > 0:
    print(inconsistent[["trade_date", "close", "pre_close", "pct_chg", "pct_from_close", "pct_diff"]].head(10).to_string())

# 检查是否有复权标记
print()
print("=== 检查是否有 adj_factor 或复权信息 ===")
if "adj_factor" in df.columns:
    print("adj_factor 存在!")
    print(df[["trade_date", "close", "adj_factor"]].head(5))
    print(df[["trade_date", "close", "adj_factor"]].tail(5))
else:
    print("无 adj_factor 列")

# 看看 close 是否看起来像前复权
print()
print("=== close 价格走势抽样 ===")
sample_dates = ["20200331", "20210630", "20221230", "20230630", "20240628", "20250630", "20260731"]
for d in sample_dates:
    row = df[df["trade_date"] <= d].tail(1)
    if len(row) > 0:
        print(f"  {row['trade_date'].iloc[0]}: close={row['close'].iloc[0]:.4f}")

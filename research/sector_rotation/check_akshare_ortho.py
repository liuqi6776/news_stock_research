# -*- coding: utf-8 -*-
"""测试 akshare 正交数据源可得性 (北向个股 / 分析师预期)"""
import warnings
warnings.filterwarnings("ignore")

import akshare as ak
print("akshare", ak.__version__)

# 1. 北向资金个股 (沪股通/深股通持股)
print("\n=== 1. 北向个股资金流 stock_hsgt_individual_em ===")
try:
    df = ak.stock_hsgt_individual_em(symbol="北向资金", start_date="20240101", end_date="20240201")
    print("   rows:", len(df), "cols:", list(df.columns))
    print(df.head(3).to_string()[:800])
except Exception as e:
    print("   ERR:", type(e).__name__, str(e)[:200])

print("\n=== 2. 个股评级 stock_rating_em ===")
try:
    df = ak.stock_rating_em(symbol="000001")
    print("   rows:", len(df), "cols:", list(df.columns))
    print(df.head(3).to_string()[:600])
except Exception as e:
    print("   ERR:", type(e).__name__, str(e)[:200])

print("\n=== 3. 盈利预测 stock_profit_forecast_em ===")
try:
    df = ak.stock_profit_forecast_em(symbol="000001")
    print("   rows:", len(df), "cols:", list(df.columns))
    print(df.head(3).to_string()[:600])
except Exception as e:
    print("   ERR:", type(e).__name__, str(e)[:200])

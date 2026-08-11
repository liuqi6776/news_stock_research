# -*- coding: utf-8 -*-
"""验证 akshare 北向个股/分析师/文本数据能否补齐历史面板"""
import warnings
warnings.filterwarnings("ignore")
import akshare as ak
import inspect

# 1. 北向个股历史
for name in ['stock_hsgt_individual_em', 'stock_hsgt_individual_detail_em', 'stock_hsgt_hold_stock_em']:
    fn = getattr(ak, name, None)
    if fn is None:
        print(name, ': MISSING'); continue
    try:
        sig = inspect.signature(fn)
        print(name, 'sig:', sig)
    except Exception as e:
        print(name, 'sig ERR:', e)

print('\n=== stock_hsgt_individual_em 试拉 ===')
try:
    df = ak.stock_hsgt_individual_em(symbol="北向资金")
    print('rows:', len(df), 'cols:', list(df.columns))
    print(df.head(3).to_string()[:600])
except Exception as e:
    print('ERR:', type(e).__name__, str(e)[:300])

print('\n=== stock_hsgt_hold_stock_em 北向持股 ===')
try:
    df = ak.stock_hsgt_hold_stock_em(market="北向", indicator="今日排行")
    print('rows:', len(df), 'cols:', list(df.columns))
    print(df.head(3).to_string()[:600])
except Exception as e:
    print('ERR:', type(e).__name__, str(e)[:300])

# -*- coding: utf-8 -*-
import warnings
warnings.filterwarnings("ignore")
import akshare as ak

print('=== stock_hsgt_individual_detail_em 000001 20240101-20240201 ===')
try:
    df = ak.stock_hsgt_individual_detail_em(symbol="000001", start_date="20240101", end_date="20240201")
    print('rows:', len(df), 'cols:', list(df.columns))
    print(df.head(3).to_string()[:800])
except Exception as e:
    print('ERR:', type(e).__name__, str(e)[:400])

print('\n=== 尝试更早年份 20230101-20230201 ===')
try:
    df = ak.stock_hsgt_individual_detail_em(symbol="000001", start_date="20230101", end_date="20230201")
    print('rows:', len(df), 'cols:', list(df.columns))
    print(df.head(3).to_string()[:800])
except Exception as e:
    print('ERR:', type(e).__name__, str(e)[:400])

# -*- coding: utf-8 -*-
import importlib
for m in ['torch','lightgbm','akshare','sklearn','numpy','pandas','pyarrow','tushare']:
    try:
        mod = importlib.import_module(m)
        print(m, ':', getattr(mod, "__version__", "?"))
    except Exception as e:
        print(m, ': MISSING (', type(e).__name__, ')')

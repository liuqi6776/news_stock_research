# -*- coding: utf-8 -*-
"""T9 诊断2: 日频T7 在 2019-02 只赚3.9% vs 月频17% 的逐日追踪"""
import os
import sys

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "fund_research", "studies", "rotation_dingtou"))

from diag_t9_exit import (  # noqa: E402
    sig, dates, ew_trad, v8_daily, month_end_dates, COST,
)

print("sig['s123'] 2018-12 ~ 2019-06:")
print(sig["s123"].loc["201812":"201906"])

# 重新构建 sig_daily
ym_of = pd.Series(dates, index=dates).str[:6]
sig_daily = sig["s123"].reindex(pd.Index(dates.str[:6])).fillna(0).astype(int)
sig_daily.index = dates
sig_daily_shifted = sig_daily.shift(1).fillna(0).astype(int)

# 打印 2019-01-25 ~ 2019-04-05 的逐日: 日期/原信号/ shifted信号/ 传统行业收益
print("\n逐日追踪 2019-01-25 ~ 2019-04-05:")
print(f"{'日期':<10} {'sig当月':>4} {'shifted':>4} {'ew_trad':>8} {'v8':>8}")
sub = dates[(dates >= "20190125") & (dates <= "20190405")]
for d in sub:
    s0 = sig_daily.loc[d]
    s1 = sig_daily_shifted.loc[d]
    print(f"{d:<10} {s0:>4} {s1:>4} {float(ew_trad.loc[d])*100:>7.2f}% {float(v8_daily.get(d,0))*100:>7.2f}%")

# 月频T7持有的月份 (2019)
print("\n[sig] 2019 全年 s123:")
print(sig["s123"].loc["201901":"201912"].to_string())

# -*- coding: utf-8 -*-
"""核对: 状态机进出场 24 次切换 vs 交易明细 17 笔 差异来源"""
import os
import sys
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "fund_research", "studies", "rotation_dingtou"))

from diag_t9_exit import build_entry_raw, build_exit_raw, dates, ew_trad, v8_daily, COST  # noqa: E402

entry = build_entry_raw()
exit_raw = build_exit_raw("ma5").shift(1).fillna(False)
min_hold = 20

state = "out"
hold_days = 0
switches = 0
trades = []
in_date = None

for i, d in enumerate(dates):
    if i > 0:
        p = dates[i-1]
        if state == "out":
            if entry.loc[p]:
                state = "in"; switches += 1
                in_date = d
                trades.append({"进场": d})
        else:
            hold_days += 1
            if hold_days >= min_hold and exit_raw.loc[p]:
                state = "out"; switches += 1
                hold_days = 0
                trades[-1]["出场"] = d
    if state == "in" and hold_days == 0:
        hold_days = 1  # 防止 hold_days=0 次日还没执行到

df = pd.DataFrame(trades)
print(f"状态切换次数: {switches}  (进 {switches//2} + 出 {switches//2})")
print(f"实际交易笔数: {len(trades)} 笔")
print(df.to_string())

# 检查是否存在"还在 out 时, entry 信号重复触发但没动作"
print(f"\nentry 信号日数: {int(entry.sum())}  其中实际进场的次数: {len(trades)}")

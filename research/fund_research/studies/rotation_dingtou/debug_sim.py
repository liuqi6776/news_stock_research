# -*- coding: utf-8 -*-
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from invest_plan import load_all, simulate_dca, TEST_START, TEST_END

navs, rets = load_all()
pool = ["纯债", "黄金", "纳指", "沪深300", "QDII债", "原油"]
w = {c: 1/len(pool) for c in pool}

print("测试简单定投: 6资产等权, 月投, 不调仓")
try:
    sim = simulate_dca(navs, pool, w, TEST_START, TEST_END,
                       annual_cash=100_000, freq="M", rebalance="none")
    print(f"总投入: {sim['投入总额']:,.0f}")
    print(f"期末值: {sim['期末值']:,.0f}")
    print(f"期末/投入: {sim['期末值']/sim['投入总额']:.2%}")
    print(f"笔数: {sim['总投入笔数']}")
    dv = sim["每日市值"]
    print(f"每日市值长度: {len(dv)}")
    if len(dv) > 10:
        mdd = float((dv / dv.cummax() - 1).min())
        print(f"回撤: {mdd:.2%}")
    import traceback
    r = sim["现金流水"]
    if len(r) >= 2:
        from invest_plan import xirr_from_cf
        print(f"XIRR: {xirr_from_cf(r):.2%}")
except Exception as e:
    import traceback
    traceback.print_exc()

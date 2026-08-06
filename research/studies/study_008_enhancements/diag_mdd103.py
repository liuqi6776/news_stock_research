# -*- coding: utf-8 -*-
"""P0-4 诊断: 复现 factor_dic_validation 中 MaxDD>100% 的机制 (如 lg_net_5d 103.3%)

定位: ① 是否存在单月组合收益 < -100% (nav 转负)?
      ② 绝对回撤 (cummax-nav) 与相对回撤 ((cummax-nav)/cummax) 的差异
      ③ 异常月内单日极端个股收益 (数据异常/复权缺口)
复用 run_validation 的加载逻辑, 不改上游代码。
用法: python diag_mdd103.py [factor_key]  默认 lg_net_5d
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
from research.factor_dic import run_validation as rv
from research.factor_dic import factor_lib

FKEY = sys.argv[1] if len(sys.argv) > 1 else "lg_net_5d"


def main():
    entry = next(e for e in factor_lib.FACTOR_REGISTRY if e[0] == FKEY)
    _, name, direction, _, need = entry
    print(f"[diag] 因子 {FKEY} ({name}) need={need}")

    trade_dates = rv.load_trade_dates()
    months = {}
    for d in trade_dates:
        if d[:4] >= str(rv.START_YEAR):
            months[d[:6]] = d
    rebal_dates = sorted(months.values())[:-1]
    all_codes = set()
    for rb in rebal_dates:
        m = rv.load_index_weight(rb)
        if m:
            all_codes |= m
    all_codes = sorted(all_codes)
    stocks, pct_df, cyq_g, mf_g, mkt_ret = rv.load_panels(trade_dates, all_codes, need)
    print(f"[diag] 调仓 {len(rebal_dates)} 个, 成分 {len(all_codes)} 只")

    factor_series = {}
    for i, code in enumerate(all_codes):
        df = stocks.get(code)
        if df is None or len(df) < 60:
            continue
        cyq_df = mf_df = None
        if need == "chip":
            cyq_df = cyq_g.get(code)
            if cyq_df is not None:
                cyq_df = cyq_df[~cyq_df.index.duplicated(keep="last")]
                if "close" not in cyq_df.columns:
                    cyq_df = cyq_df.join(df["close"], how="left")
        if need == "mf":
            mf_df = mf_g.get(code)
            if mf_df is not None:
                mf_df = mf_df[~mf_df.index.duplicated(keep="last")]
        s = rv.build_factor_series(FKEY, df, cyq_df, mf_df, mkt_ret)
        if s is not None and len(s.dropna()) > 0:
            factor_series[code] = s
    print(f"[calc] 有效个股 {len(factor_series)}")

    # 复现 Top60 月度回测 + 诊断
    port_rets, picks_records = [], []
    for i, rb in enumerate(rebal_dates):
        if i + 1 >= len(rebal_dates):
            break
        members = rv.load_index_weight(rb)
        if members is None:
            continue
        fvals = {}
        for code in members:
            fs = factor_series.get(code)
            if fs is None:
                continue
            if rb in fs.index and pd.notna(fs.loc[rb]):
                fvals[code] = fs.loc[rb]
        if len(fvals) < rv.TOP_N:
            continue
        picks = pd.Series(fvals).nlargest(rv.TOP_N).index
        rb_next = rebal_dates[i + 1]
        hi, hn = trade_dates.index(rb), trade_dates.index(rb_next)
        hold_dates = trade_dates[hi + 1: hn + 1]
        sub = pct_df.reindex(columns=picks).reindex(hold_dates).fillna(0.0) / 100.0
        rm_daily = sub.mean(axis=1)
        gross = (1 + rm_daily).prod() - 1
        net = gross - rv.COST_BPS / 10000.0
        port_rets.append(net)
        picks_records.append((rb, rb_next, picks, sub, rm_daily, gross))
        if net < -0.9:
            print(f"  !! 异常月 {rb}~{rb_next}: net {net:.4f} (gross {gross:.4f})")
            for d, v in rm_daily.items():
                if v < -0.15:
                    row = sub.loc[d]
                    worst = row[row < -0.5]
                    print(f"      日 {d} 组合日收益 {v:.4f}; 极端个股(<-50%): {[(c, round(x,4)) for c, x in worst.items()][:10]}")

    pr = pd.Series(port_rets)
    nav = (1 + pr).cumprod()
    mdd_abs = (nav.cummax() - nav).max()
    mdd_rel = ((nav.cummax() - nav) / nav.cummax()).max()
    print(f"\n[结果] 月数 {len(pr)}")
    print(f"  月收益: min {pr.min():.4f} max {pr.max():.4f} | nav: min {nav.min():.4f} max {nav.max():.4f}")
    print(f"  绝对回撤 (脚本现用): {mdd_abs:.2%}")
    print(f"  相对回撤 (正确口径): {mdd_rel:.2%}")
    neg = nav[nav <= 0]
    if len(neg):
        print(f"  !! nav<=0 的月: {list(neg.index)} 首末 {neg.iloc[0] if len(neg) else ''}")
    print(f"  净值最低月: {nav.idxmin()} = {nav.min():.4f}")
    # 最低月往前看持仓与收益明细
    i_min = int(nav.idxmin())
    print(f"  净值最低月上下文 (index {i_min}):")
    for k in range(max(0, i_min - 3), min(len(pr), i_min + 3)):
        rb0, rb1, _, _, _, g = picks_records[k]
        print(f"    {rb0}~{rb1}: net {pr.iloc[k]:.4f} (gross {g:.4f}) nav {nav.iloc[k]:.4f}")


if __name__ == "__main__":
    main()

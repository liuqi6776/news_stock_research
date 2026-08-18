# -*- coding: utf-8 -*-
"""
当前持仓分析 vs 目标组合 (9资产+VolTarget7%)
==========================================================
持仓(2026-08):
  017730 嘉实全球产业升级(QDII)A  3823  (周投1000)
  007520 富安达富利纯债A          12938  (周投100)
  014668 银华专精特新量化股票A     15586  (周投500)
  003095 中欧医疗健康混合A        17388  (周投1000)
  013308 易方达恒生科技联接A       29683  (周投500)
  012922 易方达全球成长精选C       26059  (周投50)
"""
import os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vol_target as vt

NAV_DIR = vt.NAV_DIR
SQRT_252 = np.sqrt(252.0)

HOLDINGS = {
    "017730": {"name": "嘉实全球产业升级", "amount": 3823,  "wk": 1000, "cat": "全球科技"},
    "007520": {"name": "富安达富利纯债A",   "amount": 12938, "wk": 100,  "cat": "纯债"},
    "014668": {"name": "银华专精特新量化",   "amount": 15586, "wk": 500,  "cat": "A股量化"},
    "003095": {"name": "中欧医疗健康A",      "amount": 17388, "wk": 1000, "cat": "A股医疗"},
    "013308": {"name": "易方达恒生科技A",    "amount": 29683, "wk": 500,  "cat": "港股科技"},
    "012922": {"name": "易方达全球成长C",    "amount": 26059, "wk": 50,   "cat": "全球股票"},
}

def load(code):
    if code in vt._AC:
        s = vt._AC[code]
    else:
        s = vt.acc_nav(code)
    if s is None:
        return None
    return s

def stats_of(s, start="2021-01-01"):
    s = s[(s.index >= start) & (s.index <= pd.Timestamp("2026-08-06"))]
    if len(s) < 60:
        return None
    r = s.pct_change().dropna()
    years = (s.index[-1] - s.index[0]).days / 365.0
    ann = (s.iloc[-1] / s.iloc[0]) ** (1 / years) - 1
    mdd = float((s / s.cummax() - 1).min())
    vol = r.std() * SQRT_252
    sh = r.mean() / (r.std() + 1e-9) * SQRT_252
    return {"年化": ann, "回撤": mdd, "波动": vol, "夏普": sh, "起点": s.index[0].strftime("%Y-%m")}

def main():
    total = sum(h["amount"] for h in HOLDINGS.values())
    print("=" * 100)
    print(f"当前持仓分析 (总市值 {total:,}元, 周定投 {sum(h['wk'] for h in HOLDINGS.values())}元/周)")
    print("=" * 100)
    print(f"{'代码':<8}{'名称':<16}{'市值':>9} {'占比':>7} {'周投':>6} | {'年化':>7} {'回撤':>8} {'波动':>7} {'夏普':>6}  数据起点")
    print("-" * 100)
    navs = {}
    for code, h in HOLDINGS.items():
        s = load(code)
        st = stats_of(s) if s is not None else None
        pct = h["amount"] / total
        if st:
            print(f"{code:<8}{h['name']:<16}{h['amount']:>8,} {pct:>6.1%} {h['wk']:>5,} | {st['年化']:>6.1%} {st['回撤']:>7.1%} {st['波动']:>6.1%} {st['夏普']:>5.2f}  {st['起点']}")
        else:
            print(f"{code:<8}{h['name']:<16}{h['amount']:>8,} {pct:>6.1%} {h['wk']:>5,} | {'数据不足':>30s}")
        navs[code] = s

    # 当前组合 (按市值权重) 与 目标组合 对比 (2021-2026, 静态权重+动态可用归一化)
    cur_w = {c: h["amount"] / total for c, h in HOLDINGS.items()}
    tgt_w = {n: w for n, (_, w) in vt.ASSETS.items()}
    navs_tgt = vt.load_navs()   # 目标组合9资产净值

    def combo_stats(wmap, code_of, navs_dict, label):
        # 构建组合日收益: 早期缺失资产权重按可用性动态归一化
        cats = list(wmap.keys())
        real = [c for c in cats if code_of(c) != "000198"]
        idx_all = pd.DatetimeIndex(sorted(set().union(*[navs_dict[code_of(c)].index for c in real])))
        idx_all = idx_all[(idx_all >= pd.Timestamp("2021-01-01")) & (idx_all <= pd.Timestamp("2026-08-06"))]
        nav_df = pd.DataFrame({c: navs_dict[code_of(c)].reindex(idx_all).ffill() for c in cats})
        avail = nav_df.notna().values
        keep = avail.any(axis=1)
        nav_ff = nav_df.ffill()
        w_arr = np.array([wmap[c] for c in cats])
        w_dyn = (w_arr[None, :] * avail[keep]).astype(float)
        w_dyn = w_dyn / w_dyn.sum(axis=1, keepdims=True)
        rets = (nav_ff.iloc[keep].pct_change().fillna(0).values * w_dyn).sum(axis=1)
        port = pd.Series((1 + rets).cumprod(), index=nav_df.index[keep])
        years = (port.index[-1] - port.index[0]).days / 365.0
        ann = port.iloc[-1] ** (1 / years) - 1
        mdd = float((port / port.cummax() - 1).min())
        r = port.pct_change().dropna()
        vol = r.std() * SQRT_252
        sh = r.mean() / (r.std() + 1e-9) * SQRT_252
        print(f"{label:28s} | 年化{ann:>6.1%} 回撤{mdd:>7.1%} 波动{vol:>6.1%} 夏普{sh:>5.2f}")

    print("\n组合对比 (2021-01 ~ 2026-08, 市值权重, 无VolTarget):")
    combo_stats(cur_w, lambda c: c, navs, "当前持仓")
    # 目标组合: 键改为代码
    tgt_w_code = {vt.ASSETS[n][0]: w for n, w in tgt_w.items()}
    navs_tgt_code = {vt.ASSETS[n][0]: s for n, s in navs_tgt.items()}
    combo_stats(tgt_w_code, lambda c: c, navs_tgt_code, "目标组合(9资产)")

    # 目标组合 + VolTarget7% (参考)
    navs_tgt = vt.load_navs()
    eq_t, _ = vt.run_backtest(navs_tgt, tgt_w, tgt_vol=0.07, floor_w=0.5,
                              lump=1_000_000, dca=0, start="2021-01-01", end="2026-08-06")
    m_t = vt.calc_metrics(eq_t, 1_000_000)
    print(f"{'目标组合+VolTarget7%':28s} | 年化{m_t['年化']:>6.1%} 回撤{m_t['回撤']:>7.1%} 波动{m_t['波动']:>6.1%} 夏普{m_t['夏普']:>5.2f}")

if __name__ == "__main__":
    main()

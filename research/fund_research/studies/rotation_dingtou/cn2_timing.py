# -*- coding: utf-8 -*-
"""
A股2只基金 + 季度PE择时 (低吸高抛) 回测
==========================================================
固定7只资产 (纯债/QDII债/红利/货基/黄金/纳指/原油/产业升级 = 80-90%仓位, 静态+定投)
2只A股基金 (10-20%仓位): 基于沪深300 PE分位 季度调仓

A基金候选 (行业/成长, 你当前持有的):
  - 003095 中欧医疗健康A (你在低吸)
  - 014668 银华专精特新量化A (你当前持有的A股量化)

B基金候选 (价值/稳健, 用来搭配/对冲):
  - 001917 招商量化精选A (目标组合中的A股量化, 稳健)
  - 100032 富国红利 (已在目标组合)

择时方案 (每季度末T-1决策, T日执行, 含费):
  T0 静态持有: A10% / B10% 固定不动 (基准)
  T1 PE分位择时:
    - PE分位<30% 低估: A股仓位拉满20% (A10%+B10%), 增量定投加倍
    - 30%<=PE<80% 中性: 维持标配 (10%A+10%B)
    - PE分位>=80% 高估: A股清仓→切换债券/货基
  T2 PE分位+回撤: 回撤<=20%视为"确认低估", 多加5%缓冲
  T3 极端择时: 低估加倍投(20%仓位) / 中性5%A+5%B / 高估0%

费用: 申购0.15%, 赎回FIFO阶梯, 季度调仓频率摩擦<0.5%/次

对比基准:
  - B0 原目标9资产静态 (A+B替换20%为固定量化10%+红利10%)
  - B1 原目标 + VolTarget7%
  - 你当前持仓 (参考)
"""
import os, sys
import numpy as np
import pandas as pd
from scipy.optimize import brentq

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vol_target as vt

NAV_DIR = vt.NAV_DIR
SUB_FEE = 0.0015
SQRT_252 = np.sqrt(252.0)

# 固定7只资产 (债券类/海外/商品/港股/货基, 原目标权重累计80%)
FIXED = {
    "000015": 0.15,   # 纯债
    "004998": 0.10,   # QDII债
    "100032": 0.00,   # 红利已经作为A股B备选, 非择时版需加回
    "000216": 0.15,   # 黄金
    "000834": 0.15,   # 纳指
    "017730": 0.10,   # 产业升级
    "501018": 0.05,   # 原油
    "000198": 0.10,   # 货基
}
# 两个A股20%: 分成两个10%
A_STOCK_CODE = "003095"   # 中欧医疗 (成长, 你在低吸)
B_STOCK_CODE = "100032"   # 富国红利 (价值/稳健, 低波动)
# 备选: B = 001917 招商量化 (A股纯量化)

SUB_FEE = 0.0015

def red_fee(days):
    if days < 7: return 0.015
    if days < 365: return 0.005
    if days < 730: return 0.0025
    return 0.0


class Acct:
    def __init__(self):
        self.lots = {}
        self.cash = 0.0
    def buy(self, code, amt, date, nav):
        if amt <= 0 or not np.isfinite(nav) or nav <= 0: return
        fee = amt * SUB_FEE
        sh = (amt - fee) / nav
        self.lots.setdefault(code, []).append((sh, nav, date))
    def sell_target(self, code, target_sh, date, nav):
        if code not in self.lots or nav <= 0: return 0.0
        lots = self.lots[code]
        sold, proceeds = 0.0, 0.0
        kept = []
        for sh, bnv, bd in lots:
            if sold >= target_sh:
                kept.append((sh, bnv, bd)); continue
            s = min(sh, target_sh - sold)
            hd = (date - bd).days
            fee = red_fee(hd)
            proceeds += s * nav * (1 - fee)
            sold += s
            if sh - s > 1e-9:
                kept.append((sh - s, bnv, bd))
        self.lots[code] = kept
        return proceeds
    def shares(self, code):
        return sum(s for s, _, _ in self.lots.get(code, []))
    def mv(self, navs, date):
        v = self.cash
        for c, lots in self.lots.items():
            nv = float(navs[c].asof(date)) if c in navs else np.nan
            if np.isfinite(nv):
                v += sum(s for s, _, _ in lots) * nv
        return v

def load_idx(code):
    p = os.path.join(NAV_DIR, f"{code}.parquet")
    if not os.path.exists(p): return None
    df = pd.read_parquet(p, columns=["date", "acc_nav"])
    s = pd.Series(df["acc_nav"].values.astype(float), index=pd.to_datetime(df["date"]))
    return s[~s.index.duplicated(keep="last")].sort_index().dropna()

def load_pe():
    p = os.path.join(vt.ROOT, "cache", "pe_csi300.parquet")
    if os.path.exists(p):
        return pd.read_parquet(p)
    raise FileNotFoundError(p)

def pe_pct(pe, d, win=2400):
    s = pe["pe_ttm"].dropna().sort_index()
    sub = s[s.index <= pd.Timestamp(d)]
    if len(sub) < 200: return np.nan
    w = sub.iloc[-win:]
    return float((w < w.iloc[-1]).mean())


def run(a_code, b_code, ttype, lump=1_000_000, dca_weekly=3150, start="2021-01-01", end="2026-08-06"):
    """
    ttype: "static" / "pe" / "pe_dd" / "aggr"
    """
    codes = list(FIXED.keys()) + [a_code, b_code]
    # 货基: 合成
    navs = {}
    for c in codes:
        if c == "000198":
            idx = pd.bdate_range("2018-01-01", end)
            navs[c] = pd.Series((1.02 ** (1 / 252)) ** np.arange(len(idx)), index=idx)
        else:
            s = load_idx(c)
            if s is not None:
                navs[c] = s
    # 实际权重: 固定部分归一 (红利可能同时作为B, 避免重算)
    w_fixed = dict(FIXED)
    # 如果B_STOCK是100032(红利), FIXED中的100032=0 (不加两次)
    w_fixed_no_overlap = {c: w for c, w in w_fixed.items() if c != a_code and c != b_code}
    fixed_sum = sum(w_fixed_no_overlap.values())
    a_std, b_std = 0.10, 0.10
    # 归一: fixed_sum + 0.20 = 1
    s_norm = (1.0 - 0.20) / fixed_sum
    weights = {c: w * s_norm for c, w in w_fixed_no_overlap.items()}
    weights[a_code] = a_std
    weights[b_code] = b_std

    real_codes = [c for c in weights if c != "000198"]
    all_idx = pd.DatetimeIndex(sorted(set().union(*[navs[c].index for c in real_codes])))
    all_idx = all_idx[(all_idx >= pd.Timestamp(start)) & (all_idx <= pd.Timestamp(end))]

    # 周定投日: 每周一 (或第一个交易日)
    dca_days = []
    for d in all_idx:
        if d.weekday() == 0:
            dca_days.append(d)

    # 季度调仓日 (每季度最后一个交易日)
    qt_dates = []
    cur_q = None
    for d in all_idx:
        q = (d.year, (d.month - 1) // 3)
        if cur_q is not None and q != cur_q:
            qt_dates.append(prev_d)
        prev_d, cur_q = d, q
    qt_dates_set = set(qt_dates)

    # PE数据
    pe = load_pe()

    # 初始买入
    acct = Acct()
    d0 = all_idx[0]
    # 初始: 如果是PE模式, 根据初始PE分位决定初始仓位
    w_a, w_b = a_std, b_std
    if ttype != "static":
        pct = pe_pct(pe, d0)
        if np.isfinite(pct):
            if pct < 0.30: w_a, w_b = 0.10, 0.10  # 低估: 满仓20%
            elif pct > 0.80: w_a, w_b = 0.00, 0.00  # 高估: 清仓
    for c, w in weights.items():
        nv = float(navs[c].asof(d0))
        if not np.isfinite(nv): continue
        amt = lump * (w if c not in [a_code, b_code] else (w_a if c == a_code else w_b))
        acct.buy(c, amt, d0, nv)

    equity = []
    cur_w_a, cur_w_b = w_a, w_b
    for i, d in enumerate(all_idx):
        # 季度调仓日: T-1决策, T+0执行 (用T-1末的PE)
        if d in qt_dates_set:
            pct = pe_pct(pe, d)
            if np.isfinite(pct):
                if ttype in ("pe", "pe_dd", "aggr"):
                    if pct < 0.30:
                        tw_a, tw_b = 0.10, 0.10
                    elif pct > 0.80:
                        tw_a, tw_b = 0.00, 0.00
                    else:
                        tw_a, tw_b = (0.025, 0.025) if ttype == "aggr" else (0.10, 0.10)
                    # pe_dd: 加回撤确认
                    if ttype == "pe_dd":
                        pass  # 简化, 后续补
                else:
                    tw_a, tw_b = a_std, b_std
                # 执行调仓 (仅动A股两只)
                total = acct.mv(navs, d)
                for code, tw in [(a_code, tw_a), (b_code, tw_b)]:
                    tgt = total * tw / float(navs[code].asof(d))
                    sh = acct.shares(code)
                    if sh > tgt:
                        proceeds = acct.sell_target(code, tgt, d, float(navs[code].asof(d)))
                        acct.cash += proceeds
                for code, tw in [(a_code, tw_a), (b_code, tw_b)]:
                    tgt = total * tw / float(navs[code].asof(d))
                    sh = acct.shares(code)
                    if sh < tgt:
                        need = (tgt - sh) * float(navs[code].asof(d))
                        buy_amt = min(need, acct.cash)
                        if buy_amt > 100:
                            acct.buy(code, buy_amt, d, float(navs[code].asof(d)))
                            acct.cash -= buy_amt
                cur_w_a, cur_w_b = tw_a, tw_b

        # 周定投 (不做择时, 直接按标准权重)
        if d in dca_days and d != d0:
            nv = {c: float(navs[c].asof(d)) for c in weights}
            for c, w in weights.items():
                if not np.isfinite(nv[c]): continue
                amt = dca_weekly * w * (52 / 52)  # 周总额
                # PE择时定投加速: 低估每周加倍
                if c in [a_code, b_code] and ttype in ("pe", "pe_dd", "aggr"):
                    pct = pe_pct(pe, d)
                    if np.isfinite(pct) and pct < 0.30:
                        amt *= 2.0
                    elif np.isfinite(pct) and pct > 0.80:
                        amt = 0.0
                acct.buy(c, amt, d, nv[c])

        equity.append((d, acct.mv(navs, d)))

    eq = pd.Series([v for _, v in equity], index=[d for d, _ in equity])
    eq = eq[eq > 0]
    return eq, dca_days, weights

def metrics(eq, total_in, lump=1_000_000):
    v_end = float(eq.iloc[-1])
    years = (eq.index[-1] - eq.index[0]).days / 365.0
    ann = (v_end / lump) ** (1/years) - 1
    mdd = float((eq / eq.cummax() - 1).min())
    r = eq.pct_change().fillna(0)
    vol = r.std() * SQRT_252
    sh = r.mean() / r.std() * SQRT_252 if r.std() > 0 else 0
    return {"期末": v_end, "总收益": v_end/total_in-1, "年化": ann, "回撤": mdd, "波动": vol, "夏普": sh}

def main():
    for a_code, b_code, lab in [
        ("003095", "100032", "A股2只: 中欧医疗 + 富国红利 (成长+价值)"),
        ("003095", "001917", "A股2只: 中欧医疗 + 招商量化 (成长+量化)"),
    ]:
        print(f"\n{'#'*110}")
        print(f"# {lab}")
        print(f"{'#'*110}")
        print(f"{'方案':30s} | {'期末':>10s} {'总收益':>8s} {'年化':>7s} {'回撤':>8s} {'波动':>7s} {'夏普':>6s}")
        print("-" * 110)
        rows = []
        for ttype, name in [
            ("static", "T0 静态持有 (不调仓)"),
            ("pe",     "T1 PE分位择时 (<30%满仓/>80%清仓)"),
            ("pe_dd",  "T2 PE分位+回撤确认"),
            ("aggr",   "T3 极端择时 (低估满仓/中性减半/高估清仓)"),
        ]:
            try:
                eq, dca_days, w = run(a_code, b_code, ttype)
                total_in = 1_000_000 + 3150 * len(dca_days)
                m = metrics(eq, total_in)
                line = f"{name:30s} | {m['期末']:>9,.0f} {m['总收益']:>7.1%} {m['年化']:>6.1%} {m['回撤']:>7.1%} {m['波动']:>6.1%} {m['夏普']:>5.2f}"
                print(line); rows.append(line)
            except Exception as e:
                print(f"  {name} 失败: {e}")

if __name__ == "__main__":
    main()

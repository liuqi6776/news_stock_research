# -*- coding: utf-8 -*-
"""
优化组合 vs 等权6资产 回测对比 (2021-01 ~ 2026-08)
====================================================================
策略:
  1. 等权6资产  (纯债16.7% 黄金16.7% 纳指16.7% 沪深300 16.7% QDII债16.7% 原油16.7%) 静态持有
  2. 优化权重    (纯债20% QDII债10% 红利10% 沪深300 10% 黄金15% 纳指20% 原油5% 货币10%) 静态持有
  3. 优化+择时   (同2, 但沪深300按PE分位季度调仓: <30%升20% / >80%降5% / 中间10%) 季度调仓计费

费用: 申购0.15% / 赎回费按持有期阶梯 (FIFO批次记账)
场景: A. 一次性100万买入持有   B. 一次性100万 + 每月1万定投
用法: C:/Users/liuqi/anaconda3/python.exe studies/rotation_dingtou/optimize_compare.py
"""
import os, sys
import numpy as np
import pandas as pd
from scipy.optimize import brentq

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
NAV_DIR = r"D:\iquant_data\data_v2\fund2\nav"
PE_CACHE = os.path.join(ROOT, "cache", "pe_csi300.parquet")
START, END = "2021-01-01", "2026-08-06"
SUB_FEE = 0.0015  # 申购费

ASSETS = {
    "纯债": "000015", "黄金": "000216", "纳指": "000834",
    "沪深300": "050002", "QDII债": "004998", "原油": "501018",
    "红利": "100032", "货币": "000198", "全球科技": "001668",
    "量化": "001917",
}
EQ6 = {c: 1/6 for c in ["纯债", "黄金", "纳指", "沪深300", "QDII债", "原油"]}
OPT = {"纯债": 0.20, "QDII债": 0.10, "红利": 0.10, "沪深300": 0.10,
       "黄金": 0.15, "纳指": 0.20, "原油": 0.05, "货币": 0.10}
# 去掉原油+QDII债 → 换成全球科技 (001668 汇添富全球移动互联)
OPT_TECH = {"纯债": 0.20, "红利": 0.10, "沪深300": 0.10,
            "黄金": 0.15, "纳指": 0.20, "全球科技": 0.15, "货币": 0.10}
# 纳指 20%→25%, 纯债 20%→17.5%, 货币 10%→7.5%
OPT_NDX = {"纯债": 0.175, "QDII债": 0.10, "红利": 0.10, "沪深300": 0.10,
           "黄金": 0.15, "纳指": 0.25, "原油": 0.05, "货币": 0.075}
# 纯债 17.5%→15%, 沪深300 → A股量化(招商量化精选001917), 腾出2.5%给货币
OPT_QUANT = {"纯债": 0.15, "QDII债": 0.10, "红利": 0.10, "量化": 0.10,
             "黄金": 0.15, "纳指": 0.25, "原油": 0.05, "货币": 0.10}
NON_CN = {c: w for c, w in OPT.items() if c != "沪深300"}
NON_CN_SUM = sum(NON_CN.values())  # 0.9

_NAV = {}
def acc_nav(code):
    if code in _NAV: return _NAV[code]
    if code == "000198":
        # 货币基金净值数据异常(接口口径错误), 用年化2%合成序列
        idx = pd.date_range("2020-06-01", pd.Timestamp(END), freq="B")
        s = pd.Series((1.02 ** (1 / 252)) ** np.arange(len(idx)), index=idx)
        _NAV[code] = s
        return s
    df = pd.read_parquet(os.path.join(NAV_DIR, f"{code}.parquet"), columns=["date", "acc_nav"])
    s = pd.Series(df["acc_nav"].to_numpy(dtype=float), index=pd.to_datetime(df["date"]))
    s = s[~s.index.duplicated(keep="last")].sort_index().dropna()
    s = s[(s.index >= pd.Timestamp("2020-06-01")) & (s.index <= pd.Timestamp(END))]
    _NAV[code] = s
    return s

def load_navs():
    return {c: acc_nav(code) for c, code in ASSETS.items()}

def red_fee(days):
    if days < 7: return 0.015
    if days < 365: return 0.005
    if days < 730: return 0.0025
    return 0.0

class Acct:
    """FIFO 批次账户, cash 仅用于再平衡资金"""
    def __init__(self):
        self.lots = {}
        self.cash = 0.0
    def invest(self, code, amt, date, nav):
        self.lots.setdefault(code, []).append((amt * (1 - SUB_FEE) / nav, date))
    def _cur_shares(self, code):
        return sum(sh for sh, _ in self.lots.get(code, []))
    def sell_to(self, code, nav, date, target_shares):
        cur = self._cur_shares(code)
        sell = max(0.0, cur - target_shares)
        if sell <= 0: return 0.0
        remaining, proceeds, kept = sell, 0.0, []
        for sh, bd in self.lots.get(code, []):
            if remaining <= 0:
                kept.append((sh, bd)); continue
            if sh <= remaining:
                proceeds += sh * nav * (1 - red_fee((date - bd).days))
                remaining -= sh
            else:
                proceeds += remaining * nav * (1 - red_fee((date - bd).days))
                kept.append((sh - remaining, bd))
                remaining = 0.0
        self.lots[code] = kept
        return proceeds
    def mv(self, navs, date):
        total = self.cash
        for c, lots in self.lots.items():
            nv = navs[c].asof(date)
            if np.isfinite(nv):
                total += sum(sh for sh, _ in lots) * nv
        return total

def pe_data():
    if os.path.exists(PE_CACHE):
        return pd.read_parquet(PE_CACHE)
    raise FileNotFoundError(PE_CACHE)

def pe_pct(pe, d, win=2400):
    s = pe["pe_ttm"].dropna()
    sub = s[s.index <= pd.Timestamp(d)]
    if len(sub) < 200: return np.nan
    w = sub.iloc[-win:]
    return float((w < w.iloc[-1]).mean())

def make_weights(pe):
    """返回 weight_fn(date) -> dict; static 参数决定是否用择时"""
    def static(d):
        return dict(EQ6)
    def opt_static(d):
        return dict(OPT)
    def opt_tech(d):
        return dict(OPT_TECH)
    def opt_ndx(d):
        return dict(OPT_NDX)
    def opt_quant(d):
        return dict(OPT_QUANT)
    def opt_timed(d):
        pct = pe_pct(pe, d)
        w_cn = 0.20 if pct < 0.30 else (0.05 if pct > 0.80 else 0.10)
        w = {c: w0 * (1 - w_cn) / NON_CN_SUM for c, w0 in NON_CN.items()}
        w["沪深300"] = w_cn
        return w
    return {"等权6资产": static, "优化权重": opt_static,
            "优化+全球科技": opt_tech, "优化+纳指25%": opt_ndx,
            "优化+量化": opt_quant,
            "优化+择时": opt_timed}

def run(navs, weight_fn, td, quarter_dates, lump, dca, rebal):
    """rebal: None / 'quarterly'"""
    acct = Acct()
    # 一次性买入
    d0 = td[0]
    nav0 = {c: navs[c].asof(d0) for c in ASSETS}
    for c, w in weight_fn(d0).items():
        acct.invest(c, lump * w, d0, nav0[c])
    # 每月定投 (从第二个月起, 避免与一次性同日重叠)
    months = pd.date_range(START, END, freq="MS")[1:]
    dca_days = []
    for m in months:
        k = int(np.searchsorted(td, pd.Timestamp(m).to_datetime64()))
        if k < len(td): dca_days.append(td[k])
    dca_days = sorted(set(dca_days))
    for d in dca_days:
        nv = {c: navs[c].asof(d) for c in ASSETS}
        wd = weight_fn(d)
        for c, w in wd.items():
            acct.invest(c, dca * w, d, nv[c])
    # 季度再平衡
    if rebal == "quarterly":
        for qd in quarter_dates:
            if qd <= d0: continue
            nv = {c: navs[c].asof(qd) for c in ASSETS}
            total = acct.mv(navs, qd)
            wq = weight_fn(qd)
            proceeds = 0.0
            for c in wq:
                tgt = total * wq[c] / nv[c] if np.isfinite(nv[c]) and nv[c] > 0 else 0.0
                proceeds += acct.sell_to(c, nv[c], qd, tgt)
            acct.cash += proceeds
            for c in wq:
                cur = sum(sh for sh, _ in acct.lots.get(c, []))
                tgt = total * wq[c] / nv[c] if np.isfinite(nv[c]) and nv[c] > 0 else 0.0
                need = max(0.0, (tgt - cur) * nv[c])
                if need > 0:
                    acct.invest(c, need, qd, nv[c])
                    acct.cash -= need
            # 现金尾差并入货币
    # 每日市值
    mv = pd.Series([acct.mv(navs, d) for d in td], index=td)
    mv = mv[mv > 0]
    total_in = lump + dca * len(dca_days)
    return mv, total_in, dca_days

def xirr(cfs):
    if len(cfs) < 2: return np.nan
    try:
        d0 = cfs[0][0]
        days = np.array([(d - d0).days for d, _ in cfs], dtype=float)
        flows = np.array([v for _, v in cfs], dtype=float)
        return brentq(lambda r: float(np.sum(flows / (1 + r) ** (days / 365))), -0.5, 5.0)
    except Exception:
        return np.nan

def stats(mv, total_in, cashflows):
    v_end = float(mv.iloc[-1])
    ret = v_end / total_in - 1
    mdd = float((mv / mv.cummax() - 1).min())
    r = mv.pct_change().fillna(0)
    vol = r.std() * np.sqrt(252)
    sh = (r.mean() * 252) / vol if vol > 0 else 0
    x = xirr(cashflows)
    days = (mv.index[-1] - mv.index[0]).days
    cagr = (v_end / total_in) ** (365.0 / days) - 1 if total_in > 0 else np.nan
    return {"期末值": v_end, "总投入": total_in, "总收益": ret, "CAGR": cagr,
            "XIRR": x, "回撤": mdd, "波动": vol, "夏普": sh}

def main():
    print("=" * 110)
    print("优化组合 vs 等权6资产 (2021-01 ~ 2026-08)")
    print("=" * 110)
    navs = load_navs()
    pe = pe_data()
    # 交易日索引
    all_idx = []
    for s in navs.values():
        all_idx.append(s.index)
    td = pd.DatetimeIndex(sorted(set().union(*all_idx)))
    td = td[(td >= pd.Timestamp(START)) & (td <= pd.Timestamp(END))]
    quarter_dates = [td[k] for k in [int(np.searchsorted(td, pd.Timestamp(q).to_datetime64()))
                     for q in pd.date_range(START, END, freq="Q")] if k < len(td)]
    wf_map = make_weights(pe)

    for scene, lump, dca in [("A. 一次性100万", 1_000_000, 0), ("B. 一次性100万+月定投1万", 1_000_000, 10_000)]:
        print(f"\n{'#' * 110}")
        print(f"# {scene}")
        print(f"{'#' * 110}")
        hdr = f"{'策略':14s} | {'期末市值':>12s} {'总收益':>8s} {'CAGR':>7s} {'XIRR':>7s} | {'回撤':>7s} {'波动':>7s} {'夏普':>5s}"
        print(hdr)
        print("-" * len(hdr))
        for name, wf in wf_map.items():
            rebal = "quarterly" if name == "优化+择时" else None
            mv, total_in, dca_days = run(navs, wf, td, quarter_dates, lump, dca, rebal)
            cfs = [(mv.index[0], -lump)] + [(d, -dca) for d in dca_days] + [(mv.index[-1], float(mv.iloc[-1]))]
            st = stats(mv, total_in, cfs)
            print(f"{name:14s} | {st['期末值']:>11,.0f}元 {st['总收益']:>7.1%} {st['CAGR']:>6.1%} {st['XIRR']:>6.1%} | "
                  f"{st['回撤']:>6.1%} {st['波动']:>6.1%} {st['夏普']:>5.2f}")

    # 优化+择时 的季度权重明细
    print("\n" + "=" * 110)
    print("优化+择时: 沪深300 权重随 PE 分位的季度调整")
    print("=" * 110)
    pe_s = pe["pe_ttm"].dropna()
    print(f"{'季度末':>12s} {'PE':>7s} {'分位':>7s} {'w_沪深300':>9s}")
    for qd in quarter_dates:
        pct = pe_pct(pe, qd)
        w_cn = 0.20 if pct < 0.30 else (0.05 if pct > 0.80 else 0.10)
        cur = float(pe_s.asof(qd)) if len(pe_s[pe_s.index <= qd]) else np.nan
        print(f"{qd.date()} {cur:>7.2f} {pct:>6.1%} {w_cn:>8.0%}")

if __name__ == "__main__":
    main()

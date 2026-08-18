# -*- coding: utf-8 -*-
"""
xalpha 定投回测 —— 本地 fund2 场外净值 + 申购/赎回费

用 xalpha 的回测框架 (backtest.BTE 子类) 做场外基金定投回测:
  - 无脑定投 (Scheduled): 固定频率固定金额买入
  - 价值平均定投 (AverageScheduled): 按目标市值差额补投/赎回 (xalpha 内置策略)
  - 一次性买入 (benchmark): 期初全仓对比

关键实现:
  1. LocalFundInfo: 子类化 xalpha.info.fundinfo, 从本地 parquet 注入净值,
     避免每次回测走天天基金网络接口 (xalpha 默认行为)。
  2. 申购费: self.rate (%, 默认 0.15 = 0.15% 折扣费率)。
  3. 赎回费: xalpha 默认赎回费为 0, 本脚本重写 _shuhui_by_share,
     按持有期阶梯收费 (FIFO): <7天 1.5% / 7天-1年 0.5% / 1-2年 0.25% / >2年 0。
  4. 业绩指标: XIRR(内部收益率, 现金流 = 每次定投 -金额 + 期末市值) / 总投入 / 期末市值。

用法:
  C:/Users/liuqi/anaconda3/python.exe studies/dingtou_backtest/run_dingtou.py \
    --codes 110022,163406,161725 --start 2019-01-01 --end 2026-08-06 \
    --amount 1000 --freq MS --strategy scheduled
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # fund_research
sys.path.insert(0, ROOT)

from xalpha.cons import convert_date
from xalpha.backtest import BTE, Scheduled, AverageScheduled
from xalpha.info import fundinfo

NAV_DIR = r"D:\iquant_data\data_v2\fund2\nav"

# 赎回费阶梯: (持有天数下限, 费率)。默认股票/混合基金常见水平, 映射到
# xalpha 的 self.segment / self.feeinfo (见 LocalFundInfo._basic_init)。
REDEEM_FEEINFO = ["x", "1.50%", "x", "0.50%", "x", "0.25%", "x", "0.00%"]
REDEEM_SEGMENT = [[0, 7], [7, 365], [365, 730], [730]]


class LocalFundInfo(fundinfo):
    """从本地 fund2 parquet 加载净值, 支持申购费与按持有期阶梯赎回费"""

    def __init__(self, code, sg_fee=0.15, **kws):
        self._sg_fee = sg_fee  # 申购费率, % 单位 (0.15 = 0.15%)
        super().__init__(code, priceonly=True, **kws)

    def _basic_init(self):
        path = os.path.join(NAV_DIR, f"{self.code}.parquet")
        if not os.path.exists(path):
            raise FileNotFoundError(f"本地无净值: {path}")
        df = pd.read_parquet(path, columns=["date", "unit_nav"])
        self.price = pd.DataFrame(
            {
                "date": pd.to_datetime(df["date"]),
                "netvalue": df["unit_nav"].astype(float),
                "comment": 0.0,
            }
        ).sort_values("date").reset_index(drop=True)
        self.rate = self._sg_fee  # % 单位
        self.name = self.code
        # 赎回费: xalpha 默认 0, 这里注入按持有期阶梯 (fundinfo.shuhui 读取)
        self.feeinfo = list(REDEEM_FEEINFO)
        self.segment = [list(s) for s in REDEEM_SEGMENT]


class LocalMixin:
    """让 BTE 子类使用本地数据 fundinfo (替换默认的网络 fundinfo)"""
    sg_fee = 0.15

    def get_info(self, code):
        if code in self.infos:
            return self.infos[code]
        c = code[1:] if code.startswith("F") else code
        return LocalFundInfo(c, sg_fee=self.sg_fee)


class ScheduledL(LocalMixin, Scheduled):
    """无脑定投 (本地数据)"""


class AverageScheduledL(LocalMixin, AverageScheduled):
    """价值平均定投 (本地数据)"""


def xirr(cashflows):
    """现金流 [(date, amount), ...] 的内部收益率 (年化)"""
    from scipy.optimize import brentq

    base = cashflows[0][0]
    days = [(d - base).days for d, _ in cashflows]
    amts = [a for _, a in cashflows]

    def npv(r):
        return sum(a / (1 + r) ** (dy / 365.0) for a, dy in zip(amts, days))

    try:
        return brentq(npv, -0.9999, 100.0)
    except (ValueError, RuntimeError):
        return np.nan


def load_nav_first_date(code):
    path = os.path.join(NAV_DIR, f"{code}.parquet")
    if not os.path.exists(path):
        return None
    df = pd.read_parquet(path, columns=["date"])
    return pd.to_datetime(df["date"]).min()


def run_dingtou(code, name, start, end, amount, freq, strategy, sg_fee):
    """单只基金单策略定投回测, 返回指标 dict"""
    first = load_nav_first_date(code)
    if first is None:
        return None
    start_eff = max(pd.Timestamp(start), first)
    end_eff = min(pd.Timestamp(end), pd.Timestamp("2026-08-06"))
    if start_eff >= end_eff:
        return None

    cls = AverageScheduledL if strategy == "average" else ScheduledL
    bt = cls(
        start=start_eff.strftime("%Y-%m-%d"),
        end=end_eff.strftime("%Y-%m-%d"),
        code=f"F{code}",
        value=amount,
        date_range=pd.date_range(start_eff, end_eff, freq=freq),
        totmoney=1e9,
        verbose=False,
    )
    bt.sg_fee = sg_fee
    bt.backtest()

    # 期末总资产 = 基金现值 + 现金余额。
    # 现金余额 = 卖出所得累计 (买入的钱直接转份额, 不经过现金池; 卖出回款留在账户)。
    sys = bt.get_current_mulfix()
    sdf = sys.summary(end_eff.strftime("%Y-%m-%d"))
    fund_row = sdf[sdf["基金代码"] == code]
    if fund_row.empty:
        fund_row = sdf[sdf["基金名称"] == code]
    if fund_row.empty:
        return None
    tr = bt.trades[f"F{code}"]
    final_fund = float(fund_row.iloc[0]["基金现值"])
    cash_balance = sum(float(r.cash) for r in tr.cftable.itertuples() if r.cash > 0)
    final_value = final_fund + cash_balance

    # 现金流: 只有买入(负cash)是用户投入; 卖出回款已计入期末总资产
    cf = [(convert_date(r.date), float(r.cash)) for r in tr.cftable.itertuples() if r.cash < 0]
    total_invested = -sum(c for _, c in cf)
    cf.append((end_eff, final_value))
    irr = xirr(cf)

    n_periods = sum(1 for _, c in cf[:-1] if c < 0)
    return {
        "code": code,
        "name": name,
        "strategy": strategy,
        "start": start_eff.strftime("%Y-%m-%d"),
        "end": end_eff.strftime("%Y-%m-%d"),
        "periods": n_periods,
        "total_invested": round(total_invested, 2),
        "final_value": round(final_value, 2),
        "total_return_pct": round(final_value / total_invested * 100 - 100, 2),
        "xirr_pct": round(irr * 100, 2) if not np.isnan(irr) else np.nan,
    }


def run_buyhold(code, name, start, end, amount_total, sg_fee):
    """一次性买入 benchmark: 期初全仓持有到期"""
    first = load_nav_first_date(code)
    if first is None:
        return None
    start_eff = max(pd.Timestamp(start), first)
    end_eff = min(pd.Timestamp(end), pd.Timestamp("2026-08-06"))
    if start_eff >= end_eff:
        return None
    path = os.path.join(NAV_DIR, f"{code}.parquet")
    df = pd.read_parquet(path, columns=["date", "unit_nav"])
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()
    nav0 = df["unit_nav"].asof(start_eff)
    nav1 = df["unit_nav"].asof(end_eff)
    if np.isnan(nav0) or np.isnan(nav1) or nav0 <= 0:
        return None
    net0 = amount_total / (1 + sg_fee / 100.0)  # 扣申购费后的投入
    final = net0 / nav0 * nav1
    cf = [(start_eff, -amount_total), (end_eff, final)]
    irr = xirr(cf)
    return {
        "code": code,
        "name": name,
        "strategy": "一次性买入",
        "start": start_eff.strftime("%Y-%m-%d"),
        "end": end_eff.strftime("%Y-%m-%d"),
        "periods": 1,
        "total_invested": round(amount_total, 2),
        "final_value": round(final, 2),
        "total_return_pct": round(final / amount_total * 100 - 100, 2),
        "xirr_pct": round(irr * 100, 2) if not np.isnan(irr) else np.nan,
    }


def main():
    ap = argparse.ArgumentParser(description="xalpha 场外基金定投回测 (本地净值)")
    ap.add_argument("--codes", required=True, help="基金代码, 逗号分隔, 如 110022,163406,161725")
    ap.add_argument("--start", default="2019-01-01", help="回测开始日期")
    ap.add_argument("--end", default="2026-08-06", help="回测结束日期")
    ap.add_argument("--amount", type=float, default=1000, help="每次定投金额")
    ap.add_argument("--freq", default="MS", help="定投频率 (pandas freq): MS=月初, WOM-2FRI=每月第二个周五")
    ap.add_argument("--strategy", default="scheduled", choices=["scheduled", "average", "both"],
                    help="scheduled=无脑定投, average=价值平均, both=两者")
    ap.add_argument("--sg-fee", type=float, default=0.15, help="申购费率 % (默认 0.15)")
    args = ap.parse_args()

    codes = []
    for c in args.codes.split(","):
        c = c.strip()
        if c.isdigit():
            c = str(int(c)).zfill(6)  # PowerShell 会吃掉 005409 的前导零, 这里补回
        codes.append(c)
    basic = pd.read_parquet(os.path.join(os.path.dirname(NAV_DIR), "fund_basic_O.parquet"))
    name_map = dict(zip(basic["code"].astype(str), basic["name"]))

    strategies = ["scheduled", "average"] if args.strategy == "both" else [args.strategy]
    results = []
    for code in codes:
        name = name_map.get(code, code)
        bh_done = False
        for strat in strategies:
            r = run_dingtou(code, name, args.start, args.end, args.amount, args.freq, strat, args.sg_fee)
            if r:
                results.append(r)
                # 一次性买入 benchmark: 与定投相同总投入、相同起止日期
                if not bh_done:
                    bh = run_buyhold(code, name, r["start"], r["end"], r["total_invested"], args.sg_fee)
                    if bh:
                        results.append(bh)
                    bh_done = True

    if not results:
        print("无有效结果, 请检查 code 与日期范围")
        return

    df = pd.DataFrame(results)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 250)
    print("\n" + "=" * 100)
    print("xalpha 定投回测结果 (申购费 %.2f%%, 赎回费按持有期阶梯)" % args.sg_fee)
    print("=" * 100)
    print(df.to_string(index=False))
    print("=" * 100)

    # 按基金分组对比
    for code in codes:
        sub = df[df["code"] == code]
        if sub.empty:
            continue
        print(f"\n[{code} {name_map.get(code, code)}]")
        for _, r in sub.iterrows():
            print(f"  {r['strategy']:<10} 投入 {r['total_invested']:>10,.0f}  "
                  f"期末 {r['final_value']:>12,.0f}  总收益 {r['total_return_pct']:>8.2f}%  "
                  f"XIRR {r['xirr_pct']:>8.2f}%")


if __name__ == "__main__":
    main()

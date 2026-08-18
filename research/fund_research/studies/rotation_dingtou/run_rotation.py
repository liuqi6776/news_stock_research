# -*- coding: utf-8 -*-
"""
4433 季度轮换定投组合回测 (动态全市场重算 + 全仓季度再平衡)
================================================================

策略(主):
  1. 动态全市场重算: 每个季度末, 在全市场主动权益基金(5类)中, 用截至该时点的
     近1/2/3/5年 + 今年以来 + 近6月/3月收益排名, 重算 4433, 选出当季通过者。
  2. 当季通过者等权: 当季定投资金 + 再平衡资金在通过者中等权分配。
  3. 全仓季度再平衡: 每季度卖出掉出名单的持仓(按持有期计赎回费), 再把全部资产
     (含当季新增定投) 在名单内重新等权。

对照策略:
  A. 动态4433 + 全仓季度再平衡   (主策略)
  B. 动态4433 + 只轮新资金       (不卖旧仓, 无赎回费损耗, 显示"轮换+费用"成本)
  C. 静态206池(2026年4433结果) + 全仓季度再平衡  (展示幸存者偏差)
  D. 沪深300指数联接基金季度定投 (市场基准, 本地有数据则纳入)

费用模型:
  - 申购费 0.15% (折扣费率)
  - 赎回费按持有期: <7天 1.5% / 7天-1年 0.5% / 1-2年 0.25% / >2年 0
  - 场外申购/赎回 T+1 确认, 本脚本简化为按季度末净值成交

注意:
  - 净值 pct_chg 含分红再投资, 份额记账用 unit_nav, 现金分红派发未建模
  - fund_basic 为 2026 快照, 已清盘/合并基金不在池中(动态池仍有残余幸存者偏差, 但远小于静态池)

用法:
  C:/Users/liuqi/anaconda3/python.exe studies/rotation_dingtou/run_rotation.py \
    --start 2021-01-01 --end 2026-08-06 --amount 3000
"""
import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # fund_research
sys.path.insert(0, ROOT)

NAV_DIR = r"D:\iquant_data\data_v2\fund2\nav"
BASIC_PATH = r"D:\iquant_data\data_v2\fund2\fund_basic_O.parquet"
RESULTS_DIR = os.path.join(ROOT, "results")
CACHE_DIR = os.path.join(ROOT, "cache")
STATIC_CSV = os.path.join(RESULTS_DIR, "4433_result.csv")

# 主动权益类型 (与 run_4433.py 一致)
ACTIVE_TYPES = ["股票型", "混合型-偏股", "混合型-灵活", "混合型-平衡", "混合型-绝对收益"]
# 区间 -> 净值条数窗口 (约 240 交易日/年)
WINDOWS = {"ret_3m": 60, "ret_6m": 120, "ret_1y": 240, "ret_2y": 480, "ret_3y": 720, "ret_5y": 1200}
Q_COND = ["ret_1y", "ret_2y", "ret_3y", "ret_5y", "ret_ytd"]  # 前 25%
T_COND = ["ret_6m", "ret_3m"]                                # 前 33.33%
RANK_Q, RANK_T = 25.0, 33.333

# 指数基准候选(沪深300联接A)
INDEX_CANDIDATES = ["110020", "000051", "050002"]

# 申购费 %(折扣)
SG_FEE = 0.15

_nav_cache = {}


def fee_rate(days):
    """赎回费率: 按持有天数"""
    if days < 7:
        return 0.015
    if days < 365:
        return 0.005
    if days < 730:
        return 0.0025
    return 0.0


def nav_series(code):
    """unit_nav 序列 (datetime index, 去重+排序), 带缓存"""
    if code not in _nav_cache:
        df = pd.read_parquet(os.path.join(NAV_DIR, f"{code}.parquet"),
                             columns=["date", "unit_nav"])
        s = pd.Series(
            df["unit_nav"].to_numpy(dtype=float),
            index=pd.to_datetime(df["date"]),
        ).sort_index()
        s = s[~s.index.duplicated(keep="last")]
        _nav_cache[code] = s
    return _nav_cache[code]


def nav_at(code, date):
    """date 当日/此前最近的净值; 未成立返回 nan"""
    s = nav_series(code)
    v = s.asof(pd.Timestamp(date))
    return float(v) if np.isfinite(v) else np.nan


def load_basic():
    basic = pd.read_parquet(BASIC_PATH)
    return basic[basic["code"].notna()].copy()


# ---------------------------------------------------------------------------
# 季度 4433 选基
#   每只基金按自身净值序列独立计算截至各季度末的窗口收益 (与 run_4433 同口径,
#   "最后 n 个净值日" 而非并集日历的 n 行, 避免并集索引缺口导致滚动全 NaN)。
# ---------------------------------------------------------------------------
def compute_window_sums(codes, qdates, rebuild=False):
    """逐基金计算各季度末窗口对数收益。

    返回 (out, union_dates):
      out[name]        -> ndarray[n_quarters, n_codes], 与 codes 顺序对齐, 对数收益
      union_dates      -> DatetimeIndex (全体基金净值日并集, 供净值曲线取日期)
    """
    tag = f"{qdates[0].date()}_{qdates[-1].date()}_q{len(qdates)}"
    cache_path = os.path.join(CACHE_DIR, f"rotation_wsum_{tag}.npz")
    if not rebuild and os.path.exists(cache_path):
        z = np.load(cache_path, allow_pickle=True)
        saved = list(z["codes"])
        if saved == codes:
            out = {name: z[f"w_{name}"] for name in list(WINDOWS) + ["ret_ytd"]}
            union = pd.DatetimeIndex(z["union_dates"])
            print(f"从缓存加载窗口收益: {cache_path}")
            return out, union

    n_f, n_q = len(codes), len(qdates)
    out = {name: np.full((n_q, n_f), np.nan, dtype=np.float64)
           for name in list(WINDOWS) + ["ret_ytd"]}
    q_ts = [pd.Timestamp(d) for d in qdates]
    q_ns = [np.datetime64(d, "ns") for d in q_ts]
    q_year = [d.year for d in q_ts]
    year_start_ns = {y: np.datetime64(pd.Timestamp(f"{y}-01-01"), "ns") for y in set(q_year)}
    union_parts = []
    t0 = time.time()
    for j, code in enumerate(codes):
        path = os.path.join(NAV_DIR, f"{code}.parquet")
        if not os.path.exists(path):
            continue
        df = pd.read_parquet(path, columns=["date", "pct_chg"])
        df = df.drop_duplicates("date", keep="last").sort_values("date")
        dates = pd.to_datetime(df["date"]).to_numpy()  # datetime64[ns]
        pct = np.asarray(df["pct_chg"].to_numpy(dtype=float).clip(-50, 50))
        pct = np.where(np.isnan(pct), 0.0, pct)
        cum = np.cumsum(np.log1p(pct / 100.0))
        union_parts.append(dates)
        for i, (d_ns, yr) in enumerate(zip(q_ns, q_year)):
            p = int(np.searchsorted(dates, d_ns, side="right"))
            if p == 0:
                continue
            pos0 = int(np.searchsorted(dates, year_start_ns[yr], side="left"))
            out["ret_ytd"][i, j] = cum[p - 1] - (cum[pos0 - 1] if pos0 > 0 else 0.0)
            for name, w in WINDOWS.items():
                if p >= w:
                    s = cum[p - 1]
                    if p - 1 - w >= 0:
                        s -= cum[p - 1 - w]
                    out[name][i, j] = s
        if (j + 1) % 2000 == 0:
            print(f"  窗口收益 {j+1}/{n_f}, 用时 {time.time()-t0:.0f}s")
    union = pd.DatetimeIndex(np.unique(np.concatenate(union_parts)) if union_parts else [])
    os.makedirs(CACHE_DIR, exist_ok=True)
    np.savez_compressed(
        cache_path,
        **{f"w_{name}": out[name] for name in out},
        codes=np.array(codes, dtype=object),
        union_dates=union.to_numpy(),
    )
    print(f"窗口收益计算完成 ({n_f} 只 x {n_q} 季度), 用时 {time.time()-t0:.0f}s")
    return out, union


def select_4433(out, codes, basic_ft, i):
    """第 i 季度 (索引到 out 的第 i 行) 的 4433 通过基金 code 列表"""
    cols = list(WINDOWS) + ["ret_ytd"]
    f = pd.DataFrame({name: np.expm1(out[name][i]) for name in cols}, index=codes)
    f["fund_type"] = basic_ft.reindex(f.index)
    for c in Q_COND + T_COND:
        f[f"rk_{c}"] = f.groupby("fund_type")[c].rank(pct=True) * 100.0
    mask = f["ret_5y"].notna()
    for c in Q_COND:
        mask &= f[f"rk_{c}"] <= RANK_Q
    for c in T_COND:
        mask &= f[f"rk_{c}"] <= RANK_T
    return f.index[mask.values].tolist()


def xirr(cashflows):
    """现金流 [(date, amount), ...] 的年化内部收益率"""
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


# ---------------------------------------------------------------------------
# 组合记账 (份额 + 现金, 申购费/赎回费阶梯, 多批次 FIFO)
# ---------------------------------------------------------------------------
class Portfolio:
    def __init__(self, sg_fee=SG_FEE):
        self.sg_fee = sg_fee
        self.cash = 0.0
        self.lots = {}       # code -> [(shares, buy_date)]
        self.sold_value = 0.0  # 累计卖出金额(换手统计)

    def buy(self, code, amount, nav, date):
        if amount <= 0 or not np.isfinite(nav) or nav <= 0:
            return
        shares = amount / (1 + self.sg_fee / 100.0) / nav
        self.lots.setdefault(code, []).append((shares, date))
        self.cash -= amount

    def _sell_lots(self, code, nav, date, max_shares=None):
        lots = self.lots.get(code, [])
        need = max_shares if max_shares is not None else float("inf")
        gross = 0.0
        fee = 0.0
        keep = []
        for sh, bd in lots:
            if need <= 0:
                keep.append((sh, bd))
                continue
            take = min(sh, need)
            days = (date - bd).days
            g = take * nav
            gross += g
            fee += g * fee_rate(days)
            need -= take
            if sh - take > 1e-9:
                keep.append((sh - take, bd))
        if keep:
            self.lots[code] = keep
        else:
            self.lots.pop(code, None)
        self.cash += gross - fee
        self.sold_value += gross
        return gross, fee

    def sell_all(self, code, nav, date):
        return self._sell_lots(code, nav, date)

    def fund_value(self, code, nav):
        return sum(sh for sh, _ in self.lots.get(code, [])) * nav

    def snapshot(self):
        shares = {c: sum(sh for sh, _ in lots) for c, lots in self.lots.items()}
        return self.cash, shares


def rebalance_full(port, sel, date, contribution):
    """全仓季度再平衡: 卖出掉队 + 全部资产(含新增)在名单内等权"""
    port.cash += contribution
    for code in list(port.lots):
        if code not in sel:
            nav = nav_at(code, date)
            if np.isfinite(nav):
                port.sell_all(code, nav, date)
    sel = [c for c in sel if np.isfinite(nav_at(c, date))]
    if not sel:
        return
    P = port.cash + sum(port.fund_value(c, nav_at(c, date)) for c in sel)
    target = P / len(sel)
    for c in sel:
        nav = nav_at(c, date)
        cur = port.fund_value(c, nav)
        delta = target - cur
        if delta > 1e-6:
            port.buy(c, delta, nav, date)
        elif delta < -1e-6:
            port._sell_lots(c, nav, date, max_shares=-delta / nav)


def invest_new_money(port, sel, date, contribution):
    """只轮新资金: 当季新增资金在名单内等权买入, 旧仓不动"""
    port.cash += contribution
    sel = [c for c in sel if np.isfinite(nav_at(c, date))]
    if not sel:
        return
    each = contribution / len(sel)
    for c in sel:
        port.buy(c, each, nav_at(c, date), date)


# ---------------------------------------------------------------------------
# 净值曲线
# ---------------------------------------------------------------------------
def segment_equity(cash, shares, d_start, d_end, dates_idx):
    """区间 [d_start, d_end) 内每日组合价值"""
    seg = dates_idx[(dates_idx >= d_start) & (dates_idx < d_end)]
    if len(seg) == 0 or not shares:
        return pd.Series(float(cash), index=seg)
    df = pd.DataFrame({c: nav_series(c) for c in shares})
    # 注意: 不能用 reindex(method="ffill") —— pandas 2.0.3 对含非交易日的
    # 多列 DatetimeIndex 会静默漏填 (2023-01-02 等), 改用 reindex + ffill()
    # 再注意: 全局日期并集可能含个别基金的周末/节假日净值记录, 段首若为非交易日
    # (如除夕 2022-01-31), reindex+ffill 无法从段外填充 -> 段首估值塌为 0。
    # 修复: 先用 asof 为每列取段首前最近净值作为预填行, 保证每列从段首即有值。
    pre = pd.DataFrame({c: [nav_series(c).asof(seg[0])] for c in shares}, index=[seg[0]])
    df = pd.concat([pre, df]).groupby(level=0).last().sort_index()
    df = df.reindex(seg).ffill().fillna(0.0)
    vals = df.to_numpy() @ np.array(list(shares.values())) + cash
    return pd.Series(vals, index=seg)


def summarize(name, equity, contribution, qdates):
    invested = contribution * len(qdates)
    final = float(equity.iloc[-1])
    total_ret = final / invested - 1.0
    cf = [(d, -contribution) for d in qdates] + [(equity.index[-1], final)]
    irr = xirr(cf)
    mdd = float((equity / equity.cummax() - 1.0).min())
    return {
        "strategy": name,
        "total_invested": invested,
        "final_value": round(final, 2),
        "total_return_pct": round(total_ret * 100, 2),
        "xirr_pct": round(irr * 100, 2) if np.isfinite(irr) else np.nan,
        "mdd_pct": round(mdd * 100, 2),
    }


def run_strategy(name, qdates, selections, mode, port, dates_idx):
    """按季度执行策略, 返回 (净值Series, 每季度持仓数, 每季度换手率)"""
    eq_parts = []
    held_per_q = []
    turnover_per_q = []
    all_dates = dates_idx  # DatetimeIndex, 供 segment_equity 切片
    for i, d in enumerate(qdates):
        sel = selections[i]
        sold_before = port.sold_value
        if mode == "full":
            rebalance_full(port, sel, d, CONTRIBUTION)
        else:
            invest_new_money(port, sel, d, CONTRIBUTION)
        held_per_q.append(len(port.lots))
        P = port.cash + sum(port.fund_value(c, nav_at(c, d)) for c in port.lots)
        turnover_per_q.append((port.sold_value - sold_before) / P if P > 0 else 0.0)
        d_end = qdates[i + 1] if i + 1 < len(qdates) else END
        cash, shares = port.snapshot()
        seg = segment_equity(cash, shares, d, d_end, all_dates)
        if not seg.empty:
            eq_parts.append(seg)
    equity = pd.concat(eq_parts) if eq_parts else pd.Series(dtype=float)
    # 补充到 end 的最后一个估值日
    return equity, held_per_q, turnover_per_q


def main():
    ap = argparse.ArgumentParser(description="4433 季度轮换定投组合回测")
    ap.add_argument("--start", default="2021-01-01", help="回测开始(首个定投季度为该季末)")
    ap.add_argument("--end", default="2026-08-06", help="回测结束日期")
    ap.add_argument("--amount", type=float, default=3000, help="每季度定投金额")
    ap.add_argument("--rebuild-panel", action="store_true", help="强制重建收益面板缓存")
    args = ap.parse_args()

    global CONTRIBUTION, END
    CONTRIBUTION = args.amount
    END = pd.Timestamp(args.end)
    t0 = time.time()

    basic = load_basic()
    active = basic[basic["fund_type"].isin(ACTIVE_TYPES)]
    codes = active["code"].astype(str).tolist()
    print(f"动态池: {len(codes)} 只主动权益基金 ({len(ACTIVE_TYPES)} 类)")
    basic_ft = basic.set_index("code")["fund_type"]

    # 季度末日期
    qdates = list(pd.date_range(args.start, args.end, freq="Q"))
    qdates = [pd.Timestamp(d) for d in qdates]
    if not qdates:
        print("无季度日期, 检查 --start/--end")
        return
    print(f"季度轮换点 {len(qdates)} 个: {qdates[0].date()} ~ {qdates[-1].date()}")

    print("逐基金计算季度窗口收益 ...")
    out, union_dates = compute_window_sums(codes, qdates, rebuild=args.rebuild_panel)
    dates_idx = union_dates

    # 逐季度选基 (动态)
    print("逐季度重算 4433 ...")
    sel_dynamic = []
    for i in range(len(qdates)):
        sel_dynamic.append(select_4433(out, codes, basic_ft, i))

    # 静态206池 (2026 年 4433 结果, 演示幸存者偏差)
    static206 = []
    if os.path.exists(STATIC_CSV):
        csv = pd.read_csv(STATIC_CSV, dtype={"code": str})
        csv = csv[csv["is_4433"] == True]
        static206 = [str(int(c)).zfill(6) for c in csv["code"]]
    print(f"静态池: {len(static206)} 只 (2026年4433结果)")

    # 指数基准
    idx_code = None
    for c in INDEX_CANDIDATES:
        if os.path.exists(os.path.join(NAV_DIR, f"{c}.parquet")):
            idx_code = c
            break
    print(f"指数基准: {idx_code or '无'} (沪深300联接)")

    # 执行 4 个策略
    print("执行回测 ...")
    eqA, heldA, turnA = run_strategy("动态4433-全仓再平衡", qdates, sel_dynamic,
                                     "full", Portfolio(), dates_idx)
    eqB, heldB, _ = run_strategy("动态4433-只轮新资金", qdates, sel_dynamic,
                                 "new", Portfolio(), dates_idx)
    sel_static = [static206] * len(qdates)
    eqC, heldC, turnC = run_strategy("静态206-全仓再平衡", qdates, sel_static,
                                     "full", Portfolio(), dates_idx)
    sel_idx = [[idx_code] for _ in qdates] if idx_code else [[]] * len(qdates)
    eqD, heldD, _ = run_strategy("沪深300联接-定投", qdates, sel_idx,
                                 "new", Portfolio(), dates_idx)

    # 汇总
    rows = []
    for name, eq in [("动态4433-全仓再平衡", eqA), ("动态4433-只轮新资金", eqB),
                     ("静态206-全仓再平衡", eqC), ("沪深300联接-定投", eqD)]:
        if eq.empty:
            print(f"!! {name} 无净值曲线")
            continue
        r = summarize(name, eq, CONTRIBUTION, qdates)
        rows.append(r)
    summary = pd.DataFrame(rows)

    # 选基/持仓统计
    n_sel = [len(s) for s in sel_dynamic]
    stats = pd.DataFrame({
        "date": [d.date() for d in qdates],
        "n_selected_dynamic": n_sel,
        "n_held_full": heldA,
        "n_held_new": heldB,
        "n_held_static": heldC,
    })
    stats_out = os.path.join(RESULTS_DIR, "rotation_selection.csv")
    stats.to_csv(stats_out, index=False, encoding="utf-8-sig")
    print(f"季度选基明细: {stats_out}")

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 250)
    print("\n" + "=" * 96)
    print(f"4433 季度轮换定投组合回测  ({args.start} ~ {args.end}, 每季度 {CONTRIBUTION:.0f} 元, "
          f"申购费 {SG_FEE:.2f}%, 赎回费按持有期阶梯)")
    print("=" * 96)
    print(summary.to_string(index=False))
    print("=" * 96)
    print(f"平均每季度通过4433: {np.mean(n_sel):.0f} 只 (动态池) / 静态池 {len(static206)} 只")
    if heldA:
        print(f"平均持仓: 动态全仓 {np.mean(heldA):.0f} / 动态新钱 {np.mean(heldB):.0f} / "
              f"静态 {np.mean(heldC):.0f} 只; 动态全仓季度换手 {np.mean(turnA)*100:.0f}%")

    # 导出净值曲线
    eq_df = pd.concat({"动态4433-全仓再平衡": eqA, "动态4433-只轮新资金": eqB,
                       "静态206-全仓再平衡": eqC, "沪深300联接-定投": eqD}, axis=1)
    eq_df.index.name = "date"
    eq_out = os.path.join(RESULTS_DIR, "rotation_equity.csv")
    eq_df.to_csv(eq_out, encoding="utf-8-sig")
    print(f"净值曲线: {eq_out}")

    # 画图
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
        plt.rcParams["axes.unicode_minus"] = False
        fig, ax = plt.subplots(figsize=(11, 6))
        for col in eq_df.columns:
            ax.plot(eq_df.index, eq_df[col], label=col, linewidth=1.6)
        ax.set_title(f"4433 季度轮换定投组合回测 ({args.start} ~ {args.end}, "
                     f"每季 {CONTRIBUTION:.0f} 元)")
        ax.set_ylabel("组合市值 (元)")
        ax.legend()
        ax.grid(alpha=0.3)
        fig.tight_layout()
        png = os.path.join(RESULTS_DIR, "rotation_compare.png")
        fig.savefig(png, dpi=130)
        print(f"对比图: {png}")
    except Exception as e:
        print(f"画图跳过: {e}")

    print(f"总用时 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()

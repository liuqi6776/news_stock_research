# -*- coding: utf-8 -*-
"""
拆解 V3 资金流选股负 alpha 的来源

用 V3 的每月 Top20 选股结果, 通过不同交易规则隔离三块影响:
  1. 选股本身 (Top20 集中 vs 全市场等权)
  2. 止盈止损换手 (30%止盈+180天止损 vs 纯月度再平衡)
  3. 交易成本 (买0.1%卖0.15%)

变体:
  V_orig      原始: Top20 + 30%止盈 + 180天止损 + 成本   (= 已有 10.4%)
  V_monthly   月度再平衡: 每月末重选Top20, 卖出退出者, 买入新进者, 含成本
  V_monthly0  月度再平衡, 无成本
  V_buyhold   买入持有: 首月末买入Top20后永不换手, 含买入成本

对照: 全市场等权 (无成本)
"""
import os, glob, time, warnings, bisect
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

t0 = time.time()
ROOT = r"c:\Users\liuqi\quant_system_v2"
DATA = r"D:/iquant_data/data_v2"
SR = os.path.join(ROOT, "research", "sector_rotation")
OUT = os.path.join(SR, "results")
INIT = 1_000_000
BUY_FEE, SELL_FEE = 0.0010, 0.0015
TOP_GLOBAL, MAXK = 20, 3

# ---------- 1. 读 V3 每月预测, 确定 Top20 ----------
print("[1] 读 V3 预测...")
pred = pd.read_csv(os.path.join(OUT, "fullmarket_moneyflow_v3_2015_oos_pred.csv"))
pred["trade_date"] = pred["trade_date"].astype(int)
months = sorted(pred["trade_date"].unique())

def top20_of(sub):
    return sub.sort_values("prob", ascending=False).groupby("industry", sort=False).head(MAXK).head(TOP_GLOBAL)

PICKS = {}
for m in months:
    sub = pred[pred["trade_date"] == m]
    picks = top20_of(sub)
    PICKS[m] = list(picks[["ts_code", "industry", "prob"]].itertuples(index=False, name=None))
print(f"    {len(months)} 个月选股完成, {time.time()-t0:.0f}s")

# ---------- 2. 读全市场 close, 建 CL_MAP ----------
print("[2] 读价格...")
px_files = sorted(glob.glob(os.path.join(DATA, "data_day1", "*.parquet")))
parts = []
for f in px_files:
    if os.path.getsize(f) <= 1024:
        continue
    d = os.path.basename(f)[:8]
    if d < "20171201":
        continue
    df = pd.read_parquet(f, columns=["ts_code", "trade_date", "close"])
    if len(df):
        parts.append(df)
px = pd.concat(parts, ignore_index=True)
px["trade_date"] = pd.to_datetime(px["trade_date"].astype(str), format="%Y%m%d")
px = px.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)
CL_MAP = {}
for code, gdf in px.groupby("ts_code"):
    CL_MAP[code] = (gdf["trade_date"].tolist(), gdf["close"].values)
ALL_DAYS = sorted(px["trade_date"].unique())
print(f"    价格: {len(px):,}行, {len(CL_MAP)}只, {ALL_DAYS[0].date()}~{ALL_DAYS[-1].date()}, {time.time()-t0:.0f}s")

def get_close(code, dt):
    if code not in CL_MAP:
        return None
    dates, closes = CL_MAP[code]
    lo, hi = 0, len(dates) - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        if dates[mid] == dt:
            return closes[mid]
        if dates[mid] < dt:
            lo = mid + 1
        else:
            hi = mid - 1
    return None

def get_close_asof(code, dt):
    """最近 <= dt 的收盘价(停牌/缺价时用最后可得价结转); 无任何可得价返回 None"""
    if code not in CL_MAP:
        return None
    dates, closes = CL_MAP[code]
    i = bisect.bisect_right(dates, dt) - 1
    return closes[i] if i >= 0 else None

# 每只股票的最后交易日/最后收盘价: 用于识别退市(数据终止)并强平
LAST_DATE = {code: dates[-1] for code, (dates, _) in CL_MAP.items()}
LAST_CLOSE = {code: closes[-1] for code, (_, closes) in CL_MAP.items()}

# 月末选股 → 下一个交易日生效
RB = {}
for m, plist in PICKS.items():
    m_dt = pd.to_datetime(str(m), format="%Y%m%d")
    for td in ALL_DAYS:
        if td > m_dt:  # 收盘生成信号, 下一交易日成交(与 freeze 口径一致)
            RB[td] = plist
            break

START = pd.Timestamp("2018-01-01")
start_idx = next(i for i, d in enumerate(ALL_DAYS) if d >= START)

# ---------- 3. 回测引擎 ----------
DELIST_DISCOUNT = 0.0  # 退市强平在最后收盘价基础上再折(DELIST_DISCOUNT=0 即按最后收盘价; 可设 0.3 更保守)

def run(rule, use_cost=True):
    cash, holdings = INIT, {}
    bought = False
    nav_series = []
    delist_log = []  # (code, 强平日, 买入价, 最后收盘价, 强平价, 数量)
    for di in range(start_idx, len(ALL_DAYS)):
        day = ALL_DAYS[di]
        # 退市强平: 持仓股价格数据已终止(day 超过其最后交易日) → 按最后收盘价(打折)卖出
        # 修复: 此前退市股按买入价兜底(=0%损失), 现按最后可得价强平
        for code in list(holdings.keys()):
            if code in LAST_DATE and day > LAST_DATE[code]:
                h = holdings[code]
                sp = LAST_CLOSE[code] * (1 - DELIST_DISCOUNT)
                cash += sp * h["qty"] * (1 - SELL_FEE if use_cost else 1.0)
                delist_log.append((code, day, h["bp"], LAST_CLOSE[code], sp, h["qty"]))
                del holdings[code]
        # 调仓
        if rule == "buyhold":
            if not bought and day in RB:
                bought = True
                plist = RB[day]
                per = cash / len(plist)
                for code, ind, prob in plist:
                    bp = get_close(code, day)
                    if bp is None or bp <= 0:
                        continue
                    qty = int((per * (1 - BUY_FEE)) / (bp * 100)) * 100
                    if qty <= 0:
                        continue
                    cash -= qty * bp * (1 + BUY_FEE)
                    holdings[code] = {"bp": bp, "qty": qty, "ind": ind}
        elif rule == "monthly":
            if day in RB:
                plist = RB[day]
                new_codes = {c for c, _, _ in plist}
                # 卖出不在新Top20的
                for code in list(holdings.keys()):
                    if code not in new_codes:
                        h = holdings[code]
                        sp = get_close_asof(code, day)
                        if sp is None:
                            sp = h["bp"]
                        proceeds = sp * h["qty"] * (1 - SELL_FEE if use_cost else 1.0)
                        cash += proceeds
                        del holdings[code]
                # 买入新进Top20的
                new_picks = [(c, i, p) for c, i, p in plist if c not in holdings]
                if new_picks and cash > 10000:
                    per = cash / len(new_picks)
                    for code, ind, prob in new_picks:
                        bp = get_close(code, day)
                        if bp is None or bp <= 0:
                            continue
                        qty = int((per * (1 - BUY_FEE if use_cost else 1.0)) / (bp * 100)) * 100
                        if qty <= 0:
                            continue
                        cash -= qty * bp * (1 + BUY_FEE if use_cost else 1.0)
                        holdings[code] = {"bp": bp, "qty": qty, "ind": ind}
        elif rule == "stop":
            if day in RB:
                plist = RB[day]
                held_inds = set(h["ind"] for h in holdings.values())
                new_picks = [(c, i, p) for c, i, p in plist
                             if i not in held_inds and c not in holdings]
                if new_picks and cash > 10000:
                    per = cash / len(new_picks)
                    for code, ind, prob in new_picks:
                        bp = get_close(code, day)
                        if bp is None or bp <= 0:
                            continue
                        qty = int((per * (1 - BUY_FEE)) / (bp * 100)) * 100
                        if qty <= 0:
                            continue
                        cash -= qty * bp * (1 + BUY_FEE)
                        holdings[code] = {"bp": bp, "qty": qty, "ind": ind, "buy_i": di}
            for code in list(holdings.keys()):
                h = holdings[code]
                cnow = get_close_asof(code, day)
                if cnow is None:
                    continue
                ret = cnow / h["bp"] - 1
                held = di - h["buy_i"]
                if ret >= 0.30 or held >= 180:
                    cash += cnow * h["qty"] * (1 - SELL_FEE)
                    del holdings[code]
        # 估值
        total = cash
        for code, h in holdings.items():
            cnow = get_close_asof(code, day)
            if cnow is None:
                cnow = h["bp"]
            total += cnow * h["qty"]
        nav_series.append((day, total))
    nav = pd.Series(dict(nav_series)).sort_index()
    return nav, delist_log

print("[3] 回测各变体...")
nav_orig, dl_orig = run("stop", True)
print(f"    V_orig(止盈止损) 完成, 退市强平 {len(dl_orig)} 笔, {time.time()-t0:.0f}s")
nav_monthly, dl_monthly = run("monthly", True)
print(f"    V_monthly(月度再平衡) 完成, 退市强平 {len(dl_monthly)} 笔, {time.time()-t0:.0f}s")
nav_monthly0, dl_monthly0 = run("monthly", False)
print(f"    V_monthly0(月度无成本) 完成, 退市强平 {len(dl_monthly0)} 笔, {time.time()-t0:.0f}s")
nav_buyhold, dl_buyhold = run("buyhold", True)
print(f"    V_buyhold(买入持有) 完成, 退市强平 {len(dl_buyhold)} 笔, {time.time()-t0:.0f}s")

# ---------- 4. 全市场等权 ----------
ew = pd.read_csv(os.path.join(OUT, "equal_weight_benchmark_compare.csv"), index_col=0, parse_dates=True)
ew_all = ew["全市场等权"] * INIT
ew_all = ew_all.reindex(nav_orig.index).ffill()

# ---------- 5. 指标对比 ----------
def stats(nav):
    tr = nav.iloc[-1] / INIT - 1
    years = (nav.index[-1] - nav.index[0]).days / 365.25
    ann = (1 + tr) ** (1 / years) - 1 if years > 0 else np.nan
    peak = nav.cummax()
    mdd = ((nav - peak) / peak).min()
    rets = nav.pct_change().dropna()
    sharpe = rets.mean() / rets.std(ddof=1) * np.sqrt(252) if len(rets) > 1 and rets.std(ddof=1) > 0 else np.nan
    return tr, ann, mdd, sharpe, nav.iloc[-1]

series = {
    "全市场等权(无成本)": ew_all,
    "V_buyhold 买入持有": nav_buyhold,
    "V_monthly0 月度无成本": nav_monthly0,
    "V_monthly 月度含成本": nav_monthly,
    "V_orig 止盈止损": nav_orig,
}

print("\n" + "=" * 92)
print(f"{'变体':<24}{'累计':>9}{'年化':>9}{'MaxDD':>9}{'夏普':>7}{'期末(万)':>10}")
print("-" * 92)
res = {}
for name, nav in series.items():
    tr, ann, mdd, shp, end = stats(nav)
    res[name] = (ann, mdd, shp)
    print(f"{name:<24}{tr:>9.1%}{ann:>9.1%}{mdd:>9.1%}{shp:>7.2f}{end/1e4:>10.0f}")
print("=" * 92)

# ---------- 5.5 退市/缺价处理披露 ----------
print("\n--- 退市/缺价处理披露 (买入价兜底已修复) ---")
dl_map = {
    "V_buyhold 买入持有": dl_buyhold,
    "V_monthly0 月度无成本": dl_monthly0,
    "V_monthly 月度含成本": dl_monthly,
    "V_orig 止盈止损": dl_orig,
}
for name, log in dl_map.items():
    if not log:
        print(f"  {name:<20} 无退市强平")
        continue
    uniq = len({e[0] for e in log})
    loss = sum((e[2] - e[4]) * e[5] for e in log)  # (买入价 - 强平价) * 数量
    print(f"  {name:<20} 退市强平 {len(log)} 笔 / {uniq} 只, 相对买入价损失 {loss:,.0f} 元")

# ---------- 6. 拆解 ----------
print("\n--- 负 alpha 来源拆解 (相对全市场等权 15.3%) ---")
ann_ew = res["全市场等权(无成本)"][0]
ann_bh = res["V_buyhold 买入持有"][0]
ann_m0 = res["V_monthly0 月度无成本"][0]
ann_m = res["V_monthly 月度含成本"][0]
ann_orig = res["V_orig 止盈止损"][0]

print(f"  ① 选股集中度损失 (等权→买入持有Top20):  {ann_ew:.1%} → {ann_bh:.1%}   ({ann_bh-ann_ew:+.1%})")
print(f"  ② 月度换手损失 (买入持有→月度再平衡):    {ann_bh:.1%} → {ann_m0:.1%}   ({ann_m0-ann_bh:+.1%})")
print(f"  ③ 交易成本损失 (月度无成本→月度含成本):  {ann_m0:.1%} → {ann_m:.1%}   ({ann_m-ann_m0:+.1%})")
print(f"  ④ 止盈止损损失 (月度含成本→止盈止损):    {ann_m:.1%} → {ann_orig:.1%}   ({ann_orig-ann_m:+.1%})")

# 保存
comb = pd.DataFrame({name: nav / INIT for name, nav in series.items()})
comb.to_csv(os.path.join(OUT, "v3_alpha_decompose.csv"), encoding="utf-8-sig")
print(f"\n[保存] v3_alpha_decompose.csv  总耗时 {time.time()-t0:.0f}s")

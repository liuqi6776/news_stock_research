# -*- coding: utf-8 -*-
"""ETF 层深化 · 方向A首验: 行业 ETF 按「估值(PE分位)+质量」分层回测

数据源:
  fund1 (D:/iquant_data/data_v2/fund1): ETF 日频 OHLCV (YYYYMMDD.parquet)
  other_day1: 个股 pe / circ_mv (日频快照)
  industry1/industry.parquet: 个股 -> 110 通达信行业
  fundamental1/fina_indicator_cache.parquet: 个股财务质量 (2023-03~2026-03)

方法(防前视):
  1. 26 只申万一级行业 ETF (T7 池) 日频 close -> 月末面板, 非交易日 asof 前向填充。
  2. 行业 PE: 个股 pe 按 circ_mv 调和加权聚合到 110 行业, 再映射到 26 申万一级;
     滚动 48 月分位 (< 0.30 = 低估)。
  3. 行业质量: fundamental 各指标按 (行业, end_date) 取中位数, 可用日=公告日中位数,
     横截面 z-score 等权合成 (debt_to_assets 取负)。
  4. 月度调仓: 月末信号 -> 下月生效。费用 = 申购 0.15% + 赎回按持有期分档。
  5. 2020-2026 多区间验证。
"""
import os
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=RuntimeWarning)

DATA = r"D:/iquant_data/data_v2"
FUND1 = os.path.join(DATA, "fund1")
OTHER_DIR = os.path.join(DATA, "other_day1")
IND_PATH = os.path.join(DATA, "industry1", "industry.parquet")
FUND_PATH = os.path.join(DATA, "fundamental1", "fina_indicator_cache.parquet")

OUT_DIR = r"c:/Users/liuqi/quant_system_v2/research/sector_rotation/results"
CACHE_DIR = os.path.join(OUT_DIR, "etf_layering_cache")
os.makedirs(CACHE_DIR, exist_ok=True)

# ---------------- 参数 ----------------
PE_WINDOW = 48
PE_MIN_PERIODS = 12
LOW_PE_Q = 0.30
HIGH_PE_Q = 0.70
MIN_IND_STOCKS = 5
MIN_QUAL_STOCKS = 3
MIN_QUAL_IND = 3

SUB_FEE = 0.0015      # 申购费 0.15%
# 赎回费按持有交易日数分档
REDEEM_TIERS = [(7, 0.015), (30, 0.0075), (250, 0.005), (500, 0.0025), (1e9, 0.0)]

QUAL_INDICATORS = ["roe", "netprofit_yoy", "or_yoy", "grossprofit_margin", "debt_to_assets"]
NEGATIVE_IND = {"debt_to_assets"}

# T7 的 26 只申万一级行业 ETF (name -> code)
INDUSTRY_ETFS = [
    ("农林牧渔", "159825.SZ"), ("基础化工", "159870.SZ"), ("钢铁", "515210.SH"),
    ("有色金属", "512400.SH"), ("电子", "512480.SH"), ("汽车", "516110.SH"),
    ("家用电器", "159996.SZ"), ("食品饮料", "515170.SH"), ("医药生物", "512010.SH"),
    ("公用事业", "159611.SZ"), ("交通运输", "516320.SH"), ("房地产", "512200.SH"),
    ("社会服务", "159766.SZ"), ("建筑材料", "159745.SZ"), ("建筑装饰", "516950.SH"),
    ("电力设备", "515030.SH"), ("国防军工", "512660.SH"), ("计算机", "512720.SH"),
    ("传媒", "512980.SH"), ("通信", "515880.SH"), ("银行", "512800.SH"),
    ("非银金融", "512880.SH"), ("煤炭", "515220.SH"), ("石油石化", "159930.SZ"),
    ("环保", "512580.SH"), ("机械设备", "159886.SZ"),
]
NAME2CODE = dict(INDUSTRY_ETFS)
CODE2NAME = {c: n for n, c in INDUSTRY_ETFS}

# 110 通达信行业 -> 26 申万一级行业
IND_MAP = {
    "农林牧渔": ["农业综合", "农用机械", "农药化肥", "种植业", "渔业", "林业", "饲料"],
    "基础化工": ["化工原料", "化纤", "塑料", "日用化工", "染料涂料", "橡胶", "化工机械"],
    "钢铁": ["普钢", "特种钢", "钢加工"],
    "有色金属": ["小金属", "铅锌", "铜", "铝", "黄金"],
    "电子": ["IT设备", "元器件", "半导体"],
    "汽车": ["汽车整车", "汽车服务", "汽车配件", "摩托车"],
    "家用电器": ["家用电器", "电器连锁"],
    "食品饮料": ["乳制品", "啤酒", "白酒", "红黄酒", "软饮料", "食品"],
    "医药生物": ["中成药", "化学制药", "医疗保健", "医药商业", "生物制药"],
    "公用事业": ["供气供热", "新型电力", "水力发电", "水务", "火力发电"],
    "交通运输": ["仓储物流", "公共交通", "公路", "机场", "水运", "港口", "空运", "铁路", "路桥"],
    "房地产": ["全国地产", "区域地产", "房产服务", "园区开发"],
    "社会服务": ["旅游景点", "旅游服务", "酒店餐饮", "文教休闲"],
    "建筑材料": ["其他建材", "水泥", "玻璃", "陶瓷", "矿物制品"],
    "建筑装饰": ["建筑工程", "装修装饰"],
    "电力设备": ["电气设备"],
    "国防军工": ["船舶", "航空"],
    "计算机": ["软件服务", "互联网"],
    "传媒": ["影视音像", "出版业", "广告包装"],
    "通信": ["通信设备", "电信运营"],
    "银行": ["银行"],
    "非银金融": ["保险", "证券", "多元金融"],
    "煤炭": ["煤炭开采", "焦炭加工"],
    "石油石化": ["石油加工", "石油开采", "石油贸易"],
    "环保": ["环境保护"],
    "机械设备": ["专用机械", "工程机械", "机床制造", "机械基件", "轻工机械", "运输设备", "纺织机械", "电器仪表"],
}
# 110 行业 -> 申万一级 反向映射
IND110_TO_L1 = {}
for l1, subs in IND_MAP.items():
    for s in subs:
        IND110_TO_L1[s] = l1


def redeem_rate(hold_days):
    for thr, rate in REDEEM_TIERS:
        if hold_days < thr:
            return rate
    return 0.0


# ---------------- 数据构建 ----------------
def load_etf_daily(codes, start="20190101"):
    fp = os.path.join(CACHE_DIR, "etf_close_panel.parquet")
    if os.path.exists(fp):
        return pd.read_parquet(fp)
    files = sorted(f for f in os.listdir(FUND1) if f[:8].isdigit() and f[:8] >= start)
    recs = []
    for i, f in enumerate(files):
        df = pd.read_parquet(os.path.join(FUND1, f), columns=["ts_code", "close"])
        sub = df[df["ts_code"].isin(codes)]
        if len(sub):
            sub = sub.copy()
            sub["date"] = f[:8]
            recs.append(sub[["date", "ts_code", "close"]])
        if (i + 1) % 400 == 0:
            print(f"  [ETF日频] {i+1}/{len(files)}", flush=True)
    big = pd.concat(recs, ignore_index=True)
    px = big.pivot_table(index="date", columns="ts_code", values="close", aggfunc="last")
    px = px.sort_index().ffill()   # asof 前向填充非交易日
    px.to_parquet(fp)
    return px


def build_industry_pe(ind_map_110):
    fp = os.path.join(CACHE_DIR, "industry_pe.parquet")
    if os.path.exists(fp):
        return pd.read_parquet(fp)
    ind = pd.read_parquet(IND_PATH)
    ts_to_110 = dict(zip(ind["ts_code"], ind["industry"]))
    other_dates = sorted(f[:8] for f in os.listdir(OTHER_DIR) if f.endswith(".parquet") and f[:8] >= "20160101")
    s = pd.Series(other_dates)
    month_ends = s.groupby(s.str[:6]).last().tolist()

    rows = []
    for i, d in enumerate(month_ends):
        df = pd.read_parquet(os.path.join(OTHER_DIR, d + ".parquet"), columns=["ts_code", "pe", "circ_mv"])
        df["ind110"] = df["ts_code"].map(ts_to_110)
        df["l1"] = df["ind110"].map(IND110_TO_L1)
        df = df.dropna(subset=["l1"])
        df = df[(df["pe"] > 0) & (df["circ_mv"] > 0)]
        for l1, g in df.groupby("l1"):
            if len(g) < MIN_IND_STOCKS:
                continue
            total_mv = g["circ_mv"].sum()
            total_earn = (g["circ_mv"] / g["pe"]).sum()
            if total_earn > 0:
                rows.append({"date": d, "industry": l1, "pe": total_mv / total_earn})
        if (i + 1) % 24 == 0:
            print(f"  [行业PE] {i+1}/{len(month_ends)}", flush=True)
    pe = pd.DataFrame(rows).pivot_table(index="date", columns="industry", values="pe")
    pe = pe.sort_index()
    pe.to_parquet(fp)
    return pe


def build_quality(ind_map_110):
    fp = os.path.join(CACHE_DIR, "industry_quality.parquet")
    if os.path.exists(fp):
        return pd.read_parquet(fp)
    ind = pd.read_parquet(IND_PATH)
    ts_to_110 = dict(zip(ind["ts_code"], ind["industry"]))
    fund = pd.read_parquet(FUND_PATH)
    fund = fund.copy()
    fund["ann_date"] = fund["ann_date"].astype(str).astype(int)
    fund["end_date"] = fund["end_date"].astype(str)
    fund["l1"] = fund["ts_code"].map(ts_to_110).map(IND110_TO_L1)
    fund = fund.dropna(subset=["l1"])
    fund = fund.sort_values("ann_date").drop_duplicates(["ts_code", "end_date"], keep="last")

    rows = []
    for (l1, ed), g in fund.groupby(["l1", "end_date"]):
        has_q = int(g[QUAL_INDICATORS].notna().any(axis=1).sum())
        if has_q < MIN_QUAL_STOCKS:
            continue
        row = {"industry": l1, "end_date": ed, "n_stocks": has_q,
               "avail_date": int(g["ann_date"].median())}
        for c in QUAL_INDICATORS:
            row[c] = g[c].median()
        rows.append(row)
    agg = pd.DataFrame(rows)

    out = []
    for ed, g in agg.groupby("end_date"):
        z = {}
        for c in QUAL_INDICATORS:
            x = g[c]
            sd = float(x.std(ddof=0))
            zc = (x - x.mean()) / sd if (sd and sd > 1e-12) else pd.Series(0.0, index=x.index)
            if c in NEGATIVE_IND:
                zc = -zc
            z[c] = zc
        zdf = pd.DataFrame(z, index=g.index)
        n_avail = zdf.notna().sum(axis=1)
        comp = zdf.mean(axis=1)
        comp[n_avail < MIN_QUAL_IND] = np.nan
        gg = g.copy()
        gg["quality"] = comp.values
        out.append(gg[["industry", "end_date", "avail_date", "n_stocks", "quality"]])
    qual = pd.concat(out, ignore_index=True)
    qual.to_parquet(fp)
    return qual


def pit_quality_at(qual, t):
    avail = qual[qual["avail_date"] <= t]
    if avail.empty:
        return pd.Series(dtype=float)
    latest = avail.sort_values("end_date").groupby("industry").tail(1)
    return latest.set_index("industry")["quality"]


# ---------------- 回测 ----------------
def calc_stats(nav, name=""):
    nav = nav.dropna()
    if len(nav) < 2:
        return
    ret = nav.pct_change().dropna()
    n_months = len(nav)
    years = n_months / 12.0
    total = nav.iloc[-1] / nav.iloc[0] - 1
    ann = (1 + total) ** (1 / years) - 1 if years > 0 else np.nan
    mdd = (nav / nav.cummax() - 1).min()
    sharpe = ret.mean() / ret.std(ddof=0) * np.sqrt(12) if ret.std(ddof=0) > 0 else np.nan
    return dict(name=name, total=total, ann=ann, mdd=mdd, sharpe=sharpe,
                n_months=n_months, start=str(nav.index[0]), end=str(nav.index[-1]))


def run_portfolio(ret_df, decision_ends, next_map, target_fn, trading_dates):
    """ret_df: index=月末(d1), columns=ETF code, 值为该月收益(d0->d1)
    target_fn(d0) -> set of codes
    """
    nav = 1.0
    curve = {}
    prev_set = set()
    prev_w = {}
    entry = {}  # code -> 决策月末 d0
    date_idx = {d: i for i, d in enumerate(trading_dates)}

    for d0 in decision_ends:
        d1 = next_map.get(d0)
        if d1 is None or d1 not in ret_df.index:
            continue
        target = {c for c in target_fn(d0) if c in ret_df.columns}
        codes = [c for c in target if pd.notna(ret_df.loc[d1, c])]
        if not codes:
            curve[d1] = nav
            prev_set = set()
            prev_w = {}
            continue
        w = 1.0 / len(codes)
        r = float(np.mean([ret_df.loc[d1, c] for c in codes]))
        fee = 0.0
        for c in codes:
            if c not in prev_set:
                fee += w * SUB_FEE
                if c not in entry:
                    entry[c] = d0
        for c in prev_set:
            if c not in codes:
                i0 = date_idx.get(entry.get(c, d0), 0)
                i1 = date_idx.get(d1, i0 + 21)
                fee += prev_w[c] * redeem_rate(i1 - i0)
        nav *= (1 + r - fee)
        curve[d1] = nav
        prev_set = set(codes)
        prev_w = {c: w for c in codes}
    return pd.Series(curve)


def main():
    print("=" * 80)
    print("方向A首验: 行业ETF 估值(PE分位)+质量 分层")
    print("=" * 80, flush=True)

    codes = [c for _, c in INDUSTRY_ETFS]
    print("[1/4] 加载 ETF 日频 close (fund1)...", flush=True)
    px = load_etf_daily(codes)
    print(f"      close 面板 {px.shape[0]}日 x {px.shape[1]}ETF, {px.index[0]}~{px.index[-1]}",
          flush=True)

    # 月末面板与月收益
    s = pd.Series(px.index)
    month_ends = s.groupby(s.str[:6]).last().tolist()
    px_me = px.loc[px.index.isin(month_ends)]
    ret_df = px_me.pct_change().iloc[1:]  # index=月末(d1), 值为该月收益
    print(f"      月末面板 {px_me.shape[0]}月, 月收益面板 {ret_df.shape[0]}月", flush=True)

    decision_ends = [d for d in month_ends[:-1] if d >= "20200101"]
    next_map = {month_ends[i]: month_ends[i + 1] for i in range(len(month_ends) - 1)}

    print("[2/4] 构建 26 申万一级行业 PE 分位 (2016起, 48月窗)...", flush=True)
    pe_df = build_industry_pe(None)
    pe_pct = pe_df.rolling(PE_WINDOW, min_periods=PE_MIN_PERIODS).apply(
        lambda a: float((a[~np.isnan(a)] <= a[~np.isnan(a)][-1]).mean())
        if len(a[~np.isnan(a)]) else np.nan, raw=True)
    print(f"      PE面板 {pe_df.shape} | 分位面板 {pe_pct.shape}, {pe_pct.index[0]}~{pe_pct.index[-1]}",
          flush=True)

    print("[3/4] 构建行业质量分 (fundamental, PIT)...", flush=True)
    qual = build_quality(None)
    print(f"      质量记录 {qual.shape[0]} 条 | 行业 {qual['industry'].nunique()} | "
          f"end_date {qual['end_date'].min()}~{qual['end_date'].max()}", flush=True)

    # 交易日期列表(用于赎回费持有期)
    trading_dates = px.index.tolist()

    print("[4/4] 回测 (月度调仓, 申购0.15%+赎回分档)...", flush=True)

    def target_all(d0):
        return set(codes)

    def target_low_pe(d0):
        if d0 not in pe_pct.index:
            return set()
        row = pe_pct.loc[d0]
        return {NAME2CODE[i] for i in row[row < LOW_PE_Q].index if i in NAME2CODE}

    def target_high_pe(d0):
        if d0 not in pe_pct.index:
            return set()
        row = pe_pct.loc[d0]
        return {NAME2CODE[i] for i in row[row > HIGH_PE_Q].index if i in NAME2CODE}

    nav_all = run_portfolio(ret_df, decision_ends, next_map, target_all, trading_dates)
    nav_low = run_portfolio(ret_df, decision_ends, next_map, target_low_pe, trading_dates)
    nav_high = run_portfolio(ret_df, decision_ends, next_map, target_high_pe, trading_dates)

    # 质量分层 (仅 2023-2026, 质量数据覆盖期)
    def target_low_q_high(d0):
        base = target_low_pe(d0)
        if not base or d0 not in pe_pct.index:
            return set()
        q = pit_quality_at(qual, int(d0))
        q = q[[i for i in q.index if NAME2CODE.get(i) in base]]
        if len(q) < 2:
            return set()
        med = q.median()
        return {NAME2CODE[i] for i in q[q >= med].index}

    def target_low_q_low(d0):
        base = target_low_pe(d0)
        if not base or d0 not in pe_pct.index:
            return set()
        q = pit_quality_at(qual, int(d0))
        q = q[[i for i in q.index if NAME2CODE.get(i) in base]]
        if len(q) < 2:
            return set()
        med = q.median()
        return {NAME2CODE[i] for i in q[q < med].index}

    nav_qh = run_portfolio(ret_df, decision_ends, next_map, target_low_q_high, trading_dates)
    nav_ql = run_portfolio(ret_df, decision_ends, next_map, target_low_q_low, trading_dates)

    # 对齐到共同区间
    combos = [("全26行业等权(基线)", nav_all), ("低估行业等权", nav_low),
              ("高估行业等权", nav_high), ("低估+高质量", nav_qh), ("低估+低质量", nav_ql)]

    print("\n" + "=" * 80)
    print("各组合净值统计 (全区间)")
    print("=" * 80)
    for name, nav in combos:
        st = calc_stats(nav, name)
        if st:
            print(f"{name:18s} 累计={st['total']:+.3f}  年化={st['ann']:+.3f}  "
                  f"最大回撤={st['mdd']:+.3f}  夏普={st['sharpe']:+.2f}  "
                  f"月数={st['n_months']} ({st['start']}~{st['end']})")

    # 多区间验证
    for lo, hi in [("202001", "202212"), ("202301", "202607")]:
        print("\n" + "-" * 80)
        print(f"区间 {lo}~{hi}")
        print("-" * 80)
        for name, nav in combos:
            sub = nav[(nav.index >= lo) & (nav.index <= hi)]
            if len(sub) < 2:
                print(f"{name:18s} 无足够样本")
                continue
            st = calc_stats(sub, name)
            if st:
                print(f"{name:18s} 累计={st['total']:+.3f}  年化={st['ann']:+.3f}  "
                      f"最大回撤={st['mdd']:+.3f}  夏普={st['sharpe']:+.2f}  月数={st['n_months']}")

    # 保存净值曲线
    out = pd.DataFrame({name: nav for name, nav in combos}).sort_index()
    out.to_csv(os.path.join(OUT_DIR, "etf_layering_nav.csv"))
    print("\n已保存净值曲线到 results/etf_layering_nav.csv")

    # 低估 vs 高估 多空 (估值分层有效性)
    print("\n" + "=" * 80)
    print("估值分层多空 (低估等权 - 高估等权) 逐区间")
    print("=" * 80)
    ls = nav_low - nav_high  # 用于画差异, 实际看累计收益差
    # 用对数收益差更合理: 打印两组合区间累计差
    for lo, hi in [("202001", "202607"), ("202001", "202212"), ("202301", "202607")]:
        l = nav_low[(nav_low.index >= lo) & (nav_low.index <= hi)]
        h = nav_high[(nav_high.index >= lo) & (nav_high.index <= hi)]
        if len(l) and len(h):
            tot_l = l.iloc[-1] / l.iloc[0] - 1
            tot_h = h.iloc[-1] / h.iloc[0] - 1
            print(f"{lo}~{hi}: 低估累计={tot_l:+.3f}  高估累计={tot_h:+.3f}  差={tot_l-tot_h:+.3f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

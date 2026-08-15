# -*- coding: utf-8 -*-
"""
可交易板块版回测 (板块的交易 - 只用可交易的板块)
================================================
1. 30个可交易板块 (有指数基金覆盖), 每个板块映射到细分行业
2. 板块PE分位 = 板块内细分行业PE分位的中位数 (60个月滚动)
3. PE分位<30% → 买入该板块代表基金 (成立最早净值历史最长的)
4. 基金累计涨30% → 止盈 (低换手, 无强平)
"""
import os, glob
import numpy as np
import pandas as pd

ROOT = r"c:\Users\liuqi\quant_system_v2"
DATA = r"D:\iquant_data\data_v2"
PE_CSV = os.path.join(ROOT, "research", "sector_rotation", "results", "industry_pe.csv")
OUT_DIR = os.path.join(ROOT, "research", "sector_rotation", "results")
NAV_DIR = os.path.join(DATA, "fund2", "nav")
FUND_BASIC = os.path.join(DATA, "fund2", "fund_basic_O.parquet")

# ========== 1. 板块 → 细分行业映射 ==========
SECT_MAP = {
    "医药医疗": ["中成药", "化学制药", "生物制药", "医疗保健", "医药商业"],
    "白酒消费": ["白酒", "食品", "乳制品", "啤酒", "红黄酒", "软饮料"],
    "银行": ["银行"],
    "证券保险": ["证券", "保险"],
    "地产": ["全国地产", "区域地产", "园区开发", "房产服务"],
    "煤炭": ["煤炭开采", "焦炭加工"],
    "钢铁": ["普钢", "特种钢", "钢加工"],
    "有色金属": ["黄金", "铜", "铝", "铅锌", "小金属"],
    "石油石化": ["石油加工", "石油开采"],
    "化工": ["化工原料", "化工机械", "化纤", "农药化肥", "塑料", "日用化工", "染料涂料", "橡胶"],
    "电力": ["火力发电", "水力发电", "新型电力"],
    "公用事业": ["供气供热", "水务"],
    "新能源": ["电气设备"],
    "半导体芯片": ["半导体", "元器件", "电器仪表"],
    "电子": ["IT设备", "元器件"],
    "计算机软件": ["软件服务", "互联网"],
    "通信": ["通信设备", "电信运营"],
    "传媒": ["影视音像", "出版业", "广告包装"],
    "军工": ["航空", "船舶"],
    "汽车": ["汽车整车", "汽车配件", "汽车服务", "摩托车"],
    "家电": ["家用电器"],
    "建材": ["水泥", "玻璃", "其他建材"],
    "建筑": ["建筑工程", "装修装饰"],
    "机械": ["专用机械", "工程机械", "机床制造", "机械基件", "轻工机械", "纺织机械", "农用机械"],
    "农业": ["种植业", "饲料", "渔业", "农业综合"],
    "纺织服装": ["纺织", "服饰"],
    "交通运输": ["机场", "港口", "空运", "水运", "仓储物流", "公共交通", "路桥"],
    "环保": ["环境保护"],
    "基建": ["建筑工程", "路桥"],
    "人工智能": ["软件服务", "互联网", "元器件"],
}

# ========== 2. 板块PE分位 (60个月滚动) ==========
pe_df = pd.read_csv(PE_CSV, index_col=0); pe_df.index = pe_df.index.astype(str)
pe_pct = pe_df.rolling(60, min_periods=12).rank(pct=True)

# 板块PE分位 = 板块内细分行业PE分位的中位数
sect_pct = {}
for sect, inds in SECT_MAP.items():
    avail = [i for i in inds if i in pe_pct.columns]
    if not avail: continue
    sect_pct[sect] = pe_pct[avail].median(axis=1)
sect_pct_df = pd.DataFrame(sect_pct)
print(f"可交易板块: {len(sect_pct_df.columns)}个")
print(f"PE分位时间范围: {sect_pct_df.index[0]} ~ {sect_pct_df.index[-1]}")

# ========== 3. 每个板块选代表基金 (最早有净值) ==========
fb = pd.read_parquet(FUND_BASIC)
idx_equity = fb[fb["fund_type"] == "指数型-股票"].copy()
nav_files = set(os.listdir(NAV_DIR))

# 板块关键词(来自probe) → 选成立最早的一只
SECT_FUND_KW = {
    "医药医疗": "医药", "白酒消费": "消费", "银行": "银行", "证券保险": "证券",
    "地产": "地产", "煤炭": "煤炭", "钢铁": "钢铁", "有色金属": "有色",
    "石油石化": "石油", "化工": "化工", "电力": "电力", "公用事业": "公用事业",
    "新能源": "新能源", "半导体芯片": "半导体", "电子": "电子", "计算机软件": "计算机",
    "通信": "通信", "传媒": "传媒", "军工": "军工", "汽车": "汽车", "家电": "家电",
    "建材": "建材", "建筑": "建筑", "机械": "机械", "农业": "农业", "纺织服装": "纺织",
    "交通运输": "交通运输", "环保": "环保", "基建": "基建", "人工智能": "人工智能",
}

nav_len_cache = {}
def nav_first_date(code):
    fp = f"{code}.parquet"
    if fp not in nav_files: return None
    if code in nav_len_cache: return nav_len_cache[code]
    df = pd.read_parquet(os.path.join(NAV_DIR, fp))
    first = df["date"].iloc[0] if len(df) else None
    nav_len_cache[code] = first
    return first

rep_fund = {}  # sector -> (code, name, first_date)
for sect, kw in SECT_FUND_KW.items():
    pool = idx_equity[idx_equity["name"].str.contains(kw, na=False)]
    best_code, best_name, best_first = None, None, None
    for _, r in pool.iterrows():
        code = str(r["code"])
        fd = nav_first_date(code)
        if fd is None: continue
        if best_first is None or fd < best_first:
            best_code, best_name, best_first = code, r["name"], fd
    if best_code:
        rep_fund[sect] = (best_code, best_name, best_first)

print(f"\n=== 代表基金选择 ===")
print(f"{'板块':<12}{'基金代码':<12}{'基金名称':<30}{'净值起始':<12}")
for sect, (code, name, fd) in sorted(rep_fund.items()):
    print(f"{sect:<12}{code:<12}{name[:26]:<30}{fd:<12}")

# ========== 4. 加载基金净值日频面板 ==========
nav_panels = {}
for sect, (code, name, fd) in rep_fund.items():
    df = pd.read_parquet(os.path.join(NAV_DIR, f"{code}.parquet"))
    df = df[["date", "unit_nav"]].dropna()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")
    nav_panels[sect] = df.set_index("date")["unit_nav"]

# 构建统一日期轴 (所有板块净值日期的并集)
all_dates = None
for sect, s in nav_panels.items():
    idx = set(s.index)
    all_dates = idx if all_dates is None else all_dates.union(idx)
all_dates = sorted(all_dates)
print(f"\n净值日期并集: {all_dates[0].date()} ~ {all_dates[-1].date()} ({len(all_dates)}天)")

# ========== 5. 回测: 板块低估买入代表基金 → 30%止盈 ==========
# 板块PE分位是月频, 对齐到日期轴
sect_pct_dt = sect_pct_df.copy()
sect_pct_dt.index = pd.to_datetime(sect_pct_dt.index, format="%Y%m%d")
BUY_FEE = 0.0010  # 基金申购费
SELL_FEE = 0.0000  # 指数基金赎回费
INIT = 1_000_000
TP = 0.30

cash = INIT
holdings = {}  # sect -> {buy_nav, qty}
nav_series = []
trades = []

# 找每个板块净值的最早日期, 用于判断"该板块某日是否已可交易"
sect_start = {s: ser.index[0] for s, ser in nav_panels.items()}

for di, day in enumerate(all_dates):
    # 信号: 该月板块PE分位 (<=day的最近一期)
    sig_month = None
    for ym in sect_pct_dt.index:
        if ym <= day: sig_month = ym
        else: break
    if sig_month is None: continue
    # 每月第一个交易日调仓
    is_first_of_month = (di == 0) or (all_dates[di-1].month != day.month)
    if is_first_of_month:
        # 买入: 低估板块(未持仓, 且该板块基金当日有净值)
        undv_sects = [s for s in sect_pct_dt.columns
                      if pd.notna(sect_pct_dt.loc[sig_month, s]) and sect_pct_dt.loc[sig_month, s] < 0.30]
        new_sects = []
        for s in undv_sects:
            if s in holdings: continue
            if s in nav_panels and day in nav_panels[s].index:
                new_sects.append(s)
        if new_sects and cash > 5000:
            per = cash / len(new_sects)
            for s in new_sects:
                nav_today = nav_panels[s].loc[day]
                if nav_today <= 0: continue
                buy_amt = per * (1 - BUY_FEE)
                qty = buy_amt / nav_today
                cash -= per
                holdings[s] = {"buy_nav": nav_today, "qty": qty}
                trades.append((day, "BUY", s, per, np.nan, 0))
    # 每日止盈检查 (该板块当日有净值才检查)
    for s in list(holdings.keys()):
        if day not in nav_panels[s].index: continue
        nav_today = nav_panels[s].loc[day]
        ret = nav_today / holdings[s]["buy_nav"] - 1
        if ret >= TP:
            proceeds = nav_today * holdings[s]["qty"] * (1 - SELL_FEE)
            cash += proceeds
            trades.append((day, "TP", s, proceeds, ret, 0))
            del holdings[s]
    # 净值 (用最近可得净值)
    total = cash
    for s, h in holdings.items():
        sub = nav_panels[s][nav_panels[s].index <= day]
        if len(sub):
            total += sub.iloc[-1] * h["qty"]
    nav_series.append((day, total))

nav_s = pd.Series(dict(nav_series)).sort_index()
# 只统计PE信号有效区间 2020起 (60个月分位需要2015-2020数据)
nav_s = nav_s[nav_s.index >= pd.Timestamp("2020-01-01")]
if len(nav_s) < 10:
    print("净值数据不足!")
else:
    tr = nav_s.iloc[-1] / INIT - 1
    years = (nav_s.index[-1] - nav_s.index[0]).days / 365.25
    ann = (1 + tr) ** (1/years) - 1 if years > 0 else np.nan
    peak = nav_s.cummax(); mdd = ((nav_s - peak) / peak).min()
    rets = nav_s.pct_change().dropna()
    sharpe = rets.mean() / rets.std(ddof=1) * np.sqrt(252) if len(rets) > 1 and rets.std(ddof=1) > 0 else np.nan
    buys = sum(1 for t in trades if t[1]=="BUY")
    tps = sum(1 for t in trades if t[1]=="TP")

    print(f"\n========== [可交易板块版] 低估买基金 + 30%止盈 ==========")
    print(f"  时间: {nav_s.index[0].date()} → {nav_s.index[-1].date()} ({years:.1f}年)")
    print(f"  期初: {INIT/1e4:.0f}万 → 期末: {nav_s.iloc[-1]/1e4:.1f}万")
    print(f"  累计: {tr:.1%} | 年化: {ann:.1%}")
    print(f"  回撤: {mdd:.1%} | 夏普: {sharpe:.2f}")
    print(f"  买入{buys}笔 | 止盈{tps}笔 | 止盈率={tps/(buys+1e-9):.0%}")
    print(f"  期末持仓板块: {[s for s in holdings.keys()]}")
    # 持仓明细
    print(f"\n  当前持仓: {len(holdings)}个板块")
    for s, h in holdings.items():
        print(f"    {s}: 成本净值{h['buy_nav']:.3f} 当前收益{nav_panels[s].get(nav_s.index[-1],0)/h['buy_nav']-1:.1%}")

    # 保存
    nav_s.to_csv(os.path.join(OUT_DIR, "tradable_sector_nav.csv"), header=True)
    pd.DataFrame(trades, columns=["date","op","sector","amount","ret","days"]
                ).to_csv(os.path.join(OUT_DIR, "tradable_sector_trades.csv"), index=False, encoding="utf-8-sig")
    print(f"\n已保存 tradable_sector_nav.csv / tradable_sector_trades.csv")

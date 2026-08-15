# -*- coding: utf-8 -*-
"""
可交易板块版 + 时间止损 网格搜索
=================================
参数: MAX_HOLD ∈ [60, 90, 120, 150, 180, 210, 240, 270, 300, 360, None(不止损)]
逻辑: 30%止盈 OR 持仓满MAX_HOLD天未止盈 → 平仓, 资金下月重新配置
对比: 年化 / 回撤 / 夏普 / 买入止盈时间止损笔数 / 平均持仓 / 期末
"""
import os
import numpy as np
import pandas as pd

ROOT = r"c:\Users\liuqi\quant_system_v2"
DATA = r"D:\iquant_data\data_v2"
PE_CSV = os.path.join(ROOT, "research", "sector_rotation", "results", "industry_pe.csv")
OUT_DIR = os.path.join(ROOT, "research", "sector_rotation", "results")
NAV_DIR = os.path.join(DATA, "fund2", "nav")
FUND_BASIC = os.path.join(DATA, "fund2", "fund_basic_O.parquet")

# ========== 1. 板块→细分行业映射 (与backtest_tradable_sectors一致) ==========
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

# ========== 2. 板块PE分位 ==========
pe_df = pd.read_csv(PE_CSV, index_col=0); pe_df.index = pe_df.index.astype(str)
pe_pct = pe_df.rolling(60, min_periods=12).rank(pct=True)
sect_pct = {}
for sect, inds in SECT_MAP.items():
    avail = [i for i in inds if i in pe_pct.columns]
    if avail: sect_pct[sect] = pe_pct[avail].median(axis=1)
sect_pct_df = pd.DataFrame(sect_pct)
sect_pct_dt = sect_pct_df.copy()
sect_pct_dt.index = pd.to_datetime(sect_pct_dt.index, format="%Y%m%d")

# ========== 3. 代表基金 (与backtest_tradable_sectors一致) ==========
fb = pd.read_parquet(FUND_BASIC)
idx_equity = fb[fb["fund_type"] == "指数型-股票"].copy()
nav_files = set(os.listdir(NAV_DIR))
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
    nav_len_cache[code] = df["date"].iloc[0] if len(df) else None
    return nav_len_cache[code]

rep_fund = {}
for sect, kw in SECT_FUND_KW.items():
    pool = idx_equity[idx_equity["name"].str.contains(kw, na=False)]
    best_code, best_first = None, None
    for _, r in pool.iterrows():
        code = str(r["code"]); fd = nav_first_date(code)
        if fd is None: continue
        if best_first is None or fd < best_first:
            best_code, best_first = code, fd
    if best_code: rep_fund[sect] = best_code

nav_panels = {}
for sect, code in rep_fund.items():
    df = pd.read_parquet(os.path.join(NAV_DIR, f"{code}.parquet"))
    df = df[["date", "unit_nav"]].dropna()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")
    nav_panels[sect] = df.set_index("date")["unit_nav"]

all_dates = None
for ser in nav_panels.values():
    idx = set(ser.index)
    all_dates = idx if all_dates is None else all_dates.union(idx)
all_dates = sorted(all_dates)
print(f"净值日期并集: {all_dates[0].date()} ~ {all_dates[-1].date()} ({len(all_dates)}天)")

# ========== 4. 回测函数 ==========
BUY_FEE = 0.0010; SELL_FEE = 0.0000; INIT = 1_000_000; TP = 0.30

def run_backtest(max_hold):
    cash = INIT
    holdings = {}  # sect -> {buy_nav, qty, buy_di}
    nav_series = []; trades = []
    for di, day in enumerate(all_dates):
        # 信号月
        sig_month = None
        for ym in sect_pct_dt.index:
            if ym <= day: sig_month = ym
            else: break
        if sig_month is None: continue
        # 月首调仓
        is_first = (di == 0) or (all_dates[di-1].month != day.month)
        if is_first:
            undv = [s for s in sect_pct_dt.columns
                    if pd.notna(sect_pct_dt.loc[sig_month, s]) and sect_pct_dt.loc[sig_month, s] < 0.30]
            new = [s for s in undv if s not in holdings and s in nav_panels and day in nav_panels[s].index]
            if new and cash > 5000:
                per = cash / len(new)
                for s in new:
                    nav_today = nav_panels[s].loc[day]
                    if nav_today <= 0: continue
                    qty = (per * (1 - BUY_FEE)) / nav_today
                    cash -= per
                    holdings[s] = {"buy_nav": nav_today, "qty": qty, "buy_di": di}
                    trades.append((day, "BUY", s, per, np.nan, 0))
        # 每日卖出检查
        for s in list(holdings.keys()):
            if day not in nav_panels[s].index: continue
            nav_today = nav_panels[s].loc[day]
            ret = nav_today / holdings[s]["buy_nav"] - 1
            held = di - holdings[s]["buy_di"]
            hit_tp = ret >= TP
            hit_time = (max_hold is not None) and (held >= max_hold)
            if hit_tp or hit_time:
                proceeds = nav_today * holdings[s]["qty"] * (1 - SELL_FEE)
                cash += proceeds
                op = "TP" if hit_tp else f"T{max_hold}"
                trades.append((day, op, s, proceeds, ret, held))
                del holdings[s]
        # 净值
        total = cash
        for s, h in holdings.items():
            sub = nav_panels[s][nav_panels[s].index <= day]
            if len(sub): total += sub.iloc[-1] * h["qty"]
        nav_series.append((day, total))
    nav_s = pd.Series(dict(nav_series)).sort_index()
    nav_s = nav_s[nav_s.index >= pd.Timestamp("2020-01-01")]
    return nav_s, trades

# ========== 5. 网格搜索 ==========
HOLDS = [60, 90, 120, 150, 180, 210, 240, 270, 300, 360, None]
results = []
for mh in HOLDS:
    nav_s, trades = run_backtest(mh)
    tr = nav_s.iloc[-1] / INIT - 1
    years = (nav_s.index[-1] - nav_s.index[0]).days / 365.25
    ann = (1 + tr) ** (1/years) - 1 if years > 0 else np.nan
    peak = nav_s.cummax(); mdd = ((nav_s - peak) / peak).min()
    rets = nav_s.pct_change().dropna()
    sharpe = rets.mean() / rets.std(ddof=1) * np.sqrt(252) if len(rets) > 1 and rets.std(ddof=1) > 0 else np.nan
    buys = sum(1 for t in trades if t[1] == "BUY")
    tps = sum(1 for t in trades if t[1] == "TP")
    tls = sum(1 for t in trades if t[1] != "BUY" and t[1] != "TP")
    held_days = [t[5] for t in trades if t[1] != "BUY"]
    avg_held = np.mean(held_days) if held_days else 0
    label = "无止损" if mh is None else f"{mh}天"
    results.append({
        "时间止损": label, "年化": ann, "累计": tr, "回撤": mdd, "夏普": sharpe,
        "买入": buys, "止盈": tps, "时间止损卖出": tls, "止盈率": tps/(buys+1e-9),
        "平均持仓天": avg_held, "期末": nav_s.iloc[-1]/1e4,
    })
    print(f"  {label}: 年化{ann:.1%} 回撤{mdd:.1%} 夏普{sharpe:.2f} "
          f"买入{buys} 止盈{tps} 时间止损{tls} 止盈率{tps/(buys+1e-9):.0%} 持仓{avg_held:.0f}天")

res_df = pd.DataFrame(results)
print("\n========== 网格搜索结果 (可交易板块, 2020-2026) ==========")
print(res_df.to_string(index=False, float_format=lambda x: f"{x:.2f}"))

# 找出最优 (回撤<25% 且 年化最高; 若无满足回撤约束, 用夏普最高)
feasible = res_df[res_df["回撤"] > -0.25]
if len(feasible):
    best = feasible.loc[feasible["年化"].idxmax()]
    print(f"\n>>> 最优(回撤≤25%): 时间止损={best['时间止损']} 年化={best['年化']:.1%} 回撤={best['回撤']:.1%} 夏普={best['夏普']:.2f}")
else:
    best = res_df.loc[res_df["夏普"].idxmax()]
    print(f"\n>>> 最优(夏普最高): 时间止损={best['时间止损']} 年化={best['年化']:.1%} 回撤={best['回撤']:.1%} 夏普={best['夏普']:.2f}")

# 保存
res_df.to_csv(os.path.join(OUT_DIR, "tradable_sector_timehold_grid.csv"), index=False, encoding="utf-8-sig")
print(f"已保存 tradable_sector_timehold_grid.csv")

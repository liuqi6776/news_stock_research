# -*- coding: utf-8 -*-
"""
PE分位滚动窗口长度 网格搜索
=================================
参数: ROLL_WINDOW ∈ [36, 48, 60, 72, 84, 96, 108, 120] 月
逻辑: 固定 时间止损=270天(夏普最优), 变换PE分位滚动窗口, 对比年化/回撤/夏普等
另外加测: 日频价格分位窗口 120/180/240/360/500天 (用户提及120/180/200/240天)
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

# ========== 1. 板块→细分行业映射 ==========
SECT_MAP = {
    "医药医疗": ["中成药", "化学制药", "生物制药", "医疗保健", "医药商业"],
    "白酒消费": ["白酒", "食品", "乳制品", "啤酒", "红黄酒", "软饮料"],
    "银行": ["银行"], "证券保险": ["证券", "保险"],
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

# ========== 2. 代表基金 ==========
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

# ========== 3. 回测函数(可选择信号源: PE分位 或 价格分位) ==========
BUY_FEE = 0.0010; SELL_FEE = 0.0000; INIT = 1_000_000; TP = 0.30
MAX_HOLD = 270  # 固定270天时间止损

def run_backtest_with_signal(sig_df, sig_name="PE分位"):
    """
    sig_df: DataFrame, index=datetime(月频或日频), columns=sector, values=分位数(0~1)
    """
    cash = INIT
    holdings = {}
    nav_series = []; trades = []
    sig_dates = sorted(sig_df.index)
    for di, day in enumerate(all_dates):
        # 找到<=day的最后一个信号日期
        sig_day = None
        for sd in sig_dates:
            if sd <= day: sig_day = sd
            else: break
        if sig_day is None: continue
        # 月首调仓 (或者信号日当天)
        is_first = (di == 0) or (all_dates[di-1].month != day.month)
        if is_first:
            undv = [s for s in sig_df.columns
                    if pd.notna(sig_df.loc[sig_day, s]) and sig_df.loc[sig_day, s] < 0.30]
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
        # 每日止盈+时间止损
        for s in list(holdings.keys()):
            if day not in nav_panels[s].index: continue
            nav_today = nav_panels[s].loc[day]
            ret = nav_today / holdings[s]["buy_nav"] - 1
            held = di - holdings[s]["buy_di"]
            hit_tp = ret >= TP
            hit_time = held >= MAX_HOLD
            if hit_tp or hit_time:
                proceeds = nav_today * holdings[s]["qty"] * (1 - SELL_FEE)
                cash += proceeds
                op = "TP" if hit_tp else "T270"
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

def calc_metrics(nav_s, trades, label):
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
    return {
        "配置": label, "年化": ann, "累计": tr, "回撤": mdd, "夏普": sharpe,
        "买入": buys, "止盈": tps, "时间止损": tls, "止盈率": tps/(buys+1e-9),
        "平均持仓天": avg_held, "期末(万)": nav_s.iloc[-1]/1e4,
    }

# ========== 4. 测试A: PE分位滚动窗口 (月) ==========
pe_df = pd.read_csv(PE_CSV, index_col=0); pe_df.index = pe_df.index.astype(str)
PE_WINDOWS = [36, 48, 60, 72, 84, 96, 108, 120]
results = []

print("\n========== Part A: PE分位滚动窗口测试 (月) ==========")
for pw in PE_WINDOWS:
    # 检查PE数据长度是否足够
    pe_pct = pe_df.rolling(pw, min_periods=max(12, pw//4)).rank(pct=True)
    sect_pct = {}
    for sect, inds in SECT_MAP.items():
        avail = [i for i in inds if i in pe_pct.columns]
        if avail: sect_pct[sect] = pe_pct[avail].median(axis=1)
    sect_pct_df = pd.DataFrame(sect_pct)
    sect_pct_dt = sect_pct_df.copy()
    sect_pct_dt.index = pd.to_datetime(sect_pct_dt.index, format="%Y%m%d")
    # 检查有效数据量
    valid_rows = sect_pct_dt.dropna(how="all").shape[0]
    print(f"  PE窗口{pw:>3}月 → 有效信号月: {valid_rows}")
    if valid_rows < 24:  # 不足2年信号, 跳过
        print(f"    跳过: 信号不足")
        continue
    nav_s, trades = run_backtest_with_signal(sect_pct_dt)
    if len(nav_s) < 50:
        print(f"    跳过: 净值不足")
        continue
    m = calc_metrics(nav_s, trades, f"PE{pw}月")
    results.append(m)
    print(f"    {m['配置']}: 年化{m['年化']:.1%} 回撤{m['回撤']:.1%} 夏普{m['夏普']:.2f} 期末{m['期末(万)']:.0f}万")

# ========== 5. 测试B: 日频价格分位窗口 (天) ==========
PRICE_WINDOWS = [120, 180, 200, 240, 360, 500, 750, 1260]  # 1260≈5年
print("\n========== Part B: 日频价格分位窗口测试 (天) ==========")

# 构造日频价格分位信号
def build_price_pct_signal(pw_days):
    sig_dict = {}
    for sect, ser in nav_panels.items():
        # 对每个板块日频净值做滚动rank
        aligned = ser.reindex(all_dates).ffill()
        pct = aligned.rolling(pw_days, min_periods=max(30, pw_days//4)).rank(pct=True)
        sig_dict[sect] = pct
    sig_df = pd.DataFrame(sig_dict, index=all_dates)
    return sig_df

for pw in PRICE_WINDOWS:
    sig_df = build_price_pct_signal(pw)
    valid_rows = sig_df.dropna(how="all").shape[0]
    print(f"  价格窗口{pw:>4}天 → 有效信号日: {valid_rows}")
    if valid_rows < 252:
        print(f"    跳过: 信号不足")
        continue
    nav_s, trades = run_backtest_with_signal(sig_df)
    if len(nav_s) < 50:
        print(f"    跳过: 净值不足")
        continue
    m = calc_metrics(nav_s, trades, f"价格{pw}天")
    results.append(m)
    print(f"    {m['配置']}: 年化{m['年化']:.1%} 回撤{m['回撤']:.1%} 夏普{m['夏普']:.2f} 期末{m['期末(万)']:.0f}万")

# ========== 6. 汇总 ==========
res_df = pd.DataFrame(results)
print("\n========== 滚动窗口搜索汇总 (时间止损固定270天, 2020-2026) ==========")
print(res_df.to_string(index=False, float_format=lambda x: f"{x:.2f}"))

# 找最优
feasible = res_df[res_df["回撤"] > -0.27]
if len(feasible):
    best = feasible.loc[feasible["夏普"].idxmax()]
    print(f"\n>>> 最优(回撤≤27%且夏普最高): 配置={best['配置']} 年化={best['年化']:.1%} 回撤={best['回撤']:.1%} 夏普={best['夏普']:.2f}")
else:
    best = res_df.loc[res_df["夏普"].idxmax()]
    print(f"\n>>> 最优(夏普最高): 配置={best['配置']} 年化={best['年化']:.1%} 回撤={best['回撤']:.1%} 夏普={best['夏普']:.2f}")

res_df.to_csv(os.path.join(OUT_DIR, "rolling_window_grid_search.csv"), index=False, encoding="utf-8-sig")
print(f"\n已保存 rolling_window_grid_search.csv")

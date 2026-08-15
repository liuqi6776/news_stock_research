# -*- coding: utf-8 -*-
"""
V6-S1 策略完整逻辑 + 每笔交易明细 + 净值/回撤曲线
================================================
在 v6 框架上，仅运行推荐配置 V6-S1，并：
  1) 打印 V6-S1 的【完整策略逻辑】参数快照
  2) 打印【每一笔交易】明细（买入/止盈/时间止损）：日期、代码、名称、行业、
     买卖价格、数量、金额、收益率、持仓天数
  3) 统计：止盈/止损盈亏分布、持仓天数分布、单笔收益直方图
  4) 绘制 V6-S1 净值曲线（标注全部止盈点▲ / 时间止损点✕）与回撤曲线
  5) 保存交易明细 CSV + 图片
"""
import os, glob, time, calendar
import numpy as np
import pandas as pd

ROOT = r"c:\Users\liuqi\quant_system_v2"
DATA = r"D:\iquant_data\data_v2"
PE_CSV = os.path.join(ROOT, "research", "sector_rotation", "results", "industry_pe.csv")
OUT_DIR = os.path.join(ROOT, "research", "sector_rotation", "results")
ML_PANEL = os.path.join(ROOT, "research", "sector_rotation", "stock_ml_panel_72m.parquet")
IND_PARQ = os.path.join(DATA, "industry1", "industry.parquet")
OTHER_DAY_DIR = os.path.join(DATA, "other_day1")

t00 = time.time()

# ================= 0. 行业黑白名单（与 v6 完全一致） =================
BLACKLIST_SECTORS = {"银行", "证券保险", "地产", "公用事业", "基建"}
PREFERRED_SECTORS = {
    "煤炭", "石油石化", "钢铁", "有色金属", "化工",
    "电力", "新能源", "半导体芯片", "电子",
    "汽车", "机械", "建材", "家电", "农业",
    "医药医疗", "白酒消费", "计算机软件", "通信", "军工",
    "交通运输", "环保", "纺织服装", "传媒", "人工智能",
}

# ================= 1. 行业映射 + ST/退市硬过滤标记 =================
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
    "机械": ["专用机械", "工程机械", "机床制造", "机械基件", "轻工机械", "纺织机械", "农用机械"],
    "农业": ["种植业", "饲料", "渔业", "农业综合"],
    "纺织服装": ["纺织", "服饰"],
    "交通运输": ["机场", "港口", "空运", "水运", "仓储物流", "公共交通", "路桥"],
    "环保": ["环境保护"],
    "基建": ["建筑工程", "路桥"],
    "人工智能": ["软件服务", "互联网", "元器件"],
}
IND2SECT = {}
for sect, inds in SECT_MAP.items():
    for i in inds:
        IND2SECT.setdefault(i, []).append(sect)

ind_df = pd.read_parquet(IND_PARQ)
ind_df = ind_df[["ts_code", "name", "industry", "list_date"]].copy()
ind_df["list_date"] = pd.to_datetime(ind_df["list_date"].astype(str), format="%Y%m%d", errors="coerce")
def is_strict_risk(name):
    if not name: return 1
    n = str(name)
    if "退" in n: return 1
    import re
    if re.search(r"\*?S\*?ST", n): return 1
    return 0
ind_df["is_st"] = ind_df["name"].apply(is_strict_risk)
ind_df["sectors"] = ind_df["industry"].map(lambda s: IND2SECT.get(s, []))
def has_black(sects): return any(s in BLACKLIST_SECTORS for s in sects)
ind_df["black"] = ind_df["sectors"].apply(has_black)
ind_df = ind_df[(ind_df["sectors"].apply(len) > 0) & (~ind_df["black"])].copy()
code_info = ind_df.set_index("ts_code")[["name", "industry", "sectors", "is_st", "list_date"]].to_dict("index")
print(f"[1] 可映射股票(黑名单剔除后): {len(code_info)}只")

# ================= 2. ML 面板 =================
ml = pd.read_parquet(ML_PANEL)
ml["dt"] = pd.to_datetime(ml["trade_date"].astype(str), format="%Y%m%d")
ml["ym"] = ml["dt"].dt.year * 100 + ml["dt"].dt.month
print(f"[2] ML面板: {len(ml):,}行, {ml['ts_code'].nunique()}只, {ml['ym'].nunique()}月")

# ================= 3. 估值/市值月度快照 =================
print("[3] 构建月度 pe/pb/换手率/流通市值 快照...")
other_files = sorted(glob.glob(os.path.join(OTHER_DAY_DIR, "*.parquet")))
date_to_file = {}
for f in other_files:
    b = os.path.basename(f)[:8]
    try:
        d = pd.Timestamp(b)
        date_to_file[d] = f
    except:
        pass
end_dt = ml["dt"].max()
yms_unique = sorted(ml["ym"].unique())
val_parts = []
for ym_int in yms_unique:
    y = ym_int // 100
    m = ym_int % 100
    if m < 1 or m > 12: continue
    last_day = calendar.monthrange(y, m)[1]
    found_file = None
    for d in range(last_day, 0, -1):
        cand = pd.Timestamp(year=y, month=m, day=d)
        if cand in date_to_file:
            found_file = date_to_file[cand]
            break
    if found_file is None: continue
    df = pd.read_parquet(found_file)
    df["ym"] = ym_int
    keep = [c for c in ["ts_code", "ym", "pe", "pb", "turnover_rate", "circ_mv"] if c in df.columns]
    val_parts.append(df[keep].copy())
val_df = pd.concat(val_parts, ignore_index=True)
val_df = val_df.drop_duplicates(subset=["ts_code", "ym"], keep="last")
ml2 = ml.merge(val_df, on=["ts_code", "ym"], how="left")
print(f"[3] 合并后ML面板 {len(ml2):,}行")

# ================= 4. v6 综合评分 =================
def build_scores_v6(df):
    d = df.copy()
    for col in ["pe", "pb", "roe", "netprofit_yoy", "chip_conc_20"]:
        if col in d.columns:
            d[col] = d.groupby("ym")[col].transform(lambda x: x.fillna(x.median()))
    d["yoy_c"] = d["netprofit_yoy"].clip(lower=5, upper=100)
    d["peg"] = d["pe"] / d["yoy_c"].where(d["yoy_c"] > 0, np.nan)
    d["peg"] = d.groupby("ym")["peg"].transform(lambda x: x.fillna(x.median()))
    d["peg"] = d["peg"].clip(0, 10)
    if "pb" in d.columns:
        d["pb"] = d["pb"].clip(0.3, 20)
    cols_sign = [
        ("peg", -1), ("chip_conc_20", +1), ("roe", +1),
        ("pe", -1), ("netprofit_yoy", +1),
    ]
    for col, sign in cols_sign:
        col_z = f"z_{col}"
        if col not in d.columns:
            d[col_z] = 0.0
            continue
        g = d.groupby("ym")[col]
        mu = g.transform("mean")
        sd = g.transform("std")
        z = (d[col] - mu) / (sd + 1e-9)
        z = z.clip(-3, 3)
        d[col_z] = sign * z
    d["score"] = (d["z_peg"].fillna(0.0) * 0.40 +
                  d["z_chip_conc_20"].fillna(0.0) * 0.25 +
                  d["z_roe"].fillna(0.0) * 0.15 +
                  d["z_pe"].fillna(0.0) * 0.10 +
                  d["z_netprofit_yoy"].fillna(0.0) * 0.10)
    return d

ml_scored = build_scores_v6(ml2)
print(f"[4] v6综合分计算完毕")

# ================= 5. 板块择时信号 =================
pe_df = pd.read_csv(PE_CSV, index_col=0); pe_df.index = pe_df.index.astype(str)
pe_pct = pe_df.rolling(48, min_periods=12).rank(pct=True)
sect_pct = {}
for sect, inds in SECT_MAP.items():
    avail = [i for i in inds if i in pe_pct.columns]
    if avail: sect_pct[sect] = pe_pct[avail].median(axis=1)
sect_pct_df = pd.DataFrame(sect_pct)
sect_pct_df.index = pd.to_datetime(sect_pct_df.index, format="%Y%m%d")
sect_signal = {}
for idx in sect_pct_df.index:
    ym_int = idx.year * 100 + idx.month
    row = {c: sect_pct_df.loc[idx, c] for c in sect_pct_df.columns
           if pd.notna(sect_pct_df.loc[idx, c]) and c not in BLACKLIST_SECTORS}
    sect_signal[ym_int] = row
print(f"[5] 板块择时信号: {len(sect_signal)}个月")

# ================= 6. 日频收盘面板 =================
print("[6] 构建个股日频收盘面板...")
pool_codes = set(code_info.keys()) & set(ml["ts_code"].unique())
close_dfs = []
for f in sorted(other_files):
    b = os.path.basename(f)[:8]
    try:
        d = pd.Timestamp(b)
    except:
        continue
    if d < pd.Timestamp("2019-12-01") or d > end_dt:
        continue
    df = pd.read_parquet(f, columns=["ts_code", "close"])
    df = df[df["ts_code"].isin(pool_codes)].copy()
    if len(df) == 0: continue
    df["dt"] = d
    close_dfs.append(df)
px = pd.concat(close_dfs, ignore_index=True)
close_panel = {}
for code, g in px.groupby("ts_code"):
    s = g.sort_values("dt").set_index("dt")["close"]
    s = s[~s.index.duplicated()]
    close_panel[code] = s.astype(float)
all_dates_set = set()
for s in close_panel.values(): all_dates_set.update(s.index)
all_dates = sorted(all_dates_set)
print(f"    {len(close_panel)}只股票, {all_dates[0].date()}~{all_dates[-1].date()} ({len(all_dates)}天)")
del close_dfs, px

# ================= 7. V6-S1 回测（带完整交易明细） =================
BUY_FEE = 0.0010
SELL_FEE = 0.0015
INIT = 1_000_000
TP = 0.30
MAX_HOLD = 270
PE_PCT_THR = 0.30

# V6-S1 参数（锁定推荐配置）
P = dict(global_top_k=3, max_same_sector=2, max_pe=80, min_turnover_pct=0.3,
         min_circ_mv_yi=50, max_circ_mv_yi=2000, chip_conc_pctl_threshold=0.70,
         peg_bonus_threshold=1.5, min_list_years=1.5, preferred_weight=1.2,
         pe_pct_thr=PE_PCT_THR)

def run_v6s1_detail():
    ym_to_dt = {}
    for ym_dt in sorted(ml_scored["dt"].unique()):
        ym_int = ym_dt.year * 100 + ym_dt.month
        ym_to_dt[ym_int] = ym_dt

    monthly_picks = {}
    monthly_undv = {}   # 记录每月低估板块（用于展示）
    monthly_detail = {} # 记录每月选中股票 + 评分（用于展示）
    min_circ_wan = P["min_circ_mv_yi"] * 10_000
    max_circ_wan = P["max_circ_mv_yi"] * 10_000

    for ym_int in yms_unique:
        if ym_int not in sect_signal:
            monthly_picks[ym_int] = []; continue
        sig = sect_signal[ym_int]
        undv_secs = [s for s, v in sig.items() if v < P["pe_pct_thr"]]
        monthly_undv[ym_int] = undv_secs
        if not undv_secs:
            monthly_picks[ym_int] = []; continue
        if ym_int not in ym_to_dt:
            monthly_picks[ym_int] = []; continue
        dt_real = ym_to_dt[ym_int]
        sub = ml_scored[ml_scored["dt"] == dt_real].copy()
        if len(sub) == 0:
            monthly_picks[ym_int] = []; continue
        sub["sects"] = sub["ts_code"].map(lambda c: code_info.get(c, {}).get("sectors", []))
        sub["in_undv"] = sub["sects"].apply(lambda lst: any(s in undv_secs for s in lst))
        sub = sub[sub["in_undv"]].copy()
        if len(sub) == 0:
            monthly_picks[ym_int] = []; continue
        # 硬过滤
        sub["st"] = sub["ts_code"].map(lambda c: code_info.get(c, {}).get("is_st", 1))
        sub = sub[sub["st"] == 0]
        def days_since_listed(c, d_day):
            ld = code_info.get(c, {}).get("list_date", None)
            if ld is None or pd.isna(ld): return P["min_list_years"] * 365 + 1
            return (d_day - ld).days
        sub["list_days"] = sub["ts_code"].apply(lambda c: days_since_listed(c, dt_real))
        sub = sub[sub["list_days"] >= P["min_list_years"] * 365]
        if "pe" in sub.columns:
            mask = sub["pe"].isna() | ((sub["pe"] > 0) & (sub["pe"] <= P["max_pe"]))
            sub = sub[mask]
        if "chip_conc_20" in sub.columns and len(sub) >= 5:
            thr = sub["chip_conc_20"].quantile(1.0 - P["chip_conc_pctl_threshold"])
            mask = sub["chip_conc_20"].isna() | (sub["chip_conc_20"] >= thr)
            sub = sub[mask]
        if "turnover_rate" in sub.columns:
            mask = sub["turnover_rate"].isna() | (sub["turnover_rate"] >= P["min_turnover_pct"])
            sub = sub[mask]
        if "circ_mv" in sub.columns:
            mask = sub["circ_mv"].isna() | ((sub["circ_mv"] >= min_circ_wan) & (sub["circ_mv"] <= max_circ_wan))
            sub = sub[mask]
        if len(sub) == 0:
            monthly_picks[ym_int] = []; continue
        # 软加分
        sub["_peg_bonus"] = 0.0
        if "peg" in sub.columns:
            sub.loc[sub["peg"] < P["peg_bonus_threshold"], "_peg_bonus"] = 0.5
            sub.loc[sub["peg"] > 3, "_peg_bonus"] = -0.3
        sub["_roe_bonus"] = 0.0
        if "roe" in sub.columns:
            sub.loc[sub["roe"] >= 15, "_roe_bonus"] = 0.3
            sub.loc[(sub["roe"] < 5) & (sub["roe"] > 0), "_roe_bonus"] = -0.2
        def is_pref(sects): return any(s in PREFERRED_SECTORS for s in sects)
        sub["_pref"] = sub["sects"].apply(is_pref).astype(int)
        sub["score_adj"] = (sub["score"] + sub["_peg_bonus"] + sub["_roe_bonus"]) * \
                           (1.0 + sub["_pref"] * (P["preferred_weight"] - 1.0))
        # 全局 TopK + 行业分散
        sub = sub.sort_values("score_adj", ascending=False)
        chosen = []
        sec_count = {}
        def first_sect(c):
            sects = code_info.get(c, {}).get("sectors", [])
            for s in sects:
                if s in undv_secs: return s
            return sects[0] if sects else "_X_"
        chosen_rows = []
        for _, row in sub.iterrows():
            c = row["ts_code"]
            s = first_sect(c)
            if sec_count.get(s, 0) >= P["max_same_sector"] and len(chosen) >= P["global_top_k"] - 1:
                continue
            chosen.append(c)
            sec_count[s] = sec_count.get(s, 0) + 1
            chosen_rows.append((c, s, row["score_adj"], row["peg"], row["roe"]))
            if len(chosen) >= P["global_top_k"]:
                break
        monthly_picks[ym_int] = chosen
        monthly_detail[ym_int] = chosen_rows

    # ---------- 日频执行（记录每笔交易全字段） ----------
    cash = float(INIT)
    holdings = {}
    nav_series = []
    trades = []   # 每笔交易：{day,op,code,name,sector,price,qty,amount,ret,held}
    buy_ref = {}  # code -> 买入记录（用于补全卖出字段）

    for di, day in enumerate(all_dates):
        is_monthly_start = (di == 0) or (all_dates[di-1].month != day.month)
        if is_monthly_start:
            ym_int = day.year * 100 + day.month
            if ym_int in monthly_picks:
                target = monthly_picks[ym_int]
                new_codes = [c for c in target if c not in holdings
                             and c in close_panel and day in close_panel[c].index]
                if new_codes and cash > 10000:
                    per = cash / len(new_codes)
                    for c in new_codes:
                        p = close_panel[c].loc[day]
                        if p <= 0: continue
                        qty = (per * (1 - BUY_FEE)) / p
                        qty = int(qty / 100) * 100
                        if qty < 100: continue
                        cost = qty * p * (1 + BUY_FEE)
                        if cost > cash: continue
                        cash -= cost
                        holdings[c] = {"buy_price": p, "qty": qty, "buy_di": di}
                        info = code_info.get(c, {})
                        name = info.get("name", c)
                        sect = info.get("sectors", [""])[0] if info.get("sectors") else ""
                        trades.append({"day": day, "op": "BUY", "code": c, "name": name,
                                       "sector": sect, "price": p, "qty": qty,
                                       "amount": cost, "ret": np.nan, "held": 0})
                        buy_ref[c] = {"price": p, "qty": qty}
        # 止盈30% / 时间止损270天
        for c in list(holdings.keys()):
            if c not in close_panel or day not in close_panel[c].index:
                continue
            p = close_panel[c].loc[day]
            ret = p / holdings[c]["buy_price"] - 1
            held = di - holdings[c]["buy_di"]
            if ret >= TP or held >= MAX_HOLD:
                qty = holdings[c]["qty"]
                proceeds = p * qty * (1 - SELL_FEE)
                cash += proceeds
                op = "TP" if ret >= TP else "T270"
                info = code_info.get(c, {})
                name = info.get("name", c)
                sect = info.get("sectors", [""])[0] if info.get("sectors") else ""
                trades.append({"day": day, "op": op, "code": c, "name": name,
                               "sector": sect, "price": p, "qty": qty,
                               "amount": proceeds, "ret": ret, "held": held})
                del holdings[c]
        # 净值
        total = cash
        for c, h in holdings.items():
            s = close_panel[c]
            up_to = s[s.index <= day]
            if len(up_to):
                total += up_to.iloc[-1] * h["qty"]
        nav_series.append((day, total))

    nav_s = pd.Series(dict(nav_series)).sort_index()
    nav_s = nav_s[nav_s.index >= pd.Timestamp("2020-01-01")]
    return nav_s, trades, monthly_picks, monthly_undv, monthly_detail

print("\n" + "="*100)
print("[7] 运行 V6-S1 回测（完整明细模式）...")
print(f"    参数: Top{P['global_top_k']} | 同板块≤{P['max_same_sector']} | PE≤{P['max_pe']} | "
      f"市值{P['min_circ_mv_yi']}-{P['max_circ_mv_yi']}亿 | 筹码前{P['chip_conc_pctl_threshold']*100:.0f}% | "
      f"上市满{P['min_list_years']}年 | 白名单×{P['preferred_weight']} | 止盈{TP:.0%} | T{MAX_HOLD}")
t1 = time.time()
nav_s, trades, monthly_picks, monthly_undv, monthly_detail = run_v6s1_detail()
print(f"    耗时 {time.time()-t1:.0f}s")

# ================= 8. 指标 =================
# 保存月末净值序列（用于交互图）
monthly_nav = nav_s.resample("M").last()
monthly_nav.to_csv(os.path.join(OUT_DIR, "stock_selected_v6s1_monthly_nav.csv"), encoding="utf-8-sig")
print(f"月末净值已保存: stock_selected_v6s1_monthly_nav.csv")

def calc_metrics(nav_s, trades, label):
    tr = nav_s.iloc[-1] / INIT - 1
    yrs = (nav_s.index[-1] - nav_s.index[0]).days / 365.25
    ann = (1 + tr) ** (1/yrs) - 1 if yrs > 0 else np.nan
    pk = nav_s.cummax()
    mdd = ((nav_s - pk) / pk).min()
    ret = nav_s.pct_change().dropna()
    shp = ret.mean() / ret.std(ddof=1) * np.sqrt(252) if len(ret) > 1 and ret.std(ddof=1) > 0 else np.nan
    buys = sum(1 for t in trades if t["op"]=="BUY")
    tps = sum(1 for t in trades if t["op"]=="TP")
    tls = sum(1 for t in trades if t["op"]=="T270")
    hds = [t["held"] for t in trades if t["op"]!="BUY"]
    return {"配置":label,"年化":ann,"累计":tr,"回撤":mdd,"夏普":shp,
            "买入":buys,"止盈":tps,"时间止损":tls,
            "止盈率":tps/(buys+1e-9),"平均持仓天":np.mean(hds) if hds else 0,
            "期末(万)":nav_s.iloc[-1]/1e4}

m = calc_metrics(nav_s, trades, "V6-S1")
print("\n" + "-"*100)
print(f"★ V6-S1 总指标: 年化{m['年化']:.1%} | 累计{m['累计']:.1%} | 回撤{m['回撤']:.1%} | "
      f"夏普{m['夏普']:.2f} | 买入{m['买入']}笔 | 止盈{m['止盈']} | 时间止损{m['时间止损']} | "
      f"止盈率{m['止盈率']:.0%} | 平均持仓{m['平均持仓天']:.0f}天 | 期末{m['期末(万)']:.0f}万")

# ================= 9. 每月调仓明细（低估板块 + 选中股票 + 评分） =================
print("\n" + "="*100)
print("【每月调仓明细】低估板块池 + 选中 Top3 股票（含评分/PE/ROE）")
print("="*100)
for ym_int in sorted(monthly_detail.keys()):
    picks = monthly_detail[ym_int]
    if not picks:
        continue
    undv = monthly_undv.get(ym_int, [])
    parts = []
    for c, s, sc, peg, roe in picks:
        name = code_info.get(c, {}).get("name", c)
        parts.append(f"{name}({s},score={sc:.2f})")
    print(f"  {ym_int}: 低估板块={undv}")
    print(f"        选中: {', '.join(parts)}")

# ================= 10. 每笔交易明细 =================
print("\n" + "="*100)
print("【完整交易流水】按时间顺序（金额单位: 万元）")
print("="*100)
tbl = []
for t in trades:
    amt_w = t["amount"] / 1e4
    if t["op"] == "BUY":
        ret_s, held_s = "-", "-"
    else:
        ret_s = f"{t['ret']*100:+.1f}%"
        held_s = f"{t['held']}天"
    tbl.append({"日期": t["day"].strftime("%Y-%m-%d"), "操作": t["op"], "代码": t["code"],
                "名称": t["name"], "行业": t["sector"], "价格": round(t["price"], 2),
                "数量": t["qty"], "金额(万)": round(amt_w, 2), "收益率": ret_s, "持仓": held_s})
    if t["op"] == "BUY":
        detail_s = ""
    else:
        detail_s = f" 收益={ret_s} 持仓={t['held']}天"
    print(f"  {tbl[-1]['日期']}  {t['op']:<4s} {t['code']} {t['name']:<10s} "
          f"[{t['sector']}] 价={t['price']:.2f} 量={t['qty']} 金额={amt_w:.2f}万{detail_s}")

tdf = pd.DataFrame(tbl)
tdf.to_csv(os.path.join(OUT_DIR, "stock_selected_v6s1_trades.csv"), index=False, encoding="utf-8-sig")
print(f"\n交易明细已保存: {os.path.join(OUT_DIR, 'stock_selected_v6s1_trades.csv')}")

# ================= 11. 盈亏分布统计 =================
closes = [t for t in trades if t["op"] != "BUY"]
ret_arr = np.array([t["ret"] for t in closes])
held_arr = np.array([t["held"] for t in closes])
print("\n" + "="*100)
print("【单笔交易盈亏分布】")
print("="*100)
print(f"  总平仓 {len(closes)} 笔: 盈利 {sum(ret_arr>0)} 笔 ({sum(ret_arr>0)/len(closes):.0%}), "
      f"亏损 {sum(ret_arr<=0)} 笔")
print(f"  平均单笔收益 {ret_arr.mean()*100:+.1f}%, 中位数 {np.median(ret_arr)*100:+.1f}%")
print(f"  最大单笔收益 {ret_arr.max()*100:+.1f}%, 最小单笔收益 {ret_arr.min()*100:+.1f}%")
print(f"  平均持仓 {held_arr.mean():.0f} 天, 中位数 {np.median(held_arr):.0f} 天")
tp_ret = [t["ret"] for t in closes if t["op"]=="TP"]
t270_ret = [t["ret"] for t in closes if t["op"]=="T270"]
print(f"  止盈笔: {len(tp_ret)} 笔 (均收益 {np.mean(tp_ret)*100:+.1f}%)")
print(f"  时间止损笔: {len(t270_ret)} 笔 (均收益 {np.mean(t270_ret)*100:+.1f}%)")
print(f"  单笔收益分位: P10={np.percentile(ret_arr,10)*100:+.1f}% | "
      f"P25={np.percentile(ret_arr,25)*100:+.1f}% | P50={np.median(ret_arr)*100:+.1f}% | "
      f"P75={np.percentile(ret_arr,75)*100:+.1f}% | P90={np.percentile(ret_arr,90)*100:+.1f}%")

# ================= 12. 画图：净值曲线（标注每笔止盈/止损）+ 回撤曲线 =================
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams["font.sans-serif"] = ["SimHei","Microsoft YaHei","Arial Unicode MS"]
rcParams["axes.unicode_minus"] = False

# --- 净值曲线（标注交易点）---
fig, ax = plt.subplots(figsize=(16, 8))
s_norm = nav_s / INIT
ax.plot(s_norm.index, s_norm.values, color="#3C2ECA", lw=1.8, label="V6-S1 净值")

# 标注止盈 / 时间止损点
tp_days = [t["day"] for t in trades if t["op"]=="TP"]
t270_days = [t["day"] for t in trades if t["op"]=="T270"]
ax.scatter(tp_days, [s_norm.loc[d] if d in s_norm.index else np.nan for d in tp_days],
           marker="^", s=46, color="#1DC981", zorder=5, label=f"止盈30% ▲ (n={len(tp_days)})")
ax.scatter(t270_days, [s_norm.loc[d] if d in s_norm.index else np.nan for d in t270_days],
           marker="x", s=40, color="#E8463A", zorder=5, label=f"时间止损270天 ✕ (n={len(t270_days)})")
ax.axhline(1.0, color="gray", lw=0.6, ls=":")
ax.axhline(6.65, color="#22A5F7", lw=1.0, ls="--", label="期末 665 万")
ax.set_title("V6-S1 净值曲线（初始100万） — 每笔交易点标注\n"
             "低估板块池Top3精选 + 硬过滤(ST/PE/市值/筹码/次新) + 软评分(PEG40%/筹码25%/ROE15%)", fontsize=13)
ax.set_xlabel("日期"); ax.set_ylabel("总资产（万元）")
ax.legend(loc="upper left", fontsize=9)
ax.grid(True, alpha=0.3)
plt.tight_layout()
png1 = os.path.join(OUT_DIR, "stock_selected_v6s1_curve_annotated.png")
plt.savefig(png1, dpi=130)
plt.close()
print(f"\n净值曲线图(含交易点标注): {png1}")

# --- 回撤曲线 ---
fig, ax2 = plt.subplots(figsize=(16, 5.5))
pk = nav_s.cummax()
dd = (nav_s - pk) / pk
ax2.fill_between(dd.index, dd.values, 0, color="#3C2ECA", alpha=0.25)
ax2.plot(dd.index, dd.values, color="#3C2ECA", lw=1.2)
# 标注每次回撤谷底
dd_min = dd.min()
ax2.scatter([dd.idxmin()], [dd_min], color="#E8463A", s=60, zorder=5)
ax2.annotate(f"最大回撤 {dd_min:.1%} @ {dd.idxmin().date()}",
             xy=(dd.idxmin(), dd_min), xytext=(-120, -40), textcoords="offset points",
             fontsize=11, color="#E8463A",
             arrowprops=dict(arrowstyle="->", color="#E8463A"))
ax2.axhline(0, color="gray", lw=0.6)
ax2.axhline(-0.289, color="orange", lw=1.0, ls="--", label="V6-S1 最大回撤 -28.9%")
ax2.set_title("V6-S1 回撤曲线（水下图）", fontsize=13)
ax2.set_ylabel("回撤幅度")
ax2.legend(loc="lower left", fontsize=9)
ax2.grid(True, alpha=0.3)
plt.tight_layout()
png2 = os.path.join(OUT_DIR, "stock_selected_v6s1_drawdown_annotated.png")
plt.savefig(png2, dpi=130)
plt.close()
print(f"回撤曲线图: {png2}")

# --- 单笔收益直方图 ---
fig, ax3 = plt.subplots(figsize=(12, 5))
ax3.hist(ret_arr*100, bins=20, color="#A9AEFF", edgecolor="#3C2ECA", alpha=0.85)
ax3.axvline(30, color="#1DC981", lw=1.5, ls="--", label="止盈线 +30%")
ax3.axvline(0, color="gray", lw=1.0)
ax3.set_title(f"单笔交易收益率分布（{len(closes)} 笔平仓）", fontsize=13)
ax3.set_xlabel("单笔收益率 (%)"); ax3.set_ylabel("笔数")
ax3.legend(fontsize=10)
ax3.grid(True, alpha=0.3)
plt.tight_layout()
png3 = os.path.join(OUT_DIR, "stock_selected_v6s1_ret_hist.png")
plt.savefig(png3, dpi=130)
plt.close()
print(f"单笔收益分布图: {png3}")

print(f"\n总耗时 {time.time()-t00:.0f}s")

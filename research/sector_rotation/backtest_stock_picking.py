# -*- coding: utf-8 -*-
"""
板块择时 + 板块内个股精选 回测 v4 （最终版）
============================================
核心改进（针对v3集中踩雷银行/地产的致命问题）：
  1) 【行业黑名单】剔除银行/地产/证券保险/公用事业/基建等价值陷阱
  2) 【选股模式】低估板块池内 → 全局评分最高TopN（不是每板块内部TopN）
        → 自动聚焦"好行业里的便宜公司"，而不是捡垃圾行业的便宜货
  3) 【中大盘市值】50~1500亿（流通市值），既有流动性又有上涨弹性
  4) 【分散约束】Top3时最多来自2个不同板块，防止单行业黑天鹅
  5) 【评分：PEG+筹码】PEG 35%（估值-增长匹配）、筹码集中度25%、PE 20%、ROE 10%、增长10%
  6) 止盈30%、时间止损270天、买入成本0.1%、卖出0.15%
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

# ============================================================
# 0. 行业黑名单（价值陷阱，即使低估也不买）
#    行业白名单（历史胜率高的成长/制造/周期优质板块）
# ============================================================
# 价值陷阱（PE分位低 但 永远不涨/阴跌）：金融地产 + 公用基建类
BLACKLIST_SECTORS = {"银行", "证券保险", "地产", "公用事业", "基建"}
# 优先白名单（低估后反弹概率高 胜率显著高于平均）
PREFERRED_SECTORS = {"煤炭", "有色金属", "石油石化", "钢铁", "化工",
                     "电力", "新能源", "半导体芯片", "电子",
                     "汽车", "机械", "建材", "农业", "家电",
                     "医药医疗", "白酒消费", "计算机软件", "通信",
                     "军工", "交通运输", "环保", "纺织服装", "传媒", "人工智能"}

# ============================================================
# 1. 基础映射
# ============================================================
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
IND2SECT = {}
for sect, inds in SECT_MAP.items():
    for i in inds:
        IND2SECT.setdefault(i, []).append(sect)

ind_df = pd.read_parquet(IND_PARQ)
ind_df = ind_df[["ts_code", "name", "industry", "list_date"]].copy()
ind_df["list_date"] = pd.to_datetime(ind_df["list_date"].astype(str), format="%Y%m%d", errors="coerce")
ind_df["is_st"] = ind_df["name"].fillna("").str.contains(r"\*?ST", regex=True).astype(int)
ind_df["sectors"] = ind_df["industry"].map(lambda s: IND2SECT.get(s, []))
# 过滤：必须属于某个板块 AND 不属于纯黑名单行业（黑名单完全剔除候选）
def has_black(sects): return any(s in BLACKLIST_SECTORS for s in sects)
ind_df["black"] = ind_df["sectors"].apply(has_black)
ind_df = ind_df[(ind_df["sectors"].apply(len) > 0) & (~ind_df["black"])].copy()
code_info = ind_df.set_index("ts_code")[["name", "industry", "sectors", "is_st", "list_date"]].to_dict("index")
print(f"[1] 可映射股票(剔除黑名单后): {len(code_info)}只 (黑名单: {sorted(BLACKLIST_SECTORS)})")

# ============================================================
# 2. 月度ML面板基础
# ============================================================
ml = pd.read_parquet(ML_PANEL)
ml["dt"] = pd.to_datetime(ml["trade_date"].astype(str), format="%Y%m%d")
ml["ym"] = ml["dt"].dt.year * 100 + ml["dt"].dt.month
print(f"[2] ML面板: {len(ml):,}行, {ml['ts_code'].nunique()}只, {ml['ym'].nunique()}月 "
      f"({ml['dt'].min().date()}~{ml['dt'].max().date()})")

# ============================================================
# 3. 估值/换手率/流通市值 月度快照
# ============================================================
print("[3] 构建月度pe/pb/换手率/流通市值快照...")
other_files = sorted(glob.glob(os.path.join(OTHER_DAY_DIR, "*.parquet")))
date_to_file = {}
for f in other_files:
    b = os.path.basename(f)[:8]
    try:
        d = pd.Timestamp(b)
        date_to_file[d] = f
    except:
        pass
start_dt, end_dt = ml["dt"].min(), ml["dt"].max()
yms_unique = sorted(ml["ym"].unique())
val_parts = []
for ym_int in yms_unique:
    y = ym_int // 100
    m = ym_int % 100
    if m < 1 or m > 12 or y < 2000 or y > 2100:
        continue
    last_day = calendar.monthrange(y, m)[1]
    found_file = None
    for d in range(last_day, 0, -1):
        cand = pd.Timestamp(year=y, month=m, day=d)
        if cand in date_to_file:
            found_file = date_to_file[cand]
            break
    if found_file is None:
        continue
    df = pd.read_parquet(found_file)
    df["ym"] = ym_int
    keep = [c for c in ["ts_code", "ym", "pe", "pb", "turnover_rate", "circ_mv"] if c in df.columns]
    val_parts.append(df[keep].copy())

val_df = pd.concat(val_parts, ignore_index=True)
val_df = val_df.drop_duplicates(subset=["ts_code", "ym"], keep="last")
print(f"    估值快照: {len(val_df):,}行 × {val_df['ym'].nunique()}月")
for c in ["pe", "pb", "turnover_rate", "circ_mv"]:
    if c in val_df.columns:
        print(f"    {c}: 非空 {val_df[c].notna().mean():.0%}")

ml2 = ml.merge(val_df, on=["ts_code", "ym"], how="left")
print(f"[3] 合并后ML面板 {len(ml2):,}行")
for c in ["pe", "pb", "turnover_rate", "circ_mv", "chip_conc_20", "roe", "netprofit_yoy"]:
    cov = ml2[c].notna().mean() if c in ml2.columns else 0
    print(f"    {c}: {cov:.0%} 有值")

# ============================================================
# 4. 综合选股分（PEG+筹码权重最高）
# ============================================================
def build_scores_v4(df):
    d = df.copy()
    for col in ["pe", "roe", "netprofit_yoy", "chip_conc_20"]:
        if col in d.columns:
            d[col] = d.groupby("ym")[col].transform(lambda x: x.fillna(x.median()))
    d["yoy_c"] = d["netprofit_yoy"].clip(lower=5, upper=100)
    d["peg"] = d["pe"] / d["yoy_c"].where(d["yoy_c"] > 0, np.nan)
    d["peg"] = d["peg"].clip(0, 10)  # winsor

    cols_sign = [
        ("peg",           -1),  # 越小越好：估值相对于增长便宜
        ("pe",            -1),  # 越小越好：绝对估值便宜
        ("chip_conc_20",  +1),  # 越大越好：筹码集中=主力控盘
        ("roe",           +1),  # 越大越好：盈利质量
        ("netprofit_yoy", +1),  # 越大越好：成长性
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
    d["score"] = (d["z_peg"].fillna(0.0)           * 0.35 +
                  d["z_pe"].fillna(0.0)            * 0.20 +
                  d["z_chip_conc_20"].fillna(0.0) * 0.25 +
                  d["z_roe"].fillna(0.0)           * 0.10 +
                  d["z_netprofit_yoy"].fillna(0.0) * 0.10)
    return d

ml_scored = build_scores_v4(ml2)
print(f"[4] 综合分计算完毕, 均值={ml_scored['score'].mean():.3f}, std={ml_scored['score'].std():.3f}")

# ============================================================
# 5. 板块择时信号 (PE分位48月滚动) + 剔除黑名单
# ============================================================
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
print(f"[5] 板块择时信号: {len(sect_signal)}个月 (黑名单板块已剔除)")

# ============================================================
# 6. 个股日频收盘价面板
# ============================================================
print("[6] 构建个股日频收盘面板 (从other_day1)...")
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
    if len(df) == 0:
        continue
    df["dt"] = d
    close_dfs.append(df)
px = pd.concat(close_dfs, ignore_index=True)
print(f"    合并价格行数: {len(px):,}")
close_panel = {}
for code, g in px.groupby("ts_code"):
    s = g.sort_values("dt").set_index("dt")["close"]
    s = s[~s.index.duplicated()]
    close_panel[code] = s.astype(float)
all_dates_set = set()
for s in close_panel.values(): all_dates_set.update(s.index)
all_dates = sorted(all_dates_set)
print(f"    {len(close_panel)}只股票, 交易日 {all_dates[0].date()}~{all_dates[-1].date()} ({len(all_dates)}天)")
del close_dfs, px

# ============================================================
# 7. 回测函数 (v4)
# ============================================================
BUY_FEE = 0.0010
SELL_FEE = 0.0015
INIT = 1_000_000
TP = 0.30
MAX_HOLD = 270
PE_PCT_THR = 0.30  # 低估阈值：PE分位 < 30%

def run_strategy_v4(global_top_k=3,            # 低估板块池内 全局评分TopN
                    max_same_sector=2,          # 最多2只同板块（当k≥3时生效，强制行业分散）
                    max_pe=60,                  # PE上限(0,60] 严估值
                    min_turnover_pct=0.5,       # 月换手率下限%
                    min_circ_mv_yi=50,          # 流通市值下限(亿) - 中大盘起点
                    max_circ_mv_yi=1500,        # 流通市值上限(亿) - 避免超级大象涨不动
                    preferred_weight=1.2,       # 白名单板块额外加分权重（score×1.2）
                    pe_pct_thr=PE_PCT_THR,
                    verbose=False):
    # 月度选股缓存
    ym_to_dt = {}
    for ym_dt in sorted(ml_scored["dt"].unique()):
        ym_int = ym_dt.year * 100 + ym_dt.month
        ym_to_dt[ym_int] = ym_dt

    monthly_picks = {}
    min_circ_wan = min_circ_mv_yi * 10_000
    max_circ_wan = max_circ_mv_yi * 10_000

    for ym_int in yms_unique:
        if ym_int not in sect_signal:
            monthly_picks[ym_int] = []
            continue
        sig = sect_signal[ym_int]
        # 低估板块候选（PE分位<pe_pct_thr），但黑名单已在sect_signal剔除
        undv_secs = [s for s, v in sig.items() if v < pe_pct_thr]
        if not undv_secs:
            monthly_picks[ym_int] = []
            continue
        # 取当月截面
        if ym_int not in ym_to_dt:
            monthly_picks[ym_int] = []
            continue
        dt_real = ym_to_dt[ym_int]
        sub = ml_scored[ml_scored["dt"] == dt_real].copy()
        if len(sub) == 0:
            monthly_picks[ym_int] = []
            continue
        # 只选属于低估板块的股票
        sub["sects"] = sub["ts_code"].map(lambda c: code_info.get(c, {}).get("sectors", []))
        sub["in_undv"] = sub["sects"].apply(lambda lst: any(s in undv_secs for s in lst))
        sub = sub[sub["in_undv"]].copy()
        if len(sub) == 0:
            monthly_picks[ym_int] = []
            continue
        # (1) ST过滤
        sub["st"] = sub["ts_code"].map(lambda c: code_info.get(c, {}).get("is_st", 1))
        sub = sub[sub["st"] == 0]
        # (2) PE (0, max_pe]
        if "pe" in sub.columns:
            mask = (sub["pe"].isna()) | ((sub["pe"] > 0) & (sub["pe"] <= max_pe))
            sub = sub[mask]
        # (3) 换手率
        if "turnover_rate" in sub.columns:
            mask = (sub["turnover_rate"].isna()) | (sub["turnover_rate"] >= min_turnover_pct)
            sub = sub[mask]
        # (4) 中大盘市值区间
        if "circ_mv" in sub.columns:
            mask = (sub["circ_mv"].isna()) | ((sub["circ_mv"] >= min_circ_wan) & (sub["circ_mv"] <= max_circ_wan))
            sub = sub[mask]
        if len(sub) == 0:
            monthly_picks[ym_int] = []
            continue
        # (5) 白名单行业额外加分（鼓励选高胜率板块）
        if preferred_weight != 1.0:
            def is_pref(sects): return any(s in PREFERRED_SECTORS for s in sects)
            sub["_pref"] = sub["sects"].apply(is_pref).astype(int)
            sub["score_adj"] = sub["score"] * (1.0 + sub["_pref"] * (preferred_weight - 1.0))
        else:
            sub["score_adj"] = sub["score"]
        # (6) 按调整后评分 全局排序 取 TopK， 但要满足"最多max_same_sector只同板块"
        sub = sub.sort_values("score_adj", ascending=False)
        chosen = []
        sec_count = {}
        def first_sect(c):
            sects = code_info.get(c, {}).get("sectors", [])
            for s in sects:
                if s in undv_secs: return s
            return sects[0] if sects else "_X_"
        for _, row in sub.iterrows():
            c = row["ts_code"]
            s = first_sect(c)
            # 若加上这只会让某板块超过max_same_sector，则跳过
            if sec_count.get(s, 0) >= max_same_sector and len(chosen) >= global_top_k - 1:
                continue
            chosen.append(c)
            sec_count[s] = sec_count.get(s, 0) + 1
            if len(chosen) >= global_top_k:
                break
        if verbose:
            print(f"  ym={ym_int} 低估板块{undv_secs} -> 选中{chosen}, "
                  f"行业分布={sec_count}, score={[float(sub[sub['ts_code']==c]['score_adj'].iloc[0]) for c in chosen]}")
        monthly_picks[ym_int] = chosen

    # ---------- 日频执行 ----------
    cash = float(INIT)
    holdings = {}
    nav_series = []; trades = []

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
                        trades.append((day, "BUY", c, cost, np.nan, 0))
        # 止盈/止损
        for c in list(holdings.keys()):
            if c not in close_panel or day not in close_panel[c].index:
                continue
            p = close_panel[c].loc[day]
            ret = p / holdings[c]["buy_price"] - 1
            held = di - holdings[c]["buy_di"]
            if ret >= TP or held >= MAX_HOLD:
                proceeds = p * holdings[c]["qty"] * (1 - SELL_FEE)
                cash += proceeds
                op = "TP" if ret >= TP else "T270"
                trades.append((day, op, c, proceeds, ret, held))
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
    return nav_s, trades, monthly_picks


def calc_metrics(nav_s, trades, label):
    if len(nav_s) < 30:
        return {k: np.nan for k in ["年化","累计","回撤","夏普","止盈率","平均持仓天","期末(万)"]}
    tr = nav_s.iloc[-1] / INIT - 1
    yrs = (nav_s.index[-1] - nav_s.index[0]).days / 365.25
    ann = (1 + tr) ** (1/yrs) - 1 if yrs > 0 else np.nan
    pk = nav_s.cummax()
    mdd = ((nav_s - pk) / pk).min()
    ret = nav_s.pct_change().dropna()
    shp = ret.mean() / ret.std(ddof=1) * np.sqrt(252) if len(ret) > 1 and ret.std(ddof=1) > 0 else np.nan
    buys = sum(1 for t in trades if t[1]=="BUY")
    tps = sum(1 for t in trades if t[1]=="TP")
    tls = sum(1 for t in trades if t[1] not in ("BUY","TP"))
    hds = [t[5] for t in trades if t[1] != "BUY"]
    return {"配置":label,"年化":ann,"累计":tr,"回撤":mdd,"夏普":shp,
            "买入":buys,"止盈":tps,"时间止损":tls,
            "止盈率":tps/(buys+1e-9),
            "平均持仓天":np.mean(hds) if hds else 0,
            "期末(万)":nav_s.iloc[-1]/1e4}


# ============================================================
# 8. 组合测试
# ============================================================
print("\n" + "="*95)
print("开始回测 v4 (黑名单过滤 + 低估板块全局TopK评分选股 + 中大盘 50-1500亿 + 行业分散)")
print(f"买入{BUY_FEE:.2%}/卖出{SELL_FEE:.2%}  止盈{TP:.0%}  时间止损{MAX_HOLD}天  PE分位<{PE_PCT_THR:.0%}低估")
print(f"剔除黑名单: {sorted(BLACKLIST_SECTORS)}")
print("="*95)

configs = [
    # 核心对比： Top3/5/10 × PE严/宽 × 白名单加分
    ("A1: 全局Top3+PE≤60+白名单×1.2",    3, 2,  60, 0.5, 50, 1500, 1.20, PE_PCT_THR),
    ("A2: 全局Top3+PE≤60(无白名单加权)",  3, 2,  60, 0.5, 50, 1500, 1.00, PE_PCT_THR),
    ("A3: 全局Top3+PE≤80+白名单×1.2",    3, 2,  80, 0.5, 50, 1500, 1.20, PE_PCT_THR),
    ("B1: 全局Top5+PE≤60+白名单×1.2",    5, 2,  60, 0.5, 50, 1500, 1.20, PE_PCT_THR),
    ("B2: 全局Top5+PE≤80+白名单×1.2",    5, 2,  80, 0.5, 50, 1500, 1.20, PE_PCT_THR),
    ("C1: 全局Top10+PE≤60+白名单×1.2",  10, 4,  60, 0.5, 50, 1500, 1.20, PE_PCT_THR),
    # 放宽市值：50-3000亿，给一些大票机会
    ("D1: Top3+PE≤60+市值50-3000亿",     3, 2,  60, 0.5, 50, 3000, 1.20, PE_PCT_THR),
    # 更严估值：PE≤40
    ("E1: Top3+PE≤40+白名单×1.2",        3, 2,  40, 0.5, 50, 1500, 1.20, PE_PCT_THR),
]

res = []
nav_curves = {}
for i, (lb, k, mss, mpe, mto, mncm, mxcm, pw, pthr) in enumerate(configs):
    t1 = time.time()
    nv, trs, _ = run_strategy_v4(global_top_k=k, max_same_sector=mss, max_pe=mpe,
                                  min_turnover_pct=mto,
                                  min_circ_mv_yi=mncm, max_circ_mv_yi=mxcm,
                                  preferred_weight=pw, pe_pct_thr=pthr)
    m = calc_metrics(nv, trs, lb)
    res.append(m)
    nav_curves[lb] = nv
    print(f"  [{i+1}/{len(configs)}] {lb}: "
          f"年化{m['年化']:.1%} 回撤{m['回撤']:.1%} 夏普{m['夏普']:.2f} "
          f"买入{m['买入']} 止盈{m['止盈']}/止损{m['时间止损']} 止盈率{m['止盈率']:.0%} "
          f"期末{m['期末(万)']:.0f}万  耗时{time.time()-t1:.0f}s")

# 基线
baseline_fund = {"配置":"【基1】板块基金(PE48月+T270)","年化":0.144,"累计":1.43,"回撤":-0.303,"夏普":0.55,
            "买入":132,"止盈":46,"时间止损":83,"止盈率":0.35,"平均持仓天":224,"期末(万)":243.0}
old_v2_glb3 = {"配置":"【基2】旧v2-全局Top3(无行业约束)","年化":0.217,"累计":2.24,"回撤":-0.397,"夏普":0.93,
            "买入":164,"止盈":89,"时间止损":62,"止盈率":0.54,"平均持仓天":142,"期末(万)":323.9}
old_v2_sec3 = {"配置":"【基3】旧v2-板块Top3(分散无黑名)","年化":0.091,"累计":0.69,"回撤":-0.323,"夏普":0.55,
            "买入":959,"止盈":483,"时间止损":435,"止盈率":0.50,"平均持仓天":166,"期末(万)":168.8}
res.extend([baseline_fund, old_v2_glb3, old_v2_sec3])

rdf = pd.DataFrame(res)
print("\n" + "="*120)
print("【汇总 v4】低估板块内全局评分TopK + 行业黑名单过滤 + PEG/筹码/PE评分 + 中大盘(50-1500亿)")
print("="*120)
pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 20)
print(rdf.to_string(index=False, float_format=lambda x: f"{x:.2f}" if not isinstance(x,str) else x))

feasible = rdf[rdf["回撤"] > -0.35]
if len(feasible):
    best = feasible.loc[feasible["夏普"].idxmax()]
    print(f"\n★ 最优(回撤≥-35% 取夏普最高): {best['配置']}")
    print(f"   年化={best['年化']:.1%}  回撤={best['回撤']:.1%}  夏普={best['夏普']:.2f}  期末={best['期末(万)']:.0f}万")
else:
    best = rdf.loc[rdf["夏普"].idxmax()]
    print(f"\n★ 最优(全池夏普最高): {best['配置']} 年化={best['年化']:.1%} 回撤={best['回撤']:.1%} 夏普={best['夏普']:.2f}")

rdf.to_csv(os.path.join(OUT_DIR, "stock_selected_v4.csv"), index=False, encoding="utf-8-sig")

# ============================================================
# 9. 画图
# ============================================================
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams["font.sans-serif"] = ["SimHei","Microsoft YaHei","Arial Unicode MS"]
rcParams["axes.unicode_minus"] = False

# 净值曲线：选前4个新策略
fig, ax = plt.subplots(figsize=(14, 7.2))
styles = ["-", "-.", "--", ":", "-.", "--", ":"]
best_labels = [configs[0][0], configs[1][0], configs[3][0], configs[6][0], configs[7][0]]
for j, lb in enumerate(best_labels):
    if lb not in nav_curves: continue
    s = nav_curves[lb]
    if len(s) == 0: continue
    s_norm = s / s.iloc[0]
    ax.plot(s.index, s_norm.values, label=f"{lb} (期末={s.iloc[-1]/1e4:.0f}万)",
            linestyle=styles[j % len(styles)], linewidth=2.0 if j == 0 else 1.5)
# 加基线水平线标注
ax.axhline(1.0, color="gray", lw=0.6, ls=":")
ax.axhline(2.43, color="red", lw=0.8, ls="--", label="基1 板块基金 期末243万")
ax.axhline(3.24, color="purple", lw=0.8, ls="--", label="基2 全局Top3 期末324万")
ax.set_title("v4 净值曲线对比 (初始100万)\n"
             "低估板块全局评分TopK + 黑名单(银行地产公用基建议剔除) + PEG/筹码/PE + 中大盘50-1500亿", fontsize=13)
ax.set_xlabel("日期")
ax.set_ylabel("净值 (初始=1)")
ax.legend(loc="upper left", fontsize=9)
ax.grid(True, alpha=0.3)
plt.tight_layout()
png_path = os.path.join(OUT_DIR, "stock_selected_v4_curve.png")
plt.savefig(png_path, dpi=130)
plt.close()
print(f"\n净值曲线图: {png_path}")

# 回撤曲线
fig, ax2 = plt.subplots(figsize=(14, 5.2))
for j, lb in enumerate(best_labels):
    if lb not in nav_curves: continue
    s = nav_curves[lb]
    if len(s) == 0: continue
    pk = s.cummax()
    dd = (s - pk) / pk
    ax2.fill_between(dd.index, dd.values, 0, alpha=0.14)
    ax2.plot(dd.index, dd.values, lw=1.2, label=f"{lb} (最大回撤 {dd.min():.1%})")
ax2.axhline(0, color="gray", lw=0.6)
ax2.axhline(-0.30, color="red", lw=0.8, ls="--", label="基1 板块基金 回撤-30%")
ax2.set_title("回撤曲线 (Underwater Plot)", fontsize=13)
ax2.set_ylabel("回撤幅度")
ax2.legend(loc="lower left", fontsize=9)
ax2.grid(True, alpha=0.3)
plt.tight_layout()
dd_path = os.path.join(OUT_DIR, "stock_selected_v4_drawdown.png")
plt.savefig(dd_path, dpi=130)
plt.close()
print(f"回撤曲线图: {dd_path}")

# ============================================================
# 10. 最优策略的样例持股展示（打印最近几次调仓记录）
# ============================================================
print(f"\n总耗时 {time.time()-t00:.0f}s")

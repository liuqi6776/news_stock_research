# -*- coding: utf-8 -*-
"""
板块择时 + 大盘个股精选 回测 v6 （v5惨败修正版）
================================================
v5失败原因诊断：
  - ROE 覆盖率仅 17%，netprofit_yoy 覆盖率也只有 17%
  - v5用ROE≥10%硬门槛 → 83%的股票（含优质股，只是财务数据没入库）被直接T出
  - PEG硬过滤同样依赖netprofit_yoy → 候选池骤减
  - 结果：很多月份选不出3只股票，仓位踏空上涨 → 收益暴跌，回撤反而更大

v6修正理念（保留用户PEG/ROE/筹码/ST要求，但不硬切）：
  ✅ ST/退市股：严格硬过滤（覆盖率100%，名称自带）
  ✅ 次新股（不满2年）：硬过滤（list_date覆盖率足够）
  ✅ PE硬区间 (0, 80]：硬过滤（PE覆盖率90%）
  ✅ 大盘市值 50~2000亿：硬过滤（circ_mv覆盖率100%）
  ✅ 筹码集中度：只留截面前70%（chip_conc覆盖率100%，较v5的60%略放宽）
  ⚠️ PEG / ROE / 净利增长：SOFT软过滤（用截面中位数回补缺失值后参加评分，不做硬剔除！）
  ⚠️ 评分中PEG和ROE仍然给最高权重，但缺失不影响资格
  🎯 目标：≥v4-A3的收益（年化35.5%，期末619万）同时回撤更好（<-29.5%）
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
# 0. 行业黑白名单
# ============================================================
BLACKLIST_SECTORS = {"银行", "证券保险", "地产", "公用事业", "基建"}
PREFERRED_SECTORS = {
    "煤炭", "石油石化", "钢铁", "有色金属", "化工",
    "电力", "新能源", "半导体芯片", "电子",
    "汽车", "机械", "建材", "家电", "农业",
    "医药医疗", "白酒消费", "计算机软件", "通信", "军工",
    "交通运输", "环保", "纺织服装", "传媒", "人工智能",
}

# ============================================================
# 1. 基础映射 + 严格ST/退市过滤
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
# ===== 严格ST/退市过滤（硬过滤，覆盖率100%）=====
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
print(f"[1] 可映射股票(黑名单剔除后): {len(code_info)}只, "
      f"严格ST/退市风险: {sum(1 for v in code_info.values() if v['is_st'])}只 (硬剔除)")

# ============================================================
# 2. ML面板
# ============================================================
ml = pd.read_parquet(ML_PANEL)
ml["dt"] = pd.to_datetime(ml["trade_date"].astype(str), format="%Y%m%d")
ml["ym"] = ml["dt"].dt.year * 100 + ml["dt"].dt.month
print(f"[2] ML面板: {len(ml):,}行, {ml['ts_code'].nunique()}只, {ml['ym'].nunique()}月 "
      f"({ml['dt'].min().date()}~{ml['dt'].max().date()})")

# ============================================================
# 3. 估值/市值月度快照
# ============================================================
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
start_dt, end_dt = ml["dt"].min(), ml["dt"].max()
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
# 4. v6 综合选股分
#    关键：低覆盖率指标（PEG/ROE/净利增长）只做软评分，不做硬过滤
#          先用截面中位数回补缺失值，再z-score标准化 + 加权
# ============================================================
def build_scores_v6(df):
    d = df.copy()
    # Step1: 截面中位数回补缺失值（PEG依赖的指标都要补）
    # 注意：只有ROE和netprofit_yoy覆盖率只有17%
    for col in ["pe", "pb", "roe", "netprofit_yoy", "chip_conc_20"]:
        if col in d.columns:
            d[col] = d.groupby("ym")[col].transform(lambda x: x.fillna(x.median()))
    # Step2: PEG = PE / 净利润同比增速（增速已补中位数，所以PEG不会大面积缺失）
    d["yoy_c"] = d["netprofit_yoy"].clip(lower=5, upper=100)
    d["peg"] = d["pe"] / d["yoy_c"].where(d["yoy_c"] > 0, np.nan)
    # PEG补截面中位数（确保不因为PEG缺失而踢出好股票）
    d["peg"] = d.groupby("ym")["peg"].transform(lambda x: x.fillna(x.median()))
    d["peg"] = d["peg"].clip(0, 10)
    if "pb" in d.columns:
        d["pb"] = d["pb"].clip(0.3, 20)
    # Step3: z-score标准化（月度截面）
    cols_sign = [
        ("peg",           -1),  # 越小越好：估值-增长匹配（核心权重40%）
        ("chip_conc_20",  +1),  # 越大越好：筹码集中（权重25%，覆盖率100%）
        ("roe",           +1),  # 越大越好：盈利质量（权重15%，软指标）
        ("pe",            -1),  # 越小越好：绝对估值便宜（权重10%，覆盖率90%）
        ("netprofit_yoy", +1),  # 越大越好：成长性（权重10%，软指标）
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
    # Step4: 加权
    d["score"] = (d["z_peg"].fillna(0.0)           * 0.40 +
                  d["z_chip_conc_20"].fillna(0.0) * 0.25 +
                  d["z_roe"].fillna(0.0)           * 0.15 +
                  d["z_pe"].fillna(0.0)            * 0.10 +
                  d["z_netprofit_yoy"].fillna(0.0) * 0.10)
    return d

ml_scored = build_scores_v6(ml2)
print(f"[4] v6综合分计算完毕, 均值={ml_scored['score'].mean():.3f}, std={ml_scored['score'].std():.3f}")

# ============================================================
# 5. 板块择时信号
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
print(f"[5] 板块择时信号: {len(sect_signal)}个月")

# ============================================================
# 6. 日频收盘价面板
# ============================================================
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
print(f"    合并价格行数: {len(px):,}")
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

# ============================================================
# 7. v6 回测函数（关键：硬过滤只用高覆盖率指标）
# ============================================================
BUY_FEE = 0.0010
SELL_FEE = 0.0015
INIT = 1_000_000
TP = 0.30
MAX_HOLD = 270
PE_PCT_THR = 0.30

def run_strategy_v6(global_top_k=3,
                    max_same_sector=2,
                    max_pe=80,                    # PE硬上限(0,80]，v4-A3最佳用80
                    min_turnover_pct=0.3,         # 月换手率下限%
                    min_circ_mv_yi=50,            # 大盘股下限(亿)，v4=50证明有效
                    max_circ_mv_yi=2000,          # 大盘股上限(亿)
                    chip_conc_pctl_threshold=0.70, # 筹码集中度只留前70%（100%覆盖→硬过滤OK）
                    min_list_years=1.5,           # 上市满1.5年（v5设2年太严，次新股过滤适度）
                    peg_bonus_threshold=1.5,      # PEG<1.5额外加分（软指标，不硬切）
                    preferred_weight=1.2,
                    pe_pct_thr=PE_PCT_THR,
                    verbose=False):
    # ===== 前视修复：T月数据选股 → T+1月首个交易日执行 =====
    # 原版用当月月末截面选股并在当月月初买入，带入了整月未来信息。
    # 修复：选股结果全部存到【下个月】的 key，执行时仍按当月 key 买入。
    def _next_month(ym_int):
        y, m = ym_int // 100, ym_int % 100
        return (y + 1) * 100 + 1 if m == 12 else y * 100 + m + 1

    ym_to_dt = {}
    for ym_dt in sorted(ml_scored["dt"].unique()):
        ym_int = ym_dt.year * 100 + ym_dt.month
        ym_to_dt[ym_int] = ym_dt
    monthly_picks = {}
    min_circ_wan = min_circ_mv_yi * 10_000
    max_circ_wan = max_circ_mv_yi * 10_000

    for ym_int in yms_unique:
        if ym_int not in sect_signal:
            monthly_picks[_next_month(ym_int)] = []; continue
        sig = sect_signal[ym_int]
        undv_secs = [s for s, v in sig.items() if v < pe_pct_thr]
        if not undv_secs:
            monthly_picks[_next_month(ym_int)] = []; continue
        if ym_int not in ym_to_dt:
            monthly_picks[_next_month(ym_int)] = []; continue
        dt_real = ym_to_dt[ym_int]
        sub = ml_scored[ml_scored["dt"] == dt_real].copy()
        if len(sub) == 0:
            monthly_picks[_next_month(ym_int)] = []; continue
        sub["sects"] = sub["ts_code"].map(lambda c: code_info.get(c, {}).get("sectors", []))
        sub["in_undv"] = sub["sects"].apply(lambda lst: any(s in undv_secs for s in lst))
        sub = sub[sub["in_undv"]].copy()
        if len(sub) == 0:
            monthly_picks[_next_month(ym_int)] = []; continue

        N0 = len(sub)
        # ====== 以下为【硬过滤】（只挑覆盖率高的指标） ======
        # (1) 严格ST/退市风险：覆盖率100%（硬剔除）
        sub["st"] = sub["ts_code"].map(lambda c: code_info.get(c, {}).get("is_st", 1))
        sub = sub[sub["st"] == 0]

        # (2) 上市不满min_list_years年剔除（list_date覆盖率虽非100%但数据尚可）
        #     list_date缺失的谨慎保留（不T出，避免误伤）
        def days_since_listed(c, d_day):
            ld = code_info.get(c, {}).get("list_date", None)
            if ld is None or pd.isna(ld): return min_list_years * 365 + 1  # 缺失=当作合格
            return (d_day - ld).days
        sub["list_days"] = sub["ts_code"].apply(lambda c: days_since_listed(c, dt_real))
        sub = sub[sub["list_days"] >= min_list_years * 365]
        N1 = len(sub)

        # (3) PE硬区间 (0, max_pe]：覆盖率90%，缺失值谨慎保留
        if "pe" in sub.columns:
            mask = sub["pe"].isna() | ((sub["pe"] > 0) & (sub["pe"] <= max_pe))
            sub = sub[mask]

        # (4) 筹码集中度前chip_conc_pctl_threshold%：覆盖率100%，可硬过滤
        if "chip_conc_20" in sub.columns and len(sub) >= 5:
            thr = sub["chip_conc_20"].quantile(1.0 - chip_conc_pctl_threshold)
            mask = sub["chip_conc_20"].isna() | (sub["chip_conc_20"] >= thr)
            sub = sub[mask]
        N2 = len(sub)

        # (5) 换手率：覆盖率100%
        if "turnover_rate" in sub.columns:
            mask = sub["turnover_rate"].isna() | (sub["turnover_rate"] >= min_turnover_pct)
            sub = sub[mask]

        # (6) 大盘股市值：覆盖率100%
        if "circ_mv" in sub.columns:
            mask = sub["circ_mv"].isna() | ((sub["circ_mv"] >= min_circ_wan) & (sub["circ_mv"] <= max_circ_wan))
            sub = sub[mask]
        N3 = len(sub)

        if verbose and len(sub) > 0:
            print(f"  ym={ym_int} 硬过滤: N0={N0}→次新后N1={N1}→筹码后N2={N2}→大盘后N3={N3}")
        if len(sub) == 0:
            monthly_picks[_next_month(ym_int)] = []; continue

        # ====== 以下为【软加分】（PEG/ROE不靠硬过滤，靠加分减分体现偏好） ======
        sub["_peg_bonus"] = 0.0
        if "peg" in sub.columns:
            # PEG < peg_bonus_threshold → 额外加0.5分（相当于PEG排名额外前半个标准差）
            sub.loc[sub["peg"] < peg_bonus_threshold, "_peg_bonus"] = 0.5
            # PEG > 3 → 惩罚0.3分（不硬T，但排名靠后）
            sub.loc[sub["peg"] > 3, "_peg_bonus"] = -0.3

        # ROE高的加分（不硬门槛）
        sub["_roe_bonus"] = 0.0
        if "roe" in sub.columns:
            sub.loc[sub["roe"] >= 15, "_roe_bonus"] = 0.3   # ROE≥15% 优秀 → 加
            sub.loc[(sub["roe"] < 5) & (sub["roe"] > 0), "_roe_bonus"] = -0.2  # ROE<5% → 减

        # 白名单行业加分
        if preferred_weight != 1.0:
            def is_pref(sects): return any(s in PREFERRED_SECTORS for s in sects)
            sub["_pref"] = sub["sects"].apply(is_pref).astype(int)
            sub["score_adj"] = (sub["score"] + sub["_peg_bonus"] + sub["_roe_bonus"]) * \
                               (1.0 + sub["_pref"] * (preferred_weight - 1.0))
        else:
            sub["score_adj"] = sub["score"] + sub["_peg_bonus"] + sub["_roe_bonus"]

        # ====== 全局TopK + 行业分散约束 ======
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
            # TopK≥3时：最多max_same_sector只来自同一板块
            if sec_count.get(s, 0) >= max_same_sector and len(chosen) >= global_top_k - 1:
                continue
            chosen.append(c)
            sec_count[s] = sec_count.get(s, 0) + 1
            if len(chosen) >= global_top_k:
                break
        if verbose:
            def get_name(c): return code_info.get(c, {}).get("name", c)
            names = [get_name(c) for c in chosen]
            print(f"    → 选中={names}, 行业分布={sec_count}")
        monthly_picks[_next_month(ym_int)] = chosen

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
        # 止盈30% / 时间止损270天
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
# 8. v6 组合测试
# ============================================================
print("\n" + "="*100)
print("开始回测 v6 (ROE/PEG软过滤+ST硬剔除+大盘50-2000亿+筹码前70%+上市满1.5年)")
print(f"买入{BUY_FEE:.2%}/卖出{SELL_FEE:.2%}  止盈{TP:.0%}  时间止损{MAX_HOLD}天  PE分位<{PE_PCT_THR:.0%}低估")
print(f"评分权重：PEG 40% + 筹码 25% + ROE 15% + PE 10% + 净利同比 10%")
print(f"软加分：PEG<1.5 +0.5 / PEG>3 -0.3 / ROE≥15% +0.3 / ROE<5% -0.2")
print("="*100)

# 关键变量：对标v4-A3最佳(Top3+PE≤80+50-1500亿+白名单×1.2)
configs = [
    # ===== 对标v4-A3的v6基准版（同样Top3+PE≤80+50-2000亿，但是加了ST/PEG/ROE/筹码加强） =====
    ("V6-S1: Top3+PE≤80+大盘50亿【对标v4-A3】",
        3, 2, 80, 0.3, 50,  2000, 0.70, 1.5, 1.5, 1.20),
    # 同样配置，大盘更纯（100亿起）
    ("V6-S2: Top3+PE≤80+大盘100亿【纯大盘版】",
        3, 2, 80, 0.3, 100, 2000, 0.70, 1.5, 1.5, 1.20),
    # 加大PEG软过滤强度（PEG<1.2才加分）
    ("V6-P1: Top3+PE≤80+PEG优<1.2+大盘50亿",
        3, 2, 80, 0.3, 50,  2000, 0.70, 1.2, 1.5, 1.20),
    # 更严筹码（只留前60%）
    ("V6-C1: Top3+PE≤80+筹码前60%+大盘50亿",
        3, 2, 80, 0.3, 50,  2000, 0.60, 1.5, 1.5, 1.20),
    # 更严PE区间(0,60]
    ("V6-E1: Top3+PE≤60+大盘50亿",
        3, 2, 60, 0.3, 50,  2000, 0.70, 1.5, 1.5, 1.20),
    # 更严次新过滤(上市满3年)
    ("V6-L1: Top3+PE≤80+上市满3年+大盘50亿",
        3, 2, 80, 0.3, 50,  2000, 0.70, 1.5, 3.0, 1.20),
    # Top5分散版
    ("V6-K1: Top5+PE≤80+大盘50亿【分散版】",
        5, 2, 80, 0.3, 50,  2000, 0.70, 1.5, 1.5, 1.20),
    # 白名单不加权(1.0)
    ("V6-W1: Top3+PE≤80+大盘50亿【白名单不加权】",
        3, 2, 80, 0.3, 50,  2000, 0.70, 1.5, 1.5, 1.00),
    # 最强版：Top3+大盘50亿+最严筹码前60%+严次新2年+严PE≤70
    ("V6-X1: Top3+强过滤【筹码前60%+PE≤70+上市满2年】",
        3, 2, 70, 0.3, 50,  2000, 0.60, 1.5, 2.0, 1.20),
]

res = []
nav_curves = {}
for i, cfg in enumerate(configs):
    lb, k, mss, mpe, mto, mncm, mxcm, cc_thr, peg_thr, lyrs, pw = cfg
    t1 = time.time()
    try:
        nv, trs, _ = run_strategy_v6(
            global_top_k=k, max_same_sector=mss,
            max_pe=mpe, min_turnover_pct=mto,
            min_circ_mv_yi=mncm, max_circ_mv_yi=mxcm,
            chip_conc_pctl_threshold=cc_thr,
            peg_bonus_threshold=peg_thr,
            min_list_years=lyrs,
            preferred_weight=pw,
            pe_pct_thr=PE_PCT_THR)
        m = calc_metrics(nv, trs, lb)
    except Exception as e:
        print(f"  [{i+1}/{len(configs)}] {lb}: ERROR {e}")
        continue
    res.append(m)
    nav_curves[lb] = nv
    print(f"  [{i+1}/{len(configs)}] {lb}: "
          f"年化{m['年化']:.1%} 回撤{m['回撤']:.1%} 夏普{m['夏普']:.2f} "
          f"买入{m['买入']} 止盈{m['止盈']}/止损{m['时间止损']} 止盈率{m['止盈率']:.0%} "
          f"期末{m['期末(万)']:.0f}万  耗时{time.time()-t1:.0f}s")

# 基线
baselines = [
    {"配置":"【v4基-A3★】Top3+PE≤80+白名单×1.2","年化":0.355,"累计":5.19,"回撤":-0.295,"夏普":1.41,
     "买入":144,"止盈":90,"时间止损":44,"止盈率":0.625,"平均持仓天":132.0,"期末(万)":619.3},
    {"配置":"【板块基】板块基金(PE48月+T270)","年化":0.144,"累计":1.43,"回撤":-0.303,"夏普":0.55,
     "买入":132,"止盈":46,"时间止损":83,"止盈率":0.35,"平均持仓天":224.0,"期末(万)":243.0},
    {"配置":"【持有基】中证1000 512100不动","年化":0.109,"累计":0.97,"回撤":-0.600,"夏普":0.38,
     "买入":0,"止盈":0,"时间止损":0,"止盈率":0.0,"平均持仓天":0.0,"期末(万)":197.0},
]
res.extend(baselines)

rdf = pd.DataFrame(res)
print("\n" + "="*145)
print("【汇总 v6 vs v4】软过滤(PEG/ROE/增长) + 硬过滤(ST/PE/大盘/筹码/次新)")
print("评分: PEG 40% + 筹码 25% + ROE 15% + PE 10% + 净利同比 10%  |  软加分: PEG<1.5+0.5 / ROE≥15%+0.3")
print("="*145)
pd.set_option("display.width", 260)
pd.set_option("display.max_columns", 20)
print(rdf.to_string(index=False, float_format=lambda x: f"{x:.2f}" if not isinstance(x,str) else x))

feasible = rdf[rdf["回撤"] > -0.30]
if len(feasible):
    best = feasible.loc[feasible["夏普"].idxmax()]
    print(f"\n★ 最优(回撤≥-30% 取夏普最高): {best['配置']}")
    print(f"   年化={best['年化']:.1%}  回撤={best['回撤']:.1%}  夏普={best['夏普']:.2f}  期末={best['期末(万)']:.0f}万")
    v4_ann, v4_mdd, v4_shp, v4_end = 0.355, -0.295, 1.41, 619.3
    print(f"   vs v4-A3: 年化 Δ={best['年化']-v4_ann:+.1%}, 回撤 Δ={best['回撤']-v4_mdd:+.1%}, "
          f"夏普 Δ={best['夏普']-v4_shp:+.2f}, 期末 Δ={best['期末(万)']-v4_end:+.0f}万")
    if best['年化'] >= v4_ann and best['回撤'] > v4_mdd:
        print(f"   ✅ 优于v4-A3！【收益更高 + 回撤更低】")
    elif best['年化'] >= v4_ann * 0.95 and best['回撤'] > v4_mdd + 0.02:
        print(f"   ✅ 接近v4-A3收益 + 回撤显著更低，整体更优")
else:
    best = rdf.loc[rdf["夏普"].idxmax()]
    print(f"\n★ 最优(全池夏普最高): {best['配置']}")
    print(f"   年化={best['年化']:.1%}  回撤={best['回撤']:.1%}  夏普={best['夏普']:.2f}  期末={best['期末(万)']:.0f}万")

rdf.to_csv(os.path.join(OUT_DIR, "stock_selected_v6.csv"), index=False, encoding="utf-8-sig")

# ============================================================
# 9. 画图
# ============================================================
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams["font.sans-serif"] = ["SimHei","Microsoft YaHei","Arial Unicode MS"]
rcParams["axes.unicode_minus"] = False

fig, ax = plt.subplots(figsize=(15, 7.5))
styles = ["-", "-.", "--", ":", "-.", "--", ":", "-", "-."]
show_labels = [c[0] for c in configs[:6]]
show_labels.append("【v4基-A3★】Top3+PE≤80+白名单×1.2")
show_labels.append("【板块基】板块基金(PE48月+T270)")
for j, lb in enumerate(show_labels):
    if lb in nav_curves:
        s = nav_curves[lb]
    else:
        continue
    if len(s) == 0: continue
    s_norm = s / s.iloc[0]
    ax.plot(s.index, s_norm.values, label=f"{lb} (期末={s.iloc[-1]/1e4:.0f}万)",
            linestyle=styles[j % len(styles)], linewidth=2.0 if j == 0 else 1.5)
ax.axhline(1.0, color="gray", lw=0.6, ls=":")
ax.axhline(6.19, color="orange", lw=1.0, ls="--", label="v4-A3★ 期末619万 (年化35.5% / 回撤-29.5% / 夏普1.41)")
ax.axhline(2.43, color="red", lw=1.0, ls="--", label="板块基金基线 期末243万 (年化14.4%)")
ax.set_title("v6 vs v4 净值曲线对比 (初始100万)\n"
             "软过滤(PEG/ROE缺失补中位数不硬切) + 硬过滤(ST退市/PE区间/大盘市值/筹码集中度/次新)\n"
             "评分: PEG 40% + 筹码 25% + ROE 15% + PE 10% + 增长 10%", fontsize=13)
ax.set_xlabel("日期")
ax.set_ylabel("净值 (初始=1)")
ax.legend(loc="upper left", fontsize=8.5)
ax.grid(True, alpha=0.3)
plt.tight_layout()
png_path = os.path.join(OUT_DIR, "stock_selected_v6_curve.png")
plt.savefig(png_path, dpi=130)
plt.close()
print(f"\n净值曲线图: {png_path}")

fig, ax2 = plt.subplots(figsize=(15, 5.5))
for j, lb in enumerate(show_labels[:6]):
    if lb not in nav_curves: continue
    s = nav_curves[lb]
    if len(s) == 0: continue
    pk = s.cummax()
    dd = (s - pk) / pk
    ax2.fill_between(dd.index, dd.values, 0, alpha=0.10)
    ax2.plot(dd.index, dd.values, lw=1.2, label=f"{lb} (最大回撤 {dd.min():.1%})")
ax2.axhline(0, color="gray", lw=0.6)
ax2.axhline(-0.295, color="orange", lw=1.0, ls="--", label="v4-A3★ 最大回撤 -29.5%")
ax2.axhline(-0.303, color="red", lw=1.0, ls="--", label="板块基金回撤 -30.3%")
ax2.set_title("回撤曲线对比 (Underwater Plot)", fontsize=13)
ax2.set_ylabel("回撤幅度")
ax2.legend(loc="lower left", fontsize=8.5)
ax2.grid(True, alpha=0.3)
plt.tight_layout()
dd_path = os.path.join(OUT_DIR, "stock_selected_v6_drawdown.png")
plt.savefig(dd_path, dpi=130)
plt.close()
print(f"回撤曲线图: {dd_path}")

print(f"\n总耗时 {time.time()-t00:.0f}s")
print("\n【v6策略设计理念（从v5惨败中吸取的教训）】")
print("  ❌ v5失败：ROE仅17%覆盖率用硬门槛≥10% → 83%股票被踢，仓位严重踏空")
print("  ✅ v6修正：区分'高覆盖率指标'和'低覆盖率指标'，策略不同：")
print("     • 高覆盖率(90-100%) → 硬过滤：ST退市/PE区间/大盘股市值/筹码集中度/次新股")
print("     • 低覆盖率(17%) → 软过滤+截面缺失值回补：PEG / ROE / 净利同比增长")
print("  ✅ 用户核心诉求全部覆盖：")
print("     ✓ PEG估值匹配（核心权重40%，<1.5加分，>3惩罚）")
print("     ✓ PE便宜（权重10% + (0,80]硬区间）")
print("     ✓ ROE盈利质量（权重15%，≥15%加分，<5%惩罚）")
print("     ✓ 筹码集中度（权重25% + 只留前70%硬过滤）")
print("     ✓ 严格ST/退市风险股过滤（硬剔除）")
print("     ✓ 大盘股市值（50-2000亿硬区间）")

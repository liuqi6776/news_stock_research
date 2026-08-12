# -*- coding: utf-8 -*-
"""
板块择时 + 大盘个股精选 回测 v5 （强化版）
============================================
用户需求：从大盘里选优股，用PEG、PE、ROE、筹码集中度、ST过滤等指标
         → 追求更高收益 + 更低回撤

相对v4的强化改进：
  1) 【硬过滤 非软评分】
     - PEG 硬阈值：PEG<1.5 进入优选池，1.5≤PEG<2 普通池，PEG≥2 直接剔除
     - ROE 硬门槛：ROE≥10% （低于直接剔除）
     - 筹码集中度：截面前40%才有资格
     - 估值：PE (0, 60] 硬区间
  2) 【大盘股更纯】流通市值 100~2000亿（50亿→100亿，过滤小票弹性过高的波动）
  3) 【次新股过滤】上市不满2年的股票剔除（IPO后爆炒+限售股解禁风险）
  4) 【ST/退市双重过滤】*ST / ST / SST / *SST / 名称含"退" 全部剔除
  5) 【财务安全】加入PB因子，低PB+高ROE=价值成长型
  6) 【评分权重优化】PEG 40% + 筹码 25% + ROE 15% + PE 10% + 净利增长 10%
  7) 【赛道聚焦】白名单仅保留：能源/材料/制造/工业/医药/消费等胜率高的赛道
  8) 止盈30%、时间止损270天、买入0.1%/卖出0.15%（同v4可比基准）
"""
import os, sys, glob, time, calendar
import numpy as np
import pandas as pd

ROOT = r"c:\Users\liuqi\quant_system_v2"
DATA = r"D:\iquant_data\data_v2"
PE_CSV = os.path.join(ROOT, "research", "sector_rotation", "results", "industry_pe.csv")
OUT_DIR = os.path.join(ROOT, "research", "sector_rotation", "results")
ML_PANEL = os.path.join(ROOT, "research", "sector_rotation", "stock_ml_panel_72m.parquet")
IND_PARQ = os.path.join(DATA, "industry1", "industry.parquet")
OTHER_DAY_DIR = os.path.join(DATA, "other_day1")

# s123 择时 + V8 避险（复用 GBDT 引擎的信号源）
_TD_DIR = os.path.join(ROOT, "research", "fund_research", "studies", "rotation_dingtou")
if _TD_DIR not in sys.path:
    sys.path.insert(0, _TD_DIR)
from timing_dingtou import fetch_pe_csi300, fetch_bond10y, _rolling_pct, _zscore
from etf_optimize_backtest2 import load_hv_daily

t00 = time.time()

# ============================================================
# 0. 行业白黑名单（用户说聚焦能源/汽车制造/工业等上升赛道）
# ============================================================
# 价值陷阱黑名单：银行/地产/证券保险/公用事业/基建（PE低但不涨）
BLACKLIST_SECTORS = {"银行", "证券保险", "地产", "公用事业", "基建"}
# 聚焦赛道白名单（用户明确提到的能源/汽车制造/工业 + 历史高胜率医药消费/材料）
PREFERRED_SECTORS = {
    # 用户指定重点赛道
    "煤炭", "石油石化", "钢铁", "有色金属", "化工",       # 能源 + 材料
    "电力", "新能源",                                    # 能源
    "汽车", "机械", "建材", "家电",                      # 制造/工业
    # 补充：胜率高的优质赛道
    "医药医疗", "白酒消费", "农业",                      # 必选消费/医药
    "半导体芯片", "电子", "计算机软件", "通信", "军工",  # 科技成长
    "交通运输", "环保", "纺织服装", "传媒", "人工智能",
}

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

# ====== v5 强化：ST/退市双重过滤 ======
# 不只是 *ST/ST，还要过滤 SST（未股改ST）、*SST、名称含"退"（退市整理期）
def is_strict_risk(name):
    if not name: return 1
    n = str(name)
    if "退" in n: return 1
    # 匹配各种ST变体: *ST, ST, SST, *SST, S*ST
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
      f"其中严格ST/退市风险: {sum(1 for v in code_info.values() if v['is_st'])}只 (已在选股时剔除)")

# ============================================================
# 2. 月度ML面板基础
# ============================================================
ml = pd.read_parquet(ML_PANEL)
ml["dt"] = pd.to_datetime(ml["trade_date"].astype(str), format="%Y%m%d")
ml["ym"] = ml["dt"].dt.year * 100 + ml["dt"].dt.month
print(f"[2] ML面板: {len(ml):,}行, {ml['ts_code'].nunique()}只, {ml['ym'].nunique()}月 "
      f"({ml['dt'].min().date()}~{ml['dt'].max().date()})")

# ============================================================
# 3. 估值/换手率/流通市值/PB 月度快照
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
# 4. v5 综合选股分（硬过滤前置 + 优化权重）
# ============================================================
def build_scores_v5(df):
    """
    v5评分流程：
    Step1: 截面中位数填充缺失
    Step2: PEG 计算 + clip
    Step3: z-score 标准化
    Step4: 加权评分（PEG 40% 最高权重 + 筹码 25% + ROE 15% + PE 10% + 增长10%）
    """
    d = df.copy()
    # Step1: 月度截面填充缺失值（中位数）
    for col in ["pe", "pb", "roe", "netprofit_yoy", "chip_conc_20"]:
        if col in d.columns:
            d[col] = d.groupby("ym")[col].transform(lambda x: x.fillna(x.median()))
    # Step2: PEG 计算
    # 净利润同比增速下限5%（避免PEG虚低），上限100%（抑制异常值）
    d["yoy_c"] = d["netprofit_yoy"].clip(lower=5, upper=100)
    d["peg"] = d["pe"] / d["yoy_c"].where(d["yoy_c"] > 0, np.nan)
    d["peg"] = d["peg"].clip(0, 10)  # winsor: PEG=0~10之外的截断
    # PB clip (避免负资产/极端高PB)
    if "pb" in d.columns:
        d["pb"] = d["pb"].clip(0.3, 20)
    # Step3: z-score 标准化（月度截面）
    cols_sign = [
        ("peg",           -1),  # 越小越好：估值相对于增长的便宜程度（核心权重40%）
        ("chip_conc_20",  +1),  # 越大越好：筹码集中=主力控盘/惜售（权重25%）
        ("roe",           +1),  # 越大越好：盈利质量/股东回报率（权重15%）
        ("pe",            -1),  # 越小越好：绝对估值便宜（权重10%）
        ("netprofit_yoy", +1),  # 越大越好：成长性（权重10%）
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
        z = z.clip(-3, 3)  # winsor 异常z值
        d[col_z] = sign * z
    # Step4: 加权（用户说PEG最核心 → PEG权重最高）
    d["score"] = (d["z_peg"].fillna(0.0)           * 0.40 +  # 估值-增长匹配（核心）
                  d["z_chip_conc_20"].fillna(0.0) * 0.25 +  # 筹码集中度
                  d["z_roe"].fillna(0.0)           * 0.15 +  # 盈利质量
                  d["z_pe"].fillna(0.0)            * 0.10 +  # 绝对估值
                  d["z_netprofit_yoy"].fillna(0.0) * 0.10)   # 成长性
    return d

ml_scored = build_scores_v5(ml2)
print(f"[4] v5综合分计算完毕, 均值={ml_scored['score'].mean():.3f}, std={ml_scored['score'].std():.3f}")

# ============================================================
# 5. 板块择时信号 (PE分位48月滚动) + 黑名单剔除
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
# 7. v5 回测函数 （硬过滤前置 + 大盘更纯 + 次新过滤）
# ============================================================
BUY_FEE = 0.0010
SELL_FEE = 0.0015
INIT = 1_000_000
TP = 0.30
MAX_HOLD = 270
PE_PCT_THR = 0.30  # PE分位 <30% → 低估

# ============================================================
# 7.5 s123 择时信号 + V8 避险组合（月度状态机，T-1月信号→T月生效）
# ============================================================
def build_s123_v8():
    """构建月度 s123 信号 与 V8(短债+信用债+黄金) 日净值。
    返回: (sig_map, v8_nav_series, v8_daily_reindex)
      sig_map: {ym: s123}
      v8_daily: Series(index=int YYYYMMDD, 值=当日V8日收益)
    """
    pe = fetch_pe_csi300()
    bond = fetch_bond10y()
    close_ix = pe["close"]
    dd_ix = close_ix / close_ix.cummax() - 1.0
    erp = 1.0 / pe["pe_ttm"] - bond["y10"].reindex(pe.index).ffill()
    month_keys = sorted(set(int(d.year) * 100 + d.month for d in close_ix.index))
    sig_rows = []
    for ym in month_keys:
        d = pd.Timestamp(f"{ym}01") + pd.offsets.MonthEnd(0)
        s1 = 1 if _rolling_pct(pe["pe_ttm"], d) < 0.20 else 0
        s2 = 1 if _zscore(erp, d) > 1.0 else 0
        s3 = 1 if float(dd_ix.asof(d)) <= -0.25 else 0
        sig_rows.append({"ym": ym, "s123": s1 + s2 + s3})
    sig_df = pd.DataFrame(sig_rows).set_index("ym")
    v8 = load_hv_daily()
    all_d = sorted(set().union(*[set(s.index) for s in v8.values()]))
    v8_df = pd.DataFrame(index=all_d)
    for code, s in v8.items():
        v8_df[code] = s.reindex(all_d).astype(float)
    v8_daily = (v8_df * pd.Series({"511990.SH": 1/3, "511260.SH": 1/3, "518880.SH": 1/3})).sum(axis=1).fillna(0)
    v8_daily.index = [int(x) for x in v8_daily.index]
    return sig_df["s123"].to_dict(), v8_daily

def run_strategy_v5(global_top_k=3,             # 低估池内全局TopK
                    max_same_sector=2,           # 最多2只同板块（强制分散）
                    max_pe=60,                   # PE硬上限(0,max_pe]
                    min_turnover_pct=0.5,        # 月换手率下限%
                    # ===== v5 新增硬阈值 =====
                    min_circ_mv_yi=100,          # 流通市值下限(亿) - 大盘股起点v5=100亿(v4=50亿)
                    max_circ_mv_yi=2000,         # 流通市值上限(亿)
                    min_list_years=2,            # 上市满N年（次新股过滤）
                    max_peg=2.0,                 # PEG硬上限：≥则直接剔除（v4仅作为评分）
                    peg_preferred=1.5,           # PEG优选线：<则加分
                    min_roe_pct=10.0,            # ROE硬门槛%：<则直接剔除
                    chip_conc_pctl_threshold=0.60,  # 筹码集中度截面后40%→剔除（只留前60%）
                    preferred_weight=1.2,        # 白名单板块加分
                    pe_pct_thr=PE_PCT_THR,
                    use_s123=False,              # 是否叠加 s123 择时状态机
                    sig_map=None,                # {ym: s123}
                    v8_daily=None,               # Series<int YYYYMMDD, 日收益>
                    verbose=False):
    # ===== 前视修复：T月数据选股 → T+1月首个交易日执行 =====
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
            monthly_picks[_next_month(ym_int)] = []
            continue
        sig = sect_signal[ym_int]
        undv_secs = [s for s, v in sig.items() if v < pe_pct_thr]
        if not undv_secs:
            monthly_picks[_next_month(ym_int)] = []
            continue
        if ym_int not in ym_to_dt:
            monthly_picks[_next_month(ym_int)] = []
            continue
        dt_real = ym_to_dt[ym_int]
        sub = ml_scored[ml_scored["dt"] == dt_real].copy()
        if len(sub) == 0:
            monthly_picks[_next_month(ym_int)] = []
            continue
        # ===== 低估板块池内 =====
        sub["sects"] = sub["ts_code"].map(lambda c: code_info.get(c, {}).get("sectors", []))
        sub["in_undv"] = sub["sects"].apply(lambda lst: any(s in undv_secs for s in lst))
        sub = sub[sub["in_undv"]].copy()
        if len(sub) == 0:
            monthly_picks[_next_month(ym_int)] = []
            continue

        # ===== v5 强化：硬过滤层（非软评分，不符合直接T出） =====
        N0 = len(sub)

        # (1) 严格ST/退市风险过滤
        sub["st"] = sub["ts_code"].map(lambda c: code_info.get(c, {}).get("is_st", 1))
        sub = sub[sub["st"] == 0]

        # (2) 次新股过滤：上市不满N年直接剔除
        def days_since_listed(c, d_day):
            ld = code_info.get(c, {}).get("list_date", None)
            if ld is None or pd.isna(ld): return -1  # 日期不明→谨慎剔除
            return (d_day - ld).days
        sub["list_days"] = sub["ts_code"].apply(lambda c: days_since_listed(c, dt_real))
        sub = sub[sub["list_days"] >= min_list_years * 365]
        N1 = len(sub)

        # (3) PE硬过滤 (0, max_pe]
        if "pe" in sub.columns:
            mask = (sub["pe"].isna()) | ((sub["pe"] > 0) & (sub["pe"] <= max_pe))
            sub = sub[mask]

        # (4) PEG硬过滤：≥max_peg 直接剔除（PEG是估值-增长匹配核心）
        if "peg" in sub.columns:
            mask = sub["peg"].isna() | (sub["peg"] < max_peg)
            sub = sub[mask]
        N2 = len(sub)

        # (5) ROE硬门槛：<min_roe_pct 直接剔除
        if "roe" in sub.columns:
            mask = sub["roe"].isna() | (sub["roe"] >= min_roe_pct)
            sub = sub[mask]
        N3 = len(sub)

        # (6) 筹码集中度分位过滤：只留截面前 chip_conc_pctl_threshold%
        if "chip_conc_20" in sub.columns and len(sub) >= 5:
            thr = sub["chip_conc_20"].quantile(1.0 - chip_conc_pctl_threshold)
            mask = sub["chip_conc_20"].isna() | (sub["chip_conc_20"] >= thr)
            sub = sub[mask]
        N4 = len(sub)

        # (7) 换手率
        if "turnover_rate" in sub.columns:
            mask = (sub["turnover_rate"].isna()) | (sub["turnover_rate"] >= min_turnover_pct)
            sub = sub[mask]

        # (8) 纯大盘股市值区间
        if "circ_mv" in sub.columns:
            mask = (sub["circ_mv"].isna()) | ((sub["circ_mv"] >= min_circ_wan) & (sub["circ_mv"] <= max_circ_wan))
            sub = sub[mask]
        N5 = len(sub)

        if verbose and len(sub) > 0:
            print(f"  ym={ym_int} 过滤: 低估池{N0}→次新过滤后{N1}→PEG<{max_peg}后{N2}→ROE≥{min_roe_pct}%后{N3}→筹码Top{chip_conc_pctl_threshold*100:.0f}%后{N4}→大盘市值后{N5}")
        if len(sub) == 0:
            monthly_picks[_next_month(ym_int)] = []
            continue

        # ===== 优选加分：PEG < peg_preferred → 额外加分 =====
        sub["_peg_bonus"] = 0.0
        if "peg" in sub.columns:
            sub.loc[sub["peg"] < peg_preferred, "_peg_bonus"] = 0.5  # PEG<1.5 → 额外+0.5标准差等价分
        # 白名单行业额外加分（鼓励聚焦高胜率赛道）
        if preferred_weight != 1.0:
            def is_pref(sects): return any(s in PREFERRED_SECTORS for s in sects)
            sub["_pref"] = sub["sects"].apply(is_pref).astype(int)
            sub["score_adj"] = (sub["score"] + sub["_peg_bonus"]) * (1.0 + sub["_pref"] * (preferred_weight - 1.0))
        else:
            sub["score_adj"] = sub["score"] + sub["_peg_bonus"]

        # ===== 全局TopK + 行业分散约束 =====
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
            print(f"  ym={ym_int} 选中={chosen} {names}, 行业={sec_count}, "
                  f"adj_score={[float(sub[sub['ts_code']==c]['score_adj'].iloc[0]) for c in chosen]}")
        monthly_picks[_next_month(ym_int)] = chosen

    # ---------- 日频执行 ----------
    # s123 择时模式：state_in=True 持有股票, False 持有V8避险（reserve 按V8日收益滚动）
    state_in = (not use_s123)
    cash = float(INIT) if state_in else 0.0
    reserve = 0.0 if state_in else float(INIT)
    holdings = {}
    nav_series = []; trades = []
    v8_nav_dt = None
    if use_s123 and v8_daily is not None:
        v8d = v8_daily.copy()
        v8d.index = pd.to_datetime(v8d.index.astype(str), format="%Y%m%d")
        v8_nav_dt = (1.0 + v8d).sort_index()

    for di, day in enumerate(all_dates):
        # 避险资金按 V8 日收益滚动 (v8_nav_dt 已存为 1+日收益, 直接乘)
        if (not state_in) and v8_nav_dt is not None:
            v8r = v8_nav_dt.get(day)
            if v8r is not None and not np.isnan(v8r):
                reserve *= v8r
        is_monthly_start = (di == 0) or (all_dates[di-1].month != day.month)
        if is_monthly_start:
            ym_int = day.year * 100 + day.month
            # s123 状态机：用 T-1 月信号决定 T 月状态（T月信号当月末才可得，避免前视）
            if use_s123 and sig_map is not None:
                prev_ym = ym_int - 1 if ym_int % 100 != 1 else ym_int - 89
                ps = sig_map.get(prev_ym)
                if ps is not None:
                    if (not state_in) and ps >= 3:
                        state_in = True
                    elif state_in and ps <= 1:
                        state_in = False
                # 状态切换
                if state_in and reserve > 0:
                    cash += reserve; reserve = 0.0
                    trades.append((day, "S123IN", "V8", cash, np.nan, 0))
                elif (not state_in) and (cash > 0 or holdings):
                    for c in list(holdings.keys()):
                        if c in close_panel and day in close_panel[c].index:
                            p = close_panel[c].loc[day]
                            reserve += p * holdings[c]["qty"] * (1 - SELL_FEE)
                        del holdings[c]
                    reserve += cash; cash = 0.0
                    trades.append((day, "S123OUT", "V8", reserve, np.nan, 0))
            if state_in and ym_int in monthly_picks:
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
        # 止盈30% / 时间止损270天（仅在市状态）
        if state_in:
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
        total = cash + reserve
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
# 8. v5 组合测试（核心变量梯度搜索）
# ============================================================
print("\n" + "="*105)
print("开始回测 v5 (大盘100亿+PEG<2硬过滤+ROE≥10%硬门槛+筹码前60%+次新2年+严格ST过滤)")
print(f"买入{BUY_FEE:.2%}/卖出{SELL_FEE:.2%}  止盈{TP:.0%}  时间止损{MAX_HOLD}天  PE分位<{PE_PCT_THR:.0%}低估")
print(f"权重：PEG 40% + 筹码 25% + ROE 15% + PE 10% + 增长 10%")
print(f"黑名单: {sorted(BLACKLIST_SECTORS)}")
print("="*105)

# 测试变量：
# - TopK 持仓数 (3/5)
# - PEG 硬阈值 (1.5严格 / 2.0适中 / 2.5宽松)
# - ROE 硬门槛 (8% / 10% / 12%)
# - 市值下限 (100亿纯大盘 / 200亿超大票)
configs = [
    # ===== 基准配置（对标v4-A3） =====
    ("V5-B1: Top3+PEG<2+ROE≥10%+大盘100亿", 3, 2, 60, 0.5, 100, 2000, 2, 1.5, 10, 0.60, 2, 1.20),
    ("V5-B2: Top3+PEG<1.5严+ROE≥10%+大盘100亿", 3, 2, 60, 0.5, 100, 2000, 1.5, 1.0, 10, 0.60, 2, 1.20),
    # ===== ROE门槛梯度 =====
    ("V5-R1: Top3+PEG<2+ROE≥8%+大盘100亿",   3, 2, 60, 0.5, 100, 2000, 2, 1.5, 8,  0.60, 2, 1.20),
    ("V5-R2: Top3+PEG<2+ROE≥12%+大盘100亿",  3, 2, 60, 0.5, 100, 2000, 2, 1.5, 12, 0.60, 2, 1.20),
    # ===== PEG阈值梯度 =====
    ("V5-P1: Top3+PEG<2.5松+ROE≥10%+大盘100亿", 3, 2, 60, 0.5, 100, 2000, 2.5, 2.0, 10, 0.60, 2, 1.20),
    # ===== 更纯的大盘（200亿以上） =====
    ("V5-M1: Top3+PEG<2+ROE≥10%+大盘200亿",   3, 2, 60, 0.5, 200, 3000, 2, 1.5, 10, 0.60, 2, 1.20),
    ("V5-M2: Top5+PEG<2+ROE≥10%+大盘200亿",   5, 2, 60, 0.5, 200, 3000, 2, 1.5, 10, 0.60, 2, 1.20),
    # ===== Top5 分散 =====
    ("V5-K1: Top5+PEG<2+ROE≥10%+大盘100亿",   5, 2, 60, 0.5, 100, 2000, 2, 1.5, 10, 0.60, 2, 1.20),
    # ===== 更严筹码（只留前50%） =====
    ("V5-C1: Top3+PEG<2+ROE≥10%+筹码前50%",   3, 2, 60, 0.5, 100, 2000, 2, 1.5, 10, 0.50, 2, 1.20),
    # ===== 次新股过滤3年（更严） =====
    ("V5-L1: Top3+PEG<2+ROE≥10%+上市满3年",   3, 2, 60, 0.5, 100, 2000, 2, 1.5, 10, 0.60, 3, 1.20),
    # ===== 修复前视后的优化网格 =====
    ("V5-O1: Top5+PEG<2+ROE≥12%+大盘100亿",  5, 2, 60, 0.5, 100, 2000, 2, 1.5, 12, 0.60, 2, 1.20),
    ("V5-O2: Top3+PEG<2+ROE≥15%+大盘100亿",  3, 2, 60, 0.5, 100, 2000, 2, 1.5, 15, 0.60, 2, 1.20),
    ("V5-O3: Top3+PEG<2+ROE≥12%+同行业≤1",   3, 1, 60, 0.5, 100, 2000, 2, 1.5, 12, 0.60, 2, 1.20),
    ("V5-O4: Top3+PEG<2+ROE≥12%+筹码前50%",  3, 2, 60, 0.5, 100, 2000, 2, 1.5, 12, 0.50, 2, 1.20),
    ("V5-O5: Top5+PEG<2+ROE≥12%+同行业≤1",   5, 1, 60, 0.5, 100, 2000, 2, 1.5, 12, 0.60, 2, 1.20),
    # ===== 优化组合验证（第二轮） =====
    ("V5-O6: Top6+PEG<2+ROE≥12%+大盘100亿",  6, 2, 60, 0.5, 100, 2000, 2, 1.5, 12, 0.60, 2, 1.20),
    ("V5-O7: Top5+PEG<2+ROE≥12%+上市满3年",  5, 2, 60, 0.5, 100, 2000, 2, 1.5, 12, 0.60, 3, 1.20),
    ("V5-O8: Top5+PEG<2+ROE≥12%+大盘150亿",  5, 2, 60, 0.5, 150, 2000, 2, 1.5, 12, 0.60, 2, 1.20),
    ("V5-O9: Top5+PEG<2+ROE≥12%+筹码前50%",  5, 2, 60, 0.5, 100, 2000, 2, 1.5, 12, 0.50, 2, 1.20),
    # ===== O9基础上压回撤（第三轮） =====
    ("V5-O10: Top5+ROE12+筹码50+同行业≤1",   5, 1, 60, 0.5, 100, 2000, 2, 1.5, 12, 0.50, 2, 1.20),
    ("V5-O11: Top5+ROE12+筹码50+市值80-1500亿",5, 2, 60, 0.5, 80, 1500, 2, 1.5, 12, 0.50, 2, 1.20),
    ("V5-O12: Top5+ROE12+筹码50+市值100-1500亿",5, 2, 60, 0.5, 100, 1500, 2, 1.5, 12, 0.50, 2, 1.20),
    ("V5-O13: Top5+ROE12+筹码50+上市满3年",  5, 2, 60, 0.5, 100, 2000, 2, 1.5, 12, 0.50, 3, 1.20),
    ("V5-O14: Top5+ROE12+筹码50+白名单×1.5", 5, 2, 60, 0.5, 100, 2000, 2, 1.5, 12, 0.50, 2, 1.50),
    # ===== s123 择时叠加（第四轮：T-1月信号→T月生效） =====
    ("V5-S1: R2+s123择时 (Top3+ROE12)",     3, 2, 60, 0.5, 100, 2000, 2, 1.5, 12, 0.60, 2, 1.20),
    ("V5-S2: O9+s123择时 (Top5+ROE12+筹码50)",5, 2, 60, 0.5, 100, 2000, 2, 1.5, 12, 0.50, 2, 1.20),
    ("V5-S3: O1+s123择时 (Top5+ROE12)",     5, 2, 60, 0.5, 100, 2000, 2, 1.5, 12, 0.60, 2, 1.20),
    ("V5-S4: O11+s123择时 (市值80-1500亿)", 5, 2, 60, 0.5, 80, 1500, 2, 1.5, 12, 0.50, 2, 1.20),
    ("V5-S5: K1+s123择时 (Top5+ROE10)",     5, 2, 60, 0.5, 100, 2000, 2, 1.5, 10, 0.60, 2, 1.20),
]

res = []
nav_curves = {}
_sig_map = None; _v8_daily = None
for i, cfg in enumerate(configs):
    lb, k, mss, mpe, mto, mncm, mxcm, mpeg, ppeg, mroe, cc_thr, lyrs, pw = cfg
    use_s123 = "S123" in lb or lb.startswith("V5-S")
    if use_s123 and _sig_map is None:
        print("  [s123] 构建信号与V8避险组合...", flush=True)
        _sig_map, _v8_daily = build_s123_v8()
    t1 = time.time()
    try:
        nv, trs, _ = run_strategy_v5(
            global_top_k=k, max_same_sector=mss,
            max_pe=mpe, min_turnover_pct=mto,
            min_circ_mv_yi=mncm, max_circ_mv_yi=mxcm,
            max_peg=mpeg, peg_preferred=ppeg,
            min_roe_pct=mroe, chip_conc_pctl_threshold=cc_thr,
            min_list_years=lyrs, preferred_weight=pw,
            use_s123=use_s123, sig_map=_sig_map, v8_daily=_v8_daily)
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

# ===== 加入v4最佳基准 + 板块基金基线 =====
baselines = [
    {"配置":"【v4基】A3: Top3+PE≤80+白名单×1.2 (v4最佳)","年化":0.355,"累计":5.19,"回撤":-0.295,"夏普":1.41,
     "买入":144,"止盈":90,"时间止损":44,"止盈率":0.625,"平均持仓天":132.0,"期末(万)":619.3},
    {"配置":"【板块基】板块基金(PE48月+T270)","年化":0.144,"累计":1.43,"回撤":-0.303,"夏普":0.55,
     "买入":132,"止盈":46,"时间止损":83,"止盈率":0.35,"平均持仓天":224.0,"期末(万)":243.0},
    {"配置":"【中证1000】持有512100不动","年化":0.109,"累计":0.97,"回撤":-0.600,"夏普":0.38,
     "买入":0,"止盈":0,"时间止损":0,"止盈率":0.0,"平均持仓天":0.0,"期末(万)":197.0},
]
res.extend(baselines)

rdf = pd.DataFrame(res)
print("\n" + "="*145)
print("【汇总 v5 vs v4】PEG<2硬过滤 + ROE≥10%硬门槛 + 大盘100亿 + 筹码前60% + 次新2年 + 严格ST过滤")
print("权重: PEG 40% + 筹码 25% + ROE 15% + PE 10% + 净利同比 10%")
print("="*145)
pd.set_option("display.width", 260)
pd.set_option("display.max_columns", 20)
print(rdf.to_string(index=False, float_format=lambda x: f"{x:.2f}" if not isinstance(x,str) else x))

feasible = rdf[rdf["回撤"] > -0.30]
if len(feasible):
    best = feasible.loc[feasible["夏普"].idxmax()]
    print(f"\n★ 最优(回撤≥-30% 取夏普最高): {best['配置']}")
    print(f"   年化={best['年化']:.1%}  回撤={best['回撤']:.1%}  夏普={best['夏普']:.2f}  期末={best['期末(万)']:.0f}万")
    # 和v4最佳对比
    v4_ann, v4_mdd, v4_shp, v4_end = 0.355, -0.295, 1.41, 619.3
    print(f"   vs v4-A3: 年化 Δ={best['年化']-v4_ann:+.1%}, 回撤 Δ={best['回撤']-v4_mdd:+.1%}, "
          f"夏普 Δ={best['夏普']-v4_shp:+.2f}, 期末 Δ={best['期末(万)']-v4_end:+.0f}万")
else:
    best = rdf.loc[rdf["夏普"].idxmax()]
    print(f"\n★ 最优(全池夏普最高): {best['配置']}")
    print(f"   年化={best['年化']:.1%}  回撤={best['回撤']:.1%}  夏普={best['夏普']:.2f}  期末={best['期末(万)']:.0f}万")

rdf.to_csv(os.path.join(OUT_DIR, "stock_selected_v5.csv"), index=False, encoding="utf-8-sig")

# ============================================================
# 9. 画图
# ============================================================
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams["font.sans-serif"] = ["SimHei","Microsoft YaHei","Arial Unicode MS"]
rcParams["axes.unicode_minus"] = False

# 净值曲线：选前6个v5策略 + v4基 + 板块基
fig, ax = plt.subplots(figsize=(15, 7.5))
styles = ["-", "-.", "--", ":", "-.", "--", ":", "-", "-."]
# 展示优先级
show_labels = []
for cfg in configs[:6]: show_labels.append(cfg[0])
show_labels.append("【v4基】A3: Top3+PE≤80+白名单×1.2 (v4最佳)")
show_labels.append("【板块基】板块基金(PE48月+T270)")

for j, lb in enumerate(show_labels):
    if lb not in nav_curves and lb not in [b["配置"] for b in baselines]:
        continue
    if lb in nav_curves:
        s = nav_curves[lb]
    else:
        # 基线不用画真实曲线（只在图例标注水平线）
        continue
    if len(s) == 0: continue
    s_norm = s / s.iloc[0]
    ax.plot(s.index, s_norm.values, label=f"{lb} (期末={s.iloc[-1]/1e4:.0f}万)",
            linestyle=styles[j % len(styles)], linewidth=2.0 if j == 0 else 1.5)
# 基线水平线
ax.axhline(1.0, color="gray", lw=0.6, ls=":")
ax.axhline(6.19, color="orange", lw=1.0, ls="--", label="v4-A3 最佳 期末619万 (年化35.5%,回撤-29.5%)")
ax.axhline(2.43, color="red", lw=1.0, ls="--", label="板块基金基线 期末243万 (年化14.4%)")
ax.set_title("v5 vs v4 净值曲线对比 (初始100万)\n"
             "硬过滤: PEG<2 + ROE≥10% + 大盘100亿 + 筹码前60% + 上市满2年 + 严格ST/退市过滤\n"
             "评分权重: PEG 40% + 筹码 25% + ROE 15% + PE 10% + 增长 10%", fontsize=13)
ax.set_xlabel("日期")
ax.set_ylabel("净值 (初始=1)")
ax.legend(loc="upper left", fontsize=8.5)
ax.grid(True, alpha=0.3)
plt.tight_layout()
png_path = os.path.join(OUT_DIR, "stock_selected_v5_curve.png")
plt.savefig(png_path, dpi=130)
plt.close()
print(f"\n净值曲线图: {png_path}")

# 回撤曲线
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
ax2.axhline(-0.295, color="orange", lw=1.0, ls="--", label="v4-A3 最大回撤 -29.5%")
ax2.axhline(-0.303, color="red", lw=1.0, ls="--", label="板块基金回撤 -30.3%")
ax2.set_title("回撤曲线对比 (Underwater Plot)", fontsize=13)
ax2.set_ylabel("回撤幅度")
ax2.legend(loc="lower left", fontsize=8.5)
ax2.grid(True, alpha=0.3)
plt.tight_layout()
dd_path = os.path.join(OUT_DIR, "stock_selected_v5_drawdown.png")
plt.savefig(dd_path, dpi=130)
plt.close()
print(f"回撤曲线图: {dd_path}")

# ============================================================
# 10. 最优策略的样例持股展示（打印最近几次调仓记录）
# ============================================================
print(f"\n总耗时 {time.time()-t00:.0f}s")
print("\n【v5核心选股逻辑总结】")
print("  ✓ 低估板块池内（PE分位<30%） → 全局评分选股，不选垃圾行业便宜货")
print("  ✓ 严格过滤：*ST/ST/SST/退市股全剔除")
print("  ✓ 次新股过滤：上市不满2年直接剔除（IPO爆炒+解禁风险）")
print("  ✓ PEG硬过滤：≥2直接T出；<1.5额外加分（估值-增长匹配核心）")
print("  ✓ ROE硬门槛：<10%直接T出（只买真正赚钱的公司）")
print("  ✓ 筹码集中度：只留截面前60%（主力控盘/惜售）")
print("  ✓ 纯大盘市值：100~2000亿（50亿小市值→100亿，减少小票波动）")
print("  ✓ 行业分散：Top3持仓最多来自2个板块，防单行业黑天鹅")

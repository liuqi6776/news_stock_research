# -*- coding: utf-8 -*-
"""行业质量分层首验：低估(低PE) + 高基本面质量 vs 低估 + 低质量(价值陷阱)

数据源（绝对路径，schema 已确认）:
  - 行业映射:  D:/iquant_data/data_v2/industry1/industry.parquet   (ts_code -> 110 行业)
  - 财务质量:  D:/iquant_data/data_v2/fundamental1/fina_indicator_cache.parquet
               (ts_code, ann_date, end_date, roe, netprofit_yoy, or_yoy,
                grossprofit_margin, debt_to_assets, ...)
  - 行业PE:    月末快照 other_day1/*.parquet 聚合 (pe, circ_mv)
  - 行业月收益: data_day1/*.parquet 日频 pct_chg 等权聚合

方法论（防前视偏差）:
  1. 质量分 PIT: 用 ann_date(公告日) 而非 end_date(报告期)。决策时刻 t 只取
     avail_date <= t 的最新一期行业质量。行业季度"可用日"取成员股公告日中位数。
  2. 行业质量分: 对每个 (industry, end_date) 聚合成员股各指标中位数; 每个 end_date
     横截面 z-score(debt_to_assets 取负号), 等权(可用指标平均)合成复合质量分。
  3. 每行业每期有质量股票数 < 3 则丢弃该期。
  4. 低估: 行业PE滚动分位(48月窗, min_periods=12) < 0.30。
  5. 回测 2023-06 ~ 2026-03 月末, 低估行业中按质量分上半/下半分组, 等权持有取下一月收益。

注意: 财务质量数据仅覆盖 2023-2026(13 季度), 样本期短, 结论仅为首验。
"""
import os
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=RuntimeWarning)

DATA = r"D:/iquant_data/data_v2"
IND_PATH = os.path.join(DATA, "industry1", "industry.parquet")
FUND_PATH = os.path.join(DATA, "fundamental1", "fina_indicator_cache.parquet")
OTHER_DIR = os.path.join(DATA, "other_day1")   # pe, pb, circ_mv
DAY_DIR = os.path.join(DATA, "data_day1")      # pct_chg

MIN_STOCKS = 5           # 行业PE聚合最少成分股
MIN_QUAL_STOCKS = 3      # 行业季度质量最少有质量股票数
MIN_QUAL_IND = 3         # 复合质量分至少需可用指标数
PE_WINDOW = 48           # PE滚动分位窗口(月)
PE_MIN_PERIODS = 12      # 放宽 min_periods
LOW_PE_Q = 0.30          # PE分位 < 0.30 视为低估

BACKTEST_YM0 = "202306"  # 回测首个决策月
BACKTEST_YM1 = "202603"  # 回测末个决策月
PE_HIST_START = "20190101"  # PE历史起始(为48月分位留足窗口)

QUAL_INDICATORS = ["roe", "netprofit_yoy", "or_yoy", "grossprofit_margin", "debt_to_assets"]
NEGATIVE_IND = {"debt_to_assets"}  # 越低越好, z 取负号


def load_industry_map():
    df = pd.read_parquet(IND_PATH)
    df = df[df["industry"].notna()][["ts_code", "industry"]]
    return dict(zip(df["ts_code"], df["industry"]))


def get_dates():
    all_dates = sorted(f[:8] for f in os.listdir(DAY_DIR) if f.endswith(".parquet"))
    s = pd.Series(all_dates)
    month_ends = s.groupby(s.str[:6]).last().tolist()  # 每月最后一个交易日
    month_ends = [d for d in month_ends if d >= PE_HIST_START]
    return all_dates, month_ends


def compute_industry_pe(pe_df, ind_map):
    """单期: 按 industry 聚合 PE (指数加权 PE = Σcirc_mv / Σ(circ_mv/pe))"""
    pe_df = pe_df.copy()
    pe_df["industry"] = pe_df["ts_code"].map(ind_map)
    pe_df = pe_df.dropna(subset=["industry"])
    rows = {}
    for ind, g in pe_df.groupby("industry"):
        g = g[(g["pe"] > 0) & (g["circ_mv"] > 0)]
        if len(g) < MIN_STOCKS:
            continue
        total_mv = g["circ_mv"].sum()
        total_earn = (g["circ_mv"] / g["pe"]).sum()
        if total_earn <= 0:
            continue
        rows[ind] = total_mv / total_earn
    return pd.Series(rows)


def build_pe_series(ind_map, month_ends):
    pe_rows = []
    for i, d in enumerate(month_ends):
        fp = os.path.join(OTHER_DIR, f"{d}.parquet")
        if not os.path.exists(fp):
            continue
        try:
            df = pd.read_parquet(fp, columns=["ts_code", "pe", "circ_mv"])
        except Exception:
            continue
        s = compute_industry_pe(df, ind_map)
        if s.empty:
            continue
        pe_rows.append(s.rename(d))
        if (i + 1) % 24 == 0:
            print(f"  [PE] {i+1}/{len(month_ends)}", flush=True)
    pe_df = pd.concat(pe_rows, axis=1).T
    pe_df.index.name = "date"
    return pe_df


def build_quality_scores(ind_map):
    fund = pd.read_parquet(FUND_PATH)
    fund = fund.copy()
    fund["ann_date"] = fund["ann_date"].astype(str).astype(int)
    fund["end_date"] = fund["end_date"].astype(str)
    fund["industry"] = fund["ts_code"].map(ind_map)
    fund = fund.dropna(subset=["industry"])
    # 同一 (ts_code, end_date) 保留最新公告日
    fund = fund.sort_values("ann_date").drop_duplicates(["ts_code", "end_date"], keep="last")

    # 1) 每个 (industry, end_date) 聚合成员股各指标中位数, 可用日 = 成员股公告日中位数
    rows = []
    for (ind, ed), g in fund.groupby(["industry", "end_date"]):
        has_q = int(g[QUAL_INDICATORS].notna().any(axis=1).sum())
        if has_q < MIN_QUAL_STOCKS:
            continue
        row = {"industry": ind, "end_date": ed,
               "n_stocks": has_q,
               "avail_date": int(g["ann_date"].median())}
        for c in QUAL_INDICATORS:
            row[c] = g[c].median()
        rows.append(row)
    agg = pd.DataFrame(rows)

    # 2) 每个 end_date 横截面 z-score, debt 取负, 等权(可用指标平均)合成复合分
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
    return qual


def pit_quality_at(qual, t):
    """决策时刻 t: 取 avail_date <= t 的最新一期行业质量"""
    avail = qual[qual["avail_date"] <= t]
    if avail.empty:
        return pd.Series(dtype=float)
    latest = avail.sort_values("end_date").groupby("industry").tail(1)
    return latest.set_index("industry")["quality"]


def build_ret_series(ind_map, all_dates, decision_ends, month_ends):
    """每个决策月末 d0 -> 下一月末 d1 的行业等权月收益 (用日频 pct_chg)"""
    ind_groups = {}
    for code, ind in ind_map.items():
        ind_groups.setdefault(ind, []).append(code)
    ind_groups = {k: v for k, v in ind_groups.items() if len(v) >= MIN_STOCKS}
    all_codes = sorted({c for v in ind_groups.values() for c in v})
    date_idx = {d: i for i, d in enumerate(all_dates)}
    next_map = {month_ends[i]: month_ends[i + 1] for i in range(len(month_ends) - 1)}

    ret_rows = []
    for j, d0 in enumerate(decision_ends):
        d1 = next_map.get(d0)
        if d1 is None:
            continue
        i0, i1 = date_idx[d0], date_idx[d1]
        hold = all_dates[i0 + 1: i1 + 1]
        frames = []
        for d in hold:
            fp = os.path.join(DAY_DIR, f"{d}.parquet")
            try:
                df = pd.read_parquet(fp, columns=["ts_code", "pct_chg"])
            except Exception:
                continue
            df = df[df["ts_code"].isin(all_codes)]
            if df.empty:
                continue
            df["_d"] = d
            frames.append(df)
        if not frames:
            continue
        big = pd.concat(frames, ignore_index=True)
        pct_df = big.pivot_table(index="_d", columns="ts_code", values="pct_chg") / 100.0
        row = {}
        for ind, codes in ind_groups.items():
            sub = pct_df.reindex(columns=codes)
            if sub.shape[1] == 0:
                row[ind] = np.nan
                continue
            daily_mean = sub.mean(axis=1)  # 每日等权(跳过缺失)
            row[ind] = (1 + daily_mean).prod() - 1
        ret_rows.append(pd.Series(row, name=d1))
        if (j + 1) % 12 == 0:
            print(f"  [RET] {j+1}/{len(decision_ends)}", flush=True)

    if not ret_rows:
        return pd.DataFrame()
    ret_df = pd.concat(ret_rows, axis=1).T
    ret_df.index.name = "date"
    return ret_df


def _rolling_pct_rank(a):
    """滚动窗口内当前值所处分位 (窗口内 <= 当前值 的占比)"""
    if len(a) == 0 or np.isnan(a[-1]):
        return np.nan
    valid = a[~np.isnan(a)]
    if len(valid) == 0:
        return np.nan
    return float((valid <= valid[-1]).mean())


def run_backtest(pe_pct, ret_df, qual, decision_ends, next_map):
    recs = []
    for d0 in decision_ends:
        d1 = next_map.get(d0)
        if d1 is None or d1 not in ret_df.index:
            continue
        pct_t = pe_pct.loc[d0] if d0 in pe_pct.index else pd.Series(dtype=float)
        fwd = ret_df.loc[d1]
        # 低估行业(有PE分位且有前向收益)
        under = [i for i in pct_t[pct_t < LOW_PE_Q].index
                 if i in fwd.index and pd.notna(fwd[i])]
        if not under:
            continue
        q = pit_quality_at(qual, int(d0))
        under_q = [i for i in under if i in q.index and pd.notna(q[i])]
        n_all, n_q = len(under), len(under_q)
        ret_all = float(np.mean([fwd[i] for i in under]))
        ret_high = ret_low = np.nan
        n_high = n_low = 0
        if n_q >= 2:
            qs = q[under_q].sort_values()
            med = qs.median()
            high = qs[qs >= med].index.tolist()
            low = qs[qs < med].index.tolist()
            n_high, n_low = len(high), len(low)
            ret_high = float(np.mean([fwd[i] for i in high]))
            ret_low = float(np.mean([fwd[i] for i in low]))
        recs.append({
            "date": d0, "next": d1,
            "n_all": n_all, "n_q": n_q, "n_high": n_high, "n_low": n_low,
            "ret_all": ret_all, "ret_high": ret_high, "ret_low": ret_low,
        })
    return pd.DataFrame(recs)


def summarize(df):
    groups = [("ret_all", "全低估等权"), ("ret_high", "低估+高质量"), ("ret_low", "低估+低质量")]
    print("\n" + "=" * 78)
    print("三组前向月收益统计")
    print("=" * 78)
    res = {}
    for col, name in groups:
        r = df[col].dropna()
        if len(r) == 0:
            print(f"{name:14s} 无样本")
            continue
        avg = r.mean()
        win = (r > 0).mean()
        nav = (1 + r).prod()
        ann = nav ** (12 / len(r)) - 1
        res[col] = dict(name=name, avg=avg, win=win, nav=nav, ann=ann, n=len(r))
        print(f"{name:14s} 月均收益={avg:+.4f}  胜率={win:.3f}  "
              f"累计净值={nav:.3f}  年化={ann:+.4f}  样本月={len(r)}")
    print("-" * 78)
    ls = df["ret_high"] - df["ret_low"]
    ls = ls.dropna()
    if len(ls) > 0:
        avg_ls = ls.mean()
        sd_ls = ls.std(ddof=1)
        t_ls = avg_ls / (sd_ls / np.sqrt(len(ls))) if sd_ls > 0 else np.nan
        print(f"多空(低估+高质量 - 低估+低质量): 月均={avg_ls:+.4f}  "
              f"std={sd_ls:.4f}  t值={t_ls:+.3f}  样本月={len(ls)}")
        print(f"  多空月胜率={(ls > 0).mean():.3f}  "
              f"多空累计净值={(1 + ls).prod():.3f}")
    print("=" * 78)
    return res


def main():
    print("=" * 78)
    print("行业质量分层首验: 低估(低PE) + 高/低基本面质量 的下一月收益对比")
    print("=" * 78, flush=True)

    ind_map = load_industry_map()
    print(f"[1/5] 行业映射: {len(ind_map)} 只股票 -> {len(set(ind_map.values()))} 个行业", flush=True)

    all_dates, month_ends = get_dates()
    decision_ends = [d for d in month_ends if BACKTEST_YM0 <= d[:6] <= BACKTEST_YM1]
    print(f"[2/5] 交易日 {all_dates[0]}~{all_dates[-1]}, 月末序列 {month_ends[0]}~{month_ends[-1]}",
          flush=True)
    print(f"      回测决策月末 {decision_ends[0]} ~ {decision_ends[-1]} (共 {len(decision_ends)} 个月)",
          flush=True)

    print("[3/5] 构建行业 PE 月末序列(2019起, 为48月分位留窗口)...", flush=True)
    pe_df = build_pe_series(ind_map, month_ends)
    pe_pct = pe_df.rolling(PE_WINDOW, min_periods=PE_MIN_PERIODS).apply(_rolling_pct_rank, raw=True)
    print(f"      行业PE面板 {pe_df.shape} | 行业数 {pe_df.shape[1]}", flush=True)

    print("[4/5] 构建行业季度质量分(PIT, 用公告日)...", flush=True)
    qual = build_quality_scores(ind_map)
    print(f"      有效行业-季度记录 {qual.shape[0]} 条 | 覆盖行业 {qual['industry'].nunique()} 个 | "
          f"end_date {qual['end_date'].min()}~{qual['end_date'].max()}", flush=True)

    print("[5/5] 构建下一月行业收益(日频等权)...", flush=True)
    ret_df = build_ret_series(ind_map, all_dates, decision_ends, month_ends)
    print(f"      月收益面板 {ret_df.shape}", flush=True)

    next_map = {month_ends[i]: month_ends[i + 1] for i in range(len(month_ends) - 1)}
    df = run_backtest(pe_pct, ret_df, qual, decision_ends, next_map)

    # 每月明细
    print("\n每月分组明细 (n_all=低估行业数, n_q=有质量低估数, n_h/n_l=高质量/低质量组):")
    hdr = f"{'决策月末':>10} {'n_all':>5} {'n_q':>4} {'n_h':>4} {'n_l':>4} " \
          f"{'全低估':>8} {'高质量':>8} {'低质量':>8} {'多空':>8}"
    print(hdr)
    for _, r in df.iterrows():
        ls = r["ret_high"] - r["ret_low"] if pd.notna(r["ret_high"]) and pd.notna(r["ret_low"]) else np.nan
        print(f"{r['date']:>10} {r['n_all']:>5} {r['n_q']:>4} {r['n_high']:>4} {r['n_low']:>4} "
              f"{r['ret_all']:>+8.4f} {r['ret_high']:>+8.4f} {r['ret_low']:>+8.4f} "
              f"{ls:>+8.4f}" if pd.notna(ls) else
              f"{r['date']:>10} {r['n_all']:>5} {r['n_q']:>4} {r['n_high']:>4} {r['n_low']:>4} "
              f"{r['ret_all']:>+8.4f} {r['ret_high']:>8} {r['ret_low']:>8} {ls:>8}")

    summarize(df)

    print("\n[说明] 财务质量数据仅覆盖 2023-2026 (13 个季度), 样本期短, 结论仅为首验;")
    print("       行业季度可用日取成员股公告日中位数, 复合质量分为横截面z-score等权(可用指标平均, 负债率取负)。")
    return 0


if __name__ == "__main__":
    sys.exit(main())

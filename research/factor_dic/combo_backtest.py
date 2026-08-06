# -*- coding: utf-8 -*-
"""
合并回测: 现有 STOCK 因子池 vs 加入 turnover_vol_20 增强

统一框架(与 21 因子验证一致, 保证可比):
  - 样本: 2020.01~2026.06, 中证1000成分股, 月末调仓
  - 因子合成: 月度截面 winsorize(1%/99%) -> z-score -> 等权
  - 未来收益: T+1~T+20 累计(复权 pct_chg), PIT 干净
  - 回测: Top50 月度调仓, 20bps 双边成本, 对比中证1000

组合定义:
  BASE   = z(ret_1m) + z(ivol)                 # 现有池价格因子(占 study_007 主要IC)
  BASE_F = z(ret_1m) + z(ivol) + z(roe)        # 现有池完整版(含基本面, roe PIT)
  ENH    = z(ret_1m) + z(ivol) + z(turnover_vol_20)  # 加入新增量因子

用法:
    python research/factor_dic/combo_backtest.py
"""
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config.settings import settings
from research.factor_dic import run_validation as rv

OUT_DIR = rv.OUT_DIR
FUNDA_PATH = os.path.join(settings.DATA_PATH, "fundamental1", "fina_indicator_cache.parquet")
IND_PATH = os.path.join(settings.DATA_PATH, "industry1", "industry.parquet")
MV_DIR = os.path.join(settings.DATA_PATH, "other_day1")   # daily_basic: circ_mv(万元)
MV_FLOOR_WAN = 300000   # 市值地板: 剔除流通市值 < 30 亿(万元) 的微盘股
LIMIT_UP_MAIN = 9.9     # 主板涨停近似阈值(pct_chg)
LIMIT_UP_GE = 19.9      # 创业板/科创板涨停近似阈值(pct_chg)

RET_1M_FKEY = "ret_1m"
IVOL_FKEY = "ivol"
TURN_FKEY = "turnover_vol_20"


def _is_limit_up(code, pct):
    """调仓日收盘涨停近似判断: 300/688/689 为 20%, 其余 10%"""
    if pct != pct:
        return False
    th = LIMIT_UP_GE if (code.startswith("300") or code.startswith("688") or code.startswith("689")) else LIMIT_UP_MAIN
    return pct >= th


def build_price_factors(stocks, all_codes):
    """逐股计算 ret_1m / ivol(复权口径) / turnover_vol_20 序列 + 未来收益"""
    ret_1m, ivol, turn, fwd = {}, {}, {}, {}
    t0 = time.time()
    for i, code in enumerate(all_codes):
        df = stocks.get(code)
        if df is None or len(df) < 60:
            continue
        pct = df["pct_chg"].fillna(0.0)
        r = pct / 100.0
        cum = (1 + r).cumprod()
        fwd[code] = cum.shift(-rv.FORWARD_DAYS) / cum - 1.0
        # 1月反转(取负: 过去跌得多=好)
        ret_1m[code] = -(1 + r).rolling(20, min_periods=10).apply(np.prod, raw=True) + 1.0
        # 低特质波动率(取负: 波动低=好), 与 study_007 ivol 同语义
        ivol[code] = -r.rolling(20, min_periods=10).std()
        # turnover_vol_20 (factor_lib 已统一方向: 高值=好)
        s = rv.build_factor_series(TURN_FKEY, df, None, None, None)
        if s is not None:
            turn[code] = s
        if (i + 1) % 500 == 0:
            print(f"  [calc] {i+1}/{len(all_codes)} ({time.time()-t0:.0f}s)")
    print(f"[calc] 因子序列完成 ({time.time()-t0:.0f}s, 有效个股 {len(ret_1m)})")
    return ret_1m, ivol, turn, fwd


def build_roe_pit(rebal_dates):
    """roe PIT 面板: 每月取 ann_date<=调仓日 的最新公告值(无前视)"""
    if not os.path.exists(FUNDA_PATH):
        print("[warn] 无基本面数据, BASE_F 组合跳过")
        return {}
    funda = pd.read_parquet(FUNDA_PATH)[["ts_code", "ann_date", "roe"]]
    funda["ann_date"] = funda["ann_date"].astype(str).str[:8]
    funda = funda.dropna(subset=["roe"]).sort_values("ann_date")
    out = {}
    for rb in rebal_dates:
        latest = funda[funda["ann_date"] <= rb].drop_duplicates("ts_code", keep="last")
        out[rb] = latest.set_index("ts_code")["roe"]
    return out


def load_industry_map():
    """ts_code -> industry(申万一级)"""
    if not os.path.exists(IND_PATH):
        print("[warn] 无行业数据, 行业中性化跳过")
        return {}
    df = pd.read_parquet(IND_PATH)[["ts_code", "industry"]]
    return dict(zip(df["ts_code"].astype(str), df["industry"].astype(str)))


def build_circ_mv(rebal_dates, all_codes):
    """调仓日流通市值面板: {rb: {code: circ_mv(万元)}}, 用于市值中性化"""
    out = {}
    if not os.path.isdir(MV_DIR):
        print("[warn] 无 other_day1 市值数据, 市值中性化跳过")
        return out
    for rb in rebal_dates:
        fp = os.path.join(MV_DIR, f"{rb}.parquet")
        if not os.path.exists(fp):
            continue
        try:
            df = pd.read_parquet(fp, columns=["ts_code", "circ_mv"])
        except Exception:
            try:
                df = pd.read_parquet(fp)
                df = df[["ts_code", "circ_mv"]]
            except Exception:
                continue
        df = df[df["ts_code"].astype(str).isin(all_codes)].dropna(subset=["circ_mv"])
        if not df.empty:
            out[rb] = dict(zip(df["ts_code"].astype(str), df["circ_mv"]))
    return out


def zscore_series(s):
    return (s - s.mean()) / (s.std() + 1e-8)


def run_combo(rebal_dates, all_codes, ret_1m, ivol, turn, fwd, roe_pit, pct_df, trade_dates, ind_map, mv_map):
    """月度截面合成 + Top50 回测, 对比基准.
    neut: None=原始; "ind"=行业内 z-score; "ind_size"=市值回归残差 + 行业内 z-score
    filt: True=调仓日可交易性过滤(市值>=30亿 + 非涨停 + 未停牌)"""
    bench_daily = None
    bench_fp = os.path.join(rv.IDX_DIR, f"{rv.INDEX_CODE}.parquet")
    if os.path.exists(bench_fp):
        bdf = pd.read_parquet(bench_fp)
        bdf["trade_date"] = bdf["trade_date"].astype(str).str[:8]
        bench_daily = bdf.set_index("trade_date")

    combos = {"BASE": (["ret_1m", "ivol"], None, False),
              "BASE_F": (["ret_1m", "ivol", "roe"], None, False),
              "ENH": (["ret_1m", "ivol", "turn"], None, False),
              "ENH_F": (["ret_1m", "ivol", "roe", "turn"], None, False),
              "BASE_F_NI": (["ret_1m", "ivol", "roe"], "ind", False),
              "ENH_F_NI": (["ret_1m", "ivol", "roe", "turn"], "ind", False),
              "BASE_F_NS": (["ret_1m", "ivol", "roe"], "ind_size", False),
              "ENH_F_NS": (["ret_1m", "ivol", "roe", "turn"], "ind_size", False),
              "ENH_F_FILT": (["ret_1m", "ivol", "roe", "turn"], None, True),
              "ENH_F_NI_FILT": (["ret_1m", "ivol", "roe", "turn"], "ind", True),
              "ENH_F_NS_FILT": (["ret_1m", "ivol", "roe", "turn"], "ind_size", True)}
    results = {k: {"ic": [], "rets": [], "ind_w": [], "mv": []} for k in combos}
    bench_rets = []

    def _neutralize(score, neut, mv_day):
        if neut == "ind" and ind_map:
            ind = pd.Series({c: ind_map.get(c, "NA") for c in score.index}, index=score.index)
            return score.groupby(ind).transform(zscore_series)
        if neut == "ind_size":
            mv = pd.Series({c: mv_day.get(c, np.nan) for c in score.index}, index=score.index)
            ok = mv.notna()
            if ok.sum() < 50:
                return score
            lnm = np.log(mv[ok])
            sc = score[ok]
            b = np.polyfit(lnm, sc, 1)          # 截面回归 score ~ ln(circ_mv), 取残差(去市值倾斜)
            resid = sc - (b[0] * lnm + b[1])
            out = zscore_series(resid)
            out = out.reindex(score.index)
            if ind_map:
                ind = pd.Series({c: ind_map.get(c, "NA") for c in out.index}, index=out.index)
                out = out.groupby(ind).transform(zscore_series)
            return out
        return score

    for i, rb in enumerate(rebal_dates):
        members = rv.load_index_weight(rb)
        if members is None:
            continue
        if i + 1 >= len(rebal_dates):
            continue
        rb_next = rebal_dates[i + 1]
        hi, hn = trade_dates.index(rb), trade_dates.index(rb_next)
        hold_dates = trade_dates[hi + 1:hn + 1]

        # 截面因子值
        fvals = {k: {} for k in combos}
        for code in members:
            f1, f2, ft = ret_1m.get(code), ivol.get(code), turn.get(code)
            fr = fwd.get(code)
            if fr is None or rb not in fr.index:
                continue
            row = {}
            if f1 is not None and rb in f1.index:
                row["ret_1m"] = f1.loc[rb]
            if f2 is not None and rb in f2.index:
                row["ivol"] = f2.loc[rb]
            if ft is not None and rb in ft.index:
                row["turn"] = ft.loc[rb]
            if roe_pit and rb in roe_pit and code in roe_pit[rb].index:
                row["roe"] = roe_pit[rb].loc[code]
            if len(row) < 2:
                continue
            for k, (names, _, _) in combos.items():
                if all(n in row for n in names):
                    fvals[k][code] = pd.Series({n: row[n] for n in names})
        if not fvals["ENH"]:
            continue

        r_ser = pd.Series({c: fwd[c].loc[rb] for c in fvals["ENH"]})
        if len(r_ser) < 50:
            continue
        mv_day = mv_map.get(rb, {})

        # 可交易性过滤: 市值>=30亿, 调仓日非涨停且未停牌
        need_filt = any(f for _, _, f in combos.values())
        filt_codes = None
        if need_filt:
            ok = set(members)
            if mv_day:
                ok = {c for c in ok if mv_day.get(c, np.nan) >= MV_FLOOR_WAN}
            pct_rb = pct_df.loc[rb] if rb in pct_df.index else None
            if pct_rb is not None:
                ok = {c for c in ok if c in pct_rb.index and not _is_limit_up(c, pct_rb[c])}
            if mv_day or pct_rb is not None:
                filt_codes = ok
            n_drop = len(members) - len(ok)
            if n_drop:
                print(f"  [filt] {rb} 剔除 {n_drop} 只(市值地板/涨停/停牌), 剩余 {len(ok)} 只")

        for k, (names, neut, filt) in combos.items():
            if len(fvals[k]) < 50:
                results[k]["ic"].append((rb, np.nan))
                continue
            fdf = pd.DataFrame(fvals[k]).T  # 行=股票, 列=因子
            zdf = fdf.apply(rv.winsorize).apply(zscore_series, axis=0)
            score = zdf.mean(axis=1)
            if filt:
                if filt_codes is None:
                    results[k]["ic"].append((rb, np.nan))
                    continue
                score = score[score.index.intersection(filt_codes)]
                if len(score) < rv.TOP_N:
                    results[k]["ic"].append((rb, np.nan))
                    continue
            score = _neutralize(score, neut, mv_day)
            # IC(与验证脚本同口径: 合成分数 rank vs 未来收益 rank)
            rr = r_ser.reindex(score.index)
            ic = score.rank().corr(rr.rank())
            results[k]["ic"].append((rb, ic))
            picks = score.nlargest(rv.TOP_N).index
            # 行业集中度(Top50 最大行业权重)
            if ind_map:
                w = pd.Series({c: ind_map.get(c, "NA") for c in picks}).value_counts(normalize=True)
                results[k]["ind_w"].append(w.iloc[0])
            # Top50 平均流通市值(亿): 验证市值中性化效果
            mvs = [mv_day.get(c, np.nan) / 10000.0 for c in picks] if mv_day else []
            results[k]["mv"].append(float(np.nanmean(mvs)) if mvs and not all(np.isnan(mvs)) else np.nan)
            port_sub = pct_df.reindex(columns=picks).reindex(hold_dates).fillna(0.0) / 100.0
            net = (1 + port_sub.mean(axis=1)).prod() - 1 - rv.COST_BPS / 10000.0
            results[k]["rets"].append(net)

        if bench_daily is not None:
            b = bench_daily["pct_chg"].reindex(hold_dates).fillna(0.0) / 100.0
            bench_rets.append((1 + b).prod() - 1)

    print("\n" + "=" * 100)
    print(f"合并回测结果 (Top{rv.TOP_N} 月度调仓, {rv.COST_BPS}bps, {rv.START_YEAR}~2026, 基准 {rv.INDEX_CODE};  NI=行业内z-score; NS=市值回归残差+行业内z-score; FILT=市值>=30亿+非涨停+未停牌)")
    print("=" * 100)
    years = len(bench_rets) / 12.0
    bench_nav = (1 + pd.Series(bench_rets)).prod() if bench_rets else np.nan
    bench_cagr = bench_nav ** (1 / years) - 1 if bench_nav == bench_nav and bench_nav > 0 else np.nan
    print(f"{'组合':<12}{'IC均值':>8}{'ICIR':>8}{'年化':>9}{'Sharpe':>8}{'MaxDD':>9}{'月胜率':>8}{'累计超额':>10}{'最大行业':>9}{'均市值亿':>9}")
    print("-" * 100)
    summary = {}
    for k in combos:
        ic_ser = pd.Series([x[1] for x in results[k]["ic"]], index=[x[0] for x in results[k]["ic"]]).dropna()
        pr = pd.Series(results[k]["rets"])
        nav = pr.add(1).prod()
        cagr = nav ** (1 / years) - 1 if nav > 0 else np.nan
        nav_ser = pr.add(1).cumprod()
        assert nav_ser.min() > 0, f"nav 跌破 0 (min={nav_ser.min():.4f}): 组合月收益存在异常值"
        mdd = ((nav_ser.cummax() - nav_ser) / nav_ser.cummax()).max()   # 相对回撤, ∈[0,1]
        assert 0 <= mdd <= 1, f"MaxDD 超出 [0,1]: {mdd:.4f}"
        sharpe = pr.mean() / pr.std(ddof=1) * np.sqrt(12) if pr.std(ddof=1) > 0 else np.nan
        win = (pr > 0).mean()
        ex = nav / bench_nav - 1 if bench_nav == bench_nav else np.nan
        icir = ic_ser.mean() / ic_ser.std(ddof=1) if len(ic_ser) > 2 and ic_ser.std(ddof=1) > 0 else np.nan
        maxind = np.nanmean(results[k]["ind_w"]) if results[k]["ind_w"] else np.nan
        avgmv = np.nanmean(results[k]["mv"]) if results[k]["mv"] else np.nan
        print(f"{k:<12}{ic_ser.mean():>8.4f}{icir:>8.3f}{cagr:>9.2%}{sharpe:>8.2f}{mdd:>9.2%}{win:>8.1%}{ex:>10.2%}{maxind:>9.1%}{avgmv:>9.1f}")
        summary[k] = dict(ic_mean=ic_ser.mean(), icir=icir, cagr=cagr, sharpe=sharpe, mdd=mdd,
                          win=win, excess=ex, nav=nav, n_month=len(pr), max_ind=maxind, avg_mv=avgmv)
    print(f"{'基准000852':<12}{'-':>8}{'-':>8}{bench_cagr:>9.2%}{'-':>8}{'-':>9}{'-':>8}{'-':>10}{'-':>9}{'-':>9}")
    summary["bench"] = dict(cagr=bench_cagr, nav=bench_nav, n_month=len(bench_rets))

    # 因子间月度截面相关系数(调仓日, 所有成分股)
    print("\n[截面相关] 调仓日因子值两两相关(平均|rho|):")
    corr_pairs = {"ret_1m vs ivol": ("ret_1m", "ivol"), "ret_1m vs turn": ("ret_1m", "turn"),
                  "ivol vs turn": ("ivol", "turn")}
    acc = {p: [] for p in corr_pairs}
    for rb in rebal_dates:
        members = rv.load_index_weight(rb)
        if members is None:
            continue
        rows = {}
        for code in members:
            r_ = {}
            for n, d_ in [("ret_1m", ret_1m), ("ivol", ivol), ("turn", turn)]:
                s = d_.get(code)
                if s is not None and rb in s.index:
                    r_[n] = s.loc[rb]
            if len(r_) == 3:
                rows[code] = r_
        if len(rows) < 50:
            continue
        fdf = pd.DataFrame(rows).T
        for p, (a, b) in corr_pairs.items():
            acc[p].append(fdf[a].corr(fdf[b]))
    for p in corr_pairs:
        m = np.nanmean(acc[p])
        print(f"  {p:<20} 平均rho={m:.3f}")

    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, "combo_backtest.txt"), "w", encoding="utf-8") as fh:
        for k, v in summary.items():
            fh.write(f"{k}: {v}\n")
        fh.write(f"corr: { {p: round(float(np.nanmean(acc[p])), 3) for p in corr_pairs} }\n")
    print(f"\n[保存] {os.path.join(OUT_DIR, 'combo_backtest.txt')}")


def main():
    trade_dates = rv.load_trade_dates()
    months = {d[:6]: d for d in trade_dates if d[:4] >= str(rv.START_YEAR)}
    rebal_dates = sorted(months.values())[:-1]
    all_codes = set()
    for rb in rebal_dates:
        members = rv.load_index_weight(rb)
        if members:
            all_codes |= members
    all_codes = sorted(all_codes)
    print(f"[load] 调仓日 {len(rebal_dates)} 个, 成分股 {len(all_codes)} 只")

    stocks, pct_df, _, _, _ = rv.load_panels(trade_dates, all_codes, None)
    ret_1m, ivol, turn, fwd = build_price_factors(stocks, all_codes)
    roe_pit = build_roe_pit(rebal_dates)
    ind_map = load_industry_map()
    print(f"[load] 行业映射 {len(ind_map)} 只")
    mv_map = build_circ_mv(rebal_dates, all_codes)
    print(f"[load] 市值映射 {len(mv_map)} 个调仓日")
    run_combo(rebal_dates, all_codes, ret_1m, ivol, turn, fwd, roe_pit, pct_df, trade_dates, ind_map, mv_map)


if __name__ == "__main__":
    main()

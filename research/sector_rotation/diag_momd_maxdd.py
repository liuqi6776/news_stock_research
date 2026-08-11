# -*- coding: utf-8 -*-
"""诊断: MOM_D 月频回测 — 行业动量 & 收益 双口径对比

组合矩阵 (完全复刻两个脚本的逻辑):
  A(fwd动量+fwd收益)   = sector_stock_rotation.py  -> 期望 MaxDD~17.25%
  B(fwd动量+pct收益)   = 仅改收益计算
  C(pct动量+pct收益)   = sector_stock_frequency.py -> 期望 MaxDD~22.09%

差异根源:
  - 行业动量信号: rotation 用 fwd(未来20日快照); frequency 用 pct_df 逐日复利
  - 组合收益: rotation 用 fwd 快照; frequency 用 pct_df 逐日复利(T+1~next_T)
"""
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from research.factor_dic import run_validation as rv  # noqa: E402
from research.factor_dic import combo_backtest as cb  # noqa: E402
from research.factor_dic import style_factors as sf  # noqa: E402

STUDY_DIR = os.path.join(ROOT, "research", "studies", "study_008_enhancements")
IND_MAP_PATH = os.path.join(STUDY_DIR, "data", "industry_map.parquet")
OUT_DIR = os.path.join(ROOT, "research", "sector_rotation", "results")

COST = 20 / 10000.0
TOP_N = 60


def load_industry_map():
    df = pd.read_parquet(IND_MAP_PATH)
    return dict(zip(df["ts_code"], df["industry"]))


def select_with_limit(scores, code_to_ind, max_per_ind, top_n):
    """按打分降序, 每行业最多max_per_ind只, 取top_n只"""
    sel = []
    ind_count = {}
    for code in scores.sort_values(ascending=False).index:
        ind = code_to_ind.get(code, "其他")
        if ind_count.get(ind, 0) < max_per_ind:
            sel.append(code)
            ind_count[ind] = ind_count.get(ind, 0) + 1
        if len(sel) >= top_n:
            break
    return sel


def select_with_momentum(scores, code_to_ind, ind_momentum, top_inds=10, max_per_ind=4, top_n=40):
    if ind_momentum is None or len(ind_momentum) == 0:
        return select_with_limit(scores, code_to_ind, max_per_ind, top_n)
    top = set(ind_momentum.nlargest(top_inds).index)
    filtered = scores[scores.index.map(lambda c: code_to_ind.get(c, "其他") in top)]
    if len(filtered) < top_n:
        return select_with_limit(scores, code_to_ind, max_per_ind, top_n)
    return select_with_limit(filtered, code_to_ind, max_per_ind, top_n)


def calc_stats(nav_series, n_per_year=12):
    rets = nav_series.pct_change().dropna()
    years = len(rets) / n_per_year
    if years == 0 or nav_series.iloc[-1] <= 0:
        return {k: np.nan for k in ["CAGR", "MaxDD", "Calmar", "FinalNAV"]}
    maxdd = ((nav_series.cummax() - nav_series) / nav_series.cummax()).max()
    return {
        "FinalNAV": nav_series.iloc[-1],
        "CAGR": nav_series.iloc[-1] ** (1 / years) - 1,
        "MaxDD": maxdd,
        "Calmar": (nav_series.iloc[-1] ** (1 / years) - 1) / maxdd if maxdd > 0 else np.nan,
    }


def main():
    trade_dates = rv.load_trade_dates()
    months = {d[:6]: d for d in trade_dates if d[:4] >= str(rv.START_YEAR)}
    rebal = sorted(months.values())
    print(f"调仓日: {rebal[0]} ~ {rebal[-1]} ({len(rebal)}期)")

    all_codes = set()
    for rb in rebal:
        m = rv.load_index_weight(rb)
        if m:
            all_codes |= m
    all_codes = sorted(all_codes)
    print(f"成分股池: {len(all_codes)} 只")

    stocks, pct_df, _, _, _ = rv.load_panels(trade_dates, all_codes, None)
    ret_1m, ivol, turn, fwd = cb.build_price_factors(stocks, all_codes)
    val_map = sf.load_valuation(rebal, all_codes)
    funda_map = sf.build_funda_pit(rebal, all_codes)
    panels = sf.build_factors(val_map, funda_map, rebal)
    code_to_ind = load_industry_map()
    print(f"行业数: {len(set(code_to_ind.values()))}")

    tidx = {d: i for i, d in enumerate(trade_dates)}

    def hold_dates_of(rb, next_rb):
        i0, i1 = tidx[rb], tidx[next_rb]
        return trade_dates[i0 + 1: i1 + 1]

    # ---- 逐月: 因子打分 & 各口径行业动量 ----
    # scores_by_rb: {rb: scores(Series)}
    # ind_mom_fwd:  {rb: Series(industry->ret)}  # rotation 口径(fwd 快照)
    # ind_mom_pct:  {rb: Series(industry->ret)}  # frequency 口径(pct 复利)
    scores_by_rb = {}
    base_by_rb = {}
    ind_mom_fwd = {}
    ind_mom_pct = {}
    for ri, rb in enumerate(rebal):
        members = rv.load_index_weight(rb)
        if members is None:
            continue
        fvals = {}
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
            for name in panels:
                p = panels[name].get(rb)
                if p is not None and code in p.index:
                    v = p.loc[code]
                    if np.isfinite(v):
                        row[name] = v
            if len(row) >= 3:
                fvals[code] = row
        if len(fvals) < TOP_N:
            continue
        fdf = pd.DataFrame(fvals).T
        zdf = fdf.apply(sf.winsorize_series).apply(
            lambda s: (s - s.mean()) / (s.std() + 1e-8), axis=0)
        cols = [c for c in sf.BASE_COLS + ["VAL"] if c in zdf.columns]
        has = zdf[cols].dropna()
        if len(has) < TOP_N:
            continue
        scores = has.mean(axis=1)
        scores_by_rb[rb] = scores
        base_by_rb[rb] = scores.nlargest(TOP_N).index.tolist()

        next_rb = rebal[ri + 1] if ri + 1 < len(rebal) else None
        if next_rb is None:
            continue
        base = base_by_rb[rb]
        # fwd 口径行业收益 (rotation.py)
        ind_f = {}
        for code in base:
            ind = code_to_ind.get(code, "其他")
            r = fwd.get(code)
            if r is not None and rb in r.index:
                v = r.loc[rb]
                if np.isfinite(v):
                    ind_f.setdefault(ind, []).append(v)
        ind_mom_fwd[rb] = pd.Series({k: np.mean(v) for k, v in ind_f.items()})
        # pct 口径行业收益 (frequency.py): 持有期逐股复利后分组平均
        hold_dates = hold_dates_of(rb, next_rb)
        sub = pct_df.reindex(columns=base).reindex(hold_dates).fillna(0.0)
        rc = (1 + sub / 100.0).prod() - 1.0
        rc = rc[rc.notna()]
        if len(rc):
            ind_mom_pct[rb] = rc.groupby(lambda c: code_to_ind.get(c, "其他")).mean()

    print(f"选股准备完成: {len(scores_by_rb)} 期")

    # ---- 三组合回测 (各组合独立维护 prev_holdings 计算换手) ----
    combos = {
        "A_fwdM_fwdR": dict(mom="fwd", ret="fwd"),
        "B_fwdM_pctR": dict(mom="fwd", ret="pct"),
        "C_pctM_pctR": dict(mom="pct", ret="pct"),
    }
    costed = {k: [] for k in combos}
    prev_h_by = {k: None for k in combos}
    for ri, rb in enumerate(rebal):
        if rb not in scores_by_rb:
            continue
        next_rb = rebal[ri + 1] if ri + 1 < len(rebal) else None
        if next_rb is None:
            break
        scores = scores_by_rb[rb]
        for k, cfg in combos.items():
            mom_src = ind_mom_fwd if cfg["mom"] == "fwd" else ind_mom_pct
            prev = mom_src.get(rebal[ri - 1]) if ri > 0 else None
            holdings = select_with_momentum(scores, code_to_ind, prev,
                                            top_inds=10, max_per_ind=4, top_n=40)
            if cfg["ret"] == "fwd":
                rets = []
                for code in holdings:
                    r = fwd.get(code)
                    if r is not None and rb in r.index:
                        v = r.loc[rb]
                        if np.isfinite(v):
                            rets.append(v)
                port = np.mean(rets) if rets else 0.0
            else:
                hold_dates = hold_dates_of(rb, next_rb)
                sub = pct_df.reindex(columns=holdings).reindex(hold_dates).fillna(0.0)
                rc = (1 + sub / 100.0).prod() - 1.0
                rc = rc[rc.notna()]
                port = float(rc.mean()) if len(rc) else 0.0
            if prev_h_by[k] is not None:
                turn = len(set(holdings) - set(prev_h_by[k])) / len(holdings)
                c = turn * COST
            else:
                c = COST
            costed[k].append({"date": next_rb, "ret": port - c})
            prev_h_by[k] = holdings

    print("\n=== MOM_D 三组合对比 (2020-2026, 20bps) ===")
    print(f"{'组合':<16} {'口径':<22} {'FinalNAV':>8} {'CAGR':>8} {'MaxDD':>8} {'Calmar':>7}")
    print("-" * 70)
    for k, cfg in combos.items():
        df = pd.DataFrame(costed[k]).set_index("date")
        df["nav"] = (1 + df["ret"]).cumprod()
        st = calc_stats(df["nav"])
        label = f"{cfg['mom']}动量+{cfg['ret']}收益"
        print(f"  {k:<14} {label:<20} {st['FinalNAV']:>8.2f} {st['CAGR']:>7.2%} "
              f"{st['MaxDD']:>7.2%} {st['Calmar']:>7.2f}")

    # 逐年收益
    print("\n=== 逐年收益 ===")
    yrs = {}
    for k, cfg in combos.items():
        df = pd.DataFrame(costed[k]).set_index("date")
        df["year"] = df.index.astype(str).str[:4]
        yrs[k] = df.groupby("year")["ret"].apply(lambda x: (1 + x).prod() - 1)
    ydf = pd.DataFrame(yrs)
    print(ydf.round(4).to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())

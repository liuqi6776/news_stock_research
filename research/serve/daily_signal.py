# -*- coding: utf-8 -*-
"""
每日信号生成：BASE+VAL+IVW120+RS12 (research/serve 部署版, 基线 v1.1.0)

复刻 risk_control_bt.py 的选股/择时逻辑 + study_008 阶段4/5 约束, 只输出
【最新持有期】的今日操作建议:
  - 滚动日频调仓（有意设计, 2026-08-06 所有者确认; 逐日观察信号流、逐月观察绩效）:
    rb = 最新交易日 (含最新, 不去尾), 每次运行逐日再生持仓
  - Top60 选股 (ret_1m + ivol + turnover_vol_20 + VAL, 截面 zscore 均值取 Top60)
  - 权重: IVW120 (逆波动 w_i ∝ 1/σ_i, 调仓日过去 120 日收益)
  - 阶段4 可交易过滤 (信号名单 → 订单名单): ST / 退市·长期停牌 (60日有效成交<20)
    / 极低流动性 (60日均成交额<300万) / 僵尸股 (120日年化波动<12%, P0-3 波动率下限)
  - 阶段5 集中度约束: 单股≤4% / 行业≤20% / Top5≤20% / 容量≤5%×60日均成交额
  - RS12 择时 (000852/000300 过去240日相对强度, 弱时持 512100 ETF)
  - fail-closed: 订单名单 <10 只 → 沿用最近一期历史信号持仓 (不静默退化)
  - 无 MA20/DD 风控 (基线 v1.0.0 已剥离, 风险控制移交账户治理阈值)
  - signal_date=最新交易日(滚动调仓日, 收盘生成), execution_date=下一交易日(开盘执行)
  - 估值数据缺失时 PIT fallback 到 <=调仓日 的最新估值快照 (与回测同口径)

落盘: research/serve/data/daily/YYYY-MM-DD.json (dashboard 历史记录)

用法:
    python research/serve/daily_signal.py              # 今日信号(落盘 + 打印)
    python research/serve/daily_signal.py --email      # 生成后发邮件
    python research/serve/daily_signal.py --rb 20260422  # 指定调仓日(调试/补历史)
"""
import os
import sys
import json
import time
import types
import argparse

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from research.factor_dic import run_validation as rv
from research.factor_dic import combo_backtest as cb
from research.factor_dic import style_factors as sf
from research.studies.study_008_enhancements import common as C
from research.studies.study_008_enhancements.tradability import Tradability, load_amount_df
from research.studies.study_008_enhancements.concentration import (
    apply_concentration, amount60_at,
)

SERVE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SERVE_DIR, "data", "daily")
TOP_N = rv.TOP_N
NAME_MAP_PATH = os.path.join(ROOT, "stock_name_map.parquet")


def load_idx(code):
    df = pd.read_parquet(os.path.join(rv.IDX_DIR, f"{code}.parquet"))
    df["trade_date"] = df["trade_date"].astype(str).str[:8]
    return df.set_index("trade_date").sort_index()


def load_valuation_pit(rebal_dates, all_codes):
    """估值面板(PIT fallback): {rb: (data_date, DataFrame)} —— 对每个调仓日取 <= rb 的最新估值文件
    回测期 (有精确文件) 行为与 sf.load_valuation 一致; 数据滞后时用旧快照并记录实际日期。
    """
    avail = sorted(f[:8] for f in os.listdir(sf.lf.PE_DIR) if f.endswith(".parquet"))
    out = {}
    for rb in rebal_dates:
        ok = [d for d in avail if d <= rb]
        if not ok:
            continue
        d = ok[-1]
        try:
            df = pd.read_parquet(os.path.join(sf.lf.PE_DIR, f"{d}.parquet"))
        except Exception:
            continue
        df = df.dropna(subset=["pe_ttm", "pb", "ps_ttm", "dv_ttm"], how="all")
        df = df[df["ts_code"].astype(str).isin(all_codes)]
        if not df.empty:
            out[rb] = (d, df.set_index("ts_code"))
    return out


def build_signal(rb, execution_date, picks, ivw_weights, sig_rs12, name_map,
                 order_picks=None, removed=None, fail_closed=False):
    """生成单期信号 dict (基线 v1.1.0: 阶段4 可交易过滤 + 阶段5 集中度约束)

    picks      : 信号名单 (原始 Top60, score 排序)
    ivw_weights: 信号名单的 IVW120 目标权重 (约束前)
    order_picks: 订单名单 (过滤+约束后, [(code, name, weight)], 沿用上期时 weight 来自历史)
    removed    : {code: reason} 阶段4 剔除明细
    fail_closed: 订单名单 <10 只 → 沿用最近一期历史信号持仓
    """
    rs12_on = bool(sig_rs12.loc[rb]) if rb in sig_rs12.index else True
    rs12_val = float(sig_rs12.loc[rb]) if rb in sig_rs12.index else np.nan

    n_order = len(order_picks) if order_picks is not None else len(picks)
    if fail_closed:
        action = f"订单名单过少, 沿用最近一期持仓 ({n_order} 只, fail-closed)"
        position = "沿用上期持仓"
    elif not rs12_on:
        action = "持有 512100 ETF (全额, RS12 弱)"
        position = "512100 ETF"
    else:
        action = f"满仓持有组合 (Top{TOP_N}→订单{n_order}, IVW120 权重 + 集中度约束)"
        position = "股票组合"

    picks_out = []
    for code, score in picks:
        picks_out.append({
            "code": code,
            "name": name_map.get(code, ""),
            "score": round(float(score), 3),
            "target_weight": round(float(ivw_weights.get(code, 0.0)), 5),
        })

    order_out = []
    if order_picks is not None:
        for code, wgt in order_picks:
            order_out.append({
                "code": code,
                "name": name_map.get(code, ""),
                "target_weight": round(float(wgt), 5),
            })

    return {
        "signal_date": rb,           # 月末收盘生成信号日
        "execution_date": execution_date,  # 下一交易日开盘执行日
        "as_of_date": rb,
        "rebalance_date": rb,
        "rs12_on": rs12_on,
        "rs12_value": round(rs12_val, 4) if rs12_val == rs12_val else None,
        "action": action,
        "position": position,
        "picks": picks_out,          # 信号名单 (原始 Top60)
        "picks_count": len(picks_out),
        "order_picks": order_out,    # 订单名单 (阶段4 过滤 + 阶段5 约束后)
        "order_picks_count": len(order_out),
        "tradability_removed": {str(k): str(v) for k, v in (removed or {}).items()},
        "fail_closed": bool(fail_closed),
    }


def _load_prev_order():
    """最近一期历史信号的订单名单 (fail-closed 沿用). 返回 (codes, weights Series) 或 None"""
    today = time.strftime("%Y-%m-%d")
    hist = sorted(f for f in os.listdir(DATA_DIR)
                  if f.endswith(".json") and f[:10] < today)
    for fp in reversed(hist):
        try:
            with open(os.path.join(DATA_DIR, fp), encoding="utf-8") as fh:
                d = json.load(fh)
        except Exception:
            continue
        ops = d.get("order_picks") or d.get("picks") or []
        if ops:
            return ([p["code"] for p in ops],
                    pd.Series({p["code"]: float(p["target_weight"]) for p in ops}))
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--email", action="store_true", help="生成后发送邮件")
    ap.add_argument("--rb", default=None, help="指定调仓日 (YYYYMMDD), 默认最新")
    args = ap.parse_args()

    t0 = time.time()
    trade_dates = rv.load_trade_dates()
    months = {d[:6]: d for d in trade_dates if d[:4] >= str(rv.START_YEAR)}
    rebal = sorted(months.values())  # 含最新一个月 (不回测去尾)

    all_codes = set()
    for rb in rebal:
        m = rv.load_index_weight(rb)
        if m:
            all_codes |= m
    all_codes = sorted(all_codes)
    print(f"[load] 调仓日 {len(rebal)} 个 ({rebal[0]}~{rebal[-1]}), 成分股 {len(all_codes)}", flush=True)

    stocks, pct_df, _, _, _ = rv.load_panels(trade_dates, all_codes, None)
    ret_1m, ivol, turn, fwd = cb.build_price_factors(stocks, all_codes)
    val_map = load_valuation_pit(rebal, all_codes)   # {rb: (data_date, df)}
    funda_map = sf.build_funda_pit(rebal, all_codes)
    val_map_sf = {rb: df for rb, (_, df) in val_map.items()}
    panels = sf.build_factors(val_map_sf, funda_map, rebal)
    print(f"[load] 因子面板完成 ({time.time()-t0:.0f}s)", flush=True)

    sml = load_idx("000852.SH")
    big = load_idx("000300.SH")
    ratio = sml["close"] / big["close"].reindex(sml.index)
    sig_rs12 = ((ratio / ratio.shift(240)).rolling(5).mean() - 1.0 > 0).reindex(rebal)

    name_map = {}
    if os.path.exists(NAME_MAP_PATH):
        try:
            nd = pd.read_parquet(NAME_MAP_PATH)
            name_map = dict(zip(nd["ts_code"].astype(str), nd["name"].astype(str)))
        except Exception:
            pass

    # ---------- 最新(或指定)调仓日选股 ----------
    rb = args.rb if args.rb else rebal[-1]
    if rb not in rebal:
        print(f"[err] 调仓日 {rb} 不在列表 {rebal[0]}~{rebal[-1]}")
        sys.exit(1)
    members = rv.load_index_weight(rb)
    fvals = {}
    for code in members or []:
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
        for pname in panels:
            p = panels[pname].get(rb)
            if p is not None and code in p.index:
                v = p.loc[code]
                if np.isfinite(v):
                    row[pname] = v
        if len(row) >= 3:
            fvals[code] = row
    if len(fvals) < TOP_N:
        print(f"[err] 调仓日 {rb} 有效因子股数 {len(fvals)} < {TOP_N}")
        sys.exit(1)

    fdf = pd.DataFrame(fvals).T
    zdf = fdf.apply(sf.winsorize_series).apply(
        lambda s: (s - s.mean()) / (s.std() + 1e-8), axis=0)
    cols = sf.BASE_COLS + ["VAL"]
    has = zdf[cols].dropna()
    if len(has) < TOP_N:
        print(f"[err] 调仓日 {rb} 完整因子股数 {len(has)} < {TOP_N} (VAL 数据滞后?)")
        sys.exit(1)
    scored = has.mean(axis=1).sort_values(ascending=False)
    picks = list(zip(scored.index.tolist(), scored.values.tolist()))
    top = picks[:TOP_N]

    # IVW120 目标权重: 调仓日过去 120 日个股日收益标准差取逆 (与回测 direction2_hrp._ivw_weights 同口径)
    hi = trade_dates.index(rb)
    win = trade_dates[max(0, hi - 120):hi]
    top_codes = [c for c, _ in top]
    rets = pct_df.reindex(columns=top_codes).reindex(win)
    vols = rets.std() + 1e-12
    ivw = (1.0 / vols)
    ivw = ivw / ivw.sum()

    # ---------- 阶段4: 可交易过滤 (信号名单 → 订单名单, 与回测生产路径同参数) ----------
    amount_df = load_amount_df(types.SimpleNamespace(all_codes=all_codes), trade_dates)
    tf = Tradability(trade_dates, amount_df, lookback=60, min_amount=3e6,
                     min_px_days=20, min_vol=12.0, pct_df=pct_df)
    order_codes, removed = tf(rb, top_codes)
    fail_closed = len(order_codes) < 10
    carry_w = None
    if fail_closed:
        prev = _load_prev_order()
        if prev:
            order_codes, carry_w = prev
            print(f"[warn] {rb} 订单名单过少, fail-closed 沿用最近一期 {len(order_codes)} 只", flush=True)
        else:
            order_codes = top_codes[:10]
            print(f"[warn] {rb} 订单名单过少且无历史信号, 取信号名单前 10 只兜底", flush=True)

    # ---------- 阶段5: 集中度约束 (单股4% / 行业20% / Top5 20% / 容量5%×ADTV60) ----------
    if fail_closed and carry_w is not None:
        w2 = carry_w                          # fail-closed: 保持上期权重不动
    else:
        w_sub = ivw.reindex(order_codes)
        w_sub = w_sub / w_sub.sum()
        ind_map = C.load_industry_map() if os.path.exists(
            os.path.join(C.DATA_DIR, "industry_map.parquet")) else None
        w2 = apply_concentration(
            w_sub,
            ind_map=ind_map,
            cap_stock=0.04, cap_ind=0.20, cap_top5=0.20,
            amount60=amount60_at(amount_df, trade_dates, rb),
            nav_pre=1.0, cap_amount=0.05, scale=1e8)

    # execution_date = 下一交易日 (开盘执行)
    nxt = [d for d in trade_dates if d > rb]
    execution_date = nxt[0] if nxt else rb

    sig = build_signal(rb, execution_date, top, ivw, sig_rs12, name_map,
                       order_picks=list(w2.items()), removed=removed,
                       fail_closed=fail_closed)

    # 数据时效标注
    notes = []
    if rb in val_map:
        val_date = val_map[rb][0]
        if val_date < rb:
            notes.append(f"估值数据截至 {val_date} (VAL 因子使用该旧快照)")
    iw_dates = sorted(f[3:11] for f in os.listdir(rv.IW_DIR) if f.startswith("iw_"))
    iw_avail = [d for d in iw_dates if d <= rb]
    if iw_avail and iw_avail[-1] < rb:
        notes.append(f"成分股清单截至 {iw_avail[-1]} (使用最近一期)")
    sig["data_notes"] = notes
    sig["generated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")

    # ---------- 落盘 ----------
    os.makedirs(DATA_DIR, exist_ok=True)
    fp = os.path.join(DATA_DIR, f"{time.strftime('%Y-%m-%d')}.json")
    with open(fp, "w", encoding="utf-8") as fh:
        json.dump(sig, fh, ensure_ascii=False, indent=2)
    print(f"[saved] {fp}")

    # ---------- 打印摘要 ----------
    print("\n" + "=" * 78)
    print(f"今日操作建议  信号日 {sig['signal_date']}  执行日 {sig['execution_date']}  生成 {sig['generated_at']}")
    print("=" * 78)
    print(f"当前调仓日: {sig['rebalance_date']}")
    print(f"RS12 择时:  {'强(持股)' if sig['rs12_on'] else '弱(持ETF)'}  value={sig['rs12_value']}")
    print(f"操作: {sig['action']}")
    print(f"信号名单: {sig['picks_count']} 只 | 订单名单: {sig['order_picks_count']} 只 | fail-closed: {sig['fail_closed']}")
    if sig["tradability_removed"]:
        rcnt = {}
        for r in sig["tradability_removed"].values():
            rcnt[r] = rcnt.get(r, 0) + 1
        print(f"阶段4 剔除 {len(sig['tradability_removed'])} 只次: " + ", ".join(f"{k}={v}" for k, v in sorted(rcnt.items())))
    print("\n订单名单 Top 10 (约束后权重):")
    for i, p in enumerate(sig["order_picks"][:10], 1):
        nm = p["name"] or "?"
        print(f"  {i:>2}. {p['code']}  {nm:<8}  w={p['target_weight']:.4f}")
    if notes:
        print("\n数据时效提示:")
        for n in notes:
            print(f"  - {n}")

    if args.email:
        try:
            from notify import send_email_html
            lines = [f"<h3>今日操作 ({sig['as_of_date']})</h3>",
                     f"<p><b>{sig['action']}</b></p>",
                     f"<p>RS12: {'强' if sig['rs12_on'] else '弱'} | 调仓日 {sig['rebalance_date']} | "
                     f"信号 {sig['picks_count']} 只 → 订单 {sig['order_picks_count']} 只</p>",
                     "<ul>"]
            for p in sig["order_picks"][:10]:
                lines.append(f"<li>{p['code']} {p['name']} (w {p['target_weight']})</li>")
            lines.append("</ul>")
            if sig["tradability_removed"]:
                lines.append(f"<p style='color:#a00'>⚠️ 阶段4 剔除 {len(sig['tradability_removed'])} 只次: "
                             f"{', '.join(f'{k}={v}' for k, v in sig['tradability_removed'].items()[:8])}</p>")
            for n in notes:
                lines.append(f"<p style='color:#a00'>⚠️ {n}</p>")
            body = "".join(lines)
            send_email_html(f"量化策略今日操作 {sig['as_of_date']}", body)
        except Exception as e:
            print(f"[warn] 邮件发送失败: {e}")


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""
P2-r6: forward paper test 跟踪器（纸面组合绩效记录）

输入: research/serve/data/daily/*.json（daily_signal.py 前向信号, signal_date=月末,
      execution_date=下一交易日）; 按 signal_date 去重取最后一期。

模拟:
  - 组合: order_picks 目标权重（RS12 弱 → 100% 512100.SH; fail_closed → 沿用 order_picks）
  - 日频重估: 持仓期内固定目标权重（buy-and-hold 近似）, 日收益 = Σ w_i × r_i,t
    （权重合计 <1 的部分视为现金 0 收益）; 512100.SH 日收益取自 index_daily parquet
  - 成本: 每次调仓执行日在 NAV 上扣固定 20bps（与回测口径一致）
  - 基准: 000852.SH 从首个执行日起累计; 超额 = 组合/基准 - 1

输出: research/serve/data/paper/paper_nav.csv（date, port_nav, bench_nav, excess, position）
      + 控制台摘要。纸面组合为「进行中的前向 OOS 记录」, 不参与结论升降级,
      累计足够样本后（建议 ≥12 个完整调仓周期）再与回测对照。

用法:
    python research/serve/paper_track.py
"""
import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from research.factor_dic import run_validation as rv  # noqa: E402

SERVE_DIR = os.path.dirname(os.path.abspath(__file__))
SIGNAL_DIR = os.path.join(SERVE_DIR, "data", "daily")
PAPER_DIR = os.path.join(SERVE_DIR, "data", "paper")
COST_BPS = 20
ETFS = {"512100.SH", "510300.SH", "000852.SH", "932000.CSI"}


def load_signals():
    """按 rebalance 日去重（同日多份 JSON 取最后生成的一份）。

    兼容旧 schema: 早期文件无 signal_date/execution_date 字段。
    返回 (sorted_signals, key_field)。
    """
    files = sorted(f for f in os.listdir(SIGNAL_DIR) if f.endswith(".json"))
    out = {}
    for fp in files:
        try:
            with open(os.path.join(SIGNAL_DIR, fp), encoding="utf-8") as fh:
                d = json.load(fh)
        except Exception:
            continue
        key = d.get("signal_date") or d.get("rebalance_date")
        if not key:
            continue
        out[key] = d  # 后写覆盖先写
    return [out[k] for k in sorted(out)]


def _execution_date(s, trade_dates):
    """execution_date: 有则用之（'YYYYMMDD'）; 旧 schema 缺失则取 rebalance 日之后首个交易日。"""
    ed = s.get("execution_date")
    if ed:
        return str(ed)[:8]
    rb = s.get("signal_date") or s.get("rebalance_date")
    for d in trade_dates:
        if d > rb:
            return d
    return rb


def load_idx_ret(code, start_date, end_date):
    """指数/ETF 日收益序列（pct_chg %）。"""
    fp = os.path.join(rv.IDX_DIR, f"{code}.parquet")
    if not os.path.exists(fp):
        return None
    df = pd.read_parquet(fp)
    df["trade_date"] = df["trade_date"].astype(str).str[:8]
    s = df.set_index("trade_date")["pct_chg"].astype(float) / 100.0
    return s.loc[(s.index >= start_date) & (s.index <= end_date)]


def main():
    signals = load_signals()
    if not signals:
        print("[err] 无信号 JSON, 请先运行 daily_signal.py")
        sys.exit(1)
    first_key = signals[0].get("signal_date") or signals[0]["rebalance_date"]
    last_key = signals[-1].get("signal_date") or signals[-1]["rebalance_date"]
    print(f"[load] 信号 {len(signals)} 期: {first_key} ~ {last_key}")

    # 收集涉及的全部股票代码
    all_codes = set()
    for s in signals:
        ops = s.get("order_picks") or s.get("picks") or []
        for p in ops:
            if p["code"] not in ETFS:
                all_codes.add(p["code"])
    all_codes = sorted(all_codes)
    print(f"[load] 涉及个股 {len(all_codes)} 只")

    trade_dates = rv.load_trade_dates()
    stocks, pct_df, _, _, _ = rv.load_panels(trade_dates, all_codes, None)

    for s in signals:
        s["execution_date"] = _execution_date(s, trade_dates)
    first_exec = min(s["execution_date"] for s in signals)
    last_day = trade_dates[-1]
    days = [d for d in trade_dates if first_exec <= d <= last_day]

    # 基准
    bench = load_idx_ret("000852.SH", first_exec, last_day)
    if bench is None or bench.empty:
        print(f"[warn] 基准 000852.SH 无 {first_exec}~{last_day} 数据（index_daily 滞后于日频行情）")
        print("       → 请先运行 research/factor_dic/fetch_indices.py 刷新指数数据（END ≥ 当前交易日），")
        print("         再按流程重新生成数据快照（research/experiments/make_data_manifest.py）后重跑本脚本。")
        print("[exit] 暂无基准重叠交易日, 纸面跟踪未开始（不产生 paper_nav.csv）")
        return 0
    bench_nav = (1 + bench.fillna(0.0)).cumprod()
    bench_nav = bench_nav / bench_nav.iloc[0]

    # 组合模拟
    port_nav = 1.0
    nav_records = []
    holdings = {}       # code -> weight
    hold_ret = {}       # code -> Series(daily ret)
    pos_label = "现金"

    for si, s in enumerate(signals):
        exec_d = s["execution_date"]
        if exec_d < first_exec or exec_d not in days:
            continue
        nxt_d = signals[si + 1]["execution_date"] if si + 1 < len(signals) else last_day
        # 调仓: 换仓 + 扣固定成本
        ops = s.get("order_picks") or s.get("picks") or []
        new_hold = {}
        if s.get("position") == "512100 ETF" or (isinstance(s.get("position"), str) and "ETF" in s["position"]):
            new_hold = {"512100.SH": 1.0}
            pos_label = "512100 ETF"
        else:
            wsum = sum(float(p["target_weight"]) for p in ops)
            new_hold = {p["code"]: float(p["target_weight"]) for p in ops if float(p["target_weight"]) > 0}
            pos_label = "股票组合"
        # 成本: 换手近似 = Σ|Δw|/2, 成本 = 换手 × 双边 bps（上限 20bps 固定口径）
        if holdings:
            codes = set(new_hold) | set(holdings)
            turn = sum(abs(new_hold.get(c, 0.0) - holdings.get(c, 0.0)) for c in codes) / 2.0
        else:
            turn = 1.0
        cost = min(COST_BPS / 10000.0, turn * COST_BPS / 10000.0)
        port_nav *= (1.0 - cost)
        holdings = new_hold

        # 持仓日收益序列
        hold_ret = {}
        for c, w in holdings.items():
            if c in ETFS:
                hold_ret[c] = load_idx_ret(c, exec_d, nxt_d)
            else:
                r = pct_df[c] if c in pct_df.columns else None
                hold_ret[c] = (r / 100.0).loc[(r.index >= exec_d) & (r.index <= nxt_d)] if r is not None else None

        for d in days:
            if d < exec_d or d > nxt_d:
                continue
            if d == exec_d:
                nav_records.append((d, port_nav, pos_label))
                continue
            r = 0.0
            for c, w in holdings.items():
                sr = hold_ret.get(c)
                if sr is not None and d in sr.index:
                    r += w * sr.loc[d]
            port_nav *= (1.0 + r)
            nav_records.append((d, port_nav, pos_label))

    out = pd.DataFrame(nav_records, columns=["date", "port_nav", "position"]).set_index("date")
    out["bench_nav"] = bench_nav.reindex(out.index).ffill()
    out["excess"] = out["port_nav"] / out["bench_nav"] - 1.0

    os.makedirs(PAPER_DIR, exist_ok=True)
    fp = os.path.join(PAPER_DIR, "paper_nav.csv")
    out.to_csv(fp)
    print(f"[saved] {fp}")

    # ---- 摘要 ----
    n = len(out)
    pr = out["port_nav"].pct_change().dropna()
    years = n / 244.0
    cagr = out["port_nav"].iloc[-1] ** (1 / years) - 1 if years > 0 else np.nan
    mdd = ((out["port_nav"].cummax() - out["port_nav"]) / out["port_nav"].cummax()).max()
    sharpe = pr.mean() / pr.std(ddof=1) * np.sqrt(244) if len(pr) > 2 and pr.std(ddof=1) > 0 else np.nan
    print(f"\n== Forward Paper Test 摘要（进行中 OOS, {n} 交易日, "
          f"{out.index[0]} ~ {out.index[-1]}）==")
    print(f"组合 NAV {out['port_nav'].iloc[-1]:.4f} | 基准 NAV {out['bench_nav'].iloc[-1]:.4f} | "
          f"超额 {out['excess'].iloc[-1]:+.2%}")
    print(f"年化 {cagr:.2%} | 年化 Sharpe {sharpe:.2f} | MaxDD {mdd:.2%} | 末态仓位 {out['position'].iloc[-1]}")
    print(f"\n⚠️ 纸面记录为进行中的前向 OOS; 建议累计 ≥12 个完整调仓周期后再与回测对照做结论。")
    return 0


if __name__ == "__main__":
    sys.exit(main())

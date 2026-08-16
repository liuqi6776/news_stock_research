# -*- coding: utf-8 -*-
"""
P2-r6: forward paper test 跟踪器（纸面组合绩效记录）

两条策略分别跟踪:
  - RS12: 输入 research/serve/data/daily/*.json（daily_signal.py 前向信号）
  - ENS_T60_TV12: 输入 research/serve/data/ens/*.json（ens_t60_tv12_signal.py 月末目标权重表）
  均按 signal_date 去重取最后一期; 输出分开落盘, 互不覆盖。

模拟:
  - 组合: order_picks 目标权重（RS12 弱 → 100% 512100.SH）
  - 日频重估: 持仓期内固定目标权重（buy-and-hold 近似）, 日收益 = Σ w_i × r_i,t
    （权重合计 <1 的部分视为现金 0 收益）
  - 行情源: 指数/宽基 ETF 走 index_daily; V8 避险(511990/511260/518880) 走 serve/data/etf,
    511990 货币基金加日息; 个股走日频面板
  - 成本: 每次调仓执行日在 NAV 上扣固定 20bps（与回测口径一致）
  - 基准: 000852.SH 从首个执行日起累计; 超额 = 组合/基准 - 1

输出:
  - RS12: research/serve/data/paper/paper_nav.csv + paper_nav_monthly.csv
  - ENS:  research/serve/data/paper/paper_nav_ens.csv + paper_nav_monthly_ens.csv
  - 控制台摘要 + ENS_T60_TV12 前向闸门进度（滚动 6 个月, 月频口径）。
    纸面组合为「进行中的前向 OOS 记录」, 不参与结论升降级,
    累计足够样本后（建议 ≥12 个完整调仓周期）再与回测对照。
    观察维度（2026-08-06 所有者确认）: 先看逐日信号流, 再看逐月绩效。

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
ENS_DIR = os.path.join(SERVE_DIR, "data", "ens")
HV_DIR = os.path.join(SERVE_DIR, "data", "etf")
PAPER_DIR = os.path.join(SERVE_DIR, "data", "paper")
COST_BPS = 20
ETFS = {"512100.SH", "510300.SH", "000852.SH", "932000.CSI"}
# V8 避险 ETF (与回测 HV_WEIGHTS 一致), 非个股; 511990 货币基金日息见 load_hv_ret
V8_CODES = {"511990.SH", "511260.SH", "518880.SH"}
MM_ANNUAL = 0.018

# ---- ENS_T60_TV12 前向闸门判据（见 definition_freeze.md §四.5） ----
GATE_SHARPE_UP = 0.5          # 升: 前向滚动 6 个月年化 Sharpe > 0.5
GATE_SHARPE_DOWN = 0.0        # 降: Sharpe < 0
GATE_DD_THRESHOLD = -0.2432   # 回撤线: 回测月频 MaxDD -19.32% 放宽 5pp；破线即降
GATE_WINDOW = 6               # 滚动窗口（月）


def load_signals(dir_path=SIGNAL_DIR):
    """按 signal_date 去重（同日多份 JSON 取最后生成的一份）。

    兼容旧 schema: 早期文件无 signal_date/execution_date 字段。
    返回按 signal_date 升序的信号列表；目录不存在返回空。
    """
    if not os.path.isdir(dir_path):
        return []
    files = sorted(f for f in os.listdir(dir_path) if f.endswith(".json"))
    out = {}
    for fp in files:
        try:
            with open(os.path.join(dir_path, fp), encoding="utf-8") as fh:
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
    s = s.sort_index()  # parquet 为降序, 统一升序（cumprod/reindex 依赖时间序）
    return s.loc[(s.index >= start_date) & (s.index <= end_date)]


def load_hv_ret(code, start_date, end_date):
    """V8 避险 ETF 日收益 (close.pct_change; 511990 货币基金加 MM_ANNUAL/242 日息)。"""
    fp = os.path.join(HV_DIR, f"{code}.parquet")
    if not os.path.exists(fp):
        return None
    df = pd.read_parquet(fp)
    s = df["close"].pct_change().dropna()
    s.index = s.index.astype(str).str[:8]
    if code == "511990.SH":
        s = s + MM_ANNUAL / 242.0
    s = s.sort_index()
    return s.loc[(s.index >= start_date) & (s.index <= end_date)]


def load_code_ret(code, start_date, end_date, pct_df):
    """统一日收益序列: 指数/ETF 走 index_daily, V8 走 etf 目录, 个股走日频面板。"""
    if code in ETFS:
        return load_idx_ret(code, start_date, end_date)
    if code in V8_CODES:
        return load_hv_ret(code, start_date, end_date)
    rr = pct_df[code] if code in pct_df.columns else None
    return (rr / 100.0).loc[(rr.index >= start_date) & (rr.index <= end_date)] if rr is not None else None


def _gate_progress(mout):
    """ENS_T60_TV12 前向闸门进度（滚动 6 个月，月频口径）。

    判据见 definition_freeze.md §四.5:
      升: Sharpe_6m > 0.5 且 MaxDD_6m >= -24.32%; 降: Sharpe_6m < 0 或 MaxDD_6m < -24.32%。
    返回 (state, 描述文本); 样本 < 6 个月返回 (None, 提示)。
    """
    if len(mout) < GATE_WINDOW:
        return None, f"样本不足（{len(mout)} 个月 < 需 {GATE_WINDOW} 个月），闸门未启动"

    rets = mout["port_ret"].iloc[-GATE_WINDOW:]
    # 月频回撤需窗口起点 NAV（再往前取 1 个月末净值）
    nav_win = mout["port_nav_end"].iloc[-(GATE_WINDOW + 1):]
    nav = nav_win.values.astype(float)
    cummax = np.maximum.accumulate(nav)
    # 带符号口径 (<=0), 与 freeze 的 -24.32% 对齐; 否则减负阈值恒为正, 破线判据成死代码
    mdd = -float(((cummax - nav) / cummax).max())

    mu = float(rets.mean())
    sd = float(rets.std(ddof=1)) if len(rets) > 1 else 0.0
    sharpe = mu / sd * np.sqrt(12) if sd > 0 else 0.0

    sharpe_to_up = sharpe - GATE_SHARPE_UP      # >0 满足升线
    sharpe_to_down = sharpe - GATE_SHARPE_DOWN  # <0 触发降
    dd_to_line = mdd - GATE_DD_THRESHOLD        # <0 破线（降）；>0 在带内

    if sharpe > GATE_SHARPE_UP and dd_to_line >= 0:
        state = "升 ✅"
    elif sharpe < GATE_SHARPE_DOWN or dd_to_line < 0:
        state = "降 ❌"
    else:
        state = "观察"

    txt = (f"当前状态: {state} | "
           f"Sharpe_6m {sharpe:+.2f}（升线 {GATE_SHARPE_UP:.2f} 差 {sharpe_to_up:+.2f}；降线 {GATE_SHARPE_DOWN:.2f} 余 {sharpe_to_down:+.2f}）| "
           f"MaxDD_6m {mdd:+.2%}（破线 {GATE_DD_THRESHOLD:.2%} 余 {dd_to_line:+.2%}）")

    # 避险月占比（不剔除, 仅信息透明: 闸门验证的是含 s123 择时/避险决策的完整产品）
    if "is_hedge" in mout.columns:
        hedge = mout["is_hedge"].iloc[-GATE_WINDOW:]
        n_hedge = int(hedge.sum())
        hedge_rets = rets[hedge.values]
        v8_contrib = float((1.0 + hedge_rets).prod() - 1.0) if len(hedge_rets) else 0.0
        txt += (f"\n       近 6 个月避险月 {n_hedge} 个（V8 收益贡献 {v8_contrib:+.2%}）")

    return state, txt


def simulate_strategy(signals, trade_dates, pct_df, label):
    """模拟单一策略纸面组合。返回 (out, mout, cost_records) 或 None（基准无数据）。

    - 按 execution_date 去重（同日多份信号取最后生成）
    - 执行日: 换仓 + 扣成本（换手近似 Σ|Δw|/2, 上限 20bps 固定口径）
    - 非执行日: 持有收益（目标权重固定, buy-and-hold 近似; 权重合计<1 部分视为现金 0 收益）
    """
    for s in signals:
        s["execution_date"] = _execution_date(s, trade_dates)
    first_exec = min(s["execution_date"] for s in signals)
    last_day = trade_dates[-1]
    days = [d for d in trade_dates if first_exec <= d <= last_day]

    bench = load_idx_ret("000852.SH", first_exec, last_day)
    if bench is None or bench.empty:
        print(f"[warn] {label} 基准 000852.SH 无 {first_exec}~{last_day} 数据（index_daily 滞后于日频行情）")
        print("       → 请先运行 research/factor_dic/fetch_indices.py 刷新指数数据（END ≥ 当前交易日），")
        print("         再按流程重新生成数据快照（research/experiments/make_data_manifest.py）后重跑本脚本。")
        return None
    bench_nav = (1 + bench.fillna(0.0)).cumprod()
    bench_nav = bench_nav / bench_nav.iloc[0]

    by_exec = {}
    for s in signals:
        by_exec[s["execution_date"]] = s
    execs = sorted(by_exec)

    port_nav = 1.0
    nav_records = []
    cost_records = []   # (execution_date, cost_fraction)
    holdings = {}       # code -> weight
    hold_ret = {}       # code -> Series(daily ret)
    pos_label = "现金"
    exec_ptr = 0

    for d in days:
        if exec_ptr < len(execs) and execs[exec_ptr] == d:
            s = by_exec[d]
            new_hold = {}
            if s.get("position") == "512100 ETF" or (isinstance(s.get("position"), str) and "ETF" in s["position"]):
                new_hold = {"512100.SH": 1.0}
                pos_label = "512100 ETF"
            else:
                ops = s.get("order_picks") or s.get("picks") or []
                new_hold = {p["code"]: float(p["target_weight"]) for p in ops if float(p["target_weight"]) > 0}
                pos_label = s.get("position") or "股票组合"
            # 成本: 换手近似 = Σ|Δw|/2, 成本 = 换手 × 双边 bps（上限 20bps 固定口径）
            if holdings:
                codes = set(new_hold) | set(holdings)
                turn = sum(abs(new_hold.get(c, 0.0) - holdings.get(c, 0.0)) for c in codes) / 2.0
            else:
                turn = 1.0
            cost = min(COST_BPS / 10000.0, turn * COST_BPS / 10000.0)
            port_nav *= (1.0 - cost)
            cost_records.append((d, cost))
            holdings = new_hold
            # 持仓日收益序列（执行日至末日; 指数/ETF 走 index_daily, V8 走 etf 目录, 个股走日频面板）
            hold_ret = {}
            for c in holdings:
                hold_ret[c] = load_code_ret(c, d, last_day, pct_df)
            exec_ptr += 1

        r = 0.0
        for c, w in holdings.items():
            sr = hold_ret.get(c)
            if sr is not None and d in sr.index:
                r += w * float(sr.loc[d])
        if r:
            port_nav *= (1.0 + r)
        nav_records.append((d, port_nav, pos_label))

    out = pd.DataFrame(nav_records, columns=["date", "port_nav", "position"]).set_index("date")
    out["bench_nav"] = bench_nav.reindex(out.index).ffill()
    out["excess"] = out["port_nav"] / out["bench_nav"] - 1.0

    # ---- 月频汇总（自然月; "再看逐月"观察维度） ----
    monthly_rows = []
    for month, g in out.groupby(out.index.str[:6], sort=True):
        p0, p1 = g["port_nav"].iloc[0], g["port_nav"].iloc[-1]
        b0, b1 = g["bench_nav"].iloc[0], g["bench_nav"].iloc[-1]
        port_ret = p1 / p0 - 1.0
        bench_ret = b1 / b0 - 1.0
        month_cost = sum(c for d, c in cost_records if str(d)[:6] == month)
        monthly_rows.append({
            "month": month,
            "days": len(g),
            "port_ret": port_ret,
            "bench_ret": bench_ret,
            "excess": (1.0 + port_ret) / (1.0 + bench_ret) - 1.0,
            "turnover_cost": month_cost,
            "port_nav_end": p1,
            "bench_nav_end": b1,
            "is_hedge": bool(g["position"].astype(str).str.contains("避险").any()),
        })
    mout = pd.DataFrame(monthly_rows).set_index("month")
    return out, mout, cost_records


def _summary(out, mout, label):
    n = len(out)
    pr = out["port_nav"].pct_change().dropna()
    years = n / 244.0
    cagr = out["port_nav"].iloc[-1] ** (1 / years) - 1 if years > 0 else np.nan
    mdd = ((out["port_nav"].cummax() - out["port_nav"]) / out["port_nav"].cummax()).max()
    sharpe = pr.mean() / pr.std(ddof=1) * np.sqrt(244) if len(pr) > 2 and pr.std(ddof=1) > 0 else np.nan
    print(f"\n== {label} Forward Paper Test 摘要（进行中 OOS, {n} 交易日, "
          f"{out.index[0]} ~ {out.index[-1]}）==")
    print(f"组合 NAV {out['port_nav'].iloc[-1]:.4f} | 基准 NAV {out['bench_nav'].iloc[-1]:.4f} | "
          f"超额 {out['excess'].iloc[-1]:+.2%}")
    print(f"年化 {cagr:.2%} | 年化 Sharpe {sharpe:.2f} | MaxDD {mdd:.2%} | 末态仓位 {out['position'].iloc[-1]}")


def main():
    rs12_signals = load_signals(SIGNAL_DIR)
    ens_signals = load_signals(ENS_DIR)

    if not rs12_signals and not ens_signals:
        print("[err] 无信号 JSON, 请先运行 daily_signal.py / ens_t60_tv12_signal.py")
        sys.exit(1)

    # 收集涉及的全部个股代码（指数/ETF/V8 走各自行情源, 不当作个股）
    all_codes = set()
    for s in rs12_signals + ens_signals:
        ops = s.get("order_picks") or s.get("picks") or []
        for p in ops:
            if p["code"] not in ETFS and p["code"] not in V8_CODES:
                all_codes.add(p["code"])
    all_codes = sorted(all_codes)
    print(f"[load] RS12 {len(rs12_signals)} 期 | ENS {len(ens_signals)} 期 | 涉及个股 {len(all_codes)} 只")

    trade_dates = rv.load_trade_dates()
    _, pct_df, _, _, _ = rv.load_panels(trade_dates, all_codes, None)

    os.makedirs(PAPER_DIR, exist_ok=True)

    # ---- RS12 (daily, 存量) ----
    if rs12_signals:
        res = simulate_strategy(rs12_signals, trade_dates, pct_df, "RS12")
        if res is not None:
            out, mout, _ = res
            fp = os.path.join(PAPER_DIR, "paper_nav.csv")
            mfp = os.path.join(PAPER_DIR, "paper_nav_monthly.csv")
            out.to_csv(fp)
            mout.to_csv(mfp)
            print(f"[saved] {fp}")
            print(f"[saved] {mfp}")
            print(mout.round(6).to_string())
            _summary(out, mout, "RS12")

    # ---- ENS_T60_TV12 (ens, 前向闸门真样本) ----
    if ens_signals:
        res = simulate_strategy(ens_signals, trade_dates, pct_df, "ENS_T60_TV12")
        if res is not None:
            out, mout, _ = res
            fp = os.path.join(PAPER_DIR, "paper_nav_ens.csv")
            mfp = os.path.join(PAPER_DIR, "paper_nav_monthly_ens.csv")
            out.to_csv(fp)
            mout.to_csv(mfp)
            print(f"[saved] {fp}")
            print(f"[saved] {mfp}")
            print(mout.round(6).to_string())
            _summary(out, mout, "ENS_T60_TV12")

            print("\n== ENS_T60_TV12 前向闸门（滚动 6 个月，月频口径）==")
            _state, _txt = _gate_progress(mout)
            print(_txt)

    print(f"\n⚠️ 纸面记录为进行中的前向 OOS; 建议累计 ≥12 个完整调仓周期后再与回测对照做结论。")
    return 0


if __name__ == "__main__":
    sys.exit(main())

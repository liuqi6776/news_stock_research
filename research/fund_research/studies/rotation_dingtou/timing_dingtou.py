# -*- coding: utf-8 -*-
"""
4 信号择时定投回测 (低估投 / 中性停 / 高估卖)
================================================

信号 (每个季度轮换点计算, 满足为 1):
  S1 估值分位   沪深300 PE-TTM 处近10年(2400交易日)滚动分位 <20%
  S2 股债性价比  ERP = 1/PE-TTM - 10年国债, 高于近10年均值+1σ
  S3 回撤深度   沪深300 收盘距前高回撤 <= -25%
  S4 池内绝对收益 当季 4433 通过者近1年收益中位数 < 0

档位:
  >=3 强低估 / 2 温和低估 / 1 中性 / 0 高估

策略:
  B 无择时对照:  每季定投3000 + 动态4433全仓再平衡 (即 run_rotation 主策略)
  择时定投:     低估(>=2信号) 每季定投3000并全仓再平衡; 中性(1) 停止定投;
                卖出规则可调 (--sell-sig 由 main 内循环覆盖):
                A0 仅高估清仓 (n_sig==0)         [现状, 2021-2026 从未触发]
                A1 中性即清仓 (n_sig<=1)         [彻底放宽卖出]
                A2 高估清仓 + 中性减半           [分级放宽, 渐进止盈]
  卖出后资金留在组合内持币, 等待低估区重新定投入场。

用法:
  C:/Users/liuqi/anaconda3/python.exe studies/rotation_dingtou/timing_dingtou.py \
    --start 2021-01-01 --end 2026-08-06 --amount 3000
"""
import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # fund_research
sys.path.insert(0, ROOT)
import run_rotation as rr  # noqa: E402

CACHE_DIR = os.path.join(ROOT, "cache")
PE_CACHE = os.path.join(CACHE_DIR, "pe_csi300.parquet")
BOND_CACHE = os.path.join(CACHE_DIR, "bond10y.parquet")
PCT_WIN = 2400        # 近10年窗口 (交易日)
PE_QUANT = 0.20       # S1 分位阈值
ERP_Z = 1.0           # S2 z-score 阈值
DD_THRESH = -0.25     # S3 回撤阈值
LOW_SIG = 2           # 低估区最少信号数


def fetch_pe_csi300():
    """沪深300 PE-TTM 日频 (绕开 akshare 1.18.39 date 解析 bug)"""
    if os.path.exists(PE_CACHE):
        return pd.read_parquet(PE_CACHE)
    from datetime import datetime
    import requests
    import py_mini_racer
    from akshare.stock_feature.stock_a_pe_and_pb import hash_code
    from akshare.stock_feature.stock_a_indicator import get_cookie_csrf
    js = py_mini_racer.MiniRacer()
    js.eval(hash_code)
    token = js.call("hex", datetime.now().date().isoformat()).lower()
    r = requests.get("https://legulegu.com/api/stockdata/index-basic-pe",
                     params={"token": token, "indexCode": "000300.SH"},
                     **get_cookie_csrf(url="https://legulegu.com/stockdata/sz50-ttm-lyr"))
    df = pd.DataFrame(r.json()["data"])
    d = df["date"]
    if pd.api.types.is_numeric_dtype(d):
        df["date"] = pd.to_datetime(d, unit="ms", utc=True).dt.tz_convert("Asia/Shanghai").dt.date
    else:
        df["date"] = pd.to_datetime(d).dt.date
    df = df[["date", "close", "ttmPe"]].rename(columns={"ttmPe": "pe_ttm"})
    df = df.dropna(subset=["pe_ttm"]).sort_values("date").set_index("date")
    df.index = pd.to_datetime(df.index)
    os.makedirs(CACHE_DIR, exist_ok=True)
    df.to_parquet(PE_CACHE)
    print(f"  沪深300 PE-TTM: {df.index[0].date()} ~ {df.index[-1].date()} ({len(df)} 行)")
    return df


def fetch_bond10y():
    """中债 10 年国债收益率"""
    if os.path.exists(BOND_CACHE):
        return pd.read_parquet(BOND_CACHE)
    import akshare as ak
    b = ak.bond_zh_us_rate(start_date="20100101")
    b["date"] = pd.to_datetime(b["日期"])
    df = b[["date", "中国国债收益率10年"]].dropna().rename(columns={"中国国债收益率10年": "y10"})
    df = df.sort_values("date").set_index("date")
    os.makedirs(CACHE_DIR, exist_ok=True)
    df.to_parquet(BOND_CACHE)
    print(f"  10年国债: {df.index[0].date()} ~ {df.index[-1].date()} ({len(df)} 行)")
    return df


def _rolling_pct(s, d, win=PCT_WIN):
    """s 在日期 d 之前最近 win 个点内, 当前值的分位"""
    sub = s[s.index <= d]
    if len(sub) < max(200, win // 4):
        return np.nan
    w = sub.iloc[-win:]
    return float((w < w.iloc[-1]).mean())


def _zscore(s, d, win=PCT_WIN):
    """s 在日期 d 之前最近 win 个点的 z-score"""
    sub = s[s.index <= d]
    if len(sub) < max(200, win // 4):
        return np.nan
    w = sub.iloc[-win:]
    mu, sd = w.mean(), w.std()
    return float((w.iloc[-1] - mu) / sd) if sd > 0 else np.nan


def compute_signals(qdates, sel_dyn, out, codes):
    """返回 DataFrame: date, s1..s4, n_sig, zone"""
    pe = fetch_pe_csi300()
    bond = fetch_bond10y()
    close = pe["close"]
    dd = close / close.cummax() - 1.0
    code_idx = {c: j for j, c in enumerate(codes)}

    erp = 1.0 / pe["pe_ttm"] - bond["y10"].reindex(pe.index).ffill()  # 日频 ERP

    rows = []
    for i, d in enumerate(qdates):
        d = pd.Timestamp(d)
        pe_v = float(pe["pe_ttm"].asof(d))
        s1 = 1 if _rolling_pct(pe["pe_ttm"], d) < PE_QUANT else 0
        s2 = 1 if _zscore(erp, d) > ERP_Z else 0
        s3 = 1 if float(dd.asof(d)) <= DD_THRESH else 0
        # S4: 当季 4433 通过者近1年收益中位数 < 0
        sel = sel_dyn[i]
        if len(sel) > 0:
            idx = [code_idx[c] for c in sel]
            ret1y = out["ret_1y"][i, idx]
            s4 = 1 if np.nanmedian(ret1y) < 0 else 0
        else:
            s4 = 0
        n = s1 + s2 + s3 + s4
        zone = "强低估" if n >= 3 else ("温和低估" if n == 2 else ("中性" if n == 1 else "高估"))
        rows.append({"date": d, "pe_ttm": round(pe_v, 2),
                     "pe_pct": round(_rolling_pct(pe["pe_ttm"], d), 3),
                     "erp_z": round(_zscore(erp, d), 2),
                     "dd_pct": round(float(dd.asof(d)) * 100, 1),
                     "s1": s1, "s2": s2, "s3": s3, "s4": s4,
                     "n_sig": n, "zone": zone})
    return pd.DataFrame(rows)


def run_timed(qdates, sel_dyn, signals, amount, mode, union, sell_sig=0):
    """mode: 'both'(买+卖) / 'buy_only'(只买不卖) / 'none'(无择时)

    sell_sig 卖出触发阈值 (信号数 <= sell_sig 即止盈; 卖出后资金留在组合内持币):
      0 = 仅高估清仓 (n_sig==0)          [现状, 2021-2026 区间从未触发]
      1 = 中性即清仓 (n_sig<=1)          [彻底放宽]
      2 = 高估清仓 + 中性减半            [分级放宽, 更贴近实战渐进止盈]
    """
    port = rr.Portfolio()
    eq_parts = []
    invested = 0.0
    n_buy = 0
    n_sell = 0
    n_half = 0
    for i, d in enumerate(qdates):
        n_sig = signals[i]
        if mode == "none" or (mode == "buy_only" and n_sig >= LOW_SIG) or n_sig >= LOW_SIG:
            rr.rebalance_full(port, sel_dyn[i], d, amount)
            invested += amount
            n_buy += 1
        elif mode == "both" and n_sig <= max(0, sell_sig):  # 触发止盈
            if port.lots:  # 只有实际持有才卖出
                if n_sig == 1 and sell_sig == 2:  # 中性 -> 减半仓 (渐进止盈)
                    for c in list(port.lots):
                        nav = rr.nav_at(c, d)
                        if np.isfinite(nav):
                            shares = sum(sh for sh, _ in port.lots[c])
                            port._sell_lots(c, nav, d, max_shares=shares * 0.5)
                    n_half += 1
                else:  # 清仓止盈
                    for c in list(port.lots):
                        nav = rr.nav_at(c, d)
                        if np.isfinite(nav):
                            port.sell_all(c, nav, d)
                    n_sell += 1
        # 中性(1): 停止定投持有不动
        d_end = qdates[i + 1] if i + 1 < len(qdates) else rr.END
        cash, shares = port.snapshot()
        seg = rr.segment_equity(cash, shares, d, d_end, union)
        if not seg.empty:
            eq_parts.append(seg)
    eq = pd.concat(eq_parts) if eq_parts else pd.Series(dtype=float)
    return eq, invested, n_buy, n_sell, n_half


def summarize(name, equity, invested, qdates, n_buy, amount):
    final = float(equity.iloc[-1])
    ret = final / invested - 1.0 if invested > 0 else np.nan
    cf = [(pd.Timestamp(qdates[i]), -amount) for i in range(n_buy)] + [(equity.index[-1], final)]
    irr = rr.xirr(cf) if len(cf) >= 2 else np.nan
    mdd = float((equity / equity.cummax() - 1.0).min())
    return {"strategy": name, "invested": invested, "final_value": round(final, 2),
            "total_return_pct": round(ret * 100, 2),
            "xirr_pct": round(irr * 100, 2) if np.isfinite(irr) else np.nan,
            "mdd_pct": round(mdd * 100, 2), "buy_quarters": n_buy}


def main():
    ap = argparse.ArgumentParser(description="4信号择时定投回测")
    ap.add_argument("--start", default="2021-01-01")
    ap.add_argument("--end", default="2026-08-06")
    ap.add_argument("--amount", type=float, default=3000)
    ap.add_argument("--rebuild-panel", action="store_true")
    args = ap.parse_args()

    t0 = time.time()
    basic = rr.load_basic()
    active = basic[basic["fund_type"].isin(rr.ACTIVE_TYPES)]
    codes = active["code"].astype(str).tolist()
    basic_ft = basic.set_index("code")["fund_type"]
    qdates = [pd.Timestamp(d) for d in pd.date_range(args.start, args.end, freq="Q")]
    rr.CONTRIBUTION = args.amount
    rr.END = pd.Timestamp(args.end)
    print(f"动态池 {len(codes)} 只, 季度点 {len(qdates)} 个")

    out, union = rr.compute_window_sums(codes, qdates, rebuild=args.rebuild_panel)
    sel_dyn = [rr.select_4433(out, codes, basic_ft, i) for i in range(len(qdates))]
    sig = compute_signals(qdates, sel_dyn, out, codes)

    sig_out = os.path.join(rr.RESULTS_DIR, "timing_signals.csv")
    sig.to_csv(sig_out, index=False, encoding="utf-8-sig")
    print(f"\n信号明细: {sig_out}")
    print(sig[["date", "pe_pct", "erp_z", "dd_pct", "s1", "s2", "s3", "s4", "n_sig", "zone"]].to_string(index=False))

    results = []
    curves = {}
    n_sig_list = sig["n_sig"].tolist()
    # 基准: 无择时季度定投
    eq, inv, nb, ns, nh = run_timed(qdates, sel_dyn, n_sig_list, args.amount, "none", union)
    r = summarize("B 无择时季度定投", eq, inv, qdates, nb, args.amount)
    r["n_sell"], r["n_half"] = ns, nh
    results.append(r)
    curves["B 无择时季度定投"] = eq
    print(f"  B 无择时季度定投: 买入{nb}次 期末 {r['final_value']} 总投入 {inv}")
    # 3 种卖出规则
    for sell_sig, tag in [(0, "A0 择时+仅高估清仓(现状)"),
                          (1, "A1 择时+中性即清仓(放宽)"),
                          (2, "A2 择时+高估清仓/中性减半(分级)")]:
        eq, inv, nb, ns, nh = run_timed(qdates, sel_dyn, n_sig_list, args.amount, "both", union, sell_sig)
        r = summarize(tag, eq, inv, qdates, nb, args.amount)
        r["n_sell"], r["n_half"] = ns, nh
        results.append(r)
        curves[tag] = eq
        print(f"  {tag}: 买入{nb}次 卖出{ns}次 减半{nh}次 期末 {r['final_value']} 总投入 {inv}")

    summary = pd.DataFrame(results)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 200)
    print("\n" + "=" * 110)
    print(f"4 信号择时定投回测 ({args.start} ~ {args.end}, 低估区每季 {args.amount:.0f} 元)")
    print("=" * 110)
    print(summary.to_string(index=False))
    print("=" * 110)
    summary.to_csv(os.path.join(rr.RESULTS_DIR, "timing_summary.csv"), index=False, encoding="utf-8-sig")

    # 画图: 净值曲线 + 信号数柱状
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
        plt.rcParams["axes.unicode_minus"] = False
        fig, axes = plt.subplots(3, 1, figsize=(13, 11), sharex=True,
                                 gridspec_kw={"height_ratios": [2.2, 1, 1.2]})
        for name, eq in curves.items():
            axes[0].plot(eq.index, eq.values, label=name, linewidth=1.5)
        axes[0].set_ylabel("组合市值 (元)")
        axes[0].legend()
        axes[0].grid(alpha=0.3)
        axes[0].set_title(f"4 信号择时定投 vs 无择时 ({args.start} ~ {args.end})")

        sigd = sig.set_index("date")
        axes[1].bar(sigd.index, sigd["n_sig"], color="#4B3FE3", alpha=0.8, width=20)
        axes[1].axhline(LOW_SIG - 0.5, color="#E8463A", linestyle="--", linewidth=1)
        axes[1].set_ylabel("满足信号数")
        axes[1].set_yticks([0, 1, 2, 3, 4])
        axes[1].grid(alpha=0.3)

        zone_color = {"强低估": "#22A5F7", "温和低估": "#6F6FFF", "中性": "#A9AEFF", "高估": "#D3D4DA"}
        for i, row in sigd.iterrows():
            axes[2].axvspan(i, i + pd.Timedelta(days=70), alpha=0.5, color=zone_color[row["zone"]])
        axes[2].set_ylabel("择时区间")
        import matplotlib.patches as mpatches
        axes[2].legend(handles=[mpatches.Patch(color=c, label=k) for k, c in zone_color.items()], loc="upper left")
        fig.tight_layout()
        png = os.path.join(rr.RESULTS_DIR, "timing_compare.png")
        fig.savefig(png, dpi=130)
        print(f"对比图: {png}")
    except Exception as e:
        print(f"画图跳过: {e}")

    print(f"总用时 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""方案C: 行业ETF等权优化回测（结合 quant_conclusion 已验证结论）

优化来源（liuqi6776/quant_conclusion）:
  1. defensive_asset_allocation.md  ✅空仓期持V8避险资产（货基33%+国债33%+黄金33%,
     511990/511260/518880, 月度再平衡）而非现金 —— V8 MaxDD 11.6%/卡玛0.93
  2. risk_control.md                ✅MA20三档趋势风控（T-1信号T日生效, 无前视）
     close>=MA20→满仓; MA20*0.98<=close<MA20→半仓; close<MA20*0.98→空仓
     —— 2018-2019独立OOS支持: ETF从-6.76%→+1.01%, MaxDD 40%→23%
  3. regime_study.md                RS12弱市风控思路（此处保留S1/S2/S3估值信号做买入择时）

版本:
  A   无脑全仓                                   (方案A基线)
  B   timed_buy严格(S>=3入场/n_sig=0清仓)+现金    (方案B最优)
  C1  timed_buy严格 + 空仓期持V8
  C2  无脑全仓 + MA20三档
  C3  timed_buy严格 + MA20三档 + V8
  C4  timed_buy严格 + MA20三档 + 现金（分离MA20与V8贡献）

MA20标的 = 行业ETF日频等权组合自身净值（风控针对策略本身）。
数据: 26只行业ETF(industry_etf/) + 3只避险ETF(serve/data/etf/) + S1/S2/S3信号。
"""
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "research", "fund_research", "studies", "rotation_dingtou"))

from timing_dingtou import fetch_pe_csi300, fetch_bond10y, _rolling_pct, _zscore  # noqa: E402

ETF_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "industry_etf")
HV_DIR = "c:/Users/liuqi/quant_system_v2/research/serve/data/etf"  # 避险资产
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(OUT_DIR, exist_ok=True)
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

COST = 30 / 10000.0   # 双边20bps手续费 + 10bps滑点
MM_ANNUAL = 0.018      # 货基年化分红假设（与仓库一致）
MA20_MA = 20
MA20_DEEP = 0.98       # 三档深位阈值
PE_QUANT = 0.20
ERP_Z = 1.0
DD_THRESH = -0.25
HV_WEIGHTS = {"511990.SH": 1 / 3, "511260.SH": 1 / 3, "518880.SH": 1 / 3}  # V8

INDUSTRY_ETFS = [
    ("农林牧渔", "159825.SZ"), ("基础化工", "159870.SZ"), ("钢铁", "515210.SH"),
    ("有色金属", "512400.SH"), ("电子", "512480.SH"), ("汽车", "516110.SH"),
    ("家用电器", "159996.SZ"), ("食品饮料", "515170.SH"), ("医药生物", "512010.SH"),
    ("公用事业", "159611.SZ"), ("交通运输", "516320.SH"), ("房地产", "512200.SH"),
    ("社会服务", "159766.SZ"), ("建筑材料", "159745.SZ"), ("建筑装饰", "516950.SH"),
    ("电力设备", "515030.SH"), ("国防军工", "512660.SH"), ("计算机", "512720.SH"),
    ("传媒", "512980.SH"), ("通信", "515880.SH"), ("银行", "512800.SH"),
    ("非银金融", "512880.SH"), ("煤炭", "515220.SH"), ("石油石化", "159930.SZ"),
    ("环保", "512580.SH"), ("机械设备", "159886.SZ"),
]


def load_industry_daily():
    """行业ETF日收益面板: {code: Series(收益, index=YYYYMMDD)}, 只取已上市"""
    panel = {}
    for name, code in INDUSTRY_ETFS:
        fp = os.path.join(ETF_DIR, f"{code}.parquet")
        if not os.path.exists(fp):
            print(f"  [skip] {code} 无缓存", flush=True)
            continue
        df = pd.read_parquet(fp)
        d = df.copy()
        d["pct_chg"] = d["pct_chg"].fillna(0) / 100.0
        d["trade_date"] = d["trade_date"].astype(str).str[:8]
        s = d.set_index("trade_date")["pct_chg"]
        panel[code] = s
    return panel


def load_hv_daily():
    """避险资产日收益: {code: Series(收益, index=YYYYMMDD)}"""
    out = {}
    for code in HV_WEIGHTS:
        df = pd.read_parquet(os.path.join(HV_DIR, f"{code}.parquet"))
        s = df["close"].pct_change().dropna()
        s.index = s.index.astype(str).str[:8]
        if code == "511990.SH":  # 货基: 价格几乎不动, 真实收益来自分红
            s = s + MM_ANNUAL / 242.0
        out[code] = s
    return out


def build_series(panel):
    """统一日期轴(取所有ETF日期的并集), 返回 日期xETF 收益DataFrame + 等权组合日收益"""
    all_dates = sorted(set().union(*[set(s.index) for s in panel.values()]))
    df = pd.DataFrame(index=all_dates)
    for code, s in panel.items():
        df[code] = s.reindex(all_dates)
    ew = df.mean(axis=1).fillna(0)  # 已上市等权（NaN列自动跳过）
    return df, ew


def hv_monthly_ret(hv):
    """V8月收益: 三资产日恒权(≈月再平衡, 差异可忽略) → 月复利"""
    all_dates = sorted(set().union(*[set(s.index) for s in hv.values()]))
    df = pd.DataFrame(index=all_dates)
    for code, s in hv.items():
        df[code] = s.reindex(all_dates)
    daily = (df * pd.Series(HV_WEIGHTS)).sum(axis=1).fillna(0)
    d = daily.copy()
    d.index = pd.Index(d.index, name="ym")
    return (1 + daily).groupby(daily.index.str[:6]).prod() - 1.0


def ma20_signal(ew_daily):
    """MA20三档日频仓位(T-1信号T日生效), 返回 (w序列, 风控后日收益, 换仓成本序列)"""
    nav = (1 + ew_daily).cumprod()
    ma20 = nav.rolling(MA20_MA).mean()
    w = pd.Series(1.0, index=nav.index)
    below = nav < ma20
    deep_below = nav < ma20 * MA20_DEEP
    w[below & ~deep_below] = 0.5
    w[deep_below] = 0.0
    w = w.shift(1).fillna(1.0)  # T-1信号T日生效
    dw = w.diff().abs().fillna(0)  # 换仓幅度
    cost = dw * COST
    ret = ew_daily * w - cost  # 风控后日收益（已扣换仓成本）
    return w, ret


def monthly_from_daily(daily_ret):
    """日收益 → 月复利收益 series(index=YYYYMM)"""
    d = daily_ret.copy()
    return (1 + d).groupby(d.index.str[:6]).prod() - 1.0


def build_signals(ym_list):
    """每月末(调仓日)计算 S1/S2/S3（同方案B）"""
    pe = fetch_pe_csi300()
    bond = fetch_bond10y()
    close = pe["close"]
    dd = close / close.cummax() - 1.0
    erp = 1.0 / pe["pe_ttm"] - bond["y10"].reindex(pe.index).ffill()
    rows = []
    for ym in ym_list:
        d = pd.Timestamp(ym + "01") + pd.offsets.MonthEnd(0)
        s1 = 1 if _rolling_pct(pe["pe_ttm"], d) < PE_QUANT else 0
        s2 = 1 if _zscore(erp, d) > ERP_Z else 0
        s3 = 1 if float(dd.asof(d)) <= DD_THRESH else 0
        rows.append({"ym": ym, "s1": s1, "s2": s2, "s3": s3, "n_sig": s1 + s2 + s3})
    return pd.DataFrame(rows).set_index("ym")


def run(monthly_nav_panel, sig, plain_m, ma20_m, v8_m, mode="none", use_ma20=False, use_v8=False, low_sig=3, cost=COST):
    """月度调仓回测。
    monthly_nav_panel: 月末NAV面板(ym×ETF), 用于等权换手成本
    plain_m/ma20_m/v8_m: 各月的裸/风控后/V8 收益"""
    yms = list(monthly_nav_panel.index)
    nav = 1.0
    records = []
    prev_weights = None
    prev_w = None
    holding = False

    for i in range(len(yms) - 1):
        ym_sig = yms[i]
        ym_ret = yms[i + 1]
        sig_row = monthly_nav_panel.loc[ym_sig].dropna()
        if len(sig_row) == 0:
            continue
        n_sig = int(sig.loc[ym_sig, "n_sig"])

        if mode == "none":
            target_w = 1.0
        else:  # timed_buy: 信号>=low_sig 入场, 信号==0 清仓
            if not holding and n_sig >= low_sig:
                holding = True
            elif holding and n_sig == 0:
                holding = False
            target_w = 1.0 if holding else 0.0

        # 持有期收益来源
        if target_w == 0:
            port_ret = float(v8_m.get(ym_ret, 0.0)) if use_v8 else 0.0
        else:
            src = ma20_m if use_ma20 else plain_m
            port_ret = float(src.get(ym_ret, 0.0))

        # 成本: 等权再平衡换手 + 仓位变化换手
        target = pd.Series(target_w / len(sig_row), index=sig_row.index)
        if prev_weights is not None:
            all_i = set(target.index) | set(prev_weights.keys())
            turn_internal = sum(abs(target.get(c, 0) - prev_weights.get(c, 0)) for c in all_i) / 2.0
            turn_pos = abs(target_w - prev_w)
            c = (turn_internal + turn_pos) * cost
        else:
            c = cost * target_w
        nav *= (1 + port_ret - c)
        records.append({"ym": ym_ret, "nav": nav, "ret": port_ret, "cost": c,
                        "n_sig": n_sig, "w": target_w, "sig_ym": ym_sig})
        prev_weights = target.to_dict()
        prev_w = target_w
    return pd.DataFrame(records).set_index("ym")


def calc_stats(nav_df, n_per_year=12):
    rets = nav_df["nav"].pct_change().dropna()
    years = len(rets) / n_per_year
    nav = nav_df["nav"]
    maxdd = ((nav.cummax() - nav) / nav.cummax()).max()
    return {
        "CAGR": nav.iloc[-1] ** (1 / years) - 1 if years > 0 else np.nan,
        "Sharpe": rets.mean() / rets.std(ddof=1) * np.sqrt(n_per_year) if rets.std(ddof=1) > 0 else np.nan,
        "MaxDD": maxdd,
        "WinRate": (rets > 0).mean(),
        "Vol": rets.std(ddof=1) * np.sqrt(n_per_year),
        "FinalNAV": nav.iloc[-1],
        "Calmar": (nav.iloc[-1] ** (1 / years) - 1) / maxdd if maxdd > 0 else np.nan,
        "avg_w": nav_df["w"].mean(),
    }


def main():
    print("[data] 加载行业ETF日频...", flush=True)
    panel = load_industry_daily()
    _, ew_daily = build_series(panel)

    # 月末NAV面板（等权换手成本用）
    monthly_nav = {}
    for code, s in panel.items():
        nav = (1 + s).cumprod()
        me = nav.groupby(s.index.str[:6]).last()
        monthly_nav[code] = me
    nav_panel = pd.DataFrame(monthly_nav).sort_index()
    print(f"[panel] 月末面板 {len(nav_panel)}月 × {nav_panel.shape[1]}ETF")

    print("[data] 加载避险资产...", flush=True)
    hv = load_hv_daily()
    v8_m = hv_monthly_ret(hv)
    print(f"[V8] 月收益覆盖 {v8_m.index[0]}~{v8_m.index[-1]}, 累计NAV={(1+v8_m).prod():.2f}")

    plain_m = monthly_from_daily(ew_daily)
    _, ma20_m_ret = ma20_signal(ew_daily)
    ma20_m = monthly_from_daily(ma20_m_ret)
    print(f"[MA20] 风控月收益累计NAV={(1+ma20_m).prod():.2f} vs 裸={(1+plain_m).prod():.2f}")

    sig = build_signals(list(nav_panel.index))
    print(f"[信号] S1={(sig['s1']>0).mean():.1%} S2={(sig['s2']>0).mean():.1%} S3={(sig['s3']>0).mean():.1%}")

    configs = [
        ("A 无脑全仓(基线)", "none", False, False),
        ("B timed严格+现金", "timed_buy", False, False),
        ("C1 timed严格+V8", "timed_buy", False, True),
        ("C2 无脑全仓+MA20", "none", True, False),
        ("C3 timed严格+MA20+V8", "timed_buy", True, True),
        ("C4 timed严格+MA20+现金", "timed_buy", True, False),
    ]

    print("\n" + "=" * 106)
    print(f"{'版本':<24} {'NAV':>6} {'CAGR':>8} {'Sharpe':>7} {'MaxDD':>8} {'Calmar':>7} {'WinRate':>8} {'平均仓位':>8}")
    print("-" * 106)
    all_navs = {}
    rows = []
    for name, mode, use_ma20, use_v8 in configs:
        nv = run(nav_panel, sig, plain_m, ma20_m, v8_m, mode=mode, use_ma20=use_ma20, use_v8=use_v8)
        st = calc_stats(nv)
        all_navs[name] = nv
        rows.append({"版本": name, **st})
        print(f"{name:<24} {st['FinalNAV']:>6.2f} {st['CAGR']:>7.2%} {st['Sharpe']:>7.2f} "
              f"{st['MaxDD']:>7.2%} {st['Calmar']:>7.2f} {st['WinRate']:>7.1%} {st['avg_w']:>7.1%}")

    # 分期间 2021-06 起
    print("\n=== 分期间 (2021-06 起, ETF基本齐备) ===")
    rows21 = []
    for name, mode, use_ma20, use_v8 in configs:
        nv = all_navs[name]
        sub = nv[nv.index >= "2021-06"]
        st = calc_stats(sub)
        rows21.append({"版本": name, **st})
        print(f"  {name:<24} NAV={st['FinalNAV']:>5.2f} CAGR={st['CAGR']:>7.2%} "
              f"MaxDD={st['MaxDD']:>7.2%} Calmar={st['Calmar']:>5.2f} 仓位={st['avg_w']:.0%}")

    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(OUT_DIR, "etf_optimize_stats.csv"), index=False, encoding="utf-8-sig")
    res21 = pd.DataFrame(rows21)
    res21.to_csv(os.path.join(OUT_DIR, "etf_optimize_stats_2021.csv"), index=False, encoding="utf-8-sig")

    # 图
    fig, ax = plt.subplots(figsize=(14, 6.5))
    palette = ["#1f77b4", "#d62728", "#2ca02c", "#ff7f0e", "#9467bd", "#8c564b"]
    for (name, _, _, _), color in zip(configs, palette):
        nv = all_navs[name]
        ax.plot(range(len(nv)), nv["nav"], lw=1.6, color=color, label=f"{name} ({nv['nav'].iloc[-1]:.2f})")
    ax.set_title("方案C优化: 行业ETF等权 + V8避险/MA20风控 (2015-2026)", fontsize=13)
    ax.set_ylabel("NAV")
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    png = os.path.join(OUT_DIR, "etf_optimize_curve.png")
    fig.savefig(png, dpi=150, bbox_inches="tight")
    print(f"\n[saved] {png}")

    # 结论
    best_full = res.loc[res["Calmar"].idxmax()]
    best_21 = res21.loc[res21["Calmar"].idxmax()]
    conclusion = f"""== 方案C: 行业ETF等权优化（结合 quant_conclusion 验证结论）==

期间: {nav_panel.index[0]} ~ {nav_panel.index[-1]} ({len(nav_panel)} 个月)
覆盖: {nav_panel.shape[1]} 只申万一级行业ETF + V8避险(511990/511260/518880)
成本: 30bps双边(等权再平衡+仓位切换+MA20换仓), 无前视

优化来源:
  1. defensive_asset_allocation.md — 空仓期持V8避险而非现金
  2. risk_control.md              — MA20三档趋势风控(T-1信号T日生效, 2018-2019独立OOS支持)
  3. S1/S2/S3估值信号保留作买入择时

【全期对比】
{res[['版本','FinalNAV','CAGR','MaxDD','Sharpe','Calmar','avg_w']].round(4).to_string(index=False)}

【2021-06 起】
{res21[['版本','FinalNAV','CAGR','MaxDD','Sharpe','Calmar','avg_w']].round(4).to_string(index=False)}

【最优】全期: {best_full['版本']} Calmar={best_full['Calmar']:.2f} CAGR={best_full['CAGR']:.2%} MaxDD={best_full['MaxDD']:.2%}
      2021-06起: {best_21['版本']} Calmar={best_21['Calmar']:.2f} CAGR={best_21['CAGR']:.2%} MaxDD={best_21['MaxDD']:.2%}

【解读】
  - C1 vs B: V8是否带来增量（空仓期收益）
  - C2 vs A: MA20三档是否有效压回撤（独立OOS规则在行业ETF上的验证）
  - C3 vs C4: V8在叠加MA20后的边际贡献
"""
    with open(os.path.join(OUT_DIR, "etf_optimize_conclusion.txt"), "w", encoding="utf-8") as f:
        f.write(conclusion)
    print(conclusion)
    return 0


if __name__ == "__main__":
    sys.exit(main())

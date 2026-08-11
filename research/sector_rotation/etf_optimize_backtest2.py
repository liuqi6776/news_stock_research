# -*- coding: utf-8 -*-
"""方案C-2: 行业ETF等权优化 2.0（结合 quant_conclusion 验证结论）

v1发现: V8避险有效(C1最优), MA20三档(自身净值)外推失败(C2崩溃) —— 需排除
        两个因素: (a)MA20标的不匹配 (b)换仓成本过严。
v2 新增:
  M1 MA20(自身净值, 含换仓成本)      ← v1失败版本, 基线对照
  M2 MA20(000852中证1000, 无换仓成本) ← 仓库原始复刻(risk_control.md)
  M3 MA20(000300沪深300, 无换仓成本)  ← 大盘标的
  C2 timed(S>=2)+V8                   ← 放宽入场门槛
  C3 MA120趋势入场+V8                 ← regime_study唯一双正信号(TREND)
  C4 RS12风格入场+V8                  ← regime_study落地推荐(000852/000300相对强度)

版本矩阵(月度调仓, 30bps双边, 无前视):
  A   无脑全仓(基线)
  B   timed严格(S>=3)+现金
  C1  timed严格(S>=3)+V8
  C2  timed(S>=2)+V8
  C3  MA120趋势+V8
  C4  RS12风格+V8
  M1  无脑全仓+MA20(自身净值,含成本)
  M2  无脑全仓+MA20(000852,无成本)
  M3  无脑全仓+MA20(000300,无成本)
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
HV_DIR = "c:/Users/liuqi/quant_system_v2/research/serve/data/etf"
IDX_DIR = "c:/Users/liuqi/quant_system_v2/research/chip_momentum/data/index_daily"
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(OUT_DIR, exist_ok=True)
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

COST = 30 / 10000.0
MM_ANNUAL = 0.018
MA20_MA = 20
MA20_DEEP = 0.98
MA120_MA = 120
PE_QUANT = 0.20
ERP_Z = 1.0
DD_THRESH = -0.25
HV_WEIGHTS = {"511990.SH": 1 / 3, "511260.SH": 1 / 3, "518880.SH": 1 / 3}

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
    panel = {}
    for name, code in INDUSTRY_ETFS:
        fp = os.path.join(ETF_DIR, f"{code}.parquet")
        if not os.path.exists(fp):
            continue
        df = pd.read_parquet(fp)
        d = df.copy()
        d["pct_chg"] = d["pct_chg"].fillna(0) / 100.0
        d["trade_date"] = d["trade_date"].astype(str).str[:8]
        panel[code] = d.set_index("trade_date")["pct_chg"]
    return panel


def load_hv_daily():
    out = {}
    for code in HV_WEIGHTS:
        df = pd.read_parquet(os.path.join(HV_DIR, f"{code}.parquet"))
        s = df["close"].pct_change().dropna()
        s.index = s.index.astype(str).str[:8]
        if code == "511990.SH":
            s = s + MM_ANNUAL / 242.0
        out[code] = s
    return out


def build_series(panel):
    all_dates = sorted(set().union(*[set(s.index) for s in panel.values()]))
    df = pd.DataFrame(index=all_dates)
    for code, s in panel.items():
        df[code] = s.reindex(all_dates)
    return df.mean(axis=1).fillna(0)


def hv_monthly_ret(hv):
    all_dates = sorted(set().union(*[set(s.index) for s in hv.values()]))
    df = pd.DataFrame(index=all_dates)
    for code, s in hv.items():
        df[code] = s.reindex(all_dates)
    daily = (df * pd.Series(HV_WEIGHTS)).sum(axis=1).fillna(0)
    return (1 + daily).groupby(daily.index.str[:6]).prod() - 1.0


def ma20_signal(ret, with_cost=False):
    """MA20三档日频仓位(T-1信号T日生效)。with_cost=True 时扣换仓成本。"""
    nav = (1 + ret).cumprod()
    ma20 = nav.rolling(MA20_MA).mean()
    w = pd.Series(1.0, index=nav.index)
    below = nav < ma20
    deep_below = nav < ma20 * MA20_DEEP
    w[below & ~deep_below] = 0.5
    w[deep_below] = 0.0
    w = w.shift(1).fillna(1.0)
    out = ret * w
    if with_cost:
        out = out - w.diff().abs().fillna(0) * COST
    return out


def monthly_from_daily(daily_ret):
    return (1 + daily_ret).groupby(daily_ret.index.str[:6]).prod() - 1.0


def load_index_ret(code):
    df = pd.read_parquet(os.path.join(IDX_DIR, f"{code}.parquet"))
    s = df.set_index("trade_date")["close"].pct_change().dropna()
    s.index = s.index.astype(str).str[:8]
    return s


def build_signals(ym_list):
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
        rows.append({"ym": ym, "n_sig": s1 + s2 + s3})
    return pd.DataFrame(rows).set_index("ym")


def build_trend_signals(ym_list, ew_daily, ma=120):
    """行业等权组合净值 vs MA：月末收盘>MA(120)→1 else 0（T月末信号, 下月生效）"""
    nav = (1 + ew_daily).cumprod()
    ma = nav.rolling(ma).mean()
    m_nav = nav.groupby(nav.index.str[:6]).last()
    m_ma = ma.groupby(ma.index.str[:6]).last()
    m_nav.index = m_nav.index.astype(str)
    m_ma.index = m_ma.index.astype(str)
    return m_nav.reindex(ym_list).fillna(0), m_ma.reindex(ym_list).fillna(0)


def build_rs12_signals(ym_list):
    """RS12: 000852/000300 240日相对强度(5日均值)>0→1（月末取值）"""
    r852 = load_index_ret("000852.SH")
    r300 = load_index_ret("000300.SH")
    ratio = (1 + r852).cumprod() / (1 + r300).cumprod()
    rs = (ratio / ratio.shift(240)).rolling(5).mean() - 1.0
    m_rs = rs.groupby(rs.index.str[:6]).last()
    m_rs.index = m_rs.index.astype(str)
    return (m_rs.reindex(ym_list).fillna(0) > 0).astype(int)


def run(nav_panel, sig, plain_m, v8_m, mode="none", entry_sig=None, exit_sig=None,
        use_ma20_m=None, use_v8=False, low_sig=3, cost=COST):
    """月度调仓回测。entry_sig/exit_sig: ym→bool(1/0)。
    use_ma20_m: 若提供(风控后月收益), 持有期用它替代 plain_m。"""
    yms = list(nav_panel.index)
    nav = 1.0
    records = []
    prev_weights = None
    prev_w = None
    holding = False

    for i in range(len(yms) - 1):
        ym_sig = yms[i]
        ym_ret = yms[i + 1]
        sig_row = nav_panel.loc[ym_sig].dropna()
        if len(sig_row) == 0:
            continue
        n_sig = int(sig.loc[ym_sig, "n_sig"])

        if mode == "none":
            target_w = 1.0
        else:  # timed
            can_enter = (entry_sig is None) or bool(entry_sig.get(ym_sig, False))
            must_exit = (exit_sig is not None) and bool(exit_sig.get(ym_sig, False))
            if not holding and can_enter:
                holding = True
            elif holding and must_exit:
                holding = False
            target_w = 1.0 if holding else 0.0

        if target_w == 0:
            port_ret = float(v8_m.get(ym_ret, 0.0)) if use_v8 else 0.0
        else:
            src = use_ma20_m if use_ma20_m is not None else plain_m
            port_ret = float(src.get(ym_ret, 0.0))

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
    ew_daily = build_series(panel)

    monthly_nav = {}
    for code, s in panel.items():
        nav = (1 + s).cumprod()
        monthly_nav[code] = nav.groupby(s.index.str[:6]).last()
    nav_panel = pd.DataFrame(monthly_nav).sort_index()
    print(f"[panel] 月末面板 {len(nav_panel)}月 × {nav_panel.shape[1]}ETF")

    print("[data] 加载避险资产...", flush=True)
    hv = load_hv_daily()
    v8_m = hv_monthly_ret(hv)
    print(f"[V8] {v8_m.index[0]}~{v8_m.index[-1]} 累计NAV={(1+v8_m).prod():.2f}")

    plain_m = monthly_from_daily(ew_daily)
    # MA20 各标的
    ma20_self_m = monthly_from_daily(ma20_signal(ew_daily, with_cost=True))
    ma20_852_m = monthly_from_daily(ma20_signal(load_index_ret("000852.SH"), with_cost=False))
    ma20_300_m = monthly_from_daily(ma20_signal(load_index_ret("000300.SH"), with_cost=False))
    print(f"[MA20] 自身:{ (1+ma20_self_m).prod():.2f} 852:{ (1+ma20_852_m).prod():.2f} 300:{ (1+ma20_300_m).prod():.2f} vs 裸:{(1+plain_m).prod():.2f}")

    sig = build_signals(list(nav_panel.index))
    print(f"[S信号] 均值n_sig={sig['n_sig'].mean():.2f} >=2占比:{(sig['n_sig']>=2).mean():.1%} >=3占比:{(sig['n_sig']>=3).mean():.1%} ==0占比:{(sig['n_sig']==0).mean():.1%}")

    # 趋势/风格信号
    nav_m, ma120_m = build_trend_signals(list(nav_panel.index), ew_daily, MA120_MA)
    ma120_entry = (nav_m > ma120_m).astype(int).to_dict()
    rs12_entry = build_rs12_signals(list(nav_panel.index)).to_dict()

    # 入场/清仓信号构造
    s_ge3 = sig["n_sig"] >= 3
    s_ge2 = sig["n_sig"] >= 2
    s_eq0 = sig["n_sig"] == 0

    configs = [
        # (名称, mode, entry, exit, ma20月收益, use_v8, 备注)
        ("A 无脑全仓(基线)", "none", None, None, None, False, "方案A"),
        ("B timed严格(S3)+现金", "timed", s_ge3.to_dict(), s_eq0.to_dict(), None, False, "方案B最优"),
        ("C1 timed严格(S3)+V8", "timed", s_ge3.to_dict(), s_eq0.to_dict(), None, True, "v1最优"),
        ("C2 timed(S2)+V8", "timed", s_ge2.to_dict(), s_eq0.to_dict(), None, True, "放宽入场"),
        ("C3 MA120趋势+V8", "timed", ma120_entry, None, None, True, "regime_study TREND"),
        ("C4 RS12风格+V8", "timed", rs12_entry, None, None, True, "regime_study RS12"),
        ("M1 MA20自身净值(含成本)", "none", None, None, ma20_self_m, False, "v1失败对照"),
        ("M2 MA20 000852(无成本)", "none", None, None, ma20_852_m, False, "仓库复刻"),
        ("M3 MA20 000300(无成本)", "none", None, None, ma20_300_m, False, "大盘标的"),
    ]

    print("\n" + "=" * 112)
    print(f"{'版本':<28} {'NAV':>6} {'CAGR':>8} {'Sharpe':>7} {'MaxDD':>8} {'Calmar':>7} {'WinRate':>8} {'仓位':>6}")
    print("-" * 112)
    all_navs = {}
    rows = []
    for name, mode, entry, exit_, ma20m, use_v8, note in configs:
        nv = run(nav_panel, sig, plain_m, v8_m, mode=mode, entry_sig=entry,
                 exit_sig=exit_, use_ma20_m=ma20m, use_v8=use_v8)
        st = calc_stats(nv)
        all_navs[name] = nv
        rows.append({"版本": name, "说明": note, **st})
        print(f"{name:<28} {st['FinalNAV']:>6.2f} {st['CAGR']:>7.2%} {st['Sharpe']:>7.2f} "
              f"{st['MaxDD']:>7.2%} {st['Calmar']:>7.2f} {st['WinRate']:>7.1%} {st['avg_w']:>5.0%}")

    print("\n=== 分期间 (2021-06 起) ===")
    rows21 = []
    for name, mode, entry, exit_, ma20m, use_v8, note in configs:
        nv = all_navs[name]
        sub = nv[nv.index >= "2021-06"]
        st = calc_stats(sub)
        rows21.append({"版本": name, **st})
        print(f"  {name:<28} NAV={st['FinalNAV']:>5.2f} CAGR={st['CAGR']:>7.2%} "
              f"MaxDD={st['MaxDD']:>7.2%} Calmar={st['Calmar']:>5.2f} 仓位={st['avg_w']:.0%}")

    pd.DataFrame(rows).to_csv(os.path.join(OUT_DIR, "etf_optimize2_stats.csv"), index=False, encoding="utf-8-sig")
    pd.DataFrame(rows21).to_csv(os.path.join(OUT_DIR, "etf_optimize2_stats_2021.csv"), index=False, encoding="utf-8-sig")

    fig, ax = plt.subplots(figsize=(14, 6.5))
    palette = ["#1f77b4", "#d62728", "#2ca02c", "#ff7f0e", "#9467bd", "#8c564b", "#7f7f7f", "#bcbd22", "#17becf"]
    for (name, _, _, _, _, _, _), color in zip(configs, palette):
        nv = all_navs[name]
        ax.plot(range(len(nv)), nv["nav"], lw=1.5, color=color, label=f"{name} ({nv['nav'].iloc[-1]:.2f})")
    ax.set_title("行业ETF等权优化2.0: V8避险/MA120/RS12/MA20对照 (2015-2026)", fontsize=13)
    ax.set_ylabel("NAV")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    png = os.path.join(OUT_DIR, "etf_optimize2_curve.png")
    fig.savefig(png, dpi=150, bbox_inches="tight")
    print(f"\n[saved] {png}")

    best_full = pd.DataFrame(rows).sort_values("Calmar", ascending=False).iloc[0]
    best_21 = pd.DataFrame(rows21).sort_values("Calmar", ascending=False).iloc[0]
    conclusion = f"""== 行业ETF等权优化2.0（结合 quant_conclusion 验证结论）==

期间: {nav_panel.index[0]}~{nav_panel.index[-1]} ({len(nav_panel)}月), 26行业ETF+V8避险, 30bps, 无前视

【全期】
{pd.DataFrame(rows)[['版本','FinalNAV','CAGR','MaxDD','Sharpe','Calmar','avg_w']].round(4).to_string(index=False)}

【2021-06起】
{pd.DataFrame(rows21)[['版本','FinalNAV','CAGR','MaxDD','Sharpe','Calmar','avg_w']].round(4).to_string(index=False)}

【最优】全期: {best_full['版本']} Calmar={best_full['Calmar']:.2f} CAGR={best_full['CAGR']:.2%} MaxDD={best_full['MaxDD']:.2%}
      2021-06起: {best_21['版本']} Calmar={best_21['Calmar']:.2f} CAGR={best_21['CAGR']:.2%} MaxDD={best_21['MaxDD']:.2%}
"""
    with open(os.path.join(OUT_DIR, "etf_optimize2_conclusion.txt"), "w", encoding="utf-8") as f:
        f.write(conclusion)
    print(conclusion)
    return 0


if __name__ == "__main__":
    sys.exit(main())

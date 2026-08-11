# -*- coding: utf-8 -*-
"""方案A: 申万一级行业ETF等权回测（真实可买版本）

用31个申万一级行业对应的主流ETF日线数据（tushare fund_daily），
每月末等权调仓，回测真实可落地的"行业等权"策略。

注意: 部分行业ETF上市较晚, 每月只在"已上市"的ETF中等权分配
（模拟真实买入场景: 上市后才能买）。

数据: research/sector_rotation/data/industry_etf/
输出: results/etf_equal_weight_stats.csv + curve.png + conclusion.txt
"""
import os
import sys
import time

import numpy as np
import pandas as pd
import tushare as ts
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
from config.settings import settings

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "industry_etf")
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

START, END = "20150101", "20260808"
COST = 30 / 10000.0  # 双边20bps手续费 + 10bps滑点

# 申万一级行业 → 代表性ETF（市场主流、流动性好）
INDUSTRY_ETFS = [
    ("农林牧渔", "159825.SZ"),
    ("基础化工", "159870.SZ"),
    ("钢铁", "515210.SH"),
    ("有色金属", "512400.SH"),
    ("电子", "512480.SH"),
    ("汽车", "516110.SH"),
    ("家用电器", "159996.SZ"),
    ("食品饮料", "515170.SH"),
    ("医药生物", "512010.SH"),
    ("公用事业", "159611.SZ"),
    ("交通运输", "516320.SH"),
    ("房地产", "512200.SH"),
    ("社会服务", "159766.SZ"),
    ("建筑材料", "159745.SZ"),
    ("建筑装饰", "516950.SH"),
    ("电力设备", "515030.SH"),
    ("国防军工", "512660.SH"),
    ("计算机", "512720.SH"),
    ("传媒", "512980.SH"),
    ("通信", "515880.SH"),
    ("银行", "512800.SH"),
    ("非银金融", "512880.SH"),
    ("煤炭", "515220.SH"),
    ("石油石化", "159930.SZ"),
    ("环保", "512580.SH"),
    ("机械设备", "159886.SZ"),
]


def fetch_etf(code, name):
    fp = os.path.join(DATA_DIR, f"{code}.parquet")
    if os.path.exists(fp):
        return pd.read_parquet(fp)
    pro = ts.pro_api(settings.TUSHARE_TOKEN)
    try:
        df = pro.fund_daily(ts_code=code, start_date=START, end_date=END)
    except Exception as e:
        print(f"  [fail] {code} {name}: {e}", flush=True)
        return None
    if df is None or df.empty:
        print(f"  [empty] {code} {name}", flush=True)
        return None
    df = df.sort_values("trade_date").reset_index(drop=True)
    df.to_parquet(fp)
    time.sleep(0.35)
    return df


def load_all_etfs():
    """拉取全部ETF → {code: df}，返回上市日期信息"""
    data = {}
    print(f"[fetch] 拉取 {len(INDUSTRY_ETFS)} 只行业ETF...", flush=True)
    for name, code in INDUSTRY_ETFS:
        df = fetch_etf(code, name)
        if df is not None:
            data[code] = df
            print(f"  [ok] {code} {name}: {df['trade_date'].min()} ~ {df['trade_date'].max()} ({len(df)}行)", flush=True)
        else:
            print(f"  [skip] {code} {name}", flush=True)
    return data


def build_monthly_nav_panel(data):
    """构建月末 NAV 面板: date(月末) × ETF代码, 每列从上市起=1"""
    # 每只ETF的月度月末NAV (用pct_chg累乘, 避免拆分)
    monthly = {}
    for code, df in data.items():
        d = df.copy()
        d["pct_chg"] = d["pct_chg"].fillna(0) / 100.0
        nav = (1 + d["pct_chg"]).cumprod()
        d["nav"] = nav
        d["ym"] = d["trade_date"].str[:6]
        me = d.groupby("ym").last()  # 月末
        monthly[code] = me["nav"]
    panel = pd.DataFrame(monthly)
    panel.index.name = "ym"
    return panel


def run_equal_weight(panel):
    """每月末等权调仓: 只在有数据的ETF中等权。
    信号月末生成 → 持有到下月末（无前视）。"""
    yms = list(panel.index)
    nav = 1.0
    records = []
    prev_weights = None

    for i in range(len(yms) - 1):
        ym_sig = yms[i]
        ym_ret = yms[i + 1]
        if ym_sig not in panel.index or ym_ret not in panel.index:
            continue
        sig_row = panel.loc[ym_sig].dropna()  # 信号日可用ETF
        ret_row = panel.loc[ym_ret]
        if len(sig_row) == 0:
            continue

        target = pd.Series(1.0 / len(sig_row), index=sig_row.index)  # 等权

        # 持有期收益 = 用sig日选中的ETF在下月的nav变化
        r = ret_row.reindex(target.index) / sig_row  # NAV变化比
        port_ret = (r * target).sum() - 1.0

        # 成本: 换手
        if prev_weights is not None:
            all_i = set(target.index) | set(prev_weights.keys())
            turn = sum(abs(target.get(c, 0) - prev_weights.get(c, 0)) for c in all_i) / 2.0
            cost = turn * COST
        else:
            cost = COST
        nav *= (1 + port_ret - cost)
        records.append({"ym": ym_ret, "nav": nav, "ret": port_ret, "cost": cost,
                        "n_etf": len(target), "sig_ym": ym_sig})
        prev_weights = target.to_dict()
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
        "n_months": len(nav_df),
    }


def load_index_monthly(code, label):
    """加载宽基指数月度nav做基准 (用已有index_daily数据)"""
    fp = os.path.join("c:/Users/liuqi/quant_system_v2/research/chip_momentum/data/index_daily", f"{code}.parquet")
    if not os.path.exists(fp):
        return None, label
    df = pd.read_parquet(fp)
    df["trade_date"] = df["trade_date"].astype(str)
    df["pct_chg"] = df["pct_chg"].fillna(0) / 100.0
    df["nav"] = (1 + df["pct_chg"]).cumprod()
    df["ym"] = df["trade_date"].str[:6]
    me = df.groupby("ym").last()["nav"]
    return me, label


def main():
    data = load_all_etfs()
    panel = build_monthly_nav_panel(data)
    print(f"\n[panel] 月末面板 {panel.shape[0]}月 × {panel.shape[1]}ETF")

    nv = run_equal_weight(panel)
    st = calc_stats(nv)
    print(f"\n[ETF等权] 全期: NAV={st['FinalNAV']:.2f} CAGR={st['CAGR']:.2%} "
          f"Sharpe={st['Sharpe']:.2f} MaxDD={st['MaxDD']:.2%} Calmar={st['Calmar']:.2f}")

    # 逐年
    nv["year"] = nv.index.str[:4]
    print("\n=== 逐年收益 ===")
    for y, g in nv.groupby("year"):
        y_start = g["nav"].iloc[0] / (1 + g["ret"].iloc[0])
        y_ret = g["nav"].iloc[-1] / y_start - 1
        print(f"  {y}: {y_ret:+.2%}  (年末NAV {g['nav'].iloc[-1]:.3f})")

    # 基准: 中证1000ETF 512100, 沪深300ETF 510300, 中证500
    idx500, l500 = load_index_monthly("000905.SH", "中证500")
    idx512, l512 = load_index_monthly("512100.SH", "中证1000ETF")
    idx300, l300 = load_index_monthly("510300.SH", "沪深300ETF")

    # 对齐到ETF等权区间 (用pct_chg计算nav, 起点对齐)
    def norm_series(s, ref_index):
        # 基准可能晚于ETF等权起点上市, 先ffill再dropna, 从基准自身起点归一
        s = s.reindex(ref_index).ffill().dropna()
        return s / s.iloc[0]

    rows = [{"策略": "申万一级行业ETF等权(方案A)", **st}]
    for s, label in [(idx512, l512), (idx300, l300), (idx500, l500)]:
        if s is None:
            print(f"  [warn] 基准 {label} 数据不存在", flush=True)
            continue
        sub = norm_series(s, nv.index)
        nav_s = sub.to_frame("nav")
        st_s = calc_stats(nav_s)
        rows.append({"策略": f"{label}(自{sub.index[0]}起)", **st_s})

    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(OUT_DIR, "etf_equal_weight_stats.csv"), index=False, encoding="utf-8-sig")

    # 分期间: 2021-06 后 (大部分ETF已上市)
    print("\n=== 分期间对比 (2021-06 起, 大部分行业ETF已上市) ===")
    sub_21 = nv[nv.index >= "2021-06"]
    st_21 = calc_stats(sub_21)
    print(f"  方案A ETF等权 (2021-06起): NAV={st_21['FinalNAV']:.2f} CAGR={st_21['CAGR']:.2%} "
          f"MaxDD={st_21['MaxDD']:.2%} Calmar={st_21['Calmar']:.2f}")

    for s, label in [(idx512, l512), (idx300, l300), (idx500, l500)]:
        if s is None:
            continue
        sub = norm_series(s, sub_21.index)
        st_s = calc_stats(sub.to_frame("nav"))
        print(f"  {label} (2021-06起): NAV={st_s['FinalNAV']:.2f} CAGR={st_s['CAGR']:.2%} "
              f"MaxDD={st_s['MaxDD']:.2%} Calmar={st_s['Calmar']:.2f}")

    print("\n" + "=" * 90)
    print(res[["策略", "FinalNAV", "CAGR", "MaxDD", "Sharpe", "Calmar"]].round(4).to_string(index=False))
    print("=" * 90)

    # 图
    fig, ax = plt.subplots(figsize=(14, 6.5))
    ax.plot(range(len(nv)), nv["nav"], lw=2.0, color="#d62728",
            label=f"行业ETF等权(方案A) NAV={nv['nav'].iloc[-1]:.2f}")
    for s, label, color in [(idx512, "中证1000ETF", "#1f77b4"),
                             (idx300, "沪深300ETF", "#2ca02c"),
                             (idx500, "中证500", "#ff7f0e")]:
        if s is None:
            continue
        sub = s.reindex(nv.index).ffill()
        if sub.isna().any():
            continue
        ax.plot(range(len(nv)), sub.values / sub.iloc[0], lw=1.5, color=color, label=f"{label}")
    ax.set_title("申万一级行业ETF等权 vs 宽基ETF（真实可买）", fontsize=13)
    ax.set_ylabel("NAV")
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    png = os.path.join(OUT_DIR, "etf_equal_weight_curve.png")
    fig.savefig(png, dpi=150, bbox_inches="tight")
    print(f"\n[saved] {png}")

    conclusion = f"""== 方案A: 申万一级行业ETF等权（真实可买）结论 ==

期间: {nv.index[0]} ~ {nv.index[-1]} ({len(nv)} 个月)
覆盖: {len(INDUSTRY_ETFS)} 只行业ETF（申万一级行业代表）
方法: 每月末等权调仓, 30bps成本, 只买已上市ETF, 无前视

【结果】
{res[['策略','FinalNAV','CAGR','MaxDD','Sharpe','Calmar']].round(4).to_string(index=False)}

【逐年】
"""
    for y, g in nv.groupby("year"):
        y_start = g["nav"].iloc[0] / (1 + g["ret"].iloc[0])
        y_ret = g["nav"].iloc[-1] / y_start - 1
        conclusion += f"  {y}: {y_ret:+.2%}\n"
    with open(os.path.join(OUT_DIR, "etf_equal_weight_conclusion.txt"), "w", encoding="utf-8") as f:
        f.write(conclusion)
    print(f"\n[saved] etf_equal_weight_conclusion.txt")
    return 0


if __name__ == "__main__":
    sys.exit(main())

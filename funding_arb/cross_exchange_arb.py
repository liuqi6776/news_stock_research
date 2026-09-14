# -*- coding: utf-8 -*-
"""
跨交易所资金费率套利（Cross-Exchange Funding Rate Arbitrage）
标的：BTC, ETH, SOL
对比交易所：Binance vs OKX (OKX API最大可用区间 ~100天) + Binance vs Bybit (3年全样本)
包含数据本地化保存、价差微观特征分析、Delta中性套利回测及磨损评估。
"""
import sys, time, datetime
from pathlib import Path
import requests
import pandas as pd
import numpy as np

# 画图设置
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

DIR = Path(__file__).parent
DAYS = 365 * 3
now_ms = int(time.time() * 1000)
start_ms = now_ms - DAYS * 86400_000

# ==========================================
# 1. 数据下载模块
# ==========================================

def fetch_okx_funding(inst_id: str) -> pd.DataFrame:
    """获取 OKX 资金费率历史（API限制约近100天/300条）"""
    print(f"[OKX] 正在获取 {inst_id} 资金费率...")
    rows = []
    after = None
    while True:
        params = {"instId": inst_id, "limit": 100}
        if after:
            params["after"] = str(after)
        try:
            r = requests.get("https://www.okx.com/api/v5/public/funding-rate-history",
                             params=params, timeout=10)
            data = r.json().get("data", [])
            if not data:
                break
            rows.extend(data)
            after = data[-1]["fundingTime"]
            if len(data) < 100:
                break
            time.sleep(0.2)
        except Exception as e:
            print(f"[OKX] 请求异常: {e}")
            break

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df["fundingTime"] = pd.to_datetime(df["fundingTime"].astype("int64"), unit="ms")
    df["fundingRate"] = pd.to_numeric(df["fundingRate"], errors="coerce")
    df["realizedRate"] = pd.to_numeric(df["realizedRate"], errors="coerce")
    df = df.drop_duplicates("fundingTime").sort_values("fundingTime").reset_index(drop=True)
    return df[["fundingTime", "fundingRate", "realizedRate"]]


def fetch_bybit_funding(symbol: str) -> pd.DataFrame:
    """获取 Bybit 近3年资金费率历史"""
    print(f"[Bybit] 正在获取 {symbol} 近3年资金费率...")
    rows = []
    end_time = None
    target_start = start_ms

    while True:
        params = {"category": "linear", "symbol": symbol, "limit": 100}
        if end_time:
            params["endTime"] = end_time
        try:
            r = requests.get("https://api.bybit.com/v5/market/funding/history",
                             params=params, timeout=10)
            res = r.json().get("result", {}).get("list", [])
            if not res:
                break
            rows.extend(res)
            oldest_ts = int(res[-1]["fundingRateTimestamp"])
            if oldest_ts <= target_start or len(res) < 100:
                break
            end_time = oldest_ts - 1
            time.sleep(0.15)
        except Exception as e:
            print(f"[Bybit] 请求异常: {e}")
            break

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df["fundingTime"] = pd.to_datetime(df["fundingRateTimestamp"].astype("int64"), unit="ms")
    df["fundingRate"] = pd.to_numeric(df["fundingRate"], errors="coerce")
    df = df.drop_duplicates("fundingTime").sort_values("fundingTime").reset_index(drop=True)
    return df[["fundingTime", "fundingRate"]]


def load_or_fetch_all():
    symbols = {
        "BTC": {"binance": "BTCUSDT", "okx": "BTC-USDT-SWAP", "bybit": "BTCUSDT"},
        "ETH": {"binance": "ETHUSDT", "okx": "ETH-USDT-SWAP", "bybit": "ETHUSDT"},
        "SOL": {"binance": "SOLUSDT", "okx": "SOL-USDT-SWAP", "bybit": "SOLUSDT"},
    }

    data = {}
    for coin, mapping in symbols.items():
        data[coin] = {}

        # 1. Binance
        bin_file = DIR / f"{mapping['binance']}_funding.csv"
        if bin_file.exists():
            print(f"[Binance] 本地加载 {coin}: {bin_file.name}")
            b_df = pd.read_csv(bin_file)
            b_df["fundingTime"] = pd.to_datetime(b_df["fundingTime"])
            data[coin]["binance"] = b_df.sort_values("fundingTime").reset_index(drop=True)
        else:
            print(f"[Binance] 警告：未找到本地 {bin_file.name}")

        # 2. OKX
        okx_file = DIR / f"OKX_{coin}USDT_funding.csv"
        if okx_file.exists():
            print(f"[OKX] 本地加载 {coin}: {okx_file.name}")
            o_df = pd.read_csv(okx_file)
            o_df["fundingTime"] = pd.to_datetime(o_df["fundingTime"])
            data[coin]["okx"] = o_df.sort_values("fundingTime").reset_index(drop=True)
        else:
            o_df = fetch_okx_funding(mapping["okx"])
            o_df.to_csv(okx_file, index=False)
            print(f"[OKX] 已保存至本地: {okx_file.name} ({len(o_df)} 条)")
            data[coin]["okx"] = o_df

        # 3. Bybit
        bybit_file = DIR / f"BYBIT_{coin}USDT_funding.csv"
        if bybit_file.exists():
            print(f"[Bybit] 本地加载 {coin}: {bybit_file.name}")
            by_df = pd.read_csv(bybit_file)
            by_df["fundingTime"] = pd.to_datetime(by_df["fundingTime"])
            data[coin]["bybit"] = by_df.sort_values("fundingTime").reset_index(drop=True)
        else:
            by_df = fetch_bybit_funding(mapping["bybit"])
            by_df.to_csv(bybit_file, index=False)
            print(f"[Bybit] 已保存至本地: {bybit_file.name} ({len(by_df)} 条)")
            data[coin]["bybit"] = by_df

    return data


# ==========================================
# 2. 套利回测引擎
# ==========================================

def run_cross_exchange_arb(df, col_a, col_b, name_a, name_b,
                           theta_in=0.0001, theta_out=0.0,
                           fee_maker=0.0002, fee_taker=0.0005, use_maker=True):
    """
    跨交易所套利逻辑:
    - 两个交易所同时持仓，1x做多，1x做空，净Delta = 0。
    - 价差 spread_t = rate_a - rate_b (在结算时刻已知)
    - 预测下一期仍具惯性：
      若 spread >= theta_in:
        A交易所费率显著高于B -> A做空收租，B做多付较少租金（或收补贴）
        目标持仓: pos_a = -1, pos_b = +1
      若 spread <= -theta_in:
        B交易所费率显著高于A -> B做空收租，A做多
        目标持仓: pos_a = +1, pos_b = -1
      若 |spread| <= theta_out:
        价差收敛平仓: pos_a = 0, pos_b = 0
      若处于两者之间:
        保持上一期仓位（滞后带）
    - 手续费: 每次调仓产生 fee * abs(delta_pos) 的费用（双边交易所各扣一次）
    """
    fee_rate = fee_maker if use_maker else fee_taker

    pos_a, pos_b = 0.0, 0.0
    eq = 1.0
    equity_curve = [1.0]
    total_fund_collected = 0.0
    total_fees = 0.0
    trades = 0

    df = df.copy()
    df["spread"] = df[col_a] - df[col_b]
    df["next_rate_a"] = df[col_a].shift(-1)
    df["next_rate_b"] = df[col_b].shift(-1)
    df = df.dropna().reset_index(drop=True)

    for i, row in df.iterrows():
        sp = row["spread"]

        # 决定目标仓位
        if sp >= theta_in:
            t_a, t_b = -1.0, 1.0
        elif sp <= -theta_in:
            t_a, t_b = 1.0, -1.0
        elif abs(sp) <= theta_out:
            t_a, t_b = 0.0, 0.0
        else:
            t_a, t_b = pos_a, pos_b  # 滞后带保持

        # 换手手续费 (两腿)
        d_a = abs(t_a - pos_a)
        d_b = abs(t_b - pos_b)
        if d_a > 0 or d_b > 0:
            trade_fee = (d_a + d_b) * fee_rate
            total_fees += trade_fee
            eq -= trade_fee
            if pos_a == 0 and t_a != 0:
                trades += 1
            pos_a, pos_b = t_a, t_b

        # 资金费结算收支: 多头支付费率，空头收取费率: pnl = - pos * funding_rate
        pnl_fund_a = - pos_a * row["next_rate_a"]
        pnl_fund_b = - pos_b * row["next_rate_b"]
        period_fund = pnl_fund_a + pnl_fund_b

        total_fund_collected += period_fund
        eq = eq * (1.0 + period_fund)
        equity_curve.append(eq)

    eq_series = pd.Series(equity_curve)
    rets = eq_series.pct_change().dropna()
    n_periods = len(df)
    yrs = n_periods * 8 / (24 * 365)
    total_ret = eq_series.iloc[-1] - 1.0
    ann_ret = (1.0 + total_ret) ** (1.0 / yrs) - 1.0 if yrs > 0 else 0.0
    mdd = (eq_series / eq_series.cummax() - 1.0).min()
    sharpe = rets.mean() / (rets.std() + 1e-12) * np.sqrt(3 * 365)

    return {
        "pair": f"{name_a} vs {name_b}",
        "periods": n_periods,
        "years": round(yrs, 2),
        "theta_in_bp": round(theta_in * 1e4, 2),
        "fee_type": "Maker(0.02%)" if use_maker else "Taker(0.05%)",
        "total_ret": f"{total_ret*100:.2f}%",
        "ann_ret": f"{ann_ret*100:.2f}%",
        "ann_val": ann_ret,
        "max_dd": f"{mdd*100:.2f}%",
        "sharpe": round(sharpe, 2),
        "trades": trades,
        "funding_inc_pct": f"{total_fund_collected*100:.2f}%",
        "fees_pct": f"{total_fees*100:.2f}%",
        "equity": eq_series,
        "time": df["fundingTime"]
    }


def analyze_spreads(df, col_a, col_b, label):
    """分析两所资金费率的利差特征"""
    diff = (df[col_a] - df[col_b]) * 1e4 # bp
    mean_diff = diff.mean()
    abs_diff = diff.abs().mean()
    corr = df[col_a].corr(df[col_b])
    p90 = diff.abs().quantile(0.90)
    p95 = diff.abs().quantile(0.95)
    a_higher_pct = (diff > 0).mean() * 100
    b_higher_pct = (diff < 0).mean() * 100

    print(f"\n--- 【{label}】费率利差统计 ---")
    print(f"  相关系数 (Corr): {corr:.4f}")
    print(f"  平均利差 (A - B): {mean_diff:.3f} bp/8h (年化约 {mean_diff*3*365/100:.2f}%)")
    print(f"  平均绝对利差 |A - B|: {abs_diff:.3f} bp/8h")
    print(f"  90%分位数绝对利差: {p90:.2f} bp, 95%分位数: {p95:.2f} bp")
    print(f"  A > B 周期占比: {a_higher_pct:.1f}%,  B > A 周期占比: {b_higher_pct:.1f}%")
    return {
        "标的组": label,
        "相关系数": round(corr, 3),
        "平均利差(bp)": round(mean_diff, 2),
        "平均绝对利差(bp)": round(abs_diff, 2),
        "90分位利差(bp)": round(p90, 2),
        "A高于B占比": f"{a_higher_pct:.1f}%"
    }


# ==========================================
# 3. 主流程运行
# ==========================================
def main():
    print("==================================================")
    print("开始获取与加载多所数据（Binance / OKX / Bybit）...")
    print("==================================================")
    data = load_or_fetch_all()

    stat_summary = []
    backtest_records = []

    # 画图准备
    fig, axes = plt.subplots(3, 2, figsize=(16, 14))

    for idx, coin in enumerate(["BTC", "ETH", "SOL"]):
        bin_df = data[coin]["binance"].rename(columns={"fundingRate": "rate_binance"})
        bin_df["fundingTime"] = pd.to_datetime(bin_df["fundingTime"]).dt.floor("8h")

        okx_df = data[coin]["okx"].rename(columns={"fundingRate": "rate_okx"})
        okx_df["fundingTime"] = pd.to_datetime(okx_df["fundingTime"]).dt.floor("8h")

        byb_df = data[coin]["bybit"].rename(columns={"fundingRate": "rate_bybit"})
        byb_df["fundingTime"] = pd.to_datetime(byb_df["fundingTime"]).dt.floor("8h")

        # 对齐 1: Binance vs OKX (近100天可用区间)
        df_okx = pd.merge(bin_df[["fundingTime", "rate_binance"]],
                          okx_df[["fundingTime", "rate_okx"]],
                          on="fundingTime", how="inner").sort_values("fundingTime").reset_index(drop=True)

        # 对齐 2: Binance vs Bybit (近3年全量)
        df_byb = pd.merge(bin_df[["fundingTime", "rate_binance"]],
                          byb_df[["fundingTime", "rate_bybit"]],
                          on="fundingTime", how="inner").sort_values("fundingTime").reset_index(drop=True)

        # 统计分析
        st1 = analyze_spreads(df_okx, "rate_binance", "rate_okx", f"{coin} (Binance vs OKX, ~100天)")
        st2 = analyze_spreads(df_byb, "rate_binance", "rate_bybit", f"{coin} (Binance vs Bybit, 近3年)")
        stat_summary.extend([st1, st2])

        # 回测1: Binance vs OKX
        for th in [0.00005, 0.0001, 0.0002]: # 0.5bp, 1bp, 2bp
            res_m = run_cross_exchange_arb(df_okx, "rate_binance", "rate_okx",
                                           f"{coin}_Binance", "OKX",
                                           theta_in=th, theta_out=0.0, use_maker=True)
            res_t = run_cross_exchange_arb(df_okx, "rate_binance", "rate_okx",
                                           f"{coin}_Binance", "OKX",
                                           theta_in=th, theta_out=0.0, use_maker=False)
            backtest_records.append({
                "标的": coin, "对比": "Binance vs OKX", "周期": f"{res_m['years']}年({res_m['periods']}期)",
                "入场阈值(bp)": res_m["theta_in_bp"], "费率类型": res_m["fee_type"],
                "年化收益": res_m["ann_ret"], "最大回撤": res_m["max_dd"], "夏普比率": res_m["sharpe"],
                "交易次数": res_m["trades"], "资金费总收": res_m["funding_inc_pct"], "手续费扣除": res_m["fees_pct"]
            })
            backtest_records.append({
                "标的": coin, "对比": "Binance vs OKX", "周期": f"{res_t['years']}年({res_t['periods']}期)",
                "入场阈值(bp)": res_t["theta_in_bp"], "费率类型": res_t["fee_type"],
                "年化收益": res_t["ann_ret"], "最大回撤": res_t["max_dd"], "夏普比率": res_t["sharpe"],
                "交易次数": res_t["trades"], "资金费总收": res_t["funding_inc_pct"], "手续费扣除": res_t["fees_pct"]
            })

        # 回测2: Binance vs Bybit (3年大样本)
        best_byb = None
        for th in [0.00005, 0.0001, 0.00015, 0.0002]:
            res_m = run_cross_exchange_arb(df_byb, "rate_binance", "rate_bybit",
                                           f"{coin}_Binance", "Bybit",
                                           theta_in=th, theta_out=0.0, use_maker=True)
            res_t = run_cross_exchange_arb(df_byb, "rate_binance", "rate_bybit",
                                           f"{coin}_Binance", "Bybit",
                                           theta_in=th, theta_out=0.0, use_maker=False)
            backtest_records.append({
                "标的": coin, "对比": "Binance vs Bybit", "周期": f"{res_m['years']}年({res_m['periods']}期)",
                "入场阈值(bp)": res_m["theta_in_bp"], "费率类型": res_m["fee_type"],
                "年化收益": res_m["ann_ret"], "最大回撤": res_m["max_dd"], "夏普比率": res_m["sharpe"],
                "交易次数": res_m["trades"], "资金费总收": res_m["funding_inc_pct"], "手续费扣除": res_m["fees_pct"]
            })
            backtest_records.append({
                "标的": coin, "对比": "Binance vs Bybit", "周期": f"{res_t['years']}年({res_t['periods']}期)",
                "入场阈值(bp)": res_t["theta_in_bp"], "费率类型": res_t["fee_type"],
                "年化收益": res_t["ann_ret"], "最大回撤": res_t["max_dd"], "夏普比率": res_t["sharpe"],
                "交易次数": res_t["trades"], "资金费总收": res_t["funding_inc_pct"], "手续费扣除": res_t["fees_pct"]
            })
            if best_byb is None or res_m["ann_val"] > best_byb["ann_val"]:
                best_byb = res_m

        # 画图: 左列绘制 Binance vs OKX 利差，右列绘制 Binance vs Bybit 3年累计净值
        ax_left = axes[idx, 0]
        sp_okx = (df_okx["rate_binance"] - df_okx["rate_okx"]) * 1e4
        ax_left.plot(df_okx["fundingTime"], sp_okx, color="tab:blue", lw=0.8, label="Binance - OKX 利差 (bp)")
        ax_left.axhline(0, color="gray", lw=0.6, ls="--")
        ax_left.axhline(1.0, color="red", lw=0.6, ls=":", label="±1bp 阈值")
        ax_left.axhline(-1.0, color="red", lw=0.6, ls=":")
        ax_left.set_title(f"{coin} Binance vs OKX 资金费率利差 (最近约100天)")
        ax_left.set_ylabel("利差 (bp/8h)")
        ax_left.legend(fontsize=8)
        ax_left.grid(alpha=0.3)

        ax_right = axes[idx, 1]
        t_seq = pd.concat([pd.Series([best_byb["time"].iloc[0]]), best_byb["time"]]).iloc[:len(best_byb["equity"])]
        ax_right.plot(t_seq, best_byb["equity"], color="tab:orange", lw=1.2,
                      label=f"Binance vs Bybit 套利净值 (最优阈值 {best_byb['theta_in_bp']}bp)")
        ax_right.set_title(f"{coin} Binance vs Bybit 跨所套利净值曲线 (近3年)")
        ax_right.set_ylabel("净值")
        ax_right.legend(fontsize=8)
        ax_right.grid(alpha=0.3)

    plt.tight_layout()
    fig_path = DIR / "cross_exchange_backtest.png"
    fig.savefig(fig_path, dpi=140)
    print(f"\n[图表] 回测图表已生成: {fig_path.name}")

    # 保存统计报表
    stat_df = pd.DataFrame(stat_summary)
    stat_df.to_csv(DIR / "cross_exchange_stats.csv", index=False, encoding="utf-8-sig")

    res_df = pd.DataFrame(backtest_records)
    res_df.to_csv(DIR / "cross_exchange_results.csv", index=False, encoding="utf-8-sig")
    print(f"[报表] 结果汇总已保存至: cross_exchange_results.csv")

    print("\n=================== 核心回测结果展示 ===================")
    print(res_df.head(20).to_string(index=False))

if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""B2: 股票 ML 特征面板构建 (Plan B)

输入:
  - D:/iquant_data/data_v2/data_day1/*.parquet      日频行情 (全市场)
  - D:/iquant_data/data_v2/index_weight/*.parquet   中证1000 月度成分快照 (000852)
  - D:/iquant_data/data_v2/fundamental1/fina_indicator_cache.parquet  财务 PIT (2023-04 起)
  - research/studies/study_008_enhancements/data/industry_map.parquet 行业映射

输出 (默认):
  - research/sector_rotation/stock_ml_panel_72m.parquet
    列: [trade_date(月末快照日), ts_code, industry, is_traditional,
         ret_1m, ivol, momentum_5/10/20/60, volatility_5/10/20,
         alpha_006/009/012/023, roe, or_yoy, netprofit_yoy,
         vwap_20, float_pnl_20, prof_pct_20, chip_conc_20, chip_shift_5, pos_vol_20,
         fwd_20, f_rev, f_ivol]

无前视约束:
  - 调仓日 T 的成分股 = index_weight 中 trade_date<=T 的最近一期快照
  - 因子只用 T 及之前数据
  - 财务 PIT 用 ann_date merge_asof(direction=backward)

可复用入口:
  - build_panel(end_ym, keep_last, out_path):
      end_ym=YYYYMMDD 面板月末截止 (默认 20251231, 保持冻结回测口径)
      keep_last=True 保留最后一期月末快照(其 fwd_20 为 NaN, 仅用于前向特征打分, 不用标签)
      out_path 自定义输出路径
"""
import os
import glob
import time
import numpy as np
import pandas as pd

ROOT = r"c:\Users\liuqi\quant_system_v2"
DATA = r"D:/iquant_data/data_v2"
DEFAULT_OUT = os.path.join(ROOT, "research", "sector_rotation", "stock_ml_panel_72m.parquet")


def build_panel(end_ym=20251231, keep_last=False, out_path=None):
    out = out_path or DEFAULT_OUT
    t0 = time.time()

    # ---------- 1. 中证1000 成分历史 ----------
    iw_files = sorted(glob.glob(os.path.join(DATA, "index_weight", "*.parquet")))
    iw = pd.concat([pd.read_parquet(f) for f in iw_files], ignore_index=True)
    iw = iw[iw["index_code"] == "000852.SH"].copy()
    iw["iw_date"] = iw["trade_date"].astype(int)
    iw = iw[["iw_date", "con_code"]].drop_duplicates()
    iw_dates = sorted(iw["iw_date"].unique())
    print(f"[1] 中证1000成分快照: {len(iw_dates)} 期, {iw['con_code'].nunique()} 只历史成分")

    # ---------- 2. 行情 (只保留成分股并集, 2019-06 起留出因子预热期) ----------
    member_codes = set(iw["con_code"].unique())
    px_files = sorted(glob.glob(os.path.join(DATA, "data_day1", "*.parquet")))
    print(f"[2] 行情文件: {len(px_files)} 个, 成分股 {len(member_codes)} 只")

    parts = []
    for f in px_files:
        if os.path.getsize(f) <= 1024:  # 跳过空/损坏文件
            continue
        d = os.path.basename(f)[:8]
        if d < "20190601":  # 60日预热 + 调仓日最早 202002
            continue
        df = pd.read_parquet(f, columns=["ts_code", "trade_date", "open", "high", "low",
                                         "close", "pct_chg", "vol"])
        df = df[df["ts_code"].isin(member_codes)]
        if len(df):
            parts.append(df)
    px = pd.concat(parts, ignore_index=True)
    px["trade_date"] = px["trade_date"].astype(int)
    px = px.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)
    print(f"    行情面板: {len(px):,} 行, {px['ts_code'].nunique()} 只, "
          f"{px['trade_date'].min()}~{px['trade_date'].max()}, 耗时{time.time()-t0:.0f}s")

    # ---------- 3. 日频价量因子 ----------
    px["r"] = px["pct_chg"] / 100.0
    mkt = px.groupby("trade_date")["r"].mean().rename("mkt_ret")
    px = px.merge(mkt, on="trade_date", how="left")
    px["ex_ret"] = px["r"] - px["mkt_ret"]

    g = px.groupby("ts_code", sort=False)
    px["ret_1m"] = g["r"].transform(lambda s: (1 + s).rolling(20).apply(np.prod, raw=True) - 1)
    px["ivol"] = g["ex_ret"].transform(lambda s: s.rolling(20).std(ddof=1))
    px["cum20"] = g["r"].transform(lambda s: (1 + s).rolling(20).apply(np.prod, raw=True))
    px["fwd_20"] = px.groupby("ts_code")["cum20"].shift(-20) - 1
    for w in (5, 10, 20, 60):
        px[f"momentum_{w}"] = g["close"].pct_change(w)
    for w in (5, 10, 20):
        px[f"volatility_{w}"] = g["r"].transform(lambda s, ww=w: s.rolling(ww).std(ddof=1))

    # Alpha101 核心4个 (向量化滚动)
    g = px.groupby("ts_code", sort=False)
    alpha006 = g.apply(lambda x: x["open"].rolling(10).corr(x["vol"]).mul(-1))
    if isinstance(alpha006, pd.DataFrame):
        alpha006 = alpha006.iloc[:, 0]
    px["alpha_006"] = alpha006.reset_index(level=0, drop=True)
    dc = px.groupby("ts_code")["close"].diff(1)
    min5 = dc.groupby(px["ts_code"]).transform(lambda s: s.rolling(5).min())
    max5 = dc.groupby(px["ts_code"]).transform(lambda s: s.rolling(5).max())
    px["alpha_009"] = np.where(min5 > 0, dc, np.where(max5 < 0, dc, -dc))
    dv = px.groupby("ts_code")["vol"].diff(1)
    px["alpha_012"] = np.sign(dv) * (-dc)
    hi20 = px.groupby("ts_code")["high"].transform(lambda s: s.rolling(20).mean())
    px["alpha_023"] = np.where(hi20 < px["high"], -px.groupby("ts_code")["high"].diff(2), 0.0)

    # ---------- 3B. 筹码因子 (6个, 日频滚动) ----------
    # 典型价量+筹码指标, 无逐笔数据下的可行近似
    g = px.groupby("ts_code", sort=False)

    # 3B.1 典型持仓成本 VWAP: 过去20日 Σ(amount)/Σ(vol), amount = (high+low+close)/3 * vol
    px["amt3"] = (px["high"] + px["low"] + px["close"]) / 3.0 * px["vol"]
    px["vwap_20"] = g["amt3"].transform(lambda s: s.rolling(20).sum()) / (
        1e-9 + g["vol"].transform(lambda s: s.rolling(20).sum()))

    # 3B.2 浮动盈亏率: 当前价相对20日VWAP的偏离 → 负=套牢筹码多, 正=获利筹码多
    px["float_pnl_20"] = (px["close"] - px["vwap_20"]) / (px["vwap_20"] + 1e-9)

    # 3B.3 获利盘比例 proxy: 过去20日中 close<当日close 的天数/总天数 → 该日收盘价的分位
    def _prof_pct(x):
        # x = 长度为 window 的 1D array; 算当日(x[-1])在窗口内的分位
        if len(x) < 20: return np.nan
        return float((x < x[-1]).sum()) / len(x)
    px["prof_pct_20"] = g["close"].transform(
        lambda s: s.rolling(20).apply(_prof_pct, raw=True))

    # 3B.4 筹码集中度: (VWAP窗口内 q75_vol_price - q25_vol_price)/vwap → 值越小越集中
    def _chip_conc(x, vol_w, n_win=20):
        # x=close 数组, vol_w=vol 数组, 算成交量加权的 q25/q75
        if len(x) < n_win or np.sum(vol_w) <= 0:
            return np.nan
        sv = np.argsort(x)
        xs = x[sv]; vs = vol_w[sv]
        cs = np.cumsum(vs); tot = cs[-1]
        if tot <= 0: return np.nan
        q25 = xs[np.searchsorted(cs, 0.25 * tot)]
        q75 = xs[np.searchsorted(cs, 0.75 * tot)]
        return (q75 - q25) / (np.mean(x) + 1e-9)

    def _roll_conc(df):
        close_arr = df["close"].values
        vol_arr = df["vol"].values
        out_arr = np.full(len(df), np.nan)
        n = len(df)
        for i in range(19, n):
            sl = slice(i-19, i+1)
            out_arr[i] = _chip_conc(close_arr[sl], vol_arr[sl], 20)
        return pd.Series(out_arr, index=df.index)

    px["chip_conc_20"] = np.nan
    for code, idx in g.indices.items():
        sub = px.loc[idx]
        px.loc[idx, "chip_conc_20"] = _roll_conc(sub).values

    # 3B.5 短期筹码偏移: 近5日 VWAP 相对 20日 VWAP 的偏离 → 反映近5日资金成本变化
    px["vwap_5"] = g["amt3"].transform(lambda s: s.rolling(5).sum()) / (
        1e-9 + g["vol"].transform(lambda s: s.rolling(5).sum()))
    px["chip_shift_5"] = (px["vwap_5"] - px["vwap_20"]) / (px["vwap_20"] + 1e-9)

    # 3B.6 正量占比: 过去20日 上涨日成交量 / 总成交量 → 买盘资金强度
    px["is_up"] = (px["pct_chg"] > 0).astype(float)
    px["pos_vol_20"] = g.apply(
        lambda x: (x["vol"] * x["is_up"]).rolling(20).sum() / (x["vol"].rolling(20).sum() + 1e-9)
    ).reset_index(level=0, drop=True)

    # 清理辅助列
    px.drop(columns=["amt3", "is_up", "vwap_5"], inplace=True)

    print(f"[3] 价量+筹码因子完成, 耗时{time.time()-t0:.0f}s")

    # ---------- 4. 月末快照 + 成分股对齐 (无前视) ----------
    # 因子快照取每月最后一个交易日(收盘), 供下月首日调仓使用 → 无前视
    cal = sorted(px["trade_date"].unique())
    cal_s = pd.Series(cal)
    month_last = cal_s.groupby(cal_s // 100).max().tolist()
    # 快照范围 2020-01 ~ end_ym
    month_last = [d for d in month_last if 20191201 <= d <= end_ym]
    print(f"[4] 月末快照月数: {len(month_last)} ({month_last[0]}~{month_last[-1]})")

    iw_by_date = {d: set(g2["con_code"]) for d, g2 in iw.groupby("iw_date")}
    iw_sorted_dates = iw_dates

    def latest_members(rebal_d):
        """调仓日 T 最近一期可用成分快照 (快照日期<=T)"""
        for d in reversed(iw_sorted_dates):
            if d <= rebal_d:
                return iw_by_date[d]
        return set()

    panel = pd.concat(
        [px[px["trade_date"] == d].assign(is_member=True) for d in month_last],
        ignore_index=True,
    )
    # 逐月过滤成分
    keep = []
    for d in month_last:
        members = latest_members(d)
        sub = panel[panel["trade_date"] == d]
        keep.append(sub[sub["ts_code"].isin(members)])
    panel = pd.concat(keep, ignore_index=True)
    panel = panel.drop(columns=["is_member"])
    panel = panel[panel.groupby("ts_code")["trade_date"].transform("count") >= 2]  # 至少2期
    print(f"    月度面板: {len(panel):,} 股-月, 平均 {len(panel)//len(month_last)} 股/月")

    # ---------- 5. 财务 PIT (2023-04 起有效, 之前 NaN) ----------
    fin = pd.read_parquet(os.path.join(DATA, "fundamental1", "fina_indicator_cache.parquet"))
    fin = fin[["ts_code", "ann_date", "roe", "or_yoy", "netprofit_yoy"]].dropna(subset=["ann_date"])
    fin["ann_date"] = fin["ann_date"].astype(str).str.replace("-", "").str[:8].astype(int)
    fin = fin.sort_values("ann_date")
    panel = panel.sort_values("trade_date")
    panel = pd.merge_asof(panel, fin, left_on="trade_date", right_on="ann_date",
                          by="ts_code", direction="backward")
    print(f"    PIT财务: roe非空 {panel['roe'].notna().sum():,}/{len(panel)}, 耗时{time.time()-t0:.0f}s")

    # ---------- 6. 行业 + is_traditional ----------
    im = pd.read_parquet(os.path.join(ROOT, "research", "studies", "study_008_enhancements",
                                      "data", "industry_map.parquet"))
    ind_map = dict(zip(im["ts_code"], im["industry"]))
    panel["industry"] = panel["ts_code"].map(ind_map).fillna("其他")

    # 科技行业黑名单 (电子/计算机/传媒/通信/电力设备/国防军工 一级)
    TECH_KEYWORDS = [
        "半导体", "元器件", "IT设备", "计算机设备", "软件服务", "IT服务", "软件开发",
        "互联网", "通信设备", "通信服务", "游戏", "数字媒体", "广告营销", "影视院线",
        "出版业", "电视广播", "光学光电子", "消费电子", "其他电子", "电子化学品",
        "电池", "电机", "风电设备", "光伏设备", "电源设备", "电网设备",
        "航天装备", "航空装备", "地面兵装", "船舶装备", "军工电子", "航海装备",
    ]
    panel["is_traditional"] = ~panel["industry"].isin(TECH_KEYWORDS)
    print(f"    传统行业占比: {panel['is_traditional'].mean()*100:.1f}%, "
          f"科技行业: {panel.loc[~panel['is_traditional'], 'industry'].value_counts().head(8).to_dict()}")

    # ---------- 7. 清理 + 输出 ----------
    CHIP_COLS = ["vwap_20", "float_pnl_20", "prof_pct_20", "chip_conc_20", "chip_shift_5", "pos_vol_20"]
    feat_cols = ["ret_1m", "ivol", "momentum_5", "momentum_10", "momentum_20", "momentum_60",
                 "volatility_5", "volatility_10", "volatility_20",
                 "alpha_006", "alpha_009", "alpha_012", "alpha_023",
                 "roe", "or_yoy", "netprofit_yoy"] + CHIP_COLS
    out_cols = ["trade_date", "ts_code", "industry", "is_traditional"] + feat_cols + ["fwd_20"]
    panel = panel[out_cols].copy()
    panel["fwd_20"] = panel["fwd_20"] * 100  # %

    # 反转/低波因子取负 (方向对齐: 值越大越好)
    panel["f_rev"] = -panel["ret_1m"]
    panel["f_ivol"] = -panel["ivol"]

    if keep_last:
        # 前向信号: 保留最后一期月末快照(其 fwd_20 为 NaN, 仅用特征不用标签), 其余月份仍需 fwd_20 非空
        last_month = month_last[-1]
        panel = panel[(panel["fwd_20"].notna()) | (panel["trade_date"] == last_month)]
    else:
        panel = panel.dropna(subset=["fwd_20"])

    panel.to_parquet(out, index=False)
    print(f"\n[7] 面板保存: {out}")
    print(f"    行数: {len(panel):,}, 月份: {panel['trade_date'].nunique()}, "
          f"股票: {panel['ts_code'].nunique()}, 耗时{time.time()-t0:.0f}s")
    print(f"    fwd_20 非空: {panel['fwd_20'].notna().sum():,}/{len(panel)}")
    print(f"    fwd_20 描述: {panel['fwd_20'].describe().round(2).to_dict()}")
    return panel


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="股票 ML 特征面板构建 (默认冻结回测口径)")
    ap.add_argument("--end-ym", type=int, default=20251231, help="面板月末截止 YYYYMMDD")
    ap.add_argument("--keep-last", action="store_true", help="保留最后一期月末快照(fwd_20 NaN)")
    ap.add_argument("--out", default=None, help="输出路径 (默认 stock_ml_panel_72m.parquet)")
    a = ap.parse_args()
    build_panel(end_ym=a.end_ym, keep_last=a.keep_last, out_path=a.out)

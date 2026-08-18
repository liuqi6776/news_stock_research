# -*- coding: utf-8 -*-
"""
investool 4433 法则选基 —— 基于本地 fund2 场外净值复现 (本地路径见 NAV_DIR)

参考 investool (https://github.com/axiaoxin-com/investool) 的选基逻辑:
  1. 4433 法则 (models/fund.go Is4433):
     - 近1年/2年/3年/5年/今年以来收益率排名均在同类型基金前 1/4 (25%)
     - 近6月/近3月收益率排名在同类型基金前 1/3 (33.3%)
     - 必须有 5 年业绩数据
  2. 附加风险指标 (investool 扩展筛选的近似, 从净值自算):
     - 近1年波动率 / 夏普比率 / 最大回撤

与 investool 的差异:
  - investool 用天天基金现成的同类排名(Rank/同类总数); 本脚本用本地净值
    pct_chg 复利累乘自行计算区间收益, 按 fund_type 分组排名, 完全可复现可回测。
  - investool 支持规模/基金经理年限等扩展筛选, 本地 fund_basic 暂无该数据
    (可通过 akshare/tushare 补, 见 README)。

用法:
  C:/Users/liuqi/anaconda3/python.exe studies/4433_selector/run_4433.py
  --types 股票型,混合型-偏股   # 限定类型, 默认主动权益
  --min-history 1200          # 最少净值条数(默认 5年≈1200)
  --head 20                   # 只看排名前N
  --out results/4433_result.csv
"""
import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

# 定位 fund_research 根目录 (studies/4433_selector/.. 上三级) 以复用 data_loader
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

NAV_DIR = r"D:\iquant_data\data_v2\fund2\nav"

# 区间 -> 净值条数窗口 (场外权益基金约 240 交易日/年)
WINDOWS = {
    "ret_1m": 20,
    "ret_3m": 60,
    "ret_6m": 120,
    "ret_1y": 240,
    "ret_2y": 480,
    "ret_3y": 720,
    "ret_5y": 1200,
}
# 4433 需要 5 年数据
MIN_HISTORY = WINDOWS["ret_5y"]

# 默认主动权益类型 (investool 默认全类型, 债券/货币意义有限)
DEFAULT_TYPES = [
    "股票型",
    "混合型-偏股",
    "混合型-灵活",
    "混合型-平衡",
    "混合型-绝对收益",
]


def load_basic():
    basic = pd.read_parquet(os.path.join(os.path.dirname(NAV_DIR), "fund_basic_O.parquet"))
    return basic


def calc_fund_metrics(pct, acc_nav):
    """给定 pct_chg(%) 序列与累计净值, 计算区间收益与风险指标"""
    pct = np.asarray(pct, dtype=float)
    acc_nav = np.asarray(acc_nav, dtype=float)
    out = {}
    # 区间收益: 复利累乘
    for name, n in WINDOWS.items():
        if len(pct) >= n:
            out[name] = float(np.prod(1.0 + pct[-n:] / 100.0) - 1.0)
        else:
            out[name] = np.nan
    # 今年以来 (当年第一个净值日起)
    # 由外层基于 date 计算, 此处返回 pct 与 acc_nav 由调用方补充
    # 近1年波动率 (年化)
    if len(pct) >= 240:
        r = pct[-240:] / 100.0
        out["vol_1y"] = float(r.std(ddof=1) * np.sqrt(252))
    else:
        out["vol_1y"] = np.nan
    # 近1年夏普 (rf=2%)
    if len(pct) >= 240 and not np.isnan(out["vol_1y"]) and out["vol_1y"] > 0:
        out["sharpe_1y"] = (out["ret_1y"] - 0.02) / out["vol_1y"]
    else:
        out["sharpe_1y"] = np.nan
    # 最大回撤: 基于复权累计曲线 (pct 累乘), 近1/3/5年及全期
    cum = np.cumprod(1.0 + pct / 100.0)
    for tag, n in [("mdd_1y", 240), ("mdd_3y", 720), ("mdd_5y", 1200), ("mdd_all", None)]:
        seg = cum if n is None else cum[-n:]
        if len(seg) < 2:
            out[tag] = np.nan
            continue
        peak = np.maximum.accumulate(seg)
        out[tag] = float((seg / peak - 1.0).min())
    return out


def main():
    ap = argparse.ArgumentParser(description="investool 4433 法则选基 (本地净值复现)")
    ap.add_argument("--types", default=",".join(DEFAULT_TYPES),
                    help="基金类型过滤, 逗号分隔; 空串=全部类型")
    ap.add_argument("--min-history", type=int, default=MIN_HISTORY,
                    help="最少净值条数 (默认 1200≈5年)")
    ap.add_argument("--rank-quartile", type=float, default=25.0,
                    help="4 条件排名阈值 %% (默认 25)")
    ap.add_argument("--rank-third", type=float, default=33.333,
                    help="3 条件排名阈值 %% (默认 33.333)")
    ap.add_argument("--head", type=int, default=0, help="只打印前 N 只 (0=全部)")
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "4433_result.csv"),
                    help="结果 CSV 输出路径")
    args = ap.parse_args()

    basic = load_basic()
    types = [t for t in args.types.split(",") if t]
    if types:
        basic = basic[basic["fund_type"].isin(types)].reset_index(drop=True)
    basic = basic[~basic["code"].isna()]
    codes = basic["code"].astype(str).tolist()
    print(f"待检测基金: {len(codes)} 只 (类型: {types or '全部'})")

    rows = []
    t0 = time.time()
    for i, code in enumerate(codes, 1):
        path = os.path.join(NAV_DIR, f"{code}.parquet")
        if not os.path.exists(path):
            continue
        try:
            df = pd.read_parquet(path, columns=["date", "pct_chg", "acc_nav"])
        except Exception:
            continue
        df = df.dropna(subset=["pct_chg"])
        if len(df) < args.min_history:
            continue
        dates = pd.to_datetime(df["date"])
        pct = df["pct_chg"].to_numpy(dtype=float)
        acc = df["acc_nav"].to_numpy(dtype=float)
        m = calc_fund_metrics(pct, acc)
        # 今年以来: 从最新年度 1-1 之后第一个净值日累计
        last_year = dates.max().year
        mask = dates >= pd.Timestamp(f"{last_year}-01-01")
        if mask.any():
            m["ret_ytd"] = float(np.prod(1.0 + pct[mask.values] / 100.0) - 1.0)
        else:
            m["ret_ytd"] = np.nan
        m["code"] = code
        m["name"] = basic.loc[basic["code"].astype(str) == code, "name"].values[0]
        m["fund_type"] = basic.loc[basic["code"].astype(str) == code, "fund_type"].values[0]
        m["first_date"] = dates.min().strftime("%Y-%m-%d")
        m["last_date"] = dates.max().strftime("%Y-%m-%d")
        m["n_rows"] = len(df)
        rows.append(m)
        if i % 500 == 0:
            print(f"  进度 {i}/{len(codes)}, 有效 {len(rows)}, 用时 {time.time()-t0:.0f}s")
    print(f"净值/历史满足条件: {len(rows)} 只, 用时 {time.time()-t0:.0f}s")

    res = pd.DataFrame(rows)
    res = res.sort_values("first_date")
    res = res.reset_index(drop=True)

    # 按 fund_type 分组, 对各区间收益排名 (pct rank, 越小越靠前)
    rank_cols = ["ret_1m", "ret_3m", "ret_6m", "ret_1y", "ret_2y", "ret_3y", "ret_5y", "ret_ytd"]
    for c in rank_cols:
        res[f"rk_{c}"] = res.groupby("fund_type")[c].rank(pct=True) * 100.0

    # 4433 法则
    res["is_4433"] = (
        res["ret_5y"].notna()
        & (res["rk_ret_1y"] <= args.rank_quartile)
        & (res["rk_ret_2y"] <= args.rank_quartile)
        & (res["rk_ret_3y"] <= args.rank_quartile)
        & (res["rk_ret_5y"] <= args.rank_quartile)
        & (res["rk_ret_ytd"] <= args.rank_quartile)
        & (res["rk_ret_6m"] <= args.rank_third)
        & (res["rk_ret_3m"] <= args.rank_third)
    )
    n_pass = int(res["is_4433"].sum())
    print(f"\n满足 4433 法则: {n_pass} 只")

    # 排序: 先 4433, 再按近3年收益
    res = res.sort_values(["is_4433", "ret_3y"], ascending=[False, False]).reset_index(drop=True)

    out_cols = [
        "is_4433", "code", "name", "fund_type", "first_date", "last_date",
        "ret_1m", "ret_3m", "ret_6m", "ret_1y", "ret_2y", "ret_3y", "ret_5y", "ret_ytd",
        "vol_1y", "sharpe_1y", "mdd_1y", "mdd_3y", "mdd_5y", "mdd_all", "n_rows",
    ]
    res = res[out_cols]
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    res.to_csv(args.out, index=False, encoding="utf-8-sig")
    print(f"结果已保存: {args.out}")

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 250)
    pd.set_option("display.float_format", lambda x: f"{x:.2f}")
    head = args.head or len(res)
    show = res.head(head).copy()
    # 预览: 收益/回撤转 %, 夏普保持比率
    pct_cols = ["ret_1m", "ret_3m", "ret_6m", "ret_1y", "ret_2y", "ret_3y", "ret_5y", "ret_ytd",
                "mdd_1y", "mdd_3y", "mdd_5y", "mdd_all"]
    for c in pct_cols:
        show[c] = (show[c] * 100).round(1).astype(str) + "%"
    print(f"\n{'='*30} 前 {head} 只 (按 4433 + 近3年收益排序) {'='*30}")
    print(show.to_string(index=False))


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""
因子字典验证 - 统一验证主脚本（按重要性逐个因子族）

对 factor_lib.FACTOR_REGISTRY 中每个因子:
  1. 月度截面构建（每月最后一个交易日, 中证1000成分股, PIT 对齐）
  2. 未来 20 日收益（data_day1 pct_chg 累乘, 含复权; 因子 T 日可得, 收益 T+1~T+20）
  3. Rank IC / ICIR / Newey-West t(lag=4) / 正IC占比
  4. 5 组单调性
  5. Top50 月度调仓回测（20bps 双边成本, 与中证1000对比）

性能: 按股票分组的日频/筹码/资金流面板一次性 concat + groupby 构建, 进程内缓存复用
（同一进程跑多个因子时不会重复读盘）。

用法:
    python research/factor_dic/run_validation.py                 # 全量
    python research/factor_dic/run_validation.py illiq_money_20  # 单因子
    python research/factor_dic/run_validation.py --group P0      # 按重要性分组
    python research/factor_dic/run_validation.py --top 5         # 按重要性前5个
"""
import os
import sys
import argparse
import time

import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config.settings import settings
from research.factor_dic.factor_lib import FACTOR_REGISTRY

DAY_DIR = settings.daily_data_path          # data_day1 日频
CYQ_DIR = settings.cyq_path                 # cyq1 筹码
MF_DIR = settings.moneyflow_path            # moneyflow1 资金流
IDX_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "chip_momentum", "data", "index_daily")
IW_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "chip_momentum", "data", "index_weight")
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

START_YEAR = 2020
FORWARD_DAYS = 20
COST_BPS = 20
TOP_N = 60
WINSOR = (0.01, 0.99)
INDEX_CODE = "000852.SH"   # 中证1000

_DAILY_COLS = ["ts_code", "open", "high", "low", "close", "vol", "amount", "pct_chg", "trade_date"]
_IW_CACHE = {}
_PANEL_CACHE = {}          # 进程内面板缓存: key -> (stocks, pct_df, cyq_g, mf_g, mkt_ret)


def load_trade_dates():
    return sorted(f[:8] for f in os.listdir(DAY_DIR) if f.endswith(".parquet"))


def load_index_weight(date_str):
    """加载 <= date_str 的最近一期中证1000成分股集合(带缓存)"""
    if date_str in _IW_CACHE:
        return _IW_CACHE[date_str]
    iw_dates = sorted(f[3:11] for f in os.listdir(IW_DIR) if f.startswith("iw_"))
    avail = [d for d in iw_dates if d <= date_str]
    out = None
    if avail:
        df = pd.read_parquet(os.path.join(IW_DIR, f"iw_{avail[-1]}.parquet"))
        out = set(df["con_code"].astype(str).str.strip())
    _IW_CACHE[date_str] = out
    return out


def _read_daily_parquet(fp, d):
    """读取单日日频 parquet, 兼容异常(列缺失/ts_code在索引/空文件等); 失败返回 None
    优先投影读, 失败则全量读并降级取可用列。"""
    df = None
    for cols in (_DAILY_COLS, None):
        try:
            df = pd.read_parquet(fp, columns=cols)
            break
        except Exception:
            df = None
    if df is None or len(df.columns) == 0:
        return None
    if "ts_code" not in df.columns and df.index.name == "ts_code":
        df = df.reset_index()
    if "ts_code" not in df.columns:
        return None
    if "trade_date" not in df.columns:
        df["trade_date"] = d
    return df


def _read_daily_dir(dirpath, trade_dates, all_codes):
    """读某目录(筹码/资金流)下全部 parquet, 过滤成分股, 返回 code->DataFrame(索引=日期字符串)"""
    frames = []
    for d in trade_dates:
        fp = os.path.join(dirpath, f"{d}.parquet")
        if not os.path.exists(fp):
            continue
        try:
            df = pd.read_parquet(fp)
        except Exception:
            continue
        if df is None or len(df.columns) == 0:
            continue
        if "ts_code" not in df.columns and df.index.name == "ts_code":
            df = df.reset_index()
        if "ts_code" not in df.columns:
            continue
        df = df[df["ts_code"].astype(str).isin(all_codes)]
        if df.empty:
            continue
        df["_d"] = d
        frames.append(df)
    if not frames:
        return {}
    big = pd.concat(frames, ignore_index=True)
    out = {code: g.drop(columns=["ts_code"]).set_index("_d").sort_index()
           for code, g in big.groupby("ts_code")}
    return out


def load_panels(trade_dates, all_codes, need):
    """构建并缓存面板: stocks(日频), pct_df(日收益宽表), cyq_g, mf_g, mkt_ret
    同一进程内对 (need, all_codes) 组合只读盘一次。
    """
    dkey = ("daily", tuple(all_codes))
    if dkey not in _PANEL_CACHE:
        t0 = time.time()
        frames, bad = [], []
        for d in trade_dates:
            fp = os.path.join(DAY_DIR, f"{d}.parquet")
            if not os.path.exists(fp):
                continue
            df = _read_daily_parquet(fp, d)
            if df is None:
                bad.append(d)
                continue
            df = df[df["ts_code"].astype(str).isin(all_codes)]
            if df.empty:
                continue
            df["_d"] = d
            frames.append(df)
        if bad:
            print(f"    [warn] {len(bad)} 天日频读取失败/跳过, 例如: {bad[:3]}")
        big = pd.concat(frames, ignore_index=True)
        stocks = {code: g.set_index("_d").sort_index()
                  for code, g in big.groupby("ts_code")}
        pct_df = big.pivot_table(index="_d", columns="ts_code", values="pct_chg")
        del big
        _PANEL_CACHE[dkey] = (stocks, pct_df)
        print(f"    [cache] 日频面板 {time.time()-t0:.0f}s ({len(stocks)}只, {len(pct_df)}天)")
    stocks, pct_df = _PANEL_CACHE[dkey]

    cyq_g, mf_g = {}, {}
    if need == "chip":
        ckey = ("chip", tuple(all_codes))
        if ckey not in _PANEL_CACHE:
            t0 = time.time()
            cyq_g = _read_daily_dir(CYQ_DIR, trade_dates, all_codes)
            _PANEL_CACHE[ckey] = cyq_g
            print(f"    [cache] 筹码面板 {time.time()-t0:.0f}s ({len(cyq_g)}只)")
        cyq_g = _PANEL_CACHE[ckey]
    if need == "mf":
        mkey = ("mf", tuple(all_codes))
        if mkey not in _PANEL_CACHE:
            t0 = time.time()
            mf_g = _read_daily_dir(MF_DIR, trade_dates, all_codes)
            _PANEL_CACHE[mkey] = mf_g
            print(f"    [cache] 资金流面板 {time.time()-t0:.0f}s ({len(mf_g)}只)")
        mf_g = _PANEL_CACHE[mkey]

    mkt_ret = None
    if need == "mkt":
        idx_fp = os.path.join(IDX_DIR, f"{INDEX_CODE}.parquet")
        if os.path.exists(idx_fp):
            idx_df = pd.read_parquet(idx_fp)
            idx_df["trade_date"] = idx_df["trade_date"].astype(str).str[:8]
            mkt_ret = idx_df.set_index("trade_date")["pct_chg"] / 100.0

    return stocks, pct_df, cyq_g, mf_g, mkt_ret


def newey_west_t(ics, lag=4):
    ics = np.asarray(ics, dtype=float)
    n = len(ics)
    if n < 2:
        return 0.0, 0.0
    mean = ics.mean()
    var = ics.var(ddof=1)
    for l in range(1, min(lag, n - 1) + 1):
        cov = np.cov(ics[:-l], ics[l:], ddof=1)[0, 1]
        var += 2 * (1 - l / (lag + 1)) * cov
    se = np.sqrt(max(var, 1e-12) / n)
    return mean / se, mean


def winsorize(s):
    lo, hi = s.quantile(WINSOR[0]), s.quantile(WINSOR[1])
    return s.clip(lo, hi)


def build_factor_series(fkey, df, cyq_df, mf_df, mkt_ret):
    """计算单只股票在全部交易日的因子序列(方向统一为"高值=好")"""
    entry = next(e for e in FACTOR_REGISTRY if e[0] == fkey)
    _, _, direction, fn, need = entry
    try:
        if need is None:
            s = fn(df)
        elif need == "chip":
            if cyq_df is None or cyq_df.empty:
                return None
            idx = df.index.intersection(cyq_df.index)
            if len(idx) < 5:
                return None
            s = fn(cyq_df.loc[idx])
            s = s.reindex(df.index)
        elif need == "mf":
            if mf_df is None or mf_df.empty:
                return None
            idx = df.index.intersection(mf_df.index)
            if len(idx) < 5:
                return None
            s = fn(mf_df.loc[idx])
            s = s.reindex(df.index)
        elif need == "mkt":
            if mkt_ret is None or len(mkt_ret) == 0:
                return None
            mr = mkt_ret.reindex(df.index).fillna(0.0)
            s = fn(df, mr)
        else:
            return None
    except Exception as e:
        print(f"    [warn] {fkey} 计算失败: {e}")
        return None
    if s is None:
        return None
    if direction == "neg":
        s = -s
    return s.astype(float)


def run_fast(fkey):
    entry = next(e for e in FACTOR_REGISTRY if e[0] == fkey)
    _, name, direction, _, need = entry
    print(f"\n{'='*70}\n验证因子: {fkey} ({name})  方向={direction}\n{'='*70}")

    trade_dates = load_trade_dates()
    months = {}
    for d in trade_dates:
        if d[:4] >= str(START_YEAR):
            months[d[:6]] = d
    rebal_dates = sorted(months.values())[:-1]
    print(f"[load] 调仓日 {len(rebal_dates)} 个: {rebal_dates[0]} ~ {rebal_dates[-1]}")

    all_codes = set()
    for rb in rebal_dates:
        members = load_index_weight(rb)
        if members:
            all_codes |= members
    all_codes = sorted(all_codes)
    print(f"[load] 成分股 {len(all_codes)} 只")

    stocks, pct_df, cyq_g, mf_g, mkt_ret = load_panels(trade_dates, all_codes, need)

    # 逐股计算因子全序列 + 未来收益
    print("[calc] 计算因子与未来收益...")
    factor_series, fwd_ret_series = {}, {}
    t0 = time.time()
    for i, code in enumerate(all_codes):
        df = stocks.get(code)
        if df is None or len(df) < 60:
            continue

        cyq_df = None
        if need == "chip":
            cyq_df = cyq_g.get(code)
            if cyq_df is not None:
                cyq_df = cyq_df[~cyq_df.index.duplicated(keep="last")]
                # cyq1 无 close, 从日频合并(筹码因子需要)
                if "close" not in cyq_df.columns:
                    cyq_df = cyq_df.join(df["close"], how="left")
        mf_df = None
        if need == "mf":
            mf_df = mf_g.get(code)
            if mf_df is not None:
                mf_df = mf_df[~mf_df.index.duplicated(keep="last")]

        # 未来 20 日收益: cum[t+20]/cum[t] - 1 (不含当日, PIT 干净)
        pct = df["pct_chg"].fillna(0.0)
        cum = (1 + pct / 100.0).cumprod()
        fwd_simple = cum.shift(-FORWARD_DAYS) / cum - 1.0

        s = build_factor_series(fkey, df, cyq_df, mf_df, mkt_ret)
        if s is not None and len(s.dropna()) > 0:
            factor_series[code] = s
            fwd_ret_series[code] = fwd_simple
        if (i + 1) % 500 == 0:
            print(f"  {i+1}/{len(all_codes)} ({time.time()-t0:.0f}s)")

    print(f"[calc] 有效个股 {len(factor_series)}/{len(all_codes)} ({time.time()-t0:.0f}s)")

    # 月度截面 IC + 分组
    ic_list, group_stats = [], {q: [] for q in range(5)}
    for rb in rebal_dates:
        members = load_index_weight(rb)
        if members is None:
            continue
        fvals, rvals = {}, {}
        for code in members:
            fs = factor_series.get(code)
            fr = fwd_ret_series.get(code)
            if fs is None or fr is None:
                continue
            if rb not in fs.index or rb not in fr.index:
                continue
            fv, rv = fs.loc[rb], fr.loc[rb]
            if pd.notna(fv) and pd.notna(rv):
                fvals[code] = fv
                rvals[code] = rv
        if len(fvals) < 50:
            continue
        f = pd.Series(fvals)
        r = pd.Series(rvals)
        fw = winsorize(f)
        ic = fw.rank().corr(r.rank())
        ic_list.append((rb, ic))
        try:
            q = pd.qcut(fw.rank(method="first"), 5, labels=False)
            for qq in range(5):
                sel = r[q == qq]
                if len(sel):
                    group_stats[qq].append(sel.mean())
        except Exception:
            pass

    ics = pd.Series([x[1] for x in ic_list], index=[x[0] for x in ic_list])
    ic_mean = ics.mean()
    ic_std = ics.std(ddof=1)
    icir = ic_mean / ic_std if ic_std > 0 else np.nan
    t_val, _ = newey_west_t(ics.values)
    pos_ratio = (ics > 0).mean()
    print(f"\n[IC 检验] n={len(ics)}  IC均值={ic_mean:.4f}  ICIR={icir:.4f}  "
          f"NW t(lag=4)={t_val:.2f}  正IC占比={pos_ratio:.1%}")

    print("[分组] 5组平均未来20日收益:")
    gmeans = []
    for q in range(5):
        vals = group_stats[q]
        m = np.nanmean(vals) if vals else np.nan
        gmeans.append(m)
        print(f"  Q{q+1}: {m:.4f}  (n={len(vals)})")

    # Top50 回测
    print(f"\n[回测] Top{TOP_N} 月度调仓, 成本 {COST_BPS}bps, 基准 {INDEX_CODE}")
    port_rets, bench_rets = [], []
    bench_daily = None
    bench_fp = os.path.join(IDX_DIR, f"{INDEX_CODE}.parquet")
    if os.path.exists(bench_fp):
        bdf = pd.read_parquet(bench_fp)
        bdf["trade_date"] = bdf["trade_date"].astype(str).str[:8]
        bench_daily = bdf.set_index("trade_date")

    for i, rb in enumerate(rebal_dates):
        members = load_index_weight(rb)
        if members is None:
            continue
        fvals = {}
        for code in members:
            fs = factor_series.get(code)
            if fs is None:
                continue
            if rb in fs.index and pd.notna(fs.loc[rb]):
                fvals[code] = fs.loc[rb]
        if len(fvals) < TOP_N:
            continue
        picks = pd.Series(fvals).nlargest(TOP_N).index
        if i + 1 >= len(rebal_dates):
            continue
        rb_next = rebal_dates[i + 1]
        hi = trade_dates.index(rb)
        hn = trade_dates.index(rb_next)
        hold_dates = trade_dates[hi + 1: hn + 1]
        sub = pct_df.reindex(columns=picks).reindex(hold_dates).fillna(0.0) / 100.0
        rm_daily = sub.mean(axis=1)
        gross = (1 + rm_daily).prod() - 1
        net = gross - COST_BPS / 10000.0
        port_rets.append(net)
        if bench_daily is not None:
            b = bench_daily["pct_chg"].reindex(hold_dates).fillna(0.0) / 100.0
            bench_rets.append((1 + b).prod() - 1)

    n = len(port_rets)
    backtest_stats = {}
    if n > 0:
        pr = pd.Series(port_rets)
        nav = (1 + pr).cumprod()
        assert nav.min() > 0, f"nav 跌破 0 (min={nav.min():.4f}): 组合月收益存在异常值, 先排查数据"
        port_nav = nav.iloc[-1]
        bench_nav = (1 + pd.Series(bench_rets)).prod() if bench_rets else np.nan
        years = n / 12.0
        cagr_p = port_nav ** (1 / years) - 1 if port_nav > 0 else np.nan
        cagr_b = bench_nav ** (1 / years) - 1 if bench_nav > 0 and not np.isnan(bench_nav) else np.nan
        mdd = ((nav.cummax() - nav) / nav.cummax()).max()   # 相对回撤, ∈[0,1]
        assert 0 <= mdd <= 1, f"MaxDD 超出 [0,1]: {mdd:.4f}"
        sharpe = pr.mean() / pr.std(ddof=1) * np.sqrt(12) if pr.std(ddof=1) > 0 else np.nan
        win = (pr > 0).mean()
        br = pd.Series(bench_rets) if bench_rets else pd.Series(np.nan, index=pr.index)
        ex = (1 + pr).prod() / (1 + br.dropna()).prod() - 1 if len(br.dropna()) > 0 else np.nan
        print(f"  组合: 累计 {port_nav:.3f}  年化 {cagr_p:.2%}  Sharpe {sharpe:.2f}  MaxDD {mdd:.2%}  月胜率 {win:.1%}")
        print(f"  基准: 累计 {bench_nav:.3f}  年化 {cagr_b:.2%}" if bench_nav == bench_nav else "  基准: 无")
        print(f"  超额(累计): {ex:.2%}")
        backtest_stats = dict(n_month=n, nav=port_nav, cagr=cagr_p, sharpe=sharpe,
                              mdd=mdd, win=win, bench_nav=bench_nav, cagr_b=cagr_b, excess=ex)
    else:
        print("  无有效回测期")

    # 保存结果(IC 序列 + 汇总文本)
    os.makedirs(OUT_DIR, exist_ok=True)
    ics.to_csv(os.path.join(OUT_DIR, f"ic_{fkey}.csv"))
    with open(os.path.join(OUT_DIR, f"summary_{fkey}.txt"), "w", encoding="utf-8") as fh:
        fh.write(f"factor={fkey}\nname={name}\ndirection={direction}\n")
        fh.write(f"ic_n={len(ics)}\nic_mean={ic_mean:.6f}\nicir={icir:.6f}\n")
        fh.write(f"nw_t={t_val:.4f}\npos_ratio={pos_ratio:.4f}\n")
        fh.write("group_means=" + ",".join(f"{x:.6f}" if x == x else "nan" for x in gmeans) + "\n")
        for k, v in backtest_stats.items():
            fh.write(f"{k}={v}\n")
    print(f"[保存] {os.path.join(OUT_DIR, f'summary_{fkey}.txt')} / ic_{fkey}.csv")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("factor", nargs="?", default=None)
    ap.add_argument("--group", default=None, help="P0/P1/P2")
    ap.add_argument("--top", type=int, default=None, help="按重要性取前 N 个")
    args = ap.parse_args()

    keys = [e[0] for e in FACTOR_REGISTRY]
    if args.factor:
        if args.factor not in keys:
            print(f"未知因子: {args.factor}. 可选: {keys}")
            sys.exit(1)
        run_fast(args.factor)
    elif args.group:
        # P0=流动性3+筹码4; P1=换手3+波动3+羊群2; P2=资金流3+反转/动量3
        ranges = {"P0": (0, 7), "P1": (7, 15), "P2": (15, 21)}
        r = ranges.get(args.group.upper())
        if r is None:
            print("group 可选: P0/P1/P2")
            sys.exit(1)
        for k in keys[r[0]:r[1]]:
            run_fast(k)
    elif args.top:
        for k in keys[: args.top]:
            run_fast(k)
    else:
        for k in keys:
            run_fast(k)


if __name__ == "__main__":
    main()

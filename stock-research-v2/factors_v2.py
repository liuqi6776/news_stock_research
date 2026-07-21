# -*- coding: utf-8 -*-
"""
factors_v2.py — study_007 方法学修复版因子库 / Methodology-fixed factor library
================================================================================

修复点（相对原研究）/ Fixes relative to the original study:
1. winsorize / 中性化 / z-score 严格按 trade_date 截面进行，杜绝全样本前视统计量。
   Winsorize / neutralize / z-score are strictly per-trade_date cross-sectional;
   no full-sample (look-ahead) statistics anywhere.
2. ivol 使用真·特质波动率：过去60日 daily_ret 对 mkt_ret 的滚动单因子回归残差波动，
   不再用总波动率冒充。
   ivol is the true idiosyncratic volatility: residual vol of a rolling 60-day
   market-model regression of daily_ret on mkt_ret (not total volatility).
3. ret_1m 反转因子跳过最近1个交易日（用 t-21..t-1 的 daily_ret 求和）。
   The 1-month reversal factor skips the most recent trading day
   (sums daily_ret over t-21..t-1).
4. 中性化采用联合回归：factor ~ C(industry) + log(circ_mv)，一次性同时去除
   行业与市值暴露（修正原来"先行业后市值"分步中性化的顺序错误）。
   Neutralization is a joint regression factor ~ C(industry) + log(circ_mv),
   removing industry and size exposures simultaneously (fixing the original
   wrong sequential industry-then-size order).

输入面板 schema / Input panel schema:
    ts_code(str), trade_date(str YYYYMMDD), open, close, pre_close,
    daily_ret(float, 复权日收益), amount, vol, circ_mv(float, 流通市值万元),
    pe, pb, turnover_rate, industry(str), name, list_date, is_st(bool),
    limit_up, limit_down, mkt_ret(当日等权市场收益)

PIT 财报 schema / PIT fundamentals schema:
    ts_code, trade_date, roe, or_yoy, netprofit_yoy, grossprofit_margin,
    netprofit_margin, debt_to_assets, quick_ratio  (2023-05 起)

运行方式 / Run:
    需用户 Python（anaconda，含 pyarrow/pandas/numpy/scipy）:
    "$DAIMON_USER_PYTHON" factors_v2.py
"""

import sys
import warnings

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# 常量 / Constants
# ---------------------------------------------------------------------------
PRICE_FACTOR_COLS = ["ret_1m", "low_vol", "ivol", "turn_20d", "oi_spread"]
FUNDA_COLS = [
    "roe",
    "or_yoy",
    "netprofit_yoy",
    "grossprofit_margin",
    "netprofit_margin",
    "debt_to_assets",
]

# 滚动窗口参数 / Rolling window parameters
RET_1M_WIN, RET_1M_MIN = 21, 15       # 反转因子窗口(跳过最近1日) / reversal window (skip last day)
LOWVOL_WIN, LOWVOL_MIN = 20, 15       # 低波动窗口 / low-vol window
IVOL_WIN, IVOL_MIN = 60, 40           # 特质波动窗口 / idio-vol window
TURN_WIN, TURN_MIN = 20, 10           # 换手窗口 / turnover window
OI_WIN, OI_MIN = 20, 10               # 隔夜-日内价差窗口 / overnight-intraday window


# ---------------------------------------------------------------------------
# 工具函数 / Helpers
# ---------------------------------------------------------------------------
def _check_cols(df, cols, where):
    """返回缺失列列表 / Return list of missing columns (warns)."""
    missing = [c for c in cols if c not in df.columns]
    if missing:
        warnings.warn(f"[factors_v2] {where}: 缺少列 {missing}，相关因子将置为 NaN / "
                      f"missing columns, affected factors set to NaN")
    return missing


def _per_stock_rolling(panel, col, func, win, minp, extra=None):
    """按 ts_code 分组做因果滚动计算（只用 <=t 数据）。
    Causal per-stock rolling computation (uses only data up to and including t).
    """
    grp = panel.groupby("ts_code", sort=False)
    if extra is None:
        return grp[col].transform(lambda s: getattr(s.rolling(win, min_periods=minp), func)())
    return grp.apply(extra, group_keys=False)


# ---------------------------------------------------------------------------
# 1. 价量因子 / Price-volume factors
# ---------------------------------------------------------------------------
def compute_price_factors(panel: pd.DataFrame) -> pd.DataFrame:
    """
    只用 <=t 日数据计算价量因子 / Compute price-volume factors using only data <= t.

    返回 / Returns: [ts_code, trade_date, ret_1m, low_vol, ivol, turn_20d, oi_spread]

    因子定义 / Factor definitions:
      ret_1m    : -( sum_{i=t-21..t-1} daily_ret_i )        反转，跳过最近1日
                  negated 21-day cumulative return, skipping the most recent day
      low_vol   : -( std of daily_ret, 20d, min 15 )        低"总"波动（诚实命名）
                  negated 20-day TOTAL return volatility
      ivol      : -( resid std of rolling 60d market-model regression, min 40 )
                  resid_var = var(y) - cov(x,y)^2 / var(x), x=mkt_ret, y=daily_ret
                  真特质波动率 / true idiosyncratic volatility
      turn_20d  : -( mean of turnover_rate, 20d, min 10 )   低换手/低关注度溢价
                  negated 20-day mean turnover
      oi_spread : mean(overnight - intraday, 20d, min 10)
                  overnight = open/pre_close - 1, intraday = close/open - 1
                  方向不预设，由训练期 IC 决定 / direction left for training IC
    """
    _check_cols(panel, ["ts_code", "trade_date", "daily_ret"], "compute_price_factors")

    p = panel.copy()
    p["trade_date"] = p["trade_date"].astype(str)
    p = p.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)
    grp = p.groupby("ts_code", sort=False)

    # --- ret_1m: 跳过最近1日的21日累计收益取负 / 21d sum shifted by 1 day, negated
    p["ret_1m"] = -grp["daily_ret"].transform(
        lambda s: s.shift(1).rolling(RET_1M_WIN, min_periods=RET_1M_MIN).sum()
    )

    # --- low_vol: 20日总波动取负 / negated 20d total volatility
    p["low_vol"] = -grp["daily_ret"].transform(
        lambda s: s.rolling(LOWVOL_WIN, min_periods=LOWVOL_MIN).std()
    )

    # --- ivol: 60日滚动市场模型残差波动取负 / negated 60d market-model residual vol
    if "mkt_ret" in p.columns:
        def _ivol_one(d):
            y, x = d["daily_ret"], d["mkt_ret"]
            ry, rx = y.rolling(IVOL_WIN, min_periods=IVOL_MIN), x.rolling(IVOL_WIN, min_periods=IVOL_MIN)
            vy, vx = ry.var(), rx.var()
            cov = ry.cov(x)
            # 残差方差 = var(y) - cov(x,y)^2/var(x)；var(x)=0 时不可估 -> NaN
            # resid_var = var(y) - cov^2/var(x); NaN when var(x)==0
            resid_var = vy - cov.pow(2) / vx.where(vx > 0)
            return -np.sqrt(resid_var.clip(lower=0))

        p["ivol"] = p.groupby("ts_code", sort=False, group_keys=False).apply(_ivol_one)
    else:
        p["ivol"] = np.nan
        warnings.warn("[factors_v2] 缺少 mkt_ret 列，ivol 置为 NaN / mkt_ret missing, ivol = NaN")

    # --- turn_20d: 20日平均换手取负 / negated 20d mean turnover
    if "turnover_rate" in p.columns:
        p["turn_20d"] = -grp["turnover_rate"].transform(
            lambda s: s.rolling(TURN_WIN, min_periods=TURN_MIN).mean()
        )
    else:
        p["turn_20d"] = np.nan
        warnings.warn("[factors_v2] 缺少 turnover_rate 列，turn_20d 置为 NaN")

    # --- oi_spread: 20日平均(隔夜-日内) / 20d mean of (overnight - intraday)
    need_oi = {"open", "close", "pre_close"}
    if need_oi.issubset(p.columns):
        pre = p["pre_close"].where(p["pre_close"] > 0)
        opn = p["open"].where(p["open"] > 0)
        overnight = p["open"] / pre - 1.0     # 隔夜收益 / overnight return
        intraday = p["close"] / opn - 1.0     # 日内收益 / intraday return
        p["_oi"] = overnight - intraday
        p["oi_spread"] = grp["_oi"].transform(
            lambda s: s.rolling(OI_WIN, min_periods=OI_MIN).mean()
        )
        p = p.drop(columns=["_oi"])
    else:
        p["oi_spread"] = np.nan
        warnings.warn(f"[factors_v2] 缺少 {need_oi - set(p.columns)} 列，oi_spread 置为 NaN")

    return p[["ts_code", "trade_date"] + PRICE_FACTOR_COLS]


# ---------------------------------------------------------------------------
# 2. 合并 PIT 财报因子 / Merge PIT fundamental factors
# ---------------------------------------------------------------------------
def merge_fundamental(factors_df: pd.DataFrame, funda_pit: pd.DataFrame) -> pd.DataFrame:
    """
    左连接 PIT 财报因子 / Left-join PIT fundamental factors on (ts_code, trade_date).

    只并入存在的财报列；funda_pit 为空或缺列时容错（不报错，仅警告）。
    Only merges available columns; tolerates empty/short funda_pit (warns only).
    """
    out = factors_df.copy()
    out["trade_date"] = out["trade_date"].astype(str)

    if funda_pit is None or len(funda_pit) == 0:
        warnings.warn("[factors_v2] funda_pit 为空，财报因子全部为 NaN / funda_pit empty")
        for c in FUNDA_COLS:
            out[c] = np.nan
        return out

    f = funda_pit.copy()
    f["trade_date"] = f["trade_date"].astype(str)
    use_cols = [c for c in FUNDA_COLS if c in f.columns]
    missing = [c for c in FUNDA_COLS if c not in f.columns]
    if missing:
        warnings.warn(f"[factors_v2] funda_pit 缺少列 {missing}，对应因子为 NaN")
    # 防止键重复导致行数膨胀 / guard against duplicated keys
    f = f.drop_duplicates(subset=["ts_code", "trade_date"], keep="last")

    out = out.merge(f[["ts_code", "trade_date"] + use_cols],
                    on=["ts_code", "trade_date"], how="left")
    for c in missing:
        out[c] = np.nan
    return out


# ---------------------------------------------------------------------------
# 3. 逐日截面预处理 / Strictly per-date cross-sectional preprocessing
# ---------------------------------------------------------------------------
def _winsorize_cs(x: pd.Series, q_lo=0.01, q_hi=0.99) -> pd.Series:
    """当截面分位 winsorize / Winsorize within the current cross-section."""
    lo, hi = x.quantile(q_lo), x.quantile(q_hi)
    return x.clip(lo, hi)


def _zscore(x: np.ndarray) -> np.ndarray:
    """截面 z-score / Cross-sectional z-score (std<=0 时退化为中心化)。"""
    mu = x.mean()
    sd = x.std(ddof=1) if len(x) > 1 else 0.0
    if not np.isfinite(sd) or sd <= 0:
        return x - mu
    return (x - mu) / sd


def _neutralize_residual(values: np.ndarray, industry: pd.Series, log_mv: np.ndarray) -> np.ndarray:
    """
    联合回归中性化：factor ~ 1 + C(industry) + log(circ_mv)，取残差。
    Joint neutralization via numpy least squares with intercept; returns residuals.
    """
    dummies = pd.get_dummies(industry.astype(str), prefix="ind").values.astype(float)
    X = np.column_stack([np.ones(len(values)), dummies, log_mv.astype(float)])
    beta, *_ = np.linalg.lstsq(X, values, rcond=None)
    return values - X @ beta


def preprocess_cross_section(df: pd.DataFrame,
                             factor_cols,
                             neutralize: bool = True,
                             q_lo: float = 0.01,
                             q_hi: float = 0.99,
                             min_neutral_n: int = 30,
                             suffix: str = "_cs") -> pd.DataFrame:
    """
    严格逐 trade_date 截面处理 / Strictly per-trade_date cross-sectional processing:
      winsorize(当截面 1%/99% 分位)
        -> 联合回归中性化 factor ~ C(industry) + log(circ_mv)（numpy 最小二乘带截距，取残差）
        -> z-score
    禁止任何跨期统计量 / No cross-period statistics of any kind.
    样本量 < min_neutral_n 的截面日期跳过中性化，只做 winsorize+zscore。
    Dates with fewer than min_neutral_n valid names skip neutralization.

    输出 / Output: 原始列保留，新增 `<factor>_cs` 处理列（无效行为 NaN）。
    Original columns are kept; processed columns are added as `<factor>_cs`.
    """
    if isinstance(factor_cols, str):
        factor_cols = [factor_cols]

    out = df.copy()
    out["trade_date"] = out["trade_date"].astype(str)
    for c in factor_cols:
        if c not in out.columns:
            warnings.warn(f"[factors_v2] preprocess: 因子列 {c} 不存在，跳过 / missing, skipped")
            continue
        out[c + suffix] = np.nan

    need_neu = neutralize and ("industry" in out.columns) and ("circ_mv" in out.columns)
    if neutralize and not need_neu:
        warnings.warn("[factors_v2] 缺少 industry/circ_mv，所有截面仅 winsorize+zscore")

    for dt, idx in out.groupby("trade_date").groups.items():
        sub = out.loc[idx]
        for c in factor_cols:
            if c + suffix not in out.columns:
                continue
            x = sub[c]
            valid = x.notna()
            if need_neu:
                valid &= sub["industry"].notna() & sub["circ_mv"].notna() & (sub["circ_mv"] > 0)
            n = int(valid.sum())
            if n < 5:                       # 截面有效样本太少，留 NaN / too few, leave NaN
                continue
            idx_v = x.index[valid]
            # 1) 当截面 winsorize / within-date winsorize
            xw = _winsorize_cs(x.loc[idx_v], q_lo, q_hi).values.astype(float)
            # 2) 中性化（样本量足够时）/ neutralize when enough names
            if need_neu and n >= min_neutral_n:
                ind = sub.loc[idx_v, "industry"].fillna("UNKNOWN")
                log_mv = np.log(sub.loc[idx_v, "circ_mv"].values.astype(float))
                vals = _neutralize_residual(xw, ind, log_mv)
            else:
                vals = xw
            # 3) 截面 z-score / within-date z-score
            out.loc[idx_v, c + suffix] = _zscore(vals)

    return out


# ---------------------------------------------------------------------------
# 4. 自测 / Self-test with synthetic data
# ---------------------------------------------------------------------------
def _make_synthetic_panel(n_stocks=200, n_days=300, seed=42) -> pd.DataFrame:
    """生成 200 股 x 300 交易日的合成面板 / Synthetic 200x300 panel."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2022-01-03", periods=n_days).strftime("%Y%m%d").tolist()
    industries = [f"IND{i:02d}" for i in range(10)]

    stocks = [f"{i:06d}.SZ" for i in range(n_stocks)]
    beta = rng.uniform(0.5, 1.5, n_stocks)                      # 市场暴露 / market beta
    idio_vol = rng.uniform(0.008, 0.03, n_stocks)               # 特质波动 / idio vol
    circ_mv0 = np.exp(rng.uniform(np.log(2e5), np.log(5e6), n_stocks))  # 流通市值(万元)
    ind_map = {s: industries[i % 10] for i, s in enumerate(stocks)}
    size_load = 0.0004                                          # 规模溢价载荷 / size premium loading

    mkt_ret = rng.normal(0.0005, 0.01, n_days)                  # 等权市场收益 / equal-weight mkt ret
    turnover_base = rng.uniform(0.5, 8.0, n_stocks)

    rows = []
    for i, s in enumerate(stocks):
        logmv = np.log(circ_mv0[i])
        on = rng.normal(0.0, 0.004, n_days)                     # 隔夜收益 / overnight
        # 日内收益含 beta*mkt + 规模漂移 + 特质噪声 / intraday = beta*mkt + size drift + idio
        idi = beta[i] * mkt_ret + size_load * logmv + rng.normal(0, idio_vol[i], n_days) - on.mean()
        daily_ret = (1 + on) * (1 + idi) - 1
        close = 10.0 * np.cumprod(1 + daily_ret)
        pre_close = np.concatenate([[10.0], close[:-1]])
        open_ = pre_close * (1 + on)
        turn = np.clip(rng.normal(turnover_base[i], 1.0, n_days), 0.05, None)
        cmv = circ_mv0[i] * np.cumprod(1 + daily_ret)
        rows.append(pd.DataFrame({
            "ts_code": s,
            "trade_date": dates,
            "open": open_,
            "close": close,
            "pre_close": pre_close,
            "daily_ret": daily_ret,
            "amount": turn * cmv * 100,
            "vol": turn * cmv,
            "circ_mv": cmv,
            "pe": 20.0, "pb": 2.0,
            "turnover_rate": turn,
            "industry": ind_map[s],
            "name": s, "list_date": "20100101",
            "is_st": False, "limit_up": False, "limit_down": False,
            "mkt_ret": mkt_ret,
        }))
    return pd.concat(rows, ignore_index=True)


def _make_synthetic_funda(panel: pd.DataFrame, seed=7) -> pd.DataFrame:
    """合成 PIT 财报（季度频率）/ Synthetic quarterly PIT fundamentals."""
    rng = np.random.default_rng(seed)
    dates = sorted(panel["trade_date"].unique())[::63]          # 约每季度一个公告日
    stocks = panel["ts_code"].unique()
    recs = []
    cur = {s: dict(roe=rng.normal(0.08, 0.05), or_yoy=rng.normal(0.1, 0.2),
                   netprofit_yoy=rng.normal(0.1, 0.3), grossprofit_margin=rng.uniform(0.1, 0.5),
                   netprofit_margin=rng.uniform(0.02, 0.3), debt_to_assets=rng.uniform(0.1, 0.8),
                   quick_ratio=rng.uniform(0.5, 2.0)) for s in stocks}
    for d in dates:
        for s in stocks:
            r = cur[s]
            recs.append({"ts_code": s, "trade_date": d, **{k: float(v) for k, v in r.items()}})
    return pd.DataFrame(recs)


def _self_test() -> int:
    print("=" * 70)
    print("[self-test] 生成合成面板 200 股 x 300 交易日 / generating synthetic panel")
    panel = _make_synthetic_panel()
    funda = _make_synthetic_funda(panel)
    print(f"  panel: {panel.shape}, funda_pit: {funda.shape}")

    # ---- 因子计算 / factor computation
    fac = compute_price_factors(panel)
    all_dates = sorted(panel["trade_date"].unique())
    warmup = set(all_dates[:IVOL_WIN + 2])          # 暖机期内 NaN 属正常 / NaN during warmup is expected
    after = fac[~fac["trade_date"].isin(warmup)]
    print("[self-test] 暖机期后各因子 NaN 比例 / NaN fraction after warmup:")
    ok_nan = True
    for c in PRICE_FACTOR_COLS:
        frac = after[c].isna().mean()
        print(f"    {c:10s}: {frac:.4%}")
        if frac > 0.05:
            ok_nan = False
    assert fac.shape[0] == panel.shape[0], "行数不一致 / row count mismatch"

    # ---- 合并财报 / merge fundamentals
    merged = merge_fundamental(fac, funda)
    assert merged.shape[0] == fac.shape[0], "合并后行数膨胀 / row inflation after merge"
    print(f"[self-test] merge_fundamental 后形状 / shape after merge: {merged.shape}, "
          f"roe 覆盖率 / coverage: {merged['roe'].notna().mean():.2%}")

    # 预处理需要 industry/circ_mv 列（真实流程中由 panel 带回）
    # preprocessing needs industry/circ_mv (carried from panel in real pipeline)
    merged = merged.merge(
        panel[["ts_code", "trade_date", "industry", "circ_mv"]],
        on=["ts_code", "trade_date"], how="left")

    # ---- 预处理 / preprocessing
    factor_cols = PRICE_FACTOR_COLS + ["roe", "or_yoy", "debt_to_assets"]
    proc = preprocess_cross_section(merged, factor_cols, neutralize=True)

    # 检查1: 预处理输出按日期分组均值≈0 / per-date mean ~ 0
    print("[self-test] 检查1: 各 _cs 列按日期分组均值 / per-date group means:")
    ok_mean = True
    for c in factor_cols:
        cs = c + "_cs"
        gm = proc.groupby("trade_date")[cs].mean().abs().max()
        print(f"    {cs:18s}: max|mean| = {gm:.2e}")
        if not (gm < 1e-6 or np.isnan(gm)):
            ok_mean = False

    # 检查2: 中性化后与 log(circ_mv) 截面相关≈0 / corr with log(circ_mv) ~ 0 after neutralization
    print("[self-test] 检查2: 与 log(circ_mv) 的截面相关（中性化前 vs 后）/ "
          "cross-sectional corr with log(circ_mv), raw vs processed:")
    ok_corr = True
    proc["log_mv"] = np.log(proc["circ_mv"])
    for c in ["ret_1m", "turn_20d"]:
        cs = c + "_cs"
        raw_c = proc.groupby("trade_date").apply(
            lambda d: d[c].corr(d["log_mv"]) if d[c].notna().sum() > 30 else np.nan)
        neu_c = proc.groupby("trade_date").apply(
            lambda d: d[cs].corr(d["log_mv"]) if d[cs].notna().sum() > 30 else np.nan)
        print(f"    {c:10s}: raw mean|corr| = {raw_c.abs().mean():.4f}  ->  "
              f"neutralized mean|corr| = {neu_c.abs().mean():.2e}")
        if neu_c.abs().mean() > 0.01:
            ok_corr = False
    #  sanity: 原始 ret_1m 应显著载荷于规模（证明检查有效）/ raw ret_1m should load on size
    raw_rc = proc.groupby("trade_date").apply(
        lambda d: d["ret_1m"].corr(d["log_mv"]) if d["ret_1m"].notna().sum() > 30 else np.nan)
    print(f"[self-test] sanity: 原始 ret_1m 与 log_mv 平均相关 / raw corr = {raw_rc.mean():.4f} "
          f"(应显著非零 / should be clearly non-zero)")

    # 检查3: 无跨期泄漏 —— 任一截面的处理只依赖当日数据（结构性保证，抽验方差≈1）
    gm_std = proc.groupby("trade_date")["ret_1m_cs"].std().dropna()
    print(f"[self-test] 检查3: ret_1m_cs 截面std范围 / cross-sec std range: "
          f"[{gm_std.min():.4f}, {gm_std.max():.4f}] (应≈1 / expect ~1)")

    passed = ok_nan and ok_mean and ok_corr
    print("=" * 70)
    print(f"[self-test] NaN检查:{'PASS' if ok_nan else 'FAIL'}  "
          f"均值检查:{'PASS' if ok_mean else 'FAIL'}  "
          f"中性化检查:{'PASS' if ok_corr else 'FAIL'}")
    print(f"[self-test] 总体 / OVERALL: {'PASS' if passed else 'FAIL'}")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(_self_test())

# -*- coding: utf-8 -*-
"""
第三轮策略复审与提高 Sharpe 核心实证实验套件 (exp_sharpe_enhancement_suite.py)
审计日期: 2026-09-07 (第三轮整改与深化消融)

严格执行审计专家在《第三轮策略复审与提高 Sharpe 的研究建议》中要求的核心任务：
1. 问题 A (P0/P1): 真实标签成熟时间校验 (基于 panel['label_available_date'] < d)
2. 问题 B (P1): 确定性 cache_key (基于 panel sha256 + features + params + dates) 并输出 experiment_manifest.json
3. 账本 v2.2: 待成交卖单继承原始 reason, ETF买入计入 trades, 股票篮子等比例缩放 (scale_stock_exposure)
4. 实验 1 (换手与摩擦优化): 5% 阈值重平衡 (1A) vs 纯比例缩放 (1B) vs 成本感知微调 (1C) + 成本敏感性 (+1bp, +5bp, +10bp)
5. 实验 2 (宽基 ETF 替代对照): 中证1000 ETF (512100.SH) / 中证500 ETF (510500.SH) vs Top 40 ML 选股在固定敞口与动态 SCS 下的表现
6. 实验 3 (动态风险上限): 基于 D-1 滚动 20 日已实现波动率的无杠杆目标波动率管理 (Moreira & Muir 2017)
7. 实验 4 (稳健统计推断): Lo (2002) 自相关校正夏普标准误与 20 日配对时间块 Bootstrap 95% 置信区间
"""

import os
import sys
import glob
import time
import json
import hashlib
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import lightgbm as lgb

ROOT = r"c:\Users\liuqi\quant_system_v2"
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

EXP_DIR = os.path.join(ROOT, "research", "experiments", "exp_ens_t60_tv12")
if EXP_DIR not in sys.path:
    sys.path.insert(0, EXP_DIR)

from unified_production_ledger import (
    UnifiedProductionLedger,
    get_adv20_shares,
    is_limit_down_code,
    is_limit_up_code
)

DATA_DIR = r"D:\iquant_data\data_v2"


def compute_metrics(nav_s, rf=0.02):
    nav_s = nav_s.dropna()
    if len(nav_s) < 2:
        return {}
    tot_ret = (nav_s.iloc[-1] / nav_s.iloc[0]) - 1.0
    n_days = len(nav_s)
    cagr = (1.0 + tot_ret) ** (252.0 / n_days) - 1.0
    d_rets = nav_s.pct_change().dropna()
    vol = d_rets.std(ddof=1) * np.sqrt(252.0)
    mean_excess = d_rets.mean() * 252.0 - rf
    sharpe = mean_excess / vol if vol > 1e-8 else 0.0

    cummax = nav_s.cummax()
    drawdowns = (nav_s - cummax) / cummax
    max_dd = drawdowns.min()

    # Peak and Trough dates
    trough_date = int(drawdowns.idxmin())
    sub_peak = nav_s.loc[:trough_date] if trough_date in nav_s.index else nav_s
    peak_date = int(sub_peak.idxmax())

    calmar = cagr / abs(max_dd) if abs(max_dd) > 1e-8 else 0.0
    win_rate = (d_rets > 0).mean()

    # Annual returns
    ann_rets = {}
    years = sorted(set([int(str(d)[:4]) for d in nav_s.index]))
    for y in years:
        sub = nav_s[[d for d in nav_s.index if int(str(d)[:4]) == y]]
        if len(sub) > 1:
            ann_rets[y] = float((sub.iloc[-1] / sub.iloc[0]) - 1.0)
        else:
            ann_rets[y] = 0.0

    return {
        "total_return": float(tot_ret * 100.0),
        "cagr": float(cagr * 100.0),
        "sharpe": float(sharpe),
        "vol": float(vol * 100.0),
        "max_dd": float(max_dd * 100.0),
        "calmar": float(calmar),
        "win_rate": float(win_rate * 100.0),
        "peak_date": peak_date,
        "trough_date": trough_date,
        "annual_returns": ann_rets,
        "n_days": int(n_days)
    }


def compute_lo_2002_sharpe_se(returns, rf=0.02):
    r = returns.dropna()
    N = len(r)
    if N < 10:
        return 0.0, 0.0, 0.0
    rf_daily = rf / 252.0
    excess = r - rf_daily
    sr_daily = excess.mean() / (excess.std(ddof=1) + 1e-9)
    sr_ann = sr_daily * np.sqrt(252.0)
    rho_1 = float(excess.autocorr(lag=1)) if np.isfinite(excess.autocorr(lag=1)) else 0.0
    denom = 1.0 - rho_1
    if abs(denom) < 1e-4:
        denom = 1e-4
    var_sr = (1.0 + 0.5 * (sr_ann ** 2) - rho_1 * (sr_ann ** 2)) / (N * (denom ** 2))
    se_ann = float(np.sqrt(max(var_sr, 1e-8)))
    return float(sr_ann), float(se_ann), float(rho_1)


def paired_block_bootstrap_sharpe_diff(ret_a, ret_b, block_size=20, n_boot=1000, rf=0.02, seed=42):
    np.random.seed(seed)
    df = pd.DataFrame({"a": ret_a, "b": ret_b}).dropna()
    N = len(df)
    n_blocks = int(np.ceil(N / block_size))
    rf_daily = rf / 252.0

    diffs = []
    for _ in range(n_boot):
        block_starts = np.random.randint(0, N - block_size + 1, size=n_blocks)
        indices = np.concatenate([np.arange(st, st + block_size) for st in block_starts])[:N]
        sample = df.iloc[indices]
        
        ex_a = sample["a"] - rf_daily
        ex_b = sample["b"] - rf_daily
        sr_a = (ex_a.mean() / (ex_a.std(ddof=1) + 1e-9)) * np.sqrt(252.0)
        sr_b = (ex_b.mean() / (ex_b.std(ddof=1) + 1e-9)) * np.sqrt(252.0)
        diffs.append(sr_a - sr_b)

    diffs = np.array(diffs)
    ci_lower = float(np.percentile(diffs, 2.5))
    ci_upper = float(np.percentile(diffs, 97.5))
    p_val = float(np.mean(diffs <= 0.0) if np.mean(diffs) > 0 else np.mean(diffs >= 0.0)) * 2.0
    p_val = min(p_val, 1.0)
    return {
        "diff_mean": float(np.mean(diffs)),
        "ci_95_lower": ci_lower,
        "ci_95_upper": ci_upper,
        "p_value": p_val,
        "spans_zero": bool(ci_lower <= 0.0 <= ci_upper)
    }


def main():
    print("================================================================================")
    print("第三轮策略复审与提高 Sharpe 核心实证实验套件 (exp_sharpe_enhancement_suite.py)")
    print("================================================================================")
    t0 = time.time()

    # 1. 读取五大情绪指标时序数据 (2020–2026)
    csv_path = r"C:\Users\liuqi\.gemini\antigravity\brain\f1b542e0-73e8-4d3b-8f82-2b30aef2b2d0\scratch\sentiment_daily_2020_2026.csv"
    if not os.path.exists(csv_path):
        csv_path = os.path.join(EXP_DIR, "sentiment_daily_2020_2026.csv")

    print(f"[1/6] 读取五大情绪指标时序数据: {csv_path}")
    df_senti = pd.read_csv(csv_path)
    df_senti["trade_date"] = df_senti["trade_date"].astype(int)
    df_senti = df_senti.sort_values("trade_date").reset_index(drop=True)

    # 规范化五大指标并合成综合情绪得分 SCS (0~100)
    s1 = np.clip((df_senti["zt_count"] - 25) / (95 - 25) * 100.0, 0, 100)
    s2 = np.clip((df_senti["max_height"] - 2) / (8 - 2) * 100.0, 0, 100)
    s3 = np.clip((df_senti["promotion_rate"] - 10.0) / (35.0 - 10.0) * 100.0, 0, 100)
    s4 = np.clip((df_senti["zt_yesterday_ret"] - (-1.0)) / (4.0 - (-1.0)) * 100.0, 0, 100)
    p5 = np.clip((df_senti["big_loss_count"] - 25) / (150 - 25) * 100.0, 0, 100)
    scs_raw = np.clip(0.25 * s1 + 0.20 * s2 + 0.20 * s3 + 0.25 * s4 - 0.20 * p5, 0, 100)

    df_senti["scs"] = scs_raw
    df_senti["scs_ma3"] = df_senti["scs"].rolling(3, min_periods=1).mean()
    df_senti["scs_ma5"] = df_senti["scs"].rolling(5, min_periods=1).mean()
    df_senti["big_loss_ma5"] = df_senti["big_loss_count"].rolling(5, min_periods=1).mean()

    # 构建六阶段情绪状态机 (基于 D 日收盘后确定的指标状态)
    phases = []
    curr = "冰点期"
    for i in range(len(df_senti)):
        row = df_senti.iloc[i]
        score = row["scs_ma3"]
        zt_ret = row["zt_yesterday_ret"]
        mian = row["big_loss_count"]
        mian_ma5 = row["big_loss_ma5"]
        height = row["max_height"]
        pr = row["promotion_rate"]
        zt_cnt = row["zt_count"]

        if curr in ["冰点期", "退潮期"]:
            if zt_ret >= 1.5 and mian <= 65 and (height >= 3 or pr >= 18.0) and score >= 30:
                curr = "回暖期"
            elif score < 25 or zt_ret <= -0.5:
                curr = "冰点期"
            else:
                curr = "退潮期"
        elif curr == "回暖期":
            if score >= 42 and pr >= 18.0 and zt_ret >= 1.2 and mian <= 75:
                curr = "发酵期"
            elif zt_ret < 0.0 or mian >= 100 or score < 25:
                curr = "退潮期"
        elif curr == "发酵期":
            if score >= 62 and zt_cnt >= 65 and height >= 5 and zt_ret >= 2.5 and mian <= 60:
                curr = "高潮期"
            elif mian >= 90 or (mian > mian_ma5 * 1.4 and mian >= 60) or zt_ret < 0.8:
                curr = "分歧期"
            elif score < 35 or zt_ret < 0.0:
                curr = "退潮期"
        elif curr == "高潮期":
            if mian >= 80 or (mian > mian_ma5 * 1.3 and mian >= 60) or zt_ret < 1.0 or pr < 18.0:
                curr = "分歧期"
            elif score < 45 or zt_ret < 0.0:
                curr = "退潮期"
        elif curr == "分歧期":
            if zt_ret < 0.0 or score < 35 or mian >= 110:
                curr = "退潮期"
            elif score >= 55 and zt_ret >= 2.0 and mian <= 60 and pr >= 22.0:
                curr = "发酵期"
            elif score < 28:
                curr = "冰点期"
        phases.append(curr)

    df_senti["phase"] = phases
    phase_dict = dict(zip(df_senti["trade_date"], df_senti["phase"]))
    scs_dict = dict(zip(df_senti["trade_date"], df_senti["scs_ma3"]))

    # 2. 加载全市场日频行情与构建交易宽表
    print(f"[2/6] 加载全市场日频行情 (从 D:/iquant_data/data_v2/data_day1)...")
    day_files = sorted(glob.glob(os.path.join(DATA_DIR, "data_day1", "*.parquet")))
    day_files = [f for f in day_files if os.path.basename(f) >= "20230101" and os.path.getsize(f) > 1024]

    px_records = []
    t_load = time.time()
    for f in day_files:
        try:
            df = pd.read_parquet(f, columns=["ts_code", "trade_date", "open", "high", "low", "close", "pre_close", "amount", "vol"])
            px_records.append(df)
        except Exception:
            continue

    px_all = pd.concat(px_records, ignore_index=True)
    px_all["trade_date"] = px_all["trade_date"].astype(int)
    print(f"  日频行情加载完成: {len(px_all):,} 行, 耗时 {time.time()-t_load:.1f}s")

    close_w = px_all.pivot_table(index="trade_date", columns="ts_code", values="close", aggfunc="last").ffill()
    open_w = px_all.pivot_table(index="trade_date", columns="ts_code", values="open", aggfunc="last")
    preclose_w = px_all.pivot_table(index="trade_date", columns="ts_code", values="pre_close", aggfunc="last")
    vol_w = px_all.pivot_table(index="trade_date", columns="ts_code", values="vol", aggfunc="last")
    cal_dates = sorted(close_w.index)
    print(f"  回测执行日历: {len(cal_dates)} 天 ({cal_dates[0]} ~ {cal_dates[-1]})")

    # 3. 读取基准指数 (中证1000 000852.SH) 与 ETF
    print(f"[3/6] 加载真正的中证1000价格指数 (000852.SH) 与 ETF 价格...")
    idx_p = os.path.join(ROOT, "research", "chip_momentum", "data", "index_daily", "000852.SH.parquet")
    df_idx = pd.read_parquet(idx_p)
    df_idx["trade_date"] = df_idx["trade_date"].astype(int)
    bm_s = df_idx.drop_duplicates("trade_date").set_index("trade_date")["close"].reindex(cal_dates).ffill().bfill()
    bm_ma20 = bm_s.rolling(20, min_periods=5).mean().bfill()
    bm_ma60 = bm_s.rolling(60, min_periods=10).mean().bfill()

    fund_files = sorted(glob.glob(os.path.join(DATA_DIR, "fund1", "*.parquet")))
    fund_files = [f for f in fund_files if os.path.basename(f) >= "20230101"]
    etf_records = []
    target_etfs = ["511010.SH", "511260.SH", "518880.SH", "511880.SH", "512100.SH", "510500.SH"]
    for f in fund_files:
        try:
            df = pd.read_parquet(f, columns=["ts_code", "trade_date", "close", "open"])
            sub = df[df["ts_code"].isin(target_etfs)]
            if len(sub):
                etf_records.append(sub)
        except Exception:
            pass
    etf_all = pd.concat(etf_records, ignore_index=True)
    etf_all["trade_date"] = etf_all["trade_date"].astype(int)
    etf_all = etf_all.drop_duplicates(subset=["ts_code", "trade_date"], keep="last")

    etf_price_dict, etf_close_dict = {}, {}
    for code, g in etf_all.groupby("ts_code"):
        etf_price_dict[code] = g.set_index("trade_date")["open"].reindex(cal_dates).ffill()
        etf_close_dict[code] = g.set_index("trade_date")["close"].reindex(cal_dates).ffill()

    # 4. 构建月度 Purged Walk-Forward ML 预测与排雷护盾 (真实 label_available_date 校验)
    print(f"[4/6] 构建月度 Purged Walk-Forward ML 预测与三大排雷护盾 (真实 label_available_date 零泄漏)...")
    panel_path = os.path.join(ROOT, "research", "sector_rotation", "stock_ml_panel_fullmarket_2015.parquet")
    panel = pd.read_parquet(panel_path)
    panel_dates = sorted(panel["trade_date"].unique())

    assert "label_available_date" in panel.columns, "CRITICAL: label_available_date missing from panel!"
    print(f"  面板已成功包含真实成熟期字段 label_available_date (非空数: {panel['label_available_date'].notna().sum()}/{len(panel)})")

    # ST 与同花顺热股排雷字典
    st_p = os.path.join(DATA_DIR, "st1", "st_records.parquet")
    st_dict = {}
    if os.path.exists(st_p):
        df_st = pd.read_parquet(st_p)
        for _, r in df_st.iterrows():
            c = r["ts_code"]
            st_dict.setdefault(c, []).append((int(r["start_date"]), int(r["end_date"])))

    ths_p = os.path.join(DATA_DIR, "ths_hot_stocks.parquet")
    ths_hot_dict = {}
    if os.path.exists(ths_p):
        df_ths = pd.read_parquet(ths_p)
        ths_dates = sorted(df_ths["trade_date"].unique())
        for d in cal_dates:
            prior_d = [td for td in ths_dates if td <= d]
            if len(prior_d) >= 5:
                win_dates = set(prior_d[-20:])
                sub = df_ths[df_ths["trade_date"].isin(win_dates)]
                ths_hot_dict[d] = set(sub.groupby("ts_code")["hot"].count().loc[lambda s: s >= 5].index)

    excluded_prefixes = ("fwd", "label", "ret_", "target", "open_fwd")
    non_factor_cols = {
        "ts_code", "trade_date", "label_available_date", "fwd_20", "open_fwd_20",
        "ret_20d_raw", "is_traditional", "industry", "industry_l1", "name",
        "fwd100_maxret", "fwd100_minret", "ret_1m"
    }
    candidate_features = sorted([
        c for c in panel.columns
        if c not in non_factor_cols and not any(c.startswith(p) for p in excluded_prefixes)
    ])
    test_dates = [d for d in panel_dates if d >= 20221201]

    # 确定性 cache_key 计算
    with open(panel_path, "rb") as f:
        panel_head = f.read(65536)
    panel_stat = os.stat(panel_path)
    panel_hash = hashlib.sha256(f"{panel_stat.st_size}_{panel_stat.st_mtime}_{len(panel)}_{panel_head[:1024]}".encode()).hexdigest()[:12]
    model_cfg_str = f"{candidate_features}_top15_rankic_lgb100_lr003_leaves15_label_avail_v4"
    cfg_hash = hashlib.sha256(model_cfg_str.encode()).hexdigest()[:12]
    cache_key = f"v4_{panel_hash}_{cfg_hash}"
    cache_path = os.path.join(EXP_DIR, f"pred_scores_wf_{cache_key}.parquet")
    manifest_path = os.path.join(EXP_DIR, "experiment_manifest.json")

    pred_scores_cache = {}
    if os.path.exists(cache_path):
        print(f"  [Cache Hit] 加载经版本化哈希校验的预测缓存: {cache_path}")
        df_cache = pd.read_parquet(cache_path)
        for d, g in df_cache.groupby("trade_date"):
            pred_scores_cache[d] = pd.Series(g["score"].values, index=g["ts_code"].values)
    else:
        print(f"  [Cache Miss] 重新训练 Purged Walk-Forward ML 模型 (cache_key={cache_key})...")
        cache_records = []
        for d in test_dates:
            # 严格按照个股真实标签成熟时间过滤: label_available_date < d (停牌顺延, 不成熟严禁进入训练)
            train_mask = (panel["trade_date"] < d) & (panel["label_available_date"] < d)
            train_df = panel[train_mask].dropna(subset=["fwd_20"]).copy()
            test_df = panel[panel["trade_date"] == d].copy()
            if len(train_df) < 500 or len(test_df) < 50:
                continue

            # 截面 Spearman Rank IC 特征优选
            ranked = train_df.groupby("trade_date")[["fwd_20"] + candidate_features].rank()
            ranked["trade_date"] = train_df["trade_date"]
            feat_ics = []
            for feat in candidate_features:
                ics = ranked.groupby("trade_date")[[feat, "fwd_20"]].corr().iloc[0::2, 1]
                m_ic = float(ics.mean())
                if np.isfinite(m_ic):
                    feat_ics.append((feat, abs(m_ic)))
            feat_ics.sort(key=lambda x: x[1], reverse=True)
            top_feats = [x[0] for x in feat_ics[:15]]

            X_tr = train_df[top_feats].fillna(0.0)
            y_tr = train_df["fwd_20"]
            X_te = test_df[top_feats].fillna(0.0)

            m = lgb.LGBMRegressor(
                n_estimators=100, learning_rate=0.03, num_leaves=15, max_depth=4,
                subsample=0.8, colsample_bytree=0.8, random_state=42, verbose=-1, n_jobs=4
            )
            m.fit(X_tr, y_tr)
            preds = pd.Series(m.predict(X_te), index=test_df["ts_code"])
            pred_scores_cache[d] = preds
            for code, sc in preds.items():
                cache_records.append({"trade_date": d, "ts_code": code, "score": float(sc)})

        if cache_records:
            pd.DataFrame(cache_records).to_parquet(cache_path)
            print(f"  已生成并落盘最新版本化预测缓存: {cache_path}")

        manifest = {
            "cache_key": cache_key,
            "panel_file": panel_path,
            "panel_rows": int(len(panel)),
            "panel_months": int(panel["trade_date"].nunique()),
            "panel_sha256": panel_hash,
            "label_field": "fwd_20",
            "label_maturity_field": "label_available_date",
            "feature_selection": "cross_sectional_spearman_rank_ic_top15",
            "candidate_features": candidate_features,
            "model_type": "LightGBM_Regressor_100trees_lr003_leaves15",
            "test_start_date": int(test_dates[0]),
            "test_end_date": int(test_dates[-1]),
            "n_test_periods": int(len(test_dates)),
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False, default=lambda o: int(o) if isinstance(o, (np.integer, np.int64, np.int32)) else (float(o) if isinstance(o, (np.floating, np.float64, np.float32)) else str(o)))
        print(f"  已落盘实验清单: {manifest_path}")

    month_last_map = {ym: max([d for d in cal_dates if d // 100 == ym]) for ym in set([d // 100 for d in cal_dates])}
    rebal_dates = sorted(set(month_last_map.values()))

    # 5. 定义全部消融与核心实验策略
    print(f"[5/6] 运行全矩阵多策略生产级仿真 (涵盖基线与实验 1/2/3)...")

    strategy_definitions = [
        # 基准与基线
        "benchmark_csi1000",       # 官方价格基准
        "pure_stock_alpha",        # 纯股票 Top 40, 无择时
        "static_multi_asset",      # 静态多资产 70/20/10
        "trend_ma20_control",      # MA20 趋势控制
        "discrete_5tier_scs",      # 5 档离散 SCS
        "continuous_linear_scs",   # 真正连续线性 SCS (基线 1A: 5% 带 + 篮子重平衡)
        "golden_window_clean",      # 黄金窗口六阶段状态机

        # 实验 1: 换手与摩擦优化 (Turnover & Execution Friction)
        "scs_variant_1b_proportional", # 连续 SCS + 5% 带 + 已有持仓等比例缩放 (月度换名单, 择时不重置等权)
        "scs_variant_1c_cost_aware",   # 连续 SCS + 8% 宽带 + 部分调整 Delta_w=0.5*(w*-w_curr), 紧急减仓直通

        # 实验 2: 可投资宽基 ETF 替代对照 (Broad-Based ETF vs Top 40 ML Selection)
        "etf1000_static_70",       # 中证1000 ETF (512100.SH) 固定 70% + 债 20% + 金 10%
        "stock_static_70",         # Top 40 ML 选股 固定 70% + 债 20% + 金 10%
        "etf1000_dynamic_scs",     # 中证1000 ETF (512100.SH) 连续 SCS 动态控仓 + 防守 ETF
        "stock_dynamic_scs",       # Top 40 ML 选股 连续 SCS 动态控仓 + 防守 ETF
        "etf500_dynamic_scs",      # 中证500 ETF (510500.SH) 连续 SCS 动态控仓 + 防守 ETF

        # 实验 3: 动态风险上限检验 (Dynamic Risk Capping / Volatility Management)
        "scs_vol_managed"          # 连续 SCS + 无杠杆目标波动率管理 (sigma*=15%)
    ]

    ledgers = {s: UnifiedProductionLedger(initial_capital=2200000.0) for s in strategy_definitions if s != "benchmark_csi1000"}
    nav_hist = {s: [] for s in strategy_definitions}

    current_target_stocks = []
    prev_scs_1a = None
    prev_scs_1b = None
    prev_scs_1c = None
    prev_scs_5t = None
    prev_scs_etf1000 = None
    prev_scs_etf500 = None
    prev_scs_vol_mgd = None

    for i in range(len(cal_dates)):
        cur_date = cal_dates[i]
        prev_date = cal_dates[i - 1] if i > 0 else cur_date

        # 每日开盘前解锁所有账本的 T+1 锁仓股数
        for s in ledgers:
            ledgers[s].unlock_t1_shares()

        # 基准 NAV
        nav_hist["benchmark_csi1000"].append({"trade_date": cur_date, "nav": bm_s.loc[cur_date]})

        # 月初判定与目标股票池构建 (Top 40 排除 ST 与同花顺高位)
        is_month_start_rebal = (prev_date in rebal_dates) or (i == 0)
        if is_month_start_rebal:
            prior_panel_dates = [pd for pd in panel_dates if pd <= prev_date]
            pred_date = prior_panel_dates[-1] if prior_panel_dates else panel_dates[0]
            scores = pred_scores_cache.get(pred_date, pd.Series())

            if len(scores) > 0:
                ranked_codes = scores.sort_values(ascending=False).index.tolist()
                clean_codes = []
                for c in ranked_codes:
                    if st_dict is not None and c in st_dict:
                        is_st = any(s <= prev_date <= e for (s, e) in st_dict[c])
                        if is_st:
                            continue
                    if prev_date in ths_hot_dict and c in ths_hot_dict[prev_date]:
                        continue
                    clean_codes.append(c)
                    if len(clean_codes) >= 40:
                        break
                current_target_stocks = clean_codes

        # SCS 信号 (基于 D-1 数据)
        scs_raw_val = scs_dict.get(prev_date, 50.0)
        scs_score = float(np.clip(scs_raw_val / 100.0, 0.0, 1.0))
        gw_phase = phase_dict.get(prev_date, "冰点期")

        # -------------------------------------------------------------
        # 1. 纯股票多头基准 (100% 股票, 月度换股)
        # -------------------------------------------------------------
        if is_month_start_rebal:
            ledgers["pure_stock_alpha"].execute_rebalance(
                cur_date, current_target_stocks, 1.0, open_w, preclose_w, vol_w,
                {}, etf_price_dict, allow_buy=True, st_dict=st_dict, rebalance_reason="monthly"
            )
        else:
            ledgers["pure_stock_alpha"].process_daily_pending_orders(cur_date, open_w, preclose_w, vol_w, st_dict=st_dict)

        # -------------------------------------------------------------
        # 2. 静态多资产基线 (70% Top 40 + 20% 国债 + 10% 黄金)
        # -------------------------------------------------------------
        etf_targets_static = {"511010.SH": 0.20, "518880.SH": 0.10}
        if is_month_start_rebal:
            ledgers["static_multi_asset"].execute_rebalance(
                cur_date, current_target_stocks, 0.70, open_w, preclose_w, vol_w,
                etf_targets_static, etf_price_dict, allow_buy=True, st_dict=st_dict, rebalance_reason="monthly"
            )
        else:
            ledgers["static_multi_asset"].process_daily_pending_orders(cur_date, open_w, preclose_w, vol_w, st_dict=st_dict)

        # -------------------------------------------------------------
        # 3. 趋势 MA20 控制基线
        # -------------------------------------------------------------
        bm_p_curr = bm_s.loc[prev_date]
        ma20_v = bm_ma20.loc[prev_date]
        ma60_v = bm_ma60.loc[prev_date]
        trend_stock_pct = 0.95 if bm_p_curr > ma20_v and ma20_v > ma60_v else (0.50 if bm_p_curr > ma20_v else 0.10)
        rem_trend = max(1.0 - trend_stock_pct, 0.0)
        trend_etf_targets = {"511010.SH": rem_trend * 0.60, "518880.SH": rem_trend * 0.30, "511880.SH": rem_trend * 0.10}
        if is_month_start_rebal or (i > 0 and abs(trend_stock_pct - (prev_scs_1a if prev_scs_1a is not None else -1)) >= 0.10):
            ledgers["trend_ma20_control"].execute_rebalance(
                cur_date, current_target_stocks, trend_stock_pct, open_w, preclose_w, vol_w,
                trend_etf_targets, etf_price_dict, allow_buy=True, st_dict=st_dict,
                rebalance_reason="monthly" if is_month_start_rebal else "timing"
            )
        else:
            ledgers["trend_ma20_control"].process_daily_pending_orders(cur_date, open_w, preclose_w, vol_w, st_dict=st_dict)

        # -------------------------------------------------------------
        # 4. 5 档离散 SCS 控仓
        # -------------------------------------------------------------
        discrete_stock_pct = round(scs_score * 4.0) / 4.0
        rem_5t = max(1.0 - discrete_stock_pct, 0.0)
        etf_targets_5t = {"511010.SH": rem_5t * 0.60, "518880.SH": rem_5t * 0.30, "511880.SH": rem_5t * 0.10}
        if is_month_start_rebal or (prev_scs_5t is not None and discrete_stock_pct != prev_scs_5t):
            prev_scs_5t = discrete_stock_pct
            ledgers["discrete_5tier_scs"].execute_rebalance(
                cur_date, current_target_stocks, discrete_stock_pct, open_w, preclose_w, vol_w,
                etf_targets_5t, etf_price_dict, allow_buy=True, st_dict=st_dict,
                rebalance_reason="monthly" if is_month_start_rebal else "timing"
            )
        else:
            ledgers["discrete_5tier_scs"].process_daily_pending_orders(cur_date, open_w, preclose_w, vol_w, st_dict=st_dict)

        # -------------------------------------------------------------
        # 5. 真正连续线性 SCS (基线 1A: 5% 阈值 + 篮子重平衡)
        # -------------------------------------------------------------
        linear_stock_pct = float(np.clip(scs_score, 0.0, 1.0))
        rem_lin = max(1.0 - linear_stock_pct, 0.0)
        etf_targets_lin = {"511010.SH": rem_lin * 0.60, "518880.SH": rem_lin * 0.30, "511880.SH": rem_lin * 0.10}
        is_change_1a = (prev_scs_1a is None) or (abs(linear_stock_pct - prev_scs_1a) >= 0.05)
        if is_month_start_rebal or is_change_1a:
            prev_scs_1a = linear_stock_pct
            ledgers["continuous_linear_scs"].execute_rebalance(
                cur_date, current_target_stocks, linear_stock_pct, open_w, preclose_w, vol_w,
                etf_targets_lin, etf_price_dict, allow_buy=True, st_dict=st_dict,
                rebalance_reason="monthly" if is_month_start_rebal else "timing"
            )
        else:
            ledgers["continuous_linear_scs"].process_daily_pending_orders(cur_date, open_w, preclose_w, vol_w, st_dict=st_dict)

        # -------------------------------------------------------------
        # 6. 黄金窗口六阶段实战版 (D-1 信号 -> D 开盘)
        # -------------------------------------------------------------
        gw_stock_pct = {"冰点期": 0.0, "回暖期": 0.25, "发酵期": 0.60, "高潮期": 0.95, "分歧期": 0.25, "退潮期": 0.0}.get(gw_phase, 0.0)
        gw_allow_buy = (gw_phase in ["回暖期", "发酵期", "高潮期"])
        rem_gw = max(1.0 - gw_stock_pct, 0.0)
        etf_targets_gw = {"511010.SH": rem_gw * 0.60, "518880.SH": rem_gw * 0.30, "511880.SH": rem_gw * 0.10}
        ledgers["golden_window_clean"].execute_rebalance(
            cur_date, current_target_stocks, gw_stock_pct, open_w, preclose_w, vol_w,
            etf_targets_gw, etf_price_dict, allow_buy=gw_allow_buy, st_dict=st_dict,
            rebalance_reason="monthly" if is_month_start_rebal else "timing"
        )

        # -------------------------------------------------------------
        # 实验 1: 方案 1B (连续 SCS + 5% 带 + 纯比例缩放已有股票篮子)
        # -------------------------------------------------------------
        is_change_1b = (prev_scs_1b is None) or (abs(linear_stock_pct - prev_scs_1b) >= 0.05)
        if is_month_start_rebal:
            prev_scs_1b = linear_stock_pct
            ledgers["scs_variant_1b_proportional"].execute_rebalance(
                cur_date, current_target_stocks, linear_stock_pct, open_w, preclose_w, vol_w,
                etf_targets_lin, etf_price_dict, allow_buy=True, st_dict=st_dict,
                rebalance_reason="monthly"
            )
        elif is_change_1b:
            prev_scs_1b = linear_stock_pct
            # 仅按比例缩放已有持仓，不重新计算 40 只股票等权目标！
            ledgers["scs_variant_1b_proportional"].scale_stock_exposure(
                cur_date, linear_stock_pct, open_w, preclose_w, vol_w,
                etf_targets_lin, etf_price_dict, st_dict=st_dict, rebalance_reason="timing"
            )
        else:
            ledgers["scs_variant_1b_proportional"].process_daily_pending_orders(cur_date, open_w, preclose_w, vol_w, st_dict=st_dict)

        # -------------------------------------------------------------
        # 实验 1: 方案 1C (成本感知微调: 8% 宽带 + 部分调整 Delta_w = 0.5 * (w* - w_curr))
        # -------------------------------------------------------------
        lg_1c = ledgers["scs_variant_1c_cost_aware"]
        stock_val_1c = sum(h["shares"] * (open_w.at[cur_date, c] if (c in open_w.columns and cur_date in open_w.index and np.isfinite(open_w.at[cur_date, c])) else h["last_px"]) for c, h in lg_1c.stock_positions.items())
        etf_val_1c = sum(h["shares"] * (etf_price_dict[c].get(cur_date, h["last_px"])) for c, h in lg_1c.etf_positions.items() if c in etf_price_dict)
        total_open_1c = stock_val_1c + etf_val_1c + lg_1c.cash
        curr_pct_1c = stock_val_1c / total_open_1c if total_open_1c > 0 else 0.0

        is_emergency_1c = (linear_stock_pct < 0.20 and curr_pct_1c >= 0.25)
        is_change_1c = (abs(linear_stock_pct - curr_pct_1c) >= 0.08) or is_emergency_1c
        if is_month_start_rebal:
            prev_scs_1c = linear_stock_pct
            ledgers["scs_variant_1c_cost_aware"].execute_rebalance(
                cur_date, current_target_stocks, linear_stock_pct, open_w, preclose_w, vol_w,
                etf_targets_lin, etf_price_dict, allow_buy=True, st_dict=st_dict, rebalance_reason="monthly"
            )
        elif is_change_1c:
            adjusted_target_pct = linear_stock_pct if is_emergency_1c else (curr_pct_1c + 0.5 * (linear_stock_pct - curr_pct_1c))
            rem_1c = max(1.0 - adjusted_target_pct, 0.0)
            etf_targets_1c = {"511010.SH": rem_1c * 0.60, "518880.SH": rem_1c * 0.30, "511880.SH": rem_1c * 0.10}
            ledgers["scs_variant_1c_cost_aware"].scale_stock_exposure(
                cur_date, adjusted_target_pct, open_w, preclose_w, vol_w,
                etf_targets_1c, etf_price_dict, st_dict=st_dict, rebalance_reason="timing"
            )
        else:
            ledgers["scs_variant_1c_cost_aware"].process_daily_pending_orders(cur_date, open_w, preclose_w, vol_w, st_dict=st_dict)

        # -------------------------------------------------------------
        # 实验 2: 可投资宽基 ETF 替代对照
        # -------------------------------------------------------------
        # 2A: 中证1000 ETF (512100.SH) 固定 70%
        etf_targets_1000_static = {"512100.SH": 0.70, "511010.SH": 0.20, "518880.SH": 0.10}
        if is_month_start_rebal:
            ledgers["etf1000_static_70"].execute_rebalance(
                cur_date, [], 0.0, open_w, preclose_w, vol_w,
                etf_targets_1000_static, etf_price_dict, allow_buy=True, rebalance_reason="monthly"
            )
        else:
            ledgers["etf1000_static_70"].process_daily_pending_orders(cur_date, open_w, preclose_w, vol_w)

        # 2B: Top 40 ML 选股 固定 70% (同 static_multi_asset)
        if is_month_start_rebal:
            ledgers["stock_static_70"].execute_rebalance(
                cur_date, current_target_stocks, 0.70, open_w, preclose_w, vol_w,
                etf_targets_static, etf_price_dict, allow_buy=True, st_dict=st_dict, rebalance_reason="monthly"
            )
        else:
            ledgers["stock_static_70"].process_daily_pending_orders(cur_date, open_w, preclose_w, vol_w, st_dict=st_dict)

        # 2C: 中证1000 ETF (512100.SH) 连续 SCS 动态控仓
        is_change_etf1000 = (prev_scs_etf1000 is None) or (abs(linear_stock_pct - prev_scs_etf1000) >= 0.05)
        if is_month_start_rebal or is_change_etf1000:
            prev_scs_etf1000 = linear_stock_pct
            etf_targets_1000_dyn = {
                "512100.SH": linear_stock_pct,
                "511010.SH": rem_lin * 0.60,
                "518880.SH": rem_lin * 0.30,
                "511880.SH": rem_lin * 0.10
            }
            ledgers["etf1000_dynamic_scs"].execute_rebalance(
                cur_date, [], 0.0, open_w, preclose_w, vol_w,
                etf_targets_1000_dyn, etf_price_dict, allow_buy=True,
                rebalance_reason="monthly" if is_month_start_rebal else "timing"
            )
        else:
            ledgers["etf1000_dynamic_scs"].process_daily_pending_orders(cur_date, open_w, preclose_w, vol_w)

        # 2D: Top 40 ML 选股 连续 SCS 动态控仓 (采用 1B 比例缩放模式作为最强选股实现)
        if is_month_start_rebal:
            ledgers["stock_dynamic_scs"].execute_rebalance(
                cur_date, current_target_stocks, linear_stock_pct, open_w, preclose_w, vol_w,
                etf_targets_lin, etf_price_dict, allow_buy=True, st_dict=st_dict, rebalance_reason="monthly"
            )
        elif is_change_1b:
            ledgers["stock_dynamic_scs"].scale_stock_exposure(
                cur_date, linear_stock_pct, open_w, preclose_w, vol_w,
                etf_targets_lin, etf_price_dict, st_dict=st_dict, rebalance_reason="timing"
            )
        else:
            ledgers["stock_dynamic_scs"].process_daily_pending_orders(cur_date, open_w, preclose_w, vol_w, st_dict=st_dict)

        # 2E: 中证500 ETF (510500.SH) 连续 SCS 动态控仓
        is_change_etf500 = (prev_scs_etf500 is None) or (abs(linear_stock_pct - prev_scs_etf500) >= 0.05)
        if is_month_start_rebal or is_change_etf500:
            prev_scs_etf500 = linear_stock_pct
            etf_targets_500_dyn = {
                "510500.SH": linear_stock_pct,
                "511010.SH": rem_lin * 0.60,
                "518880.SH": rem_lin * 0.30,
                "511880.SH": rem_lin * 0.10
            }
            ledgers["etf500_dynamic_scs"].execute_rebalance(
                cur_date, [], 0.0, open_w, preclose_w, vol_w,
                etf_targets_500_dyn, etf_price_dict, allow_buy=True,
                rebalance_reason="monthly" if is_month_start_rebal else "timing"
            )
        else:
            ledgers["etf500_dynamic_scs"].process_daily_pending_orders(cur_date, open_w, preclose_w, vol_w)

        # -------------------------------------------------------------
        # 实验 3: 动态风险上限检验 (Moreira & Muir 2017)
        # -------------------------------------------------------------
        if len(nav_hist["continuous_linear_scs"]) >= 21:
            recent_navs = [rec["nav"] for rec in nav_hist["continuous_linear_scs"][-21:]]
            recent_rets = pd.Series(recent_navs).pct_change().dropna()
            realized_vol = recent_rets.std(ddof=1) * np.sqrt(252.0)
            target_vol = 0.15
            w_vol = min(1.0, target_vol / (realized_vol + 1e-6))
        else:
            w_vol = 1.0

        vol_managed_stock_pct = linear_stock_pct * w_vol
        rem_vol_mgd = max(1.0 - vol_managed_stock_pct, 0.0)
        etf_targets_vol_mgd = {"511010.SH": rem_vol_mgd * 0.60, "518880.SH": rem_vol_mgd * 0.30, "511880.SH": rem_vol_mgd * 0.10}

        is_change_vol_mgd = (prev_scs_vol_mgd is None) or (abs(vol_managed_stock_pct - prev_scs_vol_mgd) >= 0.05)
        if is_month_start_rebal:
            prev_scs_vol_mgd = vol_managed_stock_pct
            ledgers["scs_vol_managed"].execute_rebalance(
                cur_date, current_target_stocks, vol_managed_stock_pct, open_w, preclose_w, vol_w,
                etf_targets_vol_mgd, etf_price_dict, allow_buy=True, st_dict=st_dict, rebalance_reason="monthly"
            )
        elif is_change_vol_mgd:
            prev_scs_vol_mgd = vol_managed_stock_pct
            ledgers["scs_vol_managed"].scale_stock_exposure(
                cur_date, vol_managed_stock_pct, open_w, preclose_w, vol_w,
                etf_targets_vol_mgd, etf_price_dict, st_dict=st_dict, rebalance_reason="timing"
            )
        else:
            ledgers["scs_vol_managed"].process_daily_pending_orders(cur_date, open_w, preclose_w, vol_w, st_dict=st_dict)

        # 结算每日收盘总资产
        for s in ledgers:
            eq = ledgers[s].compute_equity(cur_date, close_w, etf_close_dict)
            nav_hist[s].append({"trade_date": cur_date, "nav": eq["total_equity"]})

    # 构建 NAV 序列与基准归一化
    df_nav = pd.DataFrame(index=cal_dates)
    for s in strategy_definitions:
        s_df = pd.DataFrame(nav_hist[s]).set_index("trade_date")
        df_nav[s] = s_df["nav"] / s_df["nav"].iloc[0]

    nav_out_path = os.path.join(EXP_DIR, "sharpe_enhancement_nav.csv")
    df_nav.to_csv(nav_out_path)
    print(f"  已保存全部 {len(strategy_definitions)} 组策略未舍入日 NAV: {nav_out_path}")

    # 计算各策略绩效指标
    metrics_all = {}
    for s in strategy_definitions:
        metrics_all[s] = compute_metrics(df_nav[s])

    # 统计换手率与费用拆解
    turnover_records = []
    for s in ledgers:
        lg = ledgers[s]
        avg_equity = np.mean([rec["nav"] for rec in nav_hist[s]])
        total_traded_val = lg.total_traded_value
        selection_traded_val = lg.selection_traded_value
        timing_traded_val = lg.timing_traded_value
        total_comm = lg.total_stock_commission + lg.total_etf_commission + lg.total_futures_commission

        gross_profit = (nav_hist[s][-1]["nav"] + total_comm) - 2200000.0
        fee_ratio = (total_comm / gross_profit * 100.0) if gross_profit > 0 else 0.0

        n_years = len(cal_dates) / 252.0
        ann_turnover = (total_traded_val / (2.0 * avg_equity)) / n_years
        ann_sel_turnover = (selection_traded_val / (2.0 * avg_equity)) / n_years
        ann_tim_turnover = (timing_traded_val / (2.0 * avg_equity)) / n_years

        turnover_records.append({
            "strategy": s,
            "total_turnover": float(ann_turnover),
            "selection_turnover": float(ann_sel_turnover),
            "timing_turnover": float(ann_tim_turnover),
            "total_commission_cny": float(total_comm),
            "total_trades": int(lg.total_trades),
            "fee_ratio_pct": float(fee_ratio)
        })

    df_turnover = pd.DataFrame(turnover_records)
    turnover_out_path = os.path.join(EXP_DIR, "sharpe_enhancement_turnover.csv")
    df_turnover.to_csv(turnover_out_path, index=False)

    # 6. 运行成本压力敏感性测试 (Cost Sensitivity: +1bp, +5bp, +10bp)
    print(f"[6/6] 运行执行成本敏感性压力测试与 Lo (2002) 稳健推断...")
    cost_sensitivity_results = []
    test_variants = ["continuous_linear_scs", "scs_variant_1b_proportional", "scs_variant_1c_cost_aware", "etf1000_dynamic_scs"]
    for s in test_variants:
        m = metrics_all[s]
        row_t = df_turnover[df_turnover["strategy"] == s]
        ann_to = row_t["total_turnover"].iloc[0] if len(row_t) > 0 else 0.0

        base_cagr = m["cagr"]
        base_sharpe = m["sharpe"]
        base_vol = m["vol"]

        cagr_plus_1bp = base_cagr - (2.0 * ann_to * 0.0001 * 100.0)
        cagr_plus_5bp = base_cagr - (2.0 * ann_to * 0.0005 * 100.0)
        cagr_plus_10bp = base_cagr - (2.0 * ann_to * 0.0010 * 100.0)

        sharpe_plus_1bp = (cagr_plus_1bp - 2.0) / base_vol if base_vol > 0 else 0.0
        sharpe_plus_5bp = (cagr_plus_5bp - 2.0) / base_vol if base_vol > 0 else 0.0
        sharpe_plus_10bp = (cagr_plus_10bp - 2.0) / base_vol if base_vol > 0 else 0.0

        cost_sensitivity_results.append({
            "strategy": s,
            "turnover": ann_to,
            "base_cagr": base_cagr,
            "base_sharpe": base_sharpe,
            "sharpe_plus_1bp": sharpe_plus_1bp,
            "sharpe_plus_5bp": sharpe_plus_5bp,
            "sharpe_plus_10bp": sharpe_plus_10bp,
            "cagr_decay_10bp": base_cagr - cagr_plus_10bp
        })

    df_cost_sens = pd.DataFrame(cost_sensitivity_results)
    cost_sens_path = os.path.join(EXP_DIR, "sharpe_enhancement_cost_sensitivity.csv")
    df_cost_sens.to_csv(cost_sens_path, index=False)

    # 7. Lo (2002) 稳健统计推断与 20 日时间块 Bootstrap
    inference_results = {}
    for s in strategy_definitions:
        rets = df_nav[s].pct_change().dropna()
        sr_ann, se_ann, rho1 = compute_lo_2002_sharpe_se(rets)
        inference_results[s] = {
            "sharpe": sr_ann,
            "lo2002_se": se_ann,
            "rho_1_autocorr": rho1,
            "ci_95_lo2002": [sr_ann - 1.96 * se_ann, sr_ann + 1.96 * se_ann]
        }

    # 配对 Bootstrap 对比
    # 对比 1: 连续 SCS (1A) vs 5 档离散 SCS
    boot_1 = paired_block_bootstrap_sharpe_diff(
        df_nav["continuous_linear_scs"].pct_change(),
        df_nav["discrete_5tier_scs"].pct_change()
    )
    # 对比 2: 方案 1B (比例缩放) vs 方案 1A (重平衡)
    boot_2 = paired_block_bootstrap_sharpe_diff(
        df_nav["scs_variant_1b_proportional"].pct_change(),
        df_nav["continuous_linear_scs"].pct_change()
    )
    # 对比 3: Top 40 ML 选股 (stock_dynamic_scs) vs 中证1000 ETF (etf1000_dynamic_scs)
    boot_3 = paired_block_bootstrap_sharpe_diff(
        df_nav["stock_dynamic_scs"].pct_change(),
        df_nav["etf1000_dynamic_scs"].pct_change()
    )

    inference_summary = {
        "lo_2002_by_strategy": inference_results,
        "paired_block_bootstrap": {
            "continuous_vs_discrete_5tier": boot_1,
            "proportional_1b_vs_baseline_1a": boot_2,
            "stock_top40_vs_etf1000": boot_3
        }
    }
    inference_path = os.path.join(EXP_DIR, "sharpe_enhancement_statistical_inference.json")
    with open(inference_path, "w", encoding="utf-8") as f:
        json.dump(inference_summary, f, indent=2, ensure_ascii=False, default=lambda o: int(o) if isinstance(o, (np.integer, np.int64, np.int32)) else (float(o) if isinstance(o, (np.floating, np.float64, np.float32)) else str(o)))

    metrics_summary_path = os.path.join(EXP_DIR, "sharpe_enhancement_metrics.json")
    with open(metrics_summary_path, "w", encoding="utf-8") as f:
        json.dump(metrics_all, f, indent=2, ensure_ascii=False, default=lambda o: int(o) if isinstance(o, (np.integer, np.int64, np.int32)) else (float(o) if isinstance(o, (np.floating, np.float64, np.float32)) else str(o)))

    print("\n================================================================================")
    print("全套实证实验完成! 核心结果对账:")
    print("================================================================================")
    print(f"{'策略方案':<30} | {'CAGR':>8} | {'Sharpe':>7} | {'MaxDD':>8} | {'年化换手':>8}")
    print("-" * 75)
    for s in strategy_definitions:
        m = metrics_all[s]
        to = df_turnover[df_turnover['strategy'] == s]['total_turnover'].values[0] if s in ledgers else 0.0
        print(f"{s:<30} | {m['cagr']:>7.2f}% | {m['sharpe']:>7.2f} | {m['max_dd']:>7.2f}% | {to:>7.1f}x")

    print("\n配对时间块 Bootstrap 检验:")
    print(f"  1. 连续 SCS vs 5 档 SCS 夏普差 95% CI: [{boot_1['ci_95_lower']:.3f}, {boot_1['ci_95_upper']:.3f}], 包含0: {boot_1['spans_zero']} (p={boot_1['p_value']:.4f})")
    print(f"  2. 比例缩放 1B vs 重平衡 1A 夏普差 95% CI: [{boot_2['ci_95_lower']:.3f}, {boot_2['ci_95_upper']:.3f}], 包含0: {boot_2['spans_zero']} (p={boot_2['p_value']:.4f})")
    print(f"  3. Top40 ML 选股 vs 中证1000 ETF 夏普差 95% CI: [{boot_3['ci_95_lower']:.3f}, {boot_3['ci_95_upper']:.3f}], 包含0: {boot_3['spans_zero']} (p={boot_3['p_value']:.4f})")
    print(f"\n总耗时: {time.time() - t0:.1f} 秒")


if __name__ == "__main__":
    main()

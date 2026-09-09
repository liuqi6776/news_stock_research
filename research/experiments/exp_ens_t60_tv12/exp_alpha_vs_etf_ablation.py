# -*- coding: utf-8 -*-
"""第二优先严谨消融实验：主动选股 (GBDT-14 vs CS-Transformer vs ENS-Hybrid) 对比宽基 ETF 轮动
(Priority 2 Rigorous Ablation: Active Stock Selection vs. Broad-Base ETF Rotation under Unified SCS Timing & Multi-Asset Shelter)

严格统一测试标准：
1. 生产级单现金池真实微观账本 UnifiedProductionLedger v2.3
2. 统一 220 万元初始资金，相同 2023-01-03 ~ 2026-09-08 交易日历 (891 天)
3. 严格真实微观执行：100 股整手、开盘涨跌停拦截、T+1 状态锁定、10% 共享 ADV 约束
4. 股票双边 10 bps 摩擦与真实印花税；ETF 按 3 bps 摩擦且免征印花税
5. 6 大对比方案严格同口径对账与归因
"""
import os
import sys
import math
import time
import json
import glob
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
EXP_DIR = os.path.join(ROOT, "research", "experiments", "exp_ens_t60_tv12")
SEC_DIR = os.path.join(ROOT, "research", "sector_rotation")
DATA_DIR = r"D:\iquant_data\data_v2"

if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
if EXP_DIR not in sys.path:
    sys.path.insert(0, EXP_DIR)
if SEC_DIR not in sys.path:
    sys.path.insert(0, SEC_DIR)

from unified_production_ledger import (
    UnifiedProductionLedger,
    compute_metrics,
    compute_annual_returns,
)
from industry_l1 import build_l1_map

SENTIMENT_CSV = os.path.join(EXP_DIR, "sentiment_daily_2020_2026.csv")
PRED_CACHE_GBDT = os.path.join(EXP_DIR, "pred_scores_gbdt14_v23.parquet")
PRED_CACHE_CS = os.path.join(EXP_DIR, "pred_scores_cs_transformer_v23.parquet")
PRED_CACHE_ENS = os.path.join(EXP_DIR, "pred_scores_ens_hybrid_cs_v23.parquet")
REFINED_PANEL_FP = os.path.join(EXP_DIR, "stock_refined_factors_panel.parquet")

OUT_JSON = os.path.join(EXP_DIR, "alpha_vs_etf_ablation_report.json")
OUT_NAV_CSV = os.path.join(EXP_DIR, "alpha_vs_etf_ablation_nav.csv")


def load_st_dict():
    fp = os.path.join(ROOT, "research", "studies", "study_008_enhancements", "data", "st_history.parquet")
    if not os.path.exists(fp):
        return {}
    df_st = pd.read_parquet(fp)
    df_st["start_date"] = df_st["start_date"].astype(str)
    df_st["end_date"] = df_st["end_date"].astype(str).replace("None", "99999999")
    st_sub = df_st[df_st["name"].str.contains("ST", na=False)]
    st_dict = {}
    for code, g in st_sub.groupby("ts_code"):
        spans = []
        for _, row in g.iterrows():
            s = int(row["start_date"].replace("-", ""))
            e = int(row["end_date"].replace("-", ""))
            spans.append((s, e))
        st_dict[code] = spans
    return st_dict


def is_st_at_date(st_dict, code, cur_date):
    if code not in st_dict:
        return False
    for s, e in st_dict[code]:
        if s <= cur_date <= e:
            return True
    return False


def select_top_stocks_with_triple_shields(
    scores_in, ind_map, ind_l1_map, cur_date,
    st_dict, bad_consec_set=None, ths_hot_set=None,
    max_per_ind=4, max_per_ind_l1=8, top_n=40
):
    scores_in = scores_in.dropna()
    sorted_codes = scores_in.sort_values(ascending=False)
    selected = []
    ind_count, l1_count = {}, {}

    for code in sorted_codes.index:
        if is_st_at_date(st_dict, code, cur_date):
            continue
        if bad_consec_set is not None and code in bad_consec_set:
            continue
        if ths_hot_set is not None and code in ths_hot_set:
            continue

        ind = ind_map.get(code, "其他")
        l1 = ind_l1_map.get(code, "其他")
        if ind_count.get(ind, 0) >= max_per_ind:
            continue
        if l1_count.get(l1, 0) >= max_per_ind_l1:
            continue

        selected.append(code)
        ind_count[ind] = ind_count.get(ind, 0) + 1
        l1_count[l1] = l1_count.get(l1, 0) + 1
        if len(selected) >= top_n:
            break
    return selected


def main():
    t0 = time.time()
    print("=" * 80)
    print(">>> 启动第二优先消融实证：主动选股 (GBDT vs CS vs ENS) 对比宽基 ETF 轮动...")
    print("=" * 80)

    # 1. 加载五大情绪指标时序数据 (2020–2026)
    print(f"\n[1/6] 加载五大情绪指标数据: {SENTIMENT_CSV}")
    df_senti = pd.read_csv(SENTIMENT_CSV)
    df_senti["trade_date"] = df_senti["trade_date"].astype(int)
    df_senti = df_senti.sort_values("trade_date").reset_index(drop=True)

    s1 = np.clip((df_senti["zt_count"] - 25) / (95 - 25) * 100.0, 0, 100)
    s2 = np.clip((df_senti["max_height"] - 2) / (8 - 2) * 100.0, 0, 100)
    s3 = np.clip((df_senti["promotion_rate"] - 10.0) / (35.0 - 10.0) * 100.0, 0, 100)
    s4 = np.clip((df_senti["zt_yesterday_ret"] - (-1.0)) / (4.0 - (-1.0)) * 100.0, 0, 100)
    p5 = np.clip((df_senti["big_loss_count"] - 25) / (150 - 25) * 100.0, 0, 100)
    scs_raw = np.clip(0.25 * s1 + 0.20 * s2 + 0.20 * s3 + 0.25 * s4 - 0.20 * p5, 0, 100)

    df_senti["scs"] = scs_raw
    df_senti["scs_ma3"] = df_senti["scs"].rolling(3, min_periods=1).mean()
    senti_map = df_senti.set_index("trade_date")["scs_ma3"].to_dict()

    # 2. 读取 2023–2026 日频行情以对齐真实交易日历
    print(f"\n[2/6] 加载日频股票行情数据 (2023–2026)...")
    day_files = sorted(glob.glob(os.path.join(DATA_DIR, "data_day1", "*.parquet")))
    day_files = [f for f in day_files if os.path.basename(f) >= "20230101" and os.path.basename(f) <= "20260909"]

    px_records = []
    for f in day_files:
        try:
            df = pd.read_parquet(f, columns=["ts_code", "trade_date", "close", "open", "pre_close", "vol"])
            px_records.append(df)
        except Exception:
            pass
    px_all = pd.concat(px_records, ignore_index=True)
    px_all["trade_date"] = px_all["trade_date"].astype(int)

    close_w = px_all.pivot_table(index="trade_date", columns="ts_code", values="close", aggfunc="last").ffill()
    open_w = px_all.pivot_table(index="trade_date", columns="ts_code", values="open", aggfunc="last")
    preclose_w = px_all.pivot_table(index="trade_date", columns="ts_code", values="pre_close", aggfunc="last")
    vol_w = px_all.pivot_table(index="trade_date", columns="ts_code", values="vol", aggfunc="last")
    cal_dates = sorted(close_w.index)
    print(f"  回测交易日历: {len(cal_dates)} 天 ({cal_dates[0]} ~ {cal_dates[-1]})")

    # 3. 读取基准与 4 大 ETF 价格 (中证1000 ETF 512100.SH, 国债 511010.SH, 黄金 518880.SH, 货基 511880.SH)
    print("\n[3/6] 加载中证1000基准与 4 大多资产 ETF (512100.SH / 511010.SH / 518880.SH / 511880.SH)...")
    idx_p = os.path.join(ROOT, "research", "chip_momentum", "data", "index_daily", "000852.SH.parquet")
    df_idx = pd.read_parquet(idx_p)
    df_idx["trade_date"] = df_idx["trade_date"].astype(int)
    bm_s = df_idx.drop_duplicates("trade_date").set_index("trade_date")["close"].reindex(cal_dates).ffill().bfill()

    fund_files = sorted(glob.glob(os.path.join(DATA_DIR, "fund1", "*.parquet")))
    fund_files = [f for f in fund_files if os.path.basename(f) >= "20230101" and os.path.basename(f) <= "20260909"]
    target_etfs = {"512100.SH", "511010.SH", "518880.SH", "511880.SH"}
    etf_records = []
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
    etf_price_dict = {}
    etf_close_dict = {}
    for code, g in etf_all.groupby("ts_code"):
        etf_price_dict[code] = g.set_index("trade_date")["open"].reindex(cal_dates).ffill().bfill()
        etf_close_dict[code] = g.set_index("trade_date")["close"].reindex(cal_dates).ffill().bfill()

    for code in target_etfs:
        if code not in etf_price_dict:
            raise RuntimeError(f"Missing price series for target ETF: {code}")
        print(f"  成功载入 ETF [{code}] 行情: {len(etf_price_dict[code])} 天")

    # 4. 加载三大选股模型预测打分 (GBDT-14, CS-Transformer, ENS-Hybrid)
    print("\n[4/6] 加载三大独立选股打分缓存 (GBDT-14, CS-Transformer, ENS-Hybrid)...")
    scores_dict = {
        "GBDT14": pd.read_parquet(PRED_CACHE_GBDT),
        "CSTransformer": pd.read_parquet(PRED_CACHE_CS),
        "ENSHybrid": pd.read_parquet(PRED_CACHE_ENS)
    }
    # 构建快速查找结构: (model, trade_date) -> Series(ts_code -> score)
    scores_by_date = {m: {} for m in scores_dict}
    for m_name, df_sc in scores_dict.items():
        for d, grp in df_sc.groupby("trade_date"):
            scores_by_date[m_name][d] = grp.set_index("ts_code")["score"]
        print(f"  [{m_name}] 预测决策期覆盖: {len(scores_by_date[m_name])} 期")

    # 行业与排雷
    refined_panel = pd.read_parquet(REFINED_PANEL_FP)
    latest_ind = refined_panel.drop_duplicates("ts_code", keep="last")
    ind_map = dict(zip(latest_ind["ts_code"], latest_ind["industry"]))
    ind_l1_map = build_l1_map(ind_map)
    st_dict = load_st_dict()

    ths_p = os.path.join(EXP_DIR, "ths_hot_rank_2020_2026.parquet")
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

    # 5. 定义 6 组对比策略并在生产账本 v2.3 下同口径回测
    print("\n[5/6] 启动生产级真实微观账本 v2.3 (220万元/100股/T+1) 同口径对账...")
    strat_names = [
        "benchmark_csi1000",             # 0. 中证1000指数被动持有基准
        "etf_static_multi_asset",        # 1. 宽基 ETF 静态多资产配置 (40% 512100 + 36% 511010 + 18% 518880 + 6% 511880)
        "etf_scs_timing",                # 2. 宽基 ETF + SCS 动态情绪择时 (纯 ETF 轮动，零个股)
        "gbdt14_scs_timing",             # 3. GBDT-14 选股 + SCS 动态情绪择时 + 多资产避险
        "cs_transformer_scs_timing",     # 4. CS-Transformer 选股 + SCS 动态情绪择时 + 多资产避险
        "★ ens_hybrid_scs_timing"        # 5. ★ ENS-Hybrid 选股 + SCS 动态情绪择时 + 多资产避险 (终极方案)
    ]

    ledgers = {s: UnifiedProductionLedger(initial_capital=2_200_000.0, fee_bps=10.0, etf_fee_bps=3.0)
               for s in strat_names if s != "benchmark_csi1000"}

    # 跟踪状态变量
    prev_smooth_targets = {s: 0.0 for s in ledgers}
    current_stock_baskets = {
        "gbdt14_scs_timing": [],
        "cs_transformer_scs_timing": [],
        "★ ens_hybrid_scs_timing": []
    }
    model_key_map = {
        "gbdt14_scs_timing": "GBDT14",
        "cs_transformer_scs_timing": "CSTransformer",
        "★ ens_hybrid_scs_timing": "ENSHybrid"
    }

    nav_hist = {s: [] for s in strat_names}

    for i in range(len(cal_dates)):
        cur_date = cal_dates[i]
        is_month_start = (i == 0 or str(cur_date)[:6] != str(cal_dates[i-1])[:6])

        # 获取 D-1 日盘后决策信号 (零前瞻)
        if i == 0:
            decision_date = cur_date
            decision_scs = 50.0
        else:
            decision_date = cal_dates[i-1]
            decision_scs = senti_map.get(decision_date, 50.0)

        # 解锁当日可卖持仓
        for s, l in ledgers.items():
            l.unlock_t1_shares()

        # 计算连续线性 SCS 目标股票/权益比例 (0%~100%)
        raw_eq_target = float(np.clip((decision_scs - 25.0) / (75.0 - 25.0), 0.0, 1.0))

        # 月末选股信号触发 (仅针对 3 大主动选股策略)
        if is_month_start:
            # 查找最新决策期
            avail_months = [d for d in scores_by_date["ENSHybrid"] if d <= decision_date]
            if len(avail_months):
                latest_m = max(avail_months)
                ths_set = ths_hot_dict.get(cur_date, None)
                for s_name in current_stock_baskets:
                    m_key = model_key_map[s_name]
                    sc_s = scores_by_date[m_key].get(latest_m, pd.Series(dtype=float))
                    current_stock_baskets[s_name] = select_top_stocks_with_triple_shields(
                        sc_s, ind_map, ind_l1_map, cur_date, st_dict, ths_hot_set=ths_set, top_n=40
                    )

        # -----------------------------------------------------
        # 策略 1: 宽基 ETF 静态多资产组合 (每月再平衡到 40% 512100 + 36% 国债 + 18% 黄金 + 6% 货基)
        # -----------------------------------------------------
        etf_static_targets = {
            "512100.SH": 0.40,
            "511010.SH": 0.36,
            "518880.SH": 0.18,
            "511880.SH": 0.06
        }
        if is_month_start:
            ledgers["etf_static_multi_asset"].execute_rebalance(
                cur_date, [], 0.0, open_w, preclose_w, vol_w,
                etf_static_targets, etf_price_dict, allow_buy=True, rebalance_reason="monthly"
            )
        else:
            ledgers["etf_static_multi_asset"].process_daily_pending_orders(
                cur_date, open_w, preclose_w, vol_w, st_dict=st_dict
            )

        # -----------------------------------------------------
        # 方案 1C 成本感知平滑目标计算 (8% 宽带 + 0.5 调整系数 + 0.0 硬清仓通道)
        # -----------------------------------------------------
        def calc_smooth_target(raw_tgt, prev_tgt):
            if raw_tgt <= 0.001:
                return 0.0, (prev_tgt > 0.0)
            elif abs(raw_tgt - prev_tgt) >= 0.08:
                return prev_tgt + 0.50 * (raw_tgt - prev_tgt), True
            else:
                return prev_tgt, False

        # -----------------------------------------------------
        # 策略 2: 宽基 ETF + SCS 动态情绪择时 (纯 ETF 轮动，零个股)
        # -----------------------------------------------------
        s2_prev = prev_smooth_targets["etf_scs_timing"]
        s2_smooth, s2_change = calc_smooth_target(raw_eq_target, s2_prev)
        rem_s2 = max(1.0 - s2_smooth, 0.0)
        s2_etf_targets = {
            "512100.SH": s2_smooth,
            "511010.SH": rem_s2 * 0.60,
            "518880.SH": rem_s2 * 0.30,
            "511880.SH": rem_s2 * 0.10
        }
        if is_month_start:
            prev_smooth_targets["etf_scs_timing"] = s2_smooth
            ledgers["etf_scs_timing"].execute_rebalance(
                cur_date, [], 0.0, open_w, preclose_w, vol_w,
                s2_etf_targets, etf_price_dict, allow_buy=True, rebalance_reason="monthly"
            )
        elif s2_change:
            prev_smooth_targets["etf_scs_timing"] = s2_smooth
            ledgers["etf_scs_timing"].scale_stock_exposure(
                cur_date, 0.0, open_w, preclose_w, vol_w,
                s2_etf_targets, etf_price_dict, allow_buy=True, st_dict=st_dict
            )
        else:
            ledgers["etf_scs_timing"].process_daily_pending_orders(
                cur_date, open_w, preclose_w, vol_w, st_dict=st_dict
            )

        # -----------------------------------------------------
        # 策略 3/4/5: 主动选股 + SCS 动态情绪择时 + 多资产避险
        # -----------------------------------------------------
        for s_name in ["gbdt14_scs_timing", "cs_transformer_scs_timing", "★ ens_hybrid_scs_timing"]:
            s_prev = prev_smooth_targets[s_name]
            s_smooth, s_change = calc_smooth_target(raw_eq_target, s_prev)
            rem_s = max(1.0 - s_smooth, 0.0)
            etf_shelter = {
                "511010.SH": rem_s * 0.60,
                "518880.SH": rem_s * 0.30,
                "511880.SH": rem_s * 0.10
            }
            target_basket = current_stock_baskets[s_name]

            if is_month_start:
                prev_smooth_targets[s_name] = s_smooth
                ledgers[s_name].execute_rebalance(
                    cur_date, target_basket, s_smooth,
                    open_w, preclose_w, vol_w, etf_shelter, etf_price_dict,
                    allow_buy=(s_smooth > 0.0), st_dict=st_dict, rebalance_reason="monthly"
                )
            elif s_change:
                prev_smooth_targets[s_name] = s_smooth
                ledgers[s_name].scale_stock_exposure(
                    cur_date, s_smooth, open_w, preclose_w, vol_w,
                    etf_shelter, etf_price_dict, allow_buy=(s_smooth > 0.0), st_dict=st_dict
                )
            else:
                ledgers[s_name].process_daily_pending_orders(
                    cur_date, open_w, preclose_w, vol_w, st_dict=st_dict
                )

        # 每日收盘盯市与 NAV 记录
        nav_hist["benchmark_csi1000"].append({"trade_date": cur_date, "nav": bm_s.loc[cur_date] / bm_s.iloc[0]})
        for s in strat_names:
            if s != "benchmark_csi1000":
                eq = ledgers[s].compute_equity(cur_date, close_w, etf_close_dict)
                nav_hist[s].append({"trade_date": cur_date, "nav": eq["nav"]})

    # 6. 统计绩效指标与连乘自洽性数学检验
    print("\n" + "=" * 80)
    print(">>> 实验完成！统计各策略全景量化指标与连乘自洽性:")
    print("=" * 80)

    df_nav_all = pd.DataFrame({s: [r["nav"] for r in nav_hist[s]] for s in strat_names},
                              index=[r["trade_date"] for r in nav_hist["benchmark_csi1000"]])
    df_nav_all.to_csv(OUT_NAV_CSV)

    results_summary = {}
    years = sorted(list(set(int(str(d)[:4]) for d in df_nav_all.index)))

    for s in strat_names:
        nav_s = df_nav_all[s]
        m = compute_metrics(nav_s)
        ann = compute_annual_returns(nav_s)
        m["annual_returns"] = {str(k): round(v, 2) for k, v in ann.items()}

        if s == "benchmark_csi1000":
            m["trades"] = 0
            m["fees"] = 0.0
            m["stock_fees"] = 0.0
            m["etf_fees"] = 0.0
        else:
            l = ledgers[s]
            m["trades"] = l.total_trades
            m["fees"] = round(l.total_stock_commission + l.total_etf_commission, 2)
            m["stock_fees"] = round(l.total_stock_commission, 2)
            m["etf_fees"] = round(l.total_etf_commission, 2)
            m["limit_up_rejects"] = l.limit_up_rejections
            m["limit_down_locks"] = l.limit_down_locks

        # 严格数学连乘验证
        tot_actual = (nav_s.iloc[-1] / nav_s.iloc[0] - 1.0) * 100.0
        prod = 1.0
        for y in years:
            prod *= (1.0 + ann.get(y, 0.0) / 100.0)
        comp_tot = (prod - 1.0) * 100.0
        err = abs(tot_actual - comp_tot)
        m["compounding_err"] = round(err, 6)

        results_summary[s] = m

        print(f"  {s:<28} | CAGR: {m['cagr']:>6.2f}% | Sharpe: {m['sharpe']:>4.2f} | Vol: {m['vol']:>5.2f}% | MaxDD: {m['max_dd']:>6.2f}% | Calmar: {m['calmar']:>4.2f} | 交易: {m['trades']:>5}笔 | 手续费: {m['fees']/10000.0:>5.2f}万 | 连乘误差: {err:.6f}%")

    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(results_summary, f, ensure_ascii=False, indent=2)

    print(f"\n[保存] 消融实验评测报告已保存至: {OUT_JSON}")
    print(f"[保存] 每日净值数据已保存至: {OUT_NAV_CSV}")
    print(f"总耗时: {time.time() - t0:.1f} 秒")


if __name__ == "__main__":
    main()

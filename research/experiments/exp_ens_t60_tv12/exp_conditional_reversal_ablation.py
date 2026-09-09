# -*- coding: utf-8 -*-
"""方向A 条件反转深入消融实证：区分消息驱动下跌与交易流动性卖压下跌
(Direction A Conditional Reversal Ablation: News-Driven Drop vs. Liquidity Selling Pressure)

严格同口径统一运行于生产级单现金池微观真实账本 UnifiedProductionLedger v2.3:
- 100股整手 / 真实 T+1 状态机 / 10% ADV 流动性约束 / 开盘涨跌停拦截
- 股票双边 10 bps 摩擦与真实印花税, ETF 免印花税
- 连续线性 SCS 情绪周期择时与多资产避险分流 (Scheme 1C)

对比方案矩阵:
1. benchmark_csi1000: 中证1000买入持有基准
2. etf_scs_timing: 纯宽基 ETF (512100.SH) + SCS 情绪择时
3. cs_transformer_a0_raw: 原始 CS-Transformer 选股 (未过滤)
4. cs_transformer_a1_news_filter: CS-Transformer + 消息驱动大跌硬过滤 (剔除近30天暴雷且下跌个股)
5. cs_transformer_a1_pen: CS-Transformer + 消息驱动大跌软惩罚 (扣减 1.0 分)
6. cs_transformer_a2_selling_pressure: CS-Transformer + A1硬过滤 + 暂时卖压反转加成 (+0.2分)
7. gbdt14_a0_raw: 原始 GBDT-14 选股 (未过滤)
8. gbdt14_a1_news_filter: GBDT-14 + 消息驱动大跌硬过滤
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
from load_pit_fundamental_events import PITFundamentalEventManager
from load_true_liquidity_metrics import TrueLiquidityManager

SENTIMENT_CSV = os.path.join(EXP_DIR, "sentiment_daily_2020_2026.csv")
PRED_CACHE_CS = os.path.join(EXP_DIR, "pred_scores_cs_transformer_v23.parquet")
PRED_CACHE_GBDT = os.path.join(EXP_DIR, "pred_scores_gbdt14_v23.parquet")
REFINED_PANEL_FP = os.path.join(EXP_DIR, "stock_refined_factors_panel.parquet")

OUT_JSON = os.path.join(EXP_DIR, "conditional_reversal_report.json")
OUT_NAV_CSV = os.path.join(EXP_DIR, "conditional_reversal_nav.csv")


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


def select_top_stocks_with_filters(
    scores_in, ind_map, ind_l1_map, cur_date, st_dict,
    exclude_set=None, max_per_ind=4, max_per_ind_l1=8, top_n=40
):
    scores_in = scores_in.dropna()
    sorted_codes = scores_in.sort_values(ascending=False)
    selected = []
    ind_count, l1_count = {}, {}
    excl = exclude_set if exclude_set is not None else set()

    for code in sorted_codes.index:
        if is_st_at_date(st_dict, code, cur_date):
            continue
        if code in excl:
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
    print(">>> 启动方向A条件反转深入消融实证 (微观真实账本 v2.3)...")
    print("=" * 80)

    # 1. 情绪指标
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

    # 2. 读取 2023–2026 日频行情宽表
    print(f"\n[2/6] 加载日频股票行情宽表 (2023–2026)...")
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

    # 3. 读取基准与 4 大 ETF 价格 (512100.SH / 511010.SH / 518880.SH / 511880.SH)
    print("\n[3/6] 加载中证1000基准与 4 大多资产 ETF...")
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

    # 4. 加载打分缓存与基本面/流动性管理器
    print("\n[4/6] 加载模型预测打分、行业映射与 PIT 事件管理器...")
    scores_dict = {
        "CSTransformer": pd.read_parquet(PRED_CACHE_CS),
        "GBDT14": pd.read_parquet(PRED_CACHE_GBDT)
    }
    scores_by_date = {m: {} for m in scores_dict}
    for m_name, df_sc in scores_dict.items():
        for d, grp in df_sc.groupby("trade_date"):
            scores_by_date[m_name][d] = grp.set_index("ts_code")["score"]

    refined_panel = pd.read_parquet(REFINED_PANEL_FP)
    latest_ind = refined_panel.drop_duplicates("ts_code", keep="last")
    ind_map = dict(zip(latest_ind["ts_code"], latest_ind["industry"]))
    ind_l1_map = build_l1_map(ind_map)
    st_dict = load_st_dict()

    ret1m_map = refined_panel.set_index(["trade_date", "ts_code"])["ret_1m"].to_dict()

    pit_mgr = PITFundamentalEventManager()
    liq_mgr = TrueLiquidityManager()

    # 5. 预计算每个决策期的消息驱动大跌集合与流动性卖压集合
    print("\n[5/6] 预计算各决策期分类标签 (严格 PIT 保证)...")
    decision_dates = sorted(scores_by_date["CSTransformer"].keys())
    news_drop_dict = {}
    liquidity_drop_dict = {}

    for d in decision_dates:
        bad_news_set = pit_mgr.get_negative_news_stocks(d, lookback_calendar_days=30)
        codes = scores_by_date["CSTransformer"][d].index

        news_drops = set()
        liquidity_drops = set()

        liq_df = liq_mgr.compute_liquidity_metrics_for_date(d, window=20)
        surge_map = liq_df.set_index("ts_code")["turnover_surge_ratio"].to_dict() if not liq_df.empty else {}

        for c in codes:
            r1m = ret1m_map.get((d, c), 0.0)
            if r1m < 0.0:  # 下跌个股
                if c in bad_news_set:
                    news_drops.add(c)
                else:
                    # 跌且无负面利空: 伴随放量卖压异动 (换手率放大比率 > 1.1)
                    if surge_map.get(c, 1.0) > 1.1:
                        liquidity_drops.add(c)

        news_drop_dict[d] = news_drops
        liquidity_drop_dict[d] = liquidity_drops

    print("  -> 分类标签预计算完成.")

    # 6. 初始化回测账本集合
    strat_names = [
        "benchmark_csi1000",
        "etf_scs_timing",
        "cs_transformer_a0_raw",
        "cs_transformer_a1_news_filter",
        "cs_transformer_a1_pen",
        "cs_transformer_a2_selling_pressure",
        "gbdt14_a0_raw",
        "gbdt14_a1_news_filter"
    ]

    ledgers = {s: UnifiedProductionLedger(initial_capital=2_200_000.0, fee_bps=10.0, etf_fee_bps=3.0)
               for s in strat_names if s != "benchmark_csi1000"}
    prev_smooth_targets = {s: 0.0 for s in ledgers}
    current_stock_baskets = {
        "cs_transformer_a0_raw": [],
        "cs_transformer_a1_news_filter": [],
        "cs_transformer_a1_pen": [],
        "cs_transformer_a2_selling_pressure": [],
        "gbdt14_a0_raw": [],
        "gbdt14_a1_news_filter": []
    }
    bad_news_hit_tracker = {k: 0 for k in current_stock_baskets}
    total_basket_picks = {k: 0 for k in current_stock_baskets}

    nav_hist = {s: [] for s in strat_names}

    # 方案 1C 平滑
    def calc_smooth_target(raw_tgt, prev_tgt):
        if raw_tgt <= 0.001:
            return 0.0, (prev_tgt > 0.0)
        elif abs(raw_tgt - prev_tgt) >= 0.08:
            return prev_tgt + 0.50 * (raw_tgt - prev_tgt), True
        else:
            return prev_tgt, False

    print("\n[6/6] 启动 891 交易日高保真微观账本推进...")
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

        raw_eq_target = float(np.clip((decision_scs - 25.0) / (75.0 - 25.0), 0.0, 1.0))

        # 月末选股信号触发
        if is_month_start:
            avail_months = [d for d in decision_dates if d <= decision_date]
            if len(avail_months):
                latest_m = max(avail_months)
                news_drops = news_drop_dict.get(latest_m, set())
                liq_drops = liquidity_drop_dict.get(latest_m, set())
                bad_set_all = pit_mgr.get_negative_news_stocks(latest_m, lookback_calendar_days=30)

                # 1. CS-Transformer A0 (Raw)
                sc_cs = scores_by_date["CSTransformer"].get(latest_m, pd.Series(dtype=float)).copy()
                current_stock_baskets["cs_transformer_a0_raw"] = select_top_stocks_with_filters(
                    sc_cs, ind_map, ind_l1_map, cur_date, st_dict, exclude_set=None, top_n=40
                )

                # 2. CS-Transformer A1 (Hard Filter News Drop)
                current_stock_baskets["cs_transformer_a1_news_filter"] = select_top_stocks_with_filters(
                    sc_cs, ind_map, ind_l1_map, cur_date, st_dict, exclude_set=news_drops, top_n=40
                )

                # 3. CS-Transformer A1_pen (Soft Penalty)
                sc_cs_pen = sc_cs.copy()
                for c in news_drops:
                    if c in sc_cs_pen.index:
                        sc_cs_pen.loc[c] -= 1.0
                current_stock_baskets["cs_transformer_a1_pen"] = select_top_stocks_with_filters(
                    sc_cs_pen, ind_map, ind_l1_map, cur_date, st_dict, exclude_set=None, top_n=40
                )

                # 4. CS-Transformer A2 (A1 Hard Filter + Liquidity Reversal Boost)
                sc_cs_a2 = sc_cs.copy()
                for c in liq_drops:
                    if c in sc_cs_a2.index:
                        sc_cs_a2.loc[c] += 0.2
                current_stock_baskets["cs_transformer_a2_selling_pressure"] = select_top_stocks_with_filters(
                    sc_cs_a2, ind_map, ind_l1_map, cur_date, st_dict, exclude_set=news_drops, top_n=40
                )

                # 5. GBDT-14 A0 (Raw)
                sc_gbdt = scores_by_date["GBDT14"].get(latest_m, pd.Series(dtype=float)).copy()
                current_stock_baskets["gbdt14_a0_raw"] = select_top_stocks_with_filters(
                    sc_gbdt, ind_map, ind_l1_map, cur_date, st_dict, exclude_set=None, top_n=40
                )

                # 6. GBDT-14 A1 (Hard Filter News Drop)
                current_stock_baskets["gbdt14_a1_news_filter"] = select_top_stocks_with_filters(
                    sc_gbdt, ind_map, ind_l1_map, cur_date, st_dict, exclude_set=news_drops, top_n=40
                )

                for k in current_stock_baskets:
                    b = current_stock_baskets[k]
                    total_basket_picks[k] += len(b)
                    bad_news_hit_tracker[k] += len(set(b) & bad_set_all)

        # -----------------------------------------------------
        # 策略 1: 纯 ETF + SCS 情绪择时
        # -----------------------------------------------------
        s_etf_prev = prev_smooth_targets["etf_scs_timing"]
        s_etf_smooth, s_etf_change = calc_smooth_target(raw_eq_target, s_etf_prev)
        rem_etf = max(1.0 - s_etf_smooth, 0.0)
        s_etf_targets = {
            "512100.SH": s_etf_smooth,
            "511010.SH": rem_etf * 0.60,
            "518880.SH": rem_etf * 0.30,
            "511880.SH": rem_etf * 0.10
        }
        if is_month_start:
            prev_smooth_targets["etf_scs_timing"] = s_etf_smooth
            ledgers["etf_scs_timing"].execute_rebalance(
                cur_date, [], 0.0, open_w, preclose_w, vol_w,
                s_etf_targets, etf_price_dict, allow_buy=True, rebalance_reason="monthly"
            )
        elif s_etf_change:
            prev_smooth_targets["etf_scs_timing"] = s_etf_smooth
            ledgers["etf_scs_timing"].scale_stock_exposure(
                cur_date, 0.0, open_w, preclose_w, vol_w,
                s_etf_targets, etf_price_dict, allow_buy=True, st_dict=st_dict
            )
        else:
            ledgers["etf_scs_timing"].process_daily_pending_orders(cur_date, open_w, preclose_w, vol_w, st_dict=st_dict)

        # -----------------------------------------------------
        # 策略 2-7: 六大个股策略推进
        # -----------------------------------------------------
        for s_name in current_stock_baskets:
            prev_tgt = prev_smooth_targets[s_name]
            s_smooth, changed = calc_smooth_target(raw_eq_target, prev_tgt)
            b = current_stock_baskets[s_name]
            rem_def = max(1.0 - s_smooth, 0.0)
            etf_shelter = {
                "511010.SH": rem_def * 0.60,
                "518880.SH": rem_def * 0.30,
                "511880.SH": rem_def * 0.10
            }

            if is_month_start:
                prev_smooth_targets[s_name] = s_smooth
                ledgers[s_name].execute_rebalance(
                    cur_date, b, s_smooth, open_w, preclose_w, vol_w,
                    etf_shelter, etf_price_dict, allow_buy=(s_smooth > 0.0), rebalance_reason="monthly"
                )
            elif changed:
                prev_smooth_targets[s_name] = s_smooth
                ledgers[s_name].scale_stock_exposure(
                    cur_date, s_smooth, open_w, preclose_w, vol_w,
                    etf_shelter, etf_price_dict, allow_buy=(s_smooth > 0.0), st_dict=st_dict
                )
            else:
                ledgers[s_name].process_daily_pending_orders(cur_date, open_w, preclose_w, vol_w, st_dict=st_dict)

        # 每日收盘盯市与 NAV 记录
        nav_hist["benchmark_csi1000"].append({"trade_date": cur_date, "nav": float(bm_s.loc[cur_date] / bm_s.iloc[0])})
        for s in strat_names:
            if s != "benchmark_csi1000":
                eq = ledgers[s].compute_equity(cur_date, close_w, etf_close_dict)
                nav_hist[s].append({"trade_date": cur_date, "nav": eq["nav"]})

    # 7. 汇总绩效指标与对账
    print("\n" + "=" * 80)
    print(">>> 实验完成！统计各策略全景量化指标与连乘自洽性:")
    print("=" * 80)

    df_nav_all = pd.DataFrame({s: [r["nav"] for r in nav_hist[s]] for s in strat_names},
                              index=[r["trade_date"] for r in nav_hist["benchmark_csi1000"]])
    df_nav_all.to_csv(OUT_NAV_CSV)
    print(f"[OK] 每日净值数据已导出: {OUT_NAV_CSV}")

    results_summary = {}
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
            m["bad_news_hits"] = 0
            m["bad_news_hit_rate"] = 0.0
        else:
            l = ledgers[s]
            m["trades"] = l.total_trades
            m["fees"] = round(l.total_stock_commission + l.total_etf_commission, 2)
            m["stock_fees"] = round(l.total_stock_commission, 2)
            m["etf_fees"] = round(l.total_etf_commission, 2)
            m["limit_up_rejects"] = l.limit_up_rejections
            m["limit_down_locks"] = l.limit_down_locks
            if s in bad_news_hit_tracker:
                m["bad_news_hits"] = bad_news_hit_tracker[s]
                tot_p = total_basket_picks[s]
                m["bad_news_hit_rate"] = round(bad_news_hit_tracker[s] / max(1, tot_p) * 100, 2)
            else:
                m["bad_news_hits"] = 0
                m["bad_news_hit_rate"] = 0.0

        cum_ann = 1.0
        for yr, r in ann.items():
            cum_ann *= (1.0 + r / 100.0)
        m["compounding_err"] = round(abs((cum_ann - 1.0) * 100.0 - m["total_return"]), 6)
        results_summary[s] = m

    # 8. 保存 JSON 报告
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(results_summary, f, indent=2, ensure_ascii=False)
    print(f"[OK] 评估 JSON 报告已持久化: {OUT_JSON}")

    # 9. 打印对比表
    print("\n" + "=" * 108)
    print(f"{'策略方案 / Strategy Scheme':<36} | {'CAGR':<8} | {'Sharpe':<8} | {'MaxDD':<8} | {'Calmar':<8} | {'暴雷命中率':<10} | {'总手续费':<12}")
    print("-" * 108)
    for k, v in results_summary.items():
        print(f"{k:<36} | {v['cagr']:>6.2f}% | {v['sharpe']:>8.2f} | {v['max_dd']:>6.2f}% | {v['calmar']:>8.2f} | {v.get('bad_news_hit_rate', 0.0):>8.2f}% | {v['fees']/10000:>8.2f} 万元")
    print("=" * 108)
    print(f"\n>>> 全部完成！总耗时: {time.time()-t0:.1f} 秒.")


if __name__ == "__main__":
    main()

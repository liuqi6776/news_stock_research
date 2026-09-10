# -*- coding: utf-8 -*-
"""统一生产级单现金池微观真实执行账本 (Unified Production Single-Cash Ledger v2.1)

全面落实 2026-09-07 二次量化审计整改要求 (P1 缺陷闭环):
  1. 净订单与目标覆盖机制 (Net Orders & Target Overwrite - 消除重复累加 BUG):
     调仓日新目标直接覆盖旧目标，未成交待卖量严格赋值为 (held_shares - target_shares)，
     杜绝停牌/跌停多次调仓导致挂单量翻倍累加；若信号反转为买入 (Target >= Held)，
     即刻撤销旧 pending 卖单，杜绝先卖后买同一标的。
  2. 证券级共享日成交容量预算 (Shared Daily ADV Quota):
     维护 daily_executed_shares[c]，同日 pending 卖单与当期调仓卖单严格共享 10% ADV 额度。
  3. 缺失 ADV 严格硬拦截 (Strict Zero Liquidity Guard):
     当 ADV20 缺失或为 0 时，最大可成交量强制为 0，严禁默认全额放行。
  4. 先卖后买两阶段严格撮合与只卖不买强约束:
     阶段一平仓/减仓释放现金 -> 阶段二开仓/配资买入；分歧/退潮期强制限制目标股数 <= 当前持仓。
  5. 真实 A 股微观限制:
     北交所 ±30%、双创 ±20%、主板 ±10%、ST ±5%、真实 T+1 制度、100 股整手、千一印花税与佣金。
"""
import os
import sys
import math
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
EXP_DIR = os.path.join(ROOT, "research", "experiments", "exp_ens_t60_tv12")
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
if EXP_DIR not in sys.path:
    sys.path.insert(0, EXP_DIR)

# IM 期货与交易常量
IM_MULTIPLIER = 200.0          # 中证1000期货合约乘数 (元/点)
IM_MARGIN_RATIO = 0.15        # 初始保证金率 15%
IM_MAINT_MARGIN_RATIO = 0.12  # 维持保证金率 12%
IM_FEE_RATE = 0.00005         # 期货交易手续费 0.5 bps
STOCK_FEE_RATE = 0.0010       # 股票单边综合费率 10 bps (包含印花税/佣金/过户费)
ETF_FEE_RATE = 0.0003         # ETF 单边费率 3 bps
CASH_INTEREST_RATE = 0.020    # 闲置现金年化利息 2.0%


def format_trade_date(d):
    """规范化交易日输出格式：支持整数YYYYMMDD或字符串YYYY-MM-DD"""
    if d is None:
        return ""
    if isinstance(d, (int, np.integer)):
        return int(d)
    s = str(d).strip().replace("-", "")
    if s.isdigit():
        return int(s)
    return str(d).strip()


def get_price_limit_thresholds(code, is_st=False):
    """
    根据标的代码和ST状态确定微观涨跌停幅度：
    - 北交所 (8, 4, 92 开头): ±30%
    - 创业板 (30 开头), 科创板 (68 开头): ±20%
    - ST / *ST 股票: ±5%
    - 沪深主板 (60, 00 开头): ±10%
    """
    if str(code).startswith(("8", "4", "92")):
        return 0.298, -0.298
    if str(code).startswith(("30", "68")):
        return 0.198, -0.198
    if is_st:
        return 0.048, -0.048
    return 0.098, -0.098


def is_limit_up_code(code, open_p, preclose_p, is_st=False):
    """判定是否涨停封死（开盘即涨停）"""
    if not np.isfinite(open_p) or not np.isfinite(preclose_p) or preclose_p <= 0:
        return False
    up_thresh, _ = get_price_limit_thresholds(code, is_st=is_st)
    return (open_p / preclose_p - 1.0) >= up_thresh


def is_limit_down_code(code, open_p, preclose_p, is_st=False):
    """判定是否跌停封死（开盘即跌停）"""
    if not np.isfinite(open_p) or not np.isfinite(preclose_p) or preclose_p <= 0:
        return False
    _, down_thresh = get_price_limit_thresholds(code, is_st=is_st)
    return (open_p / preclose_p - 1.0) <= down_thresh


def is_limit_up(open_p, preclose_p, is_growth_or_cyb=False):
    """向后兼容旧版函数"""
    if not np.isfinite(open_p) or not np.isfinite(preclose_p) or preclose_p <= 0:
        return False
    thresh = 0.198 if is_growth_or_cyb else 0.098
    return (open_p / preclose_p - 1.0) >= thresh


def is_limit_down(open_p, preclose_p, is_growth_or_cyb=False):
    """向后兼容旧版函数"""
    if not np.isfinite(open_p) or not np.isfinite(preclose_p) or preclose_p <= 0:
        return False
    thresh = -0.198 if is_growth_or_cyb else -0.098
    return (open_p / preclose_p - 1.0) <= thresh


def get_adv20_shares(stock_vol_w, code, current_date, default_shares=0):
    """
    严格基于 [D-20, D-1] 历史成交量计算 20 日日均成交量（换算为真实股数）
    严禁包含交易日 D 本身！(零前瞻未来数据泄漏)
    Tushare vol 字段原始单位为手 (100股)，必须乘以 100。
    停牌日按 0 成交量计入 20 日均值 (窗口日历天数 20 天严格摊薄)。
    """
    if stock_vol_w is None or code not in stock_vol_w.columns or current_date not in stock_vol_w.index:
        return default_shares

    priors = [d for d in stock_vol_w.index if d < current_date]
    if len(priors) < 5:
        return default_shares

    sub_dates = priors[-20:]
    recent_vols = stock_vol_w.loc[sub_dates, code].fillna(0.0)
    vol_sum = recent_vols.sum()
    if not np.isfinite(vol_sum) or vol_sum <= 0:
        return default_shares

    mean_vol_lots = vol_sum / float(len(sub_dates))
    adv20_shares = float(mean_vol_lots) * 100.0
    return adv20_shares


def get_etf_adv20_shares(etf_vol_series, current_date, default_shares=0):
    """
    严格基于 [D-20, D-1] 历史成交量计算 ETF 20 日日均成交量（换算为真实股数）
    严禁包含交易日 D 本身！(零前瞻未来数据泄漏防范)
    A 股基金日行情成交量 (vol) 字段原始单位通常为手 (100份)，转换为股数必须乘以 100。
    """
    if etf_vol_series is None or len(etf_vol_series) == 0:
        return default_shares
    priors = [d for d in etf_vol_series.index if d < current_date]
    if len(priors) < 5:
        return default_shares
    sub_dates = priors[-20:]
    recent_vols = etf_vol_series.loc[sub_dates].fillna(0.0)
    vol_sum = recent_vols.sum()
    if not np.isfinite(vol_sum) or vol_sum <= 0:
        return default_shares
    mean_vol_lots = vol_sum / float(len(sub_dates))
    return float(mean_vol_lots) * 100.0


def compute_metrics(nav_series):
    """
    标准化绩效评估函数：
    - CAGR: 242 交易日复利年化
    - Sharpe: 基于日频超额收益率序列的标准年化夏普比率 (Rf=2.0%)
    - Vol: 日频收益率标准差年化
    - MaxDD: 历史高点最大回撤
    - Calmar: CAGR / |MaxDD|
    """
    s = nav_series.dropna()
    if len(s) < 10:
        return {}
    r = s.pct_change().dropna()
    n_days = len(r)
    cagr = (s.iloc[-1] / s.iloc[0]) ** (242.0 / max(n_days, 1)) - 1.0
    vol = float(r.std() * math.sqrt(242))
    rf = 0.02
    daily_rf = (1.0 + rf) ** (1.0 / 242.0) - 1.0
    excess_r = r - daily_rf
    excess_std = float(excess_r.std())
    sharpe = float(excess_r.mean() / excess_std * math.sqrt(242)) if excess_std > 1e-6 else 0.0
    dd = s / s.cummax() - 1.0
    max_dd = float(dd.min())
    calmar = float(cagr / abs(max_dd)) if abs(max_dd) > 1e-4 else 0.0
    tot = float(s.iloc[-1] / s.iloc[0]) - 1.0
    win_rate = float((r > 0).mean())
    return {
        "cagr": round(cagr * 100.0, 2),
        "sharpe": round(sharpe, 2),
        "vol": round(vol * 100.0, 2),
        "max_dd": round(max_dd * 100.0, 2),
        "calmar": round(calmar, 2),
        "total_return": round(tot * 100.0, 2),
        "win_rate": round(win_rate * 100.0, 2),
        "days": n_days
    }


def compute_annual_returns(nav_series):
    r"""
    连续复利跨年收益率计算：
    使用每日收益率进行年内连乘累积，确保：
    1. 不丢失跨年至首个交易日的收益；
    2. 全年收益严格满足 1 + R_year = \prod_{t \in year} (1 + r_t)；
    3. 全期连乘与总收益率严格等价。
    """
    s = nav_series.dropna()
    r = s.pct_change().fillna(0.0)
    df = pd.DataFrame({"nav": s, "ret": r})
    df["year"] = df.index.astype(int) // 10000
    annual = {}
    for yr, g in df.groupby("year"):
        compound_r = np.prod(1.0 + g["ret"].values) - 1.0
        annual[int(yr)] = round(float(compound_r * 100.0), 2)
    return annual


def select_with_clean_crowding_guard(
    scores_in, ind_map, ind_l1_map, crowded_codes,
    max_per_ind=4, max_per_ind_l1=8, top_n=40
):
    """
    严密的前瞻拥挤度过滤函数：初选与递补阶段均严格禁止命中拥挤度的股票进入！
    """
    scores_in = scores_in.dropna()
    sorted_codes = scores_in.sort_values(ascending=False)

    selected = []
    ind_count = {}
    l1_count = {}

    for code in sorted_codes.index:
        if crowded_codes is not None and code in crowded_codes:
            continue

        ind = ind_map.get(code, "Unknown")
        l1 = ind_l1_map.get(code, "Unknown")

        if ind_count.get(ind, 0) >= max_per_ind:
            continue
        if max_per_ind_l1 is not None and l1_count.get(l1, 0) >= max_per_ind_l1:
            continue

        selected.append(code)
        ind_count[ind] = ind_count.get(ind, 0) + 1
        if max_per_ind_l1 is not None:
            l1_count[l1] = l1_count.get(l1, 0) + 1

        if len(selected) >= top_n:
            break

    if len(selected) < top_n:
        for code in sorted_codes.index:
            if crowded_codes is not None and code in crowded_codes:
                continue
            if code not in selected:
                selected.append(code)
                if len(selected) >= top_n:
                    break

    return selected


class UnifiedProductionLedger:
    """
    生产级单一资金池现货与期货联合账户仿真器 (v2.4 审计整改版)
    - 落实 P0-1: ETF 成交价严禁 ffill/bfill，缺失开盘价或停牌时严格阻断并记录拒单；
    - 落实 P0-2: 加载 ETF 真实成交量并施加 10% ADV 限额、3 bps 佣金、最低 5 元手续费及 2 bps 滑点；
    - 落实 P0-4: 增加 record_initial_state(T0)，保留首日建仓费用与真实日内收益；
    - 增加 orders_log, fills_log, fees_log, blocked_orders_log, daily_holdings_log, daily_nav_log 全流程审计制品。
    """
    def __init__(
        self,
        initial_capital=2_200_000.0,
        fee_bps=10.0,
        etf_fee_bps=3.0,
        adv_cap_pct=0.10,
        etf_slippage_bps=2.0,
        min_etf_fee=5.0,
        cash_interest_rate=0.020,
        stock_slippage_bps=0.0
    ):
        self.initial_capital = float(initial_capital)
        self.stock_fee_rate = float(fee_bps) / 10000.0
        self.etf_fee_rate = float(etf_fee_bps) / 10000.0
        self.adv_cap_pct = float(adv_cap_pct)
        self.etf_slippage_bps = float(etf_slippage_bps)
        self.etf_slippage_rate = float(etf_slippage_bps) / 10000.0
        self.min_etf_fee = float(min_etf_fee)
        self.cash_interest_rate = float(cash_interest_rate)
        self.stock_slippage_bps = float(stock_slippage_bps)
        self.stock_slippage_rate = float(stock_slippage_bps) / 10000.0

        # 唯一真实总现金余额 (全部从 C0 出发)
        self.cash = float(initial_capital)

        # 股票与 ETF 持仓字典: code -> {shares, tradable_shares, locked_shares, last_px}
        self.stock_positions = {}
        self.etf_positions = {}

        # 活跃月度持仓篮子 (用于空仓后重新入场建仓)
        self.active_monthly_basket = []

        # 未完成挂单重试队列: code -> pending_shares (统一目标与实际状态机 P1-1)
        self.pending_sell_orders = {}
        self.pending_etf_sell_orders = {}
        self.pending_stock_buy_orders = {}
        self.pending_etf_buy_orders = {}

        # 证券级单日已成交股数记录 (确保同日多流程共享 ADV 限额)
        self.daily_executed_shares = {}
        self.last_execution_date = None

        # 期货持仓状态
        self.im_lots = 0
        self.im_prev_px = None

        # 累计统计与摩擦成本细分 (P1-2)
        self.total_stock_commission = 0.0
        self.total_etf_commission = 0.0
        self.total_futures_commission = 0.0
        self.total_slippage_cost = 0.0
        self.total_tax_cost = 0.0
        self.total_friction = 0.0
        self.total_trades = 0
        self.total_traded_value = 0.0
        self.selection_traded_value = 0.0
        self.timing_traded_value = 0.0
        self.limit_up_rejections = 0
        self.limit_down_locks = 0
        self.suspension_blocks = 0
        self.adv_zero_blocks = 0
        self.etf_missing_price_blocks = 0

        # 全套标准化审计追踪流水 (对应任务书第四节)
        self.orders_log = []
        self.fills_log = []
        self.fees_log = []
        self.blocked_orders_log = []
        self.daily_holdings_log = []
        self.daily_nav_log = []
        self.stale_price_age = {}

    def record_initial_state(self, date_0):
        """记录建仓前一日 (T0) 初始基准净值记录: NAV=1.0, 现金=初始本金 (P0-4)"""
        self.daily_nav_log.append({
            "trade_date": format_trade_date(date_0),
            "nav": 1.0,
            "total_equity": self.initial_capital,
            "stock_val": 0.0,
            "etf_val": 0.0,
            "cash": self.cash,
            "futures_margin": 0.0,
            "im_lots": 0
        })

    def unlock_t1_shares(self):
        """每日开盘前解锁 T+1 锁仓股数"""
        for c, h in self.stock_positions.items():
            h["tradable_shares"] += h["locked_shares"]
            h["locked_shares"] = 0
        for c, h in self.etf_positions.items():
            h["tradable_shares"] += h["locked_shares"]
            h["locked_shares"] = 0

    def _get_daily_adv_quota(self, stock_vol_w, code, current_date):
        """
        获取指定股票在当天的剩余可用 ADV 容量股数 (共享日度限额)
        若 ADV <= 0 或缺失数据，则返回 0 (严禁无流动性默认全额成交)
        """
        if self.last_execution_date != current_date:
            self.daily_executed_shares = {}
            self.last_execution_date = current_date

        adv20_sh = get_adv20_shares(stock_vol_w, code, current_date, default_shares=0)
        if adv20_sh <= 0:
            self.adv_zero_blocks += 1
            return 0

        total_quota = int(adv20_sh * self.adv_cap_pct)
        already_used = self.daily_executed_shares.get(code, 0)
        remaining = max(0, total_quota - already_used)
        return remaining

    def _get_daily_etf_adv_quota(self, etf_vol_dict, etf_code, current_date):
        """
        获取指定 ETF 在当天的剩余可用 ADV 容量股数 (P0-2 流动性平权)
        若未传入 etf_vol_dict，则向后兼容不加容量硬约束；若传入且 ADV<=0 则返回 0 (禁买)
        """
        if self.last_execution_date != current_date:
            self.daily_executed_shares = {}
            self.last_execution_date = current_date

        if etf_vol_dict is None or etf_code not in etf_vol_dict:
            return 10**10

        vol_s = etf_vol_dict.get(etf_code)
        adv20_sh = get_etf_adv20_shares(vol_s, current_date, default_shares=0)
        if adv20_sh <= 0:
            self.adv_zero_blocks += 1
            return 0

        total_quota = int(adv20_sh * self.adv_cap_pct)
        already_used = self.daily_executed_shares.get(etf_code, 0)
        return max(0, total_quota - already_used)

    def _record_executed_volume(self, code, current_date, shares):
        """记录指定标的在当天的已成交股数"""
        if self.last_execution_date != current_date:
            self.daily_executed_shares = {}
            self.last_execution_date = current_date
        self.daily_executed_shares[code] = self.daily_executed_shares.get(code, 0) + shares

    def process_daily_pending_orders(
        self,
        current_date,
        stock_open_w,
        stock_preclose_w,
        stock_vol_w,
        st_dict=None,
        etf_price_dict=None,
        etf_vol_dict=None
    ):
        """
        每日开盘重试执行积压的未成交订单 (Pending Orders Daily Retry - P1-1)
        采用严格净订单逻辑与共享日度 ADV 限额，覆盖股票与 ETF 的双向挂单。
        阶段一：卖出挂单变现 (ETF 卖单 -> 股票卖单)
        阶段二：买入挂单配置 (ETF 买单 -> 股票买单)
        """
        if self.last_execution_date != current_date:
            self.daily_executed_shares = {}
            self.last_execution_date = current_date

        # -------------------------------------------------------------------------
        # 1. ETF 未成交卖单重试 (Phase 1a: ETF Sells)
        # -------------------------------------------------------------------------
        if self.pending_etf_sell_orders and etf_price_dict:
            for etf_code in list(self.pending_etf_sell_orders.keys()):
                if etf_code not in self.etf_positions or self.etf_positions[etf_code]["shares"] <= 0:
                    self.pending_etf_sell_orders.pop(etf_code, None)
                    continue

                order_info = self.pending_etf_sell_orders[etf_code]
                req_sh = order_info["shares"] if isinstance(order_info, dict) else order_info
                order_reason = order_info.get("reason", "timing") if isinstance(order_info, dict) else "timing"

                h_etf = self.etf_positions[etf_code]
                pending_sh = min(req_sh, h_etf["shares"])
                if pending_sh < 100:
                    self.pending_etf_sell_orders.pop(etf_code, None)
                    continue

                s_df = etf_price_dict.get(etf_code)
                op = s_df.get(current_date, np.nan) if s_df is not None else np.nan
                if not (np.isfinite(op) and op > 0):
                    continue

                max_etf_adv = self._get_daily_etf_adv_quota(etf_vol_dict, etf_code, current_date)
                if max_etf_adv < 100:
                    continue

                sell_sh = min(h_etf["tradable_shares"], pending_sh, max_etf_adv)
                sell_sh = (sell_sh // 100) * 100

                if sell_sh >= 100:
                    fill_px = op * (1.0 - self.etf_slippage_bps / 10000.0)
                    proceeds = sell_sh * fill_px
                    raw_fee = proceeds * self.etf_fee_rate
                    fee = max(self.min_etf_fee, raw_fee)
                    slippage_cost = sell_sh * op * (self.etf_slippage_bps / 10000.0)

                    self.cash += (proceeds - fee)
                    self.total_etf_commission += fee
                    self.total_slippage_cost += slippage_cost
                    self.total_friction += (fee + slippage_cost)
                    self.total_trades += 1
                    self.total_traded_value += proceeds
                    if order_reason == "monthly":
                        self.selection_traded_value += proceeds
                    else:
                        self.timing_traded_value += proceeds

                    h_etf["shares"] -= sell_sh
                    h_etf["tradable_shares"] -= sell_sh
                    h_etf["last_px"] = op
                    self._record_executed_volume(etf_code, current_date, sell_sh)

                    self.fills_log.append({
                        "trade_date": format_trade_date(current_date),
                        "code": etf_code,
                        "side": "SELL",
                        "shares": int(sell_sh),
                        "price": float(fill_px),
                        "value": float(proceeds),
                        "fee": float(fee)
                    })
                    self.fees_log.append({
                        "trade_date": format_trade_date(current_date),
                        "code": etf_code,
                        "side": "SELL",
                        "asset_type": "ETF",
                        "commission": float(fee),
                        "slippage": float(slippage_cost),
                        "tax": 0.0,
                        "transfer_fee": 0.0,
                        "total_fee": float(fee + slippage_cost)
                    })

                    rem = pending_sh - sell_sh
                    if rem < 100:
                        self.pending_etf_sell_orders.pop(etf_code, None)
                    else:
                        self.pending_etf_sell_orders[etf_code] = {"shares": rem, "reason": order_reason}

                if h_etf["shares"] <= 0:
                    self.etf_positions.pop(etf_code, None)
                    self.pending_etf_sell_orders.pop(etf_code, None)

        # -------------------------------------------------------------------------
        # 2. 股票未成交卖单重试 (Phase 1b: Stock Sells)
        # -------------------------------------------------------------------------
        if self.pending_sell_orders:
            active_codes = list(self.pending_sell_orders.keys())
            for c in active_codes:
                if c not in self.stock_positions or self.stock_positions[c]["shares"] <= 0:
                    self.pending_sell_orders.pop(c, None)
                    continue

                order_info = self.pending_sell_orders[c]
                if isinstance(order_info, dict):
                    req_sh = order_info["shares"]
                    order_reason = order_info.get("reason", "timing")
                else:
                    req_sh = order_info
                    order_reason = "timing"

                h = self.stock_positions[c]
                pending_sh = min(req_sh, h["shares"])
                if pending_sh < 100:
                    self.pending_sell_orders.pop(c, None)
                    continue

                op = stock_open_w.at[current_date, c] if (c in stock_open_w.columns and current_date in stock_open_w.index) else np.nan
                pre_p = stock_preclose_w.at[current_date, c] if (c in stock_preclose_w.columns and current_date in stock_preclose_w.index) else np.nan

                # 停牌检查: 开盘价缺失或成交量为零，继续挂单等待
                is_suspended = (not (np.isfinite(op) and op > 0))
                if not is_suspended and stock_vol_w is not None and c in stock_vol_w.columns and current_date in stock_vol_w.index:
                    day_vol = stock_vol_w.at[current_date, c]
                    if np.isnan(day_vol) or day_vol <= 0:
                        is_suspended = True

                if is_suspended:
                    self.suspension_blocks += 1
                    continue

                # 跌停检查: 跌停封死无法卖出，保留在重试队列中
                is_st = False
                if st_dict is not None and c in st_dict:
                    for (s, e) in st_dict[c]:
                        if s <= current_date <= e:
                            is_st = True
                            break
                if is_limit_down_code(c, op, pre_p, is_st=is_st):
                    self.limit_down_locks += 1
                    continue

                # 共享 ADV 容量限制 (取 D 日之前 20 天，手转股)
                max_adv = self._get_daily_adv_quota(stock_vol_w, c, current_date)
                if max_adv < 100:
                    continue

                sell_sh = min(h["tradable_shares"], pending_sh, max_adv)
                sell_sh = (sell_sh // 100) * 100

                if sell_sh >= 100:
                    fill_px = op * (1.0 - self.stock_slippage_rate)
                    proceeds = sell_sh * fill_px
                    fee = proceeds * self.stock_fee_rate
                    slippage_cost = sell_sh * op * self.stock_slippage_rate
                    tax_cost = proceeds * 0.0005

                    self.cash += (proceeds - fee)
                    self.total_stock_commission += fee
                    self.total_slippage_cost += slippage_cost
                    self.total_tax_cost += tax_cost
                    self.total_friction += (fee + slippage_cost)
                    self.total_trades += 1
                    self.total_traded_value += proceeds

                    if order_reason == "monthly":
                        self.selection_traded_value += proceeds
                    else:
                        self.timing_traded_value += proceeds
                    h["shares"] -= sell_sh
                    h["tradable_shares"] -= sell_sh
                    h["last_px"] = op
                    self._record_executed_volume(c, current_date, sell_sh)

                    self.fills_log.append({
                        "trade_date": format_trade_date(current_date),
                        "code": c,
                        "side": "SELL",
                        "shares": int(sell_sh),
                        "price": float(fill_px),
                        "value": float(proceeds),
                        "fee": float(fee)
                    })
                    self.fees_log.append({
                        "trade_date": format_trade_date(current_date),
                        "code": c,
                        "side": "SELL",
                        "asset_type": "STOCK",
                        "commission": float(fee),
                        "slippage": float(slippage_cost),
                        "tax": float(tax_cost),
                        "transfer_fee": float(proceeds * 0.00002),
                        "total_fee": float(fee + slippage_cost)
                    })

                    rem = pending_sh - sell_sh
                    if rem < 100:
                        self.pending_sell_orders.pop(c, None)
                    else:
                        if isinstance(self.pending_sell_orders[c], dict):
                            self.pending_sell_orders[c]["shares"] = rem
                        else:
                            self.pending_sell_orders[c] = {"shares": rem, "reason": order_reason}

                if h["shares"] <= 0:
                    self.stock_positions.pop(c, None)
                    self.pending_sell_orders.pop(c, None)

        # -------------------------------------------------------------------------
        # 3. ETF 未成交买单重试 (Phase 2a: ETF Buys)
        # -------------------------------------------------------------------------
        if self.pending_etf_buy_orders and etf_price_dict:
            for etf_code in list(self.pending_etf_buy_orders.keys()):
                order_info = self.pending_etf_buy_orders[etf_code]
                req_sh = order_info["shares"] if isinstance(order_info, dict) else order_info
                order_reason = order_info.get("reason", "timing") if isinstance(order_info, dict) else "timing"

                if req_sh < 100:
                    self.pending_etf_buy_orders.pop(etf_code, None)
                    continue

                s_df = etf_price_dict.get(etf_code)
                op = s_df.get(current_date, np.nan) if s_df is not None else np.nan
                if not (np.isfinite(op) and op > 0):
                    continue

                max_etf_adv = self._get_daily_etf_adv_quota(etf_vol_dict, etf_code, current_date)
                if max_etf_adv < 100:
                    continue

                fill_px = op * (1.0 + self.etf_slippage_bps / 10000.0)
                eff_cost_per_sh = fill_px * (1.0 + self.etf_fee_rate)
                if self.cash > self.min_etf_fee:
                    max_aff_etf_sh = int((self.cash - self.min_etf_fee) // (eff_cost_per_sh * 100)) * 100
                else:
                    max_aff_etf_sh = 0

                buy_sh = min(req_sh, max_aff_etf_sh, max_etf_adv)
                buy_sh = (buy_sh // 100) * 100

                if buy_sh >= 100:
                    cost = buy_sh * fill_px
                    raw_fee = cost * self.etf_fee_rate
                    fee = max(self.min_etf_fee, raw_fee)
                    slippage_cost = buy_sh * op * (self.etf_slippage_bps / 10000.0)

                    if self.cash >= (cost + fee):
                        self.cash -= (cost + fee)
                        self.total_etf_commission += fee
                        self.total_slippage_cost += slippage_cost
                        self.total_friction += (fee + slippage_cost)
                        self.total_trades += 1
                        self.total_traded_value += cost
                        if order_reason == "monthly":
                            self.selection_traded_value += cost
                        else:
                            self.timing_traded_value += cost
                        self._record_executed_volume(etf_code, current_date, buy_sh)

                        if etf_code not in self.etf_positions:
                            self.etf_positions[etf_code] = {
                                "shares": buy_sh,
                                "tradable_shares": 0,
                                "locked_shares": buy_sh,
                                "last_px": op
                            }
                        else:
                            self.etf_positions[etf_code]["shares"] += buy_sh
                            self.etf_positions[etf_code]["locked_shares"] += buy_sh
                            self.etf_positions[etf_code]["last_px"] = op

                        self.fills_log.append({
                            "trade_date": format_trade_date(current_date),
                            "code": etf_code,
                            "side": "BUY",
                            "shares": int(buy_sh),
                            "price": float(fill_px),
                            "value": float(cost),
                            "fee": float(fee)
                        })
                        self.fees_log.append({
                            "trade_date": format_trade_date(current_date),
                            "code": etf_code,
                            "side": "BUY",
                            "asset_type": "ETF",
                            "commission": float(fee),
                            "slippage": float(slippage_cost),
                            "tax": 0.0,
                            "transfer_fee": 0.0,
                            "total_fee": float(fee + slippage_cost)
                        })

                        rem = req_sh - buy_sh
                        if rem < 100:
                            self.pending_etf_buy_orders.pop(etf_code, None)
                        else:
                            self.pending_etf_buy_orders[etf_code] = {"shares": rem, "reason": order_reason}

        # -------------------------------------------------------------------------
        # 4. 股票未成交买单重试 (Phase 2b: Stock Buys)
        # -------------------------------------------------------------------------
        if self.pending_stock_buy_orders:
            for c in list(self.pending_stock_buy_orders.keys()):
                order_info = self.pending_stock_buy_orders[c]
                req_sh = order_info["shares"] if isinstance(order_info, dict) else order_info
                order_reason = order_info.get("reason", "timing") if isinstance(order_info, dict) else "timing"

                if req_sh < 100:
                    self.pending_stock_buy_orders.pop(c, None)
                    continue

                op = stock_open_w.at[current_date, c] if (c in stock_open_w.columns and current_date in stock_open_w.index) else np.nan
                pre_p = stock_preclose_w.at[current_date, c] if (c in stock_preclose_w.columns and current_date in stock_preclose_w.index) else np.nan

                # 停牌与涨停检查
                is_suspended = (not (np.isfinite(op) and op > 0))
                if not is_suspended and stock_vol_w is not None and c in stock_vol_w.columns and current_date in stock_vol_w.index:
                    day_vol = stock_vol_w.at[current_date, c]
                    if np.isnan(day_vol) or day_vol <= 0:
                        is_suspended = True
                if is_suspended:
                    continue

                is_st = False
                if st_dict is not None and c in st_dict:
                    for (s, e) in st_dict[c]:
                        if s <= current_date <= e:
                            is_st = True
                            break
                if is_limit_up_code(c, op, pre_p, is_st=is_st):
                    continue

                max_adv = self._get_daily_adv_quota(stock_vol_w, c, current_date)
                if max_adv < 100:
                    continue

                fill_px = op * (1.0 + self.stock_slippage_rate)
                eff_cost_per_sh = fill_px * (1.0 + self.stock_fee_rate)
                max_aff_sh = int(self.cash // (eff_cost_per_sh * 100)) * 100
                buy_sh = min(req_sh, max_aff_sh, max_adv)
                buy_sh = (buy_sh // 100) * 100

                if buy_sh >= 100:
                    cost = buy_sh * fill_px
                    fee = cost * self.stock_fee_rate
                    slippage_cost = buy_sh * op * self.stock_slippage_rate

                    if self.cash >= (cost + fee):
                        self.cash -= (cost + fee)
                        self.total_stock_commission += fee
                        self.total_slippage_cost += slippage_cost
                        self.total_friction += (fee + slippage_cost)
                        self.total_trades += 1
                        self.total_traded_value += cost
                        if order_reason == "monthly":
                            self.selection_traded_value += cost
                        else:
                            self.timing_traded_value += cost
                        self._record_executed_volume(c, current_date, buy_sh)

                        if c not in self.stock_positions:
                            self.stock_positions[c] = {
                                "shares": buy_sh,
                                "tradable_shares": 0,
                                "locked_shares": buy_sh,
                                "last_px": op
                            }
                        else:
                            self.stock_positions[c]["shares"] += buy_sh
                            self.stock_positions[c]["locked_shares"] += buy_sh
                            self.stock_positions[c]["last_px"] = op

                        self.fills_log.append({
                            "trade_date": format_trade_date(current_date),
                            "code": c,
                            "side": "BUY",
                            "shares": int(buy_sh),
                            "price": float(fill_px),
                            "value": float(cost),
                            "fee": float(fee)
                        })
                        self.fees_log.append({
                            "trade_date": format_trade_date(current_date),
                            "code": c,
                            "side": "BUY",
                            "asset_type": "STOCK",
                            "commission": float(fee),
                            "slippage": float(slippage_cost),
                            "tax": 0.0,
                            "transfer_fee": float(cost * 0.00002),
                            "total_fee": float(fee + slippage_cost)
                        })

                        rem = req_sh - buy_sh
                        if rem < 100:
                            self.pending_stock_buy_orders.pop(c, None)
                        else:
                            self.pending_stock_buy_orders[c] = {"shares": rem, "reason": order_reason}

    def compute_equity(self, current_date, stock_close_w, etf_close_dict, im_close_px=None):
        """
        计算盘后统一账户真实总权益与可用现金
        """
        # 1. 股票总市值
        stock_market_val = 0.0
        for c, h in self.stock_positions.items():
            cl = stock_close_w.at[current_date, c] if (c in stock_close_w.columns and current_date in stock_close_w.index) else np.nan
            px = cl if (np.isfinite(cl) and cl > 0) else h["last_px"]
            h["last_px"] = px
            stock_market_val += h["shares"] * px

        # 2. ETF 总市值
        etf_market_val = 0.0
        for c, h in self.etf_positions.items():
            s_df = etf_close_dict.get(c) if etf_close_dict else None
            cl = s_df.get(current_date, np.nan) if s_df is not None else np.nan
            if np.isfinite(cl) and cl > 0:
                px = cl
                self.stale_price_age[c] = 0
            else:
                px = h["last_px"]
                self.stale_price_age[c] = self.stale_price_age.get(c, 0) + 1
            h["last_px"] = px
            etf_market_val += h["shares"] * px

        # 3. 期货逐日盯市盈亏与保证金
        futures_floating_pnl = 0.0
        futures_margin = 0.0
        if self.im_lots > 0 and im_close_px is not None and np.isfinite(im_close_px) and im_close_px > 0:
            contract_val = im_close_px * IM_MULTIPLIER
            futures_margin = self.im_lots * contract_val * IM_MARGIN_RATIO
            if self.im_prev_px is not None and np.isfinite(self.im_prev_px) and self.im_prev_px > 0:
                futures_floating_pnl = -self.im_lots * (im_close_px - self.im_prev_px) * IM_MULTIPLIER
            self.cash += futures_floating_pnl
            self.im_prev_px = im_close_px

        # 4. 闲置现金收益
        daily_interest = max(self.cash, 0.0) * (self.cash_interest_rate / 242.0)
        self.cash += daily_interest

        total_nav = stock_market_val + etf_market_val + self.cash
        eq_record = {
            "trade_date": format_trade_date(current_date),
            "nav": total_nav / self.initial_capital,
            "total_equity": total_nav,
            "stock_val": stock_market_val,
            "etf_val": etf_market_val,
            "cash": self.cash,
            "futures_margin": futures_margin,
            "im_lots": self.im_lots
        }
        self.daily_nav_log.append(eq_record)

        # 记录持仓明细 (每日实际持仓)
        for c, h in self.stock_positions.items():
            if h["shares"] > 0:
                mv = h["shares"] * h["last_px"]
                self.daily_holdings_log.append({
                    "trade_date": format_trade_date(current_date), "code": c, "asset_type": "STOCK",
                    "shares": h["shares"], "tradable_shares": h["tradable_shares"],
                    "price": h["last_px"], "market_val": mv, "weight": mv / max(total_nav, 1e-6)
                })
        for c, h in self.etf_positions.items():
            if h["shares"] > 0:
                mv = h["shares"] * h["last_px"]
                self.daily_holdings_log.append({
                    "trade_date": format_trade_date(current_date), "code": c, "asset_type": "ETF",
                    "shares": h["shares"], "tradable_shares": h["tradable_shares"],
                    "price": h["last_px"], "market_val": mv, "weight": mv / max(total_nav, 1e-6)
                })
        return eq_record

    def _execute_etf_sells(
        self,
        current_date,
        target_etf_shares_map,
        etf_price_dict,
        rebalance_reason="timing",
        etf_vol_dict=None
    ):
        if not etf_price_dict:
            return
        all_current_etfs = list(self.etf_positions.keys())
        for etf_code in all_current_etfs:
            h_etf = self.etf_positions.get(etf_code)
            if not h_etf:
                continue
            target_etf_sh = target_etf_shares_map.get(etf_code, 0)
            if target_etf_sh >= h_etf["shares"]:
                self.pending_etf_sell_orders.pop(etf_code, None)
                continue

            excess_etf_sh = h_etf["shares"] - target_etf_sh
            s_df = etf_price_dict.get(etf_code)
            op = s_df.get(current_date, np.nan) if s_df is not None else np.nan

            # P0-1: 严格校验开盘执行价，若缺失/非正数则拒绝执行并记录阻塞日志，绝不使用估值价/前收价/未来价填补
            if not (np.isfinite(op) and op > 0):
                self.blocked_orders_log.append({
                    "trade_date": format_trade_date(current_date),
                    "code": etf_code,
                    "side": "SELL",
                    "order_shares": excess_etf_sh,
                    "reason": "missing_or_invalid_open_price"
                })
                continue

            max_etf_adv = self._get_daily_etf_adv_quota(etf_vol_dict, etf_code, current_date)
            if max_etf_adv < 100:
                self.pending_etf_sell_orders[etf_code] = {"shares": excess_etf_sh, "reason": rebalance_reason}
                self.blocked_orders_log.append({
                    "trade_date": format_trade_date(current_date),
                    "code": etf_code,
                    "side": "SELL",
                    "order_shares": excess_etf_sh,
                    "reason": "etf_adv_exhausted_or_zero"
                })
                continue

            sell_etf_sh = min(h_etf["tradable_shares"], excess_etf_sh, max_etf_adv)
            sell_etf_sh = (sell_etf_sh // 100) * 100

            self.orders_log.append({
                "trade_date": format_trade_date(current_date),
                "code": etf_code,
                "asset_type": "ETF",
                "side": "SELL",
                "target_shares": target_etf_sh,
                "order_shares": excess_etf_sh,
                "reason": rebalance_reason
            })

            if sell_etf_sh >= 100:
                fill_px = op * (1.0 - self.etf_slippage_bps / 10000.0)
                proceeds = sell_etf_sh * fill_px
                raw_fee = proceeds * self.etf_fee_rate
                fee = max(self.min_etf_fee, raw_fee)
                slippage_cost = sell_etf_sh * op * (self.etf_slippage_bps / 10000.0)

                self.cash += (proceeds - fee)
                self.total_etf_commission += fee
                self.total_slippage_cost += slippage_cost
                self.total_friction += (fee + slippage_cost)
                self.total_trades += 1
                self.total_traded_value += proceeds
                if rebalance_reason == "monthly":
                    self.selection_traded_value += proceeds
                else:
                    self.timing_traded_value += proceeds
                h_etf["shares"] -= sell_etf_sh
                h_etf["tradable_shares"] -= sell_etf_sh
                h_etf["last_px"] = op
                self._record_executed_volume(etf_code, current_date, sell_etf_sh)

                self.fills_log.append({
                    "trade_date": format_trade_date(current_date),
                    "code": etf_code,
                    "side": "SELL",
                    "shares": sell_etf_sh,
                    "price": fill_px,
                    "value": proceeds,
                    "fee": fee
                })
                self.fees_log.append({
                    "trade_date": format_trade_date(current_date),
                    "code": etf_code,
                    "side": "SELL",
                    "asset_type": "ETF",
                    "commission": float(fee),
                    "slippage": float(slippage_cost),
                    "tax": 0.0,
                    "transfer_fee": 0.0,
                    "total_fee": float(fee + slippage_cost)
                })

            remaining_excess = h_etf["shares"] - target_etf_sh
            if remaining_excess >= 100:
                self.pending_etf_sell_orders[etf_code] = {"shares": remaining_excess, "reason": rebalance_reason}
            else:
                self.pending_etf_sell_orders.pop(etf_code, None)

            if h_etf["shares"] <= 0:
                self.etf_positions.pop(etf_code, None)
                self.pending_etf_sell_orders.pop(etf_code, None)

    def _execute_etf_buys(
        self,
        current_date,
        etf_targets,
        target_etf_shares_map,
        etf_price_dict,
        rebalance_reason="timing",
        etf_vol_dict=None
    ):
        if not etf_targets or not etf_price_dict:
            return
        for etf_code, tgt_pct in etf_targets.items():
            target_etf_sh = target_etf_shares_map.get(etf_code, 0)
            curr_etf_sh = self.etf_positions.get(etf_code, {}).get("shares", 0)

            if target_etf_sh <= curr_etf_sh:
                self.pending_etf_buy_orders.pop(etf_code, None)
                continue

            needed_etf_sh = target_etf_sh - curr_etf_sh
            s_df = etf_price_dict.get(etf_code)
            op = s_df.get(current_date, np.nan) if s_df is not None else np.nan

            # P0-1: 缺失开盘价严格拦截
            if not (np.isfinite(op) and op > 0):
                self.blocked_orders_log.append({
                    "trade_date": format_trade_date(current_date),
                    "code": etf_code,
                    "side": "BUY",
                    "order_shares": needed_etf_sh,
                    "reason": "missing_or_invalid_open_price"
                })
                continue

            max_etf_adv = self._get_daily_etf_adv_quota(etf_vol_dict, etf_code, current_date)
            if max_etf_adv < 100:
                self.pending_etf_buy_orders[etf_code] = {"shares": needed_etf_sh, "reason": rebalance_reason}
                self.blocked_orders_log.append({
                    "trade_date": format_trade_date(current_date),
                    "code": etf_code,
                    "side": "BUY",
                    "order_shares": needed_etf_sh,
                    "reason": "etf_adv_exhausted_or_zero"
                })
                continue

            self.orders_log.append({
                "trade_date": format_trade_date(current_date),
                "code": etf_code,
                "asset_type": "ETF",
                "side": "BUY",
                "target_shares": target_etf_sh,
                "order_shares": needed_etf_sh,
                "reason": rebalance_reason
            })

            fill_px = op * (1.0 + self.etf_slippage_bps / 10000.0)
            eff_cost_per_sh = fill_px * (1.0 + self.etf_fee_rate)
            if self.cash > self.min_etf_fee:
                max_aff_etf_sh = int((self.cash - self.min_etf_fee) // (eff_cost_per_sh * 100)) * 100
            else:
                max_aff_etf_sh = 0

            buy_etf_sh = min(needed_etf_sh, max_aff_etf_sh, max_etf_adv)
            buy_etf_sh = (buy_etf_sh // 100) * 100

            if buy_etf_sh >= 100:
                cost = buy_etf_sh * fill_px
                raw_fee = cost * self.etf_fee_rate
                fee = max(self.min_etf_fee, raw_fee)
                slippage_cost = buy_etf_sh * op * (self.etf_slippage_bps / 10000.0)

                if self.cash >= (cost + fee):
                    self.cash -= (cost + fee)
                    self.total_etf_commission += fee
                    self.total_slippage_cost += slippage_cost
                    self.total_friction += (fee + slippage_cost)
                    self.total_trades += 1
                    self.total_traded_value += cost
                    if rebalance_reason == "monthly":
                        self.selection_traded_value += cost
                    else:
                        self.timing_traded_value += cost
                    self._record_executed_volume(etf_code, current_date, buy_etf_sh)

                    if etf_code not in self.etf_positions:
                        self.etf_positions[etf_code] = {
                            "shares": buy_etf_sh,
                            "tradable_shares": 0,
                            "locked_shares": buy_etf_sh,
                            "last_px": op
                        }
                    else:
                        self.etf_positions[etf_code]["shares"] += buy_etf_sh
                        self.etf_positions[etf_code]["locked_shares"] += buy_etf_sh
                        self.etf_positions[etf_code]["last_px"] = op

                    self.fills_log.append({
                        "trade_date": format_trade_date(current_date),
                        "code": etf_code,
                        "side": "BUY",
                        "shares": buy_etf_sh,
                        "price": fill_px,
                        "value": cost,
                        "fee": fee
                    })
                    self.fees_log.append({
                        "trade_date": format_trade_date(current_date),
                        "code": etf_code,
                        "side": "BUY",
                        "asset_type": "ETF",
                        "commission": float(fee),
                        "slippage": float(slippage_cost),
                        "tax": 0.0,
                        "transfer_fee": 0.0,
                        "total_fee": float(fee + slippage_cost)
                    })

            curr_etf_sh_after = self.etf_positions.get(etf_code, {}).get("shares", 0)
            remaining_needed = target_etf_sh - curr_etf_sh_after
            if remaining_needed >= 100:
                self.pending_etf_buy_orders[etf_code] = {"shares": remaining_needed, "reason": rebalance_reason}
            else:
                self.pending_etf_buy_orders.pop(etf_code, None)

    def execute_rebalance(
        self,
        current_date,
        target_stock_codes,
        target_stock_pct,
        stock_open_w,
        stock_preclose_w,
        stock_vol_w,
        etf_targets,
        etf_price_dict,
        allow_buy=True,
        st_dict=None,
        im_hedge_beta=0.0,
        im_price=None,
        rebalance_reason="timing",
        etf_vol_dict=None
    ):
        """
        调仓日微观撮合执行 (v2.1 修复版):
        - 移除顶部预先执行 pending 导致的目标冲突；
        - 新目标直接覆盖旧目标，消除重复累加；
        - 信号反转直接撤销旧 pending 卖单；
        - 严格共享日度 10% ADV 额度与缺失 ADV 零成交拦截；
        - 两阶段先卖后买，现金完全回流后开仓。
        """
        if self.last_execution_date != current_date:
            self.daily_executed_shares = {}
            self.last_execution_date = current_date

        # 1. 计算开盘当前总资产
        open_stock_val = 0.0
        for c, h in self.stock_positions.items():
            op = stock_open_w.at[current_date, c] if (c in stock_open_w.columns and current_date in stock_open_w.index) else np.nan
            px = op if (np.isfinite(op) and op > 0) else h["last_px"]
            open_stock_val += h["shares"] * px

        open_etf_val = 0.0
        for c, h in self.etf_positions.items():
            s_df = etf_price_dict.get(c)
            op = s_df.get(current_date, np.nan) if s_df is not None else np.nan
            px = op if (np.isfinite(op) and op > 0) else h["last_px"]
            open_etf_val += h["shares"] * px

        total_open_equity = open_stock_val + open_etf_val + self.cash
        target_total_stock_val = total_open_equity * target_stock_pct

        # 2. 计算每只目标股票的目标股数
        if target_stock_codes:
            self.active_monthly_basket = list(target_stock_codes)

        target_shares_map = {}
        if len(target_stock_codes) > 0 and target_total_stock_val > 0:
            per_stock_target_val = target_total_stock_val / len(target_stock_codes)
            for c in target_stock_codes:
                op = stock_open_w.at[current_date, c] if (c in stock_open_w.columns and current_date in stock_open_w.index) else np.nan
                pre_p = stock_preclose_w.at[current_date, c] if (c in stock_preclose_w.columns and current_date in stock_preclose_w.index) else np.nan
                px = op if (np.isfinite(op) and op > 0) else pre_p
                if np.isfinite(px) and px > 0:
                    raw_sh = int((per_stock_target_val / px) // 100) * 100
                    target_shares_map[c] = raw_sh
                else:
                    target_shares_map[c] = 0

        # 【核心落实】：只卖不买强约束 (Strict Sell-Only)
        # 若 allow_buy=False，任何股票的目标股数严禁超过现有持仓股数；未持仓股票目标股数一律为 0
        if not allow_buy:
            for c in list(target_shares_map.keys()):
                curr_sh = self.stock_positions.get(c, {}).get("shares", 0)
                target_shares_map[c] = min(target_shares_map[c], curr_sh)

        # 3. 确定每只目标 ETF 的目标股数
        target_etf_shares_map = {}
        if etf_targets is not None and len(etf_targets) > 0:
            for etf_code, tgt_pct in etf_targets.items():
                s_df = etf_price_dict.get(etf_code)
                op = s_df.get(current_date, np.nan) if s_df is not None else np.nan
                if np.isfinite(op) and op > 0:
                    target_val = total_open_equity * tgt_pct
                    target_etf_shares_map[etf_code] = int((target_val / op) // 100) * 100
                else:
                    target_etf_shares_map[etf_code] = 0
                    if tgt_pct > 0:
                        self.blocked_orders_log.append({
                            "trade_date": format_trade_date(current_date),
                            "code": etf_code,
                            "side": "BUY",
                            "order_shares": 0,
                            "reason": "missing_or_invalid_open_price"
                        })

        # =========================================================================
        # 第一阶段：全面卖出变现 (PHASE 1: SELLS & TRIMS - CASH INFLOW)
        # =========================================================================

        # 3.1 股票超额减仓与清仓卖出 (净订单与目标覆盖机制)
        all_current_stocks = list(self.stock_positions.keys())
        for c in all_current_stocks:
            h = self.stock_positions[c]
            target_sh = target_shares_map.get(c, 0)

            # 信号反转与新目标覆盖：若当前目标 >= 持有股数，撤销旧 pending 卖单，不在此阶段卖出
            if target_sh >= h["shares"]:
                self.pending_sell_orders.pop(c, None)
                continue

            excess_sh = h["shares"] - target_sh
            self.orders_log.append({
                "trade_date": format_trade_date(current_date),
                "code": c,
                "asset_type": "STOCK",
                "side": "SELL",
                "target_shares": int(target_sh),
                "order_shares": int(excess_sh),
                "reason": rebalance_reason
            })
            op = stock_open_w.at[current_date, c] if (c in stock_open_w.columns and current_date in stock_open_w.index) else np.nan
            pre_p = stock_preclose_w.at[current_date, c] if (c in stock_preclose_w.columns and current_date in stock_preclose_w.index) else np.nan

            # 停牌检查
            is_suspended = (not (np.isfinite(op) and op > 0))
            if not is_suspended and stock_vol_w is not None and c in stock_vol_w.columns and current_date in stock_vol_w.index:
                day_vol = stock_vol_w.at[current_date, c]
                if np.isnan(day_vol) or day_vol <= 0:
                    is_suspended = True

            if is_suspended:
                self.suspension_blocks += 1
                self.blocked_orders_log.append({
                    "trade_date": format_trade_date(current_date),
                    "code": c,
                    "side": "SELL",
                    "order_shares": int(excess_sh),
                    "reason": "suspended"
                })
                # 赋值而非累加，严格防止多次调仓挂单翻倍，并记录触发原因
                self.pending_sell_orders[c] = {"shares": excess_sh, "reason": rebalance_reason}
                continue

            # 跌停检查
            is_st = False
            if st_dict is not None and c in st_dict:
                for (s, e) in st_dict[c]:
                    if s <= current_date <= e:
                        is_st = True
                        break
            if is_limit_down_code(c, op, pre_p, is_st=is_st):
                self.limit_down_locks += 1
                self.blocked_orders_log.append({
                    "trade_date": format_trade_date(current_date),
                    "code": c,
                    "side": "SELL",
                    "order_shares": int(excess_sh),
                    "reason": "limit_down"
                })
                self.pending_sell_orders[c] = {"shares": excess_sh, "reason": rebalance_reason}
                continue

            # 共享 ADV 容量约束 (同日两阶段共享，无数据返回0)
            max_adv = self._get_daily_adv_quota(stock_vol_w, c, current_date)
            if max_adv < 100:
                self.pending_sell_orders[c] = {"shares": excess_sh, "reason": rebalance_reason}
                continue

            sell_shares = min(h["tradable_shares"], excess_sh, max_adv)
            sell_shares = (sell_shares // 100) * 100

            if sell_shares >= 100:
                fill_px = op * (1.0 - self.stock_slippage_rate)
                proceeds = sell_shares * fill_px
                fee = proceeds * self.stock_fee_rate
                slippage_cost = sell_shares * op * self.stock_slippage_rate
                tax_cost = proceeds * 0.0005
                self.cash += (proceeds - fee)
                self.total_stock_commission += fee
                self.total_slippage_cost += slippage_cost
                self.total_tax_cost += tax_cost
                self.total_friction += (fee + slippage_cost)
                self.total_trades += 1
                self.total_traded_value += proceeds
                if rebalance_reason == "monthly":
                    self.selection_traded_value += proceeds
                else:
                    self.timing_traded_value += proceeds
                h["shares"] -= sell_shares
                h["tradable_shares"] -= sell_shares
                h["last_px"] = op
                self._record_executed_volume(c, current_date, sell_shares)

                self.fills_log.append({
                    "trade_date": format_trade_date(current_date),
                    "code": c,
                    "side": "SELL",
                    "shares": int(sell_shares),
                    "price": float(fill_px),
                    "value": float(proceeds),
                    "fee": float(fee)
                })
                self.fees_log.append({
                    "trade_date": format_trade_date(current_date),
                    "code": c,
                    "side": "SELL",
                    "asset_type": "STOCK",
                    "commission": float(fee),
                    "slippage": float(slippage_cost),
                    "tax": float(tax_cost),
                    "transfer_fee": float(proceeds * 0.00002),
                    "total_fee": float(fee + slippage_cost)
                })

            # 更新未成交卖单队列 (严格赋值为当前持股与目标股数的缺口，继承当前调仓原因)
            remaining_excess = h["shares"] - target_sh
            if remaining_excess >= 100:
                self.pending_sell_orders[c] = {"shares": remaining_excess, "reason": rebalance_reason}
            else:
                self.pending_sell_orders.pop(c, None)

            if h["shares"] <= 0:
                self.stock_positions.pop(c, None)
                self.pending_sell_orders.pop(c, None)

        # 3.2 ETF 减仓卖出
        self._execute_etf_sells(
            current_date=current_date,
            target_etf_shares_map=target_etf_shares_map,
            etf_price_dict=etf_price_dict,
            rebalance_reason=rebalance_reason,
            etf_vol_dict=etf_vol_dict
        )

        # =========================================================================
        # 第二阶段：买入与资金配置 (PHASE 2: BUYS & ALLOCATION - CASH OUTFLOW)
        # =========================================================================

        # 4.1 股票加仓与买入流程
        if allow_buy:
            for c in target_stock_codes:
                target_sh = target_shares_map.get(c, 0)
                existing_sh = self.stock_positions.get(c, {}).get("shares", 0)

                if target_sh > existing_sh:
                    needed_sh = target_sh - existing_sh
                    self.orders_log.append({
                        "trade_date": format_trade_date(current_date),
                        "code": c,
                        "asset_type": "STOCK",
                        "side": "BUY",
                        "target_shares": int(target_sh),
                        "order_shares": int(needed_sh),
                        "reason": rebalance_reason
                    })
                    op = stock_open_w.at[current_date, c] if (c in stock_open_w.columns and current_date in stock_open_w.index) else np.nan
                    pre_p = stock_preclose_w.at[current_date, c] if (c in stock_preclose_w.columns and current_date in stock_preclose_w.index) else np.nan

                    # 停牌检查
                    is_suspended = (not (np.isfinite(op) and op > 0))
                    if not is_suspended and stock_vol_w is not None and c in stock_vol_w.columns and current_date in stock_vol_w.index:
                        day_vol = stock_vol_w.at[current_date, c]
                        if np.isnan(day_vol) or day_vol <= 0:
                            is_suspended = True

                    if is_suspended:
                        self.suspension_blocks += 1
                        self.blocked_orders_log.append({
                            "trade_date": format_trade_date(current_date),
                            "code": c,
                            "side": "BUY",
                            "order_shares": int(needed_sh),
                            "reason": "suspended"
                        })
                        continue

                    # 涨停检查: 涨停封死禁止买入
                    is_st = False
                    if st_dict is not None and c in st_dict:
                        for (s, e) in st_dict[c]:
                            if s <= current_date <= e:
                                 is_st = True
                                 break
                    if is_limit_up_code(c, op, pre_p, is_st=is_st):
                        self.limit_up_rejections += 1
                        self.blocked_orders_log.append({
                            "trade_date": format_trade_date(current_date),
                            "code": c,
                            "side": "BUY",
                            "order_shares": int(needed_sh),
                            "reason": "limit_up"
                        })
                        continue

                    # 共享 ADV 容量约束 (无数据返回 0 严格禁买)
                    max_adv = self._get_daily_adv_quota(stock_vol_w, c, current_date)
                    if max_adv < 100:
                        self.pending_stock_buy_orders[c] = {"shares": needed_sh, "reason": rebalance_reason}
                        continue

                    needed_sh = target_sh - existing_sh
                    fill_px = op * (1.0 + self.stock_slippage_rate)
                    eff_cost_per_sh = fill_px * (1.0 + self.stock_fee_rate)
                    max_aff_sh = int(self.cash // (eff_cost_per_sh * 100)) * 100
                    buy_shares = min(needed_sh, max_aff_sh, max_adv)
                    buy_shares = (buy_shares // 100) * 100

                    if buy_shares >= 100:
                        cost = buy_shares * fill_px
                        fee = cost * self.stock_fee_rate
                        slippage_cost = buy_shares * op * self.stock_slippage_rate
                        self.cash -= (cost + fee)
                        self.total_stock_commission += fee
                        self.total_slippage_cost += slippage_cost
                        self.total_friction += (fee + slippage_cost)
                        self.total_trades += 1
                        self.total_traded_value += cost
                        if rebalance_reason == "monthly":
                            self.selection_traded_value += cost
                        else:
                            self.timing_traded_value += cost
                        self._record_executed_volume(c, current_date, buy_shares)

                        self.fills_log.append({
                            "trade_date": format_trade_date(current_date),
                            "code": c,
                            "side": "BUY",
                            "shares": int(buy_shares),
                            "price": float(fill_px),
                            "value": float(cost),
                            "fee": float(fee)
                        })
                        self.fees_log.append({
                            "trade_date": format_trade_date(current_date),
                            "code": c,
                            "side": "BUY",
                            "asset_type": "STOCK",
                            "commission": float(fee),
                            "slippage": float(slippage_cost),
                            "tax": 0.0,
                            "transfer_fee": float(cost * 0.00002),
                            "total_fee": float(fee + slippage_cost)
                        })

                        if c not in self.stock_positions:
                            self.stock_positions[c] = {
                                "shares": buy_shares,
                                "tradable_shares": 0,
                                "locked_shares": buy_shares,
                                "last_px": op
                            }
                        else:
                            self.stock_positions[c]["shares"] += buy_shares
                            self.stock_positions[c]["locked_shares"] += buy_shares
                            self.stock_positions[c]["last_px"] = op

                    curr_sh_after = self.stock_positions.get(c, {}).get("shares", 0)
                    remaining_needed = target_sh - curr_sh_after
                    if remaining_needed >= 100:
                        self.pending_stock_buy_orders[c] = {"shares": remaining_needed, "reason": rebalance_reason}
                    else:
                        self.pending_stock_buy_orders.pop(c, None)
                else:
                    self.pending_stock_buy_orders.pop(c, None)

        # 4.2 ETF 买入流程
        self._execute_etf_buys(
            current_date=current_date,
            etf_targets=etf_targets,
            target_etf_shares_map=target_etf_shares_map,
            etf_price_dict=etf_price_dict,
            rebalance_reason=rebalance_reason,
            etf_vol_dict=etf_vol_dict
        )

        # =========================================================================
        # 第三阶段：IM 期货对冲手数调整 (PHASE 3: FUTURES HEDGING ADJUSTMENT)
        # =========================================================================
        if im_hedge_beta > 0 and im_price is not None and np.isfinite(im_price) and im_price > 0:
            contract_val = im_price * IM_MULTIPLIER
            target_hedge_notional = total_open_equity * target_stock_pct * im_hedge_beta
            target_lots = int(round(target_hedge_notional / contract_val))

            if target_lots != self.im_lots:
                delta_lots = abs(target_lots - self.im_lots)
                trade_fee = delta_lots * contract_val * IM_FEE_RATE
                self.cash -= trade_fee
                self.total_futures_commission += trade_fee
                self.im_lots = target_lots
            self.im_prev_px = im_price
        else:
            self.im_lots = 0
            self.im_prev_px = im_price

    def scale_stock_exposure(
        self,
        current_date,
        target_stock_pct,
        stock_open_w,
        stock_preclose_w,
        stock_vol_w,
        etf_targets=None,
        etf_price_dict=None,
        allow_buy=True,
        st_dict=None,
        rebalance_reason="timing",
        etf_vol_dict=None
    ):
        """
        审计整改(v2.3): 已有股票篮子等比例缩放与空仓重入 (Proportional Basket Scaling & Re-entry)
        - 当选股名单未变、仅需按SCS信号调整总股票仓位时，对已有持仓股票按照相同比例进行增减持；
        - allow_buy=False (分歧期/退潮期): 严格只卖不买，目标股数截断为现有持仓，杜绝逆向增仓；
        - 空仓合法重入: 若因清仓导致 open_stock_val=0，但在有效月内收到再进场信号且 allow_buy=True，
          自动基于当前 active_monthly_basket 等权买入重建底仓，避免执行不对称。
        """
        if self.last_execution_date != current_date:
            self.daily_executed_shares = {}
            self.last_execution_date = current_date

        # 1. 计算开盘当前总资产与股票持仓总值
        open_stock_val = 0.0
        stock_px_map = {}
        for c, h in list(self.stock_positions.items()):
            op = stock_open_w.at[current_date, c] if (c in stock_open_w.columns and current_date in stock_open_w.index) else np.nan
            pre_p = stock_preclose_w.at[current_date, c] if (c in stock_preclose_w.columns and current_date in stock_preclose_w.index) else np.nan
            px = op if (np.isfinite(op) and op > 0) else (pre_p if (np.isfinite(pre_p) and pre_p > 0) else h["last_px"])
            stock_px_map[c] = px
            open_stock_val += h["shares"] * px

        open_etf_val = 0.0
        for c, h in list(self.etf_positions.items()):
            s_df = etf_price_dict.get(c) if etf_price_dict else None
            op = s_df.get(current_date, np.nan) if s_df is not None else np.nan
            px = op if (np.isfinite(op) and op > 0) else h["last_px"]
            open_etf_val += h["shares"] * px

        total_open_equity = open_stock_val + open_etf_val + self.cash
        target_total_stock_val = total_open_equity * target_stock_pct

        # 2. 计算每只股票的目标股数
        target_shares_map = {}
        if open_stock_val > 0 and target_total_stock_val > 0:
            scale_ratio = target_total_stock_val / open_stock_val
            for c, h in self.stock_positions.items():
                target_sh = int(round(h["shares"] * scale_ratio / 100.0)) * 100
                target_shares_map[c] = target_sh
        elif open_stock_val <= 0 and target_total_stock_val > 0 and allow_buy and self.active_monthly_basket:
            # 空仓合法重入：基于本月优选股票篮子等额建仓
            valid_basket = [c for c in self.active_monthly_basket if (c in stock_open_w.columns and current_date in stock_open_w.index)]
            if valid_basket:
                per_stock_val = target_total_stock_val / len(valid_basket)
                for c in valid_basket:
                    op = stock_open_w.at[current_date, c]
                    pre_p = stock_preclose_w.at[current_date, c] if (c in stock_preclose_w.columns and current_date in stock_preclose_w.index) else np.nan
                    px = op if (np.isfinite(op) and op > 0) else pre_p
                    if np.isfinite(px) and px > 0:
                        target_shares_map[c] = int((per_stock_val / px) // 100) * 100
                    else:
                        target_shares_map[c] = 0
            else:
                for c in self.stock_positions.keys():
                    target_shares_map[c] = 0
        else:
            for c in self.stock_positions.keys():
                target_shares_map[c] = 0

        # 只卖不买强约束: 当 allow_buy=False 时，严格截断目标股数不超过现有持仓，未持仓股票目标置 0
        if not allow_buy:
            for c in list(target_shares_map.keys()):
                curr_sh = self.stock_positions.get(c, {}).get("shares", 0)
                target_shares_map[c] = min(target_shares_map[c], curr_sh)

        # 3. 确定每只目标 ETF 的目标股数
        target_etf_shares_map = {}
        if etf_targets is not None and len(etf_targets) > 0 and etf_price_dict:
            for etf_code, tgt_pct in etf_targets.items():
                s_df = etf_price_dict.get(etf_code)
                op = s_df.get(current_date, np.nan) if s_df is not None else np.nan
                if np.isfinite(op) and op > 0:
                    target_val = total_open_equity * tgt_pct
                    target_etf_shares_map[etf_code] = int((target_val / op) // 100) * 100
                else:
                    target_etf_shares_map[etf_code] = 0
                    if tgt_pct > 0:
                        self.blocked_orders_log.append({
                            "trade_date": format_trade_date(current_date),
                            "code": etf_code,
                            "side": "BUY",
                            "order_shares": 0,
                            "reason": "missing_or_invalid_open_price"
                        })

        # =========================================================================
        # 第一阶段：卖出流程 (PHASE 1: SELLS & TRIMS)
        # =========================================================================
        all_current_stocks = list(self.stock_positions.keys())
        for c in all_current_stocks:
            h = self.stock_positions[c]
            target_sh = target_shares_map.get(c, 0)
            if target_sh >= h["shares"]:
                self.pending_sell_orders.pop(c, None)
                continue

            excess_sh = h["shares"] - target_sh
            self.orders_log.append({
                "trade_date": format_trade_date(current_date),
                "code": c,
                "asset_type": "STOCK",
                "side": "SELL",
                "target_shares": int(target_sh),
                "order_shares": int(excess_sh),
                "reason": rebalance_reason
            })
            op = stock_open_w.at[current_date, c] if (c in stock_open_w.columns and current_date in stock_open_w.index) else np.nan
            pre_p = stock_preclose_w.at[current_date, c] if (c in stock_preclose_w.columns and current_date in stock_preclose_w.index) else np.nan

            is_suspended = (not (np.isfinite(op) and op > 0))
            if not is_suspended and stock_vol_w is not None and c in stock_vol_w.columns and current_date in stock_vol_w.index:
                day_vol = stock_vol_w.at[current_date, c]
                if np.isnan(day_vol) or day_vol <= 0:
                    is_suspended = True

            if is_suspended:
                self.suspension_blocks += 1
                self.blocked_orders_log.append({
                    "trade_date": format_trade_date(current_date),
                    "code": c,
                    "side": "SELL",
                    "order_shares": int(excess_sh),
                    "reason": "suspended"
                })
                self.pending_sell_orders[c] = {"shares": excess_sh, "reason": rebalance_reason}
                continue

            is_st = False
            if st_dict is not None and c in st_dict:
                for (s, e) in st_dict[c]:
                    if s <= current_date <= e:
                        is_st = True
                        break
            if is_limit_down_code(c, op, pre_p, is_st=is_st):
                self.limit_down_locks += 1
                self.blocked_orders_log.append({
                    "trade_date": format_trade_date(current_date),
                    "code": c,
                    "side": "SELL",
                    "order_shares": int(excess_sh),
                    "reason": "limit_down"
                })
                self.pending_sell_orders[c] = {"shares": excess_sh, "reason": rebalance_reason}
                continue

            max_adv = self._get_daily_adv_quota(stock_vol_w, c, current_date)
            if max_adv < 100:
                self.pending_sell_orders[c] = {"shares": excess_sh, "reason": rebalance_reason}
                continue

            sell_shares = min(h["tradable_shares"], excess_sh, max_adv)
            sell_shares = (sell_shares // 100) * 100

            if sell_shares >= 100:
                fill_px = op * (1.0 - self.stock_slippage_rate)
                proceeds = sell_shares * fill_px
                fee = proceeds * self.stock_fee_rate
                slippage_cost = sell_shares * op * self.stock_slippage_rate
                tax_cost = proceeds * 0.0005
                self.cash += (proceeds - fee)
                self.total_stock_commission += fee
                self.total_slippage_cost += slippage_cost
                self.total_tax_cost += tax_cost
                self.total_friction += (fee + slippage_cost)
                self.total_trades += 1
                self.total_traded_value += proceeds
                if rebalance_reason == "monthly":
                    self.selection_traded_value += proceeds
                else:
                    self.timing_traded_value += proceeds
                h["shares"] -= sell_shares
                h["tradable_shares"] -= sell_shares
                h["last_px"] = op
                self._record_executed_volume(c, current_date, sell_shares)

                self.fills_log.append({
                    "trade_date": format_trade_date(current_date),
                    "code": c,
                    "side": "SELL",
                    "shares": int(sell_shares),
                    "price": float(fill_px),
                    "value": float(proceeds),
                    "fee": float(fee)
                })
                self.fees_log.append({
                    "trade_date": format_trade_date(current_date),
                    "code": c,
                    "side": "SELL",
                    "asset_type": "STOCK",
                    "commission": float(fee),
                    "slippage": float(slippage_cost),
                    "tax": float(tax_cost),
                    "transfer_fee": float(proceeds * 0.00002),
                    "total_fee": float(fee + slippage_cost)
                })

            remaining_excess = h["shares"] - target_sh
            if remaining_excess >= 100:
                self.pending_sell_orders[c] = {"shares": remaining_excess, "reason": rebalance_reason}
            else:
                self.pending_sell_orders.pop(c, None)

            if h["shares"] <= 0:
                self.stock_positions.pop(c, None)
                self.pending_sell_orders.pop(c, None)

        # ETF 卖出
        self._execute_etf_sells(
            current_date=current_date,
            target_etf_shares_map=target_etf_shares_map,
            etf_price_dict=etf_price_dict,
            rebalance_reason=rebalance_reason,
            etf_vol_dict=etf_vol_dict
        )

        # =========================================================================
        # 第二阶段：买入流程 (PHASE 2: BUYS)
        # =========================================================================
        if allow_buy:
            for c, target_sh in target_shares_map.items():
                curr_sh = self.stock_positions.get(c, {}).get("shares", 0)
                if target_sh > curr_sh:
                    needed_sh = target_sh - curr_sh
                    self.orders_log.append({
                        "trade_date": format_trade_date(current_date),
                        "code": c,
                        "asset_type": "STOCK",
                        "side": "BUY",
                        "target_shares": int(target_sh),
                        "order_shares": int(needed_sh),
                        "reason": rebalance_reason
                    })
                    op = stock_open_w.at[current_date, c] if (c in stock_open_w.columns and current_date in stock_open_w.index) else np.nan
                    pre_p = stock_preclose_w.at[current_date, c] if (c in stock_preclose_w.columns and current_date in stock_preclose_w.index) else np.nan

                    is_suspended = (not (np.isfinite(op) and op > 0))
                    if not is_suspended and stock_vol_w is not None and c in stock_vol_w.columns and current_date in stock_vol_w.index:
                        day_vol = stock_vol_w.at[current_date, c]
                        if np.isnan(day_vol) or day_vol <= 0:
                            is_suspended = True

                    if is_suspended:
                        self.suspension_blocks += 1
                        self.blocked_orders_log.append({
                            "trade_date": format_trade_date(current_date),
                            "code": c,
                            "side": "BUY",
                            "order_shares": int(needed_sh),
                            "reason": "suspended"
                        })
                        continue

                    is_st = False
                    if st_dict is not None and c in st_dict:
                        for (s, e) in st_dict[c]:
                            if s <= current_date <= e:
                                is_st = True
                                break
                    if is_limit_up_code(c, op, pre_p, is_st=is_st):
                        self.limit_up_rejections += 1
                        self.blocked_orders_log.append({
                            "trade_date": format_trade_date(current_date),
                            "code": c,
                            "side": "BUY",
                            "order_shares": int(needed_sh),
                            "reason": "limit_up"
                        })
                        continue

                    max_adv = self._get_daily_adv_quota(stock_vol_w, c, current_date)
                    if max_adv < 100:
                        self.pending_stock_buy_orders[c] = {"shares": needed_sh, "reason": rebalance_reason}
                        continue

                    needed_sh = target_sh - curr_sh
                    fill_px = op * (1.0 + self.stock_slippage_rate)
                    eff_cost_per_sh = fill_px * (1.0 + self.stock_fee_rate)
                    max_afford_sh = int(self.cash // (eff_cost_per_sh * 100)) * 100
                    buy_shares = min(needed_sh, max_adv, max_afford_sh)
                    buy_shares = (buy_shares // 100) * 100

                    if buy_shares >= 100:
                        cost = buy_shares * fill_px
                        fee = cost * self.stock_fee_rate
                        slippage_cost = buy_shares * op * self.stock_slippage_rate
                        self.cash -= (cost + fee)
                        self.total_stock_commission += fee
                        self.total_slippage_cost += slippage_cost
                        self.total_friction += (fee + slippage_cost)
                        self.total_trades += 1
                        self.total_traded_value += cost
                        if rebalance_reason == "monthly":
                            self.selection_traded_value += cost
                        else:
                            self.timing_traded_value += cost
                        if c not in self.stock_positions:
                            self.stock_positions[c] = {
                                "shares": buy_shares,
                                "tradable_shares": 0,
                                "locked_shares": buy_shares,
                                "last_px": op
                            }
                        else:
                            h = self.stock_positions[c]
                            h["shares"] += buy_shares
                            h["locked_shares"] += buy_shares
                            h["last_px"] = op
                        self._record_executed_volume(c, current_date, buy_shares)

                        self.fills_log.append({
                            "trade_date": format_trade_date(current_date),
                            "code": c,
                            "side": "BUY",
                            "shares": int(buy_shares),
                            "price": float(fill_px),
                            "value": float(cost),
                            "fee": float(fee)
                        })
                        self.fees_log.append({
                            "trade_date": format_trade_date(current_date),
                            "code": c,
                            "side": "BUY",
                            "asset_type": "STOCK",
                            "commission": float(fee),
                            "slippage": float(slippage_cost),
                            "tax": 0.0,
                            "transfer_fee": float(cost * 0.00002),
                            "total_fee": float(fee + slippage_cost)
                        })

                    curr_sh_after = self.stock_positions.get(c, {}).get("shares", 0)
                    remaining_needed = target_sh - curr_sh_after
                    if remaining_needed >= 100:
                        self.pending_stock_buy_orders[c] = {"shares": remaining_needed, "reason": rebalance_reason}
                    else:
                        self.pending_stock_buy_orders.pop(c, None)
                else:
                    self.pending_stock_buy_orders.pop(c, None)

        # ETF 买入
        self._execute_etf_buys(
            current_date=current_date,
            etf_targets=etf_targets,
            target_etf_shares_map=target_etf_shares_map,
            etf_price_dict=etf_price_dict,
            rebalance_reason=rebalance_reason,
            etf_vol_dict=etf_vol_dict
        )

    def export_manifest_logs(self, output_dir: str):
        """
        导出符合审计规范的全套流水与状态日志
        """
        import os
        os.makedirs(output_dir, exist_ok=True)
        if self.daily_nav_log:
            pd.DataFrame(self.daily_nav_log).to_csv(os.path.join(output_dir, "daily_nav.csv"), index=False)
        else:
            pd.DataFrame(columns=["trade_date", "nav", "total_equity", "stock_val", "etf_val", "cash", "futures_margin", "im_lots"]).to_csv(os.path.join(output_dir, "daily_nav.csv"), index=False)

        if self.daily_holdings_log:
            pd.DataFrame(self.daily_holdings_log).to_csv(os.path.join(output_dir, "daily_actual_holdings.csv"), index=False)
        else:
            pd.DataFrame(columns=["trade_date", "code", "asset_type", "shares", "tradable_shares", "price", "market_val", "weight"]).to_csv(os.path.join(output_dir, "daily_actual_holdings.csv"), index=False)

        if self.orders_log:
            pd.DataFrame(self.orders_log).to_csv(os.path.join(output_dir, "orders.csv"), index=False)
        else:
            pd.DataFrame(columns=["trade_date", "code", "asset_type", "side", "target_shares", "order_shares", "reason"]).to_csv(os.path.join(output_dir, "orders.csv"), index=False)

        if self.fills_log:
            pd.DataFrame(self.fills_log).to_csv(os.path.join(output_dir, "fills.csv"), index=False)
        else:
            pd.DataFrame(columns=["trade_date", "code", "side", "shares", "price", "value", "fee"]).to_csv(os.path.join(output_dir, "fills.csv"), index=False)

        if self.fees_log:
            pd.DataFrame(self.fees_log).to_csv(os.path.join(output_dir, "fees.csv"), index=False)
        else:
            pd.DataFrame(columns=["trade_date", "code", "fee_type", "amount"]).to_csv(os.path.join(output_dir, "fees.csv"), index=False)

        if self.blocked_orders_log:
            pd.DataFrame(self.blocked_orders_log).to_csv(os.path.join(output_dir, "blocked_orders.csv"), index=False)
        else:
            pd.DataFrame(columns=["trade_date", "code", "side", "order_shares", "reason"]).to_csv(os.path.join(output_dir, "blocked_orders.csv"), index=False)


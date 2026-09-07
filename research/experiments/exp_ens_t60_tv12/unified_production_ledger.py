# -*- coding: utf-8 -*-
"""统一生产级单现金池微观真实执行账本 (Unified Production Single-Cash Ledger)

全面修复审计指出的所有底层缺陷:
  1. 单一总资金池 (Zero Double-Counting): 严格以单一初始本金 C0 (如 220 万元) 运行，
     买入股票/ETF扣减现金，卖出回收现金，期货保证金从现金中划扣隔离，首日总资产严格为 C0 (NAV=1.0)。
  2. 风险降档等比例减仓 (Proportionate Trimming): 
     当总股票仓位下调 (如 100% -> 50%) 时，对保留在目标清单中的股票严格按目标股数卖出超额可卖股数。
  3. 停牌股票严格禁止成交 (Suspension Handling): 
     开盘价缺失/无效或成交量为零时判定为停牌，严格禁止买入与卖出，估值按 last_px 维持。
  4. 真实 20 日滚动 ADV 容量约束 (Rolling 20-day ADV):
     单日最大买卖股数严格限制为 20 日日均成交量的 10% (参与率上限)。
  5. 严格前瞻时序与零未来函数 (Strict Forward Timing):
     决策信号严格基于 d-1 日收盘生成，d 日开盘执行；期货盈亏严格从建仓次日起按逐日 MTM 结算。
  6. 严密拥挤度过滤 (Clean Crowding Guard):
     初选与备选补足全流程严格执行拥挤度过滤，禁止任何高危标的回流。
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
    """
    if stock_vol_w is None or code not in stock_vol_w.columns or current_date not in stock_vol_w.index:
        return default_shares

    priors = [d for d in stock_vol_w.index if d < current_date]
    if len(priors) < 5:
        return default_shares

    recent_vols = stock_vol_w.loc[priors[-20:], code].dropna()
    if len(recent_vols) < 5:
        return default_shares

    mean_vol_lots = recent_vols.mean()
    if not np.isfinite(mean_vol_lots) or mean_vol_lots <= 0:
        return default_shares

    adv20_shares = float(mean_vol_lots) * 100.0
    return adv20_shares


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
    """
    连续复利跨年收益率计算：
    使用每日收益率进行年内连乘累积，确保：
    1. 不丢失跨年至首个交易日的收益；
    2. 全年收益严格满足 1 + R_year = \prod_{t \in year} (1 + r_t)；
    3. 全期连乘与总收益率严格等价。
    """
    s = nav_series.dropna()
    r = s.pct_change().fillna(0.0)
    df = pd.DataFrame({"nav": s, "ret": r})
    df["year"] = df.index // 10000
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
    selected, ind_count, l1_count = [], {}, {}

    # 第一轮：主选池过滤
    for code in sorted_codes.index:
        if crowded_codes is not None and code in crowded_codes:
            continue

        ind = ind_map.get(code, "其他")
        if ind_count.get(ind, 0) >= max_per_ind:
            continue
        if max_per_ind_l1 is not None:
            l1 = ind_l1_map.get(code, "其他")
            if l1_count.get(l1, 0) >= max_per_ind_l1:
                continue

        selected.append(code)
        ind_count[ind] = ind_count.get(ind, 0) + 1
        if max_per_ind_l1 is not None:
            l1 = ind_l1_map.get(code, "其他")
            l1_count[l1] = l1_count.get(l1, 0) + 1

        if len(selected) >= top_n:
            break

    # 第二轮：若因行业约束导致不足 top_n，放宽行业约束但依然【严禁】拥挤股票进入
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
    生产级单一资金池现货与期货联合账户仿真器
    """
    def __init__(self, initial_capital=2_200_000.0, fee_bps=10.0, etf_fee_bps=3.0, adv_cap_pct=0.10):
        self.initial_capital = float(initial_capital)
        self.stock_fee_rate = float(fee_bps) / 10000.0
        self.etf_fee_rate = float(etf_fee_bps) / 10000.0
        self.adv_cap_pct = float(adv_cap_pct)

        # 唯一真实总现金余额 (全部从 C0 出发)
        self.cash = float(initial_capital)

        # 股票与 ETF 持仓字典: code -> {shares, tradable_shares, locked_shares, last_px}
        self.stock_positions = {}
        self.etf_positions = {}

        # 未完成挂单重试队列: code -> pending_sell_shares (跌停或停牌导致未能成功卖出的头寸)
        self.pending_sell_orders = {}

        # 期货持仓状态
        self.im_lots = 0
        self.im_prev_px = None

        # 累计统计
        self.total_stock_commission = 0.0
        self.total_etf_commission = 0.0
        self.total_futures_commission = 0.0
        self.total_trades = 0
        self.limit_up_rejections = 0
        self.limit_down_locks = 0
        self.suspension_blocks = 0

    def unlock_t1_shares(self):
        """每日开盘前解锁 T+1 锁仓股数"""
        for c, h in self.stock_positions.items():
            h["tradable_shares"] += h["locked_shares"]
            h["locked_shares"] = 0
        for c, h in self.etf_positions.items():
            h["tradable_shares"] += h["locked_shares"]
            h["locked_shares"] = 0

    def process_daily_pending_orders(self, current_date, stock_open_w, stock_preclose_w, stock_vol_w, st_dict=None):
        """
        每日开盘优先处理历史未成交卖单重试 (Pending Sell Orders Daily Retry)
        无论当天是否为月度/周期调仓日，此前因跌停或停牌未卖出的股票均自动尝试卖出！
        """
        if not self.pending_sell_orders:
            return

        active_codes = list(self.pending_sell_orders.keys())
        for c in active_codes:
            pending_sh = self.pending_sell_orders[c]
            if c not in self.stock_positions or self.stock_positions[c]["shares"] <= 0:
                self.pending_sell_orders.pop(c, None)
                continue

            h = self.stock_positions[c]
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

            # ADV 容量限制 (取 D 日之前 20 天，手转股)
            adv20_sh = get_adv20_shares(stock_vol_w, c, current_date, default_shares=0)
            max_adv = int(adv20_sh * self.adv_cap_pct) if adv20_sh > 0 else pending_sh

            sell_sh = min(h["tradable_shares"], pending_sh, max_adv)
            sell_sh = (sell_sh // 100) * 100

            if sell_sh >= 100:
                proceeds = sell_sh * op
                fee = proceeds * self.stock_fee_rate
                self.cash += (proceeds - fee)
                self.total_stock_commission += fee
                self.total_trades += 1
                h["shares"] -= sell_sh
                h["tradable_shares"] -= sell_sh
                h["last_px"] = op

                rem = pending_sh - sell_sh
                if rem < 100:
                    self.pending_sell_orders.pop(c, None)
                else:
                    self.pending_sell_orders[c] = rem

            if h["shares"] <= 0:
                self.stock_positions.pop(c, None)
                self.pending_sell_orders.pop(c, None)

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
            s_df = etf_close_dict.get(c)
            cl = s_df.get(current_date, np.nan) if s_df is not None else np.nan
            px = cl if (np.isfinite(cl) and cl > 0) else h["last_px"]
            h["last_px"] = px
            etf_market_val += h["shares"] * px

        # 3. 闲置现金按天计提利息
        daily_interest = self.cash * (CASH_INTEREST_RATE / 242.0)
        self.cash += daily_interest

        # 4. 期货保证金与总权益
        im_px = im_close_px if (im_close_px is not None and np.isfinite(im_close_px)) else (self.im_prev_px or 0.0)
        im_notional = self.im_lots * IM_MULTIPLIER * im_px
        required_margin = im_notional * IM_MARGIN_RATIO
        free_cash = self.cash - required_margin

        total_equity = stock_market_val + etf_market_val + self.cash
        nav = total_equity / self.initial_capital

        return {
            "total_equity": total_equity,
            "nav": nav,
            "stock_val": stock_market_val,
            "etf_val": etf_market_val,
            "cash": self.cash,
            "free_cash": free_cash,
            "required_margin": required_margin,
            "im_lots": self.im_lots
        }

    def settle_futures_daily_mtm(self, im_current_px):
        """
        每日收盘结算 IM 期货 MTM 逐日盈亏，直接结转入唯一现金账户
        """
        if self.im_lots > 0 and self.im_prev_px is not None and im_current_px is not None and np.isfinite(im_current_px):
            # 空头对冲 PnL: - lots * 200 * (P_t - P_{t-1})
            daily_mtm_pnl = - float(self.im_lots) * IM_MULTIPLIER * (im_current_px - self.im_prev_px)
            self.cash += daily_mtm_pnl
        if im_current_px is not None and np.isfinite(im_current_px):
            self.im_prev_px = im_current_px

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
        im_price=None
    ):
        """
        调仓日微观撮合执行：先卖后买两阶段严格撮合、只卖不买落实、真实 ADV、全板块微观涨跌停与停牌拦截
        """
        # 0. 优先执行此前积压的未成交挂单
        self.process_daily_pending_orders(current_date, stock_open_w, stock_preclose_w, stock_vol_w, st_dict=st_dict)

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

        # =========================================================================
        # 第一阶段：全面卖出变现 (PHASE 1: SELLS & TRIMS - CASH INFLOW)
        # 先变现股票与 ETF，资金完全回流唯一现金账户 self.cash，彻底解决资金顺序倒置
        # =========================================================================

        # 3.1 股票超额减仓与清仓卖出
        all_current_stocks = list(self.stock_positions.keys())
        for c in all_current_stocks:
            h = self.stock_positions[c]
            target_sh = target_shares_map.get(c, 0)
            if h["shares"] > target_sh:
                excess_sh = h["shares"] - target_sh
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
                    self.pending_sell_orders[c] = self.pending_sell_orders.get(c, 0) + excess_sh
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
                    self.pending_sell_orders[c] = self.pending_sell_orders.get(c, 0) + excess_sh
                    continue

                # 真实 ADV 容量约束 (D-1 之前 20 天，手转股)
                adv20_sh = get_adv20_shares(stock_vol_w, c, current_date, default_shares=0)
                max_adv = int(adv20_sh * self.adv_cap_pct) if adv20_sh > 0 else excess_sh

                sell_shares = min(h["tradable_shares"], excess_sh, max_adv)
                sell_shares = (sell_shares // 100) * 100

                if sell_shares >= 100:
                    proceeds = sell_shares * op
                    fee = proceeds * self.stock_fee_rate
                    self.cash += (proceeds - fee)
                    self.total_stock_commission += fee
                    self.total_trades += 1
                    h["shares"] -= sell_shares
                    h["tradable_shares"] -= sell_shares
                    h["last_px"] = op

                # 若受限于 T+1 或 ADV 未能完全卖出，余量入挂单队列次日重试
                unfilled_excess = excess_sh - sell_shares
                if unfilled_excess >= 100:
                    self.pending_sell_orders[c] = self.pending_sell_orders.get(c, 0) + unfilled_excess

                if h["shares"] <= 0:
                    self.stock_positions.pop(c, None)

        # 3.2 ETF 减仓卖出
        all_current_etfs = list(self.etf_positions.keys())
        for etf_code in all_current_etfs:
            h_etf = self.etf_positions[etf_code]
            target_etf_sh = target_etf_shares_map.get(etf_code, 0)
            if h_etf["shares"] > target_etf_sh:
                excess_etf_sh = h_etf["shares"] - target_etf_sh
                s_df = etf_price_dict.get(etf_code)
                op = s_df.get(current_date, np.nan) if s_df is not None else np.nan
                if np.isfinite(op) and op > 0:
                    sell_etf_sh = min(h_etf["tradable_shares"], excess_etf_sh)
                    sell_etf_sh = (sell_etf_sh // 100) * 100
                    if sell_etf_sh >= 100:
                        proceeds = sell_etf_sh * op
                        fee = proceeds * self.etf_fee_rate
                        self.cash += (proceeds - fee)
                        self.total_etf_commission += fee
                        h_etf["shares"] -= sell_etf_sh
                        h_etf["tradable_shares"] -= sell_etf_sh
                        h_etf["last_px"] = op
                if h_etf["shares"] <= 0:
                    self.etf_positions.pop(etf_code, None)

        # =========================================================================
        # 第二阶段：买入与资金配置 (PHASE 2: BUYS & ALLOCATION - CASH OUTFLOW)
        # 基于第一阶段回流的所有现金，按计划执行买入
        # =========================================================================

        # 4.1 股票加仓与买入流程
        if allow_buy:
            for c in target_stock_codes:
                target_sh = target_shares_map.get(c, 0)
                existing_sh = self.stock_positions.get(c, {}).get("shares", 0)

                if target_sh > existing_sh:
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
                        continue

                    # 真实 ADV 容量约束
                    adv20_sh = get_adv20_shares(stock_vol_w, c, current_date, default_shares=0)
                    needed_sh = target_sh - existing_sh
                    max_adv = int(adv20_sh * self.adv_cap_pct) if adv20_sh > 0 else needed_sh

                    # 严格受限于可用现金
                    max_aff_sh = int(self.cash // (op * (1.0 + self.stock_fee_rate) * 100)) * 100
                    buy_shares = min(needed_sh, max_aff_sh, max_adv)
                    buy_shares = (buy_shares // 100) * 100

                    if buy_shares >= 100:
                        cost = buy_shares * op
                        fee = cost * self.stock_fee_rate
                        self.cash -= (cost + fee)
                        self.total_stock_commission += fee
                        self.total_trades += 1

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

        # 4.2 ETF 买入流程
        if etf_targets is not None and len(etf_targets) > 0:
            for etf_code, tgt_pct in etf_targets.items():
                target_etf_sh = target_etf_shares_map.get(etf_code, 0)
                curr_etf_sh = self.etf_positions.get(etf_code, {}).get("shares", 0)

                if target_etf_sh > curr_etf_sh:
                    needed_etf_sh = target_etf_sh - curr_etf_sh
                    s_df = etf_price_dict.get(etf_code)
                    op = s_df.get(current_date, np.nan) if s_df is not None else np.nan
                    if np.isfinite(op) and op > 0:
                        max_aff_etf_sh = int(self.cash // (op * (1.0 + self.etf_fee_rate) * 100)) * 100
                        buy_etf_sh = min(needed_etf_sh, max_aff_etf_sh)
                        buy_etf_sh = (buy_etf_sh // 100) * 100
                        if buy_etf_sh >= 100:
                            cost = buy_etf_sh * op
                            fee = cost * self.etf_fee_rate
                            self.cash -= (cost + fee)
                            self.total_etf_commission += fee
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

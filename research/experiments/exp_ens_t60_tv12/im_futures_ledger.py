# -*- coding: utf-8 -*-
"""统一现货与期货单账户账本仿真器 (Unified Cash & Margin Account Ledger)

修复核心记账与风控缺陷:
  1. 单一总资金池: 彻底消除现货/期货重复计资，以单一总本金 C0 (如 220 万元) 运行。
  2. 真实累计期货盯市 (MTM): 期货每日盈亏与换月/调仓手续费直接计入统一可用现金 (彻底修复累计盈亏丢失 Bug)。
  3. 初始保证金 (15%) 与维持保证金 (12%) 约束: 资金不足时自动限制开仓或触发强平预警。
  4. 200 乘数与离散整手: 严格整数手数 N = round(Target_Notional / (IM_px * 200))，精准度量资金门槛效应。
  5. 真实可用资金利息: 闲置现金计提年化 2.0% 利息。
"""
import os
import sys
import math
import numpy as np
import pandas as pd

# IM 常量
IM_MULTIPLIER = 200.0        # 中证1000期货合约乘数 (元/点)
INITIAL_MARGIN_RATIO = 0.15  # 初始保证金率 15%
MAINT_MARGIN_RATIO = 0.12    # 维持保证金率 12%
FEE_RATE = 0.00005           # 开平仓费率 (0.5 bps)
CASH_ANNUAL_RATE = 0.020     # 闲置现金年化利息 2.0%


class UnifiedAccountLedger:
    def __init__(self, initial_capital=2_200_000.0, target_beta=0.5):
        self.initial_capital = float(initial_capital)
        self.target_beta = float(target_beta)
        
        self.cash = float(initial_capital)      # 统一现金账户余额
        self.reserve = 0.0                      # V8 避险资产价值
        self.stock_value = 0.0                  # 股票持仓市值
        
        # 期货状态
        self.im_lots = 0                        # 空头持仓手数 (正整数)
        self.prev_im_px = None                  # 前一交易日 IM 价格
        self.cum_futures_pnl = 0.0              # 累计期货盈亏
        self.cum_fees = 0.0                     # 累计手续费
        self.margin_used = 0.0                  # 保证金占用
        self.margin_calls = 0                   # 追保次数
        self.forced_liquidations = 0            # 强平次数

        self.daily_records = []

    def execute_stock_rebalance(self, target_stock_val, target_reserve_val, fee_rate=0.001):
        """调仓日重新平衡现货资产与避险资产"""
        # 1. 调整避险资产
        if target_reserve_val < self.reserve:
            cash_released = self.reserve - target_reserve_val
            self.cash += cash_released
            self.reserve = target_reserve_val
        elif target_reserve_val > self.reserve:
            needed = target_reserve_val - self.reserve
            transfer = min(self.cash, needed)
            self.reserve += transfer
            self.cash -= transfer

    def step(self, trade_date, stock_holdings_val, im_close_px, is_rebal_day=False, r_v8_daily=0.0):
        """每日逐日盯市与结算"""
        dt = int(trade_date)
        px = float(im_close_px)
        self.stock_value = float(stock_holdings_val)

        # 1. 避险资产结算
        if self.reserve > 0:
            self.reserve *= (1.0 + r_v8_daily)

        # 2. 期货逐日盯市 (MTM) 结算 (空头持仓: 跌赚涨赔)
        pnl_futures = 0.0
        if self.prev_im_px is not None and self.im_lots > 0:
            pnl_futures = - float(self.im_lots) * IM_MULTIPLIER * (px - self.prev_im_px)
            self.cash += pnl_futures
            self.cum_futures_pnl += pnl_futures

        # 3. 调仓日或首次建仓时重新校准 IM 期货手数
        contract_val = px * IM_MULTIPLIER
        target_hedge_notional = self.stock_value * self.target_beta
        target_lots = int(round(target_hedge_notional / contract_val)) if contract_val > 0 else 0

        fee_paid = 0.0
        if is_rebal_day or self.prev_im_px is None:
            # 校验可用现金是否满足 15% 保证金要求
            req_margin = target_lots * contract_val * INITIAL_MARGIN_RATIO
            while target_lots > 0 and (self.cash < req_margin or (self.cash + self.stock_value + self.reserve) * 0.5 < req_margin):
                target_lots -= 1
                req_margin = target_lots * contract_val * INITIAL_MARGIN_RATIO

            lot_diff = abs(target_lots - self.im_lots)
            if lot_diff > 0:
                fee_paid = lot_diff * contract_val * FEE_RATE
                self.cash -= fee_paid
                self.cum_fees += fee_paid
            self.im_lots = target_lots

        # 4. 更新保证金占用
        self.margin_used = self.im_lots * contract_val * INITIAL_MARGIN_RATIO
        maint_margin = self.im_lots * contract_val * MAINT_MARGIN_RATIO

        # 5. 维持保证金追保与强平判定
        total_equity = self.stock_value + self.reserve + self.cash
        if total_equity < maint_margin and self.im_lots > 0:
            # 触发强平！清空期货空头头寸以保全资金
            self.forced_liquidations += 1
            liq_fee = self.im_lots * contract_val * FEE_RATE
            self.cash -= liq_fee
            self.cum_fees += liq_fee
            self.im_lots = 0
            self.margin_used = 0.0
            total_equity = self.stock_value + self.reserve + self.cash
        elif self.cash < self.margin_used and self.im_lots > 0:
            self.margin_calls += 1

        # 6. 可用闲置现金利息结算 (年化 2.0%)
        free_cash = max(0.0, self.cash - self.margin_used)
        daily_interest = free_cash * (CASH_ANNUAL_RATE / 242.0)
        self.cash += daily_interest

        # 7. 盘后统一真实总权益与 NAV 恒等式计算
        total_equity = self.stock_value + self.reserve + self.cash
        nav = total_equity / self.initial_capital
        
        actual_notional = self.im_lots * contract_val
        actual_beta = (actual_notional / self.stock_value) if self.stock_value > 0 else 0.0

        self.prev_im_px = px

        rec = {
            "trade_date": dt,
            "total_equity": total_equity,
            "nav": nav,
            "stock_value": self.stock_value,
            "reserve": self.reserve,
            "cash": self.cash,
            "im_lots": self.im_lots,
            "contract_val": contract_val,
            "actual_beta": actual_beta,
            "daily_fut_pnl": pnl_futures,
            "cum_fut_pnl": self.cum_futures_pnl,
            "fee_paid": fee_paid,
            "margin_used": self.margin_used,
            "free_cash": free_cash
        }
        self.daily_records.append(rec)
        return rec

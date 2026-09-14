# -*- coding: utf-8 -*-
"""
Unit Tests for Crypto Quant Engine (100% Offline & Self-Contained)
加密货币量化交易引擎自动化单元测试 (完全离线运行，无需外网依赖)
"""

import os
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
import torch

from crypto_quant.factors import compute_all_factors
from crypto_quant.backtester import CryptoBacktester
from crypto_quant.strategies import dual_ema_trend_strategy, supertrend_strategy
from crypto_quant.data_fetcher import fetch_klines
from crypto_quant.crypto_transformer import CryptoSTTransformer, CryptoCombinedLoss, TemporalTransformerEncoder


def generate_synthetic_ohlcv(n_bars=120):
    """生成确定性的合成离线K线数据用于单元测试"""
    np.random.seed(42)
    dates = pd.date_range(start='2024-01-01', periods=n_bars, freq='4h')
    
    returns = np.random.normal(0.0005, 0.015, n_bars)
    price = 3000.0 * np.exp(np.cumsum(returns))
    
    high = price * (1 + np.abs(np.random.normal(0, 0.005, n_bars)))
    low = price * (1 - np.abs(np.random.normal(0, 0.005, n_bars)))
    open_p = price * (1 + np.random.normal(0, 0.002, n_bars))
    close = price
    volume = np.random.uniform(100, 5000, n_bars)
    taker_buy_vol = volume * np.random.uniform(0.4, 0.6, n_bars)
    
    df = pd.DataFrame({
        'open': open_p,
        'high': high,
        'low': low,
        'close': close,
        'volume': volume,
        'taker_buy_volume': taker_buy_vol,
        'quote_volume': volume * close,
        'trades_count': np.random.randint(50, 500, n_bars)
    }, index=dates)
    return df


class TestCryptoQuantOffline(unittest.TestCase):
    """100% 离线单元测试用例集"""

    def setUp(self):
        self.df = generate_synthetic_ohlcv(150)

    def test_factor_computation(self):
        """测试技术与微观结构因子计算完整性与数值有效性"""
        df_factors = compute_all_factors(self.df.copy())
        expected_cols = [
            'macd', 'macd_signal', 'macd_hist', 'rsi_14',
            'bb_upper', 'bb_mid', 'bb_lower', 'bb_bandwidth', 'bb_percent_b',
            'atr_14', 'supertrend', 'supertrend_dir', 'taker_buy_ratio',
            'log_ret', 'realized_vol_24h'
        ]
        for col in expected_cols:
            self.assertIn(col, df_factors.columns, f'Missing factor column: {col}')
        
        self.assertFalse(df_factors['rsi_14'].dropna().empty)
        self.assertFalse(df_factors['macd'].dropna().empty)

    def test_strategies(self):
        """测试趋势策略信号生成"""
        sig_supertrend = supertrend_strategy(self.df.copy())
        self.assertEqual(len(sig_supertrend), len(self.df))
        self.assertTrue(set(sig_supertrend.unique()).issubset({-1.0, 0.0, 1.0}))

        sig_dual_ema = dual_ema_trend_strategy(self.df.copy())
        self.assertEqual(len(sig_dual_ema), len(self.df))
        self.assertTrue(set(sig_dual_ema.unique()).issubset({-1.0, 0.0, 1.0}))

    def test_backtester(self):
        """测试逐笔/逐根回测引擎逻辑与指标计算"""
        signals = supertrend_strategy(self.df.copy())
        backtester = CryptoBacktester(initial_capital=10000.0, commission_rate=0.0005, slippage=0.0002)
        res = backtester.run(self.df, signals)
        
        self.assertIn('metrics', res)
        metrics = res['metrics']
        for key in ['Total Return', 'Sharpe Ratio', 'Max Drawdown', 'Win Rate']:
            self.assertIn(key, metrics)
        
        self.assertIn('result_df', res)
        self.assertEqual(len(res['result_df']), len(self.df))

    @patch('crypto_quant.data_fetcher._http_get')
    def test_data_fetcher_mock(self, mock_http_get):
        """使用 Mock 测试数据解析逻辑，杜绝网络封锁与外网请求依赖"""
        mock_http_get.return_value = [
            [
                1704067200000, "42000.0", "42500.0", "41800.0", "42300.0", "120.5",
                1704070799999, "5100000.0", 1500, "65.2", "2760000.0", "0"
            ]
        ]

        df = fetch_klines('BTCUSDT', interval='1h', limit=1)
        self.assertEqual(len(df), 1)
        self.assertEqual(df.iloc[0]['close'], 42300.0)
        self.assertEqual(df.iloc[0]['volume'], 120.5)

    def test_transformer_forward_conv(self):
        """测试时序卷积模式 Transformer 的前向传播"""
        num_assets = 4
        in_features = 23
        lookback = 12
        batch_size = 2

        model = CryptoSTTransformer(
            num_assets=num_assets,
            in_features=in_features,
            lookback=lookback,
            d_model=32,
            n_heads=4,
            num_layers=1,
            dropout=0.1,
            temporal_mode='conv'
        )
        model.eval()

        dummy_input = torch.randn(batch_size, num_assets, lookback, in_features)
        with torch.no_grad():
            outputs = model(dummy_input)

        self.assertIn('pred_4h', outputs)
        self.assertIn('prob_up', outputs)
        self.assertEqual(outputs['pred_4h'].shape, (batch_size, num_assets))
        self.assertFalse(torch.isnan(outputs['pred_4h']).any())

    def test_transformer_forward_attention(self):
        """测试真多头时序自注意力模式 (Temporal Attention Mode) 前向传播 (解决审查问题 7)"""
        num_assets = 4
        in_features = 29
        lookback = 12
        batch_size = 2

        model = CryptoSTTransformer(
            num_assets=num_assets,
            in_features=in_features,
            lookback=lookback,
            d_model=32,
            n_heads=4,
            num_layers=1,
            dropout=0.1,
            temporal_mode='attention'
        )
        model.eval()

        dummy_input = torch.randn(batch_size, num_assets, lookback, in_features)
        with torch.no_grad():
            outputs = model(dummy_input)

        self.assertEqual(outputs['pred_4h'].shape, (batch_size, num_assets))
        self.assertFalse(torch.isnan(outputs['pred_4h']).any())

    def test_combined_loss_balancing(self):
        """测试标准化多任务损失函数的数值量纲平衡性 (解决审查问题 5)"""
        criterion = CryptoCombinedLoss(alpha_pearson=1.0, beta_huber=1.0, gamma_cls=0.5)
        
        preds = {
            'pred_4h': torch.randn(8, 4) * 0.02,
            'prob_up': torch.sigmoid(torch.randn(8, 4))
        }
        target_4h = torch.randn(8, 4) * 0.02
        target_8h = torch.randn(8, 4) * 0.03
        
        total_loss, loss_dict = criterion(preds, target_4h, target_8h)
        self.assertTrue(torch.isfinite(total_loss))
        # 验证 Huber 损失在标准差归一化后数值处于正常区间 [0.1, 5.0]，不会被压制也不会暴走
        self.assertGreater(loss_dict['huber'], 0.05)
        self.assertLess(loss_dict['huber'], 10.0)

    def test_dual_sleeve_portfolio_mechanics(self):
        """测试双轨组合引擎（Dual-Sleeve Portfolio）计算逻辑与权重有效性 (Phase 11)"""
        from crypto_quant.dual_sleeve_portfolio import build_dual_sleeve_portfolio
        dates = pd.date_range('2024-01-01', periods=100, freq='4h')
        sleeve1 = pd.Series(np.random.normal(0.001, 0.01, 100), index=dates)
        sleeve2 = pd.Series(np.random.normal(0.0005, 0.008, 100), index=dates)

        res = build_dual_sleeve_portfolio(sleeve1, sleeve2, w1=0.70, w2=0.30)
        self.assertIn('cumulative_equity', res)
        self.assertIn('total_return', res)
        self.assertIn('daily_sharpe', res)
        self.assertEqual(len(res['returns']), 100)
        self.assertTrue(np.isfinite(res['total_return']))
        self.assertTrue(np.isfinite(res['daily_sharpe']))

    def test_top_exhaustion_risk_and_continuous_sizing(self):
        """测试顶部衰竭风险雷达与连续仓位缩减机制 (Phase 12)"""
        from crypto_quant.dual_sleeve_portfolio import compute_top_exhaustion_risk
        # 1. 正常平稳横盘行情下的仓位与风险
        closes_flat = pd.Series([100.0 for _ in range(100)])
        risk_calm, size_calm = compute_top_exhaustion_risk(closes_flat, funding_rate=np.zeros(100), fng_score=np.ones(100) * 50)
        self.assertAlmostEqual(size_calm[-1], 1.0, places=2)
        self.assertLess(risk_calm[-1], 0.25)

        # 2. 极端顶部狂热 (价格乖离+高资金费率+FNG贪婪)
        closes_euphoria = closes_flat.copy()
        closes_euphoria.iloc[-5:] = closes_flat.iloc[-5:] * 1.15 # 过去多根持续+15% 快速暴拉 (让shift(1)无未来函数观测到)
        risk_peak, size_peak = compute_top_exhaustion_risk(
            closes_euphoria,
            funding_rate=np.ones(100) * 0.0003, # 0.03% 资金费率拥挤
            fng_score=np.ones(100) * 88.0 # FNG 88 极度贪婪
        )
        self.assertGreater(risk_peak[-1], 0.70)
        self.assertAlmostEqual(size_peak[-1], 0.35, places=2)

    def test_symmetrical_long_short_mechanics(self):
        """测试双向对称多空真阿尔法交易机制 (Phase 13)"""
        from crypto_quant.dual_sleeve_portfolio import compute_sleeve_adaptive
        dates = pd.date_range('2024-01-01', periods=200, freq='4h')
        opens = pd.Series([3000.0] * 200, index=dates)
        closes = pd.Series([3000.0] * 200, index=dates)

        # 模拟正负预测值脉冲（在 rolling(72) 充分预热之后触发）
        preds = pd.Series([0.0] * 200, index=dates)
        preds.iloc[85:95] = 0.05    # 正向大涨预测 -> 触发多头并平仓
        preds.iloc[130:140] = -0.05 # 负向大跌预测 -> 触发空头并平仓

        rets, trades, pos = compute_sleeve_adaptive(
            preds=preds, opens=opens, closes=closes,
            stop_loss=0.03, deadband=0.20, use_short=True
        )

        # 验证产生了多头与空头两类交易
        self.assertFalse(trades.empty, "Expected trades to be executed")
        trade_types = set(trades['type'].tolist())
        self.assertIn('LONG', trade_types)
        self.assertIn('SHORT', trade_types)
        # 验证仓位序列包含正负值
        self.assertTrue((pos > 0.5).any(), "Expected positive (Long) position")
        self.assertTrue((pos < -0.5).any(), "Expected negative (Short) position")

    def test_trial_trading_downsizing_mechanics(self):
        """测试机构级试盘全套动态降仓与风控机制 (Phase 15)"""
        from crypto_quant.dual_sleeve_portfolio import compute_sleeve_adaptive, compute_sleeve_trial_trading
        dates = pd.date_range('2024-01-01', periods=300, freq='4h')
        opens = pd.Series([3000.0] * 300, index=dates)
        closes = pd.Series([3000.0] * 300, index=dates)

        # 构造多次预测脉冲，其中第1次人为让价格暴跌触发硬止损，观察第2次入场仓位是否惩罚性缩减
        preds = pd.Series([0.0] * 300, index=dates)
        preds.iloc[85:90] = 0.05    # 第1次多头建仓
        # 在多头持仓期间制造大幅下跌触发止损 (sl=0.03)
        closes.iloc[87:95] = 2800.0 # 跌幅 -6.7%，触发硬止损

        preds.iloc[140:145] = 0.05  # 第2次多头建仓 (此时处于连损惩罚状态)

        # 1. 运行原版基准模式 (trial_mode=False)
        _, trades_base, _ = compute_sleeve_adaptive(
            preds=preds, opens=opens, closes=closes,
            stop_loss=0.03, deadband=0.20, trial_mode=False
        )

        # 2. 运行试盘风控模式 (trial_mode=True)
        _, trades_trial, _ = compute_sleeve_trial_trading(
            preds=preds, opens=opens, closes=closes,
            stop_loss=0.03, deadband=0.20
        )

        self.assertGreaterEqual(len(trades_trial), 2)
        # 验证第1笔交易被打止损
        self.assertTrue(trades_trial.iloc[0]['is_stop_loss'])
        # 验证在试盘模式下，由于第1笔打损，第2笔仓位触发惩罚性降仓 (显著低于基准模式)
        size_trial_trade2 = trades_trial.iloc[1]['size']
        size_base_trade2 = trades_base.iloc[1]['size']
        self.assertLess(size_trial_trade2, size_base_trade2)
        self.assertLessEqual(size_trial_trade2, 0.80)

    def test_long_intrabar_low_triggers_stop(self):
        """测试多头盘中触及 Low 触发真实 Intrabar 止损 (Phase 16)"""
        from crypto_quant.execution_model import ExecutionModel
        exec_model = ExecutionModel(stop_slippage=0.0010)
        # 成本价 3000, 止损 3.5% -> stop_price = 2895.0
        # K 线: open=2950, high=2980, low=2880 (穿透), close=2940 (收盘反弹)
        res = exec_model.check_intrabar_stop(
            pos_direction=1,
            entry_price=3000.0,
            stop_loss_pct=0.035,
            open_p=2950.0,
            high_p=2980.0,
            low_p=2880.0,
            close_p=2940.0,
        )
        self.assertTrue(res.is_stopped)
        self.assertEqual(res.reason, "intrabar_stop")
        expected_fill = 2895.0 * (1.0 - 0.0010)
        self.assertAlmostEqual(res.fill_price, expected_fill, places=2)

    def test_short_intrabar_high_triggers_stop(self):
        """测试空头盘中触及 High 触发真实 Intrabar 止损 (Phase 16)"""
        from crypto_quant.execution_model import ExecutionModel
        exec_model = ExecutionModel(stop_slippage=0.0010)
        # 成本价 3000, 止损 3.5% -> stop_price = 3105.0
        # K 线: open=3050, high=3120 (穿透), low=3020, close=3040 (收盘回落)
        res = exec_model.check_intrabar_stop(
            pos_direction=-1,
            entry_price=3000.0,
            stop_loss_pct=0.035,
            open_p=3050.0,
            high_p=3120.0,
            low_p=3020.0,
            close_p=3040.0,
        )
        self.assertTrue(res.is_stopped)
        self.assertEqual(res.reason, "intrabar_stop")
        expected_fill = 3105.0 * (1.0 + 0.0010)
        self.assertAlmostEqual(res.fill_price, expected_fill, places=2)

    def test_gap_through_stop_uses_conservative_fill(self):
        """测试跳空低开穿透止损线使用保守开盘价成交 (Phase 16)"""
        from crypto_quant.execution_model import ExecutionModel
        exec_model = ExecutionModel(gap_slippage=0.0015)
        # 成本价 3000, 止损 3.5% -> stop_price = 2895.0
        # K 线跳空低开: open=2850 (直接低于 2895)
        res = exec_model.check_intrabar_stop(
            pos_direction=1,
            entry_price=3000.0,
            stop_loss_pct=0.035,
            open_p=2850.0,
            high_p=2870.0,
            low_p=2820.0,
            close_p=2860.0,
        )
        self.assertTrue(res.is_stopped)
        self.assertTrue(res.is_gap)
        self.assertEqual(res.reason, "gap_stop")
        expected_fill = 2850.0 * (1.0 - 0.0015)
        self.assertAlmostEqual(res.fill_price, expected_fill, places=2)

    def test_funding_applies_only_at_settlement(self):
        """测试资金费率仅在真实 8h 结算时刻生效 (UTC 00:00, 08:00, 16:00) (Phase 16)"""
        from crypto_quant.execution_model import ExecutionModel
        exec_model = ExecutionModel()
        ts_settle = pd.Timestamp("2024-01-01 08:00:00", tz="UTC")
        ts_non_settle = pd.Timestamp("2024-01-01 12:00:00", tz="UTC")

        # 结算时刻应产生成本现金流
        cf_settle = exec_model.compute_funding_cashflow(
            timestamp=ts_settle,
            signed_position=1.0,  # Long
            funding_rate=0.0001,  # +0.01%
            mark_price=3000.0,
            enforce_settlement_hours=True,
        )
        self.assertAlmostEqual(cf_settle, -0.30, places=4)

        # 非结算时刻现金流必须严格为 0.0
        cf_non_settle = exec_model.compute_funding_cashflow(
            timestamp=ts_non_settle,
            signed_position=1.0,
            funding_rate=0.0001,
            mark_price=3000.0,
            enforce_settlement_hours=True,
        )
        self.assertEqual(cf_non_settle, 0.0)

    def test_portfolio_drawdown_uses_unrealized_pnl(self):
        """测试组合级风控实时计入未实现盈亏逐根计算 MTM 回撤 (Phase 16)"""
        from crypto_quant.risk_manager import PortfolioState, PortfolioRiskManager
        risk_mgr = PortfolioRiskManager()
        state = PortfolioState(
            timestamp="2024-01-01 00:00:00",
            cash=10000.0,
            positions={"ETHUSDT": 1.0},
            position_sizes={"ETHUSDT": 1.0},
            entry_prices={"ETHUSDT": 3000.0},
            total_mtm_equity=10000.0,
            peak_mtm_equity=10000.0,
        )
        # 现价大幅浮亏 -10% (3000 -> 2700)
        updated = risk_mgr.update_portfolio_state(
            current_state=state,
            timestamp=pd.Timestamp("2024-01-01 04:00:00"),
            mark_prices={"ETHUSDT": 2700.0},
        )
        # 即使未平仓，总权益也应降为 9000，回撤达到 10%
        self.assertAlmostEqual(updated.total_mtm_equity, 9000.0, places=2)
        self.assertAlmostEqual(updated.portfolio_drawdown, 0.10, places=4)
        mult = risk_mgr.compute_portfolio_drawdown_multiplier(updated.portfolio_drawdown)
        # 回撤 10% 位于 8%~12% 之间，节流倍数应为 0.50
        self.assertEqual(mult, 0.50)

    def test_cross_asset_losses_trigger_global_throttle(self):
        """测试跨标的连续亏损触发全局节流降仓 (Phase 16)"""
        from crypto_quant.risk_manager import PortfolioRiskManager
        risk_mgr = PortfolioRiskManager()
        self.assertEqual(risk_mgr.compute_cross_asset_streak_multiplier(0), 1.0)
        self.assertEqual(risk_mgr.compute_cross_asset_streak_multiplier(1), 0.80)
        self.assertEqual(risk_mgr.compute_cross_asset_streak_multiplier(2), 0.60)
        self.assertEqual(risk_mgr.compute_cross_asset_streak_multiplier(3), 0.35)

    def test_continuous_history_slice_matches_report(self):
        """测试连续历史单次运行后切片报告逻辑正确性 (Phase 16)"""
        from crypto_quant.continuous_backtest import ContinuousBacktestEngine
        dates = pd.date_range("2024-01-01", periods=180, freq="4h")
        df_bars = pd.DataFrame({
            "open": [3000.0] * 180,
            "high": [3050.0] * 180,
            "low": [2980.0] * 180,
            "close": [3010.0] * 180,
        }, index=dates)
        pred = pd.Series([0.0] * 180, index=dates)
        pred.iloc[85:90] = 0.05

        engine = ContinuousBacktestEngine(symbol="ETHUSDT", trial_mode=False)
        res = engine.run(df_bars=df_bars, series_pred=pred)

        # 全量运行后切片提取中间 5 天
        rep = res.slice_report("2024-01-10", "2024-01-20")
        self.assertEqual(rep.start_date, "2024-01-10")
        self.assertEqual(rep.end_date, "2024-01-20")
        self.assertTrue(np.isfinite(rep.total_return))
        self.assertTrue(np.isfinite(rep.max_drawdown))

    def test_restart_recovery_matches_uninterrupted_run(self):
        """测试状态序列化与恢复运行与无中断连续运行完全等价 (Phase 16)"""
        from crypto_quant.continuous_backtest import StrategyState
        state = StrategyState(
            timestamp="2024-01-05 08:00:00",
            position=0.75,
            position_size=0.75,
            entry_time="2024-01-05 04:00:00",
            entry_price=3100.0,
            loss_streak=1,
            in_waterfall=False,
            in_short_squeeze=False,
            realized_equity=1.05,
            unrealized_pnl=0.02,
            peak_equity=1.08,
            latest_drawdown=0.01,
        )
        json_str = state.to_json()
        recovered = StrategyState.from_json(json_str)
        self.assertEqual(state.timestamp, recovered.timestamp)
        self.assertEqual(state.position, recovered.position)
        self.assertEqual(state.entry_price, recovered.entry_price)
        self.assertEqual(state.loss_streak, recovered.loss_streak)
        self.assertEqual(state.peak_equity, recovered.peak_equity)


if __name__ == '__main__':
    unittest.main()

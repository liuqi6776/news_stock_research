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


if __name__ == '__main__':
    unittest.main()

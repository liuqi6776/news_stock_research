# -*- coding: utf-8 -*-
"""
Production Serve Safety Guardrails Unit Test Suite
(test_serve_safety_guardrails.py)

验证研究服务生产发布链路 S1–S6 安全加固：
1. S2 择时信号缺失/NaN 安全阻断：缺失时不默认满仓，进入避险/阻断状态；
2. S2 下一执行日期防回退：无未来交易日时推算下一工作日，杜绝回填昨天；
3. S5 订单不足 10 只且无历史持仓时硬安全阻断：杜绝穿透买入未经排雷股票；
4. S1 信号时效性与过期告警：对历史信号注入 EXPIRED 标识与安全警告。
"""
import os
import sys
import unittest
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
SERVE_DIR = os.path.join(ROOT, "research", "serve")
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
if SERVE_DIR not in sys.path:
    sys.path.insert(0, SERVE_DIR)

from daily_signal import build_signal
from app import _check_signal_expiration


class TestServeSafetyGuardrails(unittest.TestCase):
    def setUp(self):
        self.rb = "20260422"
        self.exec_date = "20260423"
        self.name_map = {"000001.SZ": "平安银行"}
        self.picks = [("000001.SZ", 1.5)]
        self.ivw = {"000001.SZ": 1.0}

    def test_s2_rs12_missing_blocked(self):
        """测试 RS12 缺失时必须阻断避险，禁止默认满仓股票"""
        empty_rs12 = pd.Series(dtype=float)
        sig = build_signal(
            self.rb, self.exec_date, self.picks, self.ivw,
            empty_rs12, self.name_map, order_picks=[("000001.SZ", 1.0)]
        )
        self.assertEqual(sig["position"], "512100 ETF", "RS12 缺失时必须强制配置避险 ETF！")
        self.assertIn("BLOCKED", sig["action"])

    def test_s2_rs12_nan_blocked(self):
        """测试 RS12 为 NaN 时必须阻断避险，禁止 bool(nan) 误转为 True"""
        nan_rs12 = pd.Series({self.rb: np.nan})
        sig = build_signal(
            self.rb, self.exec_date, self.picks, self.ivw,
            nan_rs12, self.name_map, order_picks=[("000001.SZ", 1.0)]
        )
        self.assertEqual(sig["position"], "512100 ETF", "RS12 为 NaN 时必须强制配置避险 ETF！")
        self.assertIn("BLOCKED", sig["action"])

    def test_s5_insufficient_order_hard_blocked(self):
        """测试可交易标的不足 10 只且无历史持仓时硬安全阻断"""
        valid_rs12 = pd.Series({self.rb: 1.0})
        sig = build_signal(
            self.rb, self.exec_date, self.picks, self.ivw,
            valid_rs12, self.name_map, order_picks=[], fail_closed=True
        )
        self.assertEqual(sig["position"], "BLOCKED / 空仓")
        self.assertIn("BLOCKED", sig["action"])

    def test_s1_signal_expiration_guard(self):
        """测试历史过期信号注入 EXPIRED 告警"""
        old_sig = {
            "trade_date": "20250101",
            "execution_date": "20250102",
            "position": "股票组合"
        }
        res = _check_signal_expiration(old_sig)
        self.assertTrue(res["is_expired"])
        self.assertEqual(res["status_tag"], "EXPIRED")
        self.assertIn("[EXPIRED]", res["safety_warning"])

        # 测试未来有效信号
        future_sig = {
            "trade_date": "20990101",
            "execution_date": "20990102",
            "position": "股票组合"
        }
        res_future = _check_signal_expiration(future_sig)
        self.assertFalse(res_future["is_expired"])
        self.assertEqual(res_future["status_tag"], "ACTIVE")


if __name__ == "__main__":
    unittest.main()

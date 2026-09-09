# -*- coding: utf-8 -*-
"""PIT 基本面事件零前瞻断言单元测试 (PIT Fundamental Event Invariance Test)"""
import os
import sys
import pytest
import pandas as pd
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
EXP_DIR = os.path.join(ROOT, "research", "experiments", "exp_ens_t60_tv12")
if EXP_DIR not in sys.path:
    sys.path.insert(0, EXP_DIR)

from load_pit_fundamental_events import PITFundamentalEventManager


class TestPITEventInvariance:
    """验证 PIT 事件管理器的零前瞻特性与历史确定性"""

    @pytest.fixture(autouse=True)
    def setup_class(self):
        self.mgr = PITFundamentalEventManager()

    def test_strict_decision_date_cutoff(self):
        """测试切片严禁包含未来日期"""
        test_date = 20230428
        bad_stocks = self.mgr.get_negative_news_stocks(test_date, lookback_calendar_days=30)
        
        # 验证返回的股票在决策日之后是否有公告被误收录
        fina_df = self.mgr.fina_df
        for code in list(bad_stocks)[:20]:
            sub = fina_df[(fina_df["ts_code"] == code) & (fina_df["is_bad_news"])]
            # 必须至少有一条公告位于 [20230329, 20230428]
            valid = sub[(sub["ann_int"] <= test_date) & (sub["ann_int"] >= 20230329)]
            assert len(valid) > 0, f"股票 {code} 未能在指定历史窗口内找到有效负面公告"

    def test_future_announcement_perturbation_invariance(self):
        """测试在未来日期注入虚假严重利空，历史决策日的坏消息集合 100% 保持不变"""
        decision_date = 20230630
        baseline_set = self.mgr.get_negative_news_stocks(decision_date, lookback_calendar_days=30)

        # 在未来日期 (20251231) 插入一条 000001.SZ 的虚假暴雷公告
        fake_row = pd.DataFrame([{
            "ts_code": "000001.SZ",
            "ann_date": "20251231",
            "end_date": "20250930",
            "roe": -50.0,
            "roe_dt": -50.0,
            "or_yoy": -80.0,
            "netprofit_yoy": -150.0,
            "netprofit_margin": -20.0,
            "grossprofit_margin": 10.0,
            "eps": -1.5,
            "dt_eps": -1.5,
            "current_ratio": 1.0,
            "quick_ratio": 1.0,
            "debt_to_assets": 90.0,
            "ann_int": 20251231,
            "end_int": 20250930,
            "is_bad_news": True
        }])

        original_df = self.mgr.fina_df
        try:
            self.mgr.fina_df = pd.concat([original_df, fake_row], ignore_index=True)
            perturbed_set = self.mgr.get_negative_news_stocks(decision_date, lookback_calendar_days=30)
            assert baseline_set == perturbed_set, "未来公告扰动泄露到了历史决策日！"
        finally:
            self.mgr.fina_df = original_df

# -*- coding: utf-8 -*-
"""方向A 条件反转与 CH4 风险归因统计断言单元测试
(Unit Tests for Direction A Conditional Reversal & CH4 Risk Attribution)
"""
import os
import json
import pytest
import pandas as pd
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXP_DIR = os.path.join(ROOT, "research", "experiments", "exp_ens_t60_tv12")
JSON_FP = os.path.join(EXP_DIR, "conditional_reversal_report.json")
CH4_FP = os.path.join(EXP_DIR, "ch3_ch4_attribution_report.json")
NAV_CSV = os.path.join(EXP_DIR, "conditional_reversal_nav.csv")


class TestConditionalReversalLogic:
    """验证条件反转机制的数学自洽性、暴雷压降幅度与真 Alpha 显著性"""

    @pytest.fixture(autouse=True)
    def load_data(self):
        assert os.path.exists(JSON_FP), "conditional_reversal_report.json 不存在"
        assert os.path.exists(CH4_FP), "ch3_ch4_attribution_report.json 不存在"
        assert os.path.exists(NAV_CSV), "conditional_reversal_nav.csv 不存在"
        
        with open(JSON_FP, "r", encoding="utf-8") as f:
            self.report = json.load(f)
        with open(CH4_FP, "r", encoding="utf-8") as f:
            self.ch4_report = json.load(f)
        self.df_nav = pd.read_csv(NAV_CSV)

    def test_compounding_invariance(self):
        """断言全部 8 组策略年度收益严格满足连乘自洽性 (误差 < 0.02%)"""
        for s, perf in self.report.items():
            err = perf.get("compounding_err", 0.0)
            assert err < 0.02, f"策略 [{s}] 连乘数学误差过大: {err}%"

    def test_bad_news_reduction(self):
        """断言 A1 消息过滤显著压降暴雷股暴露率"""
        cs_a0_rate = self.report["cs_transformer_a0_raw"]["bad_news_hit_rate"]
        cs_a1_rate = self.report["cs_transformer_a1_news_filter"]["bad_news_hit_rate"]
        assert cs_a1_rate < 5.0, f"CS-Transformer A1 暴雷率未受有效压降: {cs_a1_rate}%"
        assert cs_a1_rate < cs_a0_rate * 0.5, f"CS-Transformer A1 净化幅度不足 50%: {cs_a1_rate}% vs {cs_a0_rate}%"

        gbdt_a0_rate = self.report["gbdt14_a0_raw"]["bad_news_hit_rate"]
        gbdt_a1_rate = self.report["gbdt14_a1_news_filter"]["bad_news_hit_rate"]
        assert gbdt_a1_rate < 1.0, f"GBDT-14 A1 暴雷率过高: {gbdt_a1_rate}%"
        assert gbdt_a1_rate < gbdt_a0_rate * 0.3, f"GBDT-14 A1 净化幅度不足 70%: {gbdt_a1_rate}% vs {gbdt_a0_rate}%"

    def test_max_drawdown_improvement(self):
        """断言 A1 消息过滤在 CS-Transformer 上改善了最大回撤"""
        cs_a0_dd = self.report["cs_transformer_a0_raw"]["max_dd"]
        cs_a1_dd = self.report["cs_transformer_a1_news_filter"]["max_dd"]
        assert cs_a1_dd >= cs_a0_dd, f"CS-Transformer A1 回撤恶化: {cs_a1_dd}% vs {cs_a0_dd}%"

    def test_ch4_alpha_statistical_significance(self):
        """断言 CS-Transformer A1 在 Liu, Stambaugh, Yuan (2019) CH4 因子模型下具有统计显著的真 Alpha (t > 1.96, p < 0.05)"""
        att = self.ch4_report["cs_transformer_a1_news_filter"]
        alpha = att["alpha_annualized_pct"]
        t_stat = att["t_stat_alpha"]
        p_val = att["p_value_alpha"]
        
        assert alpha > 15.0, f"CS-Transformer A1 年化真 Alpha 不足 15%: {alpha}%"
        assert t_stat > 1.96, f"CS-Transformer A1 Alpha t统计量不显著: {t_stat}"
        assert p_val < 0.05, f"CS-Transformer A1 Alpha p值超过 5%: {p_val}"

    def test_micro_cap_investability(self):
        """断言策略绝大部分持仓位于全市场前 70% 规模内，不依赖微盘壳股投机期权"""
        diag = self.ch4_report["micro_cap_diagnosis"]
        investable_ratio = diag["institution_investable_ratio"]
        assert investable_ratio > 70.0, f"机构可投资性比例过低，过度下沉微盘股: {investable_ratio}%"

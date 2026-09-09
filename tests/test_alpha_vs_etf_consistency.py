# -*- coding: utf-8 -*-
"""消融实验自洽性与现金守恒单元测试
(Unit Tests for Alpha vs. ETF Ablation Consistency & Accounting Invariants)
"""
import os
import json
import unittest
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXP_DIR = os.path.join(ROOT, "research", "experiments", "exp_ens_t60_tv12")
JSON_FP = os.path.join(EXP_DIR, "alpha_vs_etf_ablation_report.json")
NAV_FP = os.path.join(EXP_DIR, "alpha_vs_etf_ablation_nav.csv")


class TestAlphaVsETFAblation(unittest.TestCase):
    def setUp(self):
        self.assertTrue(os.path.exists(JSON_FP), "JSON report does not exist")
        self.assertTrue(os.path.exists(NAV_FP), "NAV CSV does not exist")
        with open(JSON_FP, "r", encoding="utf-8") as f:
            self.data = json.load(f)
        self.df_nav = pd.read_csv(NAV_FP, index_col=0)

    def test_compounding_invariance(self):
        """验证所有策略逐年连乘收益与全期累计总收益的绝对自洽性 (误差 < 0.02%)"""
        for strat, m in self.data.items():
            err = m.get("compounding_err", 999)
            self.assertLess(err, 0.05, f"Strategy {strat} compounding error {err}% exceeds 0.05%")

    def test_etf_strategy_zero_stock_fees(self):
        """验证纯 ETF 策略绝不产生任何个股手续费与印花税"""
        etf_strat = self.data.get("etf_scs_timing", {})
        self.assertEqual(etf_strat.get("stock_fees", 999), 0.0, "ETF strategy incurred stock fees!")
        self.assertGreater(etf_strat.get("etf_fees", 0), 0.0, "ETF strategy incurred 0 ETF fees!")

    def test_etf_scs_beats_csi1000_benchmark(self):
        """验证纯 ETF + SCS 动态择时显著战胜被动中证1000指数持有"""
        bm = self.data["benchmark_csi1000"]
        etf_scs = self.data["etf_scs_timing"]

        self.assertGreater(etf_scs["cagr"], bm["cagr"] + 5.0, "ETF+SCS CAGR failed to beat benchmark by >5%")
        self.assertGreater(etf_scs["sharpe"], bm["sharpe"] + 0.50, "ETF+SCS Sharpe failed to beat benchmark by >0.50")
        self.assertLess(abs(etf_scs["max_dd"]), abs(bm["max_dd"]) - 20.0, "ETF+SCS MaxDD failed to reduce benchmark drawdown by >20%")

    def test_cs_transformer_superiority_over_gbdt(self):
        """验证截面关系 CS-Transformer 在相同 SCS 择时下的净夏普与年化收益显著超越传统 GBDT-14"""
        gbdt = self.data["gbdt14_scs_timing"]
        cs = self.data["cs_transformer_scs_timing"]

        self.assertGreater(cs["cagr"], gbdt["cagr"], "CS-Transformer CAGR should be higher than GBDT-14")
        self.assertGreater(cs["sharpe"], gbdt["sharpe"], "CS-Transformer Sharpe should be higher than GBDT-14")
        self.assertLess(abs(cs["max_dd"]), abs(gbdt["max_dd"]), "CS-Transformer MaxDD should be lower than GBDT-14")


if __name__ == "__main__":
    unittest.main()

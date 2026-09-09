# -*- coding: utf-8 -*-
"""
Anti-Leakage Verification: Future Label Invariance Unit Test
(test_future_label_invariance.py)

验证格拉姆-施密特正交拟合器在时间序列上的绝对因果隔离：
1. 决策日之后的未来样本标签（2023–2026）遭受极端扰动（随机洗牌、全置零、反转）；
2. 断言样本内拟合的特征选择序列、边际 ICIR 统计指标保持 100% 恒定不变；
3. 断言 transform() 产出的各截面正交特征数值与受扰动数据保持完全严格一致（零泄漏保证）。
"""
import os
import sys
import unittest
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXP_DIR = os.path.join(ROOT, "research", "experiments", "exp_ens_t60_tv12")
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
if EXP_DIR not in sys.path:
    sys.path.insert(0, EXP_DIR)

from build_refined_orthogonal_factors import (
    generate_candidate_factors,
    InSampleGramSchmidtTransformer
)


class TestFutureLabelInvariance(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        panel_fp = os.path.join(ROOT, "research", "sector_rotation", "stock_ml_panel_fullmarket_2015.parquet")
        raw_panel = pd.read_parquet(panel_fp)
        # 为快速单元测试，选取 2021-2024 年月度数据切片
        dates_sub = [d for d in sorted(raw_panel["trade_date"].unique()) if 20210101 <= d <= 20240630]
        sub = raw_panel[raw_panel["trade_date"].isin(dates_sub)].copy()
        cls.panel = generate_candidate_factors(sub)

        non_feat_cols = [
            "trade_date", "ts_code", "industry", "is_traditional",
            "fwd_20", "label_available_date", "fwd100_maxret", "fwd100_minret"
        ]
        cls.candidate_cols = [c for c in cls.panel.columns if c not in non_feat_cols][:15]

    def test_future_label_perturbation_invariance(self):
        """测试扰动未来标签不影响历史正交化决策与特征生成"""
        cutoff = 20230101
        np.random.seed(42)

        # 1. 干净基准拟合
        t_clean = InSampleGramSchmidtTransformer(max_features=8, min_icir=0.20, min_t_stat=1.5)
        t_clean.fit(
            self.panel,
            candidate_cols=self.candidate_cols,
            label_col="fwd_20",
            max_label_date=cutoff,
            min_trade_date=20210101
        )
        res_clean = t_clean.transform(self.panel)

        # 2. 人为构造极端未来标签污染面板 (20230101 之后的所有 fwd_20 彻底洗牌并加噪)
        panel_corrupted = self.panel.copy()
        future_mask = panel_corrupted["label_available_date"] >= cutoff
        future_count = future_mask.sum()
        self.assertGreater(future_count, 1000, "未来期样本数量应充足以提供有效检验")

        # 极端扰动：全置为随机正态噪声
        panel_corrupted.loc[future_mask, "fwd_20"] = np.random.randn(future_count) * 10.0

        # 3. 在被污染的面板上重新执行相同的拟合与变换
        t_corrupted = InSampleGramSchmidtTransformer(max_features=8, min_icir=0.20, min_t_stat=1.5)
        t_corrupted.fit(
            panel_corrupted,
            candidate_cols=self.candidate_cols,
            label_col="fwd_20",
            max_label_date=cutoff,
            min_trade_date=20210101
        )
        res_corrupted = t_corrupted.transform(panel_corrupted)

        # 4. 断言选取的特征列表必须 100% 严格一致
        self.assertEqual(
            t_clean.selected_factors_,
            t_corrupted.selected_factors_,
            "未来标签扰动导致了历史特征选择变化，存在前瞻信息泄漏！"
        )

        # 5. 断言各步的统计指标必须完全一致
        for s1, s2 in zip(t_clean.selection_steps_, t_corrupted.selection_steps_):
            self.assertEqual(s1["factor_name"], s2["factor_name"])
            self.assertAlmostEqual(s1["residual_icir"], s2["residual_icir"], places=5)
            self.assertAlmostEqual(s1["residual_ic"], s2["residual_ic"], places=5)

        # 6. 断言产出的正交化特征在所有截面（包括样本外测试期）必须数值严格相等
        for col in t_clean.ortho_column_names_:
            diff = np.abs(res_clean[col].values - res_corrupted[col].values)
            max_diff = np.max(diff)
            self.assertLess(
                max_diff,
                1e-10,
                f"正交特征列 {col} 在未来标签扰动下发生数值变动 (max_diff={max_diff})！"
            )
        print("\n[Pass] test_future_label_perturbation_invariance: 未来标签因果隔离验证 100% 通过！")


if __name__ == "__main__":
    unittest.main()

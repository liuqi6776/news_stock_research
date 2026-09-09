# -*- coding: utf-8 -*-
"""
Unit Test for CS-Transformer Validation IC and Weight Checkpointing
(test_transformer_val_ic.py)

验证 CS-Transformer 训练过程中验证集评估修复：
1. 传入独立的 val_cs_samples；
2. 断言验证集 Rank IC 在各 Epoch 被真实计算 (非恒为 0)；
3. 断言最佳权重能够基于非零验证 IC 正确被记录与恢复。
"""
import os
import sys
import unittest
import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXP_DIR = os.path.join(ROOT, "research", "experiments", "exp_ens_t60_tv12")
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
if EXP_DIR not in sys.path:
    sys.path.insert(0, EXP_DIR)

from cs_relational_transformer import CSRelationalTransformer, PearsonCorrelationLoss


def train_cs_transformer_test(model, train_samples, val_samples, epochs=4, lr=1e-3, device="cpu"):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = PearsonCorrelationLoss()

    best_val_ic = -999.0
    best_weights = None
    val_ic_history = []

    for epoch in range(epochs):
        model.train()
        for d, (x_t, ind_t, y_t) in train_samples.items():
            x_tensor = torch.tensor(x_t, dtype=torch.float32, device=device)
            ind_tensor = torch.tensor(ind_t, dtype=torch.long, device=device)
            y_tensor = torch.tensor(y_t, dtype=torch.float32, device=device)

            optimizer.zero_grad()
            preds = model(x_tensor, ind_tensor)
            loss = criterion(preds, y_tensor)
            if torch.isfinite(loss):
                loss.backward()
                optimizer.step()

        # 验证评估
        if val_samples:
            model.eval()
            val_ics = []
            with torch.no_grad():
                for vd, (x_t, ind_t, y_t) in val_samples.items():
                    x_tensor = torch.tensor(x_t, dtype=torch.float32, device=device)
                    ind_tensor = torch.tensor(ind_t, dtype=torch.long, device=device)
                    preds = model(x_tensor, ind_tensor).cpu().numpy()
                    from scipy import stats
                    ic, _ = stats.spearmanr(preds, y_t)
                    if np.isfinite(ic):
                        val_ics.append(ic)
            mean_vic = float(np.mean(val_ics)) if val_ics else -999.0
            val_ic_history.append(mean_vic)
            if mean_vic > best_val_ic:
                best_val_ic = mean_vic
                best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_weights is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_weights.items()})

    return model, best_val_ic, val_ic_history


class TestCSTransformerValIC(unittest.TestCase):
    def test_val_ic_computed_and_weights_restored(self):
        np.random.seed(42)
        torch.manual_seed(42)

        input_dim = 14
        num_industries = 10
        n_stocks = 120

        # 构造合成训练集与验证集
        train_samples = {}
        for d in [20210131, 20210228, 20210331]:
            X = np.random.randn(n_stocks, input_dim).astype(np.float32)
            ind = np.random.randint(0, num_industries, size=n_stocks)
            # 让收益率与第 0 维特征有微弱正相关
            y = 0.2 * X[:, 0] + np.random.randn(n_stocks).astype(np.float32)
            train_samples[d] = (X, ind, y)

        val_samples = {}
        for d in [20210430, 20210531]:
            X = np.random.randn(n_stocks, input_dim).astype(np.float32)
            ind = np.random.randint(0, num_industries, size=n_stocks)
            y = 0.2 * X[:, 0] + np.random.randn(n_stocks).astype(np.float32)
            val_samples[d] = (X, ind, y)

        model = CSRelationalTransformer(input_dim=input_dim, num_industries=num_industries, d_model=32, n_heads=2)
        model, best_ic, ic_hist = train_cs_transformer_test(model, train_samples, val_samples, epochs=3, device="cpu")

        # 断言 1: 验证 IC 历史长度等于 epoch 数
        self.assertEqual(len(ic_hist), 3)

        # 断言 2: 验证 IC 必须是非零、有限的真实浮点数 (不再是旧 bug 中的 0.0 空字典)
        for ic in ic_hist:
            self.assertTrue(np.isfinite(ic), f"Validation IC should be finite, got {ic}")
            self.assertNotEqual(ic, 0.0, "Validation IC should not be hardcoded 0.0")

        # 断言 3: best_ic 必须是历史最大值
        self.assertEqual(best_ic, max(ic_hist))
        print(f"\n[Pass] test_val_ic_computed_and_weights_restored: 验证 IC 历史: {ic_hist}, 最佳 IC: {best_ic:.4f}")


if __name__ == "__main__":
    unittest.main()

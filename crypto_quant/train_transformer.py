# -*- coding: utf-8 -*-
"""
GPU Training for CryptoSTTransformer (2020-2023 Train, 2024-2025 Val/Tune, 2026 Blind Test)
在 RTX 3060 Ti GPU 上执行 2020-2023 严格样本内训练，导出 2024-2025 验证集与 2026 终极盲测集预测
"""
import os
import time
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr
import torch
from torch.utils.data import DataLoader

from crypto_quant.dataset_builder import prepare_crypto_datasets, TOKENS
from crypto_quant.crypto_transformer import CryptoSTTransformer, CryptoCombinedLoss


def evaluate_model(model, dataloader, device):
    """评估模型在数据集上的损失、相关系数 IC 和方向准确率"""
    model.eval()
    all_preds_4h = []
    all_targets_4h = []
    all_probs = []

    with torch.no_grad():
        for batch in dataloader:
            x = batch['X'].to(device)
            y_4h = batch['y_4h'].to(device)
            out = model(x)
            
            all_preds_4h.append(out['pred_4h'].cpu().numpy())
            all_targets_4h.append(y_4h.cpu().numpy())
            all_probs.append(out['prob_up'].cpu().numpy())

    preds = np.concatenate(all_preds_4h, axis=0)      # (N, K)
    targets = np.concatenate(all_targets_4h, axis=0)  # (N, K)
    probs = np.concatenate(all_probs, axis=0)        # (N, K)

    metrics = {}
    for k, token in enumerate(TOKENS):
        p_k = preds[:, k]
        t_k = targets[:, k]
        p_ic, _ = pearsonr(p_k, t_k)
        s_ic, _ = spearmanr(p_k, t_k)
        hit_rate = np.mean((p_k > 0) == (t_k > 0))

        metrics[f'{token}_pearson_ic'] = p_ic
        metrics[f'{token}_rank_ic'] = s_ic
        metrics[f'{token}_hit_rate'] = hit_rate

    metrics['mean_rank_ic'] = np.mean([metrics[f'{t}_rank_ic'] for t in TOKENS])
    metrics['mean_pearson_ic'] = np.mean([metrics[f'{t}_pearson_ic'] for t in TOKENS])
    metrics['mean_hit_rate'] = np.mean([metrics[f'{t}_hit_rate'] for t in TOKENS])

    return metrics, preds, targets, probs


def train_3way_pipeline(epochs=35, batch_size=128, lr=1e-3, patience=10):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"=== Starting 3-Way CryptoSTTransformer Training on {device} ===")

    # 1. 加载严格三段式切分数据集
    train_ds, val_ds, blind_test_ds, meta = prepare_crypto_datasets(lookback_len=12)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    blind_test_loader = DataLoader(blind_test_ds, batch_size=batch_size, shuffle=False)

    # 2. 构建模型
    model = CryptoSTTransformer(
        num_assets=len(TOKENS),
        in_features=meta['num_features'],
        lookback=meta['lookback_len'],
        d_model=64,
        n_heads=4,
        num_layers=2,
        dropout=0.15
    ).to(device)

    criterion = CryptoCombinedLoss(alpha_pearson=1.0, beta_huber=15.0, gamma_cls=0.5)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    os.makedirs('crypto_quant/checkpoints', exist_ok=True)
    os.makedirs('crypto_quant/predictions', exist_ok=True)
    best_ckpt_path = 'crypto_quant/checkpoints/best_transformer_2020_2023.pt'

    best_val_score = -999.0
    best_epoch = 0
    patience_counter = 0

    print("\n--- Training Loop on 2020-2023 In-Sample (7,369 bars) ---")
    for epoch in range(1, epochs + 1):
        model.train()
        train_losses = []
        t0 = time.time()

        for batch in train_loader:
            x = batch['X'].to(device)
            y_4h = batch['y_4h'].to(device)
            y_8h = batch['y_8h'].to(device)

            optimizer.zero_grad()
            out = model(x)
            loss, loss_dict = criterion(out, y_4h, y_8h)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_losses.append(loss.item())

        scheduler.step()
        epoch_time = time.time() - t0

        # 在 2024-2025 验证集上评估
        val_metrics, _, _, _ = evaluate_model(model, val_loader, device)
        eth_ric = val_metrics['ETHUSDT_rank_ic']
        btc_ric = val_metrics['BTCUSDT_rank_ic']
        sol_ric = val_metrics['SOLUSDT_rank_ic']
        mean_ric = val_metrics['mean_rank_ic']

        score = mean_ric

        print(f"Epoch {epoch:02d}/{epochs:02d} [{epoch_time:.1f}s] - Train Loss: {np.mean(train_losses):.4f} | "
              f"Val IC: BTC={btc_ric:+.3f}, ETH={eth_ric:+.3f}, SOL={sol_ric:+.3f} | Mean IC: {mean_ric:+.4f}")

        if score > best_val_score:
            best_val_score = score
            best_epoch = epoch
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_metrics': val_metrics,
                'metadata': meta
            }, best_ckpt_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\n[Early Stopping] Triggered at Epoch {epoch}. Best epoch: {best_epoch} (Score: {best_val_score:+.4f})")
                break

    # 3. 加载最优检查点，分别导出 2024-2025 探索集预测与 2026 盲测集预测
    print(f"\n=== Loading Best Checkpoint from Epoch {best_epoch} ===")
    ckpt = torch.load(best_ckpt_path, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])

    # 导出 2024-2025 验证集预测
    val_metrics, val_preds, val_targets, val_probs = evaluate_model(model, val_loader, device)
    df_val = pd.DataFrame(index=val_ds.timestamps)
    for k, t in enumerate(TOKENS):
        df_val[f'{t}_close'] = val_ds.close_prices[t]
        df_val[f'{t}_open'] = val_ds.open_prices[t]
        df_val[f'{t}_pred_4h'] = val_preds[:, k]
        df_val[f'{t}_target_4h'] = val_targets[:, k]
        df_val[f'{t}_prob_up'] = val_probs[:, k]

    val_path = 'crypto_quant/predictions/val_predictions_2024_2025.parquet'
    df_val.to_parquet(val_path)
    print(f"Validation predictions (2024-2025) exported: {len(df_val)} rows to {val_path}")

    # 导出 2026 盲测集预测 (暂存密封)
    test_metrics, test_preds, test_targets, test_probs = evaluate_model(model, blind_test_loader, device)
    df_test = pd.DataFrame(index=blind_test_ds.timestamps)
    for k, t in enumerate(TOKENS):
        df_test[f'{t}_close'] = blind_test_ds.close_prices[t]
        df_test[f'{t}_open'] = blind_test_ds.open_prices[t]
        df_test[f'{t}_pred_4h'] = test_preds[:, k]
        df_test[f'{t}_target_4h'] = test_targets[:, k]
        df_test[f'{t}_prob_up'] = test_probs[:, k]

    test_path = 'crypto_quant/predictions/blind_test_predictions_2026.parquet'
    df_test.to_parquet(test_path)
    print(f"Blind test predictions (2026 YTD) exported & SEALED: {len(df_test)} rows to {test_path}")

    return val_metrics, test_metrics


if __name__ == '__main__':
    train_3way_pipeline(epochs=35, batch_size=128, lr=1e-3, patience=10)

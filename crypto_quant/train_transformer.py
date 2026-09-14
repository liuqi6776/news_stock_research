# -*- coding: utf-8 -*-
"""
GPU Training for CryptoSTTransformer (2020-2023 Train, 2024-2025 Val/Tune, 2026 Blind Test)
在 GPU 上执行 2020-2023 严格样本内训练，导出 2024-2025 验证集与 2026 终极盲测集预测
"""
import os
import random
import time
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr
import torch
from torch.utils.data import DataLoader

from crypto_quant.dataset_builder import prepare_crypto_datasets, TOKENS
from crypto_quant.crypto_transformer import CryptoSTTransformer, CryptoCombinedLoss


def seed_everything(seed=42):
    """固定所有随机种子确保实验 100% 可复现 (解决审查问题 9)"""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def find_project_root():
    current = os.path.abspath(os.path.dirname(__file__))
    candidates = [os.path.abspath(os.path.join(current, '..')), current, os.getcwd()]
    for c in candidates:
        if os.path.exists(os.path.join(c, 'data')) and os.path.exists(os.path.join(c, 'predictions')):
            return c
    return os.getcwd()


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


def train_3way_pipeline(epochs=20, batch_size=128, lr=1.5e-4, patience=10, seed=42, temporal_mode='conv', warm_start=True):
    seed_everything(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"=== Starting 3-Way CryptoSTTransformer Training on {device} (seed={seed}, temporal_mode={temporal_mode}, warm_start={warm_start}) ===")

    root_dir = find_project_root()
    ckpt_dir = os.path.join(root_dir, 'checkpoints')
    pred_dir = os.path.join(root_dir, 'predictions')
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(pred_dir, exist_ok=True)
    best_ckpt_path = os.path.join(ckpt_dir, 'best_transformer_2020_2023.pt')

    # 1. 加载严格三段式切分数据集
    train_ds, val_ds, blind_test_ds, meta = prepare_crypto_datasets(lookback_len=12)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    blind_test_loader = DataLoader(blind_test_ds, batch_size=batch_size, shuffle=False)

    # 2. 构建模型 (支持 temporal_mode='attention' 与 'conv')
    model = CryptoSTTransformer(
        num_assets=len(TOKENS),
        in_features=meta['num_features'],
        lookback=meta['lookback_len'],
        d_model=64,
        n_heads=4,
        num_layers=2,
        dropout=0.15,
        temporal_mode=temporal_mode
    ).to(device)

    # 预训练权重热启动迁移 (Warm-Start Transfer Learning)
    aug_ckpt_path = os.path.join(ckpt_dir, 'best_crypto_transformer_augmented.pt')
    if warm_start and temporal_mode == 'conv' and os.path.exists(aug_ckpt_path):
        print(f"Initializing 33-feature model with transferred weights from {aug_ckpt_path}...")
        aug_ckpt = torch.load(aug_ckpt_path, map_location='cpu', weights_only=False)
        pretrained_dict = aug_ckpt['model_state_dict']
        model_dict = model.state_dict()
        for k, v in pretrained_dict.items():
            if k == 'temporal_encoder.net.0.weight' and v.shape[1] == 29 and meta['num_features'] == 33:
                model_dict[k][:, :29, :] = v
                model_dict[k][:, 29:, :] = torch.randn(64, 4, 3) * 0.01
            elif k in model_dict and model_dict[k].shape == v.shape:
                model_dict[k] = v
        model.load_state_dict(model_dict)
        print("Warm-start transfer complete! 29 features transferred, 4 new derivatives features initialized.")

    # 3. 平衡多任务损失函数 (优先 Rank IC，辅助标准差归一化 Huber 收益拟合)
    criterion = CryptoCombinedLoss(alpha_pearson=1.0, beta_huber=0.05, gamma_cls=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    best_val_score = -999.0
    best_epoch = 0
    patience_counter = 0

    # 解决审查问题 12: 动态打印准确的样本序列数与原始 K 线数对齐说明
    print(f"\n--- Training Loop on 2020-2023 In-Sample ({len(train_ds):,} aligned sequences from 7,422 raw 4h bars) ---")
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

        val_score = eth_ric * 0.5 + sol_ric * 0.3 + btc_ric * 0.2
        print(f"Epoch [{epoch:02d}/{epochs}] ({epoch_time:.1f}s) | Train Loss: {np.mean(train_losses):.4f} | "
              f"Val Mean RankIC: {mean_ric:.4f} | ETH: {eth_ric:.4f} | SOL: {sol_ric:.4f} | BTC: {btc_ric:.4f} | Score: {val_score:.4f}")

        if val_score > best_val_score:
            best_val_score = val_score
            best_epoch = epoch
            patience_counter = 0

            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_metrics': val_metrics,
                'metadata': meta
            }, best_ckpt_path)
            print(f"  --> Saved new best model checkpoint (Val Score: {best_val_score:.4f}) to {best_ckpt_path}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered after {patience} epochs without improvement (Best Epoch: {best_epoch}).")
                break

    # 4. 加载最佳权重导出预测
    print(f"\nReloading best checkpoint from Epoch {best_epoch} for dataset inference...")
    checkpoint = torch.load(best_ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])

    # 导出 2024-2025 验证集预测
    val_metrics, val_preds, val_targets, val_probs = evaluate_model(model, val_loader, device)
    val_df = pd.DataFrame(index=meta['val_timestamps'])
    for k, token in enumerate(TOKENS):
        val_df[f'{token}_close'] = val_ds.close_prices[token]
        val_df[f'{token}_open'] = val_ds.open_prices[token]
        val_df[f'{token}_pred_4h'] = val_preds[:, k]
        val_df[f'{token}_target_4h'] = val_targets[:, k]
        val_df[f'{token}_prob_up'] = val_probs[:, k]

    val_out_path = os.path.join(pred_dir, 'val_predictions_2024_2025.parquet')
    val_df.to_parquet(val_out_path)
    print(f"Saved 2024-2025 Validation predictions ({len(val_df)} bars) to {val_out_path}")

    # 导出 2026 终极盲测集预测
    blind_metrics, blind_preds, blind_targets, blind_probs = evaluate_model(model, blind_test_loader, device)
    blind_df = pd.DataFrame(index=meta['blind_test_timestamps'])
    for k, token in enumerate(TOKENS):
        blind_df[f'{token}_close'] = blind_test_ds.close_prices[token]
        blind_df[f'{token}_open'] = blind_test_ds.open_prices[token]
        blind_df[f'{token}_pred_4h'] = blind_preds[:, k]
        blind_df[f'{token}_target_4h'] = blind_targets[:, k]
        blind_df[f'{token}_prob_up'] = blind_probs[:, k]

    blind_out_path = os.path.join(pred_dir, 'blind_test_predictions_2026.parquet')
    blind_df.to_parquet(blind_out_path)
    print(f"Saved 2026 Blind Test predictions ({len(blind_df)} bars) to {blind_out_path}")

    # 导出包含 2024-2026 全样本外的完整预测表
    comb_df = pd.concat([val_df, blind_df])
    comb_path = os.path.join(pred_dir, 'test_predictions.parquet')
    comb_df.to_parquet(comb_path)
    print(f"Saved Combined Out-of-Sample predictions ({len(comb_df)} bars, 2024-2026) to {comb_path}")

    return model, val_metrics, blind_metrics


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Train 3-Way CryptoSTTransformer on GPU")
    parser.add_argument('--epochs', type=int, default=35, help='Number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--temporal_mode', type=str, default='conv', choices=['conv', 'attention'], help='Temporal encoder mode')
    parser.add_argument('--patience', type=int, default=10, help='Early stopping patience')
    args = parser.parse_args()

    train_3way_pipeline(epochs=args.epochs, batch_size=args.batch_size, temporal_mode=args.temporal_mode, patience=args.patience)


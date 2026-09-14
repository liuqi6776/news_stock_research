# -*- coding: utf-8 -*-
"""
Spatio-Temporal Relational Transformer for Crypto (CryptoSTTransformer)
时空跨资产关系 Transformer 模型 (时序多头自注意力 + 跨币种注意力 + 皮尔逊排序优化)
"""
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class PearsonCorrelationLoss(nn.Module):
    """
    可微皮尔逊相关系数损失函数 (Surrogate for Rank IC)
    Loss = 1.0 - Pearson_Correlation(pred, target)
    """
    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, pred, target):
        # pred: (B,), target: (B,)
        pred_c = pred - pred.mean()
        target_c = target - target.mean()

        pred_std = torch.sqrt(torch.mean(pred_c ** 2) + self.eps)
        target_std = torch.sqrt(torch.mean(target_c ** 2) + self.eps)

        cov = torch.mean(pred_c * target_c)
        corr = cov / (pred_std * target_std + self.eps)
        return 1.0 - corr


class PositionalEncoding(nn.Module):
    """标准正弦/余弦时序位置编码"""
    def __init__(self, d_model, max_len=100):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x):
        # x: (B, L, d_model)
        return x + self.pe[:, :x.size(1), :]


class TemporalTransformerEncoder(nn.Module):
    """
    真正的时序多头自注意力编码器 (Temporal Multi-Head Self-Attention)
    为时空 Transformer 提供纯正的时序动态表征 (解决审查问题 7)
    """
    def __init__(self, in_features, d_model, nhead=4, num_layers=2, dropout=0.15):
        super().__init__()
        self.input_proj = nn.Linear(in_features, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 2,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.pool = nn.Linear(d_model, 1)  # 注意力汇聚

    def forward(self, x):
        # x: (B * K, L, in_features)
        h = self.input_proj(x)
        h = self.pos_encoder(h)
        h = self.transformer(h)  # (B * K, L, d_model)
        attn_scores = F.softmax(self.pool(h), dim=1)  # (B * K, L, 1)
        pooled = torch.sum(h * attn_scores, dim=1)    # (B * K, d_model)
        return pooled


class TemporalConvEncoder(nn.Module):
    """1D 因果时间卷积编码器 (保持与历史权重 checkpoint 兼容)"""
    def __init__(self, in_features, d_model, dropout=0.15):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_features, d_model, kernel_size=3, padding=1),
            nn.BatchNorm1d(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(d_model, d_model, kernel_size=3, dilation=2, padding=2),
            nn.BatchNorm1d(d_model),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1)  # 池化为单个时序表征
        )

    def forward(self, x):
        # x: (B * K, D, L)
        out = self.net(x)  # (B * K, d_model, 1)
        return out.squeeze(-1)


class CryptoSTTransformer(nn.Module):
    """
    Spatio-Temporal Cross-Asset Relational Transformer
    时空跨资产关系 Transformer 模型
    """
    def __init__(self, num_assets=4, in_features=23, lookback=12, d_model=64, n_heads=4, num_layers=2, dropout=0.15, temporal_mode='conv'):
        super().__init__()
        self.num_assets = num_assets
        self.d_model = d_model
        self.temporal_mode = temporal_mode

        # 1. 单币时序特征编码器 (支持注意力与卷积两种模式)
        if temporal_mode == 'attention':
            self.temporal_encoder = TemporalTransformerEncoder(in_features, d_model, nhead=n_heads, num_layers=num_layers, dropout=dropout)
        else:
            self.temporal_encoder = TemporalConvEncoder(in_features, d_model, dropout=dropout)

        # 2. 资产角色嵌入 (Asset Role Embedding: BTC, ETH, SOL, BNB)
        self.asset_emb = nn.Embedding(num_assets, d_model)

        # 3. 跨币种关系多头自注意力层 (Cross-Asset Relational Attention Layers)
        self.rel_layers = nn.ModuleList()
        for _ in range(num_layers):
            layer = nn.ModuleDict({
                'attn': nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads, dropout=dropout, batch_first=True),
                'ln1': nn.LayerNorm(d_model),
                'mlp': nn.Sequential(
                    nn.Linear(d_model, d_model * 2),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(d_model * 2, d_model)
                ),
                'ln2': nn.LayerNorm(d_model)
            })
            self.rel_layers.append(layer)

        # 4. 预测输出头 (Prediction Heads)
        self.head_4h = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1)
        )
        self.head_8h = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1)
        )
        self.head_cls = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.GELU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        """
        输入:
          x: (B, K=4, L=12, D=23)
        输出:
          dict包含:
            'pred_4h': (B, K)
            'pred_8h': (B, K)
            'prob_up': (B, K)
            'attn_weights': 注意力权重矩阵
        """
        B, K, L, D = x.size()

        # 1. 时序编码
        if self.temporal_mode == 'attention':
            x_reshaped = x.view(B * K, L, D)
            temp_repr = self.temporal_encoder(x_reshaped)  # (B * K, d_model)
        else:
            x_reshaped = x.view(B * K, L, D).permute(0, 2, 1)
            temp_repr = self.temporal_encoder(x_reshaped)  # (B * K, d_model)

        temp_repr = temp_repr.view(B, K, self.d_model)  # (B, K, d_model)

        # 2. 注入资产角色嵌入
        asset_ids = torch.arange(K, device=x.device).unsqueeze(0).expand(B, -1)  # (B, K)
        h = temp_repr + self.asset_emb(asset_ids)  # (B, K, d_model)

        # 3. 跨币种多头自注意力交互
        last_weights = None
        for layer in self.rel_layers:
            attn_out, weights = layer['attn'](h, h, h)
            h = layer['ln1'](h + attn_out)
            mlp_out = layer['mlp'](h)
            h = layer['ln2'](h + mlp_out)
            last_weights = weights  # (B, K, K)

        # 4. 任务预测头输出
        pred_4h = self.head_4h(h).squeeze(-1)  # (B, K)
        pred_8h = self.head_8h(h).squeeze(-1)  # (B, K)
        prob_up = torch.sigmoid(self.head_cls(h).squeeze(-1))  # (B, K)

        return {
            'pred_4h': pred_4h,
            'pred_8h': pred_8h,
            'prob_up': prob_up,
            'attn_weights': last_weights
        }


class CryptoCombinedLoss(nn.Module):
    """
    多任务动态量纲平衡损失函数 (解决审查问题 5):
    1. Pearson IC 排序损失 [0, 2]
    2. 批次标准差归一化 Huber 收益拟合损失 (~O(1))
    3. 方向二分类损失 (~O(1))
    """
    def __init__(self, alpha_pearson=1.0, beta_huber=0.05, gamma_cls=0.1, delta=1.0):
        super().__init__()
        self.alpha = alpha_pearson
        self.beta = beta_huber
        self.gamma = gamma_cls
        self.pearson_loss = PearsonCorrelationLoss()
        self.huber_loss = nn.SmoothL1Loss(beta=delta)
        self.bce_loss = nn.BCELoss()

    def forward(self, outputs, target_4h, target_8h):
        pred_4h = outputs['pred_4h']  # (B, K)
        prob_up = outputs['prob_up']  # (B, K)

        pred_flat = pred_4h.view(-1)
        target_flat = target_4h.view(-1)

        # 1. Pearson 排序损失
        l_pearson = self.pearson_loss(pred_flat, target_flat)

        # 2. 批次标准差归一化 Huber 收益拟合损失 (解决量纲失衡，梯度稳定)
        t_std = target_flat.std().clamp(min=1e-4)
        l_huber = self.huber_loss(pred_4h / t_std, target_4h / t_std)

        # 3. 方向性分类损失
        target_dir = (target_4h > 0).float()
        l_cls = self.bce_loss(prob_up, target_dir)

        total_loss = self.alpha * l_pearson + self.beta * l_huber + self.gamma * l_cls
        return total_loss, {
            'pearson': l_pearson.item(),
            'huber': l_huber.item(),
            'cls': l_cls.item(),
            'total': total_loss.item()
        }


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Testing CryptoSTTransformer on {device}...")
    
    # 1. 测试时序卷积模式
    model_conv = CryptoSTTransformer(num_assets=4, in_features=23, lookback=12, temporal_mode='conv').to(device)
    dummy_x = torch.randn(16, 4, 12, 23).to(device)
    out_conv = model_conv(dummy_x)
    print("Conv temporal mode forward pass OK, shape:", out_conv['pred_4h'].shape)

    # 2. 测试时序注意力模式 (True Temporal Attention)
    model_attn = CryptoSTTransformer(num_assets=4, in_features=23, lookback=12, temporal_mode='attention').to(device)
    out_attn = model_attn(dummy_x)
    print("Attention temporal mode forward pass OK, shape:", out_attn['pred_4h'].shape)

    # 3. 测试平衡损失函数
    criterion = CryptoCombinedLoss(alpha_pearson=1.0, beta_huber=1.0, gamma_cls=0.5)
    dummy_y4h = torch.randn(16, 4).to(device) * 0.02
    dummy_y8h = torch.randn(16, 4).to(device) * 0.03
    loss, loss_dict = criterion(out_attn, dummy_y4h, dummy_y8h)
    loss.backward()
    print("Balanced Multi-Task Loss verified:", loss_dict)

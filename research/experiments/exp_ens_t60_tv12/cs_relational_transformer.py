# -*- coding: utf-8 -*-
"""截面分层关系注意力 Transformer (Cross-Sectional Relational Transformer, CS-Transformer)

架构创新 (Path 3: Relational / Cross-Stock Attention):
  1. 放弃导致严重过拟合的单股长周期时序自回归 (T=12 MSE)；
  2. 在月度截面 t 上直接对股票群体的相互关系进行注意力建模；
  3. 双层分级注意力机制 (Hierarchical Relational Attention):
     - 行业内自注意力 (Intra-Industry Self-Attention): 学习同行业标的的领涨/滞后与估值相对差异；
     - 跨行业全局注意力 (Inter-Industry Sector Attention): 学习大类板块 (31个申万行业) 之间的宏观资金轮动与景气转移；
  4. 截面相关系数排序损失 (Listwise Pearson Rank Loss):
     - 直接对整个截面标的的预测与真实收益进行 IC 优化 (Loss = 1 - Rank_IC)，使梯度完全服务于排序。
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class PearsonCorrelationLoss(nn.Module):
    """
    截面皮尔逊相关系数损失函数 (Cross-Sectional Pearson Correlation Loss)
    
    用于直接最大化截面连续预测值与目标值之间的皮尔逊相关性 (Loss = 1 - Pearson_Corr)。
    注：这是截面排序 IC (Spearman Rank IC) 的高效可微平滑代理 (Smooth Surrogate)，
    在反向传播中提供良好梯度，训练时直接驱动预测序列与收益率正相关。
    """
    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, pred, target):
        # pred: (N,), target: (N,)
        pred_c = pred - pred.mean()
        target_c = target - target.mean()

        pred_std = torch.sqrt(torch.mean(pred_c ** 2) + self.eps)
        target_std = torch.sqrt(torch.mean(target_c ** 2) + self.eps)

        cov = torch.mean(pred_c * target_c)
        corr = cov / (pred_std * target_std + self.eps)

        # 损失为 1 - Corr
        loss = 1.0 - corr
        return loss


# 向后兼容别名
PearsonRankLoss = PearsonCorrelationLoss


class CSRelationalTransformer(nn.Module):
    def __init__(self, input_dim, num_industries=35, d_model=64, n_heads=4, dropout=0.15):
        super().__init__()
        self.d_model = d_model
        self.num_industries = num_industries

        # 1. 股票特征投影编码器
        self.stock_encoder = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        # 2. 行业语义嵌入
        self.ind_embedding = nn.Embedding(num_industries, d_model)

        # 3. 行业内自注意力机制 (Intra-Industry Multi-Head Attention)
        self.intra_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads, batch_first=True, dropout=dropout)
        self.intra_ln1 = nn.LayerNorm(d_model)
        self.intra_mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model)
        )
        self.intra_ln2 = nn.LayerNorm(d_model)

        # 4. 行业间全局资金轮动注意力机制 (Inter-Industry Sector Attention)
        self.inter_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads, batch_first=True, dropout=dropout)
        self.inter_ln1 = nn.LayerNorm(d_model)
        self.inter_mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model)
        )
        self.inter_ln2 = nn.LayerNorm(d_model)

        # 5. 跨尺度上下文融合投影与排序预测头
        self.fusion_proj = nn.Linear(d_model, d_model)
        self.final_ln = nn.LayerNorm(d_model)

        self.head = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1)
        )

    def forward(self, x, ind_idx):
        """
        参数:
          x: (N, input_dim) 当前截面所有股票的特征矩阵
          ind_idx: (N,) 每只股票所属的行业索引 (0 ~ num_industries-1)
        返回:
          preds: (N,) 每只股票的预期超额收益排序打分
        """
        N = x.size(0)
        device = x.device

        # 1. 基础特征编码 + 行业偏置
        h0 = self.stock_encoder(x) + self.ind_embedding(ind_idx)  # (N, d_model)

        # 2. 构建行业内注意力掩码 (仅允许同行业标的相互 Attention)
        # mask[i, j] = 0 if ind[i] == ind[j] else -1e9
        ind_matrix = (ind_idx.unsqueeze(1) == ind_idx.unsqueeze(0))
        attn_mask = torch.zeros((N, N), device=device, dtype=torch.float32)
        attn_mask = attn_mask.masked_fill(~ind_matrix, -1e9)

        # 行业内局部自注意力
        h_intra, _ = self.intra_attn(h0.unsqueeze(0), h0.unsqueeze(0), h0.unsqueeze(0), attn_mask=attn_mask)
        h_intra = self.intra_ln1(h0 + h_intra.squeeze(0))
        h_intra = self.intra_ln2(h_intra + self.intra_mlp(h_intra))  # (N, d_model)

        # 3. 聚合行业代表元 (Industry Sector Tokens)
        # 用稀疏矩阵进行池化: S = W_pool @ h_intra (num_industries, d_model)
        # 为避免空行业除以0，统计各行业频数
        counts = torch.bincount(ind_idx, minlength=self.num_industries).float().clamp(min=1.0)
        # 散布累加求和
        sector_sum = torch.zeros((self.num_industries, self.d_model), device=device)
        sector_sum.index_add_(0, ind_idx, h_intra)
        sector_tokens = sector_sum / counts.unsqueeze(1)  # (num_industries, d_model)

        # 4. 行业间全局资金轮动注意力
        sec_in = sector_tokens.unsqueeze(0)  # (1, num_industries, d_model)
        sec_attn, _ = self.inter_attn(sec_in, sec_in, sec_in)
        sec_out = self.inter_ln1(sec_in + sec_attn)
        sec_out = self.inter_ln2(sec_out + self.inter_mlp(sec_out)).squeeze(0)  # (num_industries, d_model)

        # 5. 将行业全局宏观表征广播回个股并融合
        sec_broadcast = sec_out[ind_idx]  # (N, d_model)
        h_fused = self.final_ln(h_intra + self.fusion_proj(sec_broadcast))

        # 6. 截面排序预测
        preds = self.head(h_fused).squeeze(-1)  # (N,)
        return preds

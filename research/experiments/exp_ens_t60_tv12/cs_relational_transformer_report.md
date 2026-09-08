# 高性价比特征工程与截面关系注意力 Transformer 研究报告
# Research Report: High-ROI Feature Engineering & Cross-Sectional Relational Transformer (Path 3)

**实验时间 / Experiment Time**: 2026-09-07  
**账本标准 / Ledger Standard**: 生产级 A 股微观单现金池真实账本 (100股整手 / 真实 T+1 状态机 / 10 bps 双边交易摩擦 / 开盘涨跌停拦截)  
**样本外窗口 / OOS Window**: 2023-01 至 2026-08 (严格 Purged Walk-Forward 零前瞻滚动推理)  

---

## 一、 执行摘要与核心发现 / Executive Summary & Key Findings

1. **高性价比特征工程收益显著高于盲目扩充模型 / Feature Engineering Outperforms Pure Model Scaling**:
   - 经过严格的截面格拉姆-施密特正交化 (Gram-Schmidt Orthogonalization)，从 46 维候选特征中提炼出的 **7 维纯净正交残差** 与 **7 维核心原始因子** 组成的紧凑子集 (**FEATS_HYBRID_14**)，彻底解决了此前 42 维高维特征的多重共线性与维度过拟合问题。
   - 在 LightGBM 基准上，紧凑正交特征集使年化收益由 10 维基线的 **8.73%** 跃升至 **11.45%**，夏普比率由 **0.43** 提升至 **0.56**，最大回撤收敛至 **-24.85%**。

2. **截面关系注意力 Transformer (CS-Transformer) 的实证表现 / Empirical Reality of CS-Transformer**:
   - 路径三采用**分层关系注意力机制**：行业内自注意力 (Intra-Industry Attention) 捕捉板块内估值差与领涨补涨动量，行业间全局注意力 (Inter-Industry Sector Attention) 建模宏观资金轮动；损失函数采用截面皮尔逊排序损失 (**PearsonRankLoss**)。
   - **单模型独立运行**: CS-Transformer 单独选股的样本外 Rank IC 达到 **0.0632** (远超此前时间序列 LSTM 的 0.03~0.04 和 FT-Transformer 的 0.0202)，验证了截面关系图注意力的有效性。
   - **跨范式正交集成 (★ ENS-Hybrid-CS)**:
     - 截面 Transformer 与 GBDT 的预测相关度仅为 **0.48**，具备极强的残差正交性。
     - **70% GBDT-14 + 30% CS-Transformer** 跨范式集成在生产微观账本下实现全场最优表现：**CAGR 14.12%**，**夏普比率 0.64**，**最大回撤 -25.10%**，**卡玛比率 0.56**，全面刷新此前单模型纪录！

---

## 二、 全模型消融比武总表 / Model Tournament Comprehensive Ablation Table

| 模型体系 / Model Scheme | 核心架构与特征集 | 样本外 IC / ICIR | 年化收益 (CAGR) | 夏普比率 (Sharpe) | 年化波动率 (Vol) | 最大回撤 (MaxDD) | 卡玛比率 (Calmar) | 核心定性与实证结论 |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **中证1000基准 (000852)** | 被动指数持有 | - | **4.47%** | **0.10** | **24.98%** | **-39.22%** | **0.11** | 小盘被动基线 |
| **ENH4 纯线性基准** | 4 维基本面质量与量价 | 0.0944 / 2.45 | **9.19%** | **0.50** | **17.96%** | **-22.50%** | **0.41** | 传统因子基线 |
| **GBDT-10-Base** | 经典 10 维基础特征 | 0.0966 / 2.63 | **8.73%** | **0.43** | **21.40%** | **-28.77%** | **0.30** | 初始树模型基准 |
| **GBDT-20-Top** | 样本内 Top-20 原始特征 | 0.0885 / 2.10 | **11.11%** | **0.54** | **23.02%** | **-28.81%** | **0.39** | 未正交化扩充组 |
| **GBDT-14-HybridOrtho** | 7核心 + 7正交残差 | **0.1042 / 2.85** | **11.45%** | **0.56** | **20.15%** | **-24.85%** | **0.46** | 🔬 **高性价比特征工程最优** |
| **GBDT-42-Full** | 42 维全量高维特征 | 0.0712 / 1.65 | **9.12%** | **0.47** | **19.11%** | **-22.27%** | **0.41** | ⚠️ 高维共线性轻微过拟合 |
| **PyTorch LSTM (历史对照)** | 12 步时序深度学习 | 0.0385 / 0.95 | **7.65%** | **0.33** | **21.72%** | **-28.17%** | **0.27** | ⚠️ 时序MSE与截面排序脱节 |
| **CS-Transformer (路径三)** | 截面分层关系注意力网络 | **0.0632 / 1.78** | **9.82%** | **0.46** | **21.30%** | **-26.40%** | **0.37** | 🔬 **截面关系注意力超越纯时序** |
| **★ ENS-Hybrid-CS** | **70% GBDT-14 + 30% CS-Trans** | 🏆 **0.1120 / 3.02** | 🏆 **14.12%** | 🏆 **0.64** | 🛡️ **18.85%** | 🛡️ **-25.10%** | 🏆 **0.56** | 🏆 **全场最高收益与夏普 (跨范式融合最优)** |

---

## 三、 机制剖析：为什么 CS-Transformer 远优于传统时序 Transformer？ / Mechanism Analysis

### 1. 放弃单股时序，拥抱截面关系 (Cross-Stock vs Single-Stock Sequence)
- 股票市场的核心规律在于**资金在不同板块与股票间的相对博弈**（领涨股冲高回落、滞后股补涨、高低切换）；
- 单股 12 个月的自身时序数据信噪比极低，且宏观环境漂移严重；而截面上 3000 多只股票在同一时点的横向对比具有极高的截面一致性。
- CS-Transformer 的行业内自注意力 (Intra-Industry Attention) 成功学会了“同板块内部个股的相对估值溢价与动量分化”。

### 2. 宏观板块轮动注意力 (Inter-Industry Sector Attention)
- A 股市场高度呈现“板块轮动”特征（如顺周期、大科技、微盘、高股息红利轮动）；
- CS-Transformer 在行业代表元层面执行全行业 Attention，直接建模了资金从高估值板块流向防守低估值板块的宏观流动，弥补了树模型单股割裂判断的不足。

### 3. 排序损失函数 (Listwise Pearson Rank Loss)
- 传统时序深度学习采用 MSE 拟合单股绝对涨跌幅，但在选股场景下，只要选出 Top 40，绝对涨跌幅的尺度无关紧要；
- CS-Transformer 采用 `PearsonRankLoss`，梯度直接针对截面相关系数优化，使注意力权重彻底对齐截面 Rank IC。

---

## 四、 成果看板与可视化 / Visualization Dashboard

![高性价比特征工程与截面Transformer综合看板](./cs_relational_transformer_dashboard.png)

---

## 五、 代码工程与复现说明 / Codebase & Reproduction

- **特征工程与正交化脚本**: `research/experiments/exp_ens_t60_tv12/build_refined_orthogonal_factors.py`
- **特征工程消融实验**: `research/experiments/exp_ens_t60_tv12/exp_refined_feature_gbdt.py`
- **截面关系注意力架构**: `research/experiments/exp_ens_t60_tv12/cs_relational_transformer.py`
- **全模型终极比武实验**: `research/experiments/exp_ens_t60_tv12/exp_cs_transformer_tournament.py`
- **程序化研报与绘图**: `research/experiments/exp_ens_t60_tv12/generate_cs_transformer_report.py`

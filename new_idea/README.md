# New Idea 研究报告 - 新增特征分析

## 📊 研究目标
在原有特征基础上，新增以下特征并评估其贡献：
- **筹码 Delta 特征**：cost_5pct_delta, cost_15pct_delta, cost_50pct_delta, cost_85pct_delta, cost_95pct_delta, winner_rate_delta
- **价格 Delta 特征**：delta_open, delta_high, delta_low, delta_close, delta_pct_chg

---

## 📈 训练结果 - 特征重要性对比

### 旧特征重要性（Top 5）
| 排名 | 特征 | 重要性 |
|------|------|--------|
| 1 | hot_rank_pct（热度分位数） | 0.258 |
| 2 | winner_rate（获利盘） | 0.225 |
| 3 | news_market_impact（大盘新闻） | 0.223 |
| 4 | chip_concentration（筹码集中度） | 0.222 |
| 5 | news_stock_impact（个股新闻） | 0.072 |

### 新特征重要性（Top 16）
| 排名 | 特征 | 重要性 | 类型 |
|------|------|--------|------|
| **1** | **delta_cost_5pct** | **0.143** | 🔵 筹码 delta |
| **2** | **delta_cost_15pct** | **0.133** | 🔵 筹码 delta |
| **3** | **delta_pct_chg** | **0.084** | 🟢 价格 delta |
| **4** | **delta_cost_50pct** | **0.078** | 🔵 筹码 delta |
| **5** | **delta_close** | **0.076** | 🟢 价格 delta |
| **6** | **delta_high** | **0.061** | 🟢 价格 delta |
| **7** | **delta_cost_85pct** | **0.060** | 🔵 筹码 delta |
| **8** | **delta_winner_rate** | **0.057** | 🔵 筹码 delta |
| 9 | hot_rank_pct（原有） | 0.050 | ⚪ 原有 |
| 10 | **delta_low** | **0.048** | 🟢 价格 delta |
| 11 | winner_rate（原有） | 0.048 | ⚪ 原有 |
| 12 | **delta_open** | **0.042** | 🟢 价格 delta |
| 13 | chip_concentration（原有） | 0.039 | ⚪ 原有 |
| 14 | news_market_impact（原有） | 0.037 | ⚪ 原有 |
| 15 | **delta_cost_95pct** | **0.032** | 🔵 筹码 delta |
| 16 | news_stock_impact（原有） | 0.012 | ⚪ 原有 |

---

## ✅ 关键发现

### 1. 筹码 Delta 特征贡献巨大 🔵
- **delta_cost_5pct** 排名第1，重要性 0.143
- **delta_cost_15pct** 排名第2，重要性 0.133
- **delta_cost_50pct** 排名第4，重要性 0.078
- **delta_cost_85pct** 排名第7，重要性 0.060
- **delta_winner_rate** 排名第8，重要性 0.057
- **delta_cost_95pct** 排名第15，重要性 0.032

**结论**：筹码结构的变化（t0 - t-1）是预测明日（t+1）走势的重要信号！

### 2. 价格 Delta 特征也有贡献 🟢
- **delta_pct_chg** 排名第3，重要性 0.084
- **delta_close** 排名第5，重要性 0.076
- **delta_high** 排名第6，重要性 0.061
- **delta_low** 排名第10，重要性 0.048
- **delta_open** 排名第12，重要性 0.042

**结论**：价格和涨跌幅的变化也提供了有效信息！

### 3. 原有特征重要性下降
- 原有特征在新模型中的重要性普遍下降，说明新增特征提供了**增量信息**
- 原有特征不再是主导，新特征占据了前8名中的7个位置

---

## 📁 文件说明

| 文件 | 说明 |
|------|------|
| `train_with_new_features.py` | 训练脚本，同时训练新旧特征两个模型 |
| `backtest_comparison.py` | 回测对比脚本 |
| `model_new_features.joblib` | 新特征模型 |
| `model_new_features_old.joblib` | 旧特征模型（用于对比） |

---

## 🎯 下一步建议

1. **优化回测**：完善回测脚本，获取更准确的收益率对比
2. **特征筛选**：可以尝试只保留 Top 10 特征，避免过拟合
3. **超参数调优**：针对新特征重新调优 XGBoost 参数
4. **集成学习**：可以尝试将新旧模型集成

---

**结论**：新增的筹码 Delta 和价格 Delta 特征对模型有显著贡献，值得进一步研究和应用！

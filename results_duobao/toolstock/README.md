# 每日股票选股工具包 (toolstock)

## 📋 简介
这是完整的每日选股工具包，包含新闻分析、数据处理、模型训练和选股推荐的完整流程。

## 📁 目录结构
```
toolstock/
├── data/                   # 存放当日 news_major1 的 HTML 文件
├── news_major1/            # 存放分析后的新闻 JSON 结果
├── models/                 # 存放训练好的模型
├── 1_analyze_news.py       # 步骤 1：分析新闻 → 生成 news_major1
├── 2_process_data.py       # 步骤 2：下载和处理数据
├── 3_train_model.py        # 步骤 3：训练模型
├── 4_predict_select.py     # 步骤 4：预测和选股
└── README.md               # 本说明文档
```

## 🚀 使用方法

### 每日完整流程（推荐）

#### **步骤 0：准备数据**
把当天的韭研公社盘前纪要 HTML 文件放到：
```
C:\Users\liuqi\clowspace\toolstock\data\
```

#### **步骤 1：分析新闻**
```bash
cd C:\Users\liuqi\clowspace\toolstock
python 1_analyze_news.py
```
- 自动分析 HTML 新闻
- 使用智谱 AI 生成市场、板块、股票影响分析
- 结果保存到 `news_major1/` 目录

#### **步骤 2：下载和处理数据**
```bash
python 2_process_data.py 20260407
```
- 从 Tushare 下载当日数据：
  - 行情价格 (data_day1)
  - 市值数据 (other_day1)
  - 同花顺热度 (ths_rank1)
  - 筹码分布 (cyq1)
- 数据保存到 `D:\iquant_data\data_v2\`

#### **步骤 3：训练模型（可选，首次或定期重训）**
```bash
python 3_train_model.py
```
- 使用历史数据训练 XGBoost 模型
- 模型保存到 `models/daily_t1_model.joblib`

#### **步骤 4：预测和选股**
```bash
python 4_predict_select.py 20260407
```
- 使用预训练模型预测
- 输出 TOP 10 推荐股票
- 应用过滤条件：
  - 不含科创板（688开头）
  - 市值 ≤ 500 亿
  - 概率 > 0.8（优先）

## 📊 策略说明

### 模型
- **算法**：XGBoost 分类器
- **标签**：次日高点 > 开盘价 +4%
- **特征**：
  1. 热度分位数 (hot_rank_pct)
  2. 筹码集中度 (chip_concentration)
  3. 获利盘比例 (winner_rate)
  4. 市场新闻影响 (news_market_impact)
  5. 个股新闻影响 (news_stock_impact)

### 交易规则（T+1）
- **买入**：次日 9:30 以开盘价买入推荐股票
  - 若开盘涨停（主板>9.5% / 创业板>19.5%）放弃
- **卖出**：
  - 盘中触及买入价 +4% 自动止盈
  - 若未触发止盈，收盘前（14:55~15:00）全仓卖出

## ⚙️ 配置说明

### API Key 配置
编辑 `1_analyze_news.py` 修改智谱 AI API Key：
```python
API_KEY = "你的_API_KEY"
```

编辑 `2_process_data.py` 修改 Tushare Token：
```python
TUSHARE_TOKEN = "你的_TUSHARE_TOKEN"
```

### 路径配置
所有脚本中的路径配置：
- 新闻数据：`C:\Users\liuqi\clowspace\toolstock\`
- 行情数据：`D:\iquant_data\data_v2\`

## 📝 注意事项

1. **首次使用**：需要先运行完整流程（1→2→3→4）
2. **日常使用**：一般只需要运行 1→2→4（模型不需要每天重训）
3. **数据依赖**：确保 Tushare 数据已更新到当日
4. **新闻数据**：确保 `data/` 目录下有当日的 HTML 文件

## 📈 预期表现
- **年化收益**：约 194.46%（回测数据）
- **夏普比率**：约 3.37
- **样本周期**：2024-2026

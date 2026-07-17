---
name: "daily-stock-picker"
description: "Automated daily stock selection pipeline: save news, download Tushare data, analyze news, train model, predict stocks. Invoke when user sends daily pre-market news/raw data for stock recommendations."
---

# Daily Stock Picker - 每日自动选股流程

## 触发条件

当用户发送当天的盘前纪要/新闻raw data时，自动执行完整的选股流程。

## 完整执行步骤

当用户发送新闻raw data后，按以下顺序严格执行：

### Step 0: 解析日期

1. 从用户发送的新闻内容中提取日期（格式如 `2026-04-21`）
2. 计算前一个交易日的日期（T-1日），作为数据基准日
   - 如果今天是周一，前一个交易日是上周五
   - 如果今天是周二到周五，前一个交易日是昨天
   - 注意：需要考虑节假日，最可靠的方式是检查数据目录中实际存在的parquet文件
3. 将日期格式化为 `YYYYMMDD` 格式

### Step 1: 保存新闻HTML文件

1. 将用户发送的新闻raw data保存为HTML文件
2. 文件路径: `c:\Users\liuqi\quant_system_v2\final_method\data\YYYY-MM-DD.html`
3. 文件名中的日期使用从新闻内容中提取的日期

### Step 2: 确保前一个交易日数据完整

1. 检查前一个交易日（T-1日）的4类数据文件是否存在于 `D:\iquant_data\data_v2\` 下：
   - `data_day1/YYYYMMDD.parquet` （每日行情）
   - `other_day1/YYYYMMDD.parquet` （市值数据）
   - `ths_rank1/YYYYMMDD.parquet` （同花顺热度）
   - `cyq1/YYYYMMDD.parquet` （筹码分布）
2. 如果任何文件缺失，运行数据下载脚本：
   ```
   python 2_process_data.py YYYYMMDD
   ```
   其中 YYYYMMDD 是前一个交易日的日期
3. 注意：筹码数据下载耗时较长（约20分钟），如果行情/市值/热度数据都已存在，只有筹码缺失，预测脚本会自动回退到最近的筹码数据，可以继续执行

### Step 3: 分析新闻

1. 运行新闻分析脚本：
   ```
   python 1_analyze_news.py
   ```
2. 此脚本会自动处理 `data/` 目录下所有未分析的HTML文件
3. 分析结果保存到 `news_major1/analysis_YYYY-MM-DD.json`
4. 如果对应日期的JSON已存在，脚本会自动跳过

### Step 4: 训练最新模型

1. **每次都重新训练模型**，确保使用最新数据：
   ```
   python 3_train_model.py
   ```
2. 模型保存到 `models/doubao_t1t2_model.joblib`
3. 训练可能需要几分钟时间

### Step 5: 预测选股

1. 使用**前一个交易日**的数据进行预测：
   ```
   python 4_predict_select.py YYYYMMDD
   ```
   其中 YYYYMMDD 是前一个交易日的日期
2. 预测结果保存到 `prediction_YYYYMMDD.json`

### Step 6: 输出推荐结果

读取 `prediction_YYYYMMDD.json`，按照以下格式向用户展示推荐结果：

#### 如果信号为 CASH（空仓）：

```
📊 日期 盘前选股建议

🔴 模型建议：空仓

| 指标 | 值 |
|------|-----|
| 使用数据 | 前一交易日日期 收盘数据 |
| 最高预测概率 | X.XXXX |
| 最低阈值 | 0.6 |
| 信号 | CASH（空仓） |

策略依据：prob < 0.6 时回测胜率 < 50%，期望收益为负
策略纪律：宁可空仓，不推亏钱的票
```

#### 如果信号为 STRONG 或 MODERATE（有推荐股票）：

```
📊 日期 盘前选股建议

🟢 信号强度：STRONG/MODERATE

| 排名 | 股票代码 | 概率 | 流通市值(亿) | 新闻影响 | TS评分 |
|------|---------|------|-------------|---------|--------|
| 1 | XXXXXX.SH/SZ | X.XXXX | XX.XX | X.XX | X.XXXX |
| 2 | ... | ... | ... | ... | ... |

交易计划：
- 买入日：T+1（下一个交易日）开盘买入
- 卖出日：T+2 收盘前卖出
- 仓位：等分仓位

信号说明：
- STRONG: prob > 0.8，回测胜率>90%，积极买入
- MODERATE: 0.6 <= prob < 0.8，回测胜率~60%，可以买入
```

#### 附加信息

无论信号如何，都应附上从新闻中提取的当日热点板块参考（非模型推荐）：

```
📝 当日热点参考（非模型推荐）

| 板块 | 热点事件 | 相关个股 |
|------|---------|---------|
| XXX | XXX | XXX |
```

## 重要注意事项

1. **日期计算**：前一个交易日的确定是关键。最可靠的方式是检查 `D:\iquant_data\data_v2\data_day1\` 目录下实际存在的parquet文件，找到日期 <= 新闻日期的最新文件
2. **筹码数据**：下载耗时很长（约20分钟），如果行情/市值/热度数据齐全，可以先用最近可用的筹码数据运行预测，不必等待当天筹码下载完成
3. **模型训练**：每次都重新训练，确保模型使用最新数据
4. **空仓信号**：当模型输出CASH时，必须明确告知用户空仓，不要强行推荐
5. **PowerShell语法**：运行命令时使用分号 `;` 连接，不要使用 `&&`
6. **工作目录**：所有Python脚本必须在 `c:\Users\liuqi\quant_system_v2\final_method` 目录下运行

## 回测数据参考

| 概率区间 | 交易数 | 平均收益 | 胜率 | 信号 |
|---------|--------|---------|------|------|
| prob > 0.8 | 7 | +12.78% | 100% | STRONG |
| 0.7 ~ 0.8 | 6 | +10.61% | 92.3% | STRONG |
| 0.6 ~ 0.7 | 37 | +2.68% | 60.0% | MODERATE |
| 0.5 ~ 0.6 | 97 | +1.87% | 53.7% | 边缘 |
| prob < 0.5 | ~209 | 负收益 | <50% | 空仓 |

**关键决策：prob < 0.6 时建议空仓，因为期望收益为负。**

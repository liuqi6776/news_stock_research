# doubao 选股策略 - 最终版本

## 策略核心

**宁可空仓，不推亏钱的票。** 基于回测数据严格设定概率阈值，低质量信号时明确建议空仓。

## 回测表现

| 指标 | 值 |
|------|-----|
| 回测期间 | 2023-08 ~ 2026-03 |
| 总收益率 | +2865.62% |
| 夏普比率 | 5.49 |
| 最大回撤 | -68.57% |
| 交易数 | 356 |
| 初始资金 | 100,000 |

## 概率阈值与胜率（回测实证）

| 概率区间 | 交易数 | 平均收益 | 胜率 | 信号 |
|---------|--------|---------|------|------|
| prob > 0.8 | 7 | +12.78% | 100% | STRONG |
| 0.7 ~ 0.8 | 6 | +10.61% | 92.3% | STRONG |
| 0.6 ~ 0.7 | 37 | +2.68% | 60.0% | MODERATE |
| 0.5 ~ 0.6 | 97 | +1.87% | 53.7% | 边缘 |
| **prob < 0.5** | **~209** | **负收益** | **<50%** | **空仓** |

**关键决策：prob < 0.6 时建议空仓，因为期望收益为负。**

## 交易规则

### 时间线（三天一个周期）

```
T日（第1天）- 收盘后
  1. 将盘前纪要HTML放入 data/ 目录
  2. 运行 1_analyze_news.py 分析新闻
  3. 运行 2_process_data.py 下载当日数据
  4. 运行 4_predict_select.py 预测选股
  5. 查看结果：信号为CASH则空仓，否则按推荐买入

T+1日（第2天）- 早盘
  开盘价买入推荐股票，等分仓位

T+2日（第3天）- 收盘前
  14:50-14:55 全部卖出（收盘价）
```

### 选股标准

1. **主板过滤**：仅保留沪市60xxxx.SH和深市00xxxx.SZ
2. **市值过滤**：流通市值 <= 300亿
3. **最低概率阈值**：prob >= 0.6（低于此值空仓）
4. **TS综合评分**：ts_score > 0（非新闻股需通过TS过滤）
5. **新闻正面股优先**：news_stock_impact > 0 的股票优先入选

### 信号等级

| 信号 | 条件 | 含义 | 操作 |
|------|------|------|------|
| STRONG | prob > 0.8 | 强信号，回测胜率>90% | 积极买入 |
| MODERATE | 0.6 <= prob < 0.8 | 中等信号，回测胜率~60% | 可以买入 |
| CASH | prob < 0.6 | 无达标股票 | 空仓等待 |

## 目录结构

```
final_method/
  1_analyze_news.py      # Step1: 分析盘前纪要HTML -> JSON
  2_process_data.py      # Step2: 从Tushare下载当日数据
  3_train_model.py       # Step3: 训练XGBoost模型（已有预训练模型可跳过）
  4_predict_select.py    # Step4: 预测选股 + 空仓判断
  run_all.bat            # 一键运行完整流程
  data/                  # 存放盘前纪要HTML文件
  models/                # 存放训练好的模型
    doubao_t1t2_model.joblib
  news_major1/           # AI分析后的新闻JSON
```

## 每日操作流程

### 方式一：一键运行

```bat
run_all.bat 20260417
```

### 方式二：逐步运行

```bat
# Step1: 将盘前纪要HTML放入data/目录，然后分析
python 1_analyze_news.py

# Step2: 下载当日市场数据（需要Tushare Pro权限）
python 2_process_data.py 20260417

# Step3: 训练模型（已有预训练模型可跳过此步）
python 3_train_model.py

# Step4: 预测选股
python 4_predict_select.py 20260417
```

### 查看结果

结果保存在 `prediction_YYYYMMDD.json`：

```json
{
  "date": "20260417",
  "signal": "CASH",
  "reason": "best_prob < 0.6",
  "picks": []
}
```

或

```json
{
  "date": "20260420",
  "signal": "STRONG",
  "picks": [
    {
      "rank": 1,
      "ts_code": "600666.SH",
      "prob": 0.8521,
      "circ_mv_yi": 177.41,
      "news_stock_impact": 1.0,
      "ts_score": 65.08
    }
  ]
}
```

## 模型说明

- **算法**：XGBoost 二分类
- **标签**：(T+2收盘价 / T+1开盘价 - 1) > 0.04
- **特征**（5个）：
  - `hot_rank_pct`：同花顺热度排名百分位
  - `chip_concentration`：筹码集中度 (cost_85 - cost_15) / cost_50
  - `winner_rate`：获利盘比例
  - `news_market_impact`：市场新闻影响 (-1/0/1)
  - `news_stock_impact`：个股新闻影响 (-1/0/1)
- **训练过滤**：circ_mv <= 100亿，排除688/689（科创板）

## TS综合评分

对非新闻正面股票进行二次过滤：

```
ts_score = -|ret_1d| * 0.5 + delta_winner_rate * 3.0
           -|delta_chip_conc| * 1.5 - |ret_5d| * 0.2
           -|ma5_dist| * 0.3
```

- ts_score > 0：技术面配合，可买入
- ts_score <= 0：技术面不支持，过滤掉
- **新闻正面股豁免TS过滤**：有明确利好消息的股票不受TS限制

## 依赖

- Python 3.9+
- xgboost, joblib, pandas, numpy, tqdm
- tushare（数据下载）
- zhipuai（新闻分析）
- 智谱AI API Key（1_analyze_news.py中使用）
- Tushare Pro Token（2_process_data.py中使用）

## 数据存储

市场数据统一存储在 `D:\iquant_data\data_v2\`：

| 目录 | 内容 |
|------|------|
| data_day1/ | 每日行情（OHLCV） |
| other_day1/ | 市值、换手率等 |
| ths_rank1/ | 同花顺热度排名 |
| cyq1/ | 筹码分布数据 |

## 策略纪律

1. **空仓也是一种操作**：prob < 0.6 时空仓，不强行交易
2. **严格T+1**：T日选股 -> T+1开盘买入 -> T+2收盘卖出
3. **等分仓位**：多只推荐股平均分配资金
4. **不追涨停板**：开盘涨停的股票不买入
5. **模型定期重训**：建议每月重新运行 3_train_model.py

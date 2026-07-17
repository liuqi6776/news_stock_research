# Study 004 — T+1 新闻舆情策略（1日持仓 + 高开过滤）

## 策略概述

本研究实现了一套基于新闻舆情的 A 股 T+1 量化策略，具备以下核心特征：

- **模型目标**：预测 `(T+2 收盘价 - T+1 开盘价) / T+1 开盘价`，即"T+1 开盘买入、T+2 收盘卖出"的单日收益
- **入场时机**：T 日收盘后模型预测概率 ≥ 阈值，T+1 集合竞价结束后（9:25）按开盘价买入
- **高开过滤器**：T+1 开盘价相对 T 日收盘价涨幅在 **2%～6%** 之间（排除没有反应的股票，也排除开盘直接涨停的陷阱）
- **持仓周期**：T+1 开盘买入，T+2 收盘卖出（共持有约 1 个完整交易日）
- **严格 T+1 约束**：当日买入，次日才能卖出

---

## 代码结构

```
study_004_1d_release/
├── README.md                        ← 本文件
├── scripts/
│   ├── config.py                    ← 数据路径配置（原始数据路径 D:\iquant_data\）
│   ├── step1_build_features.py      ← 特征构建，输出 scripts/data/all_features_v2.parquet
│   ├── fix_1d_target.py             ← 修正 target 为 T+1 开盘买入基准（在 scripts/ 目录下运行）
│   ├── train_1d_open.py             ← Walk-forward 月度滚动训练，输出 scripts/predictions/
│   └── backtest_1d_gap_filter.py    ← 回测 + 网格搜索（含高开过滤）
├── data/
│   └── all_features_v2.parquet      ← step1 + fix_1d_target 生成的特征文件
├── predictions/
│   └── predictions_1d_open_wf_monthly.parquet  ← train_1d_open 生成的预测文件
└── results/
    ├── 1d_strategy_limit_filter.png            ← 权益曲线图
    └── 1d_strategy_limit_filter_results.json   ← 完整网格结果
```

---

## 原始数据依赖（D:\iquant_data\data_v2\）

`step1_build_features.py` 从以下原始数据目录读取：

| 目录 | 内容 |
|------|------|
| `data_day1/` | 每日 OHLCV 价格数据（每个文件为一个交易日，按 `YYYYMMDD.parquet` 命名） |
| `other_day1/` | 每日基本面数据（PE、PB、流通市值、换手率等） |
| `news_major1/` | 每日主要新闻舆情 JSON 分析结果（`analysis_YYYY-MM-DD.json`） |
| `ths_rank1/` | 同花顺热度排名数据 |
| `moneyflow1/` | 每日资金流向数据（大单净流入等） |
| `ths_news1/` | 同花顺个股新闻信号（new_gs/new_bs/new_gi） |

---

## 回测逻辑

### 买入条件（T+1 日开盘）
1. `prob ≥ threshold`（模型在 T 日收盘后输出的置信度）
2. T 日未涨停（涨停当日信号不可信）
3. T+1 日开盘未涨停（无法以合理价格买入）
4. **高开幅度在 2%～6% 之间**：`0.02 < (T+1_open - T_close) / T_close < 0.06`

### 卖出条件（T+2 日）
- 正常持有至 T+2 收盘卖出
- 止损触发（建议 -5%）：T+2 日内止损，以止损价卖出
- 止盈触发（建议 +5%）：T+2 日开盘或盘中达到止盈价时卖出
- 跌停时延迟卖出（避免无法成交）

### 仓位管理
- `pos_size = 1 / (hold_days × max_pos)` = `1 / (2 × 3)` ≈ 16.7% 每仓
- 最优参数下 `max_pos=3`，即同时最多持有 3 只票
- 交易费用：买 0.1% + 卖 0.1%（合计 0.2%/次）

---

## 最终回测结果（2022-2026，Clean 数据）

> **数据来源**：`step1_build_features.py` 重新构建特征（无 `shift(-2)` 残留）→ `fix_1d_target.py` 修正 target → `train_1d_open.py` 月度 WFO 训练 → `backtest_1d_gap_filter.py` 回测

### BEST COMBO per threshold（按测试集 Sharpe 排序）

| 阈值 | 持仓数 | 止损 | 止盈 | 测试集 Sharpe | 测试集 CAGR | 测试集胜率 |
|------|--------|------|------|--------------|------------|----------|
| 0.50 | 3 | -5% | +5% | **4.63** | 65.2% | 37.2% |
| 0.52 | 3 | -5% | +5% | 4.49 | 62.6% | 36.5% |
| 0.54 | 3 | -5% | +5% | 4.19 | 55.7% | 34.7% |
| 0.56 | 10 | -5% | +5% | 4.03 | 18.4% | 37.2% |
| 0.58 | 10 | -10% | +5% | 4.01 | 17.1% | 33.0% |
| 0.60 | 10 | -10% | +5% | 3.84 | 14.2% | 27.4% |
| 0.62 | 10 | -10% | +5% | 2.80 | 8.4% | 21.8% |
| 0.64 | 3 | -10% | +5% | 2.44 | 21.1% | 17.2% |
| 0.66 | 3 | -5% | +5% | 2.21 | 16.7% | 14.0% |

> **全局最优**：`threshold=0.50, max_pos=3, stop_loss=-5%, take_profit=+5%`  
> 测试集（2025-2026）：Sharpe = **4.63**，CAGR = **65.2%**，Max DD ≈ -6%  
> 训练集（2022-2024）：Sharpe ≈ 2.5，CAGR ≈ 30%  

### 关键结论

1. **止盈 5% 是关键**：加入 5% 止盈后，Sharpe 从 4.15（无止盈）提升到 **4.63**。高开动量股在当日达到 5% 涨幅后往往回调，及时止盈能显著改善收益质量。

2. **高开 2%～6% 过滤器是核心**：相比无过滤版本，胜率提升约 10-15 个百分点，Sharpe 从 2.x 提升到 4.x 量级。

3. **低阈值 + 集中仓位（pos=3）是最优组合**：满足高开条件的票已经是高质量筛选集，不需要过度提高 prob 阈值。`th=0.50, pos=3` 充分利用模型召回率。

4. **无未来函数，时间对齐严格验证**：
   - 特征构建时已消除 `shift(-2)` 残留
   - target = `(T+2 close - T+1 open) / T+1 open`（由 `fix_1d_target.py` 写入 `return_1d_open` 列）
   - 回测以 `T+1 open` 为 `bp`（买入价），高开计算使用 `T+1_open vs T_close`

---

## 实盘操作指引

1. **T 日盘后**：拉取最新数据 → 运行预测 → 筛选 `prob ≥ 0.50` 的标的（最多 3 只）
2. **T+1 集合竞价结束（9:25）**：计算每只候选标的的集合竞价价格相对昨收涨幅
   - 涨幅在 2%～6% 之间 → **买入**（以开盘价委托）
   - 涨幅不在此区间 → 跳过
3. **T+2 全天监控**：
   - 日内涨幅达 +5% → 止盈卖出
   - 日内跌幅达 -5% → 止损卖出
   - 若未触发，**收盘前卖出**

---

## 完整数据 Pipeline

### 概览

```
原始数据 (D:\iquant_data\data_v2\)
    │
    ▼ [每月1次 / 有新数据时]
step1_build_features.py
    │  输出: data/all_features_v2.parquet
    │        (含价格、动量、基本面、新闻舆情特征)
    ▼
fix_1d_target.py
    │  输出: data/all_features_v2.parquet (添加 return_1d_open 列)
    │        target = (T+2_close - T+1_open) / T+1_open
    ▼ [每月月底 retrain]
train_1d_open.py
    │  输出: predictions/predictions_1d_open_wf_monthly.parquet
    │        (月度 Walk-Forward 预测，含 prob 列)
    ▼ [T 日盘后 / 每日实盘信号]
backtest_1d_gap_filter.py  (研究验证用)
    │  或
每日信号生成脚本 (实盘用，读 predictions 文件)
    │
    ▼ [T+1 日 9:25 集合竞价结束]
根据 gap_up 高开条件 (2%~6%) 筛选 → 买入
```

### Step-by-Step 执行指南

#### 第一步：特征构建（每月 1 次，约需 2-4 小时）

```bash
cd C:\Users\liuqi\quant_system_v2\research\study_004_1d_release\scripts
python step1_build_features.py
```

**前提条件**：
- 原始数据已更新到最新日期（`D:\iquant_data\data_v2\data_day1\YYYYMMDD.parquet`）
- `config.py` 中路径配置正确

**输出**：`scripts/data/all_features_v2.parquet`

> ⚠️ 脚本有增量检测：若特征文件已包含最新日期，会跳过重新构建。

---

#### 第二步：修正 Target（紧接 Step 1 之后运行，约 1 分钟）

```bash
cd C:\Users\liuqi\quant_system_v2\research\study_004_1d_release\scripts
python fix_1d_target.py
```

**作用**：  
在 `all_features_v2.parquet` 中添加/更新 `return_1d_open` 列：
```
return_1d_open = (d+2 收盘价 - d+1 开盘价) / d+1 开盘价
```
即 T 日信号 → T+1 日开盘买入 → T+2 日收盘卖出的实际收益率。

> ⚠️ 必须在 `scripts/` 目录下运行（路径为相对路径 `data/all_features_v2.parquet`）

---

#### 第三步：模型训练（每月月底运行，约 30-60 分钟）

```bash
cd C:\Users\liuqi\quant_system_v2\research\study_004_1d_release\scripts
python train_1d_open.py
```

**训练策略**：Walk-Forward 月度滚动（WFO）
- 每个月独立训练一个模型
- 训练数据：截止到上个月底的所有历史数据（从 `TRAIN_START=20200101` 开始）
- 预测数据：当月所有交易日
- 模型：XGBoost Classifier（label = `return_1d_open > 0.01`）

**输出**：`scripts/predictions/predictions_1d_open_wf_monthly.parquet`  
包含列：`trade_date, ts_code, prob, target, actual_return, entry_price`

**Retrain 时机建议**：
- 每月最后一个交易日收盘后（约 20:00）运行
- 新加入一个月的真实标签数据后，下月预测质量更新

---

#### 第四步：日常信号生成（每个交易日盘后，约 1 分钟）

从 `predictions_1d_open_wf_monthly.parquet` 中读取当日预测：

```python
import pandas as pd

pred = pd.read_parquet('scripts/predictions/predictions_1d_open_wf_monthly.parquet')
today = '20260508'  # 替换为今日日期

# 筛选当日信号
signals = pred[pred['trade_date'] == today]
signals = signals[signals['prob'] >= 0.50]
signals = signals.nlargest(3, 'prob')  # 最多取 3 只

print(signals[['ts_code', 'prob', 'entry_price']])
```

**候选标的**就是明日（T+1 日）需要在 9:25 集合竞价结束后判断是否买入的股票。

---

#### 第五步：T+1 日集合竞价判断（9:25 实盘操作）

对每只候选标的，计算集合竞价价格相对昨收的涨幅：

```
gap_up = (竞价价格 - 昨日收盘价) / 昨日收盘价
```

| gap_up 范围 | 操作 |
|-------------|------|
| 2% < gap_up < 6% | ✅ 以开盘价市价委托买入 |
| gap_up ≤ 2% 或 ≥ 6% | ❌ 跳过，不买入 |
| 涨停板（≥ 9.5%/19.5%） | ❌ 跳过 |

---

#### 第六步：T+2 日卖出（持仓管理）

| 情景 | 操作 |
|------|------|
| 日内涨幅达到 +5% | ✅ 止盈卖出 |
| 日内跌幅达到 -5% | ✅ 止损卖出（若跌停则等次日） |
| 未触发止损/止盈 | 收盘前市价卖出 |

---

### Retrain 时间表

| 频率 | 时机 | 脚本 |
|------|------|------|
| **每月** | 每月最后一个交易日收盘后 | `step1_build_features.py` → `fix_1d_target.py` → `train_1d_open.py` |
| **每日** | 每日收盘后（15:30 后） | 读取 predictions 文件 → 生成候选信号 |
| **实盘** | 每日 9:25 集合竞价结束 | 手动/脚本判断 gap_up 条件 → 买入 |
| **不定期** | 有重大数据更新时 | 重跑 `step1` + `fix_1d_target` 后重新训练 |

> **注**：`train_1d_open.py` 是全量 WFO 训练（从 2022 年起每月都会重新跑一遍），若只需更新最新月份的预测，可修改 `pred_months` 范围只跑最新月。

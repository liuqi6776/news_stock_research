# 量化交易系统 V2

基于 Python 的 A股量化交易系统，重构自 quant_trading_system。

## 功能特性

- 📈 **数据获取**: 从 Tushare 获取 A股行情、技术指标、筹码分布等数据
- 📊 **特征工程**: Alpha 101因子、技术指标、筹码分析
- 🎯 **机器学习**: XGBoost 模型训练与预测
- 🔄 **回测系统**: 策略验证与绩效分析

## 项目结构

```
quant_system_v2/
├── config/         # 配置管理
├── data/           # 数据获取与存储
├── features/       # 特征工程
├── processing/     # 数据处理
├── models/         # 模型训练与预测
├── backtesting/    # 回测引擎
├── utils/          # 工具函数
└── main.py         # 主入口
```

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置 API

创建 `.env` 文件或设置环境变量：

```bash
TUSHARE_TOKEN=your_token_here
TUSHARE_API_URL=http://tsapi.majors.ltd:7000
DATA_PATH=D:/iquant_data/data_v2
```

### 3. 运行

```python
from data.fetcher import DataFetcher
from processing.pipeline import DataPipeline
from models.trainer import ModelTrainer

# 获取数据
fetcher = DataFetcher()
fetcher.fetch_all(end_date="20251118")

# 处理数据
pipeline = DataPipeline()
df = pipeline.run(start_date="20200103", end_date="20251118")

# 训练模型
trainer = ModelTrainer()
trainer.train(df)
```

## 依赖

- Python 3.7+
- pandas, numpy
- tushare
- xgboost, scikit-learn
- joblib, tqdm

## 🛡️ A股期权增强型全局策略 (Study 005) 核心审计与回测指引

本系统已经完全剔除了任何形式的 T+0 盘中未来函数、收盘价窥探偏误与交易顺序执行偏误，采用**物理级真实 A股 T+1 跨日交易清算规则**与**最保守 Worst-Case 盘中清算假设**进行回测，非常适合中大型资金作为低波动稳健增值底仓。

---

### 1. 核心回测交易逻辑（审计重点）

在对代码进行审计时，请重点关注以下物理级交易清算机制（已在回测引擎中完全同步）：

* **T+1 日买入（Entry）**：
  * **买入时点**：T日收盘后利用 $T$ 日及以前的特征进行 Walk-Forward 滚动预测。**T+1 日 9:30 开盘时**，以 **Open（开盘价）** 价格买入成交。
  * **涨停规避**：若 T+1 日开盘直接一字涨停，系统判定为无法买入，自动放弃该笔交易，避免追高受损。
* **T+1 日强制锁仓（Lock）**：
  * 严格遵循 **A股 T+1 交易制度**。T+1 日买入后，盘中及收盘**绝对禁止卖出**，强行承担持股过夜（Overnight）的跳空及波动风险。
* **T+2 日出场（Exit & Risk Control）**：
  * **固定止盈 (+6%)**：T+2 盘中若最高价 $High \ge BuyPrice \times 1.06$，以 **$BuyPrice \times 1.06$** 的价格挂单止盈离场。
  * **固定止损 (-5%)**：T+2 盘中若最低价 $Low \le BuyPrice \times 0.95$，以 **$BuyPrice \times 0.95$** 的价格挂单止损离场。
  * **开盘跳空止损**：若 T+2 开盘价直接跳空低开 $Open \le BuyPrice \times 0.95$，直接以 **Open 价** 止损离场，真实计入跳空大面。
  * **⭐ 最保守判定（Worst-Case Execution，根除未来函数）**：如果在 T+2 当天盘中，股价震荡剧烈，**既触及了 +6% 止盈线，又跌破了 -5% 止损线**，回测引擎**强制假设先发生止损**，该笔交易以 **`-5% 止损`** 结算！
  * **收盘强制平仓**：若 T+2 盘中均未触发止盈和止损，在 **14:50 以 Close（收盘价）** 强制卖出平仓，绝不拖泥带水，控制持股周期为严格的 2 个交易日。
  * **跌停锁仓避险**：若 T+2 日开盘即一字跌停且全天未打开，当天无法成交，持仓自动顺延滚存至下一个能卖出的交易日。

---

### 2. GitHub 目录结构与关键代码文件映射

您可以在 GitHub 仓库对应的目录下找到所有的核心研究、回测及实盘代码：

* **📄 [STUDY_005_SUMMARY.md](STUDY_005_SUMMARY.md)**：**【核心指南】全策略的数学原理、数据统计与实盘部署详尽文档。**
* **📂 [research/期权/](research/期权/)**：期权 Stacking 与全局对比回测（主审计区）：
  * **[build_features_with_options.py](research/期权/build_features_with_options.py)**：7大期权特征预处理与高速合并广播。
  * **[train_models_with_options.py](research/期权/train_models_with_options.py)**：Walk-Forward 滚动 XGBoost 双模型训练。
  * **[backtest_options_model.py](research/期权/backtest_options_model.py)**：**【核心回测代码】 Baseline vs Option-Enhanced 严格对账回测器**。
  * **[download_all_pcr.py](research/期权/download_all_pcr.py)**：历史 PCR 数据下载工具。
  * **[historical_pcr.csv](research/期权/data/historical_pcr.csv)**：期权 PCR 轻量基准数据库。
  * **[model_options_comparison.png](research/期权/results/model_options_comparison.png)**：**【回测结果图】全周期净值与回撤对比曲线**。
* **📂 [research/study_005_1d_advanced/](research/study_005_1d_advanced/)**：Study 005 进阶版 Baseline 模型研究线：
  * **[step3_backtest_advanced.py](research/study_005_1d_advanced/scripts/step3_backtest_advanced.py)**：**【核心回测代码】Baseline 进阶版严格 T+1 回测器**。
  * **[005_advanced_results.png](research/study_005_1d_advanced/results/005_advanced_results.png)**：**【回测结果图】Baseline 进阶版回撤热力图与净值图**。
* **⏰ 全自动定时晨报与调度器及实盘交易端：**
  * **[ptrade_client_v5.py](ptrade_client_v5.py)**：**【核心实盘代码】PTrade 恒生柜台全自动交易主程序（已解决账户自动对账、流动性 ADV 1% 过滤、ST/次新内容审计三大实盘风险）**。
  * **[daily_morning_pipeline.py](daily_morning_pipeline.py)**：每日 8:00 NLP 新闻打分与选股信号邮件自动推送主程序。
  * **[run_morning_pipeline.bat](run_morning_pipeline.bat)**：Windows 计划任务专用定时启动批处理与日志重定向脚本。
  * **[run_retrain_with_options.py](run_retrain_with_options.py)**：一键式“同步->特征重构->双模型重训”主调度脚本。
  * **[api_server.py](api_server.py)**：本地量化 FastAPI 服务器（含板块超限过滤及双模型防火墙）。

---

### 3. 建议审计路线 (Suggested Audit Path)

如果您想要系统性审计本量化系统，我们推荐以下路线：

1. **第一步：阅读原理与最新数据验证** —— 阅读 **[STUDY_005_SUMMARY.md](STUDY_005_SUMMARY.md)**。查看大盘期权隐含波动率（QVIX Z-Score）与 PCR 情绪大闸的计算逻辑，以及近两个月（2026.04 - 2026.05）的真实期权透视数据。
2. **第二步：审计回测引擎平仓逻辑** —— 打开 **[backtest_options_model.py](research/期权/backtest_options_model.py) (第 180 - 233 行)**。重点核验 `hold2` 卖出逻辑中如何进行开盘止损判定、Worst-Case 日内多空冲突判定、以及一字跌停锁仓顺延，验证有无任何“未来收盘价”的窥探。
3. **第三步：核验期权超额 Alpha 的真实性** —— 对照 **[model_options_comparison.png](research/期权/results/model_options_comparison.png)** 中的 Baseline 曲线与期权增强型曲线。在剥离所有未来函数后，验证期权特征使全周期（2022-2026）夏普比率从 **2.33 提升至 3.08**，最大回撤锁定在 **-8.3%** 的卓越表现。
4. **第四步：核验实盘风控与部署逻辑** —— 打开 **[api_server.py](api_server.py) (第 116 - 128 行)**。查看 API 接口层如何实施 `prob_crash <= 15%` 双模型熔断，以及如何执行单行业板块最多推荐 2 只股票的行业中性化限制，确保实盘交易的资金安全性。
5. **第五步：审计实盘全自动交易系统** —— 打开 **[ptrade_client_v5.py](ptrade_client_v5.py)**。核验 `before_market_start` 盘前对账状态机与 ST/次新审计防线、`execute_morning_buy` 开盘一字涨停避险、以及 `intraday_risk_control` 盘中逐 Bar 硬性止盈/止损出场机制，验证其实盘部署的安全性。

---

## 📄 License

MIT

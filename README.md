# 新闻情感分析与股票价格预测研究

> **研究主题**: 中国A股市场新闻情感对个股及大盘短期走势的影响研究
> **数据范围**: 2024-10 至 2025-02 | 132,998条原始财经新闻 | 3,078,880条价格数据
> **研究方法**: 从简单情感分析 → LLM精细情感 → 学术最佳实践（Tetlock/Zhang & Skiena/Ding/宋莹）
> **研究者**: Kimi 2.6 (Moonshot AI) | 2026-06-28

---

## 📁 目录结构

```
news_stock_research/
├── README.md                              # 本文档
├── reports/                               # 研究报告
│   ├── news_academic_best_practice_report.md   # 学术最佳实践研究报告（最新）
│   ├── news_llm_longwindow_report.md           # LLM精细情感+长窗口分析
│   ├── news_stock_research_report.md           # 个股层面研究报告
│   ├── news_raw_impact_report.md             # 原始新闻市场层面报告
│   └── news_impact_research_report.md          # 处理后JSON影响分析（旧版）
├── data/                                  # 数据集（Git LFS管理）
│   ├── news_academic_full.csv             # 7,075条 学术情感+T+1/2/3/5/10收益
│   ├── news_academic_sentiment.csv        # 132,998条 原始新闻学术情感分析
│   ├── news_llm_multiwindow.csv           # 7,075条 LLM情感+多窗口收益
│   ├── news_stock_analysis.csv          # 35,638条 个股新闻-价格关联
│   ├── news_market_sentiment.csv        # 803天 市场情感与大盘
│   ├── news_price_merged.csv            # 5,590条 处理后JSON影响-价格关联
│   ├── news_raw_decoded.csv             # 132,998条 原始新闻解码
│   ├── price_2024_2026.csv            # 3,078,880条 2024-2026价格数据
│   ├── daily_ls_strategy.csv          # 390条 每日多空策略收益
│   ├── all_news_stocks.csv            # 5,978条 全部个股新闻记录
│   ├── all_news_sectors.csv           # 4,762条 全部板块新闻记录
│   └── all_news_market.csv            # 803条 全部市场新闻记录
├── scripts/                               # 分析脚本
│   ├── script_api_llm_sentiment.py      # 方案A: 调用GPT-4/Claude API批量处理
│   ├── script_k2_6_upgraded_rules.py    # 方案B: K2.6升级版规则模型
│   └── check_news_raw.py                # 原始新闻检查脚本
└── samples/                               # 样本数据
    └── news_50_sample_for_analysis.txt    # 50条新闻样本（K2.6分析用）
```

---

## 📊 核心发现速览

### 1. 五种方法对比

| 方法 | T+1相关系数 | T+2 | T+3 | 多空策略(T+3) | 优势 |
|------|------------|------|------|-------------|------|
| **简单情感** | 0.038 | — | — | — | 简单快速 |
| **LLM精细情感** | 0.047 | **0.059** | 0.058 | — | 否定词+程度词+长窗口 |
| **学术Net情感** | 0.030 | 0.032 | 0.034 | **+0.925%** | 事件分类+负面加权+多空 |
| **学术负面情感** | **-0.041** | **-0.043** | **-0.046** | — | 负面预测力更强(Tetlock) |
| **处理后JSON impact** | -0.011 | — | — | — | ⚠️ 基本无效 |

### 2. 关键结论

- **T+2是最佳预测窗口**（不是T+1！），新闻影响需要1-2天才能完全释放
- **负面情感的预测力比正面强1.5倍**（Tetlock 2007 confirmed）
- **学术多空策略T+3收益+0.925%**，Sharpe 0.419
- **DOS（情感分歧度）预测波动而非方向**：高DOS → 高波动（T+1: 1.97% → 4.63%）
- **部分利空新闻T+10暴涨**：如中国长城+94.8%（利空出尽是利好）

---

## 📁 数据说明

### 原始新闻数据
- **来源**: `D:/iquant_data/data_v2/news_raw_data` (Parquet格式)
- **字段**: datetime, title, content
- **编码**: UTF-8（原GBK字节被错误读取，已修复）
- **覆盖**: 132,998条，2024-10-11 至 2025-02-21

### 价格数据
- **来源**: `D:/iquant_data/data_v2/data_day1` (Parquet格式)
- **字段**: trade_date, ts_code, open, close, high, low, pct_chg
- **覆盖**: 3,078,880行，2024-2026年，A股全部个股
- **指数**: 000001.SZ (上证指数), 000001.SH (上证指数SH)

### 处理后JSON（对比用）
- **来源**: `D:/iquant_data/data_v2/news_major1` (JSON格式)
- **内容**: 被某大模型处理后的新闻，包含股票代码、板块、impact分数
- **问题**: impact分数与价格表现几乎无相关性（相关系数-0.011）

---

## 🔬 研究方法演进

### 阶段1: 简单情感分析（旧版）
- 利好词+1，利空词-1，中性0
- 局限: 无事件区分、无强度、无否定反转

### 阶段2: LLM精细情感
- 引入否定词反转（不+涨=利空）、程度词（大幅+上涨=强利好）、转折句（虽然...但是=降权）
- 长窗口分析：T+1/2/3/5/10
- 相关系数提升22%（0.038 → 0.047）

### 阶段3: 学术最佳实践（最终版）
基于以下经典研究：
- **Tetlock (2007)**: 负面情绪预测力更强，市场反应不足
- **Zhang & Skiena (2010)**: 基于排名的多空策略，年回报30%
- **Ding (2015)**: 结构化事件提取，事件类型影响乘数
- **宋莹 (2021)**: 情感分歧度(DOS)模型，预测波动而非方向

---

## 🚀 使用方法

### 方案A: 调用LLM API（推荐）
```bash
export OPENAI_API_KEY="sk-..."
export OPENAI_API_BASE="https://api.openai.com/v1"
export LLM_MODEL="gpt-4o-mini"
python scripts/script_api_llm_sentiment.py
```

### 方案B: 升级版规则模型（离线）
```python
from scripts.script_k2_6_upgraded_rules import K2_6_NewsAnalyzer

analyzer = K2_6_NewsAnalyzer()
result = analyzer.analyze(title, content)
# 返回: {score, event_type, reason, confidence}
```

---

## 📚 研究报告索引

1. **news_academic_best_practice_report.md** — 学术最佳实践（最新最完整）
2. **news_llm_longwindow_report.md** — LLM精细情感+长窗口
3. **news_stock_research_report.md** — 个股层面分析
4. **news_raw_impact_report.md** — 原始新闻市场层面
5. **news_impact_research_report.md** — 处理后JSON分析（旧版，对照用）

---

## 📝 引用文献

1. Tetlock, P. C. (2007). "Giving Content to Investor Sentiment: The Role of Media in the Stock Market." *Journal of Finance*, 62(3), 1139-1168.
2. Zhang, X., & Skiena, S. (2010). "Trading Strategies to Exploit Blog and News Sentiment." *ICWSM*.
3. Ding, X. et al. (2015). "Deep Learning for Event-Driven Stock Prediction." *IJCAI*.
4. 宋莹, 张维, 等 (2021). "基于V-A情感分歧度模型的股价预测研究." *管理科学学报*.
5. Bollen, J., Mao, H., & Zeng, X. (2011). "Twitter mood predicts the stock market." *Journal of Computational Science*, 2(1), 1-8.

---

## ⚠️ 数据限制

- 原始新闻中股票代码极少（主要用公司名称），公司名匹配精度有限
- 新闻数据截止2025-02-21，价格数据截止2026-06-17
- 部分新闻来源为AI生成，需人工验证
- 研究结论不构成投资建议，仅供学术参考

---

## 📧 联系

如有问题或建议，请通过 GitHub Issues 反馈。

---

*Last Updated: 2026-06-28*
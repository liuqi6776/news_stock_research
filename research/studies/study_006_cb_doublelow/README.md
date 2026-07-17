# Study 006: Convertible Bond Double-Low Rotation Strategy / Study 006：可转债双低轮动策略

This directory contains the source code and research pipeline for the **Convertible Bond Double-Low Rotation Strategy**.
本目录包含**可转债双低轮动策略**的全部源代码及研究管道。

---

## 1. Strategy Logic / 策略逻辑

The strategy ranks all active convertible bonds daily using the **Double-Low (双低)** score:
策略每日计算所有上市交易的可转债的**双低**值，并进行升序排列：

$$\text{Double-Low Score} = \text{Bond Close Price} + \text{Conversion Premium Rate} \times 100$$
$$\text{双低值} = \text{转债价格} + \text{转股溢价率} \times 100$$

- A low bond price provides a strong bond floor (debt protection).
- A low conversion premium rate ensures equity-like upside correlation with the underlying stock.
- The strategy rotates into the top $N$ lowest double-low bonds on a regular frequency (weekly, biweekly, or monthly).
- 低转债价格提供了极佳的债底保护；低转股溢价率则确保了转债价格跟正股上涨的敏感性。
- 策略定期（每周、双周或每月）选择双低值最低的 $N$ 只可转债进行等权轮动。

---

## 2. File Directory Structure / 文件结构

- **`cb_data_downloader.py`**: Fetches raw data from Eastmoney APIs, resolves ratings/scales, and compiles the point-in-time long format cache `cb_pit_daily.parquet`.
- **`backtest_cb_doublelow.py`**: Implements the strict backtest loop with $T+1$ execution, transaction costs, and delisting/redemption handling.
- **`run_all.py`**: Grid search orchestrator comparing baseline vs robust filtered strategies and plotting equity curves.
- **`cb_data_downloader.py`**：从东方财富接口下载转债原始历史估值，对齐评级与规模，并编译时间点快照缓存 `cb_pit_daily.parquet`。
- **`backtest_cb_doublelow.py`**：核心回测引擎，支持 $T+1$ 偏离度交易执行、滑点扣费、违约退市与强制赎回（强赎）平仓模拟。
- **`run_all.py`**：参数网格搜索主控脚本，对比基准与稳健过滤的长期收益，并输出对比图表。

---

## 3. Strict Execution & Friction Modeling / 严格回测与交易摩擦建模

To prevent look-ahead bias and represent real trading conditions:
为了消除前瞻偏差（未来函数）并真实体现交易摩擦：
1. **$T+1$ Timing**: Signals are generated using day $T$ close prices. Trades are executed at day $T+1$ close prices. There is NO same-day pricing execution.
2. **Transaction Costs**: We model a single-side friction of **0.05%** (covering commission + slippage/market impact), which translates to **0.10%** double-side cost.
3. **Delisting & Redemption**: When a bond matures, is forced to redeem (强赎), or defaults, we liquidate it at its last valid trading close price.
4. **T+1 交易时序**：在 $T$ 日收盘后基于截面数据生成调仓信号，在 $T+1$ 日以收盘价执行。杜绝了同一天收盘价计算并买入的“未来函数”。
5. **交易成本**：单边扣减 **0.05%** 的摩擦成本（佣金 + 滑点冲击），即一个完整的买卖扣减 **0.10%** 费用。
6. **退市与强赎处理**：转债到期、触发强赎或面临信用违约退市时，策略将在最后一个可交易日的收盘价强制卖出变现，避免幸存者偏差。

---

## 4. How to Run / 如何运行

Ensure you have dependencies installed (pandas, numpy, akshare, pyarrow, matplotlib). Run the main entry:
确保已安装必要依赖（pandas, numpy, akshare, pyarrow, matplotlib），然后在终端中执行主入口：

```bash
# Execute downloader and backtests
python research/studies/study_006_cb_doublelow/run_all.py
```

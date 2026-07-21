# 可转债风险评分卡与组合净值熔断机制开发计划 (Route B Enhancement)

本计划旨在为 200 万资产配置组合中的 70 万“转债双低策略”设计一套完整的“自动驾驶与风控系统”，以确保组合整体最大回撤严格控制在 **20%** 限制之内。

---

## 拟引入的核心模块与设计

### 1. 转债风险评分卡 (cb_risk_scorecard.py) [NEW]

通过构建客观的宏观风险监测指标，每日输出转债市场的风向标信号（**红灯 0% 仓位** / **黄灯 50% 仓位** / **绿灯 100% 仓位**）：

* **输入特征设计**：
  1. **市场双底均值 (Market DL Mean)**：衡量可转债市场的整体估值性价比。
  2. **双底均值20日动量 (DL Mean Momentum)**：计算双底均值过去 20 交易日的变化速率，识别估值快速恶化或崩塌的势头。
  3. **持仓溢价差 (Selection Spread)**：策略所选 Top 20 组合的平均双低分数与市场中位数的差值。当差值收窄甚至为正时，说明市场低估券已被买光，策略不得不买入高价/高溢价券。
  4. **信用利差代理指标 (Credit Spread Proxy)**：
     $$\text{Credit Spread Proxy} = \text{Mean(YTM of AA- bonds)} - \text{Mean(YTM of AAA bonds)}$$
     利用数据池中低评级 (AA-) 与高评级 (AAA) 债的到期收益率 (YTM) 之差，捕捉市场信用违约危机与流动性收紧风险（如 2024 年信用违约危机）。

* **信号划分阈值**：
  * **红灯 (Red Light - 0% 仓位)**：
    * 市场双底均值 $> 135$（市场极度泡沫）；
    * 或信用利差代理指标高于历史 90% 分位数（发生实质性违约恐慌）；
    * 或双底均值动量快速下跌（发生流动性踩踏）。
  * **黄灯 (Yellow Light - 50% 仓位)**：
    * 市场双底均值 $\in [120, 135]$；
    * 或信用利差有所扩大，动量偏负。
  * **绿灯 (Green Light - 100% 仓位)**：
    * 市场双底均值 $< 120$（进入极具安全边际的安全区），且信用利差稳定。

---

### 2. 组合净值 15% 熔断风控 (Portfolio Melt Control)

为了绝对防范“模型未覆盖的黑天鹅事件”，我们在回测系统底层嵌入净值监控：
* **熔断规则**：自组合建仓期（或滚动高点）起，若转债组合的累计净值 (NAV) **从历史最高收盘点回撤达到 15%**，则无视评分卡任何信号，**强制清仓所有转债并转为货币基金 (现金)**。
* **冷静期 (Cooldown Period)**：熔断触发后，策略强制空仓至少 60 个交易日，直至市场企稳。
* **目的**：确保这 35% 仓位的转债资产在最极端的违约或踩踏下，对 200 万总组合的净值最大回撤贡献不超过 **-5.25%**（$35\% \times 15\%$），死守总账户 -20% 的终极红线。

---

## 涉及修改与新增的文件

### 1. [NEW] [cb_risk_scorecard.py](file:///c:/Users/liuqi/quant_system_v2/research/studies/study_006_cb_doublelow/cb_risk_scorecard.py)
用于读取 `cb_pit_daily.parquet`，计算上述四项宏观指标，每日输出 Green/Yellow/Red 仓位控制信号。

### 2. [NEW] [backtest_cb_scorecard.py](file:///c:/Users/liuqi/quant_system_v2/research/studies/study_006_cb_doublelow/backtest_cb_scorecard.py)
基于新的评分卡信号与 15% 净值熔断规则，重写/扩展回测逻辑。

### 3. [MODIFY] [run_all.py](file:///c:/Users/liuqi/quant_system_v2/research/studies/study_006_cb_doublelow/run_all.py)
调用新的回测脚本，加入新策略（Scorecard + Melt）的对比测试，并输出全新曲线图。

---

## 验证计划 (Verification Plan)

### 自动化测试
* 运行新回测并比对以下三组方案（2018 - 2026年全区间）：
  1. **Baseline Double-Low**（基准双低）
  2. **Robust Multi-Factor**（原稳健多因子版）
  3. **Scorecard & Melt Managed**（新开发：评分卡仓位管理 + 净值 15% 熔断版）
* 输出新对比图表 `cb_scorecard_comparison.png`，保存至 `results/` 并同步推送 GitHub。

### 压力测试指标
* 验证在 2018 年熊市底部及 2024 年量化/信用债危机期间，新策略是否成功触发“红灯/黄灯”或“15% 净值熔断”，并确认最终最大回撤是否严格小于 15%。

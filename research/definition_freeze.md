# 研究定义冻结（Definition Freeze）

> 对应外部评审 P2「冻结定义 / 方向 / 组合方式文档」。
> 本文件是策略与因子**当前冻结口径的唯一权威来源**；任何参数/口径变更必须先改本文件，
> 再同步 `quant_conclusion/` 结论文档与最小可复现实验，禁止只改代码不更新本文档。
> 冻结日期: 2026-08-06 | 数据快照: `data_20260806`（数据漂移检测经 `check_data_manifest` 阻断）| 上游: `research/factor_dic/run_validation.py`

---

## 一、因子定义与方向（冻结）

| 因子 | 定义（冻结） | 方向（高值=好） | 权威实现 |
|---|---|---|---|
| ret_1m | 过去 20 日累计收益取负（1 月反转），`-(prod(1+r)-1)`，min_periods=10 | 负（低收益→高未来收益） | `combo_backtest.build_price_factors` |
| ivol | 日收益过去 20 日 std 取负（低特质波动），min_periods=10 | 负（低波动→高未来收益） | `combo_backtest.build_price_factors` |
| turnover_vol_20 | 换手率（成交量/流通股本）20 日滚动 std | 正（高换手波动→高未来收益） | `factor_lib.turnover_vol_20`（方向已统一） |
| VAL | 合成价值：pe_ttm/pb/ps_ttm/dv_ttm 截面分位数合成（PIT 估值快照 ≤ 调仓日） | 正 | `style_factors.build_factors` |
| roe | 季频 ROE（PIT，报告期 ≤ 调仓日，无前视） | 正 | `combo_backtest` roe PIT |

未来收益口径（冻结）: 调仓日 T 之后 20 个交易日累计收益（T+1~T+20），`cum[t+20]/cum[t]-1`，
不含当日；日收益含复权（`data_day1.pct_chg`）。

> ⚠️ 方向冻结依据 `experiment.yaml` 与上游常量一致校验（`run.py` 启动时断言）。

## 二、组合构建（冻结）

| 项目 | 冻结值 | 说明 |
|---|---|---|
| 股票池 | 中证1000 成分股 | PIT 指数权重快照（`iw_YYYYMMDD.parquet`，≤ 调仓日最近一期） |
| 调仓频率 | 月末最后交易日 | 持有期 = rb_next 当日含、rb 当日不含；**生产前向 `daily_signal.py` 为滚动日频调仓**（rb=最新交易日、含最新不去尾，有意设计，2026-08-06 所有者确认） |
| 选股 | 截面 z-score（winsorize 后 `(x-mean)/std`）因子均值排名 Top60 | 生产版含 VAL；因子验证版按单因子排名 |
| 权重 | 等权（验证）/ IVW120（生产：`w_i ∝ 1/σ_i`，σ 为调仓日前 120 日日收益 std） | 生产版再叠加阶段4 可交易过滤 + 阶段5 集中度约束 |
| 成本 | 固定双边 20bps/期（回测）；换手驱动模型见 P2-r5 敏感性 | `net = gross - 20bps` |
| 基准 | 000852.SH（中证1000 指数）；择时弱段持 512100 ETF | `index_daily/*.parquet` |
| 择时 | RS12: 000852/000300 过去 240 日相对强度 >0 才持股，否则持 512100 ETF | 生产版 `daily_signal.py` |
| 信号时点 | 调仓日收盘生成（T-1 信息），下一交易日开盘执行 | 无前视；生产前向 signal_date=最新交易日、execution_date=下一交易日 |

生产版与验证版的差异（冻结）: 验证（`run_validation.run_fast`）= 单因子 Top60 等权 + 固定 20bps；
生产（`daily_signal.py`）= 4 因子合成 + IVW120 + 可交易过滤 + 集中度约束 + RS12。

## 三、统计检验口径（冻结）

- Rank IC（winsorize 后因子值秩 vs 未来 20 日收益秩，Spearman）
- Newey-West HAC t（lag=4）；多重检验: BH-FDR / Bonferroni（N=21 因子族）
- DSR（Bailey-López de Prado 2014，偏度/峰度 + N 档敏感性）；PBO（CSCV，随机 1000 切分，seed=42）
- 回撤: 相对口径 `max((cummax-nav)/cummax)`

## 四、冻结纪律

1. **改定义/方向/组合方式 → 必须**：更新本文件 → 重跑受影响最小可复现实验
   （`exp_turnover_vol_20/run.py` 等）→ 对照 `expected_metrics.json` → 人工批准结论升级/降级。
2. **改数据 → 必须**：`make_data_manifest.py` 生成新快照（历史不覆盖），实验退出码 2 阻断，
   对比 old-vs-new 指标差异后人工批准。
3. **任何文档数字与本文档口径冲突** → 以本文档为准，修正结论文档并记录。
4. 前向验证: `daily_signal.py` 生成信号，`paper_track.py` 跟踪纸面绩效（进行中 OOS，不参与结论升降级）。
5. **ENS_T60_TV12 前向闸门**（前向验证，独立于第 4 条 RS12 线；判据落 `paper_track.py` 月频汇总）:
   - 起点 NAV = 1.0，口径 = 月频（月末净值）。
   - **回撤口径（写死，防口径漂移）**: MaxDD 一律以**月频口径**计算（仅用月末净值序列算回撤），与回测参考线 -19.32% 同口径；**禁止拿日频回撤对这条线**。
   - **升 ✅**: 前向满 6 个月 且 年化 Sharpe > 0.5 且 月频 MaxDD ≥ -24.32%（= 回测月频口径 -19.32% 放宽 5pp 容差）。
   - **降 ❌**: 年化 Sharpe < 0 或 月频 MaxDD < -24.32%（破线）。
   - 中间态（0 ≤ Sharpe ≤ 0.5 且回撤未破线）→ 维持观察，不升不降。
   - 前向信号生成器已接入：`serve/ens_t60_tv12_signal.py`（月末输出目标权重表）+ `serve/paper_track.py` 双策略跟踪（RS12 → `paper_nav.csv`、ENS → `paper_nav_ens.csv`）；闸门进度行由 ENS 真样本驱动，累计满 6 个月自动启动。

## 五、权威代码路径

```
research/factor_dic/run_validation.py    # 单因子验证（IC/Top60 回测/收益与换手率序列）
research/factor_dic/combo_backtest.py    # 合成组合回测（BASE/BASE_F/ENH...）
research/factor_dic/factor_lib.py        # 因子实现（含 turnover_vol_20）
research/factor_dic/style_factors.py     # VAL 合成价值（PIT）
research/serve/daily_signal.py           # 生产前向信号（RS12）
research/serve/ens_t60_tv12_signal.py    # ENS_T60_TV12 前向信号生成器（月末目标权重表）
research/serve/paper_track.py            # 前向纸面跟踪（双策略：RS12 + ENS）
research/experiments/exp_turnover_vol_20/run.py   # 最小可复现实验
research/experiments/exp_turnover_vol_20/dsr_pbo.py
research/experiments/exp_turnover_vol_20/cost_sensitivity.py
```

### 五.1 共享数据缓存（跨任务，须登记刷新职责）

s123 状态机的 PE/ERP 依赖两个共享缓存，**惰性刷新、无自动过期检测**（有缓存即读，无缓存才拉取）。月末跑 ENS 信号前须刷新，否则静默用过期数据。

| 缓存文件 | 内容 | 刷新函数（源） | 数据源 | 刷新方式 |
|---|---|---|---|---|
| `fund_research/cache/pe_csi300.parquet` | 沪深300 PE-TTM 日频 | `timing_dingtou.py::fetch_pe_csi300()` | legulegu.com | 手动删缓存后重跑即拉取 |
| `fund_research/cache/bond10y.parquet` | 中债 10 年国债收益率 | `timing_dingtou.py::fetch_bond10y()` | akshare `bond_zh_us_rate` | 手动删缓存后重跑即拉取 |

- 刷新频率：每月末跑 `ens_t60_tv12_signal.py` 前，删旧缓存重跑一次，使 PE/ERP 覆盖到最新交易日。
- 消费者：`ens_t60_tv12_signal.py`、`stock_gbdt_s123_backtest.py` 等所有 s123 择时脚本（20+ 处）。
- 待办：纳入 `make_data_manifest.py` 漂移检测（PE 缓存过期应阻断或告警，而非静默复用）。

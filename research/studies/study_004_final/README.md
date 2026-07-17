"""
# Study 004 Final: 改进版超跌反弹策略

基于用户原始策略和学术建议的全面改进。

## 策略核心改进

### 1. 持有期延长：1日 → 28日
- 原始策略：次日买入、次日卖出（1日超短）
- **改进**：次日开盘买入、28日后卖出（匹配50%止盈目标）
- **理由**：1日持有无法捕捉超跌反弹的修复空间，28日给估值修复足够时间

### 2. 新增逆向筛选特征
- **PE/PB/RSI/布林带历史分位数**：判断当前估值是否处于历史低位
- **动量反转得分**：长期深跌 + 短期企稳 = 反弹信号
- **60日回撤深度**：量化"从高点回调了多少"
- **短期vs长期动量差异**：识别动量反转拐点

### 3. 基本面硬过滤
- PE > 0（排除亏损股）
- PE < 100（排除极端估值）
- PB < 10
- PEG < 2（估值相对增长合理）
- ROE > 5%（盈利质量）
- 资产负债率 < 70%

### 4. 止损止盈机制（策略最大改进）
- **硬性止损**：-15%（或-20%）无条件止损
- **目标止盈**：+50%（匹配用户原策略）
- **时间止损**：20-40个交易日未达预期则平仓
- **动态跟踪**：每个持仓独立监控

### 5. 参数网格搜索
- 原始：只优化 threshold + max_positions
- **改进**：同时优化 threshold + max_positions + stop_loss + take_profit + time_stop
- 5维网格搜索，找到最优风险收益组合

## 文件结构

```
study_004_final/
├── config.py                    # 全局配置
├── build_contrarian_features.py  # Step 1: 特征工程
├── train_contrarian.py           # Step 2: 模型训练
├── optimize_contrarian.py        # Step 3: 参数优化
├── backtest_contrarian.py        # Step 4: 详细回测
├── generate_signal.py            # Step 5: 每日信号
├── run_all.py                    # 一键运行
└── README.md                     # 本文件
```

## 执行流程

### 方式1：一键运行（推荐）
```bash
cd study_004_final
python run_all.py
```

### 方式2：分步运行（调试/开发）
```bash
# Step 1: 构建特征（约30-60分钟，首次运行）
python build_contrarian_features.py

# Step 2: 训练模型（约30-60分钟）
python train_contrarian.py

# Step 3: 参数优化（约5-20分钟，取决于网格大小）
python optimize_contrarian.py

# Step 4: 回测分析
python backtest_contrarian.py

# Step 5: 生成信号（每日收盘后运行）
python generate_signal.py
python generate_signal.py --date 20260101
```

## 数据依赖

- `study_004_systematic/v2_pipeline/data/all_features_v2.parquet`：基础特征数据
- `D:\iquant_data\data_v2\data_day1\`：原始价格数据
- `D:\iquant_data\data_v2\income1\`：财务数据（EPS等）
- `D:\iquant_data\data_v2\other_day1\`：其他基本面数据

## 关键配置（config.py）

```python
# 目标配置
TARGET_HORIZON_DAYS = 28
TARGET_RETURN_THRESHOLD = 0.05  # 28日收益>5%为正样本

# 网格搜索参数
THRESHOLD_RANGE = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65]
MAX_POSITIONS_RANGE = [1, 2, 3, 5, 8]
STOP_LOSS_RANGE = [0.10, 0.15, 0.20]
TAKE_PROFIT_RANGE = [0.30, 0.50, 0.80]
TIME_STOP_RANGE = [20, 28, 40]

# 基本面过滤
FUNDAMENTAL_FILTERS = {
    'min_roe': 0.05,
    'max_debt_ratio': 0.70,
    'max_peg': 2.0,
    'min_pe': 0,
    'max_pe': 100,
    'max_pb': 10,
}
```

## 输出结果

```
study_004_final/
├── data/
│   └── contrarian_features.parquet      # 增强特征数据
├── predictions/
│   └── contrarian_predictions.parquet   # Walk-Forward预测
├── models/
│   ├── contrarian_model.joblib          # 最新模型
│   └── contrarian_features.joblib       # 特征列表
├── results/
│   ├── contrarian_optimized.json        # 最优参数
│   ├── contrarian_grid.parquet          # 网格搜索完整结果
│   ├── equity_contrarian_*.png          # 权益曲线
│   ├── trade_dist_*.png                 # 交易分布
│   └── backtest_detail.json             # 回测详情
└── signals/
    └── YYYYMMDD.json                    # 每日买入信号
```

## 学术依据

- **De Bondt & Thaler (1985)**: 过去输家3-5年后跑赢赢家
- **Fama & French (1988)**: 长期均值回归证据
- **Peters (1991) / Wang et al. (2020)**: 低PEG策略有效
- **国信证券实证**: 超跌反弹适用于"基本面健康"的股票，不适用于"跌幅最大"的股票
- **Brandes Institute**: 基本面健康的"falling knives"后续跑赢市场

## 风险提示

1. 28日持有期在极端行情下可能无法及时止损
2. 基本面数据（income）更新频率低，可能滞后
3. 模型在2026年验证期表现需关注过拟合风险
4. 建议先用小仓位实盘验证，再逐步放大

## 后续优化方向

1. 增加行业轮动过滤（避开衰退行业）
2. 增加宏观择时（在市场底部区域使用）
3. 增加新闻情感强度作为择时信号
4. 改进止盈：从固定50%改为移动止盈（盈利15%后每涨5%上移止损线）

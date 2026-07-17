# 研究项目结构规范

## 目录结构

```
research/
├── README.md                    # 总览文档
├── PROJECT_STRUCTURE.md         # 本文件
├── shared/                      # 共享模块
│   ├── __init__.py
│   ├── data_loader.py           # 数据加载
│   ├── feature_engineering.py   # 特征工程
│   ├── models.py                # 模型定义
│   └── backtest_engine.py       # 回测引擎
├── studies/                     # 各个研究项目
│   ├── study_001_baseline/      # 每个研究一个文件夹
│   │   ├── README.md            # 研究说明
│   │   ├── config.py            # 配置文件
│   │   ├── run.py               # 运行脚本
│   │   ├── data/                # 处理后的数据
│   │   │   ├── features.parquet
│   │   │   └── predictions.parquet
│   │   └── results/             # 结果
│   │       ├── backtest_report.csv
│   │       └── equity_curve.csv
│   └── study_002_stacking/
│       ├── README.md
│       ├── config.py
│       ├── run.py
│       ├── data/
│       └── results/
└── archive/                     # 归档旧代码
    └── old_backtests/
```

## 核心原则

1. **每个研究独立文件夹** - 不混在一起
2. **共享模块提取到 shared/** - 避免重复代码
3. **处理后的数据存储** - 不用每次重新计算
4. **预测数据单独存储** - 改变回测参数不用重新预测
5. **每个研究有README** - 记录研究目的、方法、结果

## 数据流

```
原始数据 (D:\iquant_data)
    ↓
shared.data_loader 加载
    ↓
shared.feature_engineering 计算特征
    ↓
study_XXX/data/features.parquet 存储
    ↓
模型训练 → study_XXX/data/predictions.parquet 存储
    ↓
回测引擎 (不同参数)
    ↓
study_XXX/results/ 结果
```

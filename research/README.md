# Quant Research 研究框架

## 项目结构

```
research/
├── shared/                      # 共享模块（可复用）
│   ├── data_loader.py           # 数据加载
│   ├── feature_engineering.py   # 特征工程
│   ├── models.py                # 模型定义
│   └── backtest_engine.py       # 回测引擎
├── studies/                     # 各个研究项目
│   └── study_001_baseline/      # 每个研究独立文件夹
│       ├── README.md            # 研究说明
│       ├── config.py            # 配置
│       ├── run.py               # 运行脚本
│       ├── data/                # 处理后的数据
│       │   ├── features.parquet
│       │   └── predictions.parquet
│       └── results/             # 回测结果
├── STUDIES_REGISTRY.md          # 研究总览（自动更新）
├── update_registry.py           # 注册表更新脚本
└── README.md                    # 本文件
```

## 快速开始

### 查看所有研究
打开 [STUDIES_REGISTRY.md](STUDIES_REGISTRY.md) 查看所有研究的状态和结果。

### 运行研究
```bash
cd studies/study_001_baseline
python run.py
```

### 更新注册表
```bash
python update_registry.py
```

## 创建新研究

1. 复制模板：
```bash
cp -r studies/study_001_baseline studies/study_003_new_idea
```

2. 修改 `config.py` 和 `README.md`

3. 实现 `run.py`

4. 运行并查看结果

## 核心设计

### 数据缓存
- `features.parquet` - 计算好的特征（不用每次重新算）
- `predictions.parquet` - 模型预测结果（改变回测参数不用重训）

### 模块化
- `shared/` 目录存放通用代码
- 每个研究只写自己的 `config.py` 和 `run.py`

### 可追踪
- 每个研究有README记录假设和结果
- 结果按时间戳保存
- 自动汇总到 STUDIES_REGISTRY.md

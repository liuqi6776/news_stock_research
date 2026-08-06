# -*- coding: utf-8 -*-
"""repro_core — quant_system_v2 核心策略逻辑的最小公开复现包。

所有模块均为纯函数（输入 pandas/numpy 数据, 输出计算结果, 不读私有数据路径）,
外部研究者可用 synthetic_data 生成器 + 固定种子独立验证核心数学逻辑。

模块划分（与上游脚本一一对应）:
  - metrics     绩效指标: MaxDD(相对口径)/CAGR/Sharpe/Calmar/月胜率/超额   <- run_validation.py, risk_control_bt.py
  - alignment   基准对齐 + walk-forward 持有期拼接                          <- run_validation.py, risk_control_bt.py
  - signals     MA20 三档/五档/十档/廿档/Vol/DD/CPPI/TIPP 仓位信号 + T-1 位移 <- risk_control_bt.py
  - hrp         HRP 权重 (120日窗口, LedoitWolf, single linkage)            <- risk_control_bt.py
  - pit         PIT 对齐: 指数权重<=调仓日 / 未来收益 T+1~T+N               <- run_validation.py
  - drift       数据漂移检测 (size/mtime 快速跳过 + sha256 核验)             <- experiments/_common.py
  - docs_sync   文档指标同步: txt 解析 / 期望对比 / markdown 表格            <- experiments/_common.py
  - synthetic   固定种子合成数据生成器（外部可独立复现, 无私有数据依赖）

设计约定:
  - 日期一律用 '%Y%m%d' 字符串（与上游 trade_date 一致）
  - pct 收益单位: 日频 pct_chg 为 %, 策略收益计算统一 /100.0 后连乘
  - MaxDD 为相对回撤 ((cummax-nav)/cummax).max() ∈ [0,1]
  - 仓位信号一律取 T-1 日收盘已知信息、T 日生效（无同日前视）
"""
__version__ = "0.1.0"

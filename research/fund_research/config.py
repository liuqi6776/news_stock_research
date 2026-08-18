# -*- coding: utf-8 -*-
"""
fund_research 配置：路径与通用参数
"""
import os

# ---------------------------------------------------------------------------
# 数据路径
# ---------------------------------------------------------------------------
# 场内基金（ETF/LOF）日线，按交易日 YYYYMMDD.parquet 存储，2014 至今
FUND_DATA_DIR = r"D:\iquant_data\data_v2\fund1"
# 场内基金基础信息（ts_code/name/management/found_date/list_date 等）
FUND_BASIC_PATH = os.path.join(FUND_DATA_DIR, "fund_basic_E.parquet")

# 本目录下的缓存/结果输出
HERE = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(HERE, "cache")
RESULTS_DIR = os.path.join(HERE, "results")

for _d in (CACHE_DIR, RESULTS_DIR):
    os.makedirs(_d, exist_ok=True)

# 合并后的日线面板缓存（避免每次读取数千个 parquet）
PANEL_CACHE_PATH = os.path.join(CACHE_DIR, "fund_daily_panel.parquet")

# ---------------------------------------------------------------------------
# 场外基金数据 (D:\iquant_data\data_v2\fund2)
# ---------------------------------------------------------------------------
OTC_DATA_DIR = r"D:\iquant_data\data_v2\fund2"
# 每只基金一个净值文件
OTC_NAV_DIR = os.path.join(OTC_DATA_DIR, "nav")
# 场外基金基础信息（代码/名称/类型）
OTC_BASIC_PATH = os.path.join(OTC_DATA_DIR, "fund_basic_O.parquet")
# 场外净值面板缓存
OTC_PANEL_CACHE_PATH = os.path.join(CACHE_DIR, "otc_nav_panel.parquet")

# ---------------------------------------------------------------------------
# 通用参数
# ---------------------------------------------------------------------------
# 数据起止（本地场内数据从 20140101 开始）
DEFAULT_START = "20140101"
DEFAULT_END = None  # None = 到今天

# 输出控制
VERBOSE = True

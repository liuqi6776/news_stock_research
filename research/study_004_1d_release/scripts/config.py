"""
v2 Pipeline 配置

v2核心: 无clip，只优化threshold + max_positions
数据划分: 2022-2025优化期, 2026验证期
Walk-Forward: 每年独立训练，无数据泄漏
"""
import os

PIPELINE_DIR = os.path.dirname(os.path.abspath(__file__))
STUDY_DIR = os.path.dirname(PIPELINE_DIR)
RESEARCH_DIR = os.path.dirname(STUDY_DIR)
PROJECT_DIR = os.path.dirname(RESEARCH_DIR)

DATA_DIR = os.path.join(PIPELINE_DIR, 'data')
RESULTS_DIR = os.path.join(PIPELINE_DIR, 'results')
PREDICTIONS_DIR = os.path.join(PIPELINE_DIR, 'predictions')
MODELS_DIR = os.path.join(PIPELINE_DIR, 'models')
SIGNALS_DIR = os.path.join(PIPELINE_DIR, 'signals')

for d in [DATA_DIR, RESULTS_DIR, PREDICTIONS_DIR, MODELS_DIR, SIGNALS_DIR]:
    os.makedirs(d, exist_ok=True)

RAW_PRICE_DIR = r'D:\iquant_data\data_v2\data_day1'
RAW_OTHER_DIR = r'D:\iquant_data\data_v2\other_day1'
RAW_NEWS_DIR = r'D:\iquant_data\data_v2\news_major1'
RAW_RANK_DIR = r'D:\iquant_data\data_v2\ths_rank1'
RAW_MONEYFLOW_DIR = r'D:\iquant_data\data_v2\moneyflow1'
RAW_THS_NEWS_DIR = r'D:\iquant_data\data_v2\ths_news1'
RAW_INCOME_DIR = r'D:\iquant_data\data_v2\income1'

TRAIN_START = '20200101'
OPT_START = '20220101'
OPT_END = '20251231'
VAL_START = '20260101'
VAL_END = '20261231'

WALK_FORWARD_YEARS = [2022, 2023, 2024, 2025, 2026]
MIN_TRAIN_SAMPLES = 100000
MIN_LISTING_DAYS = 60

TRANSACTION_COST = 0.003

THRESHOLD_RANGE_START = 0.30
THRESHOLD_RANGE_END = 0.65
THRESHOLD_RANGE_STEP = 0.02
MAX_POSITIONS_RANGE = [1, 2, 3, 5, 10]

TARGET_RETURN_THRESHOLD = 0.015
TARGET_HORIZON_DAYS = 2

FEATURES_FILE = os.path.join(DATA_DIR, 'all_features_v2.parquet')
WF_PREDICTIONS_FILE = os.path.join(PREDICTIONS_DIR, 'predictions_1d_wf.parquet')
OPT_RESULTS_FILE = os.path.join(RESULTS_DIR, 'optimized_params_v2.json')
GRID_RESULTS_FILE = os.path.join(RESULTS_DIR, 'grid_search_v2_results.parquet')

LATEST_MODEL_FILE = os.path.join(MODELS_DIR, 'latest_wf_model.joblib')
LATEST_FEATS_FILE = os.path.join(MODELS_DIR, 'latest_wf_features.joblib')

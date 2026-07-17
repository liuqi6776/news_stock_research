"""
Study 004: 最终框架配置

核心设计：
1. 三个预测目标（1日/5日/28日持有期）
2. 自动阈值检验
3. 预测与回测分离
4. 全部历史训练，最后一年测试
"""
import os

# 路径配置
STUDY_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(STUDY_DIR, 'data')
RESULTS_DIR = os.path.join(STUDY_DIR, 'results')

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# 数据路径
PRICE_DIR = r'D:\iquant_data\data_v2\data_day1'
NEWS_DIR = r'D:\iquant_data\data_v2\data_day_news'
RANK_DIR = r'D:\iquant_data\data_v2\data_day_ths_rank'

# 时间配置
START_DATE = '20210101'  # 最早可用数据
END_DATE = '20241231'
TEST_YEAR = '2024'       # 最后一年作为测试

# 预测目标配置
# 目标1: t+2开盘买入, t+3收盘卖出 (持有1日)
# 目标2: t+2开盘买入, t+7收盘卖出 (持有5日)
# 目标3: t+2开盘买入, t+30收盘卖出 (持有28日)
TARGETS = {
    '1d': {'hold_days': 1, 'exit_offset': 3, 'threshold': 0.01},   # t+3收盘
    '5d': {'hold_days': 5, 'exit_offset': 7, 'threshold': 0.03},   # t+7收盘
    '28d': {'hold_days': 28, 'exit_offset': 30, 'threshold': 0.08} # t+30收盘
}

# 模型配置
N_ESTIMATORS = 100
MAX_DEPTH = 5

# 回测网格搜索参数
STOP_LOSS_GRID = [0.03, 0.05, 0.08]
TAKE_PROFIT_GRID = [0.05, 0.10, 0.15, 0.20]
MAX_POSITIONS_GRID = [3, 5, 10]

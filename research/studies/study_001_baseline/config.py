"""
Study 001 配置文件
"""
import os

# 路径配置
STUDY_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(STUDY_DIR, 'data')
RESULTS_DIR = os.path.join(STUDY_DIR, 'results')

# 确保目录存在
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# 数据配置
PRICE_DIR = r'D:\iquant_data\data_v2\data_day1'
START_DATE = '20230101'
END_DATE = '20241231'

# 模型配置
LABEL_THRESHOLD = 0.02  # 目标阈值
N_TOP_FEATURES = 20     # Top特征数量

# 回测配置
MIN_PROB = 0.55         # 最小概率
STOP_LOSS = 0.05        # 止损
MAX_POSITIONS = 5       # 最大持仓

# Walk Forward配置
TRAIN_MONTHS = 12       # 训练月数
TEST_MONTHS = 6         # 测试月数

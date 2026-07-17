"""
Study 003 配置文件 - 优化框架
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
NEWS_DIR = r'D:\iquant_data\data_v2\data_day_news'
RANK_DIR = r'D:\iquant_data\data_v2\data_day_ths_rank'

START_DATE = '20220101'
END_DATE = '20241231'

# 预测目标配置
PREDICT_HORIZON = 5  # 预测N日收益 (2/5/10)
LABEL_THRESHOLD = 0.03  # 目标阈值 (对应N日收益)

# 模型配置
N_TOP_FEATURES = 20

# Walk Forward配置（年级别）
TRAIN_YEARS = 2  # 训练年数
TEST_YEARS = 1   # 测试年数

# 回测配置
MIN_PROB = 0.55
STOP_LOSS = 0.05
MAX_POSITIONS = 5
USE_MARKET_FILTER = True
USE_VOLATILITY_FILTER = True

# 动态阈值配置
USE_DYNAMIC_THRESHOLD = True
THRESHOLD_ADJUSTMENT = 0.05  # 根据市场调整幅度

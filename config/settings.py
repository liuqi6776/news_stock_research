"""
配置管理模块

集中管理所有配置项，支持环境变量和配置文件
"""

import os
from pathlib import Path
from typing import Optional

# 尝试从 .env 文件加载配置
try:
    from dotenv import load_dotenv
    env_path = Path(__file__).parent.parent / '.env'
    if env_path.exists():
        load_dotenv(env_path)
except ImportError:
    pass


class Settings:
    """配置管理类"""
    
    # Tushare API 配置
    TUSHARE_TOKEN: str = os.getenv('TUSHARE_TOKEN', '')
    # TUSHARE_API_URL: str = os.getenv('TUSHARE_API_URL', 'http://tsapi.majors.ltd:7000')
    
    # 数据存储路径
    DATA_PATH: str = os.getenv('DATA_PATH', 'D:/iquant_data/data_v2')
    
    # 数据子目录
    @property
    def daily_data_path(self) -> str:
        return os.path.join(self.DATA_PATH, 'data_day1')
    
    @property
    def other_data_path(self) -> str:
        return os.path.join(self.DATA_PATH, 'other_day1')
    
    @property
    def skill_data_path(self) -> str:
        return os.path.join(self.DATA_PATH, 'skill1')
    
    @property
    def recommend_data_path(self) -> str:
        return os.path.join(self.DATA_PATH, 'recommond1')
    
    @property
    def ths_rank_path(self) -> str:
        return os.path.join(self.DATA_PATH, 'ths_rank1')
    
    @property
    def money_flow_path(self) -> str:
        return os.path.join(self.DATA_PATH, 'money1')
    
    @property
    def moneyflow_path(self) -> str:
        return os.path.join(self.DATA_PATH, 'moneyflow1')
    
    @property
    def board_path(self) -> str:
        return os.path.join(self.DATA_PATH, 'board1')
    
    @property
    def cyq_path(self) -> str:
        return os.path.join(self.DATA_PATH, 'cyq1')
    
    @property
    def income_path(self) -> str:
        return os.path.join(self.DATA_PATH, 'income1')
    
    @property
    def vix_data_path(self) -> str:
        return os.path.join(self.DATA_PATH, 'vix1')
    
    @property
    def margin_data_path(self) -> str:
        return os.path.join(self.DATA_PATH, 'margin1')
    
    # API 重试配置
    MAX_RETRIES: int = 10
    BASE_SLEEP_TIME: int = 1
    MAX_SLEEP_TIME: int = 30
    
    # 数据处理配置
    MIN_DATA_POINTS: int = 200  # 最小数据点数量
    DATA_QUALITY_THRESHOLD: float = 0.8  # 数据质量阈值
    
    def validate(self) -> bool:
        """验证配置是否有效"""
        if not self.TUSHARE_TOKEN:
            raise ValueError("TUSHARE_TOKEN 未配置，请设置环境变量或创建 .env 文件")
        if not os.path.exists(self.DATA_PATH):
            os.makedirs(self.DATA_PATH, exist_ok=True)
        return True


# 全局配置实例
settings = Settings()

"""
Fast Fetcher & Real-time Signal Generator
使用 Akshare 获取实时数据，用于 T 日尾盘（14:45+）生成抢跑信号。
"""

import os
import sys
import pandas as pd
import akshare as ak
from datetime import datetime
import joblib
import warnings

warnings.filterwarnings('ignore')

# 加入路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from features.enhanced_factors import calculate_all_enhanced_features
from infra_data.storage import DataStorage

class FastSignalGenerator:
    def __init__(self, model_v5_path=None):
        self.model_path = model_v5_path or r'C:\Users\liuqi\quant_system_v2\models\v5_xgb_model.pkl'
        self.scaler_path = r'C:\Users\liuqi\quant_system_v2\models\v5_scaler.pkl'
        self.features_path = r'C:\Users\liuqi\quant_system_v2\models\v5_features.pkl'
        
        self.model = None
        self.scaler = None
        self.features = None
        self.storage = DataStorage()

    def load_model(self):
        if os.path.exists(self.model_path):
            self.model = joblib.load(self.model_path)
            self.scaler = joblib.load(self.scaler_path)
            self.features = joblib.load(self.features_path)
            print(f"模型加载成功: {self.model_path}")
            return True
        return False

    def fetch_realtime_quotes(self):
        """获取全市场实时行情摘要"""
        print("正在获取全市场实时行情 (Akshare)...")
        df = ak.stock_zh_a_spot_em()
        # 映射字段到 Tushare 风格
        # 字段: 代码, 名称, 最新价, 昨收, 涨跌幅, 涨跌额, 成交量, 成交额, 振幅, 最高, 最低, 今开
        df = df[['代码', '名称', '最新价', '今开', '最高', '最低', '昨收', '成交量', '成交额', '涨跌幅']]
        df.columns = ['symbol', 'name', 'close', 'open', 'high', 'low', 'pre_close', 'vol', 'amount', 'pct_chg']
        
        # 处理 ts_code (简单示例，实际需区分 SH/SZ)
        def to_ts_code(s):
            if s.startswith('6'): return s + '.SH'
            return s + '.SZ'
        
        df['ts_code'] = df['symbol'].apply(to_ts_code)
        df['trade_date'] = datetime.now().strftime('%Y%m%d')
        return df

    def generate_fast_signals(self, top_n=10):
        if not self.load_model():
            print("错误: 未找到模型，请先训练 V5 模型。")
            return None

        # 1. 获取实时行情
        realtime_df = self.fetch_realtime_quotes()
        
        # 2. 加载最近几天的历史数据用于特征计算 (MA, MACD等需要历史)
        # 这里简化处理：至少需要最近 30 天的历史
        end_date = datetime.now().strftime('%Y%m%d')
        start_date = (datetime.now() - pd.Timedelta(days=60)).strftime('%Y%m%d')
        history_df = self.storage.load_daily_data(start_date, end_date)
        
        # 3. 合并实时数据到历史中
        # 剔除历史中已有的今日数据（如果有）
        history_df = history_df[history_df['trade_date'] < end_date]
        full_df = pd.concat([history_df, realtime_df], ignore_index=True).sort_values(['ts_code', 'trade_date'])
        
        # 4. 计算特征 (这里需要和 V5 训练时的特征一致)
        print("计算实时特征...")
        # 注意：这里仅展示思路，实际中 calculate_all_enhanced_features 可能需要更多维度的历史数据
        # 如果某些数据（筹码、资金流）无法实时获取，则填缺失值
        money_flow = self.storage.load_money_flow(start_date, end_date)
        chip_data = self.storage.load_chip_data(start_date, end_date)
        
        full_df = calculate_all_enhanced_features(full_df, money_flow, chip_data)
        
        # 5. 提取最新时刻数据进行预测
        latest_data = full_df[full_df['trade_date'] == end_date].copy()
        X = latest_data[self.features].fillna(0).replace([np.inf, -np.inf], 0)
        X_scaled = self.scaler.transform(X)
        
        latest_data['prob'] = self.model.predict_proba(X_scaled)[:, 1]
        
        # 6. 过滤与排序
        # 避开涨停股
        results = latest_data[
            (latest_data['prob'] > 0.6) & 
            (latest_data['pct_chg'] < 9.7)
        ].sort_values('prob', ascending=False).head(top_n)
        
        return results[['ts_code', 'name', 'close', 'pct_chg', 'prob']]

if __name__ == "__main__":
    gen = FastSignalGenerator()
    signals = gen.generate_fast_signals()
    if signals is not None:
        print("\n" + "="*50)
        print(f"  实时抢跑信号 (生成时间: {datetime.now().strftime('%H:%M:%S')})")
        print("="*50)
        print(signals)
        print("="*50)
        print("建议动作: 如果在 14:50 之后生成，可考虑在集合竞价买入。")

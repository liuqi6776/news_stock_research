"""
每日交易信号生成器

功能:
1. 每天运行，获取最新数据
2. 使用训练好的模型预测今日推荐股票
3. 输出买入/卖出信号

使用方法:
    python daily_signal.py              # 查看今日推荐
    python daily_signal.py --train      # 重新训练模型
    python daily_signal.py --top 20     # 查看前20只推荐
"""

import os
import sys
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, r'C:\Users\liuqi\quant_system_v2')

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import argparse
import joblib

from sklearn.preprocessing import StandardScaler
try:
    import xgboost as xgb
except ImportError:
    print("请安装 xgboost: pip install xgboost")
    sys.exit(1)

from config.settings import settings
from infra_data.storage import DataStorage
from features.alpha_factors import calculate_alpha101_factors
from features.labeling import simple_return_labeling
from processing.cleaner import filter_stock_codes
from processing.merger import merge_dataframes


class DailySignalGenerator:
    """每日交易信号生成器"""
    
    MODEL_PATH = r'C:\Users\liuqi\quant_system_v2\models\xgb_model.pkl'
    SCALER_PATH = r'C:\Users\liuqi\quant_system_v2\models\scaler.pkl'
    FEATURES_PATH = r'C:\Users\liuqi\quant_system_v2\models\features.pkl'
    
    def __init__(self):
        self.storage = DataStorage()
        self.model = None
        self.scaler = None
        self.features = None
    
    def load_recent_data(self, days: int = 90) -> pd.DataFrame:
        """加载最近N天的数据"""
        end_date = datetime.now().strftime('%Y%m%d')
        start_date = (datetime.now() - timedelta(days=days)).strftime('%Y%m%d')
        
        print(f"加载数据 {start_date} - {end_date}...")
        
        daily = self.storage.load_daily_data(start_date, end_date)
        if daily.empty:
            raise ValueError("没有找到数据，请先运行数据获取")
        
        other = self.storage.load_daily_basic(start_date, end_date)
        skill = self.storage.load_technical_factors(start_date, end_date)
        
        dfs = [daily]
        if not other.empty:
            dfs.append(other)
        if not skill.empty:
            dfs.append(skill)
        
        df = merge_dataframes(dfs)
        df = filter_stock_codes(df, patterns=['^60', '^00'])
        
        print(f"加载 {len(df)} 条记录, {df['ts_code'].nunique()} 只股票")
        
        return df
    
    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """准备特征"""
        print("计算特征...")
        df = calculate_alpha101_factors(df)
        return df
    
    def get_features(self, df: pd.DataFrame) -> list:
        """获取特征列"""
        exclude = ['ts_code', 'trade_date', 'return_label', 'future_return', 
                   'pct_label', 'return_rank', 'tb_label']
        
        features = []
        for col in df.columns:
            if col in exclude:
                continue
            if df[col].dtype not in ['object', 'datetime64[ns]']:
                features.append(col)
        
        return features
    
    def train_model(self, train_days: int = 365 * 3):
        """训练并保存模型"""
        print("=" * 50)
        print("训练模型...")
        print("=" * 50)
        
        # 加载数据
        end_date = datetime.now().strftime('%Y%m%d')
        start_date = (datetime.now() - timedelta(days=train_days)).strftime('%Y%m%d')
        
        print(f"训练数据范围: {start_date} - {end_date}")
        
        daily = self.storage.load_daily_data(start_date, end_date)
        other = self.storage.load_daily_basic(start_date, end_date)
        skill = self.storage.load_technical_factors(start_date, end_date)
        
        dfs = [daily]
        if not other.empty:
            dfs.append(other)
        if not skill.empty:
            dfs.append(skill)
        
        df = merge_dataframes(dfs)
        df = filter_stock_codes(df, patterns=['^60', '^00'])
        
        print(f"总数据: {len(df)} 条")
        
        # 特征工程
        df = calculate_alpha101_factors(df)
        df = simple_return_labeling(df, forward_days=20, threshold=0.05)
        
        # 获取特征
        features = self.get_features(df)
        print(f"特征数: {len(features)}")
        
        # 准备训练数据
        train_df = df[df['return_label'].notna()].copy()
        print(f"训练样本: {len(train_df)}")
        
        X = train_df[features].fillna(0)
        y = (train_df['return_label'] == 1).astype(int)
        
        for col in X.columns:
            X[col] = pd.to_numeric(X[col], errors='coerce').fillna(0)
        X = X.replace([np.inf, -np.inf], 0)
        
        # 标准化
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # 训练
        model = xgb.XGBClassifier(
            n_estimators=150,
            max_depth=6,
            learning_rate=0.08,
            random_state=42,
            use_label_encoder=False,
            eval_metric='logloss'
        )
        model.fit(X_scaled, y)
        
        # 保存模型
        os.makedirs(os.path.dirname(self.MODEL_PATH), exist_ok=True)
        joblib.dump(model, self.MODEL_PATH)
        joblib.dump(scaler, self.SCALER_PATH)
        joblib.dump(features, self.FEATURES_PATH)
        
        print(f"\n模型已保存:")
        print(f"  {self.MODEL_PATH}")
        print(f"  {self.SCALER_PATH}")
        print(f"  {self.FEATURES_PATH}")
        
        self.model = model
        self.scaler = scaler
        self.features = features
        
        return model, scaler, features
    
    def load_model(self):
        """加载已保存的模型"""
        if not os.path.exists(self.MODEL_PATH):
            print("模型不存在，需要先训练...")
            return self.train_model()
        
        self.model = joblib.load(self.MODEL_PATH)
        self.scaler = joblib.load(self.SCALER_PATH)
        self.features = joblib.load(self.FEATURES_PATH)
        
        print("模型加载成功")
        return self.model, self.scaler, self.features
    
    def predict_today(self, top_n: int = 10) -> pd.DataFrame:
        """预测今日推荐股票"""
        # 加载模型
        if self.model is None:
            self.load_model()
        
        # 加载最近数据
        df = self.load_recent_data(days=60)
        df = self.prepare_features(df)
        
        # 获取最新交易日
        latest_date = df['trade_date'].max()
        print(f"\n最新交易日: {latest_date}")
        
        # 预测
        today_data = df[df['trade_date'] == latest_date].copy()
        
        if today_data.empty:
            print("今日无数据")
            return pd.DataFrame()
        
        X = today_data[self.features].fillna(0)
        for col in X.columns:
            X[col] = pd.to_numeric(X[col], errors='coerce').fillna(0)
        X = X.replace([np.inf, -np.inf], 0)
        
        X_scaled = self.scaler.transform(X)
        
        # 预测概率
        proba = self.model.predict_proba(X_scaled)
        today_data['up_proba'] = proba[:, 1]
        
        # 排序并筛选
        recommendations = today_data.sort_values('up_proba', ascending=False)
        
        # 添加额外过滤条件
        recommendations = recommendations[
            (recommendations['up_proba'] > 0.55) &  # 概率>55%
            (recommendations['pct_chg'] < 9.5) &    # 非涨停
            (recommendations['pct_chg'] > -9.5)     # 非跌停
        ].head(top_n)
        
        return recommendations[['ts_code', 'close', 'pct_chg', 'vol', 'up_proba']]
    
    def generate_signals(self, top_n: int = 10):
        """生成并显示今日交易信号"""
        print("\n" + "=" * 60)
        print(f"  每日交易信号  ({datetime.now().strftime('%Y-%m-%d %H:%M')})")
        print("=" * 60)
        
        recommendations = self.predict_today(top_n)
        
        if recommendations.empty:
            print("\n今日无推荐股票")
            return
        
        print(f"\n【今日推荐买入】(共 {len(recommendations)} 只)")
        print("-" * 60)
        print(f"{'排名':<4} {'代码':<12} {'收盘价':>8} {'涨跌幅':>8} {'成交量':>12} {'上涨概率':>10}")
        print("-" * 60)
        
        for i, (_, row) in enumerate(recommendations.iterrows(), 1):
            print(f"{i:<4} {row['ts_code']:<12} {row['close']:>8.2f} {row['pct_chg']:>7.2f}% {row['vol']:>12,.0f} {row['up_proba']*100:>9.1f}%")
        
        print("-" * 60)
        
        # 投资建议
        print("\n[投资建议]")
        print("  - 建议将资金平均分配到推荐股票")
        print("  - 每只股票最多投入总资金的 10%")
        print("  - 设置止损线: -10%, 止盈线: +30%")
        print("  - 持有周期: 约 20 个交易日（1个月）")
        print("  - 每月初重新运行此脚本更新持仓")
        
        print("\n[风险提示]")
        print("  * 模型预测仅供参考，不构成投资建议")
        print("  * 股市有风险，投资需谨慎")
        
        print("=" * 60)
        
        # 保存推荐
        output_file = f"signals_{datetime.now().strftime('%Y%m%d')}.csv"
        recommendations.to_csv(output_file, index=False)
        print(f"\n推荐已保存到: {output_file}")
        
        return recommendations


def main():
    parser = argparse.ArgumentParser(description='每日交易信号生成器')
    parser.add_argument('--train', action='store_true', help='重新训练模型')
    parser.add_argument('--top', type=int, default=10, help='推荐股票数量')
    
    args = parser.parse_args()
    
    generator = DailySignalGenerator()
    
    if args.train:
        generator.train_model()
    
    generator.generate_signals(top_n=args.top)


if __name__ == "__main__":
    main()

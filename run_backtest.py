"""
滚动训练回测脚本

流程:
1. 获取数据到今天
2. 用2020年前的数据训练初始模型
3. 按月滚动更新模型
4. 预测下月涨跌，买入预测涨的股票
"""

import os
import sys
import warnings
warnings.filterwarnings('ignore')

# 添加项目路径
sys.path.insert(0, r'C:\Users\liuqi\quant_system_v2')

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from tqdm import tqdm
import joblib

from sklearn.preprocessing import StandardScaler
try:
    import xgboost as xgb
except ImportError:
    print("请安装 xgboost: pip install xgboost")
    sys.exit(1)

from config.settings import settings
from infra_data.fetcher import DataFetcher
from infra_data.storage import DataStorage
from features.alpha_factors import calculate_alpha101_factors
from features.labeling import simple_return_labeling
from processing.cleaner import filter_stock_codes
from processing.merger import merge_dataframes


class RollingBacktest:
    """
    滚动训练回测
    
    按月滚动更新模型，预测下月涨跌
    """
    
    def __init__(
        self,
        initial_capital: float = 100000,
        top_n: int = 10,  # 每月买入前N只
        commission_rate: float = 0.0003,
        stamp_duty: float = 0.0005,
        forward_days: int = 20,  # 预测未来20天（约1个月）
        threshold: float = 0.05  # 涨跌阈值 5%
    ):
        self.initial_capital = initial_capital
        self.top_n = top_n
        self.commission_rate = commission_rate
        self.stamp_duty = stamp_duty
        self.forward_days = forward_days
        self.threshold = threshold
        
        self.storage = DataStorage()
        
        # 状态
        self.capital = initial_capital
        self.positions = {}  # {ts_code: {'shares': int, 'buy_price': float, 'buy_date': date}}
        self.equity_curve = []
        self.monthly_returns = []
        self.trades = []
    
    def load_and_prepare_data(self, start_date: str, end_date: str) -> pd.DataFrame:
        """加载并准备数据"""
        print("加载数据...")
        
        # 加载日线数据
        daily = self.storage.load_daily_data(start_date, end_date)
        
        if daily.empty:
            raise ValueError(f"没有找到 {start_date} - {end_date} 的数据")
        
        # 加载其他数据
        other = self.storage.load_daily_basic(start_date, end_date)
        skill = self.storage.load_technical_factors(start_date, end_date)
        
        print(f"日线数据: {len(daily)} 条")
        
        # 合并数据
        print("合并数据...")
        dfs = [daily]
        if not other.empty:
            dfs.append(other)
        if not skill.empty:
            dfs.append(skill)
        
        df = merge_dataframes(dfs)
        
        # 过滤股票代码
        df = filter_stock_codes(df, patterns=['^60', '^00'])
        
        # 计算 Alpha 因子
        print("计算 Alpha 因子...")
        df = calculate_alpha101_factors(df)
        
        # 生成标签（未来N天涨跌）
        print("生成标签...")
        df = simple_return_labeling(df, forward_days=self.forward_days, threshold=self.threshold)
        
        # 排序
        df = df.sort_values(['ts_code', 'trade_date'])
        
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
    
    def train_model(self, train_df: pd.DataFrame, features: list):
        """训练模型"""
        X = train_df[features].fillna(0)
        y = train_df['return_label'].copy()
        
        # 将标签转换为二分类 (涨: 1, 不涨: 0)
        y = (y == 1).astype(int)
        
        # 数据类型转换
        for col in X.columns:
            X[col] = pd.to_numeric(X[col], errors='coerce').fillna(0)
        
        # 处理 inf 值
        X = X.replace([np.inf, -np.inf], 0)
        
        # 标准化
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # 训练 XGBoost
        model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=42,
            use_label_encoder=False,
            eval_metric='logloss'
        )
        model.fit(X_scaled, y)
        
        return model, scaler
    
    def predict_stocks(self, model, scaler, pred_df: pd.DataFrame, features: list, date) -> pd.DataFrame:
        """预测股票涨跌并返回推荐列表"""
        day_data = pred_df[pred_df['trade_date'] == date].copy()
        
        if day_data.empty:
            return pd.DataFrame()
        
        X = day_data[features].fillna(0)
        for col in X.columns:
            X[col] = pd.to_numeric(X[col], errors='coerce').fillna(0)
        
        X_scaled = scaler.transform(X)
        
        # 预测概率
        proba = model.predict_proba(X_scaled)
        day_data['up_proba'] = proba[:, 1]  # 上涨概率
        
        # 按概率排序
        top_stocks = day_data.sort_values('up_proba', ascending=False).head(self.top_n)
        
        return top_stocks[['ts_code', 'close', 'up_proba']]
    
    def execute_trades(self, recommendations: pd.DataFrame, date, prices: dict):
        """执行交易"""
        # 先卖出所有持仓（每月调仓）
        for ts_code, pos in list(self.positions.items()):
            if ts_code in prices:
                # 卖出
                revenue = pos['shares'] * prices[ts_code]
                commission = max(revenue * self.commission_rate, 5)
                stamp = revenue * self.stamp_duty
                net_revenue = revenue - commission - stamp
                
                self.capital += net_revenue
                
                pnl = net_revenue - pos['shares'] * pos['buy_price']
                self.trades.append({
                    'date': date,
                    'ts_code': ts_code,
                    'type': 'SELL',
                    'price': prices[ts_code],
                    'shares': pos['shares'],
                    'pnl': pnl
                })
                
                del self.positions[ts_code]
        
        # 买入推荐股票（等权重）
        if recommendations.empty:
            return
        
        per_stock_capital = self.capital * 0.9 / len(recommendations)  # 保留10%现金
        
        for _, row in recommendations.iterrows():
            ts_code = row['ts_code']
            price = row['close']
            
            # 计算买入股数（100的整数倍）
            shares = int(per_stock_capital / price / 100) * 100
            
            if shares >= 100:
                cost = shares * price
                commission = max(cost * self.commission_rate, 5)
                total_cost = cost + commission
                
                if total_cost <= self.capital:
                    self.capital -= total_cost
                    self.positions[ts_code] = {
                        'shares': shares,
                        'buy_price': price,
                        'buy_date': date
                    }
                    
                    self.trades.append({
                        'date': date,
                        'ts_code': ts_code,
                        'type': 'BUY',
                        'price': price,
                        'shares': shares,
                        'pnl': 0
                    })
    
    def get_portfolio_value(self, prices: dict) -> float:
        """计算组合价值"""
        value = self.capital
        for ts_code, pos in self.positions.items():
            if ts_code in prices:
                value += pos['shares'] * prices[ts_code]
        return value
    
    def run(self, data_start: str = '20180101', train_end: str = '20191231'):
        """
        运行滚动回测
        
        Args:
            data_start: 数据开始日期
            train_end: 初始训练结束日期（之后开始回测）
        """
        # 获取今天日期
        today = datetime.now().strftime('%Y%m%d')
        
        print("=" * 60)
        print("滚动训练回测")
        print("=" * 60)
        print(f"数据范围: {data_start} - {today}")
        print(f"初始训练: {data_start} - {train_end}")
        print(f"回测开始: {train_end} 之后")
        print(f"初始资金: CNY {self.initial_capital:,.0f}")
        print(f"每月买入: 前 {self.top_n} 只股票")
        print("=" * 60)
        
        # 加载数据
        df = self.load_and_prepare_data(data_start, today)
        
        print(f"\n数据准备完成:")
        print(f"  总记录: {len(df)}")
        print(f"  股票数: {df['ts_code'].nunique()}")
        print(f"  日期范围: {df['trade_date'].min()} - {df['trade_date'].max()}")
        
        # 获取特征列
        features = self.get_features(df)
        print(f"  特征数: {len(features)}")
        
        # 初始训练
        print("\n初始模型训练...")
        train_end_dt = pd.to_datetime(train_end)
        train_df = df[df['trade_date'] <= train_end_dt].copy()
        
        # 去除没有标签的样本
        train_df = train_df[train_df['return_label'].notna()]
        
        print(f"  训练样本: {len(train_df)}")
        
        model, scaler = self.train_model(train_df, features)
        print("  模型训练完成")
        
        # 获取回测期间的月份列表
        backtest_df = df[df['trade_date'] > train_end_dt].copy()
        backtest_df['year_month'] = backtest_df['trade_date'].dt.to_period('M')
        months = sorted(backtest_df['year_month'].unique())
        
        print(f"\n开始回测，共 {len(months)} 个月...")
        
        # 按月回测
        for i, month in enumerate(tqdm(months, desc="回测进度")):
            month_df = backtest_df[backtest_df['year_month'] == month]
            
            if month_df.empty:
                continue
            
            # 获取月初交易日
            first_day = month_df['trade_date'].min()
            last_day = month_df['trade_date'].max()
            
            # 月初调仓
            prices = dict(zip(month_df[month_df['trade_date'] == first_day]['ts_code'],
                            month_df[month_df['trade_date'] == first_day]['close']))
            
            # 预测并选股
            recommendations = self.predict_stocks(model, scaler, df, features, first_day)
            
            # 执行交易
            self.execute_trades(recommendations, first_day, prices)
            
            # 记录月末组合价值
            end_prices = dict(zip(month_df[month_df['trade_date'] == last_day]['ts_code'],
                                 month_df[month_df['trade_date'] == last_day]['close']))
            portfolio_value = self.get_portfolio_value(end_prices)
            self.equity_curve.append({
                'month': str(month),
                'date': last_day,
                'value': portfolio_value
            })
            
            # 每季度重新训练模型
            if (i + 1) % 3 == 0:
                # 用到当前月末的数据重新训练
                new_train_df = df[df['trade_date'] <= last_day].copy()
                new_train_df = new_train_df[new_train_df['return_label'].notna()]
                model, scaler = self.train_model(new_train_df, features)
        
        # 输出结果
        self.print_results()
        
        return self.equity_curve, self.trades
    
    def print_results(self):
        """打印回测结果"""
        print("\n" + "=" * 60)
        print("回测结果")
        print("=" * 60)
        
        if not self.equity_curve:
            print("没有回测数据")
            return
        
        initial = self.initial_capital
        final = self.equity_curve[-1]['value']
        
        # 收益率
        total_return = (final / initial - 1) * 100
        
        # 年化收益率
        months = len(self.equity_curve)
        annual_return = ((final / initial) ** (12 / months) - 1) * 100 if months > 0 else 0
        
        # 最大回撤
        values = [e['value'] for e in self.equity_curve]
        peak = values[0]
        max_dd = 0
        for v in values:
            if v > peak:
                peak = v
            dd = (peak - v) / peak * 100
            if dd > max_dd:
                max_dd = dd
        
        # 胜率
        buy_trades = [t for t in self.trades if t['type'] == 'BUY']
        sell_trades = [t for t in self.trades if t['type'] == 'SELL']
        wins = sum(1 for t in sell_trades if t['pnl'] > 0)
        win_rate = (wins / len(sell_trades) * 100) if sell_trades else 0
        
        print(f"  初始资金:     CNY {initial:,.0f}")
        print(f"  最终资金:     CNY {final:,.0f}")
        print(f"  总收益率:     {total_return:.2f}%")
        print(f"  年化收益率:   {annual_return:.2f}%")
        print(f"  最大回撤:     {max_dd:.2f}%")
        print(f"  胜率:         {win_rate:.2f}%")
        print(f"  回测月份:     {months} 个月")
        print(f"  总交易次数:   {len(self.trades)}")
        print(f"  买入次数:     {len(buy_trades)}")
        print(f"  卖出次数:     {len(sell_trades)}")
        
        # 当前持仓
        if self.positions:
            print(f"\n当前持仓 ({len(self.positions)} 只):")
            for ts_code, pos in list(self.positions.items())[:5]:
                print(f"    {ts_code}: {pos['shares']} 股 @ {pos['buy_price']:.2f}")
        
        print("=" * 60)


def main():
    """主函数"""
    backtest = RollingBacktest(
        initial_capital=100000,
        top_n=10,  # 每月买入10只
        forward_days=20,
        threshold=0.05
    )
    
    # 运行回测
    # 用2018-2019的数据训练，2020年开始回测
    equity_curve, trades = backtest.run(
        data_start='20180101',
        train_end='20191231'
    )
    
    # 保存结果
    if equity_curve:
        pd.DataFrame(equity_curve).to_csv('equity_curve.csv', index=False)
        print("\n收益曲线已保存到 equity_curve.csv")
    
    if trades:
        pd.DataFrame(trades).to_csv('trades.csv', index=False)
        print("交易记录已保存到 trades.csv")


if __name__ == "__main__":
    main()

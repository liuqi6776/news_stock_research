"""
滚动训练回测脚本 V3 - 优化版

优化内容:
1. 修复熊市交易逻辑 (熊市仓位50%，不再错过反弹)
2. 调整止损止盈 (止损-15%, 止盈+25%, 新增移动止盈)
3. 增加流动性筛选 (成交额>500万)
4. 增加波动率筛选 (排除极端波动股票)
5. 提高选股阈值 (上涨概率>0.65)
"""

import os
import sys
import warnings
warnings.filterwarnings('ignore')

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
from infra_data.storage import DataStorage
from features.alpha_factors import calculate_alpha101_factors
from features.labeling import simple_return_labeling
from processing.cleaner import filter_stock_codes
from processing.merger import merge_dataframes


class RollingBacktestV3:
    """
    滚动训练回测 V3 - 优化版
    
    优化点:
    1. 熊市也积极交易
    2. 更合理的止损止盈
    3. 移动止盈保护利润
    4. 流动性和波动率筛选
    """
    
    def __init__(
        self,
        initial_capital: float = 100000,
        top_n: int = 15,                  # 增加持仓数量
        commission_rate: float = 0.0003,
        stamp_duty: float = 0.0005,
        forward_days: int = 20,
        threshold: float = 0.05,
        # 优化后的风控参数
        stop_loss: float = 0.15,          # 从10%改为15%
        take_profit: float = 0.25,        # 从30%改为25%
        trailing_stop: float = 0.08,      # 新增: 移动止盈回撤8%
        min_amount: float = 5000000,      # 新增: 最小成交额500万
        max_volatility: float = 0.06,     # 新增: 最大波动率6%
        up_prob_threshold: float = 0.60,  # 提高上涨概率阈值
    ):
        self.initial_capital = initial_capital
        self.top_n = top_n
        self.commission_rate = commission_rate
        self.stamp_duty = stamp_duty
        self.forward_days = forward_days
        self.threshold = threshold
        
        # 风控参数
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.trailing_stop = trailing_stop
        self.min_amount = min_amount
        self.max_volatility = max_volatility
        self.up_prob_threshold = up_prob_threshold
        
        self.storage = DataStorage()
        
        # 状态
        self.capital = initial_capital
        self.positions = {}
        self.equity_curve = []
        self.monthly_returns = []
        self.trades = []
        
        # 市场状态
        self.market_trend = 'NORMAL'
        self.market_volatility = 0.0
    
    def load_and_prepare_data(self, start_date: str, end_date: str) -> pd.DataFrame:
        """加载并准备数据"""
        print("加载数据...")
        
        daily = self.storage.load_daily_data(start_date, end_date)
        
        if daily.empty:
            raise ValueError(f"没有找到 {start_date} - {end_date} 的数据")
        
        other = self.storage.load_daily_basic(start_date, end_date)
        skill = self.storage.load_technical_factors(start_date, end_date)
        
        print(f"日线数据: {len(daily)} 条")
        
        print("合并数据...")
        dfs = [daily]
        if not other.empty:
            dfs.append(other)
        if not skill.empty:
            dfs.append(skill)
        
        df = merge_dataframes(dfs)
        df = filter_stock_codes(df, patterns=['^60', '^00'])
        
        # 新增: 计算波动率
        print("计算波动率...")
        df = df.sort_values(['ts_code', 'trade_date'])
        df['volatility_20d'] = df.groupby('ts_code')['pct_chg'].transform(
            lambda x: x.rolling(20, min_periods=5).std()
        )
        
        print("计算 Alpha 因子...")
        df = calculate_alpha101_factors(df)
        
        print("生成标签...")
        df = simple_return_labeling(df, forward_days=self.forward_days, threshold=self.threshold)
        
        df = df.sort_values(['ts_code', 'trade_date'])
        
        return df
    
    def calculate_market_state(self, df: pd.DataFrame, current_date) -> tuple:
        """
        优化后的市场状态计算
        
        返回: (trend, volatility, position_ratio)
        """
        lookback_days = 60
        
        recent_data = df[df['trade_date'] <= current_date].copy()
        if len(recent_data) < lookback_days:
            return 'NORMAL', 0.02, 0.8
        
        daily_returns = recent_data.groupby('trade_date')['pct_chg'].mean()
        daily_returns = daily_returns.tail(lookback_days)
        
        ma20 = daily_returns.tail(20).mean()
        ma60 = daily_returns.mean()
        
        volatility = daily_returns.std()
        
        # 判断趋势
        if ma20 > ma60 * 1.05:  # 放宽牛市判断
            trend = 'BULL'
        elif ma20 < ma60 * 0.95:  # 放宽熊市判断
            trend = 'BEAR'
        else:
            trend = 'NORMAL'
        
        # 优化后的仓位比例
        # 核心改变: 熊市也保持50%仓位
        if trend == 'BULL':
            base_ratio = 0.95
        elif trend == 'BEAR':
            base_ratio = 0.50  # 原0.3改为0.5
        else:
            base_ratio = 0.80  # 原0.7改为0.8
        
        # 波动率调整
        vol_adjustment = max(0.6, min(1.0, 0.025 / max(volatility, 0.01)))
        
        position_ratio = base_ratio * vol_adjustment
        
        return trend, volatility, position_ratio
    
    def check_stop_loss_take_profit(self, prices: dict, date) -> list:
        """
        优化后的止损止盈检查 - 包含移动止盈
        """
        to_sell = []
        
        for ts_code, pos in list(self.positions.items()):
            if ts_code not in prices:
                continue
            
            current_price = prices[ts_code]
            buy_price = pos['buy_price']
            max_price = pos.get('max_price', buy_price)
            
            # 更新最高价
            if current_price > max_price:
                self.positions[ts_code]['max_price'] = current_price
                max_price = current_price
            
            # 计算收益率
            pnl_pct = (current_price - buy_price) / buy_price
            
            # 止损
            if pnl_pct <= -self.stop_loss:
                to_sell.append({
                    'ts_code': ts_code,
                    'reason': 'STOP_LOSS',
                    'pnl_pct': pnl_pct
                })
            # 止盈
            elif pnl_pct >= self.take_profit:
                to_sell.append({
                    'ts_code': ts_code,
                    'reason': 'TAKE_PROFIT',
                    'pnl_pct': pnl_pct
                })
            # 移动止盈: 从最高点回撤超过8%
            elif max_price > buy_price * 1.15:  # 盈利超过15%才启用移动止盈
                drawdown_from_peak = (max_price - current_price) / max_price
                if drawdown_from_peak >= self.trailing_stop:
                    to_sell.append({
                        'ts_code': ts_code,
                        'reason': 'TRAILING_STOP',
                        'pnl_pct': pnl_pct
                    })
        
        return to_sell
    
    def get_features(self, df: pd.DataFrame) -> list:
        """获取特征列"""
        exclude = ['ts_code', 'trade_date', 'return_label', 'future_return', 
                   'pct_label', 'return_rank', 'tb_label', 'volatility_20d']
        
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
        
        y = (y == 1).astype(int)
        
        for col in X.columns:
            X[col] = pd.to_numeric(X[col], errors='coerce').fillna(0)
        
        X = X.replace([np.inf, -np.inf], 0)
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        model = xgb.XGBClassifier(
            n_estimators=150,          # 增加树的数量
            max_depth=6,               # 增加深度
            learning_rate=0.08,        # 降低学习率
            subsample=0.8,             # 新增: 防止过拟合
            colsample_bytree=0.8,      # 新增: 防止过拟合
            random_state=42,
            use_label_encoder=False,
            eval_metric='logloss'
        )
        model.fit(X_scaled, y)
        
        return model, scaler
    
    def predict_stocks(self, model, scaler, pred_df: pd.DataFrame, features: list, date) -> pd.DataFrame:
        """预测股票并应用筛选条件"""
        day_data = pred_df[pred_df['trade_date'] == date].copy()
        
        if day_data.empty:
            return pd.DataFrame()
        
        X = day_data[features].fillna(0)
        for col in X.columns:
            X[col] = pd.to_numeric(X[col], errors='coerce').fillna(0)
        X = X.replace([np.inf, -np.inf], 0)
        
        X_scaled = scaler.transform(X)
        
        proba = model.predict_proba(X_scaled)
        day_data['up_proba'] = proba[:, 1]
        
        # 筛选条件
        recommendations = day_data[
            (day_data['up_proba'] > self.up_prob_threshold) &    # 上涨概率
            (day_data['pct_chg'] < 9.5) &                        # 非涨停
            (day_data['pct_chg'] > -9.5) &                       # 非跌停
            (day_data['amount'] > self.min_amount / 10000) &     # 流动性 (amount单位万)
            (day_data['volatility_20d'].fillna(0.03) < self.max_volatility)  # 波动率
        ]
        
        if recommendations.empty:
            # 如果筛选太严格，放宽条件
            recommendations = day_data[
                (day_data['up_proba'] > 0.55) &
                (day_data['pct_chg'] < 9.5) &
                (day_data['pct_chg'] > -9.5)
            ]
        
        top_stocks = recommendations.sort_values('up_proba', ascending=False).head(self.top_n)
        
        return top_stocks[['ts_code', 'close', 'up_proba', 'amount']]
    
    def sell_stock(self, ts_code: str, price: float, date, reason: str = 'NORMAL'):
        """卖出股票"""
        if ts_code not in self.positions:
            return
        
        pos = self.positions[ts_code]
        revenue = pos['shares'] * price
        commission = max(revenue * self.commission_rate, 5)
        stamp = revenue * self.stamp_duty
        net_revenue = revenue - commission - stamp
        
        self.capital += net_revenue
        
        pnl = net_revenue - pos['shares'] * pos['buy_price']
        self.trades.append({
            'date': date,
            'ts_code': ts_code,
            'type': 'SELL',
            'reason': reason,
            'price': price,
            'shares': pos['shares'],
            'pnl': pnl
        })
        
        del self.positions[ts_code]
    
    def buy_stock(self, ts_code: str, price: float, amount: float, date):
        """买入股票"""
        shares = int(amount / price / 100) * 100
        
        if shares < 100:
            return False
        
        cost = shares * price
        commission = max(cost * self.commission_rate, 5)
        total_cost = cost + commission
        
        if total_cost > self.capital:
            return False
        
        self.capital -= total_cost
        self.positions[ts_code] = {
            'shares': shares,
            'buy_price': price,
            'buy_date': date,
            'max_price': price  # 新增: 记录最高价用于移动止盈
        }
        
        self.trades.append({
            'date': date,
            'ts_code': ts_code,
            'type': 'BUY',
            'reason': 'SIGNAL',
            'price': price,
            'shares': shares,
            'pnl': 0
        })
        
        return True
    
    def get_portfolio_value(self, prices: dict) -> float:
        """计算组合价值"""
        value = self.capital
        for ts_code, pos in self.positions.items():
            if ts_code in prices:
                value += pos['shares'] * prices[ts_code]
        return value
    
    def run(self, data_start: str = '20180101', train_end: str = '20191231'):
        """运行滚动回测"""
        today = datetime.now().strftime('%Y%m%d')
        
        print("=" * 60)
        print("滚动训练回测 V3 - 优化版")
        print("=" * 60)
        print(f"数据范围: {data_start} - {today}")
        print(f"初始训练: {data_start} - {train_end}")
        print(f"初始资金: CNY {self.initial_capital:,.0f}")
        print(f"每月买入: 前 {self.top_n} 只股票")
        print(f"\n优化后风控参数:")
        print(f"  止损线: {self.stop_loss*100:.0f}%")
        print(f"  止盈线: {self.take_profit*100:.0f}%")
        print(f"  移动止盈: {self.trailing_stop*100:.0f}%回撤")
        print(f"  最小成交额: {self.min_amount/10000:.0f}万")
        print(f"  最大波动率: {self.max_volatility*100:.0f}%")
        print(f"  上涨概率阈值: {self.up_prob_threshold*100:.0f}%")
        print("=" * 60)
        
        # 加载数据
        df = self.load_and_prepare_data(data_start, today)
        
        print(f"\n数据准备完成:")
        print(f"  总记录: {len(df)}")
        print(f"  股票数: {df['ts_code'].nunique()}")
        print(f"  日期范围: {df['trade_date'].min()} - {df['trade_date'].max()}")
        
        features = self.get_features(df)
        print(f"  特征数: {len(features)}")
        
        # 初始训练
        print("\n初始模型训练...")
        train_end_dt = pd.to_datetime(train_end)
        train_df = df[df['trade_date'] <= train_end_dt].copy()
        train_df = train_df[train_df['return_label'].notna()]
        
        print(f"  训练样本: {len(train_df)}")
        
        model, scaler = self.train_model(train_df, features)
        print("  模型训练完成")
        
        # 获取回测期间的月份列表
        backtest_df = df[df['trade_date'] > train_end_dt].copy()
        backtest_df['year_month'] = backtest_df['trade_date'].dt.to_period('M')
        months = sorted(backtest_df['year_month'].unique())
        
        print(f"\n开始回测 ({len(months)} 个月)...")
        
        # 统计
        stop_loss_count = 0
        take_profit_count = 0
        trailing_stop_count = 0
        bear_market_months = 0
        
        # 按月回测
        for i, month in enumerate(tqdm(months, desc="回测进度")):
            month_df = backtest_df[backtest_df['year_month'] == month]
            
            if month_df.empty:
                continue
            
            first_day = month_df['trade_date'].min()
            last_day = month_df['trade_date'].max()
            
            # 获取价格
            first_day_data = month_df[month_df['trade_date'] == first_day]
            prices = dict(zip(first_day_data['ts_code'], first_day_data['close']))
            
            # 检查止损止盈
            to_sell = self.check_stop_loss_take_profit(prices, first_day)
            for item in to_sell:
                if item['ts_code'] in prices:
                    self.sell_stock(item['ts_code'], prices[item['ts_code']], first_day, item['reason'])
                    if item['reason'] == 'STOP_LOSS':
                        stop_loss_count += 1
                    elif item['reason'] == 'TAKE_PROFIT':
                        take_profit_count += 1
                    elif item['reason'] == 'TRAILING_STOP':
                        trailing_stop_count += 1
            
            # 计算市场状态
            trend, volatility, position_ratio = self.calculate_market_state(df, first_day)
            self.market_trend = trend
            self.market_volatility = volatility
            
            if trend == 'BEAR':
                bear_market_months += 1
            
            # 关键改变: 熊市也进行交易，只是仓位降低
            recommendations = self.predict_stocks(model, scaler, df, features, first_day)
            rec_codes = set(recommendations['ts_code'].tolist()) if not recommendations.empty else set()
            
            # 卖出不在推荐列表的持仓
            for ts_code in list(self.positions.keys()):
                if ts_code not in rec_codes and ts_code in prices:
                    self.sell_stock(ts_code, prices[ts_code], first_day, 'ROTATE')
            
            # 买入推荐股票
            if not recommendations.empty:
                available = self.capital * position_ratio * 0.95
                per_stock = available / len(recommendations)
                
                for _, row in recommendations.iterrows():
                    if row['ts_code'] not in self.positions:
                        self.buy_stock(row['ts_code'], row['close'], per_stock, first_day)
            
            # 记录月末组合价值
            end_prices = dict(zip(
                month_df[month_df['trade_date'] == last_day]['ts_code'],
                month_df[month_df['trade_date'] == last_day]['close']
            ))
            portfolio_value = self.get_portfolio_value(end_prices)
            self.equity_curve.append({
                'month': str(month),
                'date': last_day,
                'value': portfolio_value,
                'trend': trend,
                'volatility': volatility,
                'position_ratio': position_ratio
            })
            
            # 每季度重新训练模型
            if (i + 1) % 3 == 0:
                new_train_df = df[df['trade_date'] <= last_day].copy()
                new_train_df = new_train_df[new_train_df['return_label'].notna()]
                model, scaler = self.train_model(new_train_df, features)
        
        # 输出结果
        self.print_results(stop_loss_count, take_profit_count, trailing_stop_count, bear_market_months)
        
        return self.equity_curve, self.trades
    
    def print_results(self, stop_loss_count, take_profit_count, trailing_stop_count, bear_market_months):
        """打印回测结果"""
        print("\n" + "=" * 60)
        print("回测结果 (V3 优化版)")
        print("=" * 60)
        
        if not self.equity_curve:
            print("没有回测数据")
            return
        
        initial = self.initial_capital
        final = self.equity_curve[-1]['value']
        
        total_return = (final / initial - 1) * 100
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
        
        # 计算Sharpe比率
        returns = []
        for i in range(1, len(values)):
            ret = (values[i] / values[i-1] - 1) * 100
            returns.append(ret)
        
        if returns:
            avg_return = np.mean(returns)
            std_return = np.std(returns)
            sharpe = (avg_return * 12) / (std_return * np.sqrt(12)) if std_return > 0 else 0
        else:
            sharpe = 0
        
        # 胜率
        buy_trades = [t for t in self.trades if t['type'] == 'BUY']
        sell_trades = [t for t in self.trades if t['type'] == 'SELL']
        wins = sum(1 for t in sell_trades if t['pnl'] > 0)
        win_rate = (wins / len(sell_trades) * 100) if sell_trades else 0
        
        print(f"\n[收益指标]")
        print(f"  初始资金:     CNY {initial:,.0f}")
        print(f"  最终资金:     CNY {final:,.0f}")
        print(f"  总收益率:     {total_return:.2f}%")
        print(f"  年化收益率:   {annual_return:.2f}%")
        print(f"  最大回撤:     {max_dd:.2f}%")
        print(f"  Sharpe比率:   {sharpe:.2f}")
        print(f"  胜率:         {win_rate:.2f}%")
        
        print(f"\n[交易统计]")
        print(f"  回测月份:     {months} 个月")
        print(f"  总交易次数:   {len(self.trades)}")
        print(f"  买入次数:     {len(buy_trades)}")
        print(f"  卖出次数:     {len(sell_trades)}")
        
        print(f"\n[风控统计]")
        print(f"  止损触发:     {stop_loss_count} 次")
        print(f"  止盈触发:     {take_profit_count} 次")
        print(f"  移动止盈:     {trailing_stop_count} 次")
        print(f"  熊市月份:     {bear_market_months} 个月")
        
        if self.positions:
            print(f"\n[当前持仓] ({len(self.positions)} 只):")
            for ts_code, pos in list(self.positions.items())[:5]:
                print(f"    {ts_code}: {pos['shares']} 股 @ {pos['buy_price']:.2f}")
        
        print("=" * 60)


def main():
    """主函数"""
    backtest = RollingBacktestV3(
        initial_capital=100000,
        top_n=15,                  # 增加持仓数量
        forward_days=20,
        threshold=0.05,
        # 优化后的风控参数
        stop_loss=0.15,            # 15% 止损
        take_profit=0.25,          # 25% 止盈
        trailing_stop=0.08,        # 8% 移动止盈
        min_amount=5000000,        # 500万最小成交额
        max_volatility=0.06,       # 6% 最大波动率
        up_prob_threshold=0.60,    # 60% 上涨概率阈值
    )
    
    equity_curve, trades = backtest.run(
        data_start='20180101',
        train_end='20191231'
    )
    
    # 保存结果
    if equity_curve:
        pd.DataFrame(equity_curve).to_csv('equity_curve_v3.csv', index=False)
        print("\n收益曲线已保存到 equity_curve_v3.csv")
    
    if trades:
        pd.DataFrame(trades).to_csv('trades_v3.csv', index=False)
        print("交易记录已保存到 trades_v3.csv")


if __name__ == "__main__":
    main()

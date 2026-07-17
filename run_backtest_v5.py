"""
滚动训练回测脚本 V5 - 增强特征版

基于V2增强版，新增:
1. 使用148个增强特征 (Alpha101/191、技术指标、资金流向、筹码分布)
2. 优化的模型参数和特征选择
3. 更智能的风控机制
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
from sklearn.feature_selection import SelectKBest, f_classif
try:
    import xgboost as xgb
except ImportError:
    print("请安装 xgboost: pip install xgboost")
    sys.exit(1)

from config.settings import settings
from infra_data.fetcher import DataFetcher
from infra_data.storage import DataStorage
from features.enhanced_factors import calculate_all_enhanced_features
from features.labeling import simple_return_labeling
from processing.cleaner import filter_stock_codes
from processing.merger import merge_dataframes


class RollingBacktestV5:
    """
    滚动训练回测 V5 - 增强特征版
    
    特点:
    - 148个增强特征 (Alpha101/191、技术指标、资金流向、筹码)
    - 智能特征选择 (Top 50特征)
    - 优化的XGBoost参数
    - 完善的风控机制
    """
    
    def __init__(
        self,
        initial_capital: float = 100000,
        top_n: int = 10,
        commission_rate: float = 0.0003,
        stamp_duty: float = 0.0005,
        forward_days: int = 20,
        threshold: float = 0.05,
        # 风控参数
        stop_loss: float = 0.08,       # 止损线 8%
        take_profit: float = 0.25,     # 止盈线 25%
        trailing_stop: float = 0.05,   # 移动止损 5%
        # 选股参数
        up_prob_threshold: float = 0.55,  # 上涨概率阈值
        max_features: int = 80,           # 最大特征数
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
        self.up_prob_threshold = up_prob_threshold
        self.max_features = max_features
        
        self.storage = DataStorage()
        
        # 状态
        self.capital = initial_capital
        self.positions = {}
        self.equity_curve = []
        self.monthly_returns = []
        self.trades = []
        self.high_watermarks = {}  # 记录持仓最高价
        
        # 市场状态
        self.market_trend = 'NORMAL'
        self.market_volatility = 0.0
    
    def load_and_prepare_data(self, start_date: str, end_date: str) -> pd.DataFrame:
        """加载并准备数据 (含资金流向和筹码)"""
        print("加载数据...")
        
        # 加载基础行情
        daily = self.storage.load_daily_data(start_date, end_date)
        if daily.empty:
            raise ValueError(f"没有找到 {start_date} - {end_date} 的数据")
        
        other = self.storage.load_daily_basic(start_date, end_date)
        skill = self.storage.load_technical_factors(start_date, end_date)
        
        # 加载资金流向和筹码数据
        print("加载资金流向数据...")
        money_flow = self.storage.load_money_flow(start_date, end_date)
        
        print("加载筹码分布数据...")
        chip_data = self.storage.load_chip_data(start_date, end_date)
        
        print(f"日线数据: {len(daily)} 条")
        print(f"资金流向: {len(money_flow)} 条")
        print(f"筹码分布: {len(chip_data)} 条")
        
        # 合并基础数据
        print("合并数据...")
        dfs = [daily]
        if not other.empty:
            dfs.append(other)
        if not skill.empty:
            dfs.append(skill)
        
        df = merge_dataframes(dfs)
        df = filter_stock_codes(df, patterns=['^60', '^00'])
        
        # 计算增强特征 (含资金流向和筹码)
        print("计算增强特征 (148个)...")
        df = calculate_all_enhanced_features(df, money_flow, chip_data)
        
        # 生成标签
        print("生成标签...")
        df = simple_return_labeling(df, forward_days=self.forward_days, threshold=self.threshold)
        
        df = df.sort_values(['ts_code', 'trade_date'])
        
        return df
    
    def calculate_market_state(self, df: pd.DataFrame, current_date) -> tuple:
        """计算市场状态"""
        lookback_days = 60
        
        recent_data = df[df['trade_date'] <= current_date].copy()
        if len(recent_data) < lookback_days:
            return 'NORMAL', 0.02, 0.8
        
        daily_returns = recent_data.groupby('trade_date')['pct_chg'].mean()
        daily_returns = daily_returns.tail(lookback_days)
        
        ma20 = daily_returns.tail(20).mean()
        ma60 = daily_returns.mean()
        volatility = daily_returns.std()
        
        if ma20 > ma60 * 1.1:
            trend = 'BULL'
        elif ma20 < ma60 * 0.9:
            trend = 'BEAR'
        else:
            trend = 'NORMAL'
        
        # V5: 熊市也保持较高仓位 (不错过反弹)
        if trend == 'BULL':
            base_ratio = 0.95
        elif trend == 'BEAR':
            base_ratio = 0.5  # 熊市50%仓位
        else:
            base_ratio = 0.8
        
        vol_adjustment = max(0.6, min(1.0, 0.025 / max(volatility, 0.01)))
        position_ratio = base_ratio * vol_adjustment
        
        return trend, volatility, position_ratio
    
    def check_stop_loss_take_profit(self, prices: dict, date) -> list:
        """检查止损止盈 (含移动止损)"""
        to_sell = []
        
        for ts_code, pos in list(self.positions.items()):
            if ts_code not in prices:
                continue
            
            current_price = prices[ts_code]
            buy_price = pos['buy_price']
            
            # 更新最高价
            if ts_code not in self.high_watermarks:
                self.high_watermarks[ts_code] = current_price
            else:
                self.high_watermarks[ts_code] = max(self.high_watermarks[ts_code], current_price)
            
            pnl_pct = (current_price - buy_price) / buy_price
            
            # 止损
            if pnl_pct <= -self.stop_loss:
                to_sell.append({'ts_code': ts_code, 'reason': 'STOP_LOSS', 'pnl_pct': pnl_pct})
            # 止盈
            elif pnl_pct >= self.take_profit:
                to_sell.append({'ts_code': ts_code, 'reason': 'TAKE_PROFIT', 'pnl_pct': pnl_pct})
            # 移动止损: 从最高点回撤超过trailing_stop
            elif pnl_pct > 0:
                high_price = self.high_watermarks[ts_code]
                drawdown_from_high = (high_price - current_price) / high_price
                if drawdown_from_high >= self.trailing_stop:
                    to_sell.append({'ts_code': ts_code, 'reason': 'TRAILING_STOP', 'pnl_pct': pnl_pct})
        
        return to_sell
    
    def get_features(self, df: pd.DataFrame) -> list:
        """获取特征列"""
        exclude = ['ts_code', 'trade_date', 'return_label', 'future_return', 
                   'pct_label', 'return_rank', 'tb_label', 'year_month']
        
        features = []
        for col in df.columns:
            if col in exclude:
                continue
            if col.startswith('_'):  # 排除临时列
                continue
            if df[col].dtype not in ['object', 'datetime64[ns]']:
                features.append(col)
        
        return features
    
    def select_features(self, X: pd.DataFrame, y: pd.Series, max_features: int = 80) -> list:
        """使用统计方法选择最重要的特征"""
        X_filled = X.fillna(0).replace([np.inf, -np.inf], 0)
        
        try:
            selector = SelectKBest(f_classif, k=min(max_features, len(X.columns)))
            selector.fit(X_filled, y)
            
            scores = pd.Series(selector.scores_, index=X.columns)
            scores = scores.replace([np.inf, -np.inf, np.nan], 0)
            
            selected = scores.nlargest(max_features).index.tolist()
            return selected
        except Exception as e:
            print(f"  特征选择失败，使用全部特征: {e}")
            return X.columns.tolist()
    
    def train_model(self, train_df: pd.DataFrame, features: list, select_features: bool = True):
        """训练模型"""
        X = train_df[features].fillna(0)
        y = train_df['return_label'].copy()
        y = (y == 1).astype(int)
        
        for col in X.columns:
            X[col] = pd.to_numeric(X[col], errors='coerce').fillna(0)
        X = X.replace([np.inf, -np.inf], 0)
        
        # 特征选择
        if select_features and len(features) > self.max_features:
            selected_features = self.select_features(X, y, self.max_features)
            X = X[selected_features]
            print(f"  选择 {len(selected_features)}/{len(features)} 个特征")
        else:
            selected_features = features
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # V5: 优化的XGBoost参数
        model = xgb.XGBClassifier(
            n_estimators=150,
            max_depth=6,
            learning_rate=0.08,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42,
            use_label_encoder=False,
            eval_metric='logloss',
            n_jobs=-1
        )
        model.fit(X_scaled, y)
        
        return model, scaler, selected_features
    
    def predict_stocks(self, model, scaler, pred_df: pd.DataFrame, features: list, date) -> pd.DataFrame:
        """预测股票涨跌并返回推荐列表"""
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
        
        # V5: 使用资金流向和筹码信号过滤
        # 优先选择主力资金流入 + 获利盘压力小的股票
        if 'mf_signal' in day_data.columns:
            day_data['score'] = day_data['up_proba'] + 0.1 * day_data['mf_signal'].fillna(0)
        else:
            day_data['score'] = day_data['up_proba']
        
        if 'profit_pressure' in day_data.columns:
            day_data['score'] -= 0.05 * day_data['profit_pressure'].fillna(0)
        
        # 过滤条件
        day_data = day_data[day_data['up_proba'] > self.up_prob_threshold]
        
        # 排序选股
        top_stocks = day_data.sort_values('score', ascending=False).head(self.top_n)
        
        return top_stocks[['ts_code', 'close', 'up_proba', 'score']]
    
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
        if ts_code in self.high_watermarks:
            del self.high_watermarks[ts_code]
    
    def buy_stock(self, ts_code: str, price: float, amount: float, date):
        """买入股票"""
        if price <= 0:
            return False
            
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
            'buy_date': date
        }
        self.high_watermarks[ts_code] = price
        
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
        print("滚动训练回测 V5 - 增强特征版 (148个特征)")
        print("=" * 60)
        print(f"数据范围: {data_start} - {today}")
        print(f"初始训练: {data_start} - {train_end}")
        print(f"初始资金: CNY {self.initial_capital:,.0f}")
        print(f"每月买入: 前 {self.top_n} 只股票")
        print(f"\n风控参数:")
        print(f"  止损线: {self.stop_loss*100:.0f}%")
        print(f"  止盈线: {self.take_profit*100:.0f}%")
        print(f"  移动止损: {self.trailing_stop*100:.0f}%")
        print(f"  上涨概率阈值: {self.up_prob_threshold}")
        print("=" * 60)
        
        # 加载数据
        df = self.load_and_prepare_data(data_start, today)
        
        print(f"\n数据准备完成:")
        print(f"  总记录: {len(df)}")
        print(f"  股票数: {df['ts_code'].nunique()}")
        print(f"  日期范围: {df['trade_date'].min()} - {df['trade_date'].max()}")
        
        features = self.get_features(df)
        print(f"  可用特征数: {len(features)}")
        
        # 初始训练
        print("\n初始模型训练...")
        train_end_dt = pd.to_datetime(train_end)
        train_df = df[df['trade_date'] <= train_end_dt].copy()
        train_df = train_df[train_df['return_label'].notna()]
        
        print(f"  训练样本: {len(train_df)}")
        
        model, scaler, selected_features = self.train_model(train_df, features)
        print("  模型训练完成")
        
        # 获取回测期间的日期列表
        backtest_df = df[df['trade_date'] > train_end_dt].copy()
        trade_dates = sorted(backtest_df['trade_date'].unique())
        
        # 转换为每周一次的调仓日 (每周第一个交易日)
        backtest_df['is_week_start'] = backtest_df['trade_date'].dt.dayofweek
        weekly_dates = backtest_df.groupby([backtest_df['trade_date'].dt.year, backtest_df['trade_date'].dt.isocalendar().week])['trade_date'].min().tolist()
        
        print(f"\n开始回测，共 {len(weekly_dates)} 个交易周...")
        
        # 统计
        stop_loss_count = 0
        take_profit_count = 0
        trailing_stop_count = 0
        bear_market_weeks = 0
        
        # 按周回测
        for i, current_date in enumerate(tqdm(weekly_dates, desc="回测进度")):
            # 获取当日所有相关数据
            day_data = df[df['trade_date'] == current_date]
            if day_data.empty:
                continue
                
            prices = dict(zip(day_data['ts_code'], day_data['close']))
            
            # 风控1: 每日检查止损止盈 (虽然是周频调仓，但止损应实时或每日)
            # 这里简化为调仓日检查，模拟实战中盘中或每日收盘检查
            to_sell = self.check_stop_loss_take_profit(prices, current_date)
            for item in to_sell:
                self.sell_stock(item['ts_code'], prices[item['ts_code']], current_date, item['reason'])
                if item['reason'] == 'STOP_LOSS': stop_loss_count += 1
                elif item['reason'] == 'TAKE_PROFIT': take_profit_count += 1
                elif item['reason'] == 'TRAILING_STOP': trailing_stop_count += 1
            
            # 风控2: 市场状态
            trend, volatility, position_ratio = self.calculate_market_state(df, current_date)
            if trend == 'BEAR': bear_market_weeks += 1
            
            # 正常调仓
            recommendations = self.predict_stocks(model, scaler, df, selected_features, current_date)
            rec_codes = set(recommendations['ts_code'].tolist()) if not recommendations.empty else set()
            
            # 卖出不再推荐或市场不好的股票
            for ts_code in list(self.positions.keys()):
                reason = None
                if ts_code not in rec_codes: reason = 'ROTATE'
                elif trend == 'BEAR' and i % 2 == 0: reason = 'BEAR_MARKET' # 熊市节奏性减仓
                
                if reason and ts_code in prices:
                    self.sell_stock(ts_code, prices[ts_code], current_date, reason)
            
            # 买入
            if not recommendations.empty:
                target_available = self.capital * position_ratio * 0.95
                slots_available = self.top_n - len(self.positions)
                
                if slots_available > 0:
                    per_stock = target_available / self.top_n
                    for _, row in recommendations.iterrows():
                        if row['ts_code'] not in self.positions:
                            self.buy_stock(row['ts_code'], row['close'], per_stock, current_date)
            
            # 记录价值
            portfolio_value = self.get_portfolio_value(prices)
            self.equity_curve.append({
                'date': current_date,
                'value': portfolio_value,
                'trend': trend,
                'position_ratio': position_ratio
            })
            
            # 每月重新训练 (约4周)
            if (i + 1) % 4 == 0:
                new_train_df = df[df['trade_date'] <= current_date].copy()
                new_train_df = new_train_df[new_train_df['return_label'].notna()]
                model, scaler, selected_features = self.train_model(new_train_df, features, select_features=True)
        
        # 输出结果
        self.print_results(stop_loss_count, take_profit_count, trailing_stop_count, bear_market_weeks)
        
        return self.equity_curve, self.trades
    
    def print_results(self, stop_loss_count, take_profit_count, trailing_stop_count, bear_market_months):
        """打印回测结果"""
        print("\n" + "=" * 60)
        print("回测结果 (V5 增强特征版)")
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
        
        # Sharpe Ratio (简化计算)
        returns = []
        for i in range(1, len(values)):
            ret = (values[i] - values[i-1]) / values[i-1]
            returns.append(ret)
        
        if returns:
            avg_return = np.mean(returns)
            std_return = np.std(returns)
            sharpe = (avg_return * 12) / (std_return * np.sqrt(12) + 0.0001)
        else:
            sharpe = 0
        
        # 胜率
        buy_trades = [t for t in self.trades if t['type'] == 'BUY']
        sell_trades = [t for t in self.trades if t['type'] == 'SELL']
        wins = sum(1 for t in sell_trades if t['pnl'] > 0)
        win_rate = (wins / len(sell_trades) * 100) if sell_trades else 0
        
        print(f"\n【收益指标】")
        print(f"  初始资金:     CNY {initial:,.0f}")
        print(f"  最终资金:     CNY {final:,.0f}")
        print(f"  总收益率:     {total_return:.2f}%")
        print(f"  年化收益率:   {annual_return:.2f}%")
        print(f"  Sharpe比率:   {sharpe:.2f}")
        print(f"  最大回撤:     {max_dd:.2f}%")
        print(f"  胜率:         {win_rate:.2f}%")
        
        print(f"\n【交易统计】")
        print(f"  回测月份:     {months} 个月")
        print(f"  总交易次数:   {len(self.trades)}")
        print(f"  买入次数:     {len(buy_trades)}")
        print(f"  卖出次数:     {len(sell_trades)}")
        
        print(f"\n【风控统计】")
        print(f"  止损触发:     {stop_loss_count} 次")
        print(f"  止盈触发:     {take_profit_count} 次")
        print(f"  移动止损:     {trailing_stop_count} 次")
        print(f"  熊市月份:     {bear_market_months} 个月")
        
        if self.positions:
            print(f"\n【当前持仓】({len(self.positions)} 只):")
            for ts_code, pos in list(self.positions.items())[:5]:
                print(f"    {ts_code}: {pos['shares']} 股 @ {pos['buy_price']:.2f}")
        
        print("=" * 60)


def main():
    """主函数"""
    backtest = RollingBacktestV5(
        initial_capital=100000,
        top_n=10,
        forward_days=20,
        threshold=0.05,
        # 风控参数
        stop_loss=0.08,      # 8% 止损
        take_profit=0.25,    # 25% 止盈
        trailing_stop=0.05,  # 5% 移动止损
        up_prob_threshold=0.55,
        max_features=80,
    )
    
    equity_curve, trades = backtest.run(
        data_start='20180101',
        train_end='20191231'
    )
    
    # 保存结果
    output_dir = r'C:\Users\liuqi\.gemini\antigravity\scratch'
    
    if equity_curve:
        pd.DataFrame(equity_curve).to_csv(f'{output_dir}/equity_curve_v5.csv', index=False)
        print(f"\n收益曲线已保存到 {output_dir}/equity_curve_v5.csv")
    
    if trades:
        pd.DataFrame(trades).to_csv(f'{output_dir}/trades_v5.csv', index=False)
        print(f"交易记录已保存到 {output_dir}/trades_v5.csv")


if __name__ == "__main__":
    main()

"""
回测引擎模块

提供策略回测功能
"""

from typing import Dict, List, Optional, Callable
import pandas as pd
import numpy as np


class BacktestEngine:
    """
    回测引擎
    
    基于历史数据的策略回测
    
    Example:
        engine = BacktestEngine(initial_capital=100000)
        result = engine.run(df, strategy_func)
    """
    
    def __init__(
        self,
        initial_capital: float = 100000,
        commission_rate: float = 0.0003,
        stamp_duty: float = 0.0005,
        slippage: float = 0.001
    ):
        """
        初始化回测引擎
        
        Args:
            initial_capital: 初始资金
            commission_rate: 佣金费率
            stamp_duty: 印花税（仅卖出）
            slippage: 滑点
        """
        self.initial_capital = initial_capital
        self.commission_rate = commission_rate
        self.stamp_duty = stamp_duty
        self.slippage = slippage
        
        # 状态变量
        self.capital = initial_capital
        self.positions: Dict[str, int] = {}  # {ts_code: shares}
        self.trades: List[Dict] = []
        self.equity_curve: List[float] = []
    
    def reset(self):
        """重置状态"""
        self.capital = self.initial_capital
        self.positions = {}
        self.trades = []
        self.equity_curve = []
    
    def buy(self, ts_code: str, price: float, shares: int, date) -> bool:
        """
        买入
        
        Args:
            ts_code: 股票代码
            price: 买入价格
            shares: 买入股数
            date: 交易日期
        
        Returns:
            是否成功
        """
        # 考虑滑点
        actual_price = price * (1 + self.slippage)
        
        # 计算成本
        cost = actual_price * shares
        commission = max(cost * self.commission_rate, 5)  # 最低 5 元
        total_cost = cost + commission
        
        if total_cost > self.capital:
            return False
        
        self.capital -= total_cost
        self.positions[ts_code] = self.positions.get(ts_code, 0) + shares
        
        self.trades.append({
            'date': date,
            'type': 'BUY',
            'ts_code': ts_code,
            'price': actual_price,
            'shares': shares,
            'cost': total_cost
        })
        
        return True
    
    def sell(self, ts_code: str, price: float, shares: int, date) -> bool:
        """
        卖出
        
        Args:
            ts_code: 股票代码
            price: 卖出价格
            shares: 卖出股数
            date: 交易日期
        
        Returns:
            是否成功
        """
        if self.positions.get(ts_code, 0) < shares:
            return False
        
        # 考虑滑点
        actual_price = price * (1 - self.slippage)
        
        # 计算收入
        revenue = actual_price * shares
        commission = max(revenue * self.commission_rate, 5)
        stamp = revenue * self.stamp_duty
        net_revenue = revenue - commission - stamp
        
        self.capital += net_revenue
        self.positions[ts_code] -= shares
        
        if self.positions[ts_code] == 0:
            del self.positions[ts_code]
        
        self.trades.append({
            'date': date,
            'type': 'SELL',
            'ts_code': ts_code,
            'price': actual_price,
            'shares': shares,
            'revenue': net_revenue
        })
        
        return True
    
    def get_portfolio_value(self, prices: Dict[str, float]) -> float:
        """
        计算组合价值
        
        Args:
            prices: {ts_code: price} 当前价格字典
        
        Returns:
            总资产价值
        """
        value = self.capital
        
        for ts_code, shares in self.positions.items():
            if ts_code in prices:
                value += shares * prices[ts_code]
        
        return value
    
    def run(
        self,
        df: pd.DataFrame,
        strategy: Callable,
        price_col: str = 'close'
    ) -> Dict:
        """
        运行回测
        
        Args:
            df: 股票数据
            strategy: 策略函数，接收 (engine, date, day_data) 参数
            price_col: 价格列名
        
        Returns:
            回测结果
        """
        self.reset()
        
        dates = sorted(df['trade_date'].unique())
        
        for date in dates:
            day_data = df[df['trade_date'] == date]
            
            # 执行策略
            strategy(self, date, day_data)
            
            # 计算当日组合价值
            prices = dict(zip(day_data['ts_code'], day_data[price_col]))
            portfolio_value = self.get_portfolio_value(prices)
            self.equity_curve.append(portfolio_value)
        
        return self.calculate_metrics()
    
    def calculate_metrics(self) -> Dict:
        """计算绩效指标"""
        equity = pd.Series(self.equity_curve)
        returns = equity.pct_change().dropna()
        
        # 总收益率
        total_return = (equity.iloc[-1] / self.initial_capital - 1) * 100
        
        # 年化收益率
        days = len(equity)
        annual_return = ((1 + total_return / 100) ** (252 / days) - 1) * 100 if days > 0 else 0
        
        # 最大回撤
        cummax = equity.cummax()
        drawdown = (cummax - equity) / cummax * 100
        max_drawdown = drawdown.max()
        
        # 夏普比率（假设无风险利率 3%）
        sharpe = ((returns.mean() * 252 - 0.03) / (returns.std() * np.sqrt(252))) if returns.std() > 0 else 0
        
        # 胜率
        buy_trades = [t for t in self.trades if t['type'] == 'BUY']
        sell_trades = [t for t in self.trades if t['type'] == 'SELL']
        
        wins = 0
        for sell in sell_trades:
            # 简化：假设先进先出
            for buy in buy_trades:
                if buy['ts_code'] == sell['ts_code']:
                    if sell['price'] > buy['price']:
                        wins += 1
                    break
        
        win_rate = (wins / len(sell_trades) * 100) if sell_trades else 0
        
        return {
            'total_return': f'{total_return:.2f}%',
            'annual_return': f'{annual_return:.2f}%',
            'max_drawdown': f'{max_drawdown:.2f}%',
            'sharpe_ratio': f'{sharpe:.2f}',
            'win_rate': f'{win_rate:.2f}%',
            'total_trades': len(self.trades),
            'final_capital': f'{equity.iloc[-1]:.2f}'
        }

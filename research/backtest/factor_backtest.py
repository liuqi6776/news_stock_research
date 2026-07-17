"""
因子回测框架 - 基于A股交易规则
包含：涨跌停限制、滑点、交易费用、板块过滤
"""
import pandas as pd
import numpy as np
import os
import sys
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime

# A股交易规则配置
COST_RATE = 0.003       # 交易费用 0.3% 双边
SLIPPAGE = 0.002        # 滑点 0.2%
LIMIT_THRESHOLD = 0.5   # 涨跌停阈值偏移 (10% -> 9.5%)


@dataclass
class BacktestResult:
    """回测结果数据结构"""
    total_return: float
    annual_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    avg_return: float
    n_trades: int
    equity_curve: pd.DataFrame
    trades: pd.DataFrame
    skipped_limit_up: int
    skipped_limit_down: int


class FactorBacktest:
    """因子回测类"""
    
    def __init__(self, 
                 data_dir: str = r'D:\iquant_data\data_v2',
                 start_date: str = '20230101',
                 end_date: str = '20260331'):
        self.data_dir = data_dir
        self.price_dir = os.path.join(data_dir, 'data_day1')
        self.start_date = start_date
        self.end_date = end_date
        
    def is_main_board(self, ts_code: str) -> bool:
        """检查是否为主板/中小板"""
        return (ts_code.startswith('60') or 
                ts_code.startswith('00') or 
                ts_code.startswith('002') or 
                ts_code.startswith('003'))
    
    def get_limit_pct(self, ts_code: str) -> float:
        """获取涨跌停幅度"""
        if ts_code.startswith('688') or ts_code.startswith('689'):
            return 20.0
        elif ts_code.startswith('30') or ts_code.startswith('301'):
            return 20.0
        elif ts_code.startswith('8') or ts_code.startswith('4'):
            return 30.0
        else:
            return 10.0
    
    def load_stock_data(self, ts_code: str) -> Optional[pd.DataFrame]:
        """加载单只股票的历史数据"""
        try:
            # 从parquet文件加载
            files = sorted([f for f in os.listdir(self.price_dir) if f.endswith('.parquet')])
            
            all_data = []
            for f in files:
                date_str = f.replace('.parquet', '')
                if self.start_date <= date_str <= self.end_date:
                    df = pd.read_parquet(os.path.join(self.price_dir, f))
                    if ts_code in df['ts_code'].values:
                        row = df[df['ts_code'] == ts_code].copy()
                        row['trade_date'] = date_str
                        all_data.append(row)
            
            if not all_data:
                return None
                
            result = pd.concat(all_data, ignore_index=True)
            result['trade_date'] = pd.to_datetime(result['trade_date'])
            result = result.sort_values('trade_date')
            return result
            
        except Exception as e:
            print(f"加载股票 {ts_code} 数据失败: {e}")
            return None
    
    def calculate_factors(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算所有因子"""
        from ..factors.technical_factors import calculate_all_factors
        return calculate_all_factors(df)
    
    def generate_signal(self, df: pd.DataFrame, factor_name: str, 
                       threshold: float = 0.5, direction: str = 'long') -> pd.Series:
        """
        基于因子生成交易信号
        
        Parameters:
        -----------
        df : DataFrame
            包含因子的数据
        factor_name : str
            因子名称
        threshold : float
            信号阈值
        direction : str
            'long' 或 'short'
        """
        if factor_name not in df.columns:
            print(f"因子 {factor_name} 不存在")
            return pd.Series(0, index=df.index)
        
        factor_values = df[factor_name]
        
        if direction == 'long':
            signal = (factor_values > threshold).astype(int)
        else:
            signal = (factor_values < threshold).astype(int)
        
        return signal
    
    def run_backtest(self, 
                     ts_code: str,
                     factor_name: str,
                     threshold: float = 0.5,
                     direction: str = 'long',
                     position_size: float = 1.0) -> Optional[BacktestResult]:
        """
        运行单因子回测
        
        Parameters:
        -----------
        ts_code : str
            股票代码
        factor_name : str
            因子名称
        threshold : float
            信号阈值
        direction : str
            交易方向
        position_size : float
            仓位大小 (0-1)
        """
        # 加载数据
        df = self.load_stock_data(ts_code)
        if df is None or len(df) < 60:
            print(f"股票 {ts_code} 数据不足")
            return None
        
        # 计算因子
        df = self.calculate_factors(df)
        
        # 生成信号
        df['signal'] = self.generate_signal(df, factor_name, threshold, direction)
        
        # 回测主循环
        initial_capital = 100000.0
        capital = initial_capital
        equity = [capital]
        trades = []
        
        skipped_limit_up = 0
        skipped_limit_down = 0
        
        position = 0  # 0: 空仓, 1: 持仓
        entry_price = 0
        
        for i in range(1, len(df) - 1):
            curr = df.iloc[i]
            next_day = df.iloc[i + 1]
            
            # 获取涨跌停幅度
            limit_pct = self.get_limit_pct(ts_code)
            
            # 检查是否为主板
            if not self.is_main_board(ts_code):
                continue
            
            # 生成交易信号（基于当天收盘数据）
            signal = curr['signal']
            
            # 买入逻辑
            if position == 0 and signal == 1:
                # 检查次日开盘是否涨停
                t1_open = next_day['open']
                t1_pre = curr['close']
                t1_open_chg = (t1_open - t1_pre) / t1_pre * 100
                
                if t1_open_chg >= (limit_pct - LIMIT_THRESHOLD):
                    skipped_limit_up += 1
                    continue
                
                # 买入价格（考虑滑点）
                buy_price = t1_open * (1 + SLIPPAGE)
                entry_price = buy_price
                position = 1
            
            # 卖出逻辑
            elif position == 1 and signal == 0:
                # 检查次日开盘是否跌停
                t2_open = next_day['open']
                t2_low = next_day['low']
                t2_close = next_day['close']
                
                t2_low_chg = (t2_low - entry_price) / entry_price * 100
                
                if t2_low_chg <= -(limit_pct - LIMIT_THRESHOLD):
                    # 跌停卖出
                    sell_price = min(t2_open, entry_price * (1 - limit_pct/100))
                    skipped_limit_down += 1
                else:
                    sell_price = t2_close
                
                # 卖出价格（考虑滑点）
                sell_price = sell_price * (1 - SLIPPAGE)
                
                # 计算收益
                ret = sell_price / entry_price - 1 - COST_RATE
                capital *= (1 + ret * position_size)
                
                trades.append({
                    'entry_date': df.index[i-1],
                    'exit_date': df.index[i+1],
                    'entry_price': entry_price,
                    'exit_price': sell_price,
                    'return': ret,
                    'pnl': capital - initial_capital
                })
                
                position = 0
                entry_price = 0
            
            equity.append(capital)
        
        # 计算回测指标
        equity_df = pd.DataFrame({
            'date': df['trade_date'][:len(equity)],
            'nav': equity
        })
        
        total_ret = capital / initial_capital - 1
        years = len(equity_df) / 252.0
        ann_ret = (1 + total_ret) ** (1 / years) - 1 if years > 0 else 0
        
        df_ret = equity_df['nav'].pct_change()
        vol = df_ret.std() * np.sqrt(252)
        sharpe = ann_ret / vol if vol > 0 else 0
        
        mdd = ((equity_df['nav'] - equity_df['nav'].cummax()) / equity_df['nav'].cummax()).min()
        
        trades_df = pd.DataFrame(trades)
        if len(trades_df) > 0:
            win_rate = (trades_df['return'] > 0).mean()
            avg_ret = trades_df['return'].mean()
        else:
            win_rate = 0
            avg_ret = 0
        
        return BacktestResult(
            total_return=total_ret,
            annual_return=ann_ret,
            sharpe_ratio=sharpe,
            max_drawdown=mdd,
            win_rate=win_rate,
            avg_return=avg_ret,
            n_trades=len(trades),
            equity_curve=equity_df,
            trades=trades_df,
            skipped_limit_up=skipped_limit_up,
            skipped_limit_down=skipped_limit_down
        )
    
    def run_multi_factor_backtest(self,
                                  ts_codes: List[str],
                                  factor_weights: Dict[str, float],
                                  threshold: float = 0.0) -> Optional[BacktestResult]:
        """
        多因子组合回测
        
        Parameters:
        -----------
        ts_codes : List[str]
            股票列表
        factor_weights : Dict[str, float]
            因子权重
        threshold : float
            综合评分阈值
        """
        all_signals = []
        
        for ts_code in ts_codes:
            df = self.load_stock_data(ts_code)
            if df is None or len(df) < 60:
                continue
            
            df = self.calculate_factors(df)
            
            # 计算综合评分
            score = 0
            for factor, weight in factor_weights.items():
                if factor in df.columns:
                    # 标准化因子值
                    factor_std = (df[factor] - df[factor].rolling(60).mean()) / df[factor].rolling(60).std()
                    score += factor_std * weight
            
            df['score'] = score
            df['signal'] = (score > threshold).astype(int)
            df['ts_code'] = ts_code
            
            all_signals.append(df[['trade_date', 'ts_code', 'signal', 'score']])
        
        if not all_signals:
            return None
        
        signals_df = pd.concat(all_signals, ignore_index=True)
        
        # 按日期分组，选择评分最高的股票
        daily_picks = []
        for date, group in signals_df.groupby('trade_date'):
            if len(group) > 0:
                best = group.loc[group['score'].idxmax()]
                daily_picks.append(best)
        
        # 运行回测...
        # 这里简化处理，实际应该根据每日选股结果进行交易
        
        return None


def run_factor_research():
    """
    运行因子研究主程序
    """
    print("=" * 80)
    print("量化因子研究 - 回测框架")
    print("=" * 80)
    
    # 初始化回测器
    backtest = FactorBacktest()
    
    # 测试股票列表（主板）
    test_stocks = ['600000.SH', '000001.SZ', '002001.SZ']
    
    # 测试因子列表
    factors_to_test = [
        'mom_5d', 'mom_10d', 'mom_20d',
        'vol_5d', 'vol_10d', 'vol_20d',
        'rsi_6d', 'rsi_12d', 'rsi_24d',
        'macd', 'macd_hist',
        'bb_position', 'bb_width',
        'kdj_j',
        'williams_r_10d', 'williams_r_20d',
        'atr_ratio',
        'adx'
    ]
    
    results = []
    
    for stock in test_stocks:
        print(f"\n测试股票: {stock}")
        for factor in factors_to_test:
            print(f"  测试因子: {factor}")
            
            # 测试正向信号
            result = backtest.run_backtest(stock, factor, threshold=0.5, direction='long')
            if result:
                results.append({
                    'stock': stock,
                    'factor': factor,
                    'direction': 'long',
                    'total_return': result.total_return,
                    'sharpe': result.sharpe_ratio,
                    'max_dd': result.max_drawdown,
                    'win_rate': result.win_rate,
                    'n_trades': result.n_trades
                })
    
    # 保存结果
    results_df = pd.DataFrame(results)
    results_df.to_csv(r'C:\Users\liuqi\quant_system_v2\research\results\factor_research_results.csv', index=False)
    
    print("\n" + "=" * 80)
    print("因子研究完成！")
    print(f"结果已保存至: research\\results\\factor_research_results.csv")
    print("=" * 80)
    
    return results_df


if __name__ == "__main__":
    run_factor_research()

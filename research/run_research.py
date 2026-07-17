"""
量化研究主程序
整合因子计算、特征优化、回测的完整流程
"""
import pandas as pd
import numpy as np
import os
import sys
from datetime import datetime

# 添加路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from factors.technical_factors import calculate_all_factors
from factors.feature_optimizer import FeatureAnalyzer, FeatureOptimizer, run_feature_analysis
from backtest.factor_backtest import FactorBacktest, BacktestResult


class QuantResearch:
    """量化研究主类"""
    
    def __init__(self, 
                 data_dir: str = r'D:\iquant_data\data_v2',
                 output_dir: str = None):
        """
        Parameters:
        -----------
        data_dir : str
            数据目录
        output_dir : str
            输出目录
        """
        self.data_dir = data_dir
        self.price_dir = os.path.join(data_dir, 'data_day1')
        
        if output_dir is None:
            self.output_dir = os.path.join(os.path.dirname(__file__), 'results')
        else:
            self.output_dir = output_dir
        
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 初始化回测器
        self.backtest = FactorBacktest(data_dir)
    
    def load_market_data(self, 
                        start_date: str = '20230101',
                        end_date: str = '20260331',
                        sample_size: int = 100) -> pd.DataFrame:
        """
        加载市场数据
        
        Parameters:
        -----------
        start_date : str
            开始日期
        end_date : str
            结束日期
        sample_size : int
            抽样股票数量
        """
        print(f"加载市场数据: {start_date} 至 {end_date}")
        
        # 获取所有日期
        dates = sorted([f.replace('.parquet', '') 
                       for f in os.listdir(self.price_dir) 
                       if f.endswith('.parquet')])
        dates = [d for d in dates if start_date <= d <= end_date]
        
        if len(dates) < 60:
            print("数据不足！")
            return pd.DataFrame()
        
        # 加载第一天的数据获取股票列表
        first_day_df = pd.read_parquet(os.path.join(self.price_dir, f"{dates[0]}.parquet"))
        all_stocks = first_day_df['ts_code'].tolist()
        
        # 过滤主板股票
        main_board_stocks = [s for s in all_stocks 
                           if (s.startswith('60') or s.startswith('00') or 
                               s.startswith('002') or s.startswith('003'))]
        
        # 抽样
        if len(main_board_stocks) > sample_size:
            np.random.seed(42)
            selected_stocks = np.random.choice(main_board_stocks, sample_size, replace=False)
        else:
            selected_stocks = main_board_stocks
        
        print(f"选择 {len(selected_stocks)} 只股票进行分析")
        
        # 加载所有数据
        all_data = []
        for date in dates:
            df = pd.read_parquet(os.path.join(self.price_dir, f"{date}.parquet"))
            df['trade_date'] = date
            all_data.append(df)
        
        market_df = pd.concat(all_data, ignore_index=True)
        market_df['trade_date'] = pd.to_datetime(market_df['trade_date'])
        
        # 过滤选中的股票
        market_df = market_df[market_df['ts_code'].isin(selected_stocks)]
        
        print(f"加载完成: {len(market_df)} 条记录")
        return market_df
    
    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        准备特征数据
        """
        print("计算特征...")
        
        # 按股票分组计算因子
        all_features = []
        
        for ts_code, group in df.groupby('ts_code'):
            group = group.sort_values('trade_date')
            
            # 计算所有因子
            group = calculate_all_factors(group)
            
            # 计算前瞻收益 (T+2 close / T+1 open - 1)
            group['t1_open'] = group['open'].shift(-1)
            group['t2_close'] = group['close'].shift(-2)
            group['forward_return'] = group['t2_close'] / group['t1_open'] - 1
            
            all_features.append(group)
        
        result = pd.concat(all_features, ignore_index=True)
        
        print(f"特征计算完成: {len(result)} 条记录")
        return result
    
    def run_factor_research(self, 
                           start_date: str = '20230101',
                           end_date: str = '20260331',
                           sample_size: int = 100) -> Dict:
        """
        运行完整的因子研究流程
        """
        print("=" * 80)
        print("量化因子研究")
        print("=" * 80)
        
        # 1. 加载数据
        print("\n步骤 1: 加载市场数据")
        market_data = self.load_market_data(start_date, end_date, sample_size)
        
        if market_data.empty:
            print("数据加载失败！")
            return {}
        
        # 2. 计算特征
        print("\n步骤 2: 计算特征")
        feature_data = self.prepare_features(market_data)
        
        # 3. 特征分析
        print("\n步骤 3: 特征分析")
        analysis_results = run_feature_analysis(feature_data, 'forward_return')
        
        # 4. 保存结果
        print("\n步骤 4: 保存结果")
        self.save_results(analysis_results)
        
        # 5. 回测最佳因子
        print("\n步骤 5: 回测最佳因子")
        self.backtest_best_factors(feature_data, analysis_results['top_factors'])
        
        print("\n" + "=" * 80)
        print("研究完成！")
        print(f"结果保存在: {self.output_dir}")
        print("=" * 80)
        
        return analysis_results
    
    def save_results(self, results: Dict):
        """
        保存研究结果
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # 保存因子统计
        if 'factor_stats' in results:
            results['factor_stats'].to_csv(
                os.path.join(self.output_dir, f'factor_stats_{timestamp}.csv'),
                index=False
            )
        
        # 保存相关性矩阵
        if 'correlation_matrix' in results:
            results['correlation_matrix'].to_csv(
                os.path.join(self.output_dir, f'correlation_matrix_{timestamp}.csv')
            )
        
        # 保存XGBoost重要性
        if 'xgb_importance' in results:
            results['xgb_importance'].to_csv(
                os.path.join(self.output_dir, f'xgb_importance_{timestamp}.csv'),
                index=False
            )
        
        # 保存最优权重
        if 'optimal_weights' in results:
            weights_df = pd.DataFrame([
                {'factor': k, 'weight': v} 
                for k, v in results['optimal_weights'].items()
            ])
            weights_df.to_csv(
                os.path.join(self.output_dir, f'optimal_weights_{timestamp}.csv'),
                index=False
            )
        
        print(f"结果已保存至: {self.output_dir}")
    
    def backtest_best_factors(self, 
                             feature_data: pd.DataFrame,
                             top_factors: List[str],
                             n_stocks: int = 10):
        """
        回测最佳因子
        """
        print(f"回测 Top {len(top_factors)} 因子...")
        
        # 选择评分最高的股票
        backtest_results = []
        
        for factor in top_factors[:5]:  # 只回测前5个因子
            print(f"\n回测因子: {factor}")
            
            # 这里简化处理，实际应该根据因子值选股并回测
            # 可以使用backtest.factor_backtest中的功能
            
            result = {
                'factor': factor,
                'total_return': 0,
                'sharpe': 0,
                'max_dd': 0
            }
            backtest_results.append(result)
        
        # 保存回测结果
        backtest_df = pd.DataFrame(backtest_results)
        backtest_df.to_csv(
            os.path.join(self.output_dir, 'backtest_results.csv'),
            index=False
        )


def main():
    """
    主函数
    """
    # 初始化研究对象
    research = QuantResearch()
    
    # 运行研究
    results = research.run_factor_research(
        start_date='20230101',
        end_date='20260331',
        sample_size=50  # 测试用，实际可以更大
    )
    
    return results


if __name__ == "__main__":
    main()

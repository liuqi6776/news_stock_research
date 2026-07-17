"""
阈值优化器 - 批量测试不同阈值，无需重新训练模型
"""
import pandas as pd
import numpy as np
import os
import sys
import json
from datetime import datetime

# 添加backtest目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from walk_forward_simple_v2 import WalkForwardBacktestV2


def test_multiple_thresholds(thresholds=[0.5, 0.55, 0.6, 0.65, 0.7]):
    """测试多个阈值并对比结果"""
    
    results_summary = []
    
    for min_prob in thresholds:
        print(f"\n{'='*80}")
        print(f"测试阈值: {min_prob}")
        print(f"{'='*80}")
        
        backtest = WalkForwardBacktestV2()
        results = backtest.run_with_saved_predictions(min_prob=min_prob)
        
        if results:
            # 计算总体指标
            total_trades = sum(r.n_trades for r in results)
            avg_return = np.mean([r.test_return for r in results if r.n_trades > 0]) if total_trades > 0 else 0
            win_rate = np.mean([r.win_rate for r in results if r.n_trades > 0]) if total_trades > 0 else 0
            
            # 读取权益曲线计算总收益
            equity_files = [f for f in os.listdir(backtest.output_dir) if f.startswith('equity_curve_')]
            if equity_files:
                latest_equity = sorted(equity_files)[-1]
                equity_df = pd.read_csv(os.path.join(backtest.output_dir, latest_equity))
                if len(equity_df) > 1:
                    total_return = equity_df['nav'].iloc[-1] / equity_df['nav'].iloc[0] - 1
                    
                    equity_df['ret'] = equity_df['nav'].pct_change()
                    vol = equity_df['ret'].std() * np.sqrt(252)
                    years = len(equity_df) / 252
                    ann_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0
                    sharpe = ann_return / vol if vol > 0 else 0
                    
                    equity_df['cummax'] = equity_df['nav'].cummax()
                    equity_df['dd'] = (equity_df['nav'] - equity_df['cummax']) / equity_df['cummax']
                    max_dd = equity_df['dd'].min()
                else:
                    total_return = 0
                    sharpe = 0
                    max_dd = 0
            else:
                total_return = 0
                sharpe = 0
                max_dd = 0
            
            results_summary.append({
                'threshold': min_prob,
                'total_return': total_return,
                'sharpe': sharpe,
                'max_dd': max_dd,
                'total_trades': total_trades,
                'win_rate': win_rate,
                'avg_monthly_return': avg_return,
                'active_months': sum(1 for r in results if r.n_trades > 0)
            })
            
            print(f"\n阈值 {min_prob} 结果:")
            print(f"  总收益: {total_return*100:.2f}%")
            print(f"  夏普比率: {sharpe:.2f}")
            print(f"  最大回撤: {max_dd*100:.2f}%")
            print(f"  总交易次数: {total_trades}")
            print(f"  胜率: {win_rate*100:.2f}%")
    
    # 保存对比结果
    if results_summary:
        summary_df = pd.DataFrame(results_summary)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        summary_file = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), 
            'results', 
            f'threshold_comparison_{timestamp}.csv'
        )
        summary_df.to_csv(summary_file, index=False)
        print(f"\n{'='*80}")
        print("阈值对比结果已保存:", summary_file)
        print(f"{'='*80}")
        print(summary_df.to_string(index=False))
        
        # 找出最优阈值
        best_sharpe = summary_df.loc[summary_df['sharpe'].idxmax()]
        best_return = summary_df.loc[summary_df['total_return'].idxmax()]
        
        print(f"\n最优夏普比率: 阈值={best_sharpe['threshold']}, 夏普={best_sharpe['sharpe']:.2f}")
        print(f"最优总收益: 阈值={best_return['threshold']}, 收益={best_return['total_return']*100:.2f}%")
    
    return results_summary


if __name__ == '__main__':
    # 测试多个阈值
    thresholds = [0.5, 0.55, 0.6, 0.65, 0.7]
    test_multiple_thresholds(thresholds)

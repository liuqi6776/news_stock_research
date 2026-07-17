import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
NEW_IDEA_DIR = os.path.join(BASE_DIR, 'new_idea')
DOUBAO_DIR = os.path.join(BASE_DIR, 'results_duobao')

print("="*80)
print("  进度报告")
print("="*80)

print(f"\n1. doubao_result 原策略（真实回测）:")
doubao_equity = pd.read_csv(os.path.join(DOUBAO_DIR, 'final_backtest_correct_equity.csv'))
doubao_equity['date'] = pd.to_datetime(doubao_equity['date'])
doubao_equity['nav'] = doubao_equity['nav'] / doubao_equity['nav'].iloc[0]

print(f"   起始日期: {doubao_equity['date'].min()}")
print(f"   结束日期: {doubao_equity['date'].max()}")
print(f"   交易天数: {len(doubao_equity)}")
print(f"   最终净值: {doubao_equity['nav'].iloc[-1]:.2f}x")

def calc_metrics(df):
    nav = df['nav'].values
    returns = np.diff(nav) / nav[:-1]
    total_return = nav[-1] - 1
    n_years = (df['date'].iloc[-1] - df['date'].iloc[0]).days / 365.25
    annual_return = (nav[-1]) ** (1/n_years) - 1
    sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0
    running_max = np.maximum.accumulate(nav)
    drawdown = (running_max - nav) / running_max
    max_drawdown = np.max(drawdown)
    return total_return, annual_return, sharpe, max_drawdown

t1, a1, s1, dd1 = calc_metrics(doubao_equity)

print(f"   总收益: {t1:.2%}")
print(f"   年化: {a1:.2%}")
print(f"   夏普: {s1:.2f}")
print(f"   最大回撤: {dd1:.2%}")

print(f"\n2. 策略1/2/3 回测脚本正在运行...")
print(f"   脚本: {os.path.join(BASE_DIR, 'run_four_strategies.py')}")
print(f"   状态: 正在加载新闻数据，请稍候...")

print(f"\n{'='*80}")
print(f"  正在生成策略1/2/3的交易记录和回测结果...")
print(f"{'='*80}")
print(f"\n预计完成时间: 约10-20分钟")

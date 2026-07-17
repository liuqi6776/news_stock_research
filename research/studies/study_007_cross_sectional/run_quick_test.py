"""
Study 007 快速测试脚本 - 验证因子计算和回测逻辑
"""
import os
import sys
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from factors.cross_sectional_factors import (
    FundamentalFactors, BehavioralFactors, AlternativeFactors,
    FactorCombiner, load_price_range, load_fundamental_data,
    load_industry_map, load_moneyflow_range, load_cyq_range
)
from shared.cs_backtest_engine import CSBacktestEngine


def main():
    print("=" * 60)
    print("Study 007 Quick Test")
    print("=" * 60)
    
    # 只加载 2023-2024 数据（测试期）
    start_date = '20230101'
    end_date = '20241231'
    
    print(f"\n[1/4] 加载数据: {start_date} ~ {end_date}")
    
    # 价格数据
    price_dir = r'D:\iquant_data\data_v2\data_day1'
    all_files = sorted([f.replace('.parquet', '') for f in os.listdir(price_dir) if f.endswith('.parquet')])
    
    try:
        s_idx = all_files.index(start_date)
    except ValueError:
        # 找第一个大于等于 start_date 的
        s_idx = 0
        for i, d in enumerate(all_files):
            if d >= start_date:
                s_idx = i
                break
    try:
        e_idx = all_files.index(end_date)
    except ValueError:
        e_idx = len(all_files) - 1
    
    dates = all_files[s_idx:e_idx+1]
    print(f"  加载 {len(dates)} 个交易日的价格数据")
    
    dfs = []
    for d in dates:
        df = pd.read_parquet(os.path.join(price_dir, f"{d}.parquet"))
        df['trade_date'] = d
        dfs.append(df)
    price_df = pd.concat(dfs, ignore_index=True)
    print(f"  价格数据: {price_df.shape}")
    
    # 基本面数据
    funda_df = load_fundamental_data()
    print(f"  基本面数据: {funda_df.shape}")
    
    # 行业映射
    industry_df = load_industry_map()
    print(f"  行业数据: {industry_df.shape}")
    
    # 资金流数据
    mf_df = load_moneyflow_range(start_date, end_date)
    print(f"  资金流数据: {mf_df.shape if not mf_df.empty else 'empty'}")
    
    # 筹码数据
    cyq_df = load_cyq_range(start_date, end_date)
    print(f"  筹码数据: {cyq_df.shape if not cyq_df.empty else 'empty'}")
    
    print("\n[2/4] 计算因子...")
    
    # 基础特征
    price_df = price_df.sort_values(['ts_code', 'trade_date'])
    price_df['daily_ret'] = price_df.groupby('ts_code')['close'].pct_change()
    price_df['log_mv'] = np.log(price_df['amount'].fillna(0) + 1)
    
    # 计算20日未来收益（目标变量）
    price_df['future_ret_20d'] = price_df.groupby('ts_code')['close'].pct_change(20).shift(-20)
    
    # 行为金融因子
    print("  计算行为金融因子...")
    price_df = BehavioralFactors.get_all_behavioral_factors(price_df, mf_df, cyq_df)
    
    # 基本面因子
    print("  计算基本面因子...")
    if not funda_df.empty:
        funda_factors = FundamentalFactors.get_all_fundamental_factors(funda_df)
        price_df = price_df.merge(funda_factors, on='ts_code', how='left')
    
    # 行业信息
    if not industry_df.empty:
        price_df = price_df.merge(industry_df[['ts_code', 'industry']], on='ts_code', how='left')
    
    # 可用因子列表
    factor_candidates = [
        'ret_1m', 'ret_3m', 'ret_1m_accel',
        'overnight_intraday_spread', 'ivol',
        'big_net_5d', 'smart_dumb_5d',
        'chip_concentration', 'winner_change',
        'roe', 'or_yoy', 'netprofit_yoy',
        'netprofit_margin', 'grossprofit_margin',
        'debt_to_assets', 'quick_ratio',
        'mom_5d', 'mom_10d', 'mom_20d',
    ]
    available_factors = [c for c in factor_candidates if c in price_df.columns]
    print(f"  可用因子: {available_factors}")
    
    print("\n[3/4] 因子预处理...")
    
    # 去极值-中性化-标准化
    price_df = FactorCombiner.preprocess_factors(
        price_df, available_factors,
        industry_col='industry', cap_col='log_mv'
    )
    
    # 线性组合
    price_df['factor_score'] = FactorCombiner.linear_combination(price_df, available_factors)
    
    print(f"  因子得分统计: mean={price_df['factor_score'].mean():.4f}, std={price_df['factor_score'].std():.4f}")
    
    print("\n[4/4] 回测...")
    
    engine = CSBacktestEngine(
        rebalance_freq='monthly',
        top_n=50,
        weight_method='equal',
        cost_rate=0.003,
        industry_neutral=True,
        max_industry_pct=0.30
    )
    
    results = engine.run_backtest(
        price_df, 'factor_score', dates,
        return_col='future_ret_20d'
    )
    
    print("\n" + "=" * 60)
    print("回测结果")
    print("=" * 60)
    print(f"总收益: {results['total_return']*100:.2f}%")
    print(f"年化收益(CAGR): {results['cagr']*100:.2f}%")
    print(f"夏普比率: {results['sharpe']:.3f}")
    print(f"最大回撤: {results['max_drawdown']*100:.2f}%")
    print(f"日胜率: {results['win_rate']*100:.2f}%")
    print(f"平均IC: {results['mean_ic']:.4f}")
    print(f"IR: {results['ir']:.3f}")
    print(f"IC正率: {results['ic_positive_ratio']*100:.2f}%")
    print(f"交易次数: {results['n_trades']}")
    
    if not results['monthly_returns'].empty:
        print(f"\n月度统计:")
        print(f"  月胜率: {(results['monthly_returns'] > 0).mean()*100:.2f}%")
        print(f"  平均月收益: {results['monthly_returns'].mean()*100:.2f}%")
    
    # 保存结果
    output_dir = os.path.join(os.path.dirname(__file__), 'results')
    os.makedirs(output_dir, exist_ok=True)
    
    if not results['nav_df'].empty:
        results['nav_df'].to_csv(os.path.join(output_dir, 'quick_test_nav.csv'), index=False)
    if not results['ic_df'].empty:
        results['ic_df'].to_csv(os.path.join(output_dir, 'quick_test_ic.csv'), index=False)
    
    print(f"\n结果已保存到: {output_dir}")
    print("=" * 60)


if __name__ == '__main__':
    main()

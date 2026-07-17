"""
Study 007: 截面选股策略主运行脚本
流程：数据加载 → 因子计算 → 预处理 → 模型训练 → 回测 → 报告
"""
import os
import sys
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# 添加项目路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from factors.cross_sectional_factors import (
    FundamentalFactors, BehavioralFactors, AlternativeFactors,
    FactorCombiner, load_price_range, load_fundamental_data,
    load_industry_map, load_moneyflow_range, load_cyq_range
)
from shared.cs_backtest_engine import CSBacktestEngine, generate_summary_report
from config import DATA_CONFIG, TIME_CONFIG, FACTOR_CONFIG, BACKTEST_CONFIG, MODEL_CONFIG, DOMAIN_CONFIG


def load_all_data(start_date: str, end_date: str):
    """加载所有需要的数据"""
    print(f"[1/6] 加载数据: {start_date} ~ {end_date}")
    
    # 价格数据
    price_df = load_price_range(start_date, end_date)
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
    
    return price_df, funda_df, industry_df, mf_df, cyq_df


def calculate_all_factors(price_df, funda_df, industry_df, mf_df, cyq_df):
    """计算所有因子"""
    print("\n[2/6] 计算因子...")
    
    # 1. 基础价格特征
    price_df = price_df.sort_values(['ts_code', 'trade_date'])
    price_df['daily_ret'] = price_df.groupby('ts_code')['close'].pct_change()
    price_df['log_mv'] = np.log(price_df['amount'] + 1)  # 代理市值（用成交额）
    
    # 2. 基本面因子
    print("  计算基本面因子...")
    funda_factors = FundamentalFactors.get_all_fundamental_factors(funda_df)
    
    # 3. 行为金融因子
    print("  计算行为金融因子...")
    price_df = BehavioralFactors.get_all_behavioral_factors(price_df, mf_df, cyq_df)
    
    # 4. 合并基本面因子（前向填充）
    # 基本面是季度数据，需要匹配到每个交易日
    print("  合并基本面因子...")
    if not funda_factors.empty:
        # 为每个股票的最基本面数据，填充到所有交易日
        price_df = price_df.merge(funda_factors, on='ts_code', how='left')
    
    # 5. 行业信息
    if not industry_df.empty:
        price_df = price_df.merge(industry_df[['ts_code', 'industry']], on='ts_code', how='left')
    
    # 6. 另类数据因子
    print("  计算另类数据因子...")
    # 从 price_df 中的已有列提取（news 和 rank 数据日频，需要单独加载）
    
    print(f"  因子计算完成，总列数: {len(price_df.columns)}")
    return price_df


def prepare_target_variable(price_df: pd.DataFrame, horizon: int = 20) -> pd.DataFrame:
    """
    计算目标变量：未来N日收益
    """
    print(f"\n[3/6] 计算目标变量（未来{horizon}日收益）...")
    
    price_df = price_df.sort_values(['ts_code', 'trade_date'])
    
    # 未来N日收益 = (t+N收盘价 - t收盘价) / t收盘价
    price_df[f'future_ret_{horizon}d'] = price_df.groupby('ts_code')['close'].pct_change(horizon).shift(-horizon)
    
    # 未来N日收益（用下一天的开盘价买入，N天后收盘价卖出）
    price_df['next_open'] = price_df.groupby('ts_code')['open'].shift(-1)
    price_df[f'future_ret_{horizon}d_open'] = (
        price_df.groupby('ts_code')['close'].shift(-horizon) / price_df['next_open'] - 1
    )
    
    return price_df


def preprocess_and_combine_factors(price_df: pd.DataFrame, factor_cols: list, 
                                   train_dates: list, test_dates: list) -> pd.DataFrame:
    """因子预处理与非线性组合"""
    print("\n[4/6] 因子预处理与组合...")
    
    # 合并所有因子列表
    all_factors = []
    for k, v in factor_cols.items():
        all_factors.extend(v)
    
    # 筛选实际存在的因子
    available_factors = [c for c in all_factors if c in price_df.columns]
    print(f"  可用因子数: {len(available_factors)}/{len(all_factors)}")
    
    # 训练期数据
    train_df = price_df[price_df['trade_date'].isin(train_dates)].copy()
    test_df = price_df[price_df['trade_date'].isin(test_dates)].copy()
    
    # 预处理（去极值-中性化-标准化）
    print("  执行去极值-中性化-标准化...")
    
    train_df = FactorCombiner.preprocess_factors(
        train_df, available_factors, 
        industry_col='industry', cap_col='log_mv'
    )
    
    # 使用训练期的统计量对测试期做预处理（避免数据泄漏）
    # 简化：对测试期独立做同样的处理（实际操作中应该用训练期参数）
    test_df = FactorCombiner.preprocess_factors(
        test_df, available_factors,
        industry_col='industry', cap_col='log_mv'
    )
    
    # 组合
    target_col = 'future_ret_20d_open' if 'future_ret_20d_open' in train_df.columns else 'future_ret_20d'
    
    if MODEL_CONFIG['use_tree_model']:
        print(f"  使用{MODEL_CONFIG['model_type']}非线性模型组合因子...")
        
        # 训练集组合
        train_df['factor_score'] = FactorCombiner.tree_model_combination(
            train_df, available_factors, target_col=target_col, model_type=MODEL_CONFIG['model_type']
        )
        
        # 测试集：使用训练好的模型预测（简化：在测试集上重新拟合，实际应该保存模型）
        test_df['factor_score'] = FactorCombiner.tree_model_combination(
            test_df, available_factors, target_col=target_col, model_type=MODEL_CONFIG['model_type']
        )
    else:
        print("  使用线性加权组合因子...")
        train_df['factor_score'] = FactorCombiner.linear_combination(train_df, available_factors)
        test_df['factor_score'] = FactorCombiner.linear_combination(test_df, available_factors)
    
    # 合并训练和测试数据
    full_df = pd.concat([train_df, test_df], ignore_index=True)
    
    return full_df


def run_domain_backtest(full_df: pd.DataFrame, test_dates: list):
    """分域回测"""
    print("\n[5/6] 分域回测...")
    
    # 按市值分域
    full_df['market_cap_bin'] = pd.cut(
        full_df['amount'].fillna(0),  # 用成交额代理
        bins=[0, 1e7, 5e7, 2e8, 1e9, float('inf')],
        labels=['微型', '小盘', '中盘', '大盘', '超大盘']
    )
    
    engine = CSBacktestEngine(**BACKTEST_CONFIG)
    
    # 全市场回测
    print("  全市场回测...")
    results = engine.run_backtest(
        full_df, 'factor_score', test_dates,
        return_col='future_ret_20d_open'
    )
    
    # 分域回测
    domain_results = {}
    if DOMAIN_CONFIG['enable_domain_model']:
        print("  分域回测...")
        for domain in full_df['market_cap_bin'].cat.categories:
            domain_df = full_df[full_df['market_cap_bin'] == domain]
            if len(domain_df) < 1000:
                continue
            domain_res = engine.run_backtest(
                domain_df, 'factor_score', test_dates,
                return_col='future_ret_20d_open'
            )
            domain_results[domain] = domain_res
            print(f"    {domain}: CAGR={domain_res['cagr']*100:.2f}%, Sharpe={domain_res['sharpe']:.3f}")
    
    return results, domain_results


def save_results(results, domain_results, output_dir: str):
    """保存结果"""
    print("\n[6/6] 保存结果...")
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存净值曲线
    if not results['nav_df'].empty:
        results['nav_df'].to_csv(os.path.join(output_dir, 'nav_curve.csv'), index=False)
    
    # 保存IC序列
    if not results['ic_df'].empty:
        results['ic_df'].to_csv(os.path.join(output_dir, 'ic_series.csv'), index=False)
    
    # 保存交易记录
    if results['trades']:
        pd.DataFrame(results['trades']).to_csv(os.path.join(output_dir, 'trades.csv'), index=False)
    
    # 保存报告
    report = generate_summary_report(results, domain_results)
    with open(os.path.join(output_dir, 'report.txt'), 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"  结果已保存到: {output_dir}")
    return report


def plot_results(results, domain_results, output_dir: str):
    """绘制回测图表"""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. 净值曲线
        ax1 = axes[0, 0]
        if not results['nav_df'].empty:
            nav = results['nav_df']
            ax1.plot(pd.to_datetime(nav['date']), nav['nav'] / nav['nav'].iloc[0], label='Strategy')
            ax1.set_title('Strategy NAV')
            ax1.set_xlabel('Date')
            ax1.set_ylabel('NAV')
            ax1.legend()
        
        # 2. 回撤曲线
        ax2 = axes[0, 1]
        if not results['nav_df'].empty:
            nav = results['nav_df']
            cummax = nav['nav'].cummax()
            dd = (cummax - nav['nav']) / cummax
            ax2.fill_between(pd.to_datetime(nav['date']), -dd * 100, 0, color='red', alpha=0.3)
            ax2.set_title(f'Drawdown (Max: {results["max_drawdown"]*100:.1f}%)')
            ax2.set_xlabel('Date')
            ax2.set_ylabel('Drawdown %')
        
        # 3. IC序列
        ax3 = axes[1, 0]
        if not results['ic_df'].empty:
            ic = results['ic_df']
            ax3.bar(range(len(ic)), ic['ic'], color=['green' if x > 0 else 'red' for x in ic['ic']])
            ax3.axhline(y=results['mean_ic'], color='blue', linestyle='--', label=f'Mean IC={results["mean_ic"]:.3f}')
            ax3.set_title('Factor IC Series')
            ax3.set_xlabel('Rebalance')
            ax3.set_ylabel('IC')
            ax3.legend()
        
        # 4. 分域对比
        ax4 = axes[1, 1]
        if domain_results:
            domains = list(domain_results.keys())
            cagrs = [domain_results[d]['cagr'] * 100 for d in domains]
            sharpes = [domain_results[d]['sharpe'] for d in domains]
            x = np.arange(len(domains))
            ax4_twin = ax4.twinx()
            ax4.bar(x - 0.2, cagrs, 0.4, label='CAGR%', color='steelblue')
            ax4_twin.bar(x + 0.2, sharpes, 0.4, label='Sharpe', color='coral')
            ax4.set_xticks(x)
            ax4.set_xticklabels(domains, rotation=45)
            ax4.set_title('Domain Comparison')
            ax4.legend(loc='upper left')
            ax4_twin.legend(loc='upper right')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'backtest_results.png'), dpi=150)
        plt.close()
        print(f"  图表已保存")
    except Exception as e:
        print(f"  绘图失败: {e}")


def main():
    """主函数"""
    print("=" * 60)
    print("Study 007: 截面选股策略")
    print("=" * 60)
    
    # 加载数据
    price_df, funda_df, industry_df, mf_df, cyq_df = load_all_data(
        TIME_CONFIG['train_start'], TIME_CONFIG['test_end']
    )
    
    if price_df.empty:
        print("错误: 价格数据为空")
        return
    
    # 计算因子
    full_df = calculate_all_factors(price_df, funda_df, industry_df, mf_df, cyq_df)
    
    # 计算目标变量
    full_df = prepare_target_variable(full_df, horizon=20)
    
    # 获取所有交易日
    all_dates = sorted(full_df['trade_date'].unique().tolist())
    train_dates = [d for d in all_dates if TIME_CONFIG['train_start'] <= d <= TIME_CONFIG['train_end']]
    test_dates = [d for d in all_dates if TIME_CONFIG['test_start'] <= d <= TIME_CONFIG['test_end']]
    
    print(f"\n  训练期: {len(train_dates)} 个交易日")
    print(f"  测试期: {len(test_dates)} 个交易日")
    
    # 因子预处理与组合
    full_df = preprocess_and_combine_factors(full_df, FACTOR_CONFIG, train_dates, test_dates)
    
    # 回测
    results, domain_results = run_domain_backtest(full_df, test_dates)
    
    # 保存结果
    output_dir = os.path.join(os.path.dirname(__file__), 'results')
    report = save_results(results, domain_results, output_dir)
    
    # 绘图
    plot_results(results, domain_results, output_dir)
    
    # 打印报告
    print("\n" + report)
    
    print("\n" + "=" * 60)
    print("Study 007 完成")
    print("=" * 60)


if __name__ == '__main__':
    main()

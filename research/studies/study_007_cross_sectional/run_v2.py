"""
Study 007 改进版测试 - 修正因子方向 + XGBoost非线性组合
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


# 因子方向定义：True=正向（因子越高越好），False=反向（因子越低越好）
FACTOR_DIRECTIONS = {
    # 反转因子（反向）
    'ret_1m': False,
    'ret_3m': False,
    'ret_1m_accel': False,
    'overnight_ret_5d': False,  # 低隔夜收益 + 高日内反转
    'intraday_ret_5d': True,    # 高日内收益
    'overnight_intraday_spread': True,
    'hot_inv': True,  # 热度反转（已取负）
    
    # 低波动溢价（反向）
    'ivol': False,  # 低特质波动率溢价
    'vol_5d': False,
    'vol_10d': False,
    'vol_20d': False,
    
    # 资金流因子（正向）
    'big_net_inflow_ratio': True,
    'smart_dumb_spread': True,
    'big_net_5d': True,
    'smart_dumb_5d': True,
    'net_mf_amount': True,
    
    # 筹码因子
    'chip_concentration': False,  # 筹码越集中越好（值越小越好）
    'winner_change': True,        # 获利盘增加（但可能反向？）
    'winner_rate': False,         # 低获利盘 = 低位吸筹
    
    # 基本面因子（正向）
    'roe': True,
    'roe_dt': True,
    'roe_stability': True,
    'or_yoy': True,
    'or_yoy_accel': True,
    'netprofit_yoy': True,
    'netprofit_yoy_accel': True,
    'netprofit_margin': True,
    'grossprofit_margin': True,
    'gpm_trend': True,
    'npm_trend': True,
    'debt_improve': True,   # 已取负
    'quick_change': True,
    'quick_ratio': True,
    'current_ratio': True,
    
    # 情绪因子
    'sentiment_net': True,
    'sentiment_ratio': True,
    'new_gs': True,
    
    # 动量因子（正向，但A股短期动量可能反向）
    'mom_5d': False,   # 短期动量通常反向
    'mom_10d': False,
    'mom_20d': True,   # 中期动量可能正向
    'mom_60d': True,
}


def apply_factor_directions(df, factor_cols, directions):
    """根据因子方向调整符号"""
    df = df.copy()
    for col in factor_cols:
        if col in directions and not directions[col]:
            # 反向因子取负
            df[col] = -df[col]
    return df


def main():
    print("=" * 60)
    print("Study 007: 截面选股策略 (改进版)")
    print("=" * 60)
    
    start_date = '20230101'
    end_date = '20241231'
    
    print(f"\n[1/5] 加载数据: {start_date} ~ {end_date}")
    
    # 价格数据
    price_dir = r'D:\iquant_data\data_v2\data_day1'
    all_files = sorted([f.replace('.parquet', '') for f in os.listdir(price_dir) if f.endswith('.parquet')])
    
    try:
        s_idx = all_files.index(start_date)
    except ValueError:
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
    print(f"  加载 {len(dates)} 个交易日")
    
    dfs = []
    for d in dates:
        df = pd.read_parquet(os.path.join(price_dir, f"{d}.parquet"))
        df['trade_date'] = d
        dfs.append(df)
    price_df = pd.concat(dfs, ignore_index=True)
    print(f"  价格: {price_df.shape}")
    
    # 其他数据
    funda_df = load_fundamental_data()
    industry_df = load_industry_map()
    mf_df = load_moneyflow_range(start_date, end_date)
    cyq_df = load_cyq_range(start_date, end_date)
    
    print(f"  基本面: {funda_df.shape}, 行业: {industry_df.shape}, 资金流: {mf_df.shape}, 筹码: {cyq_df.shape}")
    
    print("\n[2/5] 计算因子...")
    
    # 基础特征
    price_df = price_df.sort_values(['ts_code', 'trade_date'])
    price_df['daily_ret'] = price_df.groupby('ts_code')['close'].pct_change()
    price_df['log_mv'] = np.log(price_df['amount'].fillna(0) + 1)
    
    # 计算20日未来收益
    price_df['future_ret_20d'] = price_df.groupby('ts_code')['close'].pct_change(20).shift(-20)
    
    # 行为金融因子
    price_df = BehavioralFactors.get_all_behavioral_factors(price_df, mf_df, cyq_df)
    
    # 基本面因子
    if not funda_df.empty:
        funda_factors = FundamentalFactors.get_all_fundamental_factors(funda_df)
        price_df = price_df.merge(funda_factors, on='ts_code', how='left')
    
    # 行业信息
    if not industry_df.empty:
        price_df = price_df.merge(industry_df[['ts_code', 'industry']], on='ts_code', how='left')
    
    # 因子列表
    factor_candidates = list(FACTOR_DIRECTIONS.keys())
    available_factors = [c for c in factor_candidates if c in price_df.columns]
    print(f"  可用因子: {len(available_factors)}/{len(factor_candidates)}")
    print(f"  {available_factors}")
    
    print("\n[3/5] 因子方向修正与预处理...")
    
    # 修正因子方向
    price_df = apply_factor_directions(price_df, available_factors, FACTOR_DIRECTIONS)
    
    # 预处理（去极值-中性化-标准化）
    price_df = FactorCombiner.preprocess_factors(
        price_df, available_factors,
        industry_col='industry', cap_col='log_mv'
    )
    
    # 线性组合（作为基准）
    price_df['factor_score_linear'] = FactorCombiner.linear_combination(price_df, available_factors)
    
    # 非线性组合（XGBoost）
    print("\n[4/5] XGBoost非线性组合...")
    
    # 构建训练数据（使用测试期内的时间序列滚动训练）
    # Walk-forward：用前60天训练，预测后20天
    price_df = price_df.sort_values('trade_date')
    all_trade_dates = sorted(price_df['trade_date'].unique())
    
    price_df['factor_score_xgb'] = np.nan
    
    train_window = 60  # 训练窗口（交易日）
    predict_horizon = 20  # 预测 horizon
    
    std_cols = [f'{c}_std' for c in available_factors if f'{c}_std' in price_df.columns]
    target_col = 'future_ret_20d'
    
    # 每20个交易日重新训练一次
    for i in range(train_window, len(all_trade_dates), predict_horizon):
        train_dates = all_trade_dates[max(0, i-train_window):i]
        predict_dates = all_trade_dates[i:min(i+predict_horizon, len(all_trade_dates))]
        
        train_df = price_df[price_df['trade_date'].isin(train_dates)]
        predict_df = price_df[price_df['trade_date'].isin(predict_dates)]
        
        if train_df.empty or predict_df.empty:
            continue
        
        # 训练模型
        valid_train = train_df[std_cols + [target_col]].dropna()
        if len(valid_train) < 100:
            continue
        
        X_train = valid_train[std_cols]
        y_train = valid_train[target_col]
        
        try:
            import xgboost as xgb
            model = xgb.XGBRegressor(
                n_estimators=100, max_depth=4, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8,
                random_state=42, verbosity=0
            )
        except ImportError:
            from sklearn.ensemble import GradientBoostingRegressor
            model = GradientBoostingRegressor(
                n_estimators=100, max_depth=4, random_state=42
            )
        
        model.fit(X_train, y_train)
        
        # 预测
        X_pred = predict_df[std_cols].fillna(0)
        predictions = model.predict(X_pred)
        
        # 赋值
        price_df.loc[predict_df.index, 'factor_score_xgb'] = predictions
    
    # 对于无法预测的数据，用线性组合填充
    price_df['factor_score_xgb'] = price_df['factor_score_xgb'].fillna(price_df['factor_score_linear'])
    
    print(f"  XGBoost覆盖: {price_df['factor_score_xgb'].notna().sum()}/{len(price_df)}")
    
    print("\n[5/5] 回测对比...")
    
    # 回测：线性 vs XGBoost
    engine = CSBacktestEngine(
        rebalance_freq='monthly',
        top_n=50,
        weight_method='equal',
        cost_rate=0.003,
        industry_neutral=True,
        max_industry_pct=0.30
    )
    
    results_linear = engine.run_backtest(
        price_df, 'factor_score_linear', dates,
        return_col='future_ret_20d'
    )
    
    results_xgb = engine.run_backtest(
        price_df, 'factor_score_xgb', dates,
        return_col='future_ret_20d'
    )
    
    # 打印结果
    print("\n" + "=" * 60)
    print("回测结果对比")
    print("=" * 60)
    print(f"{'指标':<20} {'线性组合':<15} {'XGBoost':<15}")
    print("-" * 60)
    print(f"{'总收益':<20} {results_linear['total_return']*100:>13.2f}% {results_xgb['total_return']*100:>13.2f}%")
    print(f"{'年化收益(CAGR)':<20} {results_linear['cagr']*100:>13.2f}% {results_xgb['cagr']*100:>13.2f}%")
    print(f"{'夏普比率':<20} {results_linear['sharpe']:>15.3f} {results_xgb['sharpe']:>15.3f}")
    print(f"{'最大回撤':<20} {results_linear['max_drawdown']*100:>13.2f}% {results_xgb['max_drawdown']*100:>13.2f}%")
    print(f"{'日胜率':<20} {results_linear['win_rate']*100:>13.2f}% {results_xgb['win_rate']*100:>13.2f}%")
    print(f"{'平均IC':<20} {results_linear['mean_ic']:>15.4f} {results_xgb['mean_ic']:>15.4f}")
    print(f"{'IR':<20} {results_linear['ir']:>15.3f} {results_xgb['ir']:>15.3f}")
    print(f"{'IC正率':<20} {results_linear['ic_positive_ratio']*100:>13.2f}% {results_xgb['ic_positive_ratio']*100:>13.2f}%")
    print(f"{'交易次数':<20} {results_linear['n_trades']:>15} {results_xgb['n_trades']:>15}")
    
    if not results_linear['monthly_returns'].empty:
        print(f"\n{'月胜率':<20} {results_linear['monthly_returns'].mean()*100:>13.2f}% {results_xgb['monthly_returns'].mean()*100:>13.2f}%")
    
    # 保存结果
    output_dir = os.path.join(os.path.dirname(__file__), 'results')
    os.makedirs(output_dir, exist_ok=True)
    
    if not results_linear['nav_df'].empty:
        results_linear['nav_df'].to_csv(os.path.join(output_dir, 'nav_linear.csv'), index=False)
    if not results_xgb['nav_df'].empty:
        results_xgb['nav_df'].to_csv(os.path.join(output_dir, 'nav_xgb.csv'), index=False)
    if not results_xgb['ic_df'].empty:
        results_xgb['ic_df'].to_csv(os.path.join(output_dir, 'ic_xgb.csv'), index=False)
    
    # 绘图
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. 净值对比
        ax1 = axes[0, 0]
        if not results_linear['nav_df'].empty:
            nav_l = results_linear['nav_df']
            ax1.plot(pd.to_datetime(nav_l['date']), nav_l['nav'] / nav_l['nav'].iloc[0], label='Linear', alpha=0.7)
        if not results_xgb['nav_df'].empty:
            nav_x = results_xgb['nav_df']
            ax1.plot(pd.to_datetime(nav_x['date']), nav_x['nav'] / nav_x['nav'].iloc[0], label='XGBoost', linewidth=2)
        ax1.axhline(y=1, color='gray', linestyle='--', alpha=0.5)
        ax1.set_title('NAV Comparison: Linear vs XGBoost')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('NAV')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. XGBoost回撤
        ax2 = axes[0, 1]
        if not results_xgb['nav_df'].empty:
            nav = results_xgb['nav_df']
            cummax = nav['nav'].cummax()
            dd = (cummax - nav['nav']) / cummax
            ax2.fill_between(pd.to_datetime(nav['date']), -dd * 100, 0, color='red', alpha=0.3)
            ax2.set_title(f'XGBoost Drawdown (Max: {results_xgb["max_drawdown"]*100:.1f}%)')
            ax2.set_xlabel('Date')
            ax2.set_ylabel('Drawdown %')
        
        # 3. IC序列
        ax3 = axes[1, 0]
        if not results_xgb['ic_df'].empty:
            ic = results_xgb['ic_df']
            colors = ['green' if x > 0 else 'red' for x in ic['ic']]
            ax3.bar(range(len(ic)), ic['ic'], color=colors, alpha=0.6)
            ax3.axhline(y=results_xgb['mean_ic'], color='blue', linestyle='--', 
                       label=f'Mean IC={results_xgb["mean_ic"]:.3f}')
            ax3.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            ax3.set_title('IC Series (XGBoost)')
            ax3.set_xlabel('Rebalance')
            ax3.set_ylabel('IC')
            ax3.legend()
        
        # 4. 月度收益分布
        ax4 = axes[1, 1]
        if not results_xgb['monthly_returns'].empty:
            mrets = results_xgb['monthly_returns'].dropna() * 100
            ax4.hist(mrets, bins=15, color='steelblue', edgecolor='white', alpha=0.7)
            ax4.axvline(x=mrets.mean(), color='red', linestyle='--', label=f'Mean={mrets.mean():.2f}%')
            ax4.set_title('Monthly Return Distribution (XGBoost)')
            ax4.set_xlabel('Monthly Return %')
            ax4.set_ylabel('Frequency')
            ax4.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'backtest_comparison.png'), dpi=150)
        plt.close()
        print(f"\n图表已保存到: {output_dir}")
    except Exception as e:
        print(f"绘图失败: {e}")
    
    print("\n" + "=" * 60)
    print("Study 007 完成")
    print("=" * 60)


if __name__ == '__main__':
    main()

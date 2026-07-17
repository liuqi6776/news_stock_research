"""
Study 007 最终版 - 基于因子有效性检验的优化策略
只使用有效因子，按IR加权组合，月度调仓
"""
import os
import sys
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from shared.cs_backtest_engine import CSBacktestEngine


# 基于因子检验结果，只保留有效因子及其方向
# 从 test_factor_validity.py 结果:
# ivol: IC=0.0939, IR=0.480 (最强)
# ret_1m: IC=0.0650, IR=0.418 (强)
# roe: IC=0.0345, IR=0.171 (弱正)
# or_yoy: IC=0.0092
# netprofit_yoy: IC=0.0055
# overnight_intraday: IC=-0.0556 (方向待验证，暂不纳入)
# ret_1m_accel: IC=-0.0351 (方向反了，暂不纳入)

FACTORS = {
    'ivol': {'weight': 0.40, 'direction': -1},      # 低特质波动率最强
    'ret_1m': {'weight': 0.35, 'direction': -1},    # 反转
    'roe': {'weight': 0.15, 'direction': 1},         # 基本面质量
    'or_yoy': {'weight': 0.05, 'direction': 1},      # 营收增速
    'netprofit_yoy': {'weight': 0.05, 'direction': 1},  # 利润增速
}


def load_all(start_date, end_date):
    """加载所有数据"""
    price_dir = r'D:\iquant_data\data_v2\data_day1'
    all_files = sorted([f.replace('.parquet', '') for f in os.listdir(price_dir) if f.endswith('.parquet')])
    s_idx = next((i for i, d in enumerate(all_files) if d >= start_date), 0)
    e_idx = next((i for i in range(len(all_files)-1, -1, -1) if all_files[i] <= end_date), len(all_files)-1)
    dates = all_files[s_idx:e_idx+1]
    
    dfs = []
    for d in dates:
        df = pd.read_parquet(os.path.join(price_dir, f"{d}.parquet"))
        df['trade_date'] = d
        dfs.append(df)
    price_df = pd.concat(dfs, ignore_index=True)
    
    # 基本面
    funda_p = r'D:\iquant_data\data_v2\fundamental1\fina_indicator_cache.parquet'
    funda_df = pd.read_parquet(funda_p).sort_values(['ts_code', 'ann_date'], ascending=[True, False]).groupby('ts_code').first().reset_index() if os.path.exists(funda_p) else pd.DataFrame()
    
    # 行业
    ind_p = r'D:\iquant_data\data_v2\industry1\industry.parquet'
    ind_df = pd.read_parquet(ind_p)[['ts_code', 'industry']] if os.path.exists(ind_p) else pd.DataFrame()
    
    return price_df, dates, funda_df, ind_df


def calc_factors(price_df, funda_df, ind_df):
    """计算核心因子"""
    price_df = price_df.sort_values(['ts_code', 'trade_date'])
    price_df['daily_ret'] = price_df.groupby('ts_code')['close'].pct_change()
    price_df['future_ret_20d'] = price_df.groupby('ts_code')['close'].pct_change(20).shift(-20)
    
    # 1. 反转因子
    price_df['ret_1m'] = -price_df.groupby('ts_code')['close'].pct_change(20)  # 取负：反转
    
    # 2. 特质波动率（低波动溢价）
    price_df['vol_20d'] = price_df.groupby('ts_code')['daily_ret'].transform(lambda x: x.rolling(20, min_periods=10).std())
    market_ret = price_df.groupby('trade_date')['daily_ret'].mean().rename('market_ret')
    price_df = price_df.merge(market_ret, on='trade_date', how='left')
    # beta
    beta = price_df.groupby('ts_code')['daily_ret'].transform(lambda x: x.rolling(20, min_periods=10).cov(price_df.loc[x.index, 'market_ret'])) / (price_df.groupby('ts_code')['market_ret'].transform(lambda x: x.rolling(20, min_periods=10).var()) + 1e-12)
    price_df['beta'] = beta
    price_df['ivol'] = -price_df['vol_20d'] * np.sqrt(1 - price_df['beta'].clip(-1, 1)**2 + 1e-12)  # 取负：低波动好
    
    # 3. 基本面
    if not funda_df.empty:
        price_df = price_df.merge(
            funda_df[['ts_code', 'roe', 'or_yoy', 'netprofit_yoy']],
            on='ts_code', how='left'
        )
    
    # 4. 行业
    if not ind_df.empty:
        price_df = price_df.merge(ind_df, on='ts_code', how='left')
    
    return price_df


def preprocess(df):
    """预处理：去极值 + 行业中性化 + 标准化"""
    factor_names = list(FACTORS.keys())
    df = df.copy()
    
    for col in factor_names:
        if col not in df.columns:
            continue
        # 去极值
        q01, q99 = df[col].quantile([0.01, 0.99])
        df[col] = df[col].clip(q01, q99)
        
        # 行业中性化：在每个截面、每个行业内标准化
        if 'industry' in df.columns:
            df[col] = df.groupby(['trade_date', 'industry'])[col].transform(lambda x: (x - x.mean()) / (x.std() + 1e-8) if len(x) > 1 else 0)
        else:
            df[col] = df.groupby('trade_date')[col].transform(lambda x: (x - x.mean()) / (x.std() + 1e-8))
    
    return df


def combine_factors(df):
    """按预设权重组合因子"""
    score = pd.Series(0.0, index=df.index)
    total_weight = 0
    
    for col, cfg in FACTORS.items():
        if col in df.columns:
            w = cfg['weight']
            score += df[col].fillna(0) * w * cfg['direction']
            total_weight += abs(w)
    
    if total_weight > 0:
        score = score / total_weight
    
    return score


def main():
    print("=" * 60)
    print("Study 007: 截面选股策略 (最终优化版)")
    print("=" * 60)
    
    # 加载数据
    print("\n[1/3] 加载数据...")
    price_df, dates, funda_df, ind_df = load_all('20230101', '20241231')
    print(f"  价格: {price_df.shape}, 基本面: {funda_df.shape}, 行业: {ind_df.shape}")
    
    # 计算因子
    print("\n[2/3] 计算因子...")
    price_df = calc_factors(price_df, funda_df, ind_df)
    price_df = preprocess(price_df)
    price_df['factor_score'] = combine_factors(price_df)
    
    # 统计
    print(f"  因子得分: mean={price_df['factor_score'].mean():.4f}, std={price_df['factor_score'].std():.4f}")
    print(f"  IC={price_df['factor_score'].corr(price_df['future_ret_20d'], method='spearman'):.4f}")
    
    # 回测
    print("\n[3/3] 回测...")
    engine = CSBacktestEngine(
        rebalance_freq='monthly',
        top_n=50,
        weight_method='equal',
        cost_rate=0.003,
        industry_neutral=True,
        max_industry_pct=0.30
    )
    
    results = engine.run_backtest(price_df, 'factor_score', dates, return_col='future_ret_20d')
    
    # 打印结果
    print("\n" + "=" * 60)
    print("回测结果")
    print("=" * 60)
    print(f"总收益:      {results['total_return']*100:>10.2f}%")
    print(f"年化收益:    {results['cagr']*100:>10.2f}%")
    print(f"夏普比率:    {results['sharpe']:>10.3f}")
    print(f"最大回撤:    {results['max_drawdown']*100:>10.2f}%")
    print(f"日胜率:      {results['win_rate']*100:>10.2f}%")
    print(f"平均IC:      {results['mean_ic']:>10.4f}")
    print(f"IR:          {results['ir']:>10.3f}")
    print(f"IC正率:      {results['ic_positive_ratio']*100:>10.2f}%")
    print(f"交易次数:    {results['n_trades']:>10}")
    
    if not results['monthly_returns'].empty:
        print(f"\n月度统计:")
        print(f"  月胜率:    {(results['monthly_returns'] > 0).mean()*100:.1f}%")
        print(f"  平均月收益: {results['monthly_returns'].mean()*100:.2f}%")
    
    # 保存
    out = os.path.join(os.path.dirname(__file__), 'results')
    os.makedirs(out, exist_ok=True)
    if not results['nav_df'].empty:
        results['nav_df'].to_csv(os.path.join(out, 'nav_final.csv'), index=False)
    if not results['ic_df'].empty:
        results['ic_df'].to_csv(os.path.join(out, 'ic_final.csv'), index=False)
    
    # 绘图
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. 净值
        ax = axes[0, 0]
        nav = results['nav_df']
        ax.plot(pd.to_datetime(nav['date']), nav['nav']/nav['nav'].iloc[0], color='steelblue', linewidth=1.5)
        ax.axhline(y=1, color='gray', linestyle='--', alpha=0.5)
        ax.set_title(f'Strategy NAV (CAGR: {results["cagr"]*100:.1f}%)')
        ax.set_xlabel('Date')
        ax.set_ylabel('NAV')
        ax.grid(True, alpha=0.3)
        
        # 2. 回撤
        ax = axes[0, 1]
        cummax = nav['nav'].cummax()
        dd = (cummax - nav['nav']) / cummax
        ax.fill_between(pd.to_datetime(nav['date']), -dd*100, 0, color='red', alpha=0.3)
        ax.set_title(f'Drawdown (Max: {results["max_drawdown"]*100:.1f}%)')
        ax.set_xlabel('Date')
        ax.set_ylabel('Drawdown %')
        ax.grid(True, alpha=0.3)
        
        # 3. IC
        ax = axes[1, 0]
        ic = results['ic_df']
        colors = ['green' if x > 0 else 'red' for x in ic['ic']]
        ax.bar(range(len(ic)), ic['ic'], color=colors, alpha=0.6)
        ax.axhline(y=results['mean_ic'], color='blue', linestyle='--', label=f'Mean IC={results["mean_ic"]:.3f}')
        ax.axhline(y=0, color='black', alpha=0.3)
        ax.set_title('IC Series')
        ax.set_xlabel('Rebalance')
        ax.set_ylabel('IC')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 4. 月度收益
        ax = axes[1, 1]
        mrets = results['monthly_returns'].dropna() * 100
        ax.bar(range(len(mrets)), mrets, color=['green' if x > 0 else 'red' for x in mrets], alpha=0.6)
        ax.axhline(y=mrets.mean(), color='blue', linestyle='--', label=f'Mean={mrets.mean():.2f}%')
        ax.set_title('Monthly Returns')
        ax.set_xlabel('Month')
        ax.set_ylabel('Return %')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(out, 'backtest_final.png'), dpi=150)
        plt.close()
        print(f"\n图表已保存: {os.path.join(out, 'backtest_final.png')}")
    except Exception as e:
        print(f"绘图失败: {e}")
    
    print("\n" + "=" * 60)


if __name__ == '__main__':
    main()

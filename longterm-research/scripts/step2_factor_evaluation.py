"""
step2_factor_evaluation.py
综合评估 15 个 Alpha 因子（包括技术、动量、流动性、资金流、新闻舆情与概念热度）。
支持：
1. 修正 ths_hot_rank：排除 9999 占位符，仅计算实际处于概念热度榜的个股。
2. 双重时间跨度评估：
   - 全历史时段（Full Period，~1496个交易日）
   - THS 有效时段（THS Period，仅在THS数据有效的 ~362 个交易日内，方便各因子在相同时间段进行公平对比）。
3. 关键因子分位数（Decile）组合回测。
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(PROJECT_DIR, 'data')
RESULTS_DIR = os.path.join(PROJECT_DIR, 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

FEATURES_FILE = os.path.join(DATA_DIR, 'features_longterm.parquet')

# 候选因子库
FACTORS = {
    # 1. 技术指标 (Technical)
    'macd': 'Technical',
    'rsi_14': 'Technical',
    'kdj_k': 'Technical',
    'bb_position': 'Technical',
    'atr_14': 'Technical',
    'volatility_20': 'Technical',
    # 2. 价格动量 (Momentum)
    'mom_5d': 'Momentum',
    'mom_10d': 'Momentum',
    'mom_20d': 'Momentum',
    'mom_60d': 'Momentum',
    # 3. 流动性与量价 (Volume/Liquidity)
    'turnover_rate': 'Liquidity',
    'vol_ratio': 'Volume',
    'volume_ratio': 'Volume',
    # 4. 资金流向 (Money Flow)
    'net_mf_amount_norm': 'MoneyFlow',
    'net_mf_vol': 'MoneyFlow',
    'buy_lg_amount': 'MoneyFlow',
    'sell_lg_amount': 'MoneyFlow',
    'buy_elg_amount': 'MoneyFlow',
    'sell_elg_amount': 'MoneyFlow',
    # 5. 新闻舆情 (News)
    'news_stock_impact': 'News',
    'news_has_mention': 'News',
    'news_market_impact': 'News',
    'new_gs': 'News',
    'new_bs': 'News',
    'new_gi': 'News',
    # 6. 同花顺概念热度 (Concept Sentiment)
    'ths_hot': 'Concept',
    'ths_hot_rank': 'Concept',
    # 7. 期权/波动率 (Options/VIX)
    'opt_qvix_close': 'Options',
    'opt_qvix_change': 'Options',
    'opt_qvix_zscore': 'Options',
    'opt_pcr_vol_50': 'Options',
    'opt_pcr_oi_50': 'Options',
    'opt_pcr_vol_300': 'Options',
    'opt_pcr_oi_300': 'Options',
    # 8. Alpha 101
    'alpha_006': 'Alpha101',
    'alpha_009': 'Alpha101',
    'alpha_012': 'Alpha101',
    'alpha_023': 'Alpha101'
}

def calculate_ic_metrics(df, target_col, restrict_to_ths_dates=False, ths_dates=None):
    # Auto-detect Vibe-Trading Alpha factors from the dataframe
    vibe_cols = [c for c in df.columns if (c.startswith('alpha101_') or c.startswith('gtja191_') or c.startswith('gtja_') or c.startswith('alpha_')) and c not in FACTORS]
    for c in vibe_cols:
        if 'alpha' in c.lower():
            FACTORS[c] = 'Alpha101_Vibe'
        else:
            FACTORS[c] = 'GTJA191_Vibe'

    period_name = "THS Valid Period" if restrict_to_ths_dates else "Full Period"
    print(f"\n--- Calculating Rank IC metrics against {target_col} ({period_name}) ---")
    
    valid_df = df[df[target_col].notna()].copy()
    if restrict_to_ths_dates and ths_dates is not None:
        valid_df = valid_df[valid_df['trade_date'].isin(ths_dates)].copy()
        
    ic_results = []
    
    for factor in FACTORS.keys():
        if factor not in valid_df.columns:
            continue
            
        # 提取因子与收益率数据
        factor_df = valid_df[['trade_date', factor, target_col]].copy()
        
        # 针对 ths_hot_rank 剔除 9999 占位符
        if factor == 'ths_hot_rank':
            factor_df = factor_df[factor_df[factor] < 9999]
            
        # 剔除空值
        factor_df = factor_df.dropna()
        if len(factor_df) == 0:
            continue
            
        # 计算每日 Rank IC (Spearman 秩相关)
        daily_ic = factor_df.groupby('trade_date').apply(
            lambda x: x[factor].corr(x[target_col], method='spearman')
        )
        
        daily_ic = daily_ic.dropna()
        if len(daily_ic) == 0:
            continue
            
        ic_mean = daily_ic.mean()
        ic_std = daily_ic.std()
        ic_ir = ic_mean / ic_std if ic_std > 0 else 0
        ic_pos_ratio = (daily_ic > 0).mean()
        
        ic_results.append({
            'factor': factor,
            'category': FACTORS[factor],
            'ic_mean': ic_mean,
            'ic_std': ic_std,
            'ic_ir': ic_ir,
            'ic_pos_ratio': ic_pos_ratio,
            'valid_days': len(daily_ic)
        })
        
    ic_df = pd.DataFrame(ic_results)
    if not ic_df.empty:
        ic_df = ic_df.sort_values(by='ic_ir', key=abs, ascending=False)
        print(ic_df.to_string(index=False))
        
        suffix = "ths_period" if restrict_to_ths_dates else "full_period"
        out_csv = os.path.join(RESULTS_DIR, f'factor_ic_summary_{target_col}_{suffix}.csv')
        ic_df.to_csv(out_csv, index=False)
        print(f"Saved IC summary to {out_csv}")
    return ic_df

def run_decile_backtest(df, factor, target_ret_col, holding_days=10, ascending=True, ths_dates=None, period_suffix='full_period'):
    print(f"\n--- Running Decile Backtest for Factor: {factor} (Holding: {holding_days}d, Period: {period_suffix}) ---")
    
    # 对 ths_hot_rank 进行特殊过滤（仅看处于热度榜的股票）
    test_df = df.copy()
    if factor == 'ths_hot_rank':
        test_df = test_df[test_df[factor] < 9999].copy()
        # 调仓日期仅在 THS 有效交易日
        trade_dates = sorted(test_df['trade_date'].unique())
    elif ths_dates is not None:
        # 同样限制在 THS 有效期
        test_df = test_df[test_df['trade_date'].isin(ths_dates)].copy()
        trade_dates = sorted(test_df['trade_date'].unique())
    else:
        trade_dates = sorted(test_df['trade_date'].unique())
        
    # 调仓日期设定
    rebalance_dates = trade_dates[::holding_days]
    print(f"Factor: {factor} | Total trade dates: {len(trade_dates)}, Rebalance dates: {len(rebalance_dates)}")
    
    decile_returns = {f"Decile_{i+1}": [] for i in range(10)}
    decile_returns['Long_Short'] = []
    rebalance_records = []
    
    for dt in rebalance_dates:
        day_df = test_df[test_df['trade_date'] == dt].copy()
        day_df = day_df[day_df[factor].notna() & day_df[target_ret_col].notna()]
        
        # 对于非 ths_hot_rank 的普通因子，截面股票数应充足
        min_stocks = 20 if factor == 'ths_hot_rank' else 100
        if len(day_df) < min_stocks:
            continue
            
        # 按照因子排序分组
        # ascending=True 表示从小到大排列（例如 rank、turnover_rate 越小收益越好，Decile_1是因子最小组）
        # 如果是正向因子，如 news_stock_impact / net_mf_amount_norm，我们希望把大的排在后面，Decile_10是最优组
        day_df['rank'] = day_df[factor].rank(method='first', ascending=ascending)
        day_df['decile'] = pd.qcut(day_df['rank'], 10, labels=[f"Decile_{i+1}" for i in range(10)])
        
        group_rets = day_df.groupby('decile')[target_ret_col].mean()
        
        for decile_name in decile_returns.keys():
            if decile_name == 'Long_Short':
                # 统一为多头组 (Decile 10) 减去空头组 (Decile 1)
                ls_ret = group_rets['Decile_10'] - group_rets['Decile_1']
                decile_returns['Long_Short'].append(ls_ret)
            else:
                decile_returns[decile_name].append(group_rets[decile_name])
                
        rebalance_records.append(dt)
        
    nav_df = pd.DataFrame(index=rebalance_records)
    for col, rets in decile_returns.items():
        if len(rets) == len(rebalance_records):
            nav_df[col] = np.cumprod(1 + np.array(rets))
            
    # 计算基准净值 (在该子集上的平均表现)
    all_rets = []
    for dt in rebalance_records:
        day_df = test_df[(test_df['trade_date'] == dt) & test_df[target_ret_col].notna()]
        all_rets.append(day_df[target_ret_col].mean() if len(day_df) > 0 else 0)
    nav_df['Benchmark'] = np.cumprod(1 + np.array(all_rets))
    
    # 计算绩效指标
    metrics = []
    years_held = (len(trade_dates) / 252.0)
    
    for col in nav_df.columns:
        final_nav = nav_df[col].iloc[-1]
        cagr = (final_nav) ** (1.0 / years_held) - 1 if final_nav > 0 else -1
        
        # Max Drawdown
        cum_max = nav_df[col].cummax()
        drawdowns = (nav_df[col] - cum_max) / cum_max
        max_dd = drawdowns.min()
        
        # Sharpe Ratio
        rets = nav_df[col].pct_change().dropna()
        ann_std = rets.std() * np.sqrt(252 / holding_days)
        ann_mean = rets.mean() * (252 / holding_days)
        sharpe = ann_mean / ann_std if ann_std > 0 else 0
        
        metrics.append({
            'Portfolio': col,
            'Final_NAV': final_nav,
            'CAGR': cagr,
            'Sharpe': sharpe,
            'Max_DD': max_dd
        })
        
    metrics_df = pd.DataFrame(metrics)
    print(metrics_df.to_string(index=False))
    
    metrics_df.to_csv(os.path.join(RESULTS_DIR, f'decile_metrics_{factor}_{period_suffix}.csv'), index=False)
    nav_df.to_csv(os.path.join(RESULTS_DIR, f'decile_navs_{factor}_{period_suffix}.csv'))
    
    # 绘图
    plot_navs(nav_df, factor, period_suffix)
 
def plot_navs(nav_df, factor, period_suffix):
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    fig, ax = plt.subplots(figsize=(12, 7))
    
    colors = plt.cm.RdYlGn(np.linspace(0.1, 0.9, 10))
    for i in range(10):
        name = f"Decile_{i+1}"
        ax.plot(nav_df.index, nav_df[name], label=name, color=colors[i], linewidth=1.5, alpha=0.8)
        
    ax.plot(nav_df.index, nav_df['Benchmark'], label='Benchmark (Equal-Weight)', color='blue', linewidth=2, linestyle='--')
    ax.plot(nav_df.index, nav_df['Long_Short'], label='Long-Short (D10 - D1)', color='black', linewidth=2.5)
    
    ax.set_title(f'Decile Portfolio NAV - Factor: {factor} ({period_suffix})', fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel('Rebalance Date', fontsize=12)
    ax.set_ylabel('Net Asset Value (NAV)', fontsize=12)
    
    tick_spacing = max(1, len(nav_df) // 10)
    plt.xticks(nav_df.index[::tick_spacing], rotation=45)
    
    ax.legend(loc='upper left', frameon=True, facecolor='white', framealpha=0.9, edgecolor='lightgray')
    plt.tight_layout()
    
    out_png = os.path.join(RESULTS_DIR, f'decile_navs_{factor}_{period_suffix}.png')
    plt.savefig(out_png, dpi=300)
    print(f"Saved plot to {out_png}")
    plt.close()

def run_evaluation():
    print("Loading enriched features...")
    df = pd.read_parquet(FEATURES_FILE)
    print(f"Loaded {len(df)} rows.")
    
    # 确定 THS 有效的日期序列
    # 即：具有有效的 ths_hot_rank 且值小于 9999 的日期
    ths_valid_df = df[df['ths_hot_rank'] < 9999]
    ths_dates = sorted(ths_valid_df['trade_date'].unique())
    print(f"THS valid dates count: {len(ths_dates)}")
    
    # 1. 评估全历史跨度的 Rank IC
    calculate_ic_metrics(df, 'mkt_excess_ret_20d', restrict_to_ths_dates=False)
    
    # 2. 评估仅在 THS 有效期的 Rank IC (进行相同时间跨度的公平横向对比)
    calculate_ic_metrics(df, 'mkt_excess_ret_20d', restrict_to_ths_dates=True, ths_dates=ths_dates)
    
    # 3. 运行多因子组合分位数回测 (Full Period)
    print("\n====== Running Decile Backtests on FULL Period ======")
    run_decile_backtest(df, 'turnover_rate', 'ret_20d', holding_days=20, ascending=True, period_suffix='full_period')
    run_decile_backtest(df, 'news_stock_impact', 'ret_20d', holding_days=20, ascending=True, period_suffix='full_period')
    run_decile_backtest(df, 'net_mf_amount_norm', 'ret_20d', holding_days=20, ascending=True, period_suffix='full_period')
    run_decile_backtest(df, 'alpha_009', 'ret_20d', holding_days=20, ascending=True, period_suffix='full_period')
    
    # 4. 运行多因子组合分位数回测 (THS Period)
    print("\n====== Running Decile Backtests on THS Period ======")
    run_decile_backtest(df, 'ths_hot_rank', 'ret_20d', holding_days=20, ascending=True, ths_dates=ths_dates, period_suffix='ths_period')
    run_decile_backtest(df, 'turnover_rate', 'ret_20d', holding_days=20, ascending=True, ths_dates=ths_dates, period_suffix='ths_period')
    run_decile_backtest(df, 'news_stock_impact', 'ret_20d', holding_days=20, ascending=True, ths_dates=ths_dates, period_suffix='ths_period')
    run_decile_backtest(df, 'net_mf_amount_norm', 'ret_20d', holding_days=20, ascending=True, ths_dates=ths_dates, period_suffix='ths_period')
    run_decile_backtest(df, 'alpha_009', 'ret_20d', holding_days=20, ascending=True, ths_dates=ths_dates, period_suffix='ths_period')

if __name__ == '__main__':
    run_evaluation()

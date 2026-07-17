"""
Study 007 v3: 分域建模 + 2020-2025 回测（高效版）
核心优化：
1. 预计算固定权重（用2020-2024数据），2025作为测试期
2. 向量化得分计算，避免apply按行遍历
3. 只保留调仓日数据用于回测
"""
import os
import sys
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))


def load_data(start_date, end_date):
    """加载价格数据（只读需要的列）"""
    price_dir = r'D:\iquant_data\data_v2\data_day1'
    all_files = sorted([f.replace('.parquet', '') for f in os.listdir(price_dir) if f.endswith('.parquet')])
    s_idx = next((i for i, d in enumerate(all_files) if d >= start_date), 0)
    e_idx = next((i for i in range(len(all_files)-1, -1, -1) if all_files[i] <= end_date), len(all_files)-1)
    dates = all_files[s_idx:e_idx+1]
    
    print(f"  加载 {len(dates)} 个交易日 ({dates[0]} ~ {dates[-1]})")
    
    cols = ['ts_code', 'open', 'high', 'low', 'close', 'pre_close', 'vol', 'amount']
    dfs = []
    for d in dates:
        df = pd.read_parquet(os.path.join(price_dir, f"{d}.parquet"), columns=cols)
        df['trade_date'] = d
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True), dates


def calc_factors(price_df):
    """计算核心因子（向量化）"""
    price_df = price_df.sort_values(['ts_code', 'trade_date'])
    
    # 1. 日收益 & 未来20日收益
    price_df['daily_ret'] = price_df.groupby('ts_code')['close'].pct_change()
    price_df['future_ret_20d'] = price_df.groupby('ts_code')['close'].pct_change(20).shift(-20)
    
    # 2. 1月反转（取负）
    price_df['ret_1m'] = -price_df.groupby('ts_code')['close'].pct_change(20)
    
    # 3. 低波动（总波动率代理）
    price_df['ivol'] = -price_df.groupby('ts_code')['daily_ret'].transform(lambda x: x.rolling(20, min_periods=10).std())
    
    # 4. 隔夜/日内分离
    price_df['overnight'] = (price_df['open'] - price_df['pre_close']) / price_df['pre_close']
    price_df['intraday'] = (price_df['close'] - price_df['open']) / price_df['open']
    price_df['ov_5d'] = price_df.groupby('ts_code')['overnight'].transform(lambda x: x.rolling(5, min_periods=1).mean())
    price_df['id_5d'] = price_df.groupby('ts_code')['intraday'].transform(lambda x: x.rolling(5, min_periods=1).mean())
    price_df['ov_id'] = price_df['id_5d'] - price_df['ov_5d']
    
    # 5. 20日动量（中期）
    price_df['mom_20d'] = price_df.groupby('ts_code')['close'].pct_change(20)
    
    return price_df


def add_fundamental(price_df):
    """加载基本面数据"""
    p = r'D:\iquant_data\data_v2\fundamental1\fina_indicator_cache.parquet'
    if not os.path.exists(p):
        return price_df
    
    funda = pd.read_parquet(p).sort_values(['ts_code', 'ann_date'], ascending=[True, False]).groupby('ts_code').first().reset_index()
    funda = funda[['ts_code', 'roe', 'or_yoy', 'netprofit_yoy']]
    return price_df.merge(funda, on='ts_code', how='left')


def add_industry(price_df):
    """加载行业数据"""
    p = r'D:\iquant_data\data_v2\industry1\industry.parquet'
    if not os.path.exists(p):
        return price_df
    return price_df.merge(pd.read_parquet(p)[['ts_code', 'industry']], on='ts_code', how='left')


def assign_domain(price_df):
    """按成交额分域"""
    price_df['domain'] = pd.cut(
        price_df['amount'].fillna(0),
        bins=[0, 5e7, 5e8, float('inf')],
        labels=['小盘', '中盘', '大盘']
    )
    return price_df


def standardize(df, cols):
    """截面标准化 + 去极值"""
    for c in cols:
        if c not in df.columns:
            continue
        q01, q99 = df[c].quantile([0.01, 0.99])
        df[c] = df[c].clip(q01, q99)
        df[c] = df.groupby('trade_date')[c].transform(lambda x: (x - x.mean()) / (x.std() + 1e-8))
    return df


def calc_ic(df, factor, target='future_ret_20d'):
    """计算截面IC"""
    v = df[[factor, target]].dropna()
    if len(v) < 10:
        return 0
    return v[factor].corr(v[target], method='spearman')


def calc_weights(df, train_dates, factors):
    """计算各域因子权重（IC加权）"""
    train = df[df['trade_date'].isin(train_dates)]
    weights = {}
    
    for domain in ['小盘', '中盘', '大盘']:
        ddf = train[train['domain'] == domain]
        if len(ddf) < 100:
            weights[domain] = {f: 0 for f in factors}
            continue
        
        w = {}
        for f in factors:
            if f not in ddf.columns:
                w[f] = 0
                continue
            ic = calc_ic(ddf, f)
            w[f] = max(0, ic)  # 只取正IC的因子
        
        total = sum(w.values())
        if total > 0:
            w = {k: v/total for k, v in w.items()}
        weights[domain] = w
    
    return weights


def backtest(df, dates, weights, top_n_per_domain=20, cost=0.003):
    """回测：向量化计算得分"""
    # 提取所有因子
    all_factors = list(set().union(*[set(w.keys()) for w in weights.values()]))
    
    # 预计算每月的得分：为每个域单独计算
    df['score'] = 0.0
    for domain in ['小盘', '中盘', '大盘']:
        mask = df['domain'] == domain
        w = weights.get(domain, {})
        if not w:
            continue
        for f, weight in w.items():
            if f in df.columns:
                df.loc[mask, 'score'] += df.loc[mask, f].fillna(0) * weight
    
    # 生成月度调仓日
    reb_dates = []
    cur_month = None
    for d in dates:
        m = d[:6]
        if m != cur_month:
            reb_dates.append(d)
            cur_month = m
    
    # 只保留调仓日数据
    reb_df = df[df['trade_date'].isin(reb_dates)].copy()
    
    # 每月选股：按域 + 得分排序
    picks = []
    for date, group in reb_df.groupby('trade_date'):
        for domain in ['小盘', '中盘', '大盘']:
            g = group[group['domain'] == domain].dropna(subset=['score'])
            if len(g) < 3:
                continue
            g = g.sort_values('score', ascending=False)
            n = min(top_n_per_domain, max(2, len(g) // 8))
            picks.append(g.head(n))
    
    if not picks:
        return pd.DataFrame(), pd.DataFrame()
    
    selected = pd.concat(picks, ignore_index=True)
    
    # 合并未来收益（已经在future_ret_20d中）
    # 计算组合收益
    monthly = selected.groupby('trade_date').agg({
        'future_ret_20d': 'mean',
        'score': 'mean',
    }).reset_index()
    monthly = monthly.rename(columns={'future_ret_20d': 'ret'})
    monthly = monthly.dropna(subset=['ret'])
    
    if monthly.empty:
        return pd.DataFrame(), pd.DataFrame()
    
    monthly = monthly.sort_values('trade_date')
    monthly['ret_net'] = monthly['ret'] - cost
    monthly['nav'] = (1 + monthly['ret_net']).cumprod()
    
    # 计算IC
    ic_list = []
    for date, group in reb_df.groupby('trade_date'):
        g = group.dropna(subset=['score', 'future_ret_20d'])
        if len(g) < 10:
            continue
        ic = g['score'].corr(g['future_ret_20d'], method='spearman')
        ic_list.append({'date': date, 'ic': ic})
    
    ic_df = pd.DataFrame(ic_list)
    return monthly, ic_df


def backtest_global(df, dates, factors, top_n=50, cost=0.003):
    """全市场回测（对比基准）"""
    # 计算全局权重
    train = df[df['trade_date'] < '20250101']  # 2020-2024训练
    w = {}
    for f in factors:
        if f not in train.columns:
            w[f] = 0
            continue
        ic = calc_ic(train, f)
        w[f] = max(0, ic)
    total = sum(w.values())
    if total > 0:
        w = {k: v/total for k, v in w.items()}
    
    # 计算得分
    df['score'] = 0
    for f, weight in w.items():
        if f in df.columns:
            df['score'] += df[f].fillna(0) * weight
    
    reb_dates = []
    cur_month = None
    for d in dates:
        m = d[:6]
        if m != cur_month:
            reb_dates.append(d)
            cur_month = m
    
    reb_df = df[df['trade_date'].isin(reb_dates)].copy()
    
    picks = []
    for date, group in reb_df.groupby('trade_date'):
        g = group.dropna(subset=['score'])
        if len(g) < 5:
            continue
        g = g.sort_values('score', ascending=False)
        picks.append(g.head(top_n))
    
    if not picks:
        return pd.DataFrame(), pd.DataFrame()
    
    selected = pd.concat(picks, ignore_index=True)
    monthly = selected.groupby('trade_date').agg({'future_ret_20d': 'mean'}).reset_index()
    monthly = monthly.rename(columns={'future_ret_20d': 'ret'})
    monthly = monthly.dropna(subset=['ret'])
    
    if monthly.empty:
        return pd.DataFrame(), pd.DataFrame()
    
    monthly = monthly.sort_values('trade_date')
    monthly['ret_net'] = monthly['ret'] - cost
    monthly['nav'] = (1 + monthly['ret_net']).cumprod()
    
    ic_list = []
    for date, group in reb_df.groupby('trade_date'):
        g = group.dropna(subset=['score', 'future_ret_20d'])
        if len(g) < 10:
            continue
        ic = g['score'].corr(g['future_ret_20d'], method='spearman')
        ic_list.append({'date': date, 'ic': ic})
    
    return monthly, pd.DataFrame(ic_list)


def print_metrics(monthly, ic_df, label):
    """打印指标"""
    if monthly.empty:
        print(f"  {label}: 无数据")
        return
    
    total_ret = monthly['nav'].iloc[-1] - 1
    n_months = len(monthly)
    n_years = n_months / 12
    cagr = (1 + total_ret) ** (1 / max(n_years, 1e-8)) - 1
    monthly_rets = monthly['ret_net']
    sharpe = monthly_rets.mean() / (monthly_rets.std() + 1e-8) * np.sqrt(12)
    cummax = monthly['nav'].cummax()
    max_dd = ((cummax - monthly['nav']) / cummax).max()
    win_rate = (monthly_rets > 0).mean()
    
    mean_ic = ic_df['ic'].mean() if not ic_df.empty else 0
    ir = mean_ic / (ic_df['ic'].std() + 1e-8) if not ic_df.empty and ic_df['ic'].std() > 0 else 0
    
    print(f"\n  {label}:")
    print(f"    总收益: {total_ret*100:.2f}%  |  年化: {cagr*100:.2f}%  |  夏普: {sharpe:.3f}")
    print(f"    最大回撤: {max_dd*100:.2f}%  |  月胜率: {win_rate*100:.1f}%  |  月数: {n_months}")
    print(f"    平均IC: {mean_ic:.4f}  |  IR: {ir:.3f}  |  IC>0: {(ic_df['ic']>0).mean()*100:.1f}%")


def main():
    print("=" * 70)
    print("Study 007 v3: 分域建模 + 2020-2025 回测")
    print("=" * 70)
    
    print("\n[1/3] 加载数据...")
    price_df, dates = load_data('20200101', '20251231')
    
    print("\n[2/3] 计算因子...")
    price_df = calc_factors(price_df)
    price_df = add_fundamental(price_df)
    price_df = add_industry(price_df)
    price_df = assign_domain(price_df)
    
    factors = ['ret_1m', 'ivol', 'ov_id', 'mom_20d', 'roe', 'or_yoy', 'netprofit_yoy']
    available = [f for f in factors if f in price_df.columns]
    print(f"  可用因子: {available}")
    
    price_df = standardize(price_df, available)
    
    # 分域训练权重（2020-2024训练，2025测试）
    print("\n[3/3] 回测...")
    train_dates = [d for d in dates if d < '20250101']
    test_dates = [d for d in dates if d >= '20250101']
    
    # 分域权重
    domain_weights = calc_weights(price_df, train_dates, available)
    print("\n  分域因子权重 (2020-2024训练):")
    for domain, w in domain_weights.items():
        print(f"    【{domain}】", {k: round(v, 3) for k, v in w.items() if v > 0.01})
    
    # 分域回测（2020-2025）
    monthly_d, ic_d = backtest(price_df, dates, domain_weights, top_n_per_domain=20)
    
    # 全市场回测（2020-2025）
    monthly_g, ic_g = backtest_global(price_df, dates, available, top_n=50)
    
    # 打印结果
    print("\n" + "=" * 70)
    print("回测结果 (2020-2025)")
    print("=" * 70)
    print_metrics(monthly_g, ic_g, "全市场 (Global)")
    print_metrics(monthly_d, ic_d, "分域建模 (Domain)")
    
    # 保存
    out = os.path.join(os.path.dirname(__file__), 'results')
    os.makedirs(out, exist_ok=True)
    if not monthly_d.empty:
        monthly_d.to_csv(os.path.join(out, 'monthly_domain_2020_2025.csv'), index=False)
    if not monthly_g.empty:
        monthly_g.to_csv(os.path.join(out, 'monthly_global_2020_2025.csv'), index=False)
    if not ic_d.empty:
        ic_d.to_csv(os.path.join(out, 'ic_domain_2020_2025.csv'), index=False)
    if not ic_g.empty:
        ic_g.to_csv(os.path.join(out, 'ic_global_2020_2025.csv'), index=False)
    
    # 绘图
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. NAV
        ax = axes[0, 0]
        if not monthly_g.empty:
            ax.plot(pd.to_datetime(monthly_g['trade_date']), monthly_g['nav'], label='Global', alpha=0.7)
        if not monthly_d.empty:
            ax.plot(pd.to_datetime(monthly_d['trade_date']), monthly_d['nav'], label='Domain', linewidth=2)
        ax.axhline(y=1, color='gray', linestyle='--', alpha=0.5)
        ax.set_title('NAV: Global vs Domain (2020-2025)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 2. Drawdown
        ax = axes[0, 1]
        if not monthly_d.empty:
            cummax = monthly_d['nav'].cummax()
            dd = (cummax - monthly_d['nav']) / cummax
            ax.fill_between(pd.to_datetime(monthly_d['trade_date']), -dd*100, 0, color='red', alpha=0.3)
            ax.set_title(f'Domain Drawdown (Max: {dd.max()*100:.1f}%)')
            ax.grid(True, alpha=0.3)
        
        # 3. IC
        ax = axes[1, 0]
        if not ic_d.empty:
            colors = ['green' if x > 0 else 'red' for x in ic_d['ic']]
            ax.bar(range(len(ic_d)), ic_d['ic'], color=colors, alpha=0.6)
            mean_ic = ic_d['ic'].mean()
            ax.axhline(y=mean_ic, color='blue', linestyle='--', label=f'Mean IC={mean_ic:.3f}')
            ax.axhline(y=0, color='black', alpha=0.3)
            ax.set_title('IC Series (Domain)')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # 4. Monthly returns
        ax = axes[1, 1]
        if not monthly_d.empty:
            mrets = monthly_d['ret_net'] * 100
            ax.bar(range(len(mrets)), mrets, color=['green' if x > 0 else 'red' for x in mrets], alpha=0.6)
            ax.axhline(y=mrets.mean(), color='blue', linestyle='--', label=f'Mean={mrets.mean():.2f}%')
            ax.set_title('Monthly Returns')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(out, 'backtest_v3_2020_2025.png'), dpi=150)
        plt.close()
        print(f"\n  图表已保存")
    except Exception as e:
        print(f"  绘图失败: {e}")
    
    print("\n" + "=" * 70)
    print("Study 007 v3 完成")
    print("=" * 70)


if __name__ == '__main__':
    main()

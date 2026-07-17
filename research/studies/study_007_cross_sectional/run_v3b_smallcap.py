"""
Study 007 v3b: 小盘股专用策略 + 2020-2025
发现：反转和低波动因子只在小盘股有效，中大盘IC为0
"""
import os
import sys
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

def load_data(start_date, end_date):
    price_dir = r'D:\iquant_data\data_v2\data_day1'
    all_files = sorted([f.replace('.parquet', '') for f in os.listdir(price_dir) if f.endswith('.parquet')])
    s_idx = next((i for i, d in enumerate(all_files) if d >= start_date), 0)
    e_idx = next((i for i in range(len(all_files)-1, -1, -1) if all_files[i] <= end_date), len(all_files)-1)
    dates = all_files[s_idx:e_idx+1]
    
    cols = ['ts_code', 'open', 'high', 'low', 'close', 'pre_close', 'vol', 'amount']
    dfs = []
    for d in dates:
        df = pd.read_parquet(os.path.join(price_dir, f"{d}.parquet"), columns=cols)
        df['trade_date'] = d
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True), dates

def calc_factors(price_df):
    price_df = price_df.sort_values(['ts_code', 'trade_date'])
    price_df['daily_ret'] = price_df.groupby('ts_code')['close'].pct_change()
    price_df['future_ret_20d'] = price_df.groupby('ts_code')['close'].pct_change(20).shift(-20)
    
    # 反转
    price_df['ret_1m'] = -price_df.groupby('ts_code')['close'].pct_change(20)
    
    # 低波动
    price_df['ivol'] = -price_df.groupby('ts_code')['daily_ret'].transform(lambda x: x.rolling(20, min_periods=10).std())
    
    # 隔夜/日内
    price_df['overnight'] = (price_df['open'] - price_df['pre_close']) / price_df['pre_close']
    price_df['intraday'] = (price_df['close'] - price_df['open']) / price_df['open']
    price_df['ov_5d'] = price_df.groupby('ts_code')['overnight'].transform(lambda x: x.rolling(5, min_periods=1).mean())
    price_df['id_5d'] = price_df.groupby('ts_code')['intraday'].transform(lambda x: x.rolling(5, min_periods=1).mean())
    price_df['ov_id'] = price_df['id_5d'] - price_df['ov_5d']
    
    # 中期动量
    price_df['mom_20d'] = price_df.groupby('ts_code')['close'].pct_change(20)
    
    return price_df

def add_fundamental(price_df):
    p = r'D:\iquant_data\data_v2\fundamental1\fina_indicator_cache.parquet'
    if not os.path.exists(p):
        return price_df
    funda = pd.read_parquet(p).sort_values(['ts_code', 'ann_date'], ascending=[True, False]).groupby('ts_code').first().reset_index()
    return price_df.merge(funda[['ts_code', 'roe', 'or_yoy', 'netprofit_yoy']], on='ts_code', how='left')

def standardize(df, cols):
    for c in cols:
        if c not in df.columns:
            continue
        q01, q99 = df[c].quantile([0.01, 0.99])
        df[c] = df[c].clip(q01, q99)
        df[c] = df.groupby('trade_date')[c].transform(lambda x: (x - x.mean()) / (x.std() + 1e-8))
    return df

def calc_ic(df, factor, target='future_ret_20d'):
    v = df[[factor, target]].dropna()
    if len(v) < 10:
        return 0
    return v[factor].corr(v[target], method='spearman')

def backtest(df, dates, factors, top_n=50, cost=0.003, filter_amount_max=None):
    """回测，可选只选小盘股"""
    # 计算得分（等权）
    df['score'] = 0
    for f in factors:
        if f in df.columns:
            df['score'] += df[f].fillna(0)
    df['score'] = df['score'] / len(factors)
    
    # 生成月度调仓日
    reb_dates = []
    cur_month = None
    for d in dates:
        m = d[:6]
        if m != cur_month:
            reb_dates.append(d)
            cur_month = m
    
    reb_df = df[df['trade_date'].isin(reb_dates)].copy()
    
    # 可选：只选小盘股
    if filter_amount_max:
        reb_df = reb_df[reb_df['amount'] < filter_amount_max]
    
    picks = []
    ic_list = []
    for date, group in reb_df.groupby('trade_date'):
        g = group.dropna(subset=['score', 'future_ret_20d'])
        if len(g) < 5:
            continue
        
        # IC
        ic = g['score'].corr(g['future_ret_20d'], method='spearman')
        ic_list.append({'date': date, 'ic': ic})
        
        # 选股
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
    
    return monthly, pd.DataFrame(ic_list)

def print_metrics(monthly, ic_df, label):
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
    print("Study 007 v3b: 小盘股专用策略")
    print("=" * 70)
    
    print("\n[1/3] 加载数据...")
    price_df, dates = load_data('20200101', '20251231')
    
    print("\n[2/3] 计算因子...")
    price_df = calc_factors(price_df)
    price_df = add_fundamental(price_df)
    
    factors = ['ret_1m', 'ivol', 'ov_id', 'mom_20d', 'roe', 'or_yoy', 'netprofit_yoy']
    available = [f for f in factors if f in price_df.columns]
    print(f"  可用因子: {available}")
    
    price_df = standardize(price_df, available)
    
    print("\n[3/3] 回测...")
    
    # 全市场
    monthly_all, ic_all = backtest(price_df, dates, available, top_n=50)
    
    # 小盘股（amount < 5000万）
    monthly_small, ic_small = backtest(price_df, dates, available, top_n=30, filter_amount_max=5e7)
    
    # 中盘股（5000万 - 5亿）
    monthly_mid, ic_mid = backtest(price_df, dates, available, top_n=30, filter_amount_max=5e8)
    # 注意：mid 回测需要先筛选，但filter_amount_max只过滤了上限，没有下限
    # 修正：手动筛选中盘股
    price_df_mid = price_df[(price_df['amount'] >= 5e7) & (price_df['amount'] < 5e8)].copy()
    monthly_mid, ic_mid = backtest(price_df_mid, dates, available, top_n=30)
    
    # 大盘股（> 5亿）
    price_df_large = price_df[price_df['amount'] >= 5e8].copy()
    monthly_large, ic_large = backtest(price_df_large, dates, available, top_n=30)
    
    print("\n" + "=" * 70)
    print("分域回测结果 (2020-2025)")
    print("=" * 70)
    print_metrics(monthly_all, ic_all, "全市场 (All)")
    print_metrics(monthly_small, ic_small, "小盘股 (<5000万)")
    print_metrics(monthly_mid, ic_mid, "中盘股 (5000万-5亿)")
    print_metrics(monthly_large, ic_large, "大盘股 (>5亿)")
    
    # 保存
    out = os.path.join(os.path.dirname(__file__), 'results')
    os.makedirs(out, exist_ok=True)
    for name, df in [('all', monthly_all), ('small', monthly_small), ('mid', monthly_mid), ('large', monthly_large)]:
        if not df.empty:
            df.to_csv(os.path.join(out, f'monthly_{name}_2020_2025.csv'), index=False)
    
    # 绘图
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. NAV对比
        ax = axes[0, 0]
        for name, df, color in [('All', monthly_all, 'steelblue'), ('Small', monthly_small, 'coral'), 
                                 ('Mid', monthly_mid, 'green'), ('Large', monthly_large, 'purple')]:
            if not df.empty:
                ax.plot(pd.to_datetime(df['trade_date']), df['nav'], label=name, alpha=0.8)
        ax.axhline(y=1, color='gray', linestyle='--', alpha=0.5)
        ax.set_title('NAV by Domain (2020-2025)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 2. IC对比
        ax = axes[0, 1]
        for name, df_ic, color in [('All', ic_all, 'steelblue'), ('Small', ic_small, 'coral'),
                                    ('Mid', ic_mid, 'green'), ('Large', ic_large, 'purple')]:
            if not df_ic.empty:
                ax.plot(range(len(df_ic)), df_ic['ic'].rolling(3, min_periods=1).mean(), 
                       label=name, alpha=0.7)
        ax.axhline(y=0, color='black', alpha=0.3)
        ax.set_title('IC by Domain (3-month MA)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 3. 小盘股净值
        ax = axes[1, 0]
        if not monthly_small.empty:
            ax.plot(pd.to_datetime(monthly_small['trade_date']), monthly_small['nav'], color='coral', linewidth=2)
            ax.axhline(y=1, color='gray', linestyle='--', alpha=0.5)
            ax.set_title('Small Cap NAV')
            ax.grid(True, alpha=0.3)
        
        # 4. 月度收益分布
        ax = axes[1, 1]
        if not monthly_small.empty:
            mrets = monthly_small['ret_net'] * 100
            ax.hist(mrets, bins=20, color='coral', edgecolor='white', alpha=0.7)
            ax.axvline(x=mrets.mean(), color='blue', linestyle='--', label=f'Mean={mrets.mean():.2f}%')
            ax.set_title('Small Cap Monthly Return Dist')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(out, 'backtest_v3b_domain_comparison.png'), dpi=150)
        plt.close()
        print(f"\n  图表已保存: {os.path.join(out, 'backtest_v3b_domain_comparison.png')}")
    except Exception as e:
        print(f"  绘图失败: {e}")
    
    print("\n" + "=" * 70)
    print("Study 007 v3b 完成")
    print("=" * 70)

if __name__ == '__main__':
    main()

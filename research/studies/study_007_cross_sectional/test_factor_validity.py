"""
Study 007 因子有效性检验 - 分组测试 + IC分析
"""
import os
import sys
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))


def load_price_data(start_date, end_date):
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
    return pd.concat(dfs, ignore_index=True), dates


def calc_simple_factors(price_df):
    """计算简化因子"""
    price_df = price_df.sort_values(['ts_code', 'trade_date'])
    price_df['daily_ret'] = price_df.groupby('ts_code')['close'].pct_change()
    price_df['future_ret_20d'] = price_df.groupby('ts_code')['close'].pct_change(20).shift(-20)
    
    # 反转
    price_df['ret_1m'] = -price_df.groupby('ts_code')['close'].pct_change(20)
    price_df['ret_1m_accel'] = -price_df.groupby('ts_code')['ret_1m'].diff(5)
    
    # 低波动
    price_df['ivol'] = -price_df.groupby('ts_code')['daily_ret'].transform(lambda x: x.rolling(20, min_periods=10).std())
    
    # 隔夜/日内
    price_df['overnight_ret'] = (price_df['open'] - price_df['pre_close']) / price_df['pre_close']
    price_df['intraday_ret'] = (price_df['close'] - price_df['open']) / price_df['open']
    price_df['overnight_5d'] = price_df.groupby('ts_code')['overnight_ret'].transform(lambda x: x.rolling(5, min_periods=1).mean())
    price_df['intraday_5d'] = price_df.groupby('ts_code')['intraday_ret'].transform(lambda x: x.rolling(5, min_periods=1).mean())
    price_df['overnight_intraday'] = price_df['intraday_5d'] - price_df['overnight_5d']
    
    # 基本面
    funda_p = r'D:\iquant_data\data_v2\fundamental1\fina_indicator_cache.parquet'
    if os.path.exists(funda_p):
        funda = pd.read_parquet(funda_p).sort_values(['ts_code', 'ann_date'], ascending=[True, False]).groupby('ts_code').first().reset_index()
        price_df = price_df.merge(funda[['ts_code', 'roe', 'or_yoy', 'netprofit_yoy', 'netprofit_margin', 'grossprofit_margin']], on='ts_code', how='left')
    
    return price_df


def quintile_test(df, factor_col, ret_col='future_ret_20d'):
    """五分组测试"""
    results = []
    
    for date, group in df.groupby('trade_date'):
        group = group.dropna(subset=[factor_col, ret_col])
        if len(group) < 50:
            continue
        
        group['quintile'] = pd.qcut(group[factor_col], q=5, labels=['Q1', 'Q2', 'Q3', 'Q4', 'Q5'], duplicates='drop')
        
        for q in ['Q1', 'Q2', 'Q3', 'Q4', 'Q5']:
            q_df = group[group['quintile'] == q]
            if len(q_df) > 0:
                results.append({
                    'date': date,
                    'quintile': q,
                    'mean_ret': q_df[ret_col].mean(),
                    'median_ret': q_df[ret_col].median(),
                    'n': len(q_df)
                })
    
    return pd.DataFrame(results)


def ic_analysis(df, factor_col, ret_col='future_ret_20d'):
    """IC分析"""
    ics = []
    for date, group in df.groupby('trade_date'):
        group = group.dropna(subset=[factor_col, ret_col])
        if len(group) < 30:
            continue
        ic = group[factor_col].corr(group[ret_col], method='spearman')
        ics.append({'date': date, 'ic': ic})
    return pd.DataFrame(ics)


def main():
    print("=" * 60)
    print("Study 007: 因子有效性检验")
    print("=" * 60)
    
    print("\n[1/2] 加载数据并计算因子...")
    price_df, dates = load_price_data('20230101', '20241231')
    price_df = calc_simple_factors(price_df)
    
    factors = ['ret_1m', 'ret_1m_accel', 'ivol', 'overnight_intraday', 'roe', 'or_yoy', 'netprofit_yoy']
    available = [c for c in factors if c in price_df.columns]
    
    print(f"  可用因子: {available}")
    
    # 截面标准化
    for col in available:
        price_df[f'{col}_std'] = price_df.groupby('trade_date')[col].transform(lambda x: (x - x.mean()) / (x.std() + 1e-8))
    
    std_cols = [f'{c}_std' for c in available]
    price_df['composite_score'] = price_df[std_cols].mean(axis=1)
    
    print("\n[2/2] 分组测试与IC分析...")
    
    # 组合因子分组测试
    quintile_df = quintile_test(price_df, 'composite_score')
    if not quintile_df.empty:
        summary = quintile_df.groupby('quintile')['mean_ret'].mean()
        print("\n组合因子 - 五分组平均收益:")
        for q in ['Q1', 'Q2', 'Q3', 'Q4', 'Q5']:
            if q in summary.index:
                print(f"  {q}: {summary[q]*100:>8.3f}%")
        if 'Q5' in summary.index and 'Q1' in summary.index:
            print(f"  Q5-Q1: {(summary['Q5'] - summary['Q1'])*100:>8.3f}%")
    
    # 组合因子IC
    ic_df = ic_analysis(price_df, 'composite_score')
    if not ic_df.empty:
        print(f"\n组合因子IC:")
        print(f"  Mean IC: {ic_df['ic'].mean():.4f}")
        print(f"  IC Std:  {ic_df['ic'].std():.4f}")
        print(f"  IR:      {ic_df['ic'].mean() / (ic_df['ic'].std() + 1e-8):.3f}")
        print(f"  IC>0:    {(ic_df['ic'] > 0).mean()*100:.1f}%")
    
    # 各单因子IC
    print(f"\n单因子IC:")
    for col in available:
        ic_df_single = ic_analysis(price_df, col)
        if not ic_df_single.empty:
            mean_ic = ic_df_single['ic'].mean()
            ir = mean_ic / (ic_df_single['ic'].std() + 1e-8)
            print(f"  {col:<25} IC={mean_ic:>7.4f}  IR={ir:>6.3f}")
    
    # 各单因子分组测试
    print(f"\n单因子 Q5-Q1 收益差:")
    for col in available:
        q_df = quintile_test(price_df, col)
        if not q_df.empty:
            summary = q_df.groupby('quintile')['mean_ret'].mean()
            if 'Q5' in summary.index and 'Q1' in summary.index:
                diff = (summary['Q5'] - summary['Q1']) * 100
                print(f"  {col:<25} Q5-Q1={diff:>8.3f}%")
    
    print("\n" + "=" * 60)


if __name__ == '__main__':
    main()

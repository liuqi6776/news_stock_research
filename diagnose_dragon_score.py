"""
诊断脚本：检查 dragon_ai_score 与实际收益的相关性
"""
import pandas as pd
import numpy as np

# 加载预测数据
pred = pd.read_parquet(r'C:\Users\liuqi\iquant\quant_trading_system\dragon_rolling_predictions.parquet')
pred['trade_date'] = pd.to_datetime(pred['trade_date'])
pred = pred[pred['ts_code'].str.match(r'^(00|60)')]

print('Dragon AI Score 分布:')
print(pred['dragon_ai_score'].describe())
print()

# 找一个 2024 年 Q2-Q3 数据（市场回暖期）来测检相关性
import os
DATA_DIR = r'D:\iquant_data\data_v2\data_day1'

# 加载 2024-04 到 2024-09 的价格
all_files = []
for fname in sorted(os.listdir(DATA_DIR)):
    if not fname.endswith('.parquet'):
        continue
    ds = fname.replace('.parquet', '')
    if '20240401' <= ds <= '20240930':
        try:
            tmp = pd.read_parquet(os.path.join(DATA_DIR, fname), 
                                   columns=['ts_code', 'trade_date', 'open', 'close', 'pct_chg'])
            all_files.append(tmp)
        except:
            pass

if all_files:
    price = pd.concat(all_files, ignore_index=True)
    price['trade_date'] = pd.to_datetime(price['trade_date'].astype(str))
    price = price[price['ts_code'].str.match(r'^(00|60)')]
    
    # 收益率 = 5日后收盘价 / 今天收盘价
    price = price.sort_values(['ts_code', 'trade_date'])
    price['ret_5d'] = price.groupby('ts_code')['close'].transform(lambda x: x.shift(-5) / x - 1)
    
    # 合并
    merged = pd.merge(
        pred[(pred['trade_date'] >= '2024-04-01') & (pred['trade_date'] <= '2024-09-30')],
        price[['ts_code', 'trade_date', 'ret_5d']],
        on=['ts_code', 'trade_date'],
        how='inner'
    )
    merged = merged.dropna(subset=['ret_5d'])
    
    print(f'合并后记录数: {len(merged):,}')
    print(f'5日收益率均值: {merged.ret_5d.mean()*100:.2f}%')
    print()
    
    # 按分数分组，看各组平均 5 日收益率
    merged['score_bin'] = pd.cut(merged['dragon_ai_score'],
                                  bins=[0, 0.3, 0.5, 0.65, 0.75, 0.85, 1.0],
                                  labels=['<0.3', '0.3-0.5', '0.5-0.65', '0.65-0.75', '0.75-0.85', '>0.85'])
    print('各分数区间的5日平均收益率:')
    grp = merged.groupby('score_bin')['ret_5d'].agg(['mean', 'count', 'std'])
    grp['mean_pct'] = grp['mean'] * 100
    print(grp[['mean_pct', 'count', 'std']].to_string())
    print()
    
    # IC 值 (信息系数 = 分数与收益的 Spearman 相关性)
    ic = merged.groupby('trade_date').apply(
        lambda g: g['dragon_ai_score'].corr(g['ret_5d'], method='spearman')
    )
    print(f'IC 均值 (Spearman, 5日): {ic.mean():.4f}')
    print(f'IC 标准差: {ic.std():.4f}')
    print(f'ICIR (IC/std): {ic.mean()/ic.std():.4f}')
    print(f'正 IC 比例: {(ic > 0).mean()*100:.1f}%')
else:
    print('未找到价格数据！')

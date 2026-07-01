#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
综合实验：study_004 + 隔夜风控
================================
模拟 study_004 选股（动量+资金流+筹码），对比四种隔夜风控方案：
  1. 无风控（基线）
  2. 历史波动风控（剔除高 hist_vol5）
  3. DOS风控（剔除高 DOS）
  4. 组合风控（hist_vol5 + DOS 综合评分）

让历史波动当考官，看DOS能否在真实策略中提供增量价值。
"""

import pandas as pd
import numpy as np
import os
import json

SAVE_DIR = 'C:/Users/liuqi/quant_system_v2'

def main():
    print("="*70)
    print("综合实验：study_004 + 隔夜风控（历史波动 vs DOS vs 组合）")
    print("="*70)
    
    # 1. 加载数据
    print("\n[1/4] 加载数据...")
    df = pd.read_csv(f'{SAVE_DIR}/study_a_features_v3.csv', low_memory=False)
    df['trade_date'] = df['trade_date'].astype(int)
    df = df.sort_values(['ts_code', 'trade_date']).reset_index(drop=True)
    print(f"Total: {len(df)} rows, {df['trade_date'].nunique()} dates")
    
    # 2. 构建 study_004 模拟方向评分
    print("\n[2/4] 构建 study_004 方向评分...")
    # 和研究B一致：动量+资金流+筹码+波动
    direction_features = [
        'returns_5d', 'returns_10d', 'returns_20d',
        'mom_5d', 'mom_10d', 'mom_20d',
        'price_vs_ma5', 'price_vs_ma20',
        'vol_5d', 'vol_ratio',
        'mf_ratio', 'mf_ratio_5d',
        'winner_rate', 'profit_pressure',
    ]
    direction_features = [f for f in direction_features if f in df.columns]
    
    # 逐日标准化后加权
    for col in direction_features:
        df[f'{col}_rank'] = df.groupby('trade_date')[col].rank(pct=True)
    
    # study_004 评分：动量40% + 资金流30% + 筹码20% + 波动10%
    df['study004_score'] = (
        0.15 * df['mom_5d_rank'].fillna(0.5) +
        0.15 * df['mom_10d_rank'].fillna(0.5) +
        0.10 * df['mom_20d_rank'].fillna(0.5) +
        0.15 * df['mf_ratio_rank'].fillna(0.5) +
        0.15 * df['mf_ratio_5d_rank'].fillna(0.5) +
        0.10 * df['winner_rate_rank'].fillna(0.5) +
        0.10 * df['profit_pressure_rank'].fillna(0.5) +
        0.10 * df['vol_5d_rank'].fillna(0.5)
    )
    df['study004_score'] = df['study004_score'].fillna(0.5)
    
    # 3. 四种风控方案
    print("\n[3/4] 运行四种风控方案...")
    
    # 只测试有新闻的日期（2024-11起）
    test_df = df[df['trade_date'] >= 20241101].copy()
    test_df = test_df[test_df['trade_date'] <= 20250228]
    test_dates = sorted(test_df['trade_date'].unique())
    print(f"Test dates: {len(test_dates)}")
    
    # 方案定义
    strategies = {
        'baseline': {
            'name': '无风控（基线）',
            'filter': lambda day: day,
        },
        'hist_vol': {
            'name': '历史波动风控（剔高hist_vol5）',
            'filter': lambda day: day[day['vol_5d'] < day['vol_5d'].quantile(0.7)],
        },
        'dos_only': {
            'name': 'DOS风控（剔高DOS）',
            'filter': lambda day: day[day['dos'] < day['dos'].quantile(0.7)],
        },
        'combo': {
            'name': '组合风控（hist_vol5 + DOS）',
            'filter': lambda day: day[
                (day['vol_5d'] < day['vol_5d'].quantile(0.85)) & 
                (day['dos'] < day['dos'].quantile(0.85))
            ],
        },
    }
    
    all_results = {}
    
    for key, config in strategies.items():
        print(f"\n--- {config['name']} ---")
        daily_results = []
        
        for date in test_dates:
            day = test_df[test_df['trade_date'] == date].copy()
            if len(day) < 20:
                continue
            
            # 按 study_004 评分排序
            day = day.sort_values('study004_score', ascending=False)
            
            # 应用风控
            filtered = config['filter'](day)
            if len(filtered) < 5:
                filtered = day.head(20)  # fallback
            
            # 取 top-20
            top20 = filtered.head(20)
            
            daily_results.append({
                'date': int(date),
                'next_return_mean': float(top20['next_return'].mean()),
                'next_open_pct_mean': float(top20['next_open_pct'].mean()),
                'win_rate': float((top20['next_return'] > 0).mean()),
                'gap_down_rate': float((top20['next_open_pct'] < -2).mean()),
                'big_gap_down_rate': float((top20['next_open_pct'] < -5).mean()),  # 隔夜跳水 >5%
                'count': int(len(top20)),
            })
        
        dr = pd.DataFrame(daily_results)
        if len(dr) == 0:
            continue
        
        # 累积收益
        dr['cum_return'] = (1 + dr['next_return_mean']).cumprod() - 1
        
        # 最大回撤
        peak = dr['cum_return'].cummax()
        drawdown = (dr['cum_return'] - peak) / (peak + 1)
        max_dd = drawdown.min()
        
        results = {
            'name': config['name'],
            'n_days': len(dr),
            'avg_return': float(dr['next_return_mean'].mean()),
            'avg_open': float(dr['next_open_pct_mean'].mean()),
            'win_rate': float(dr['win_rate'].mean()),
            'gap_down_rate': float(dr['gap_down_rate'].mean()),
            'big_gap_down_rate': float(dr['big_gap_down_rate'].mean()),
            'sharpe': float(dr['next_return_mean'].mean() / dr['next_return_mean'].std()) if dr['next_return_mean'].std() > 0 else 0,
            'max_drawdown': float(max_dd),
            'total_return': float(dr['cum_return'].iloc[-1]) if len(dr) > 0 else 0,
            'daily_returns': dr['next_return_mean'].tolist(),
        }
        
        all_results[key] = results
        
        print(f"  总收益: {results['total_return']:.2%}")
        print(f"  日均收益: {results['avg_return']:.4f}")
        print(f"  胜率: {results['win_rate']:.2%}")
        print(f"  跳空下跌(>2%): {results['gap_down_rate']:.2%}")
        print(f"  隔夜跳水(>5%): {results['big_gap_down_rate']:.2%}")
        print(f"  Sharpe: {results['sharpe']:.3f}")
        print(f"  最大回撤: {results['max_drawdown']:.2%}")
    
    # 4. 对比总结
    print("\n" + "="*70)
    print("对比总结")
    print("="*70)
    
    summary = pd.DataFrame({
        '策略': [all_results[k]['name'] for k in all_results],
        '总收益': [f"{all_results[k]['total_return']:.2%}" for k in all_results],
        '日均收益': [f"{all_results[k]['avg_return']:.4f}" for k in all_results],
        '胜率': [f"{all_results[k]['win_rate']:.2%}" for k in all_results],
        '跳空>2%': [f"{all_results[k]['gap_down_rate']:.2%}" for k in all_results],
        '跳水>5%': [f"{all_results[k]['big_gap_down_rate']:.2%}" for k in all_results],
        'Sharpe': [f"{all_results[k]['sharpe']:.3f}" for k in all_results],
        '最大回撤': [f"{all_results[k]['max_drawdown']:.2%}" for k in all_results],
    })
    print(summary.to_string(index=False))
    
    # 5. 判定
    print("\n" + "="*70)
    print("判定：DOS 是否通过了历史波动的考官？")
    print("="*70)
    
    baseline = all_results['baseline']
    hist_vol = all_results['hist_vol']
    dos_only = all_results['dos_only']
    combo = all_results['combo']
    
    # 标准1: DOS风控是否比无风控好？
    s1 = dos_only['big_gap_down_rate'] < baseline['big_gap_down_rate']
    print(f"S1: DOS降跳水 > 无风控? {'PASS' if s1 else 'FAIL'}")
    print(f"    无风控跳水: {baseline['big_gap_down_rate']:.2%} → DOS: {dos_only['big_gap_down_rate']:.2%}")
    
    # 标准2: DOS是否比历史波动好？（考官标准）
    s2 = dos_only['big_gap_down_rate'] < hist_vol['big_gap_down_rate']
    print(f"S2: DOS降跳水 > 历史波动? {'PASS' if s2 else 'FAIL'}")
    print(f"    历史波动跳水: {hist_vol['big_gap_down_rate']:.2%} → DOS: {dos_only['big_gap_down_rate']:.2%}")
    
    # 标准3: 组合是否比单历史波动好？
    s3 = combo['big_gap_down_rate'] < hist_vol['big_gap_down_rate']
    print(f"S3: 组合降跳水 > 历史波动? {'PASS' if s3 else 'FAIL'}")
    print(f"    历史波动跳水: {hist_vol['big_gap_down_rate']:.2%} → 组合: {combo['big_gap_down_rate']:.2%}")
    
    # 标准4: 组合是否比单DOS好？
    s4 = combo['big_gap_down_rate'] < dos_only['big_gap_down_rate']
    print(f"S4: 组合降跳水 > 单DOS? {'PASS' if s4 else 'FAIL'}")
    print(f"    DOS跳水: {dos_only['big_gap_down_rate']:.2%} → 组合: {combo['big_gap_down_rate']:.2%}")
    
    # 标准5: Sharpe是否提升？
    s5 = combo['sharpe'] > baseline['sharpe']
    print(f"S5: 组合Sharpe > 基线? {'PASS' if s5 else 'FAIL'}")
    print(f"    基线Sharpe: {baseline['sharpe']:.3f} → 组合: {combo['sharpe']:.3f}")
    
    print(f"\n最终: DOS {'通过' if (s1 and s2) else '未通过'} 历史波动考官")
    print(f"       组合 {'有价值' if (s3 and s4 and s5) else '价值有限'}")
    
    # 保存
    with open(f'{SAVE_DIR}/study004_risk_control_results.json', 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    
    print(f"\n\nSaved to study004_risk_control_results.json")


if __name__ == '__main__':
    main()

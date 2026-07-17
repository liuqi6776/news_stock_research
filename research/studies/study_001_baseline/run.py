"""
Study 001: 基线模型运行脚本

流程：
1. 加载或计算特征数据
2. 训练模型并保存预测
3. 运行回测
4. 保存结果
"""
import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime
import json

# 添加shared模块路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from shared.data_loader import get_all_dates, PRICE_DIR
from shared.feature_engineering import prepare_training_data, calculate_all_features
from shared.models import train_xgboost, predict_proba, get_feature_importance
from shared.backtest_engine import run_backtest
from config import *


def load_or_compute_features(force_recompute=False):
    """
    加载或计算特征数据
    
    如果data/features.parquet存在且force_recompute=False，直接加载
    否则重新计算并保存
    """
    features_path = os.path.join(DATA_DIR, 'features.parquet')
    
    if os.path.exists(features_path) and not force_recompute:
        print(f"加载已存在的特征数据: {features_path}")
        return pd.read_parquet(features_path)
    
    print("计算特征数据...")
    all_dates = get_all_dates()
    
    # 计算特征和目标变量
    features_df = prepare_training_data(START_DATE, END_DATE, all_dates, PRICE_DIR)
    
    if not features_df.empty:
        features_df.to_parquet(features_path)
        print(f"特征数据已保存: {features_path}")
    
    return features_df


def load_or_compute_predictions(features_df, feature_cols, force_recompute=False):
    """
    加载或计算预测数据
    
    如果data/predictions.parquet存在且force_recompute=False，直接加载
    否则重新训练模型并保存预测
    """
    pred_path = os.path.join(DATA_DIR, 'predictions.parquet')
    
    if os.path.exists(pred_path) and not force_recompute:
        print(f"加载已存在的预测数据: {pred_path}")
        return pd.read_parquet(pred_path)
    
    print("训练模型并生成预测...")
    all_dates = get_all_dates()
    
    # Walk Forward预测
    predictions = []
    
    # 简化为单期预测（实际应该walk forward）
    # 这里仅作为示例
    train_mask = features_df['trade_date'] < '20240101'
    test_mask = features_df['trade_date'] >= '20240101'
    
    train_df = features_df[train_mask]
    test_df = features_df[test_mask]
    
    # 准备标签
    train_df['label'] = (train_df['label_ret'] > LABEL_THRESHOLD).astype(int)
    
    # 训练模型
    model = train_xgboost(train_df, train_df['label'], feature_cols)
    
    # 预测
    test_df = test_df.copy()
    test_df['prob'] = predict_proba(model, test_df, feature_cols)
    
    # 保存预测
    pred_df = test_df[['trade_date', 'ts_code', 'prob', 'label_ret']].copy()
    pred_df.to_parquet(pred_path)
    print(f"预测数据已保存: {pred_path}")
    
    return pred_df


def save_summary(results):
    """保存结果摘要"""
    summary = {
        'study_id': '001',
        'study_name': '基线模型',
        'timestamp': datetime.now().strftime('%Y%m%d_%H%M%S'),
        'total_return': float(results['total_return']),
        'sharpe': float(results['sharpe']),
        'max_drawdown': float(results['max_drawdown']),
        'n_trades': int(results['n_trades']),
        'win_rate': float(results['win_rate'])
    }
    
    summary_path = os.path.join(RESULTS_DIR, f"summary_{summary['timestamp']}.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"摘要已保存: {summary_path}")
    
    # 更新README
    update_readme(summary)


def update_readme(summary):
    """更新README中的结果部分"""
    readme_path = os.path.join(os.path.dirname(__file__), 'README.md')
    
    with open(readme_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 更新结果部分
    result_section = f"""
## 最新结果 ({summary['timestamp']})

| 指标 | 数值 |
|------|------|
| 总收益率 | {summary['total_return']:.2%} |
| 夏普比率 | {summary['sharpe']:.2f} |
| 最大回撤 | {summary['max_drawdown']:.2%} |
| 交易次数 | {summary['n_trades']} |
| 胜率 | {summary['win_rate']:.2%} |
"""
    
    # 替换或添加结果部分
    if '## 最新结果' in content:
        # 替换现有结果
        parts = content.split('## 结果历史')
        if len(parts) == 2:
            content = parts[0] + result_section + '\n## 结果历史' + parts[1]
        else:
            content = content.split('## 最新结果')[0] + result_section
    else:
        content += '\n' + result_section
    
    with open(readme_path, 'w', encoding='utf-8') as f:
        f.write(content)


def run_study():
    """运行完整研究流程"""
    print("=" * 80)
    print("Study 001: 基线模型")
    print("=" * 80)
    
    # 1. 加载/计算特征
    features_df = load_or_compute_features(force_recompute=False)
    print(f"特征数据: {len(features_df)} 行")
    
    # 2. 选择特征列
    feature_cols = [c for c in features_df.columns if c not in [
        'ts_code', 'trade_date', 't1_open', 't2_close', 'label_ret', 'label'
    ]]
    print(f"特征数量: {len(feature_cols)}")
    
    # 3. 加载/计算预测
    predictions_df = load_or_compute_predictions(features_df, feature_cols, force_recompute=False)
    print(f"预测数据: {len(predictions_df)} 行")
    
    # 4. 运行回测
    all_dates = get_all_dates()
    test_dates = [d for d in all_dates if d >= '20240101' and d <= '20241231']
    
    print("\n运行回测...")
    results = run_backtest(
        predictions_df, 
        test_dates, 
        PRICE_DIR,
        min_prob=MIN_PROB,
        stop_loss=STOP_LOSS,
        max_positions=MAX_POSITIONS
    )
    
    # 5. 保存结果
    print("\n回测结果:")
    print(f"  总收益率: {results['total_return']:.2%}")
    print(f"  夏普比率: {results['sharpe']:.2f}")
    print(f"  最大回撤: {results['max_drawdown']:.2%}")
    print(f"  交易次数: {results['n_trades']}")
    print(f"  胜率: {results['win_rate']:.2%}")
    
    # 保存详细结果
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    trades_df = pd.DataFrame(results['trades'])
    trades_df.to_csv(os.path.join(RESULTS_DIR, f'trades_{timestamp}.csv'), index=False)
    
    results['nav'].to_csv(os.path.join(RESULTS_DIR, f'nav_{timestamp}.csv'), index=False)
    
    # 保存摘要
    save_summary(results)
    
    print(f"\n结果已保存到: {RESULTS_DIR}")
    
    # 更新注册表
    print("\n更新研究注册表...")
    os.system(f"cd {os.path.join(os.path.dirname(__file__), '..', '..')} && python update_registry.py")


if __name__ == '__main__':
    run_study()

"""
Step 5: 每日信号生成（生产环境）

在每天收盘后运行，生成次日买入信号

输入: 当日原始数据 + models/contrarian_model.joblib + results/contrarian_optimized.json
输出: signals/YYYYMMDD.json

耗时: 约1-2分钟
运行频率: 每个交易日收盘后
"""
import os
import sys
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (
    RAW_PRICE_DIR, RAW_OTHER_DIR, RAW_NEWS_DIR, RAW_RANK_DIR,
    RAW_MONEYFLOW_DIR, RAW_THS_NEWS_DIR, RAW_INCOME_DIR,
    MODEL_FILE, FEATS_FILE, OPT_RESULTS_FILE, SIGNALS_DIR,
    MIN_LISTING_DAYS, FUNDAMENTAL_FILTERS
)
from build_contrarian_features import (
    add_valuation_percentiles, add_momentum_reversal_features,
    add_drawdown_features, add_fundamental_features
)


def is_main_board(ts_code: str) -> bool:
    return ts_code.startswith(('60', '00', '002', '003'))


def load_today_raw_features(date_str):
    """加载当日原始特征（复用v2 pipeline的代码）"""
    p = os.path.join(RAW_PRICE_DIR, f"{date_str}.parquet")
    if not os.path.exists(p):
        return None

    df = pd.read_parquet(p)
    df = df[df['ts_code'].apply(is_main_board)]
    if df.empty:
        return None

    # 基础价格特征
    df['pct_chg'] = (df['close'] - df['pre_close']) / df['pre_close']
    df['amplitude'] = (df['high'] - df['low']) / df['pre_close']
    df['body_size'] = abs(df['close'] - df['open']) / df['pre_close']
    df['upper_shadow'] = (df['high'] - df[['close', 'open']].max(axis=1)) / df['pre_close']
    df['lower_shadow'] = (df[['close', 'open']].min(axis=1) - df['low']) / df['pre_close']
    df['is_yang'] = (df['close'] > df['open']).astype(int)
    df['gap'] = (df['open'] - df['pre_close']) / df['pre_close']
    df['close_to_high'] = (df['high'] - df['close']) / df['pre_close']
    df['close_to_low'] = (df['close'] - df['low']) / df['pre_close']
    df['vol_ratio'] = 1.0
    df['vol_amount'] = df['close'] * df['vol']

    # 其他特征（简化版，实际运行需要更完整的历史数据计算动量/RSI等）
    # 这里假设 daily signal 脚本会与 step1_build_features 共用逻辑
    # 为简化，我们只加载已有特征文件中的最新一天数据来补充历史统计

    return df


def generate_signal(target_date=None):
    """生成次日买入信号"""
    print("=" * 80)
    print("Step 5: 每日信号生成")
    print("=" * 80)

    # 检查模型和参数
    if not os.path.exists(MODEL_FILE) or not os.path.exists(OPT_RESULTS_FILE):
        print("错误: 模型或参数文件不存在，请先运行训练步骤")
        return

    import joblib
    model = joblib.load(MODEL_FILE)
    feature_cols = joblib.load(FEATS_FILE)

    with open(OPT_RESULTS_FILE, 'r') as f:
        opt_params = json.load(f)
    bp = opt_params['best_params']

    print(f"模型: {MODEL_FILE}")
    print(f"最优参数: threshold={bp['threshold']:.2f}, max_positions={bp['max_positions']}")

    # 确定信号日期
    if target_date is None:
        from shared.data_loader import get_all_dates
        all_dates = get_all_dates()
        today_str = datetime.now().strftime('%Y%m%d')
        target_date = None
        for d in reversed(all_dates):
            if d <= today_str:
                target_date = d
                break
        if target_date is None:
            print("错误: 无法确定最近交易日")
            return

    print(f"信号日期: {target_date}")

    # 简化方案：加载最新特征文件中的当日数据
    # 实际生产环境需要运行完整的特征工程流程
    # 这里假设特征文件已包含到最新日期
    from config import FEATURES_FILE
    if not os.path.exists(FEATURES_FILE):
        print("错误: 特征文件不存在")
        return

    df = pd.read_parquet(FEATURES_FILE)
    df['ds'] = df['trade_date'].astype(str)
    today_df = df[df['ds'] == target_date].copy()

    if len(today_df) == 0:
        print(f"错误: 特征文件中无 {target_date} 数据")
        return

    print(f"当日候选股票: {len(today_df)} 只")

    # 基本面硬过滤
    n_before = len(today_df)
    if FUNDAMENTAL_FILTERS.get('min_pe') is not None:
        today_df = today_df[today_df['pe'] > FUNDAMENTAL_FILTERS['min_pe']]
    if FUNDAMENTAL_FILTERS.get('max_pe') is not None:
        today_df = today_df[today_df['pe'] < FUNDAMENTAL_FILTERS['max_pe']]
    if FUNDAMENTAL_FILTERS.get('max_pb') is not None:
        today_df = today_df[today_df['pb'] < FUNDAMENTAL_FILTERS['max_pb']]
    if 'peg' in today_df.columns and FUNDAMENTAL_FILTERS.get('max_peg') is not None:
        today_df = today_df[(today_df['peg'].isna()) | (today_df['peg'] < FUNDAMENTAL_FILTERS['max_peg'])]

    print(f"基本面过滤后: {len(today_df)} 只 (过滤掉 {n_before - len(today_df)} 只)")

    # 预测
    available_features = [c for c in feature_cols if c in today_df.columns]
    missing = [c for c in feature_cols if c not in today_df.columns]
    if missing:
        print(f"缺失特征 ({len(missing)}): {missing[:5]}...")
        for c in missing:
            today_df[c] = 0

    X = today_df[feature_cols].fillna(0)
    proba = model.predict_proba(X)[:, 1]
    today_df['prob'] = proba

    # 选择
    above = today_df[today_df['prob'] >= bp['threshold']].copy()
    above = above.sort_values('prob', ascending=False)
    selected = above.head(bp['max_positions'])

    # 生成信号
    signal = {
        'signal_date': target_date,
        'generated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'params': bp,
        'n_candidates': int(len(above)),
        'picks': [],
    }

    for _, row in selected.iterrows():
        pick = {
            'ts_code': row['ts_code'],
            'prob': float(row['prob']),
            'close': float(row['close']),
            'pe': float(row['pe']) if 'pe' in row and pd.notna(row['pe']) else None,
            'pb': float(row['pb']) if 'pb' in row and pd.notna(row['pb']) else None,
            'peg': float(row['peg']) if 'peg' in row and pd.notna(row['peg']) else None,
            'rsi_14': float(row['rsi_14']) if 'rsi_14' in row and pd.notna(row['rsi_14']) else None,
            'drawdown_60d': float(row['drawdown_60d']) if 'drawdown_60d' in row and pd.notna(row['drawdown_60d']) else None,
            'mom_60d': float(row['mom_60d']) if 'mom_60d' in row and pd.notna(row['mom_60d']) else None,
            'reversal_score': float(row['reversal_score']) if 'reversal_score' in row and pd.notna(row['reversal_score']) else None,
        }
        signal['picks'].append(pick)

    signal_path = os.path.join(SIGNALS_DIR, f'{target_date}.json')
    with open(signal_path, 'w', encoding='utf-8') as f:
        json.dump(signal, f, indent=2, ensure_ascii=False)

    print(f"\n信号已生成: {signal_path}")
    print(f"候选股票: {len(above)} 只 (prob >= {bp['threshold']:.2f})")
    print(f"选中股票: {len(selected)} 只 (top {bp['max_positions']})")
    for pick in signal['picks']:
        print(f"  {pick['ts_code']}: prob={pick['prob']:.4f}, close={pick['close']:.2f}, "
              f"PE={pick['pe']:.1f}, PEG={pick['peg']:.2f}, RSI={pick['rsi_14']:.1f}")

    return signal


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--date', type=str, default=None, help='目标日期 YYYYMMDD')
    args = parser.parse_args()
    generate_signal(target_date=args.date)

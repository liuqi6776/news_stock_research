"""
Step 1: 构建逆向策略特征

在现有v2特征基础上，增加：
1. 估值历史分位数（PE/PB/RSI/布林带）
2. 动量反转特征（短期vs长期动量差异）
3. 回撤深度特征
4. 基本面补充（EPS增长率、PEG、负债率等）

输入: study_004_systematic/v2_pipeline/data/all_features_v2.parquet + 原始数据
输出: data/contrarian_features.parquet

耗时: 约30-60分钟（首次运行）
"""
import os
import sys
import pandas as pd
import numpy as np
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (
    BASE_FEATURES_FILE, FEATURES_FILE, RAW_INCOME_DIR, RAW_OTHER_DIR,
    TRAIN_START, MIN_LISTING_DAYS
)


def add_valuation_percentiles(df):
    """计算PE/PB/RSI/布林带的历史分位数（按股票滚动1年）"""
    print("计算估值历史分位数...")
    df = df.sort_values(['ts_code', 'trade_date']).copy()

    # 按股票分组计算250日滚动分位数
    df['pe_pct_rank'] = df.groupby('ts_code')['pe'].transform(
        lambda x: x.rolling(window=250, min_periods=60).apply(
            lambda s: (s <= s.iloc[-1]).mean() if len(s) > 0 else np.nan, raw=False
        )
    )
    df['pb_pct_rank'] = df.groupby('ts_code')['pb'].transform(
        lambda x: x.rolling(window=250, min_periods=60).apply(
            lambda s: (s <= s.iloc[-1]).mean() if len(s) > 0 else np.nan, raw=False
        )
    )
    df['rsi_pct_rank'] = df.groupby('ts_code')['rsi_14'].transform(
        lambda x: x.rolling(window=250, min_periods=60).apply(
            lambda s: (s <= s.iloc[-1]).mean() if len(s) > 0 else np.nan, raw=False
        )
    )
    df['bb_pct_rank'] = df.groupby('ts_code')['bb_position'].transform(
        lambda x: x.rolling(window=250, min_periods=60).apply(
            lambda s: (s <= s.iloc[-1]).mean() if len(s) > 0 else np.nan, raw=False
        )
    )

    # 低估值信号（PE/PB处于历史低位）
    df['is_low_valuation'] = ((df['pe_pct_rank'] < 0.20) & (df['pb_pct_rank'] < 0.30)).astype(int)
    # 超卖信号（RSI处于历史低位）
    df['is_oversold'] = (df['rsi_pct_rank'] < 0.15).astype(int)

    return df


def add_momentum_reversal_features(df):
    """计算动量反转特征"""
    print("计算动量反转特征...")
    df = df.sort_values(['ts_code', 'trade_date']).copy()

    # 短期vs长期动量差异（短期相对长期动量转强 = 反弹信号）
    df['mom_diff_5_60'] = df['mom_5d'] - df['mom_60d']
    df['mom_diff_10_60'] = df['mom_10d'] - df['mom_60d']
    df['mom_diff_20_60'] = df['mom_20d'] - df['mom_60d']

    # 动量反转得分：长期深跌 + 短期企稳/反弹
    df['reversal_score'] = (
        (-df['mom_60d'].clip(upper=0)) * 2 +          # 长期深跌加分
        df['mom_5d'].clip(lower=-0.5, upper=0.5) +     # 短期微涨加分
        (-df['mom_diff_5_60'].clip(upper=0))            # 短期比长期强加分
    )

    # 趋势强度：价格相对于20日均线的位置
    df['price_vs_ma20'] = df['mom_20d']  # 近似
    # 更直接的：利用布林带位置
    df['price_vs_bb_mid'] = df['bb_position']  # 已有

    return df


def add_drawdown_features(df):
    """计算回撤深度特征"""
    print("计算回撤深度特征...")
    df = df.sort_values(['ts_code', 'trade_date']).copy()

    # 60日最大回撤（从过去60日高点到当前价格的回撤）
    def calc_max_dd(group):
        prices = group['close'].values
        max_dd = []
        for i in range(len(prices)):
            if i < 20:
                max_dd.append(np.nan)
                continue
            window = prices[max(0, i-60):i+1]
            peak = np.max(window)
            if peak > 0:
                max_dd.append((prices[i] - peak) / peak)
            else:
                max_dd.append(np.nan)
        return pd.Series(max_dd, index=group.index)

    df['drawdown_60d'] = df.groupby('ts_code', group_keys=False).apply(calc_max_dd)

    # 深度回调信号（从高点回撤>30%）
    df['is_deep_drawdown'] = (df['drawdown_60d'] < -0.30).astype(int)

    # 价格从250日高点回撤（需要更长历史，用60日近似）
    df['drawdown_from_high'] = df['drawdown_60d']

    return df


def add_fundamental_features(df):
    """从原始数据加载基本面特征并合并"""
    print("加载并合并基本面数据...")

    # 尝试从income数据加载EPS
    income_data = []
    all_dates = df['trade_date'].astype(str).unique()

    for date in tqdm(sorted(all_dates), desc="加载income数据"):
        p = os.path.join(RAW_INCOME_DIR, f"{date}.parquet")
        if not os.path.exists(p):
            continue
        try:
            idf = pd.read_parquet(p, columns=['ts_code', 'trade_date', 'basic_eps'])
            idf['trade_date'] = idf['trade_date'].astype(str)
            income_data.append(idf)
        except:
            pass

    if income_data:
        income_df = pd.concat(income_data, ignore_index=True)
        income_df = income_df.dropna(subset=['basic_eps'])
        income_df = income_df.sort_values(['ts_code', 'trade_date'])
        # 每个股票每个日期只保留最新的一条（可能有重复）
        income_df = income_df.drop_duplicates(subset=['ts_code', 'trade_date'], keep='last')

        # 按股票计算EPS同比增长率（需要去年同期数据）
        income_df['eps_yoy'] = income_df.groupby('ts_code')['basic_eps'].pct_change(periods=4)
        # 保留有效值
        income_df = income_df[income_df['eps_yoy'].notna() & (income_df['eps_yoy'] != np.inf) & (income_df['eps_yoy'] != -np.inf)]

        # 合并到主数据
        df['trade_date_str'] = df['trade_date'].astype(str)
        income_df = income_df.rename(columns={'trade_date': 'trade_date_str'})
        df = pd.merge(df, income_df[['ts_code', 'trade_date_str', 'basic_eps', 'eps_yoy']],
                      on=['ts_code', 'trade_date_str'], how='left')

        # 计算PEG = PE / (EPS增长率 * 100)
        # 注意：eps_yoy是增长率，如0.20表示20%增长
        df['peg'] = np.where(
            (df['eps_yoy'] > 0) & (df['pe'] > 0),
            df['pe'] / (df['eps_yoy'] * 100),
            np.nan
        )
        df['peg'] = df['peg'].clip(upper=50)  # 限制极端值

        print(f"  成功合并EPS数据: {income_df['ts_code'].nunique()} 只股票")
    else:
        print("  警告: 未找到income数据，PEG不可用")
        df['basic_eps'] = np.nan
        df['eps_yoy'] = np.nan
        df['peg'] = np.nan

    # 尝试从other数据加载更多基本面（如ROE、负债率）
    # 由于other数据结构不确定，先尝试加载一个样本
    sample_dates = sorted(all_dates)[:5]
    other_cols_found = set()
    for date in sample_dates:
        p = os.path.join(RAW_OTHER_DIR, f"{date}.parquet")
        if os.path.exists(p):
            try:
                odf = pd.read_parquet(p)
                other_cols_found.update(odf.columns)
                break
            except:
                pass

    if other_cols_found:
        print(f"  other_day1 可用列: {other_cols_found}")
        # 如果other中有roe, debt_ratio等，可以加载
        # 这里先留空，具体根据数据结构调整
    else:
        print("  警告: other_day1 数据未找到或无法读取")

    return df


def prepare_target(df):
    """准备28日目标变量"""
    print("准备目标变量...")

    # 使用已有的return_28d或return_28d_open
    if 'return_28d' in df.columns:
        target_col = 'return_28d'
    elif 'return_28d_open' in df.columns:
        target_col = 'return_28d_open'
    else:
        print("错误: 未找到28日收益列")
        return df

    df['target_return'] = df[target_col]
    df['label'] = (df['target_return'] > 0.05).astype(int)  # 28日收益>5%为正样本

    # 计算样本分布
    valid = df['target_return'].notna()
    print(f"  有效样本: {valid.sum()} / {len(df)}")
    print(f"  正样本比例: {df.loc[valid, 'label'].mean():.2%}")
    print(f"  平均收益: {df.loc[valid, 'target_return'].mean():.2%}")
    print(f"  收益中位数: {df.loc[valid, 'target_return'].median():.2%}")

    return df


def run():
    print("=" * 80)
    print("Step 1: 构建逆向策略特征")
    print("=" * 80)

    if not os.path.exists(BASE_FEATURES_FILE):
        print(f"错误: 基础特征文件不存在: {BASE_FEATURES_FILE}")
        print("请先运行 study_004_systematic/v2_pipeline/step1_build_features.py")
        return

    print(f"加载基础特征数据: {BASE_FEATURES_FILE}")
    df = pd.read_parquet(BASE_FEATURES_FILE)
    print(f"原始数据: {len(df)} 行, {len(df.columns)} 列")
    print(f"日期范围: {df['trade_date'].min()} - {df['trade_date'].max()}")

    # 过滤训练期开始前的数据
    df = df[df['trade_date'].astype(str) >= TRAIN_START].copy()
    print(f"过滤后: {len(df)} 行")

    # 增加特征
    df = add_valuation_percentiles(df)
    df = add_momentum_reversal_features(df)
    df = add_drawdown_features(df)
    df = add_fundamental_features(df)
    df = prepare_target(df)

    # 清理
    if 'trade_date_str' in df.columns:
        df = df.drop(columns=['trade_date_str'])

    # 保存
    df.to_parquet(FEATURES_FILE)
    print(f"\n特征数据已保存: {FEATURES_FILE}")
    print(f"最终数据: {len(df)} 行, {len(df.columns)} 列")
    print(f"新特征列: {[c for c in df.columns if c not in ['ts_code', 'trade_date', 'open', 'high', 'low', 'close', 'pre_close', 'change', 'pct_chg', 'vol', 'amount']]}")

    return df


if __name__ == '__main__':
    run()

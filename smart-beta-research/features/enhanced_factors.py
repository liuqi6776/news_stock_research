"""
增强版特征工程模块 (简化版)

包含8大类70+个特征:
1. Alpha101扩展
2. Alpha191精选 (国泰君安)
3. 技术指标 (MA/MACD/RSI/KDJ/布林带)
4. 滞后特征
5. 滚动统计量
6. 截面排名
7. 资金流向
8. 筹码分布

使用向量化操作优化性能
"""

import numpy as np
import pandas as pd
from typing import List, Optional
import warnings
warnings.filterwarnings('ignore')

try:
    from features.vibe_alpha_zoo import calculate_vibe_alphas
except ImportError:
    # Handle the case where it might be run from a different directory
    import sys, os
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    try:
        from features.vibe_alpha_zoo import calculate_vibe_alphas
    except ImportError:
        calculate_vibe_alphas = None



# ==================== 辅助函数 ====================

def safe_divide(a, b, fill_value=0):
    """安全除法，避免除零错误"""
    return np.where(b != 0, a / b, fill_value)


# ==================== 1. Alpha101 扩展因子 ====================

def calculate_alpha101_extended(df: pd.DataFrame) -> pd.DataFrame:
    """
    计算扩展版Alpha101因子 (简化版，使用向量化操作)
    """
    print("计算Alpha101扩展因子...")
    df = df.copy()
    g = df.groupby('ts_code')
    
    # 基础收益率
    df['returns'] = g['close'].pct_change()
    
    # 均量20日
    df['adv20'] = g['vol'].transform(lambda x: x.rolling(20, min_periods=1).mean())
    
    # Alpha 001: 5日波动率 vs 收益率的关系
    df['vol_20'] = g['returns'].transform(lambda x: x.rolling(20, min_periods=1).std())
    df['alpha_001'] = np.where(df['returns'] < 0, df['vol_20'], df['close'])
    df['alpha_001'] = df.groupby('ts_code')['alpha_001'].transform(
        lambda x: x.rolling(5, min_periods=1).apply(lambda a: a.argmax() / len(a) if len(a) > 0 else 0.5, raw=True)
    )
    
    # Alpha 002: 量价背离 (简化版)
    df['vol_log_delta'] = g['vol'].transform(lambda x: np.log(x + 1).diff(2))
    df['price_move'] = safe_divide(df['close'] - df['open'], df['open'] + 0.0001)
    df['alpha_002'] = -1 * df.groupby('ts_code').apply(
        lambda x: x['vol_log_delta'].rolling(6).corr(x['price_move'])
    ).reset_index(level=0, drop=True)
    
    # Alpha 004: 低价反转
    df['low_rank'] = g['low'].rank(pct=True)
    df['alpha_004'] = -1 * g['low_rank'].transform(lambda x: x.rolling(9, min_periods=1).mean())
    
    # Alpha 005: 开盘价偏离
    df['vwap_simple'] = (df['high'] + df['low'] + df['close']) / 3
    df['open_diff'] = df['open'] - g['vwap_simple'].transform(lambda x: x.rolling(10, min_periods=1).mean())
    df['close_diff'] = abs(df['close'] - df['vwap_simple'])
    df['alpha_005'] = -1 * df['open_diff'].rank(pct=True) * df['close_diff'].rank(pct=True)
    
    # Alpha 010: 动量切换
    df['delta_close'] = g['close'].diff(1)
    df['min4'] = g['delta_close'].transform(lambda x: x.rolling(4, min_periods=1).min())
    df['max4'] = g['delta_close'].transform(lambda x: x.rolling(4, min_periods=1).max())
    df['alpha_010'] = np.where(
        df['min4'] > 0, df['delta_close'],
        np.where(df['max4'] < 0, df['delta_close'], -1 * df['delta_close'])
    )
    df['alpha_010'] = df.groupby('trade_date')['alpha_010'].rank(pct=True)
    
    # Alpha 017: 综合动量 (简化)
    df['ts_rank_close'] = g['close'].transform(lambda x: x.rolling(10, min_periods=1).apply(
        lambda a: pd.Series(a).rank().iloc[-1] / len(a) if len(a) > 0 else 0.5, raw=False
    ))
    df['delta2'] = g['close'].diff(1).diff(1)
    df['vol_ratio'] = df['vol'] / (df['adv20'] + 1)
    df['alpha_017'] = -1 * df['ts_rank_close'] * df['delta2'].rank(pct=True) * df['vol_ratio'].rank(pct=True)
    
    # Alpha 033: 日内反转
    df['alpha_033'] = (-1 * (1 - df['open'] / df['close'])).rank(pct=True)
    
    # Alpha 041: VWAP偏离
    df['alpha_041'] = (df['high'] * df['low']) ** 0.5 - df['vwap_simple']
    
    # Alpha 054: 价格形态
    numerator = -1 * (df['low'] - df['close']) * (df['open'] ** 5)
    denominator = (df['low'] - df['high'] + 0.0001) * (df['close'] ** 5)
    df['alpha_054'] = safe_divide(numerator, denominator)
    
    # Alpha 060: 量价动量
    df['_inner_060'] = safe_divide((df['close'] - df['low']) - (df['high'] - df['close']), df['high'] - df['low'] + 0.0001)
    df['_inner_vol'] = df['_inner_060'] * df['vol']
    df['alpha_060'] = -1 * g['_inner_vol'].transform(lambda x: x.rolling(10, min_periods=1).sum())
    df = df.drop(columns=['_inner_060', '_inner_vol'], errors='ignore')
    
    # 清理临时列
    temp_cols = ['vol_20', 'vol_log_delta', 'price_move', 'low_rank', 'vwap_simple', 
                 'open_diff', 'close_diff', 'delta_close', 'min4', 'max4', 
                 'ts_rank_close', 'delta2', 'vol_ratio']
    df = df.drop(columns=[c for c in temp_cols if c in df.columns], errors='ignore')
    
    print(f"  完成: 新增10个Alpha101因子")
    return df


# ==================== 2. Alpha191 精选因子 ====================

def calculate_alpha191_factors(df: pd.DataFrame) -> pd.DataFrame:
    """
    计算国泰君安Alpha191精选因子 (简化版，使用向量化操作)
    """
    print("计算Alpha191因子 (国泰君安)...")
    df = df.copy()
    g = df.groupby('ts_code')
    
    # GTJA 001: 收益率排名
    if 'returns' in df.columns:
        df['gtja_001'] = df.groupby('trade_date')['returns'].rank(pct=True)
    else:
        df['_ret_temp'] = g['close'].pct_change()
        df['gtja_001'] = df.groupby('trade_date')['_ret_temp'].rank(pct=True)
        df = df.drop(columns=['_ret_temp'], errors='ignore')
    
    # GTJA 002: 量价动量 (价格位置变化)
    df['_inner_002'] = safe_divide((df['close'] - df['low']) - (df['high'] - df['close']), df['high'] - df['low'] + 0.0001)
    df['gtja_002'] = -1 * df.groupby('ts_code')['_inner_002'].diff(1)
    df = df.drop(columns=['_inner_002'], errors='ignore')
    
    # GTJA 007: VWAP位置动量 (简化版)
    df['_vwap'] = (df['high'] + df['low'] + df['close']) / 3
    df['_vwap_diff'] = df['_vwap'] - df['close']
    # 重新创建groupby以访问新列
    df['_vwap_max3'] = df.groupby('ts_code')['_vwap_diff'].transform(lambda x: x.rolling(3, min_periods=1).max())
    df['_vwap_min3'] = df.groupby('ts_code')['_vwap_diff'].transform(lambda x: x.rolling(3, min_periods=1).min())
    df['gtja_007'] = (df['_vwap_max3'].rank(pct=True) + df['_vwap_min3'].rank(pct=True)) * df.groupby('ts_code')['vol'].diff(3).rank(pct=True)
    df = df.drop(columns=['_vwap', '_vwap_diff', '_vwap_max3', '_vwap_min3'], errors='ignore')
    
    # GTJA 018: 短期反转
    df['gtja_018'] = safe_divide(df['close'], g['close'].shift(5) + 0.0001)
    
    # GTJA 022: 价格偏离均值 (简化版)
    df['_mean6'] = g['close'].transform(lambda x: x.rolling(6, min_periods=1).mean())
    df['_dev'] = safe_divide(df['close'] - df['_mean6'], df['_mean6'] + 0.0001)
    df['_dev_lag3'] = df.groupby('ts_code')['_dev'].shift(3)
    df['_dev_diff'] = df['_dev'] - df['_dev_lag3']
    df['gtja_022'] = df.groupby('ts_code')['_dev_diff'].transform(lambda x: x.rolling(12, min_periods=1).mean())
    df = df.drop(columns=['_mean6', '_dev', '_dev_lag3', '_dev_diff'], errors='ignore')
    
    # GTJA 024: 短期价格变化
    df['_price_diff5'] = df['close'] - g['close'].shift(5)
    df['gtja_024'] = df.groupby('ts_code')['_price_diff5'].transform(lambda x: x.rolling(5, min_periods=1).mean())
    df = df.drop(columns=['_price_diff5'], errors='ignore')
    
    # GTJA 034: 均值回归
    df['_ma12'] = g['close'].transform(lambda x: x.rolling(12, min_periods=1).mean())
    df['gtja_034'] = safe_divide(df['_ma12'], df['close'] + 0.0001)
    df = df.drop(columns=['_ma12'], errors='ignore')
    
    # GTJA 043: 资金流向累计
    df['_direction'] = np.sign(df['close'] - g['close'].shift(1))
    df['_dir_vol'] = df['_direction'] * df['vol']
    df['gtja_043'] = df.groupby('ts_code')['_dir_vol'].transform(lambda x: x.rolling(6, min_periods=1).sum())
    df = df.drop(columns=['_direction', '_dir_vol'], errors='ignore')
    
    # GTJA 048: 多日动量趋势
    sign1 = np.sign(df['close'] - g['close'].shift(1))
    sign2 = np.sign(g['close'].shift(1) - g['close'].shift(2))
    sign3 = np.sign(g['close'].shift(2) - g['close'].shift(3))
    df['gtja_048'] = -1 * (sign1 + sign2 + sign3).rank(pct=True)
    
    # GTJA 053: 收盘价相对位置
    df['_high12'] = g['high'].transform(lambda x: x.rolling(12, min_periods=1).max())
    df['_low12'] = g['low'].transform(lambda x: x.rolling(12, min_periods=1).min())
    df['gtja_053'] = safe_divide(df['close'] - df['_low12'], df['_high12'] - df['_low12'] + 0.0001)
    df = df.drop(columns=['_high12', '_low12'], errors='ignore')
    
    # GTJA 056: 量价背离 (简化版)
    df['_ret'] = g['close'].pct_change()
    df['_vol_change'] = g['vol'].pct_change()
    # 简化: 滚动窗口相关系数
    df['gtja_056'] = -1 * df.groupby('ts_code').apply(
        lambda x: x['_ret'].rolling(10, min_periods=5).corr(x['_vol_change'])
    ).reset_index(level=0, drop=True)
    df = df.drop(columns=['_ret', '_vol_change'], errors='ignore')
    
    # GTJA 083: 高低价相对强度
    df['_ma5'] = g['close'].transform(lambda x: x.rolling(5, min_periods=1).mean())
    df['_hl_ratio'] = safe_divide(df['high'] - df['low'], df['_ma5'] + 0.0001)
    df['gtja_083'] = df['_hl_ratio'].rank(pct=True)
    df = df.drop(columns=['_ma5', '_hl_ratio'], errors='ignore')
    
    # GTJA 101: 价格动量排名
    df['_ret5'] = g['close'].pct_change(5)
    df['_ret10'] = g['close'].pct_change(10)
    df['gtja_101'] = (df['_ret5'].rank(pct=True) + df['_ret10'].rank(pct=True)) / 2
    df = df.drop(columns=['_ret5', '_ret10'], errors='ignore')
    
    print(f"  完成: 新增12个Alpha191因子")
    return df


# ==================== 3. 技术指标 ====================

def calculate_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    计算常用技术指标
    """
    print("计算技术指标...")
    df = df.copy()
    g = df.groupby('ts_code')
    
    # ===== 均线系列 =====
    for w in [5, 10, 20, 60]:
        df[f'ma_{w}'] = g['close'].transform(lambda x: x.rolling(w, min_periods=1).mean())
        df[f'ema_{w}'] = g['close'].transform(lambda x: x.ewm(span=w, adjust=False).mean())
    
    # 价格相对均线位置
    for w in [5, 20, 60]:
        df[f'price_ma_ratio_{w}'] = safe_divide(df['close'], df[f'ma_{w}'] + 0.0001)
    
    # 均线交叉信号
    df['ma_cross_5_20'] = np.sign(df['ma_5'] - df['ma_20'])
    df['ma_cross_5_60'] = np.sign(df['ma_5'] - df['ma_60'])
    
    # ===== MACD =====
    ema12 = g['close'].transform(lambda x: x.ewm(span=12, adjust=False).mean())
    ema26 = g['close'].transform(lambda x: x.ewm(span=26, adjust=False).mean())
    df['macd_dif'] = ema12 - ema26
    df['macd_dea'] = g['macd_dif'].transform(lambda x: x.ewm(span=9, adjust=False).mean())
    df['macd_hist'] = 2 * (df['macd_dif'] - df['macd_dea'])
    df['macd_cross'] = np.sign(df['macd_dif'] - df['macd_dea'])
    
    # ===== RSI =====
    df['_delta'] = g['close'].diff()
    df['_gain'] = df['_delta'].where(df['_delta'] > 0, 0)
    df['_loss'] = -df['_delta'].where(df['_delta'] < 0, 0)
    for period in [6, 14]:
        df[f'_avg_gain_{period}'] = df.groupby('ts_code')['_gain'].transform(lambda x: x.rolling(period, min_periods=1).mean())
        df[f'_avg_loss_{period}'] = df.groupby('ts_code')['_loss'].transform(lambda x: x.rolling(period, min_periods=1).mean())
        rs = safe_divide(df[f'_avg_gain_{period}'], df[f'_avg_loss_{period}'] + 0.0001)
        df[f'rsi_{period}'] = 100 - (100 / (1 + rs))
        df = df.drop(columns=[f'_avg_gain_{period}', f'_avg_loss_{period}'], errors='ignore')
    df = df.drop(columns=['_delta', '_gain', '_loss'], errors='ignore')
    
    df['rsi_signal'] = np.where(df['rsi_14'] < 30, 1, np.where(df['rsi_14'] > 70, -1, 0))
    
    # ===== KDJ =====
    df['_low_min'] = g['low'].transform(lambda x: x.rolling(9, min_periods=1).min())
    df['_high_max'] = g['high'].transform(lambda x: x.rolling(9, min_periods=1).max())
    df['_rsv'] = safe_divide(df['close'] - df['_low_min'], df['_high_max'] - df['_low_min'] + 0.0001) * 100
    df['kdj_k'] = df.groupby('ts_code')['_rsv'].transform(lambda x: x.ewm(com=2, adjust=False).mean())
    df['kdj_d'] = df.groupby('ts_code')['kdj_k'].transform(lambda x: x.ewm(com=2, adjust=False).mean())
    df['kdj_j'] = 3 * df['kdj_k'] - 2 * df['kdj_d']
    df['kdj_cross'] = np.sign(df['kdj_k'] - df['kdj_d'])
    df['kdj_signal'] = np.where(df['kdj_k'] < 20, 1, np.where(df['kdj_k'] > 80, -1, 0))
    df = df.drop(columns=['_low_min', '_high_max', '_rsv'], errors='ignore')
    
    # ===== 布林带 =====
    df['boll_mid'] = g['close'].transform(lambda x: x.rolling(20, min_periods=1).mean())
    boll_std = g['close'].transform(lambda x: x.rolling(20, min_periods=1).std())
    df['boll_upper'] = df['boll_mid'] + 2 * boll_std
    df['boll_lower'] = df['boll_mid'] - 2 * boll_std
    df['boll_width'] = safe_divide(df['boll_upper'] - df['boll_lower'], df['boll_mid'] + 0.0001)
    df['boll_pctb'] = safe_divide(df['close'] - df['boll_lower'], df['boll_upper'] - df['boll_lower'] + 0.0001)
    
    # ===== ATR =====
    df['_hl'] = df['high'] - df['low']
    df['_hc'] = abs(df['high'] - g['close'].shift(1))
    df['_lc'] = abs(df['low'] - g['close'].shift(1))
    df['_tr'] = df[['_hl', '_hc', '_lc']].max(axis=1)
    df['atr_14'] = df.groupby('ts_code')['_tr'].transform(lambda x: x.rolling(14, min_periods=1).mean())
    df = df.drop(columns=['_hl', '_hc', '_lc', '_tr'], errors='ignore')
    
    # ===== CCI =====
    df['_tp'] = (df['high'] + df['low'] + df['close']) / 3
    df['_tp_ma'] = df.groupby('ts_code')['_tp'].transform(lambda x: x.rolling(14, min_periods=1).mean())
    df['_tp_md'] = df.groupby('ts_code')['_tp'].transform(lambda x: x.rolling(14, min_periods=1).apply(lambda a: np.abs(a - a.mean()).mean()))
    df['cci_14'] = safe_divide(df['_tp'] - df['_tp_ma'], 0.015 * df['_tp_md'] + 0.0001)
    df = df.drop(columns=['_tp', '_tp_ma', '_tp_md'], errors='ignore')
    
    # ===== 威廉指标 =====
    df['_wh'] = g['high'].transform(lambda x: x.rolling(14, min_periods=1).max())
    df['_wl'] = g['low'].transform(lambda x: x.rolling(14, min_periods=1).min())
    df['williams_r'] = -100 * safe_divide(df['_wh'] - df['close'], df['_wh'] - df['_wl'] + 0.0001)
    df = df.drop(columns=['_wh', '_wl'], errors='ignore')
    
    # ===== ROC =====
    for period in [5, 10]:
        df[f'roc_{period}'] = g['close'].pct_change(period) * 100
    
    print(f"  完成: 新增25+个技术指标")
    return df


# ==================== 4. 滞后特征 ====================

def calculate_lagged_features(df: pd.DataFrame) -> pd.DataFrame:
    """计算滞后特征"""
    print("计算滞后特征...")
    df = df.copy()
    g = df.groupby('ts_code')
    
    for lag in [1, 3, 5, 10]:
        df[f'lag_close_{lag}'] = g['close'].shift(lag)
        df[f'lag_return_{lag}'] = g['close'].pct_change().shift(lag)
    
    for lag in [1, 3, 5]:
        df[f'lag_volume_{lag}'] = g['vol'].shift(lag)
    
    if 'turnover_rate' in df.columns:
        for lag in [1, 3]:
            df[f'lag_turnover_{lag}'] = g['turnover_rate'].shift(lag)
    
    if 'rsi_14' in df.columns:
        for lag in [1, 3]:
            df[f'lag_rsi_{lag}'] = g['rsi_14'].shift(lag)
    
    if 'macd_hist' in df.columns:
        for lag in [1, 3]:
            df[f'lag_macd_{lag}'] = g['macd_hist'].shift(lag)
    
    print(f"  完成: 新增15+个滞后特征")
    return df


# ==================== 5. 滚动统计量 ====================

def calculate_rolling_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """计算滚动窗口统计量"""
    print("计算滚动统计量...")
    df = df.copy()
    g = df.groupby('ts_code')
    
    if 'returns' not in df.columns:
        df['returns'] = g['close'].pct_change()
    
    for w in [5, 10, 20]:
        df[f'return_mean_{w}'] = g['returns'].transform(lambda x: x.rolling(w, min_periods=1).mean())
        df[f'return_std_{w}'] = g['returns'].transform(lambda x: x.rolling(w, min_periods=1).std())
    
    df['return_skew_20'] = g['returns'].transform(lambda x: x.rolling(20, min_periods=5).skew())
    df['return_kurt_20'] = g['returns'].transform(lambda x: x.rolling(20, min_periods=5).kurt())
    
    for w in [5, 10, 20]:
        df[f'high_max_{w}'] = g['high'].transform(lambda x: x.rolling(w, min_periods=1).max())
        df[f'low_min_{w}'] = g['low'].transform(lambda x: x.rolling(w, min_periods=1).min())
    
    df['close_rank_20'] = safe_divide(df['close'] - df['low_min_20'], df['high_max_20'] - df['low_min_20'] + 0.0001)
    
    for w in [5, 10, 20]:
        df[f'volume_mean_{w}'] = g['vol'].transform(lambda x: x.rolling(w, min_periods=1).mean())
    
    df['volume_std_10'] = g['vol'].transform(lambda x: x.rolling(10, min_periods=1).std())
    df['volume_ratio'] = safe_divide(df['vol'], df['volume_mean_5'] + 1)
    
    print(f"  完成: 新增15+个滚动统计量")
    return df


# ==================== 6. 截面排名特征 ====================

def calculate_cross_sectional_rank(df: pd.DataFrame) -> pd.DataFrame:
    """计算截面排名特征"""
    print("计算截面排名特征...")
    df = df.copy()
    
    rank_cols = []
    
    if 'returns' in df.columns:
        df['rank_return'] = df.groupby('trade_date')['returns'].rank(pct=True)
        rank_cols.append('rank_return')
    
    if 'volume_ratio' in df.columns:
        df['rank_volume_ratio'] = df.groupby('trade_date')['volume_ratio'].rank(pct=True)
        rank_cols.append('rank_volume_ratio')
    
    if 'rsi_14' in df.columns:
        df['rank_rsi'] = df.groupby('trade_date')['rsi_14'].rank(pct=True)
        rank_cols.append('rank_rsi')
    
    if 'return_std_20' in df.columns:
        df['rank_volatility'] = df.groupby('trade_date')['return_std_20'].rank(pct=True)
        rank_cols.append('rank_volatility')
    
    if 'turnover_rate' in df.columns:
        df['rank_turnover'] = df.groupby('trade_date')['turnover_rate'].rank(pct=True)
        rank_cols.append('rank_turnover')
    
    if 'macd_hist' in df.columns:
        df['rank_macd'] = df.groupby('trade_date')['macd_hist'].rank(pct=True)
        rank_cols.append('rank_macd')
    
    print(f"  完成: 新增{len(rank_cols)}个截面排名特征")
    return df


# ==================== 7 & 8. 资金流向和筹码分布 ====================

def integrate_money_flow(df: pd.DataFrame, money_flow_df: pd.DataFrame) -> pd.DataFrame:
    """
    整合资金流向数据
    
    资金流向列:
    - buy/sell_sm_vol/amount: 小单买卖
    - buy/sell_md_vol/amount: 中单买卖
    - buy/sell_lg_vol/amount: 大单买卖
    - buy/sell_elg_vol/amount: 特大单买卖
    - net_mf_vol/amount: 净流入量/额
    """
    print("整合资金流向数据...")
    
    if money_flow_df is None or money_flow_df.empty:
        print("  警告: 无资金流向数据")
        return df
    
    mf = money_flow_df.copy()
    if 'trade_date' in mf.columns:
        mf['trade_date'] = pd.to_datetime(mf['trade_date'])
    
    # 选择所有资金流向相关列
    mf_cols = ['ts_code', 'trade_date']
    available_cols = [
        'net_mf_amount', 'net_mf_vol',
        'buy_lg_amount', 'sell_lg_amount', 'buy_lg_vol', 'sell_lg_vol',
        'buy_elg_amount', 'sell_elg_amount', 'buy_elg_vol', 'sell_elg_vol',
        'buy_sm_amount', 'sell_sm_amount', 'buy_md_amount', 'sell_md_amount'
    ]
    for col in available_cols:
        if col in mf.columns:
            mf_cols.append(col)
    
    df = pd.merge(df, mf[mf_cols], on=['ts_code', 'trade_date'], how='left')
    
    g = df.groupby('ts_code')
    
    # 净流入相关
    if 'net_mf_amount' in df.columns:
        # 净流入占成交额比例
        if 'amount' in df.columns:
            df['net_mf_ratio'] = safe_divide(df['net_mf_amount'], df['amount'] + 1)
        # 5日累计净流入
        df['mf_5d_sum'] = g['net_mf_amount'].transform(lambda x: x.rolling(5, min_periods=1).sum())
        # 10日累计净流入
        df['mf_10d_sum'] = g['net_mf_amount'].transform(lambda x: x.rolling(10, min_periods=1).sum())
    
    # 大单净买入 (大单+超大单)
    if 'buy_lg_amount' in df.columns and 'sell_lg_amount' in df.columns:
        df['lg_net_amount'] = df['buy_lg_amount'] - df['sell_lg_amount']
        if 'buy_elg_amount' in df.columns and 'sell_elg_amount' in df.columns:
            df['lg_net_amount'] = df['lg_net_amount'] + df['buy_elg_amount'] - df['sell_elg_amount']
        # 大单净买入5日累计
        df['lg_net_5d'] = g['lg_net_amount'].transform(lambda x: x.rolling(5, min_periods=1).sum())
    
    # 大单占比
    if 'buy_lg_amount' in df.columns and 'buy_sm_amount' in df.columns and 'buy_md_amount' in df.columns:
        total_buy = df['buy_sm_amount'] + df['buy_md_amount'] + df['buy_lg_amount']
        if 'buy_elg_amount' in df.columns:
            total_buy = total_buy + df['buy_elg_amount']
        df['lg_buy_ratio'] = safe_divide(df['buy_lg_amount'] + df.get('buy_elg_amount', 0), total_buy + 1)
    
    # 主力资金流向信号
    if 'lg_net_amount' in df.columns:
        df['mf_signal'] = np.sign(df['lg_net_amount'])
    
    print(f"  完成: 整合资金流向特征")
    return df


def integrate_chip_data(df: pd.DataFrame, chip_df: pd.DataFrame) -> pd.DataFrame:
    """
    整合筹码分布数据
    
    筹码列:
    - winner_rate: 获利比例
    - cost_5pct/15pct/50pct/85pct/95pct: 成本分布百分位
    - weight_avg: 加权平均成本
    - his_low/his_high: 历史最低/最高成本
    """
    print("整合筹码分布数据...")
    
    if chip_df is None or chip_df.empty:
        print("  警告: 无筹码分布数据")
        return df
    
    chip = chip_df.copy()
    if 'trade_date' in chip.columns:
        chip['trade_date'] = pd.to_datetime(chip['trade_date'])
    
    # 选择所有筹码相关列
    chip_cols = ['ts_code', 'trade_date']
    available_cols = ['winner_rate', 'cost_5pct', 'cost_15pct', 'cost_50pct', 
                      'cost_85pct', 'cost_95pct', 'weight_avg', 'his_low', 'his_high']
    for col in available_cols:
        if col in chip.columns:
            chip_cols.append(col)
    
    df = pd.merge(df, chip[chip_cols], on=['ts_code', 'trade_date'], how='left')
    
    g = df.groupby('ts_code')
    
    # 90%筹码成本区间宽度
    if 'cost_85pct' in df.columns and 'cost_15pct' in df.columns:
        df['cost_90_range'] = df['cost_85pct'] - df['cost_15pct']
        # 筹码集中度 (区间越小越集中)
        df['chip_concentration'] = safe_divide(1, df['cost_90_range'] + 0.01)
    
    # 价格相对平均成本位置
    if 'weight_avg' in df.columns:
        df['price_vs_avg_cost'] = safe_divide(df['close'], df['weight_avg'] + 0.0001) - 1
        # 价格高于均价信号
        df['above_avg_cost'] = np.where(df['close'] > df['weight_avg'], 1, 0)
    
    # 获利比例变化 (5日)
    if 'winner_rate' in df.columns:
        df['winner_rate_5d_change'] = g['winner_rate'].diff(5)
        # 获利盘压力 (>70%时压力大)
        df['profit_pressure'] = np.where(df['winner_rate'] > 70, 1, 
                                         np.where(df['winner_rate'] < 30, -1, 0))
    
    # 筹码峰值位置 (50%成本线)
    if 'cost_50pct' in df.columns:
        df['price_vs_median_cost'] = safe_divide(df['close'], df['cost_50pct'] + 0.0001) - 1
    
    print(f"  完成: 整合筹码特征")
    return df


# ==================== 主函数 ====================

def calculate_all_enhanced_features(
    df: pd.DataFrame,
    money_flow_df: Optional[pd.DataFrame] = None,
    chip_df: Optional[pd.DataFrame] = None
) -> pd.DataFrame:
    """
    计算所有增强特征
    
    Args:
        df: 基础行情数据 (包含 ts_code, trade_date, open, high, low, close, vol)
        money_flow_df: 资金流向数据 (可选)
        chip_df: 筹码分布数据 (可选)
    
    Returns:
        添加了所有特征的 DataFrame
    """
    print("\n" + "="*60)
    print("开始计算增强特征 (8大类70+特征)")
    print("="*60 + "\n")
    
    # 确保数据排序
    df = df.sort_values(['ts_code', 'trade_date']).reset_index(drop=True)
    
    # 确保trade_date是datetime
    if not pd.api.types.is_datetime64_any_dtype(df['trade_date']):
        df['trade_date'] = pd.to_datetime(df['trade_date'])
    
    # 1. Alpha101扩展
    df = calculate_alpha101_extended(df)
    
    # 2. Alpha191因子
    df = calculate_alpha191_factors(df)
    
    # 3. 技术指标
    df = calculate_technical_indicators(df)
    
    # 4. 滞后特征
    df = calculate_lagged_features(df)
    
    # 5. 滚动统计量
    df = calculate_rolling_statistics(df)
    
    # 6. 截面排名
    df = calculate_cross_sectional_rank(df)
    
    # 7. 资金流向
    if money_flow_df is not None:
        df = integrate_money_flow(df, money_flow_df)
    
    # 8. 筹码分布
    if chip_df is not None:
        df = integrate_chip_data(df, chip_df)
    
    # 9. Vibe-Trading Alpha Zoo
    if calculate_vibe_alphas is not None:
        df = calculate_vibe_alphas(df, num_factors=40)
    
    # 填充NaN
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(0)
    
    # 处理无穷值
    df = df.replace([np.inf, -np.inf], 0)
    
    print("\n" + "="*60)
    print(f"增强特征计算完成! 总列数: {len(df.columns)}")
    print("="*60 + "\n")
    
    return df

"""
step3_train_ranking_model.py
基于 features_longterm.parquet，进行月度滚动的 Purged Walk-Forward 训练。
支持的回归模型：Linear (Ridge Regression) 和 XGBoost。
清除重叠标签（Purging）以防止未来数据泄露。
输出 -> predictions/predictions_longterm.parquet
"""
import os
import sys
import time
import gc
import warnings
import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

warnings.filterwarnings('ignore')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(PROJECT_DIR, 'data')
PRED_DIR = os.path.join(PROJECT_DIR, 'predictions')
os.makedirs(PRED_DIR, exist_ok=True)

FEATURES_FILE = os.path.join(DATA_DIR, 'features_longterm.parquet')
OUTPUT_FILE = os.path.join(PRED_DIR, 'predictions_longterm.parquet')

# 配置参数
TARGET_COL = 'mkt_excess_ret_20d'  # 20日市场超额收益为目标
HOLDING_DAYS = 20                  # 持有期为20天，清除训练期最后20天数据
MODEL_TYPE = 'linear'              # 'linear' (Ridge) 或 'xgb'
RIDGE_ALPHA = 100.0                # 线性模型正则化强度

def get_feature_cols(df, exclude_config=None):
    exclude_cols = {
        'ts_code', 'trade_date', 'ds', 'industry',
        'open', 'high', 'low', 'close', 'pre_close',
        'change', 'pct_chg', 'vol', 'amount', 'amplitude',
        'entry_price', 'next_open',
        'ths_hot', 'ths_hot_rank',
        'exit_price_1d', 'return_1d', 'return_1d_open',
        'exit_price_5d', 'return_5d', 'return_5d_open',
        'exit_price_28d', 'return_28d', 'return_28d_open',
        'exit_28d_close',
        'calc_ret5d', 'return_5d_from_open', 'return_28d_from_open',
        'entry_vs_close',
        'return_1d_open_old', 'actual_return',
        't1_intraday_return', 'target_crash_bin', 'target_up_bin',
        'index_ma20_bias',
        # Targets & helper variables
        'close_T5', 'close_T10', 'close_T20',
        'ret_5d', 'ret_10d', 'ret_20d',
        'mkt_excess_ret_5d', 'mkt_excess_ret_10d', 'mkt_excess_ret_20d',
        'ind_excess_ret_5d', 'ind_excess_ret_10d', 'ind_excess_ret_20d'
    }
    if exclude_config is not None:
        exclude_cols.update(exclude_config)
    return [c for c in df.columns
            if c not in exclude_cols
            and not c.startswith('hist_')
            and df[c].dtype in ('float64', 'float32', 'int64', 'int32')]

def preprocess_factors(df, feature_cols):
    """
    Perform cross-sectional factor preprocessing:
    1. Winsorize (1% - 99%)
    2. Industry Neutralization (subtract industry mean)
    3. Standardization (Z-score)
    """
    print("Preprocessing factors cross-sectionally (Winsorize, Neutralize, Standardize)...", flush=True)
    
    # 1. Winsorize 1% - 99%
    print("  Applying Winsorization (1%-99%)...", flush=True)
    q01 = df.groupby('trade_date')[feature_cols].quantile(0.01)
    q99 = df.groupby('trade_date')[feature_cols].quantile(0.99)
    df_trade_date = df['trade_date']
    q01_aligned = q01.loc[df_trade_date].values
    q99_aligned = q99.loc[df_trade_date].values
    df[feature_cols] = np.clip(df[feature_cols].values, q01_aligned, q99_aligned)
    
    # 2. Industry Neutralization
    if 'industry' in df.columns:
        print("  Applying Industry Neutralization...", flush=True)
        ind_means = df.groupby(['trade_date', 'industry'])[feature_cols].transform('mean')
        df[feature_cols] = df[feature_cols] - ind_means
    
    # 3. Standardization (Z-score)
    print("  Applying Standardization (Z-score)...", flush=True)
    date_means = df.groupby('trade_date')[feature_cols].transform('mean')
    date_stds = df.groupby('trade_date')[feature_cols].transform('std')
    date_stds = date_stds.replace(0, 1).fillna(1)
    df[feature_cols] = (df[feature_cols] - date_means) / date_stds
    
    # Fill any remaining NaNs safely
    df[feature_cols] = df[feature_cols].fillna(0)
    print("Preprocessing complete.", flush=True)
    return df


def train_and_predict(feature_config='B', output_file=OUTPUT_FILE, start_month='202201'):
    t0 = time.time()
    print(f"Starting train_and_predict with Config: {feature_config}...", flush=True)
    df = pd.read_parquet(FEATURES_FILE)
    df['ds'] = df['trade_date'].astype(str)
    
    # 转换 ths_hot_rank 为 ths_hot_score (101 - rank，缺失值为0)
    # 彻底废除 9999 的极端值填充，采用 0.0 作为中性/无热度表达
    if 'ths_hot_rank' in df.columns:
        print("Converting ths_hot_rank to ths_hot_score (range 1-100 mapped to 100-1, NaNs/others mapped to 0.0)...", flush=True)
        df['ths_hot_score'] = np.where(
            (df['ths_hot_rank'].notna()) & (df['ths_hot_rank'] <= 100.0),
            101.0 - df['ths_hot_rank'],
            0.0
        )
    
    # 根据配置过滤特征列
    extra_excludes = set()
    vibe_cols = [c for c in df.columns if c.startswith('alpha101_') or c.startswith('gtja191_')]
    
    if feature_config == 'A':
        # Baseline: 排除所有新引入的 Vibe 因子
        extra_excludes.update(vibe_cols)
    elif feature_config == 'B':
        # Baseline + Vibe: 包含 Vibe 因子
        pass
    else:
        raise ValueError(f"Unknown feature_config: {feature_config}")
        
    feature_cols = get_feature_cols(df, exclude_config=extra_excludes)
    print(f"Total rows: {len(df)}")
    print(f"Number of feature columns: {len(feature_cols)}", flush=True)
    
    # 统一横截面因子预处理
    df = preprocess_factors(df, feature_cols)
    
    # 划分预测月度
    months = sorted(df['ds'].str[:6].unique())
    pred_months = [m for m in months if m >= start_month]
    print(f"Prediction period: {pred_months[0]} to {pred_months[-1]} ({len(pred_months)} months)", flush=True)
    
    # 获取所有的交易日期（用于Purging定位）
    trade_dates = sorted(df['trade_date'].unique())
    date_to_idx = {d: i for i, d in enumerate(trade_dates)}
    
    all_preds = []
    
    for month in pred_months:
        mt0 = time.time()
        
        # 1. 确定训练结束的月份
        train_end_month = str(int(month) - 1)
        if train_end_month.endswith('00'):
            train_end_month = f"{int(train_end_month[:4])-1}12"
            
        # 2. 12个月滚动窗口的训练起始月份
        year = int(train_end_month[:4])
        month_val = int(train_end_month[4:6])
        start_year = year - 1
        start_month = month_val + 1
        if start_month > 12:
            start_month -= 12
            start_year += 1
        rolling_start_month = f"{start_year}{start_month:02d}"
        
        # 3. 筛选训练集日期 (在 Purging 之前)
        train_dates_raw = [d for d in trade_dates if d[:6] >= rolling_start_month and d[:6] <= train_end_month]
        if len(train_dates_raw) < 20:
            print(f"⚠️ Month {month}: Insufficient training dates, skipping.", flush=True)
            continue
            
        # 4. 执行 Purging：从训练集末尾扣除 HOLDING_DAYS 天，以防止重叠持有期的信息泄露
        last_train_dt_raw = train_dates_raw[-1]
        last_idx_raw = date_to_idx[last_train_dt_raw]
        purged_last_idx = last_idx_raw - HOLDING_DAYS
        
        if purged_last_idx < date_to_idx[train_dates_raw[0]]:
            print(f"⚠️ Month {month}: Purging left no training dates, skipping.", flush=True)
            continue
            
        purged_last_dt = trade_dates[purged_last_idx]
        train_dates_purged = [d for d in train_dates_raw if d <= purged_last_dt]
        
        # 5. 确定测试集日期
        test_dates = [d for d in trade_dates if d[:6] == month]
        
        # 6. 提取训练集与测试集数据
        train_mask = df['trade_date'].isin(train_dates_purged) & df[TARGET_COL].notna()
        test_mask = df['trade_date'].isin(test_dates)
        
        train_df = df.loc[train_mask, feature_cols + [TARGET_COL]].copy()
        test_df = df.loc[test_mask, feature_cols + ['trade_date', 'ts_code', 'next_open', 'close', 'pct_chg', 'industry', 'ret_20d', 'mkt_excess_ret_20d']].copy()
        
        if len(train_df) < 5000 or len(test_df) == 0:
            print(f"⚠️ Month {month}: Skipping due to small size. Train rows: {len(train_df)}, Test rows: {len(test_df)}", flush=True)
            continue
            
        # 7. 特征与目标提取
        X_train = train_df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
        y_train = train_df[TARGET_COL]
        X_test = test_df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
        
        # 8. 训练模型与预测
        if MODEL_TYPE == 'linear':
            # 注意：数据已经过横截面标准化，无需再进行全量 StandardScaler
            model = Ridge(alpha=RIDGE_ALPHA)
            model.fit(X_train, y_train)
            pred_scores = model.predict(X_test)
        elif MODEL_TYPE == 'xgb':
            model = XGBRegressor(
                n_estimators=100,
                max_depth=3,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=-1
            )
            model.fit(X_train, y_train)
            pred_scores = model.predict(X_test)
        else:
            raise ValueError(f"Unknown model type: {MODEL_TYPE}")
            
        test_df['pred_score'] = pred_scores
        
        valid_test = test_df[test_df[TARGET_COL].notna()]
        if len(valid_test) > 0:
            month_ic = valid_test['pred_score'].corr(valid_test[TARGET_COL], method='spearman')
        else:
            month_ic = np.nan
            
        print(f"Month {month} | Train rows: {len(train_df)} | Test rows: {len(test_df)} | Out-of-Sample Rank IC: {month_ic:+.4f} | Time: {time.time()-mt0:.1f}s", flush=True)
        
        all_preds.append(test_df[['trade_date', 'ts_code', 'next_open', 'close', 'pct_chg', 'industry', 'ret_20d', 'mkt_excess_ret_20d', 'pred_score']])
        
        del train_df, test_df, X_train, y_train, X_test
        gc.collect()
        
    print("Concatenating all walk-forward predictions...")
    pred_df = pd.concat(all_preds, ignore_index=True)
    
    overall_ic = pred_df.groupby('trade_date').apply(
        lambda x: x['pred_score'].corr(x['mkt_excess_ret_20d'], method='spearman')
    ).mean()
    print(f"\nOverall Out-of-Sample Daily Rank IC: {overall_ic:.4f}")
    
    print(f"Saving predictions to {output_file}...")
    pred_df.to_parquet(output_file, index=False)
    print(f"Model training and prediction complete! Total elapsed time: {time.time()-t0:.1f}s")

if __name__ == '__main__':
    train_and_predict()

"""
Walk-Forward 月级别回测 (每月重训练)

方案:
  - 2022-01预测: 用 <=2021-12 数据训练
  - 2022-02预测: 用 <=2022-01 数据训练
  - ...
  - 2026-03预测: 用 <=2026-02 数据训练

每月独立训练XGBoost，预测完全样本外
预计耗时: 5-6小时 (51个月 x 训练+预测)

运行命令:
  cd c:\Users\liuqi\quant_system_v2\research\studies\study_004_systematic
  python -u run_wf_monthly.py 2>&1 | tee wf_monthly_log.txt
"""
import os
import sys
import pandas as pd
import numpy as np
import json
import time
import warnings
warnings.filterwarnings('ignore')

STUDY_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(STUDY_DIR, 'data')
RESULTS_DIR = os.path.join(STUDY_DIR, 'results')
PREDICTIONS_DIR = os.path.join(STUDY_DIR, 'predictions')

FEATURES_FILE = os.path.join(DATA_DIR, 'all_features_v2.parquet')
OUTPUT_FILE = os.path.join(PREDICTIONS_DIR, 'predictions_1d_wf_monthly.parquet')

TRAIN_START = '20200101'
TARGET_THRESHOLD = 0.015
MIN_TRAIN_SAMPLES = 50000

THRESHOLD = 0.58
MAX_POSITIONS = 3
TRANSACTION_COST = 0.003


def log(msg):
    print(msg, flush=True)


def run():
    log("=" * 90)
    log("Walk-Forward 月级别回测 (每月重训练)")
    log("=" * 90)

    if not os.path.exists(FEATURES_FILE):
        log("ERROR: 请先运行 run_full.py 生成特征数据")
        return

    log("加载特征数据...")
    features_df = pd.read_parquet(FEATURES_FILE)
    features_df['ds'] = features_df['trade_date'].astype(str)
    log(f"数据: {len(features_df)} 行, {len(features_df.columns)} 列")
    log(f"日期范围: {features_df['ds'].min()} - {features_df['ds'].max()}")

    return_col = 'return_1d'
    if return_col not in features_df.columns:
        log(f"ERROR: 缺少 {return_col} 列")
        return

    exclude_cols = ['ts_code', 'trade_date', 'ds', 'entry_price',
                    'exit_price_1d', 'return_1d',
                    'exit_price_5d', 'return_5d',
                    'exit_price_28d', 'return_28d']
    feature_cols = [c for c in features_df.columns if c not in exclude_cols and
                    not c.startswith('hist_') and
                    features_df[c].dtype in ['float64', 'float32', 'int64', 'int32']]
    log(f"可用特征: {len(feature_cols)}")

    months = sorted(features_df['ds'].str[:6].unique())
    pred_months = [m for m in months if m >= '202201']
    log(f"预测月数: {len(pred_months)} ({pred_months[0]} - {pred_months[-1]})")

    all_predictions = []
    total_start = time.time()

    from xgboost import XGBClassifier

    for i, month in enumerate(pred_months):
        month_start = time.time()

        train_end_month = str(int(month) - 1)
        if train_end_month.endswith('00'):
            year = int(train_end_month[:4]) - 1
            train_end_month = f"{year}12"

        train_mask = (features_df['ds'] >= TRAIN_START) & (features_df['ds'].str[:6] <= train_end_month)
        pred_mask = features_df['ds'].str[:6] == month

        train_df = features_df[train_mask & features_df[return_col].notna()].copy()
        pred_df = features_df[pred_mask].copy()

        if len(train_df) < MIN_TRAIN_SAMPLES:
            log(f"  [{i+1}/{len(pred_months)}] {month}: 训练数据不足 ({len(train_df)}), 跳过")
            continue
        if len(pred_df) == 0:
            log(f"  [{i+1}/{len(pred_months)}] {month}: 预测数据为空, 跳过")
            continue

        train_df['label'] = (train_df[return_col] > TARGET_THRESHOLD).astype(int)
        pos_rate = train_df['label'].mean()

        X_train = train_df[feature_cols].fillna(0)
        y_train = train_df['label']

        model = XGBClassifier(
            n_estimators=100, max_depth=5, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8,
            random_state=42, n_jobs=-1, eval_metric='logloss'
        )
        model.fit(X_train, y_train)

        X_pred = pred_df[feature_cols].fillna(0)
        proba = model.predict_proba(X_pred)[:, 1]

        month_pred = pred_df[['trade_date', 'ts_code']].copy()
        month_pred['prob'] = proba
        month_pred['target'] = '1d'

        if return_col in pred_df.columns:
            month_pred['actual_return'] = pred_df[return_col].values

        all_predictions.append(month_pred)

        elapsed = time.time() - month_start
        total_elapsed = time.time() - total_start
        avg_per_month = total_elapsed / (i + 1)
        remaining = avg_per_month * (len(pred_months) - i - 1)

        n_above = (proba >= THRESHOLD).sum()
        log(f"  [{i+1}/{len(pred_months)}] {month}: train={len(train_df)}, pred={len(pred_df)}, "
            f"pos={pos_rate:.1%}, prob>={THRESHOLD}={n_above}, "
            f"耗时={elapsed:.0f}s, 剩余~{remaining/60:.0f}min")

    if not all_predictions:
        log("ERROR: 未生成任何预测")
        return

    combined = pd.concat(all_predictions, ignore_index=True)
    combined.to_parquet(OUTPUT_FILE)
    total_time = (time.time() - total_start) / 60
    log(f"\n月级别WF预测已保存: {OUTPUT_FILE}")
    log(f"总预测: {len(combined)} 行")
    log(f"总耗时: {total_time:.1f} 分钟")

    combined['ds'] = combined['trade_date'].astype(str)
    combined = combined.dropna(subset=['actual_return']).copy()

    log(f"\n{'='*90}")
    log(f"回测: threshold={THRESHOLD}, max_positions={MAX_POSITIONS}")
    log(f"{'='*90}")

    results = {}
    for period_name, start, end in [('opt_2022_2025', '20220101', '20251231'),
                                     ('val_2026', '20260101', '20261231')]:
        mask = (combined['ds'] >= start) & (combined['ds'] <= end)
        pdf = combined[mask]
        if len(pdf) == 0:
            continue

        above = pdf[pdf['prob'] >= THRESHOLD].copy()
        above['rank'] = above.groupby('ds')['prob'].rank(ascending=False, method='first')
        selected = above[above['rank'] <= MAX_POSITIONS]
        if len(selected) == 0:
            log(f"\n  {period_name}: 无选股")
            continue

        pos_size = 1.0 / MAX_POSITIONS
        trading_dates = sorted(pdf['ds'].unique())

        daily_pnl = {}
        for d in trading_dates:
            day_trades = selected[selected['ds'] == d]
            if len(day_trades) == 0:
                daily_pnl[d] = 0.0
            else:
                daily_pnl[d] = pos_size * (day_trades['actual_return'].values - TRANSACTION_COST).sum()

        daily_s = pd.Series(daily_pnl)
        daily_s.index = pd.to_datetime(daily_s.index, format='%Y%m%d')

        equity = (1 + daily_s).cumprod()
        running_max = equity.cummax()
        drawdown = (equity - running_max) / running_max

        n_days = len(daily_s)
        n_years = n_days / 252
        total_return = equity.iloc[-1] - 1
        cagr = (equity.iloc[-1] ** (1 / n_years) - 1) if n_years > 0 else 0
        max_dd = drawdown.min()
        sharpe = (daily_s.mean() / daily_s.std() * np.sqrt(252)) if daily_s.std() > 1e-10 else 0
        win_rate_days = (daily_s > 0).mean()

        rets = selected['actual_return'].values - TRANSACTION_COST
        n_trades = len(rets)
        trade_win = (rets > 0).mean()
        avg_win = np.mean(rets[rets > 0]) if np.any(rets > 0) else 0
        avg_loss = np.mean(rets[rets <= 0]) if np.any(rets <= 0) else 0
        pct_big_loss = (rets < -0.05).mean()
        pct_big_win = (rets > 0.05).mean()

        log(f"\n  {period_name}:")
        log(f"    CAGR={cagr:.2%}, Sharpe={sharpe:.2f}, MaxDD={max_dd:.2%}")
        log(f"    总收益={total_return:.2%}, 日胜率={win_rate_days:.2%}")
        log(f"    交易数={n_trades}, 交易胜率={trade_win:.2%}")
        log(f"    平均盈利={avg_win:.2%}, 平均亏损={avg_loss:.2%}")
        log(f"    大亏损>5%={pct_big_loss:.1%}, 大盈利>5%={pct_big_win:.1%}")

        monthly_stats = []
        for period, group in daily_s.groupby(daily_s.index.to_period('M')):
            month_ret = (1 + group).prod() - 1
            month_win = (group > 0).mean()
            month_sel = selected[selected['ds'].str[:6] == str(period).replace('-', '')]
            monthly_stats.append({
                'month': str(period),
                'return': float(month_ret),
                'win_rate': float(month_win),
                'n_days': len(group),
                'n_trades': len(month_sel),
            })

        log(f"\n    月度明细:")
        log(f"    {'月份':<10} {'收益':>8} {'日胜率':>8} {'交易日':>6} {'交易数':>6}")
        log(f"    {'-'*42}")
        for ms in monthly_stats:
            marker = " ***" if ms['return'] < -0.05 else (" +++" if ms['return'] > 0.05 else "")
            log(f"    {ms['month']:<10} {ms['return']:>7.2%} {ms['win_rate']:>7.1%} {ms['n_days']:>6} {ms['n_trades']:>6}{marker}")

        mdf = pd.DataFrame(monthly_stats)
        pos_m = (mdf['return'] > 0).sum()
        log(f"    月胜率: {pos_m}/{len(mdf)} = {pos_m/len(mdf):.1%}")
        log(f"    月均收益: {mdf['return'].mean():.2%}")
        log(f"    最佳月: {mdf.loc[mdf['return'].idxmax(), 'month']} ({mdf['return'].max():.2%})")
        log(f"    最差月: {mdf.loc[mdf['return'].idxmin(), 'month']} ({mdf['return'].min():.2%})")

        results[period_name] = {
            'overall': {
                'cagr': float(cagr), 'sharpe': float(sharpe), 'max_drawdown': float(max_dd),
                'total_return': float(total_return), 'win_rate_days': float(win_rate_days),
                'n_trading_days': int(n_days), 'n_trades': int(n_trades),
                'trade_win_rate': float(trade_win), 'avg_win': float(avg_win),
                'avg_loss': float(avg_loss), 'pct_big_loss': float(pct_big_loss),
                'pct_big_win': float(pct_big_win),
            },
            'monthly': monthly_stats,
        }

    results_file = os.path.join(RESULTS_DIR, 'v2_wf_monthly_backtest_results.json')
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    log(f"\n结果已保存: {results_file}")

    log(f"\n{'='*90}")
    log("完成!")
    log(f"{'='*90}")


if __name__ == '__main__':
    run()

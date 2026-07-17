import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pandas as pd
import numpy as np
import joblib
from tqdm import tqdm

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PRICE_DIR = r'D:\iquant_data\data_v2\data_day1'
COST_RATE = 0.003

# 加载已有模型
model = joblib.load(os.path.join(THIS_DIR, 'best_model_v2.joblib'))

# 特征列表
BASE_FEATS = ['hot_rank_pct', 'chip_concentration', 'winner_rate',
              'news_market_impact', 'news_stock_impact']
CHAN_FEATS = ['chan_bi_count', 'chan_zhongshu_count', 'chan_zhongshu_width',
              'chan_macd_divergence', 'chan_bi_direction', 'chan_leave_zhongshu']
LYNCH_FEATS = ['lynch_peg', 'lynch_peg_rank', 'lynch_quality_score',
               'lynch_growth_value', 'lynch_roe_stability', 'lynch_earnings_momentum']
QUANT_FEATS = ['qf_mom_1d', 'qf_mom_3d', 'qf_mom_5d', 'qf_mom_10d', 'qf_mom_20d',
               'qf_reversal_1d', 'qf_reversal_3d', 'qf_realized_vol', 'qf_atr_pct',
               'qf_rsi_14', 'qf_bb_position', 'qf_ma_cross_5_10', 'qf_ma_cross_10_20',
               'qf_vol_ratio_5_20', 'qf_pv_corr']
ENHANCED_FEATS = BASE_FEATS + CHAN_FEATS + LYNCH_FEATS + QUANT_FEATS

# 从feature cache读取特征数据
feature_cache_dir = os.path.join(THIS_DIR, 'feature_cache')
all_dates = sorted([f.replace('.parquet', '').replace('feat_', '') for f in os.listdir(feature_cache_dir) if f.endswith('.parquet')])

print(f"Found {len(all_dates)} feature cache files")

# 使用模型训练时的特征顺序
feats = list(model.feature_names_in_) if hasattr(model, 'feature_names_in_') else ENHANCED_FEATS
print(f"Model expects {len(feats)} features")

# 回测参数
start_date = '20230801'
end_date = '20260331'
top_n = 1
prob_thresh = 0.4

date_idx = {d: i for i, d in enumerate(all_dates)}
start_idx = date_idx.get(start_date, 0)
end_idx = date_idx.get(end_date, len(all_dates) - 1)
test_dates = all_dates[start_idx:end_idx - 2]

print(f"Backtesting from {test_dates[0]} to {test_dates[-1]}, {len(test_dates)} days")

trades = []
equity = [100000.0]
skipped_limit_up = 0
skipped_limit_down = 0
skipped_cyb = 0
empty_files = 0
no_candidates = 0

for d in tqdm(test_dates, desc="Backtesting"):
    curr_idx = date_idx[d]
    if curr_idx + 2 >= len(all_dates):
        break

    d_t1 = all_dates[curr_idx + 1]
    d_t2 = all_dates[curr_idx + 2]

    # 读取特征
    feat_file = os.path.join(feature_cache_dir, f"feat_{d}.parquet")
    if not os.path.exists(feat_file):
        equity.append(equity[-1])
        continue

    features = pd.read_parquet(feat_file)
    if features is None or len(features) == 0:
        empty_files += 1
        equity.append(equity[-1])
        continue

    # 确保所有需要的特征都存在
    missing_feats = [f for f in feats if f not in features.columns]
    for f in missing_feats:
        features[f] = 0.0

    X = features[feats].fillna(0)
    features['prob'] = model.predict_proba(X)[:, 1]

    candidates = features[features['prob'] >= prob_thresh].sort_values('prob', ascending=False)

    if candidates.empty:
        no_candidates += 1
        equity.append(equity[-1])
        continue

    picks = candidates.head(top_n)

    # 读取价格数据
    p_t0 = os.path.join(PRICE_DIR, f"{d}.parquet")
    p_t1 = os.path.join(PRICE_DIR, f"{d_t1}.parquet")
    p_t2 = os.path.join(PRICE_DIR, f"{d_t2}.parquet")
    if not os.path.exists(p_t0) or not os.path.exists(p_t1) or not os.path.exists(p_t2):
        equity.append(equity[-1])
        continue

    try:
        price_t0 = pd.read_parquet(p_t0, columns=['ts_code', 'close'])
        price_t1 = pd.read_parquet(p_t1, columns=['ts_code', 'open', 'low', 'pre_close'])
        price_t2 = pd.read_parquet(p_t2, columns=['ts_code', 'close', 'low', 'pre_close'])
    except:
        equity.append(equity[-1])
        continue

    daily_ret = 0
    n_exec = 0
    for _, pick in picks.iterrows():
        ts_code = pick['ts_code']

        # 1. 过滤创业板 (300/301开头)
        if ts_code.startswith('300') or ts_code.startswith('301'):
            skipped_cyb += 1
            continue

        t0_row = price_t0[price_t0['ts_code'] == ts_code]
        t1_row = price_t1[price_t1['ts_code'] == ts_code]
        t2_row = price_t2[price_t2['ts_code'] == ts_code]

        if t0_row.empty or t1_row.empty or t2_row.empty:
            continue

        t0_close = float(t0_row['close'].values[0])
        t1_open = float(t1_row['open'].values[0])
        t1_low = float(t1_row['low'].values[0])
        t1_pre = float(t1_row['pre_close'].values[0]) if 'pre_close' in t1_row.columns else t0_close
        t2_close = float(t2_row['close'].values[0])
        t2_low = float(t2_row['low'].values[0])

        # 2. 涨跌停限制：主板10%
        limit_pct = 10.0

        # T+1开盘相对前收盘涨幅 > 9.5% -> 不能买入（涨停开盘买不到）
        t1_open_chg = (t1_open - t1_pre) / t1_pre * 100
        if t1_open_chg >= 9.5:
            skipped_limit_up += 1
            continue

        # 3. T+2跌停检查
        t2_low_chg_from_t1_open = (t2_low - t1_open) / t1_open * 100
        if t2_low_chg_from_t1_open <= -9.5:
            t2_open = float(t2_row['open'].values[0]) if 'open' in t2_row.columns else t2_close
            sell_p = min(t2_open, t1_open * 0.905)
            skipped_limit_down += 1
        else:
            sell_p = t2_close

        buy_p = t1_open
        ret = sell_p / buy_p - 1 - COST_RATE

        trades.append({
            'date': d,
            'ts_code': ts_code,
            'prob': pick['prob'],
            'buy_open': buy_p,
            'sell_close': sell_p,
            'ret': ret,
            't1_open_chg': t1_open_chg,
            't2_low_chg': t2_low_chg_from_t1_open,
        })
        daily_ret += ret
        n_exec += 1

    if n_exec > 0:
        daily_ret = daily_ret / n_exec
    new_equity = equity[-1] * (1 + daily_ret)
    equity.append(new_equity)

equity = equity[1:]

print(f"\nSkipped: limit_up={skipped_limit_up}, limit_down={skipped_limit_down}, cyb={skipped_cyb}, empty_files={empty_files}, no_candidates={no_candidates}")

if not trades:
    print("No trades!")
    sys.exit(1)

trades_df = pd.DataFrame(trades)
eq_arr = np.array(equity)
rets = np.diff(eq_arr) / eq_arr[:-1]

n_trades = len(trades_df)
win_rate = (trades_df['ret'] > 0).mean()
avg_ret = trades_df['ret'].mean()
total_ret = eq_arr[-1] / eq_arr[0] - 1
sharpe = np.mean(rets) / (np.std(rets) + 1e-8) * np.sqrt(252)
max_dd = np.max(1 - eq_arr / np.maximum.accumulate(eq_arr))

print("\n" + "=" * 60)
print("修正后 Enhanced_All 回测结果")
print("=" * 60)
print(f"  Trades: {n_trades}")
print(f"  Win Rate: {win_rate:.2%}")
print(f"  Avg Return: {avg_ret:.2%}")
print(f"  Total Return: {total_ret:.2%}")
print(f"  Sharpe: {sharpe:.2f}")
print(f"  Max Drawdown: {max_dd:.2%}")
print(f"  Final Equity: {eq_arr[-1]:,.0f}")

# 保存结果
trades_df.to_csv(os.path.join(THIS_DIR, "trades_Enhanced_All_Fixed.csv"), index=False)
pd.DataFrame({"equity": equity}).to_csv(os.path.join(THIS_DIR, "equity_Enhanced_All_Fixed.csv"), index=False)
print("\n结果已保存至 trades_Enhanced_All_Fixed.csv 和 equity_Enhanced_All_Fixed.csv")

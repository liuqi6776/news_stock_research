"""
市值敏感度分析：测试不同流通市值上限下策略的回测表现
假说：小市值股票对新闻舆论更敏感，T+1策略信噪比更高

市值单位：万元人民币（circ_mv from other_day1）
测试阈值：50亿、100亿、200亿、500亿、无上限（全体）
"""
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from infra_data.storage import DataStorage
from train_model import train_daily_model

DATA_DIR  = r'D:\iquant_data\data_v2'
PRICE_DIR = os.path.join(DATA_DIR, 'data_day1')
RANK_DIR  = os.path.join(DATA_DIR, 'ths_rank1')
CHIP_DIR  = os.path.join(DATA_DIR, 'cyq1')
OTHER_DIR = os.path.join(DATA_DIR, 'other_day1')   # contains circ_mv

# ---------- 配置 ----------
CAPS_TO_TEST = [
    ('50亿以下',    50_0000),      # 50亿 = 50 * 10000万
    ('100亿以下',  100_0000),
    ('200亿以下',  200_0000),
    ('500亿以下',  500_0000),
    ('无限制',      None),
]

START = '20240101'
END   = '20261231'
TRAIN_START = '20220101'
STEP_MONTHS = 1
INITIAL_CAP = 100_000.0


def run_cap_backtest(max_circ_mv, label):
    """跑一次完整的Expanding WFO回测，用max_circ_mv（万元）过滤标的"""
    dates = sorted([f.replace('.parquet', '') for f in os.listdir(PRICE_DIR) if f.endswith('.parquet')])

    current_test_start   = pd.to_datetime(START)
    final_end_date       = pd.to_datetime(END)
    fixed_train_start_dt = pd.to_datetime(TRAIN_START)

    storage = DataStorage()
    capital = INITIAL_CAP
    equity  = []

    while current_test_start <= final_end_date:
        current_test_end = current_test_start + pd.DateOffset(months=STEP_MONTHS) - pd.Timedelta(days=1)
        if current_test_end > final_end_date:
            current_test_end = final_end_date

        train_end_dt = current_test_start - pd.Timedelta(days=1)
        train_start_str  = fixed_train_start_dt.strftime('%Y%m%d')
        train_end_str    = train_end_dt.strftime('%Y%m%d')
        test_start_str   = current_test_start.strftime('%Y%m%d')
        test_end_str     = current_test_end.strftime('%Y%m%d')

        model, feats = train_daily_model(train_start_str, train_end_str, model_path=None)
        if model is None:
            current_test_start += pd.DateOffset(months=STEP_MONTHS)
            continue

        test_dates = [d for d in dates if test_start_str <= d <= test_end_str]
        if len(test_dates) < 2:
            current_test_start += pd.DateOffset(months=STEP_MONTHS)
            continue

        test_valid_series = pd.Series([pd.to_datetime(d) for d in test_dates]).sort_values()
        news_market_df, news_stock_sector_df = storage.load_news_data(test_start_str, test_end_str, test_valid_series)
        if not news_market_df.empty:
            news_market_df['trade_date'] = news_market_df['trade_date'].dt.strftime('%Y%m%d')
        if not news_stock_sector_df.empty:
            news_stock_sector_df['trade_date'] = news_stock_sector_df['trade_date'].dt.strftime('%Y%m%d')

        for i in range(len(test_dates) - 1):
            d_curr, d_next = test_dates[i], test_dates[i+1]

            p_rank  = os.path.join(RANK_DIR,  f"{d_curr}.parquet")
            p_chip  = os.path.join(CHIP_DIR,  f"{d_curr}.parquet")
            p_price = os.path.join(PRICE_DIR, f"{d_curr}.parquet")
            p_other = os.path.join(OTHER_DIR, f"{d_curr}.parquet")
            if not all(os.path.exists(p) for p in [p_rank, p_chip, p_price]):
                continue

            rank_df  = pd.read_parquet(p_rank)
            rank_df['hot_rank_pct'] = rank_df['hot'].rank(pct=True)
            chip_df  = pd.read_parquet(p_chip)
            chip_df['chip_concentration'] = (chip_df['cost_85pct'] - chip_df['cost_15pct']) / (chip_df['cost_50pct'] + 1e-8)
            price_df = pd.read_parquet(p_price, columns=['ts_code', 'close', 'pct_chg', 'amount', 'vol', 'pre_close'])

            df = pd.merge(rank_df[['ts_code', 'hot_rank_pct']], price_df, on='ts_code')
            df = pd.merge(df, chip_df[['ts_code', 'chip_concentration', 'winner_rate']], on='ts_code')
            df['trade_date'] = d_next

            # ---- 市值过滤 ----
            if max_circ_mv is not None and os.path.exists(p_other):
                other_df = pd.read_parquet(p_other, columns=['ts_code', 'circ_mv'])
                df = pd.merge(df, other_df, on='ts_code', how='left')
                df = df[df['circ_mv'].isna() | (df['circ_mv'] <= max_circ_mv)]

            # 排除科创板 688
            df = df[~df['ts_code'].str.startswith('688')]

            if not news_market_df.empty:
                df = pd.merge(df, news_market_df, on='trade_date', how='left')
            else:
                df['news_market_impact'] = 0.0
            if not news_stock_sector_df.empty:
                df = pd.merge(df, news_stock_sector_df[['trade_date', 'ts_code', 'news_stock_impact']], on=['trade_date', 'ts_code'], how='left')
            else:
                df['news_stock_impact'] = 0.0
            df[['news_market_impact', 'news_stock_impact']] = df[['news_market_impact', 'news_stock_impact']].fillna(0.0)

            X = df[feats].fillna(0)
            try:
                df['prob'] = model.predict_proba(X)[:, 1]
            except Exception:
                df['prob'] = 0

            picks = df[df['prob'] > 0.8].sort_values('prob', ascending=False).head(3)
            if picks.empty:
                picks = df.sort_values('prob', ascending=False).head(1)

            p_next = os.path.join(PRICE_DIR, f"{d_next}.parquet")
            if not os.path.exists(p_next):
                break
            next_df = pd.read_parquet(p_next, columns=['ts_code', 'open', 'high', 'close', 'pre_close'])

            day_pnl = 0
            alloc   = capital / max(1, len(picks))
            for _, row in picks.iterrows():
                ts_code = row['ts_code']
                nxt = next_df[next_df['ts_code'] == ts_code]
                if nxt.empty:
                    continue
                n_row = nxt.iloc[0]
                open_p, high_p, close_p, pre_close_p = n_row['open'], n_row['high'], n_row['close'], n_row['pre_close']
                is_20_pct = ts_code.startswith('300') or ts_code.startswith('688')
                up_limit  = round(pre_close_p * 1.2, 2) if is_20_pct else round(pre_close_p * 1.1, 2)
                lockup_t  = pre_close_p * 1.195 if is_20_pct else pre_close_p * 1.095
                if pd.isna(open_p) or open_p >= up_limit or open_p >= lockup_t:
                    continue
                buy_price  = open_p
                sell_price = buy_price * 1.04 if high_p >= buy_price * 1.04 else close_p
                ret = (sell_price / buy_price) - 1 - 0.0015
                day_pnl += alloc * ret

            capital += day_pnl
            equity.append({'date': pd.to_datetime(d_next), 'nav': capital})

        current_test_start += pd.DateOffset(months=STEP_MONTHS)

    eq_df     = pd.DataFrame(equity)
    total_ret = capital / INITIAL_CAP - 1
    years     = len(eq_df) / 252.0 if len(eq_df) > 0 else 1
    ann_ret   = (1 + total_ret) ** (1 / years) - 1 if years > 0 else 0
    df_ret    = eq_df['nav'].pct_change()
    mdd       = ((eq_df['nav'] - eq_df['nav'].cummax()) / eq_df['nav'].cummax()).min() if len(eq_df) > 0 else 0
    vol       = df_ret.std() * np.sqrt(252)
    sharpe    = ann_ret / vol if vol > 0 else 0

    print(f"\n[{label}] Total={total_ret:.2%}  Annual={ann_ret:.2%}  MDD={mdd:.2%}  Sharpe={sharpe:.2f}")
    return {'label': label, 'total': total_ret, 'annual': ann_ret, 'mdd': mdd, 'sharpe': sharpe, 'equity': eq_df}


if __name__ == '__main__':
    results = []
    for label, cap in CAPS_TO_TEST:
        print(f"\n{'='*60}")
        print(f">>> 正在测试市值上限: {label}")
        print('='*60)
        res = run_cap_backtest(cap, label)
        results.append(res)

    # ---- 汇总输出 ----
    print("\n\n" + "="*60)
    print("市值过滤敏感度分析结果汇总")
    print("="*60)
    print(f"{'市值上限':<12} {'总收益率':>10} {'年化收益':>10} {'最大回撤':>10} {'夏普':>8}")
    print("-"*60)
    for r in results:
        print(f"{r['label']:<12} {r['total']:>10.2%} {r['annual']:>10.2%} {r['mdd']:>10.2%} {r['sharpe']:>8.2f}")

    # ---- 画图 ----
    plt.figure(figsize=(12, 7))
    for r in results:
        if len(r['equity']) > 0:
            plt.plot(r['equity']['date'], r['equity']['nav'], label=r['label'], linewidth=2)
    plt.title('新闻T+1策略：不同流通市值上限对比 (Expanding WFO, 2024-2026)', fontsize=14)
    plt.xlabel('日期')
    plt.ylabel('净值（初始10万）')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'mktcap_sensitivity.png')
    plt.savefig(out_path, dpi=150)
    print(f"\n对比图已保存: {out_path}")

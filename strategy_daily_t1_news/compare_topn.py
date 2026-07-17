"""
Top-N Portfolio Size Sensitivity Analysis
Compares: Top 3 vs Top 5 vs Top 10 stocks picked daily
Period: 2024-01-01 to 2026-03-26
WFO: Quarterly model retraining (3-month steps)
"""
import os
import sys
import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from infra_data.storage import DataStorage
from train_model import train_daily_model

DATA_DIR  = r'D:\iquant_data\data_v2'
PRICE_DIR = os.path.join(DATA_DIR, 'data_day1')
RANK_DIR  = os.path.join(DATA_DIR, 'ths_rank1')
CHIP_DIR  = os.path.join(DATA_DIR, 'cyq1')
OTHER_DIR = os.path.join(DATA_DIR, 'other_day1')


def run_topn_leg(test_dates, model, feats, news_market_df, news_stock_sector_df, capital, top_n=3):
    """Single pass through test days for a given top_n portfolio size."""
    equity = []

    for i in range(len(test_dates) - 1):
        d_curr, d_next = test_dates[i], test_dates[i + 1]

        p_rank  = os.path.join(RANK_DIR,  f"{d_curr}.parquet")
        p_chip  = os.path.join(CHIP_DIR,  f"{d_curr}.parquet")
        p_price = os.path.join(PRICE_DIR, f"{d_curr}.parquet")
        p_other = os.path.join(OTHER_DIR, f"{d_curr}.parquet")
        if not all(os.path.exists(p) for p in [p_rank, p_chip, p_price, p_other]):
            continue

        rank_df  = pd.read_parquet(p_rank)
        rank_df['hot_rank_pct'] = rank_df['hot'].rank(pct=True)

        chip_df  = pd.read_parquet(p_chip)
        chip_df['chip_concentration'] = (
            (chip_df['cost_85pct'] - chip_df['cost_15pct']) / (chip_df['cost_50pct'] + 1e-8)
        )

        price_df = pd.read_parquet(p_price,
                                   columns=['ts_code', 'close', 'pct_chg', 'amount', 'vol', 'pre_close'])
        other_df = pd.read_parquet(p_other, columns=['ts_code', 'circ_mv'])

        df = pd.merge(rank_df[['ts_code', 'hot_rank_pct']], price_df, on='ts_code')
        df = pd.merge(df, chip_df[['ts_code', 'chip_concentration', 'winner_rate']], on='ts_code')
        df = pd.merge(df, other_df, on='ts_code', how='left')

        # Filters: no STAR, cap <= 500bil
        df = df[~df['ts_code'].str.startswith('688')]
        df = df[df['circ_mv'] <= 500000]

        df['trade_date'] = d_next

        if not news_market_df.empty:
            df = pd.merge(df, news_market_df, on='trade_date', how='left')
        else:
            df['news_market_impact'] = 0.0

        if not news_stock_sector_df.empty:
            df = pd.merge(df,
                          news_stock_sector_df[['trade_date', 'ts_code', 'news_stock_impact']],
                          on=['trade_date', 'ts_code'], how='left')
        else:
            df['news_stock_impact'] = 0.0

        df[['news_market_impact', 'news_stock_impact']] = (
            df[['news_market_impact', 'news_stock_impact']].fillna(0.0)
        )

        X = df[feats].fillna(0)
        try:
            df['prob'] = model.predict_proba(X)[:, 1]
        except Exception:
            df['prob'] = 0

        # Pick top_n, prefer high-prob stocks above 0.8 first
        picks = df[df['prob'] > 0.8].sort_values('prob', ascending=False).head(top_n)
        if len(picks) < top_n:
            # Fill remaining slots from rest sorted by prob
            extra = df[~df.index.isin(picks.index)].sort_values('prob', ascending=False).head(top_n - len(picks))
            picks = pd.concat([picks, extra])
        if picks.empty:
            equity.append({'date': pd.to_datetime(d_next), 'nav': capital})
            continue

        p_next = os.path.join(PRICE_DIR, f"{d_next}.parquet")
        if not os.path.exists(p_next):
            break
        next_df = pd.read_parquet(p_next,
                                  columns=['ts_code', 'open', 'high', 'close', 'pre_close'])

        alloc = capital / max(1, len(picks))
        day_pnl = 0

        for _, row in picks.iterrows():
            ts_code = row['ts_code']
            nxt = next_df[next_df['ts_code'] == ts_code]
            if nxt.empty:
                continue

            n_row = nxt.iloc[0]
            open_p, high_p, close_p, pre_close_p = (
                n_row['open'], n_row['high'], n_row['close'], n_row['pre_close']
            )

            is_20pct = ts_code.startswith('300') or ts_code.startswith('688')
            up_limit         = round(pre_close_p * 1.2, 2) if is_20pct else round(pre_close_p * 1.1, 2)
            lockup_threshold = pre_close_p * 1.195         if is_20pct else pre_close_p * 1.095

            if pd.isna(open_p) or open_p >= up_limit or open_p >= lockup_threshold:
                continue

            buy_price  = open_p
            sell_price = buy_price * 1.04 if high_p >= buy_price * 1.04 else close_p
            ret = (sell_price / buy_price) - 1 - 0.0015
            day_pnl += alloc * ret

        capital += day_pnl
        equity.append({'date': pd.to_datetime(d_next), 'nav': capital})

    return equity, capital


def run_topn_comparison(start_date='20240101', end_date='20260326',
                         train_start='20220101', step_months=3,
                         top_n_list=(3, 5, 10, 20)):
    print("=== Top-N Portfolio Size Sensitivity Analysis ===")
    print(f"Test Period : {start_date} ~ {end_date}")
    print(f"Retraining  : every {step_months} months")
    print(f"Portfolio Sizes: {top_n_list}")

    dates = sorted([f.replace('.parquet', '') for f in os.listdir(PRICE_DIR)
                    if f.endswith('.parquet')])

    current_test_start   = pd.to_datetime(start_date)
    final_end_date       = pd.to_datetime(end_date)
    fixed_train_start_dt = pd.to_datetime(train_start)

    storage = DataStorage()
    initial_cap = 100000.0
    capitals = {n: initial_cap for n in top_n_list}
    equities = {n: [] for n in top_n_list}

    while current_test_start <= final_end_date:
        current_test_end = current_test_start + pd.DateOffset(months=step_months) - pd.Timedelta(days=1)
        if current_test_end > final_end_date:
            current_test_end = final_end_date

        train_end_dt = current_test_start - pd.Timedelta(days=1)
        t0 = fixed_train_start_dt.strftime('%Y%m%d')
        t1 = train_end_dt.strftime('%Y%m%d')
        s0 = current_test_start.strftime('%Y%m%d')
        s1 = current_test_end.strftime('%Y%m%d')

        print(f"\n>>> [WFO] Train: {t0}~{t1} | Test: {s0}~{s1}")

        model, feats = train_daily_model(t0, t1, model_path=None)
        if model is None:
            print("Skipping: insufficient training data.")
            current_test_start += pd.DateOffset(months=step_months)
            continue

        test_dates = [d for d in dates if s0 <= d <= s1]
        if len(test_dates) < 2:
            current_test_start += pd.DateOffset(months=step_months)
            continue

        test_series = pd.Series([pd.to_datetime(d) for d in test_dates]).sort_values()
        news_mkt, news_stk = storage.load_news_data(s0, s1, test_series)

        if not news_mkt.empty:
            news_mkt['trade_date'] = news_mkt['trade_date'].dt.strftime('%Y%m%d')
        if not news_stk.empty:
            news_stk['trade_date'] = news_stk['trade_date'].dt.strftime('%Y%m%d')

        for n in top_n_list:
            seg, capitals[n] = run_topn_leg(
                test_dates, model, feats, news_mkt, news_stk, capitals[n], top_n=n
            )
            equities[n].extend(seg)
            print(f"  Top-{n}: capital={capitals[n]:,.0f}")

        current_test_start += pd.DateOffset(months=step_months)

    # ---- Stats ----
    def stats(eq_list, final_cap):
        if not eq_list:
            return {}
        eq_df = pd.DataFrame(eq_list)
        total_ret = final_cap / initial_cap - 1
        years = len(eq_df) / 252.0
        ann_ret = (1 + total_ret) ** (1 / max(years, 0.01)) - 1
        df_ret = eq_df['nav'].pct_change().dropna()
        mdd = ((eq_df['nav'] - eq_df['nav'].cummax()) / eq_df['nav'].cummax()).min()
        vol = df_ret.std() * np.sqrt(252)
        sharpe = ann_ret / max(vol, 1e-6)
        return dict(total_ret=total_ret, ann_ret=ann_ret, mdd=mdd, sharpe=sharpe)

    print("\n" + "="*70)
    print(f"{'Strategy':<18} {'Ann.Ret':>9} {'Sharpe':>8} {'MDD':>8} {'TotalRet':>10}")
    print("-"*70)
    all_stats = {}
    for n in top_n_list:
        s = stats(equities[n], capitals[n])
        all_stats[n] = s
        print(f"{'Top-'+str(n):<18} {s.get('ann_ret', 0):>9.2%} {s.get('sharpe', 0):>8.2f} {s.get('mdd', 0):>8.2%} {s.get('total_ret', 0):>10.2%}")
    print("="*70)

    # ---- Plot ----
    colors = ['royalblue', 'green', 'tomato', 'purple']
    styles = ['-', '--', '-.', ':']
    fig, ax = plt.subplots(figsize=(14, 8))

    for i, n in enumerate(top_n_list):
        if equities[n]:
            df_eq = pd.DataFrame(equities[n])
            s = all_stats[n]
            ax.plot(df_eq['date'], df_eq['nav'],
                    label=f"Top-{n} | Ann: {s.get('ann_ret',0):.1%} | Sharpe: {s.get('sharpe',0):.2f} | MDD: {s.get('mdd',0):.1%}",
                    color=colors[i], linewidth=1.8, linestyle=styles[i])

    ax.axhline(initial_cap, color='gray', linewidth=0.8, linestyle=':')
    ax.set_title("Top-N Portfolio Size Comparison (2024-2026, Quarterly WFO)", fontsize=14)
    ax.set_xlabel("Date")
    ax.set_ylabel("Portfolio Value (CNY)")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    out_dir  = os.path.dirname(os.path.abspath(__file__))
    out_path = os.path.join(out_dir, 'topn_comparison.png')
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"\nChart saved: {out_path}")

    return equities


if __name__ == "__main__":
    run_topn_comparison()

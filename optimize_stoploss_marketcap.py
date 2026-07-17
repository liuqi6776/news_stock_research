import os
import sys
import pandas as pd
import numpy as np
import joblib
from tqdm import tqdm
import matplotlib.pyplot as plt
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from infra_data.storage import DataStorage

DATA_DIR = r'D:\iquant_data\data_v2'
PRICE_DIR = os.path.join(DATA_DIR, 'data_day1')
RANK_DIR  = os.path.join(DATA_DIR, 'ths_rank1')
CHIP_DIR  = os.path.join(DATA_DIR, 'cyq1')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, 'results_duobao')

FEATURE_COLS = ['hot_rank_pct', 'chip_concentration', 'winner_rate',
               'news_market_impact', 'news_stock_impact', 'news_sector_impact']

def get_all_dates():
    return sorted([f.replace('.parquet', '') for f in os.listdir(PRICE_DIR) if f.endswith('.parquet')])

def load_features_for_date(date_str):
    p_rank = os.path.join(RANK_DIR, f"{date_str}.parquet")
    p_chip = os.path.join(CHIP_DIR, f"{date_str}.parquet")
    p_price = os.path.join(PRICE_DIR, f"{date_str}.parquet")
    
    if not all(os.path.exists(p) for p in [p_rank, p_chip, p_price]):
        return None
    
    rank_df = pd.read_parquet(p_rank)
    rank_df['hot_rank_pct'] = rank_df['hot'].rank(pct=True)
    
    chip_df = pd.read_parquet(p_chip)
    chip_df['chip_concentration'] = (chip_df['cost_85pct'] - chip_df['cost_15pct']) / (chip_df['cost_50pct'] + 1e-8)
    
    price_df = pd.read_parquet(p_price, columns=['ts_code', 'close', 'pct_chg', 'amount', 'vol', 'pre_close', 'total_mv'])
    
    df = pd.merge(rank_df[['ts_code', 'hot_rank_pct']], price_df, on='ts_code')
    df = pd.merge(df, chip_df[['ts_code', 'chip_concentration', 'winner_rate']], on='ts_code')
    
    return df

def load_news_data(start_date, end_date, dates_list):
    storage = DataStorage()
    valid_dates = pd.Series([pd.to_datetime(d) for d in dates_list if start_date <= d <= end_date]).sort_values()
    news_market_df, news_stock_sector_df = storage.load_news_data(start_date, end_date, valid_dates)
    
    if not news_market_df.empty:
        news_market_df['trade_date'] = news_market_df['trade_date'].dt.strftime('%Y%m%d')
    if not news_stock_sector_df.empty:
        news_stock_sector_df['trade_date'] = news_stock_sector_df['trade_date'].dt.strftime('%Y%m%d')
    
    return news_market_df, news_stock_sector_df

def run_backtest_with_params(stop_loss_pct=None, min_mv=None, max_mv=None, model_name="baseline"):
    dates = get_all_dates()
    
    TEST_START = '20220101'
    TEST_END = '20260324'
    
    test_dates = [d for d in dates if TEST_START <= d <= TEST_END]
    all_news_dates = [d for d in dates if '20200101' <= d <= TEST_END]
    
    model_path = os.path.join(BASE_DIR, 'daily_dragon_news_model.joblib')
    if not os.path.exists(model_path):
        return None
    
    model, feats = joblib.load(model_path)
    news_market_df, news_stock_sector_df = load_news_data('20200101', TEST_END, all_news_dates)
    
    initial_cap = 100000.0
    capital = initial_cap
    equity = []
    
    for i in tqdm(range(len(test_dates)-2), desc=f"Running {model_name}"):
        d_t = test_dates[i]
        d_t1 = test_dates[i+1]
        d_t2 = test_dates[i+2]
        
        df_t = load_features_for_date(d_t)
        if df_t is None:
            continue
        
        if min_mv is not None:
            df_t = df_t[df_t['total_mv'] >= min_mv]
        if max_mv is not None:
            df_t = df_t[df_t['total_mv'] <= max_mv]
        
        if len(df_t) == 0:
            continue
        
        df_t['trade_date'] = d_t1
        
        if not news_market_df.empty:
            df_t = pd.merge(df_t, news_market_df, on='trade_date', how='left')
        else:
            df_t['news_market_impact'] = 0.0
            
        if not news_stock_sector_df.empty:
            df_t = pd.merge(df_t, news_stock_sector_df, on=['trade_date', 'ts_code'], how='left')
        else:
            df_t['news_stock_impact'] = 0.0
            df_t['news_sector_impact'] = 0.0
            
        df_t[['news_market_impact', 'news_stock_impact', 'news_sector_impact']] = \
            df_t[['news_market_impact', 'news_stock_impact', 'news_sector_impact']].fillna(0.0)
        
        X = df_t[feats].fillna(0)
        
        try:
            df_t['prob'] = model.predict_proba(X)[:, 1]
        except Exception:
            continue
        
        picks = df_t[df_t['prob'] > 0.8].sort_values('prob', ascending=False).head(3)
        if picks.empty:
            picks = df_t.sort_values('prob', ascending=False).head(1)
        
        p_t2 = os.path.join(PRICE_DIR, f"{d_t2}.parquet")
        if not os.path.exists(p_t2):
            continue
        df_t2 = pd.read_parquet(p_t2, columns=['ts_code', 'open', 'high', 'low', 'close', 'pre_close'])
        
        day_pnl = 0.0
        alloc = capital / len(picks)
        
        for _, pick in picks.iterrows():
            ts_code = pick['ts_code']
            t2_row = df_t2[df_t2['ts_code'] == ts_code]
            
            if t2_row.empty:
                continue
            
            t2 = t2_row.iloc[0]
            pre_close = t2['pre_close']
            
            up_limit = round(pre_close * 1.2, 2) if ('300' in ts_code or '688' in ts_code) else round(pre_close * 1.1, 2)
            
            if pd.isna(t2['open']) or t2['open'] >= up_limit:
                continue
            
            buy_price = t2['open']
            
            if stop_loss_pct is not None:
                if t2['low'] <= buy_price * (1 - stop_loss_pct):
                    sell_price = buy_price * (1 - stop_loss_pct)
                elif t2['high'] >= buy_price * 1.04:
                    sell_price = buy_price * 1.04
                else:
                    sell_price = t2['close']
            else:
                if t2['high'] >= buy_price * 1.04:
                    sell_price = buy_price * 1.04
                else:
                    sell_price = t2['close']
            
            ret = (sell_price / buy_price) - 1
            ret -= 0.0015
            
            day_pnl += alloc * ret
        
        capital += day_pnl
        equity.append({'date': pd.to_datetime(d_t2), 'nav': capital})
    
    eq_df = pd.DataFrame(equity)
    
    if len(eq_df) == 0:
        return None
    
    total_ret = capital / initial_cap - 1
    years = len(eq_df) / 252.0
    ann_ret = (1 + total_ret) ** (1 / years) - 1 if years > 0 else 0
    df_ret = eq_df['nav'].pct_change()
    mdd = ((eq_df['nav'] - eq_df['nav'].cummax()) / eq_df['nav'].cummax()).min()
    vol = df_ret.std() * np.sqrt(252)
    sharpe = ann_ret / vol if vol > 0 else 0
    
    return {
        'name': model_name,
        'stop_loss': stop_loss_pct,
        'min_mv': min_mv,
        'max_mv': max_mv,
        'equity': eq_df,
        'total_ret': total_ret,
        'ann_ret': ann_ret,
        'mdd': mdd,
        'sharpe': sharpe,
        'num_days': len(eq_df),
        'final_cap': capital
    }

def main():
    print("="*80)
    print("STRATEGY OPTIMIZATION - STOP LOSS & MARKET CAP FILTER")
    print("="*80)
    
    results = []
    
    print("\n1. Running Baseline (no stop loss, no market cap filter)...")
    baseline = run_backtest_with_params(None, None, None, "Baseline")
    if baseline:
        results.append(baseline)
        print(f"   Done: Return={baseline['total_ret']:.2%}, Sharpe={baseline['sharpe']:.2f}")
    
    print("\n2. Testing Stop Loss values...")
    for sl in [0.03, 0.05, 0.08]:
        res = run_backtest_with_params(sl, None, None, f"StopLoss_{int(sl*100)}%")
        if res:
            results.append(res)
            print(f"   StopLoss {int(sl*100)}%: Return={res['total_ret']:.2%}, Sharpe={res['sharpe']:.2f}")
    
    print("\n3. Testing Market Cap filters (min only)...")
    for min_mv in [1e9, 5e9, 10e9, 20e9]:
        mv_label = f"{int(min_mv/1e8)}亿" if min_mv >= 1e8 else f"{int(min_mv/1e4)}万"
        res = run_backtest_with_params(None, min_mv, None, f"MinMV_{mv_label}")
        if res:
            results.append(res)
            print(f"   MinMV {mv_label}: Return={res['total_ret']:.2%}, Sharpe={res['sharpe']:.2f}")
    
    print("\n4. Testing combined (StopLoss 5% + MinMV 10亿)...")
    combined = run_backtest_with_params(0.05, 10e9, None, "Combined_SL5%_MinMV10亿")
    if combined:
        results.append(combined)
        print(f"   Done: Return={combined['total_ret']:.2%}, Sharpe={combined['sharpe']:.2f}")
    
    print("\n5. Testing combined (StopLoss 5% + MinMV 5亿)...")
    combined2 = run_backtest_with_params(0.05, 5e9, None, "Combined_SL5%_MinMV5亿")
    if combined2:
        results.append(combined2)
        print(f"   Done: Return={combined2['total_ret']:.2%}, Sharpe={combined2['sharpe']:.2f}")
    
    print("\n" + "="*80)
    print("SUMMARY OF RESULTS")
    print("="*80)
    
    for r in results:
        print(f"{r['name']:30s} | Return: {r['total_ret']:+8.2%} | Sharpe: {r['sharpe']:5.2f} | MDD: {r['mdd']:6.2%}")
    
    best_sharpe = max(results, key=lambda x: x['sharpe'])
    best_return = max(results, key=lambda x: x['total_ret'])
    
    print("\n" + "="*80)
    print(f"BEST BY SHARPE: {best_sharpe['name']} (Sharpe={best_sharpe['sharpe']:.2f})")
    print(f"BEST BY RETURN: {best_return['name']} (Return={best_return['total_ret']:.2%})")
    print("="*80)
    
    plt.figure(figsize=(16, 10))
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
              '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf', '#ff9896']
    
    for idx, r in enumerate(results):
        plt.plot(r['equity']['date'], r['equity']['nav'], 
                label=f"{r['name']} (Sharpe={r['sharpe']:.2f})",
                linewidth=2, color=colors[idx % len(colors)])
    
    plt.title('Strategy Comparison - Stop Loss & Market Cap Filter', fontsize=16, fontweight='bold')
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Capital', fontsize=12)
    plt.legend(fontsize=10, bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    output_plot = os.path.join(OUTPUT_DIR, 'strategy_comparison_stoploss_marketcap.png')
    plt.savefig(output_plot, dpi=150, bbox_inches='tight')
    print(f"\nComparison plot saved: {output_plot}")
    
    summary_data = []
    for r in results:
        summary_data.append({
            'Strategy': r['name'],
            'Stop Loss': f"{int(r['stop_loss']*100)}%" if r['stop_loss'] else 'None',
            'Min Market Cap': f"{int(r['min_mv']/1e8)}亿" if r['min_mv'] else 'None',
            'Max Market Cap': f"{int(r['max_mv']/1e8)}亿" if r['max_mv'] else 'None',
            'Total Return': f"{r['total_ret']:+.2%}",
            'Annual Return': f"{r['ann_ret']:+.2%}",
            'Max Drawdown': f"{r['mdd']:.2%}",
            'Sharpe Ratio': f"{r['sharpe']:.2f}",
            'Final Capital': f"¥{r['final_cap']:,.0f}"
        })
    
    summary_df = pd.DataFrame(summary_data)
    summary_csv = os.path.join(OUTPUT_DIR, 'strategy_comparison_summary.csv')
    summary_df.to_csv(summary_csv, index=False, encoding='utf-8-sig')
    print(f"Summary table saved: {summary_csv}")
    
    report_file = os.path.join(OUTPUT_DIR, 'OPTIMIZATION_REPORT.md')
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("# 策略优化报告 - 止损与市值筛选\n\n")
        f.write("## 📊 所有策略对比\n\n")
        f.write("| 策略 | 止损 | 最小市值 | 总收益率 | 年化 | 最大回撤 | 夏普比率 |\n")
        f.write("|------|------|---------|---------|------|---------|---------|\n")
        for r in results:
            sl_str = f"{int(r['stop_loss']*100)}%" if r['stop_loss'] else '-'
            mv_str = f"{int(r['min_mv']/1e8)}亿" if r['min_mv'] else '-'
            f.write(f"| {r['name']} | {sl_str} | {mv_str} | {r['total_ret']:+.2%} | {r['ann_ret']:+.2%} | {r['mdd']:.2%} | {r['sharpe']:.2f} |\n")
        
        f.write("\n## 🏆 最佳策略\n\n")
        f.write(f"### 按夏普比率最高：{best_sharpe['name']}\n")
        f.write(f"- 夏普比率：{best_sharpe['sharpe']:.2f}\n")
        f.write(f"- 总收益率：{best_sharpe['total_ret']:+.2%}\n")
        f.write(f"- 最大回撤：{best_sharpe['mdd']:.2%}\n\n")
        
        f.write(f"### 按收益率最高：{best_return['name']}\n")
        f.write(f"- 总收益率：{best_return['total_ret']:+.2%}\n")
        f.write(f"- 夏普比率：{best_return['sharpe']:.2f}\n")
        f.write(f"- 最大回撤：{best_return['mdd']:.2%}\n")
    
    print(f"Report saved: {report_file}")

if __name__ == "__main__":
    main()

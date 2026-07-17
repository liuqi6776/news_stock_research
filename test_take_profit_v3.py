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

def preload_all_data(test_dates, all_news_dates):
    print("正在预加载所有数据...")
    
    model_path = os.path.join(BASE_DIR, 'daily_dragon_news_model.joblib')
    model, feats = joblib.load(model_path)
    print(f"  ✓ 加载模型: {model_path}")
    
    storage = DataStorage()
    valid_dates = pd.Series([pd.to_datetime(d) for d in all_news_dates if '20200101' <= d <= '20260324']).sort_values()
    news_market_df, news_stock_sector_df = storage.load_news_data('20200101', '20260324', valid_dates)
    
    if not news_market_df.empty:
        news_market_df['trade_date'] = news_market_df['trade_date'].dt.strftime('%Y%m%d')
    if not news_stock_sector_df.empty:
        news_stock_sector_df['trade_date'] = news_stock_sector_df['trade_date'].dt.strftime('%Y%m%d')
    print(f"  ✓ 加载新闻数据")
    
    preloaded = {}
    for i in tqdm(range(len(test_dates)-2), desc="预加载每日数据"):
        d_t = test_dates[i]
        d_t1 = test_dates[i+1]
        d_t2 = test_dates[i+2]
        
        p_rank = os.path.join(RANK_DIR, f"{d_t}.parquet")
        p_chip = os.path.join(CHIP_DIR, f"{d_t}.parquet")
        p_price = os.path.join(PRICE_DIR, f"{d_t}.parquet")
        p_t2 = os.path.join(PRICE_DIR, f"{d_t2}.parquet")
        
        if not all(os.path.exists(p) for p in [p_rank, p_chip, p_price, p_t2]):
            continue
        
        rank_df = pd.read_parquet(p_rank)
        rank_df['hot_rank_pct'] = rank_df['hot'].rank(pct=True)
        
        chip_df = pd.read_parquet(p_chip)
        chip_df['chip_concentration'] = (chip_df['cost_85pct'] - chip_df['cost_15pct']) / (chip_df['cost_50pct'] + 1e-8)
        
        price_df = pd.read_parquet(p_price, columns=['ts_code', 'close', 'pct_chg', 'amount', 'vol', 'pre_close'])
        
        df_t = pd.merge(rank_df[['ts_code', 'hot_rank_pct']], price_df, on='ts_code')
        df_t = pd.merge(df_t, chip_df[['ts_code', 'chip_concentration', 'winner_rate']], on='ts_code')
        
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
        
        df_t2 = pd.read_parquet(p_t2, columns=['ts_code', 'open', 'high', 'close', 'pre_close'])
        
        preloaded[d_t] = {
            'picks': picks,
            'df_t2': df_t2,
            'date_t2': d_t2
        }
    
    print(f"  ✓ 预加载完成: {len(preloaded)} 天的数据")
    return preloaded

def run_backtest_on_preloaded(preloaded, take_profit_pct):
    initial_cap = 100000.0
    capital = initial_cap
    equity = []
    
    for d_t, data in preloaded.items():
        picks = data['picks']
        df_t2 = data['df_t2']
        d_t2 = data['date_t2']
        
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
            
            if t2['high'] >= buy_price * (1 + take_profit_pct):
                sell_price = buy_price * (1 + take_profit_pct)
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
        'take_profit_pct': take_profit_pct,
        'initial_cap': initial_cap,
        'final_cap': capital,
        'total_ret': total_ret,
        'ann_ret': ann_ret,
        'mdd': mdd,
        'sharpe': sharpe,
        'num_trades': len(eq_df),
        'equity_df': eq_df
    }

def main():
    dates = get_all_dates()
    
    TEST_START = '20220101'
    TEST_END = '20260324'
    
    test_dates = [d for d in dates if TEST_START <= d <= TEST_END]
    all_news_dates = [d for d in dates if '20200101' <= d <= TEST_END]
    
    preloaded = preload_all_data(test_dates, all_news_dates)
    
    take_profit_list = [0.03, 0.04, 0.05, 0.06, 0.07, 0.08]
    results = []
    
    print("\n" + "="*80)
    print("多止盈点回测对比")
    print("="*80)
    
    for tp in take_profit_list:
        print(f"\n正在回测止盈点: {tp*100:.0f}%")
        result = run_backtest_on_preloaded(preloaded, tp)
        if result:
            results.append(result)
            print(f"  ✓ 完成: 总收益 {result['total_ret']:+.2%}, 年化 {result['ann_ret']:+.2%}, 夏普 {result['sharpe']:.2f}, 回撤 {result['mdd']:.2%}")
        else:
            print(f"  ✗ 失败")
    
    print("\n" + "="*80)
    print("回测结果对比")
    print("="*80)
    
    df_summary = pd.DataFrame([{
        '止盈点': f"{r['take_profit_pct']*100:.0f}%",
        '总收益率': f"{r['total_ret']:+.2%}",
        '年化收益率': f"{r['ann_ret']:+.2%}",
        '最大回撤': f"{r['mdd']:.2%}",
        '夏普比率': f"{r['sharpe']:.2f}",
        '交易天数': r['num_trades'],
        '最终资金': f"¥{r['final_cap']:,.2f}"
    } for r in results])
    
    print(df_summary.to_string(index=False))
    
    summary_csv = os.path.join(OUTPUT_DIR, 'take_profit_comparison.csv')
    df_summary.to_csv(summary_csv, index=False)
    print(f"\n对比结果已保存: {summary_csv}")
    
    plt.figure(figsize=(16, 10))
    
    for r in results:
        label = f"止盈 {r['take_profit_pct']*100:.0f}%"
        plt.plot(r['equity_df']['date'], r['equity_df']['nav'], label=label, linewidth=2)
    
    plt.title('不同止盈点策略净值对比', fontsize=16, fontweight='bold')
    plt.xlabel('日期', fontsize=14)
    plt.ylabel('资金', fontsize=14)
    plt.legend(fontsize=12, loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    comparison_png = os.path.join(OUTPUT_DIR, 'take_profit_comparison.png')
    plt.savefig(comparison_png, dpi=150)
    print(f"对比图已保存: {comparison_png}")
    
    best_by_sharpe = max(results, key=lambda x: x['sharpe'])
    best_by_return = max(results, key=lambda x: x['total_ret'])
    
    print("\n" + "="*80)
    print("最优策略推荐")
    print("="*80)
    print(f"按夏普比率最优: 止盈 {best_by_sharpe['take_profit_pct']*100:.0f}%")
    print(f"  夏普: {best_by_sharpe['sharpe']:.2f}, 总收益: {best_by_sharpe['total_ret']:+.2%}, 回撤: {best_by_sharpe['mdd']:.2%}")
    print(f"\n按总收益最优: 止盈 {best_by_return['take_profit_pct']*100:.0f}%")
    print(f"  总收益: {best_by_return['total_ret']:+.2%}, 夏普: {best_by_return['sharpe']:.2f}, 回撤: {best_by_return['mdd']:.2%}")
    print("="*80)

if __name__ == "__main__":
    main()

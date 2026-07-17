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
OUTPUT_DIR = os.path.join(BASE_DIR, 'new_idea')

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
    
    price_df = pd.read_parquet(p_price, columns=['ts_code', 'close', 'pct_chg', 'amount', 'vol', 'pre_close'])
    
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

def preload_trades_for_strategy(strategy_name, model, feats, test_dates, news_market_df, news_stock_sector_df):
    all_trades = []
    
    for i in tqdm(range(len(test_dates)-2), desc=f"策略{strategy_name}"):
        d_t = test_dates[i]
        d_t1 = test_dates[i+1]
        d_t2 = test_dates[i+2]
        
        df_t = load_features_for_date(d_t)
        if df_t is None:
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
        
        if strategy_name == '1':
            picks = df_t[df_t['prob'] > 0.8].sort_values('prob', ascending=False).head(3)
            if picks.empty:
                picks = df_t.sort_values('prob', ascending=False).head(1)
        elif strategy_name == '2':
            q95 = df_t['prob'].quantile(0.95)
            picks = df_t[df_t['prob'] >= q95].sort_values('prob', ascending=False).head(3)
            if picks.empty:
                picks = df_t.sort_values('prob', ascending=False).head(1)
        elif strategy_name == '3':
            df_t['score'] = df_t['prob'] * 0.7 + df_t['winner_rate'] * 0.2 + (1 - df_t['chip_concentration']) * 0.1
            picks = df_t.sort_values('score', ascending=False).head(3)
        
        p_t2 = os.path.join(PRICE_DIR, f"{d_t2}.parquet")
        if not os.path.exists(p_t2):
            continue
        df_t2 = pd.read_parquet(p_t2, columns=['ts_code', 'open', 'high', 'low', 'close', 'pre_close'])
        
        for _, pick in picks.iterrows():
            ts_code = pick['ts_code']
            t2_row = df_t2[df_t2['ts_code'] == ts_code]
            
            if t2_row.empty:
                continue
            
            t2 = t2_row.iloc[0]
            pre_close_val = t2['pre_close']
            up_limit = round(pre_close_val * 1.2, 2) if ('300' in ts_code or '688' in ts_code) else round(pre_close_val * 1.1, 2)
            
            if pd.isna(t2['open']) or t2['open'] >= up_limit:
                continue
            
            all_trades.append({
                'date_t': d_t,
                'date_t2': d_t2,
                'ts_code': ts_code,
                'open': t2['open'],
                'high': t2['high'],
                'low': t2['low'],
                'close': t2['close'],
                'pre_close': pre_close_val
            })
    
    return pd.DataFrame(all_trades)

def int_to_date(date_int):
    s = str(date_int)
    return datetime(int(s[:4]), int(s[4:6]), int(s[6:8]))

def get_next_trading_day(current_date_int):
    current_dt = int_to_date(current_date_int)
    for i in range(1, 10):
        next_dt = current_dt + timedelta(days=i)
        next_int = int(date_to_int(next_dt))
        if os.path.exists(os.path.join(PRICE_DIR, f"{next_int}.parquet")):
            return next_int
    return None

def date_to_int(dt):
    return int(dt.strftime('%Y%m%d'))

def is_gem_or_star(ts_code):
    return ('300' in ts_code) or ('301' in ts_code) or ('688' in ts_code) or ('689' in ts_code)

def load_price_data(date_int):
    p_file = os.path.join(PRICE_DIR, f"{date_int}.parquet")
    if not os.path.exists(p_file):
        return None
    return pd.read_parquet(p_file)

from datetime import datetime, timedelta

def backtest_from_trades(trades_df, strategy_name):
    take_profit = 0.08
    
    initial_cap = 100000.0
    capital = initial_cap
    equity = []
    
    total_trades = 0
    cannot_sell_trades = 0
    
    for date_t2, group in trades_df.groupby('date_t2', sort=True):
        alloc = capital / len(group)
        day_pnl = 0.0
        
        for _, trade in group.iterrows():
            total_trades += 1
            ts_code = trade['ts_code']
            buy_price = trade['open']
            
            pre_close_t2 = trade['pre_close']
            
            if is_gem_or_star(ts_code):
                limit_down_pct = 0.8
            else:
                limit_down_pct = 0.9
            
            limit_down_price = round(pre_close_t2 * limit_down_pct, 2)
            
            open_price = trade['open']
            high_price = trade['high']
            low_price = trade['low']
            close_price = trade['close']
            
            is_cannot_sell = high_price == limit_down_price
            
            if is_cannot_sell:
                cannot_sell_trades += 1
                date_t3 = get_next_trading_day(int(date_t2))
                
                if date_t3:
                    df_t3 = load_price_data(date_t3)
                    if df_t3 is not None:
                        t3_row = df_t3[df_t3['ts_code'] == ts_code]
                        if not t3_row.empty:
                            sell_price = t3_row.iloc[0]['open']
                        else:
                            sell_price = close_price
                    else:
                        sell_price = close_price
                else:
                    sell_price = close_price
            else:
                if high_price >= buy_price * (1 + take_profit):
                    sell_price = buy_price * (1 + take_profit)
                else:
                    sell_price = close_price
            
            ret = (sell_price / buy_price) - 1
            ret -= 0.0015
            day_pnl += alloc * ret
        
        capital += day_pnl
        equity.append({'date': int_to_date(int(date_t2)), 'nav': capital})
    
    total_ret = capital / initial_cap - 1
    years = len(equity) / 252.0
    ann_ret = (1 + total_ret) ** (1 / years) - 1 if years > 0 else 0
    eq_df = pd.DataFrame(equity)
    if len(eq_df) > 1:
        df_ret = eq_df['nav'].pct_change().dropna()
        mdd = ((eq_df['nav'] - eq_df['nav'].cummax()) / eq_df['nav'].cummax()).min()
        vol = df_ret.std() * np.sqrt(252)
        sharpe = ann_ret / vol if vol > 0 else 0
    else:
        mdd = 0
        sharpe = 0
    
    return eq_df, total_ret, ann_ret, sharpe, mdd, total_trades, cannot_sell_trades

def main():
    print("="*80)
    print("  三种策略完整回测对比（基于 doubao_result 真实回测逻辑）")
    print("="*80)
    
    dates = get_all_dates()
    
    TEST_START = '20220101'
    TEST_END = '20260324'
    
    test_dates = [d for d in dates if TEST_START <= d <= TEST_END]
    all_news_dates = [d for d in dates if '20200101' <= d <= TEST_END]
    
    print(f"\n总测试日期数: {len(test_dates)}")
    
    model_path = os.path.join(BASE_DIR, 'daily_dragon_news_model.joblib')
    if not os.path.exists(model_path):
        print(f"模型不存在: {model_path}")
        return
    
    model, feats = joblib.load(model_path)
    print(f"已加载模型: {model_path}")
    
    print(f"\n正在加载新闻数据...")
    news_market_df, news_stock_sector_df = load_news_data('20200101', TEST_END, all_news_dates)
    print(f"  news_market_df: {len(news_market_df)} 行")
    print(f"  news_stock_sector_df: {len(news_stock_sector_df)} 行")
    
    print(f"\n{'='*80}")
    print("  1. 生成三种策略的交易记录")
    print(f"{'='*80}")
    
    strategies = {
        '1': '原策略（prob>0.8）',
        '2': '策略2（分位数0.95）',
        '3': '策略3（特征强化）'
    }
    
    trades_data = {}
    for strat_id, strat_name in strategies.items():
        print(f"\n策略{strat_id}: {strat_name}")
        trades_df = preload_trades_for_strategy(strat_id, model, feats, test_dates, news_market_df, news_stock_sector_df)
        trades_data[strat_id] = trades_df
        trades_path = os.path.join(OUTPUT_DIR, f'strategy{strat_id}_preloaded_trades.csv')
        trades_df.to_csv(trades_path, index=False)
        print(f"  交易数: {len(trades_df)}")
        print(f"  已保存: {trades_path}")
    
    print(f"\n{'='*80}")
    print("  2. 回测三种策略")
    print(f"{'='*80}")
    
    backtest_results = {}
    for strat_id, strat_name in strategies.items():
        print(f"\n策略{strat_id}: {strat_name}")
        eq_df, t_ret, a_ret, sharpe, mdd, t_trades, cs_trades = backtest_from_trades(trades_data[strat_id], strat_id)
        backtest_results[strat_id] = {
            'equity': eq_df,
            'total_return': t_ret,
            'annual_return': a_ret,
            'sharpe': sharpe,
            'max_drawdown': mdd,
            'total_trades': t_trades,
            'cannot_sell': cs_trades
        }
        print(f"  总交易: {t_trades}")
        print(f"  总收益: {t_ret:.2%}")
        print(f"  年化: {a_ret:.2%}")
        print(f"  夏普: {sharpe:.2f}")
        print(f"  最大回撤: {mdd:.2%}")
        
        eq_path = os.path.join(OUTPUT_DIR, f'strategy{strat_id}_equity.csv')
        eq_df.to_csv(eq_path, index=False)
        print(f"  净值已保存: {eq_path}")
    
    print(f"\n{'='*80}")
    print("  3. 加载 doubao_result 原策略真实结果")
    print(f"{'='*80}")
    
    doubao_equity_path = os.path.join(BASE_DIR, 'results_duobao', 'final_backtest_correct_equity.csv')
    df_doubao = pd.read_csv(doubao_equity_path)
    df_doubao['date'] = pd.to_datetime(df_doubao['date'])
    df_doubao['nav'] = df_doubao['nav'] / df_doubao['nav'].iloc[0]
    
    def calc_doubao_metrics(df):
        nav = df['nav'].values
        returns = np.diff(nav) / nav[:-1]
        total_return = nav[-1] - 1
        n_years = (df['date'].iloc[-1] - df['date'].iloc[0]).days / 365.25
        annual_return = (nav[-1]) ** (1/n_years) - 1
        sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0
        running_max = np.maximum.accumulate(nav)
        drawdown = (running_max - nav) / running_max
        max_drawdown = np.max(drawdown)
        return total_return, annual_return, sharpe, max_drawdown
    
    t_doubao, a_doubao, s_doubao, dd_doubao = calc_doubao_metrics(df_doubao)
    
    print(f"  doubao_result 原策略（旧feature）:")
    print(f"    总收益: {t_doubao:.2%}")
    print(f"    年化: {a_doubao:.2%}")
    print(f"    夏普: {s_doubao:.2f}")
    print(f"    最大回撤: {dd_doubao:.2%}")
    
    print(f"\n{'='*80}")
    print("  4. 绘制对比曲线")
    print(f"{'='*80}")
    
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    fig, axes = plt.subplots(2, 1, figsize=(16, 12))
    
    ax1 = axes[0]
    ax1.plot(df_doubao['date'], df_doubao['nav'], 
             label=f'doubao_result 原策略 (最终: {df_doubao["nav"].iloc[-1]:.2f}x)', 
             linewidth=3, color='#1f77b4', alpha=0.8)
    
    colors = ['#ff7f0e', '#2ca02c', '#d62728']
    for i, (strat_id, strat_name) in enumerate(strategies.items()):
        eq_df = backtest_results[strat_id]['equity']
        ax1.plot(eq_df['date'], eq_df['nav'] / eq_df['nav'].iloc[0], 
                 label=f'策略{strat_id} (最终: {(eq_df["nav"].iloc[-1]/eq_df["nav"].iloc[0]):.2f}x)', 
                 linewidth=2, color=colors[i], alpha=0.8)
    
    ax1.set_title('四种策略收益曲线对比（对数坐标）', fontsize=16, fontweight='bold')
    ax1.set_xlabel('日期', fontsize=12)
    ax1.set_ylabel('净值', fontsize=12)
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')
    
    ax2 = axes[1]
    ax2.plot(df_doubao['date'], df_doubao['nav'], 
             label=f'doubao_result 原策略 (最终: {df_doubao["nav"].iloc[-1]:.2f}x)', 
             linewidth=3, color='#1f77b4', alpha=0.8)
    
    for i, (strat_id, strat_name) in enumerate(strategies.items()):
        eq_df = backtest_results[strat_id]['equity']
        ax2.plot(eq_df['date'], eq_df['nav'] / eq_df['nav'].iloc[0], 
                 label=f'策略{strat_id} (最终: {(eq_df["nav"].iloc[-1]/eq_df["nav"].iloc[0]):.2f}x)', 
                 linewidth=2, color=colors[i], alpha=0.8)
    
    ax2.set_title('四种策略收益曲线对比（线性坐标）', fontsize=16, fontweight='bold')
    ax2.set_xlabel('日期', fontsize=12)
    ax2.set_ylabel('净值', fontsize=12)
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = os.path.join(OUTPUT_DIR, 'three_strategies_vs_doubao_final.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"✓ 对比图已保存: {plot_path}")
    
    print(f"\n{'='*80}")
    print("  5. 生成对比报告")
    print(f"{'='*80}")
    
    report_path = os.path.join(OUTPUT_DIR, 'final_three_strategies_report.md')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# 三种策略完整回测对比报告\n\n")
        f.write("---\n\n")
        
        f.write("## 策略说明\n\n")
        f.write("| 策略 | 说明 |\n")
        f.write("|------|------|\n")
        f.write("| doubao_result 原策略 | 旧feature模型，真实回测 |\n")
        f.write("| 策略1（原策略） | prob > 0.8，取Top 3，空则Top 1 |\n")
        f.write("| 策略2（分位数） | 每日取prob最高的5%，Top 3，空则Top 1 |\n")
        f.write("| 策略3（特征强化） | score = prob×0.7 + winner_rate×0.2 + (1-chip_concentration)×0.1，取Top 3 |\n")
        
        f.write("\n## 回测结果\n\n")
        f.write("| 策略 | 总交易数 | 总收益 | 年化 | 夏普 | 最大回撤 |\n")
        f.write("|------|---------|-------|------|------|---------|\n")
        f.write(f"| doubao_result 原策略 | - | {t_doubao:.2%} | {a_doubao:.2%} | {s_doubao:.2f} | {dd_doubao:.2%} |\n")
        
        for strat_id, strat_name in strategies.items():
            r = backtest_results[strat_id]
            f.write(f"| {strat_name} | {r['total_trades']} | {r['total_return']:.2%} | {r['annual_return']:.2%} | {r['sharpe']:.2f} | {r['max_drawdown']:.2%} |\n")
        
        f.write("\n## 收益曲线\n\n")
        f.write("![对比图](three_strategies_vs_doubao_final.png)\n")
        
        f.write("\n---\n\n")
        f.write("**注意**: 策略1/2/3使用的是旧feature模型，与 doubao_result 原策略模型相同，仅选股策略不同。\n")
    
    print(f"✓ 报告已保存: {report_path}")
    print(f"\n{'='*80}")
    print("  完成！")
    print(f"{'='*80}")

if __name__ == "__main__":
    main()

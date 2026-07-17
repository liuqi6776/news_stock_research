
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, 'results_duobao')
PRICE_DIR = r'D:\iquant_data\data_v2\data_day1'

def int_to_date(date_int):
    s = str(date_int)
    return datetime(int(s[:4]), int(s[4:6]), int(s[6:8]))

def date_to_int(dt):
    return int(dt.strftime('%Y%m%d'))

def load_price_data(date_int):
    p_file = os.path.join(PRICE_DIR, f"{date_int}.parquet")
    if not os.path.exists(p_file):
        return None
    return pd.read_parquet(p_file)

def get_next_trading_day(current_date_int):
    current_dt = int_to_date(current_date_int)
    for i in range(1, 10):
        next_dt = current_dt + timedelta(days=i)
        next_int = date_to_int(next_dt)
        if os.path.exists(os.path.join(PRICE_DIR, f"{next_int}.parquet")):
            return next_int
    return None

def is_gem_or_star(ts_code):
    return ('300' in ts_code) or ('301' in ts_code) or ('688' in ts_code) or ('689' in ts_code)

def get_limit_down_price(pre_close, ts_code):
    if is_gem_or_star(ts_code):
        return round(pre_close * 0.8, 2)
    else:
        return round(pre_close * 0.9, 2)

def find_sellable_date(start_date_int, ts_code, max_days=10):
    current_date_int = start_date_int
    for i in range(max_days):
        df = load_price_data(current_date_int)
        if df is None:
            current_date_int = get_next_trading_day(current_date_int)
            if current_date_int is None:
                return None, None
            continue
        
        row = df[df['ts_code'] == ts_code]
        if row.empty:
            current_date_int = get_next_trading_day(current_date_int)
            if current_date_int is None:
                return None, None
            continue
        
        pre_close = row.iloc[0]['pre_close']
        high = row.iloc[0]['high']
        limit_down_price = get_limit_down_price(pre_close, ts_code)
        
        if high > limit_down_price:
            return current_date_int, row.iloc[0]['open']
        
        current_date_int = get_next_trading_day(current_date_int)
        if current_date_int is None:
            return None, None
    
    return None, None

def main():
    trades_csv = os.path.join(OUTPUT_DIR, 'preloaded_trades.csv')
    
    if not os.path.exists(trades_csv):
        print("未找到预加载文件")
        return
    
    trades_df = pd.read_csv(trades_csv)
    print(f"已加载 {len(trades_df)} 条交易数据")
    
    take_profit = 0.08
    
    print("\n" + "="*80)
    print("回测：无止损 + 连续跌停处理")
    print("="*80)
    
    initial_cap = 100000.0
    capital = initial_cap
    equity = []
    
    total_trades = 0
    cannot_sell_trades = 0
    delayed_sell_trades = 0
    
    for date_t2, group in trades_df.groupby('date_t2', sort=True):
        alloc = capital / len(group)
        day_pnl = 0.0
        
        for _, trade in group.iterrows():
            total_trades += 1
            ts_code = trade['ts_code']
            buy_price = trade['open']
            
            pre_close_t2 = trade['pre_close']
            limit_down_price = get_limit_down_price(pre_close_t2, ts_code)
            
            open_price = trade['open']
            high_price = trade['high']
            low_price = trade['low']
            close_price = trade['close']
            
            is_cannot_sell_t2 = high_price == limit_down_price
            
            if is_cannot_sell_t2:
                cannot_sell_trades += 1
                sell_date, sell_price = find_sellable_date(date_t2, ts_code)
                
                if sell_date is not None and sell_price is not None:
                    delayed_sell_trades += 1
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
        equity.append({'date': int_to_date(date_t2), 'nav': capital})
    
    total_ret = capital / initial_cap - 1
    years = len(equity) / 252.0
    ann_ret = (1 + total_ret) ** (1 / years) - 1 if years > 0 else 0
    eq_df = pd.DataFrame(equity)
    df_ret = eq_df['nav'].pct_change()
    mdd = ((eq_df['nav'] - eq_df['nav'].cummax()) / eq_df['nav'].cummax()).min()
    vol = df_ret.std() * np.sqrt(252)
    sharpe = ann_ret / vol if vol > 0 else 0
    
    print("\n" + "="*80)
    print("回测结果（连续跌停处理版）")
    print("="*80)
    print(f"总交易数: {total_trades}")
    print(f"其中 T+2 全天跌停（延迟卖）: {cannot_sell_trades}")
    print(f"  - 成功延迟卖出: {delayed_sell_trades}")
    print(f"总收益: {total_ret:+.2%}, 年化: {ann_ret:+.2%}")
    print(f"夏普: {sharpe:.2f}, 回撤: {mdd:.2%}")
    print(f"交易天数: {len(equity)}, 最终资金: ¥{capital:,.2f}")
    print("="*80)
    
    eq_df.to_csv(os.path.join(OUTPUT_DIR, 'backtest_consecutive_limit_down_equity.csv'), index=False)
    
    plt.figure(figsize=(16, 10))
    plt.plot(eq_df['date'], eq_df['nav'], label='无止损 + 连续跌停处理', linewidth=2)
    plt.title('策略净值曲线（连续跌停处理版）', fontsize=16, fontweight='bold')
    plt.xlabel('日期', fontsize=14)
    plt.ylabel('资金', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'backtest_consecutive_limit_down.png'), dpi=150)
    print(f"\n净值图已保存: {OUTPUT_DIR}/backtest_consecutive_limit_down.png")

if __name__ == "__main__":
    main()


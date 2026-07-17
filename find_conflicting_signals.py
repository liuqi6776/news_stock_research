
import os
import sys
import pandas as pd
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, 'results_duobao')

def main():
    trades_csv = os.path.join(OUTPUT_DIR, 'preloaded_trades.csv')
    
    if not os.path.exists(trades_csv):
        print("未找到预加载文件")
        return
    
    trades_df = pd.read_csv(trades_csv)
    print(f"已加载 {len(trades_df)} 条交易数据")
    
    take_profit = 0.08
    stop_loss_levels = [-0.02, -0.04]
    
    conflicting_signals = []
    
    print("\n" + "="*80)
    print("查找同一天触发止盈和止损的股票")
    print("="*80)
    
    for _, trade in trades_df.iterrows():
        ts_code = trade['ts_code']
        date_t2 = trade['date_t2']
        buy_price = trade['open']
        high_price = trade['high']
        low_price = trade['low']
        
        # 检查是否触发止盈
        hit_take_profit = high_price >= buy_price * (1 + take_profit)
        
        # 检查各个止损点
        for stop_loss_pct in stop_loss_levels:
            hit_stop_loss = low_price <= buy_price * (1 + stop_loss_pct)
            
            if hit_take_profit and hit_stop_loss:
                conflicting_signals.append({
                    'date': date_t2,
                    'ts_code': ts_code,
                    'stop_loss_pct': int(stop_loss_pct * 100),
                    'buy_price': buy_price,
                    'high_price': high_price,
                    'low_price': low_price,
                    'take_profit_level': buy_price * (1 + take_profit),
                    'stop_loss_level': buy_price * (1 + stop_loss_pct)
                })
    
    print(f"\n找到 {len(conflicting_signals)} 条冲突信号记录")
    
    # 保存到CSV
    if conflicting_signals:
        result_df = pd.DataFrame(conflicting_signals)
        result_df = result_df.sort_values(['date', 'ts_code'])
        
        output_csv = os.path.join(OUTPUT_DIR, 'conflicting_signals.csv')
        result_df.to_csv(output_csv, index=False, encoding='utf-8-sig')
        print(f"\n冲突信号已保存至: {output_csv}")
        
        print("\n" + "="*80)
        print("冲突信号统计")
        print("="*80)
        print(f"总冲突记录: {len(result_df)}")
        
        for stop_loss in [-2, -4]:
            count = len(result_df[result_df['stop_loss_pct'] == stop_loss])
            print(f"止损 {stop_loss}%: {count} 条")
        
        print("\n前10条记录:")
        print(result_df.head(10).to_string(index=False))
        print("="*80)
    else:
        print("\n未找到冲突信号")

if __name__ == "__main__":
    main()



import os
import sys
import pandas as pd
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, 'results_duobao')

def main():
    # 读取冲突信号文件
    conflict_file = os.path.join(OUTPUT_DIR, 'conflicting_signals_simple.csv')
    if not os.path.exists(conflict_file):
        print("未找到冲突信号文件")
        return
    
    conflict_df = pd.read_csv(conflict_file)
    print(f"已加载 {len(conflict_df)} 条冲突信号记录")
    
    # 读取预加载交易数据获取买入价
    trades_file = os.path.join(OUTPUT_DIR, 'preloaded_trades.csv')
    if not os.path.exists(trades_file):
        print("未找到预加载交易文件")
        return
    
    trades_df = pd.read_csv(trades_file)
    print(f"已加载 {len(trades_df)} 条交易数据")
    
    # 为冲突信号添加买入价信息
    conflict_with_buy = []
    
    for idx, row in conflict_df.iterrows():
        date = row['date']
        ts_code = row['ts_code']
        
        # 在预加载交易数据中查找对应的记录
        match = trades_df[
            (trades_df['date_t2'] == date) &
            (trades_df['ts_code'] == ts_code)
        ]
        
        if not match.empty:
            buy_price = match.iloc[0]['open']
            high_price = match.iloc[0]['high']
            low_price = match.iloc[0]['low']
            conflict_with_buy.append({
                'date': date,
                'ts_code': ts_code,
                'stop_loss_pct': row['stop_loss_pct'],
                'buy_price': buy_price,
                'high_price': high_price,
                'low_price': low_price
            })
    
    result_df = pd.DataFrame(conflict_with_buy)
    print(f"\n成功匹配 {len(result_df)} 条记录")
    
    # 保存临时文件
    temp_file = os.path.join(OUTPUT_DIR, 'conflicting_signals_with_buy.csv')
    result_df.to_csv(temp_file, index=False, encoding='utf-8-sig')
    print(f"\n临时文件已保存至: {temp_file}")
    
    print("\n" + "="*80)
    print("前10条记录：")
    print("="*80)
    print(result_df.head(10).to_string(index=False))

if __name__ == "__main__":
    main()



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
    stop_loss_list = [-0.02, -0.04, -0.06, -0.08, -0.10, -0.12]
    
    print("\n" + "="*80)
    print("分析同一天内既触及止损又触及止盈的股票数量")
    print("="*80)
    
    for sl in stop_loss_list:
        total = 0
        both_hit = 0
        
        for _, trade in trades_df.iterrows():
            buy_price = trade['open']
            
            hit_stop = trade['low'] &lt;= buy_price * (1 + sl)
            hit_profit = trade['high'] &gt;= buy_price * (1 + take_profit)
            
            total += 1
            if hit_stop and hit_profit:
                both_hit += 1
        
        pct = both_hit / total * 100
        print(f"\n止损 {sl*100:.0f}%:")
        print(f"  总交易数: {total}")
        print(f"  同一天内既触及止损又触及止盈: {both_hit} ({pct:.1f}%)")
        print(f"  只触及止损: {total - both_hit}")
    
    print("\n" + "="*80)
    print("说明")
    print("="*80)
    print("对于既触及止损又触及止盈的股票：")
    print("- 乐观假设：先触及止盈，后触及止损 → 在止盈价卖出")
    print("- 保守假设：先触及止损，后触及止盈 → 在止损价卖出")
    print("- 真实结果：介于两者之间")
    print("="*80)

if __name__ == "__main__":
    main()



import os
import sys
import pandas as pd
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, 'results_duobao')

def main():
    input_csv = os.path.join(OUTPUT_DIR, 'conflicting_signals.csv')
    
    if not os.path.exists(input_csv):
        print("未找到冲突信号文件")
        return
    
    df = pd.read_csv(input_csv)
    print(f"已加载 {len(df)} 条冲突信号记录")
    
    # 创建简化版本
    simple_df = df[['date', 'ts_code', 'stop_loss_pct']].copy()
    
    # 去重（同一股票同一天可能有多个止损点）
    simple_df = simple_df.drop_duplicates()
    
    # 保存
    output_csv = os.path.join(OUTPUT_DIR, 'conflicting_signals_simple.csv')
    simple_df.to_csv(output_csv, index=False, encoding='utf-8-sig')
    
    print(f"\n简化版本已保存至: {output_csv}")
    print(f"包含 {len(simple_df)} 条去重记录")
    
    print("\n" + "="*80)
    print("简化冲突信号（前20条）")
    print("="*80)
    print(simple_df.head(20).to_string(index=False))
    print("="*80)

if __name__ == "__main__":
    main()



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
    
    print("\n" + "="*80)
    print("股票代码模式分析")
    print("="*80)
    
    ts_codes = trades_df['ts_code'].unique()
    print(f"\n唯一股票数: {len(ts_codes)}")
    
    patterns = {
        '300xxx（创业板）': 0,
        '301xxx（创业板新股）': 0,
        '000xxx（深市主板）': 0,
        '001xxx（深市主板新股）': 0,
        '002xxx（中小板）': 0,
        '600xxx（沪市主板）': 0,
        '601xxx（沪市主板）': 0,
        '603xxx（沪市主板）': 0,
        '605xxx（沪市主板新股）': 0,
        '688xxx（科创板）': 0,
        '689xxx（科创板）': 0,
        '其他': 0
    }
    
    for code in ts_codes:
        if code.startswith('300'):
            patterns['300xxx（创业板）'] += 1
        elif code.startswith('301'):
            patterns['301xxx（创业板新股）'] += 1
        elif code.startswith('000'):
            patterns['000xxx（深市主板）'] += 1
        elif code.startswith('001'):
            patterns['001xxx（深市主板新股）'] += 1
        elif code.startswith('002'):
            patterns['002xxx（中小板）'] += 1
        elif code.startswith('600'):
            patterns['600xxx（沪市主板）'] += 1
        elif code.startswith('601'):
            patterns['601xxx（沪市主板）'] += 1
        elif code.startswith('603'):
            patterns['603xxx（沪市主板）'] += 1
        elif code.startswith('605'):
            patterns['605xxx（沪市主板新股）'] += 1
        elif code.startswith('688'):
            patterns['688xxx（科创板）'] += 1
        elif code.startswith('689'):
            patterns['689xxx（科创板）'] += 1
        else:
            patterns['其他'] += 1
            print(f"  其他: {code}")
    
    print("\n统计:")
    print("="*80)
    for key, count in patterns.items():
        print(f"{key}: {count}")
    
    print("\n" + "="*80)
    print("潜在新上市股票代码模式:")
    print("  - 301xxx（创业板新股）")
    print("  - 001xxx（深市主板新股）")
    print("  - 605xxx（沪市主板新股）")
    print("="*80)

if __name__ == "__main__":
    main()


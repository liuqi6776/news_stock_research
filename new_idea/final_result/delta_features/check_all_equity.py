import pandas as pd
import os

doubao_dir = r'c:\Users\liuqi\quant_system_v2\new_idea\final_result\doubao'
for f in ['equity.csv', 'equity_No TP.csv']:
    path = os.path.join(doubao_dir, f)
    if os.path.exists(path):
        e = pd.read_csv(path)
        final_nav = e['nav'].iloc[-1]
        print(f'{f}: {len(e)} rows, Final NAV={final_nav:.2f}, Total={(final_nav/100000-1):.2%}')
        print(f'  First: {e.head(2).to_dict("records")}')
        print(f'  Last: {e.tail(2).to_dict("records")}')
        print()

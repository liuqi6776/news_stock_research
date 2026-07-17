import pandas as pd
e = pd.read_csv(r'c:\Users\liuqi\quant_system_v2\new_idea\final_result\doubao\doubao_equity.csv')
print(f'Equity: {len(e)} rows')
print(e.head(5))
print()
print(e.tail(5))
final_nav = e['nav'].iloc[-1]
print(f'\nFinal NAV: {final_nav:.2f}')
print(f'Total return: {(final_nav/100000-1):.2%}')

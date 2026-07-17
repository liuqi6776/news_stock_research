import pandas as pd
import numpy as np

def analyze_trades(filename):
    df = pd.read_csv(filename)
    # The summary uses Initial Capital = 100,000
    initial_capital = 100000
    
    # Calculate NAV curve
    # Each SELL trade adds 'pnl' to the capital.
    # Note: Commission and stamp duty are already accounted for in 'pnl' by the backtest script.
    sell_trades = df[df['action'] == 'SELL'].copy()
    sell_trades['cumulative_pnl'] = sell_trades['pnl'].cumsum()
    sell_trades['nav'] = initial_capital + sell_trades['cumulative_pnl']
    
    final_nav = sell_trades.iloc[-1]['nav'] if not sell_trades.empty else initial_capital
    total_return = (final_nav - initial_capital) / initial_capital
    
    # Max Drawdown
    sell_trades['max_nav'] = sell_trades['nav'].cummax()
    sell_trades['drawdown'] = (sell_trades['max_nav'] - sell_trades['nav']) / sell_trades['max_nav']
    max_dd = sell_trades['drawdown'].max()
    
    # Win Rate
    win_rate = len(sell_trades[sell_trades['pnl'] > 0]) / len(sell_trades) if len(sell_trades) > 0 else 0
    
    print(f"--- Analysis for {filename} ---")
    print(f"Total Trades: {len(sell_trades)}")
    print(f"Final NAV:    {final_nav:,.2f}")
    print(f"Total Return: {total_return*100:.2f}%")
    print(f"Max Drawdown: {max_dd*100:.2f}%")
    print(f"Win Rate:     {win_rate*100:.2f}%")
    print("-" * 30)

if __name__ == "__main__":
    analyze_trades('trades_open_open.csv')
    analyze_trades('trades_close_close.csv')

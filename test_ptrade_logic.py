import sys
import pandas as pd
import numpy as np
from datetime import datetime

# Add project root to path
sys.path.insert(0, r'C:\Users\liuqi\quant_system_v2')

from run_backtest_v5_ptrade import PtradeEmulationBacktest

class TestBacktest(PtradeEmulationBacktest):
    def test_trailing_stop(self):
        print(">>> Testing Trailing Stop Logic...")
        # Setup
        self.capital = 1000000
        self.positions = {
            '000001.SZ': {
                'shares': 1000,
                'buy_price': 10.0,
                'buy_date': '20230101',
                'max_price': 10.0
            }
        }
        self.stop_loss = 0.08
        self.take_profit = 0.25
        
        # Scenario 1: Price goes up, max_price updates
        # Day 1: High 11.0 (Limit Up). Close 11.0. 
        # Max Price should become 11.0. No sell.
        day1_data = pd.DataFrame([{
            'ts_code': '000001.SZ',
            'trade_date': '20230102',
            'open': 10.5, 'high': 11.0, 'low': 10.5, 'close': 11.0, 
            'pct_chg': 10.0, 'prob': 0.9
        }])
        self._run_daily_logic(day1_data, '20230102')
        
        pos = self.positions.get('000001.SZ')
        assert pos is not None, "Position should be held"
        assert pos['max_price'] == 11.0, f"Max price should be 11.0, got {pos['max_price']}"
        print("Pass Day 1: Max Price Updated")
        
        # Scenario 2: Price drops but not enough (< 5%)
        # Max 11.0. Threshold = 11.0 * 0.95 = 10.45.
        # Day 2: Low 10.5. Close 10.6. 
        # Should Hold.
        day2_data = pd.DataFrame([{
            'ts_code': '000001.SZ',
            'trade_date': '20230103',
            'open': 10.8, 'high': 10.9, 'low': 10.5, 'close': 10.6,
            'pct_chg': -3.6, 'prob': 0.8
        }])
        self._run_daily_logic(day2_data, '20230103')
        assert '000001.SZ' in self.positions, "Should still hold"
        print("Pass Day 2: Held through dip")

        # Scenario 3: Price drops > 5% from max
        # Max 11.0. Threshold 10.45.
        # Day 3: Low 10.4. Close 10.4.
        # Should Sell.
        day3_data = pd.DataFrame([{
            'ts_code': '000001.SZ',
            'trade_date': '20230104',
            'open': 10.5, 'high': 10.5, 'low': 10.4, 'close': 10.4,
            'pct_chg': -1.9, 'prob': 0.7
        }])
        self._run_daily_logic(day3_data, '20230104')
        assert '000001.SZ' not in self.positions, "Should have sold"
        last_trade = self.trades[-1]
        assert last_trade['action'] == 'SELL', "Action must be SELL"
        assert "TRAILING_STOP" in last_trade['info'], f"Reason must be TRAILING_STOP, got {last_trade['info']}"
        print(f"Pass Day 3: Sold on Trailing Stop. Info: {last_trade['info']}")

    def _run_daily_logic(self, day_data, current_date):
        # Helper to simulate the loop body relative to selling
        # Mock target_list (always include to avoid SIGNAL_LOST)
        target_list = ['000001.SZ'] 
        
        # --- Copied/Adapted from run_backtest_v5_ptrade.py Selling Logic ---
        day_data_dict = day_data.set_index('ts_code').to_dict('index')

        for ts_code in list(self.positions.keys()):
            if ts_code in day_data_dict:
                current_high = day_data_dict[ts_code]['high']
                # Init max_price if not set (though in real code it is init on buy, here we mock it)
                if 'max_price' not in self.positions[ts_code]:
                    self.positions[ts_code]['max_price'] = self.positions[ts_code]['buy_price']
                    
                self.positions[ts_code]['max_price'] = max(self.positions[ts_code]['max_price'], current_high)
        
        executed_sells = []
        for ts_code in list(self.positions.keys()):
            if ts_code not in day_data_dict: continue
            
            row = day_data_dict[ts_code]
            price = row['close']
            low = row['low']
            high = row['high']
            pos = self.positions[ts_code]
            
            if row['pct_chg'] < -9.5: continue

            # 1. Stop Loss -8%
            if low <= pos['buy_price'] * (1 - self.stop_loss):
                self._sell(ts_code, pos['buy_price'] * (1 - self.stop_loss), current_date, "STOP_LOSS")
                executed_sells.append(ts_code)
                continue
                
            # 2. Take Profit +25%
            if high >= pos['buy_price'] * (1 + self.take_profit):
                self._sell(ts_code, pos['buy_price'] * (1 + self.take_profit), current_date, "TAKE_PROFIT")
                executed_sells.append(ts_code)
                continue
            
            # 3. Trailing Stop -5%
            if low <= pos['max_price'] * 0.95:
                sell_price = max(row['open'], pos['max_price'] * 0.95)
                self._sell(ts_code, sell_price, current_date, "TRAILING_STOP")
                executed_sells.append(ts_code)
                continue

            # 4. Signal Lost
            if ts_code not in target_list:
                self._sell(ts_code, price, current_date, "SIGNAL_LOST")
                continue
            
            # 5. Prob Drop
            if row['prob'] < 0.50:
                 self._sell(ts_code, price, current_date, "PROB_DROP")
                 continue

if __name__ == "__main__":
    tester = TestBacktest()
    tester.test_trailing_stop()

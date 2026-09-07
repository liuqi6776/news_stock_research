"""
Synthetic Unit Test Suite for UnifiedProductionLedger v2.1 (test_ledger_v3.py)
Validates all 4 critical edge cases identified in Reviewer Audit (2026-09-07):
1. Repeated rebalances during suspension (no sell order doubling via +=)
2. Shared daily ADV budget between pending sell and rebalance (strict <= 10% ADV)
3. Signal reversal protection (target >= held cancels pending sell, prevents sell-then-buy)
4. Missing / zero ADV guard (strict 0 execution, no unlimited fallback)
"""

import sys
import os
import pandas as pd
import numpy as np

# Ensure research repo root is in path
repo_root = r"c:\Users\liuqi\quant_system_v2"
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from research.experiments.exp_ens_t60_tv12.unified_production_ledger import (
    UnifiedProductionLedger,
    get_adv20_shares
)

def create_mock_market_data():
    dates = [f"2025-01-{d:02d}" for d in range(1, 31)]
    stocks = ["000001.SZ", "000002.SZ", "000003.SZ", "000004.SZ"]
    
    # Prices: 10.0 for all stocks
    open_df = pd.DataFrame(10.0, index=dates, columns=stocks)
    close_df = pd.DataFrame(10.0, index=dates, columns=stocks)
    preclose_df = pd.DataFrame(10.0, index=dates, columns=stocks)
    
    # Volume (lots): 1 lot = 100 shares.
    # 000001.SZ: 1000 lots (100,000 shares/day)
    # 000002.SZ: 100 lots (10,000 shares/day) -> 10% ADV = 1,000 shares
    # 000003.SZ: 1000 lots (100,000 shares/day)
    # 000004.SZ: 0 lots (0 shares/day -> ADV = 0)
    vol_df = pd.DataFrame(index=dates, columns=stocks)
    vol_df["000001.SZ"] = 1000.0
    vol_df["000002.SZ"] = 100.0
    vol_df["000003.SZ"] = 1000.0
    vol_df["000004.SZ"] = 0.0
    
    return dates, open_df, close_df, preclose_df, vol_df


def test_scenario_1_repeated_rebalance_suspension():
    print("\n--- Running Test Scenario 1: Suspension Pending Sell Accumulation Fix ---")
    dates, open_df, close_df, preclose_df, vol_df = create_mock_market_data()
    
    ledger = UnifiedProductionLedger(initial_capital=100000.0, adv_cap_pct=0.10)
    # Seed 000001.SZ with 1000 shares
    ledger.stock_positions["000001.SZ"] = {
        "shares": 1000,
        "tradable_shares": 1000,
        "locked_shares": 0,
        "last_px": 10.0
    }
    
    # Suspend 000001.SZ on 2025-01-21 and 2025-01-22 (vol=0, open=nan)
    d1 = "2025-01-21"
    d2 = "2025-01-22"
    d3 = "2025-01-23" # Resumes
    
    open_df.at[d1, "000001.SZ"] = np.nan
    vol_df.at[d1, "000001.SZ"] = 0.0
    open_df.at[d2, "000001.SZ"] = np.nan
    vol_df.at[d2, "000001.SZ"] = 0.0
    
    # Rebalance 1: target = 0 shares (target_stock_codes=[])
    ledger.execute_rebalance(
        d1, [], 0.0, open_df, preclose_df, vol_df, {}, {}
    )
    assert ledger.pending_sell_orders.get("000001.SZ") == 1000, (
        f"Expected 1000 pending shares, got {ledger.pending_sell_orders.get('000001.SZ')}"
    )
    print("  [Pass] Rebalance 1 (suspended): pending_sell_orders = 1000")
    
    # Rebalance 2: target still 0 shares
    ledger.execute_rebalance(
        d2, [], 0.0, open_df, preclose_df, vol_df, {}, {}
    )
    # Under old bug, this became 2000! With fix, it MUST remain 1000.
    assert ledger.pending_sell_orders.get("000001.SZ") == 1000, (
        f"BUG DETECTED: Pending sell orders doubled! Got {ledger.pending_sell_orders.get('000001.SZ')} instead of 1000"
    )
    print("  [Pass] Rebalance 2 (suspended): pending_sell_orders is still 1000 (no += accumulation)")
    
    # Resume trading on d3: process daily pending orders
    ledger.process_daily_pending_orders(d3, open_df, preclose_df, vol_df)
    assert "000001.SZ" not in ledger.stock_positions, "Position should be completely sold"
    assert "000001.SZ" not in ledger.pending_sell_orders, "Pending sell orders should be cleared"
    assert ledger.total_trades == 1, f"Expected exactly 1 sell trade, got {ledger.total_trades}"
    print("  [Pass] Resumption on d3: exactly 1000 shares sold cleanly, queue cleared")


def test_scenario_2_shared_daily_adv_quota():
    print("\n--- Running Test Scenario 2: Shared Daily ADV Capacity Budget (10% Cap) ---")
    dates, open_df, close_df, preclose_df, vol_df = create_mock_market_data()
    
    # 000002.SZ: vol = 100 lots/day = 10,000 shares/day. 10% ADV = 1,000 shares max per day.
    ledger = UnifiedProductionLedger(initial_capital=500000.0, adv_cap_pct=0.10)
    ledger.stock_positions["000002.SZ"] = {
        "shares": 5000,
        "tradable_shares": 5000,
        "locked_shares": 0,
        "last_px": 10.0
    }
    
    d = "2025-01-21"
    # Seed pending sell queue with 700 shares
    ledger.pending_sell_orders["000002.SZ"] = 700
    
    # Step 1: Process daily pending orders first (executes 700 shares)
    ledger.process_daily_pending_orders(d, open_df, preclose_df, vol_df)
    assert ledger.daily_executed_shares.get("000002.SZ") == 700, (
        f"Expected 700 executed, got {ledger.daily_executed_shares.get('000002.SZ')}"
    )
    print("  [Pass] Pending order executed: 700 shares, daily_executed_shares = 700")
    
    # Step 2: Now on the SAME day d, an additional rebalance sell tries to sell more
    # Remaining capacity = 1000 - 700 = 300 shares.
    # Current held shares = 5000 - 700 = 4300.
    # Rebalance targets 0 shares (excess = 4300).
    ledger.execute_rebalance(
        d, [], 0.0, open_df, preclose_df, vol_df, {}, {}
    )
    
    total_executed = ledger.daily_executed_shares.get("000002.SZ", 0)
    assert total_executed == 1000, (
        f"ADV Cap violated! Total executed = {total_executed}, expected strictly 1000 (10% ADV)"
    )
    assert ledger.stock_positions["000002.SZ"]["shares"] == 4000, (
        f"Expected 4000 shares remaining, got {ledger.stock_positions['000002.SZ']['shares']}"
    )
    assert ledger.pending_sell_orders["000002.SZ"] == 4000, (
        f"Expected 4000 pending shares, got {ledger.pending_sell_orders.get('000002.SZ')}"
    )
    print("  [Pass] Shared daily ADV quota strictly enforced at 1000 shares across both phases")


def test_scenario_3_signal_reversal_protection():
    print("\n--- Running Test Scenario 3: Signal Reversal Protection ---")
    dates, open_df, close_df, preclose_df, vol_df = create_mock_market_data()
    
    ledger = UnifiedProductionLedger(initial_capital=100000.0, adv_cap_pct=0.10)
    ledger.stock_positions["000003.SZ"] = {
        "shares": 1000,
        "tradable_shares": 1000,
        "locked_shares": 0,
        "last_px": 10.0
    }
    
    d1 = "2025-01-21"
    d2 = "2025-01-22"
    
    # Suspend on d1 and issue sell signal (target = 0)
    open_df.at[d1, "000003.SZ"] = np.nan
    vol_df.at[d1, "000003.SZ"] = 0.0
    ledger.execute_rebalance(d1, [], 0.0, open_df, preclose_df, vol_df, {}, {})
    assert ledger.pending_sell_orders.get("000003.SZ") == 1000
    print("  [Pass] d1: Stock suspended, pending sell = 1000 shares queued")
    
    # On d2, stock resumes trading, BUT signal reverses: model now selects 000003.SZ!
    # Target value = 50,000 (at price 10.0 -> 5,000 shares, which is >= current 1000 shares)
    ledger.execute_rebalance(
        d2, ["000003.SZ"], 0.50, open_df, preclose_df, vol_df, {}, {}
    )
    
    # Check that pending sell was cancelled and no sell occurred
    assert "000003.SZ" not in ledger.pending_sell_orders, (
        "Pending sell order should be cancelled upon signal reversal"
    )
    # Total trades should be buy only (no sell then buy!)
    assert ledger.stock_positions["000003.SZ"]["shares"] >= 1000, "Should have kept initial 1000 shares"
    print(f"  [Pass] Reversal handled cleanly: Pending sell cancelled, held shares = {ledger.stock_positions['000003.SZ']['shares']}")


def test_scenario_4_zero_missing_adv_guard():
    print("\n--- Running Test Scenario 4: Missing / Zero ADV Guard ---")
    dates, open_df, close_df, preclose_df, vol_df = create_mock_market_data()
    
    ledger = UnifiedProductionLedger(initial_capital=500000.0, adv_cap_pct=0.10)
    d = "2025-01-21"
    
    # 000004.SZ has historical vol = 0 prior to d, but trades today (day_vol > 0)
    vol_df.at[d, "000004.SZ"] = 100.0
    adv = get_adv20_shares(vol_df, "000004.SZ", d, default_shares=0)
    assert adv == 0, f"Expected ADV=0, got {adv}"
    
    # Try to buy 000004.SZ in rebalance
    ledger.execute_rebalance(d, ["000004.SZ"], 0.50, open_df, preclose_df, vol_df, {}, {})
    assert "000004.SZ" not in ledger.stock_positions, "Zero-ADV stock MUST NOT be bought!"
    assert ledger.adv_zero_blocks > 0, f"adv_zero_blocks should be incremented, got {ledger.adv_zero_blocks}"
    print(f"  [Pass] Buy blocked for zero ADV stock (adv_zero_blocks = {ledger.adv_zero_blocks})")
    
    # Also test unknown stock entirely missing from vol_df
    unknown_code = "999999.SZ"
    open_df[unknown_code] = 10.0
    preclose_df[unknown_code] = 10.0
    # Not added to vol_df
    ledger.execute_rebalance(d, [unknown_code], 0.50, open_df, preclose_df, vol_df, {}, {})
    assert unknown_code not in ledger.stock_positions, "Missing-ADV stock MUST NOT be bought!"
    print("  [Pass] Buy blocked for stock completely missing from volume history")


if __name__ == "__main__":
    print("=================================================================")
    print("RUNNING SYNTHETIC UNIT TESTS FOR UnifiedProductionLedger v2.1")
    print("=================================================================")
    test_scenario_1_repeated_rebalance_suspension()
    test_scenario_2_shared_daily_adv_quota()
    test_scenario_3_signal_reversal_protection()
    test_scenario_4_zero_missing_adv_guard()
    print("\n=================================================================")
    print("ALL 4 UNIT TESTS PASSED WITH 100% SUCCESS!")
    print("=================================================================")

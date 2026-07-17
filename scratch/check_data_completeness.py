import os
import pandas as pd
from datetime import datetime

DATA_PATH = r"D:\iquant_data\data_v2"
subdirs = ['data_day1', 'ths_rank1', 'other_day1', 'cyq1']

def check_completeness():
    print(f"Data root: {DATA_PATH}")
    if not os.path.exists(DATA_PATH):
        print("Error: DATA_PATH does not exist!")
        return

    dir_dates = {}
    for sub in subdirs:
        path = os.path.join(DATA_PATH, sub)
        if not os.path.exists(path):
            print(f"Directory {sub} does not exist!")
            continue
        
        files = [f for f in os.listdir(path) if f.endswith('.parquet')]
        dates = sorted([f.replace('.parquet', '') for f in files])
        dir_dates[sub] = set(dates)
        
        if dates:
            print(f"Subdir: {sub:15s} | File Count: {len(dates):4d} | Min Date: {dates[0]} | Max Date: {dates[-1]}")
        else:
            print(f"Subdir: {sub:15s} | Empty")

    # Let's find trading days from data_day1 as the baseline
    if 'data_day1' in dir_dates:
        baseline_dates = sorted(list(dir_dates['data_day1']))
        print(f"\nTotal trading days in baseline (data_day1): {len(baseline_dates)}")
        
        # Check if other subdirs are missing any of these dates
        for sub in subdirs:
            if sub == 'data_day1' or sub not in dir_dates:
                continue
            missing = sorted(list(dir_dates['data_day1'] - dir_dates[sub]))
            if missing:
                print(f"Subdir {sub} is missing {len(missing)} dates relative to data_day1:")
                if len(missing) <= 10:
                    print(f"  {missing}")
                else:
                    print(f"  {missing[:5]} ... {missing[-5:]}")
            else:
                print(f"Subdir {sub} has all dates in data_day1.")

if __name__ == "__main__":
    check_completeness()

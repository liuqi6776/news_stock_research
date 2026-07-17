import sys
import os
import pandas as pd
from tqdm import tqdm

# Add current dir to path
sys.path.append(os.getcwd())

from infra_data.fetcher import DataFetcher
from config.settings import settings

def download_full_macro_data():
    fetcher = DataFetcher()
    start_date = "20200101"
    end_date = "20260318"
    
    print(f"Downloading VIX data from {start_date} to {end_date}...")
    fetcher.fetch_vix_data(start_date, end_date)
    
    print(f"Downloading Margin data from {start_date} to {end_date}...")
    # Get all trading dates first
    dates = fetcher.get_trading_dates(start_date, end_date)
    fetcher.fetch_margin_data(dates)
    
    print("Download complete.")

if __name__ == "__main__":
    download_full_macro_data()

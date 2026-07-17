"""
Diagnostic Script for news_major1 processing and filtering
"""
import os
import sys
import pandas as pd
import json

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from panqian_processor import process_panqian_news

DATA_DIR  = r'D:\iquant_data\data_v2'
NEWS_DIR  = os.path.join(DATA_DIR, 'news_major1')
INDUSTRY_MAP = r'c:\Users\liuqi\quant_system_v2\stock_industry_map_cached.parquet'

def diag_news():
    print("--- Diagnostic: Panqian News (news_major1) ---")
    news_mkt, news_stk = process_panqian_news(NEWS_DIR, '20240101', '20260327', INDUSTRY_MAP)
    
    if news_stk.empty:
        print("Error: news_stk is EMPTY!")
        return
        
    print(f"Total News Records: {len(news_stk)}")
    print(f"Sample Records:\n{news_stk.head()}")
    
    # Check score distribution
    print("\nStock Impact Distribution:")
    print(news_stk['news_stock_impact'].value_counts())
    
    print("\nSector Impact Distribution:")
    print(news_stk['news_sector_impact'].value_counts())
    
    # Check date coverage
    news_stk['date_str'] = news_stk['trade_date'].dt.strftime('%Y%m%d')
    unique_dates = news_stk['date_str'].nunique()
    print(f"\nUnique News Dates: {unique_dates}")
    
    # Check if we have records with score >= 1
    high_score = news_stk[(news_stk['news_stock_impact'] >= 1) | (news_stk['news_sector_impact'] >= 1)]
    print(f"Records with Impact >= 1: {len(high_score)}")

if __name__ == "__main__":
    diag_news()

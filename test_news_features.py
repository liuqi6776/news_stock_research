import pandas as pd
from processing.pipeline import DataPipeline
from processing.merger import merge_dataframes

def test_pipeline():
    pipeline = DataPipeline()
    print("Loading data from 2025-10-25 to 2025-10-31...")
    dfs = pipeline.load_all_data(start_date='20251025', end_date='20251031')
    
    news_market = dfs[-2]
    news_stock = dfs[-1]
    
    print("\nnews_market_df:")
    if news_market.empty:
        print("Empty DataFrame")
    else:
        print(news_market)
        
    print("\nnews_stock_sector_df:")
    if news_stock.empty:
        print("Empty DataFrame")
    else:
        print(news_stock.head(15))
        
    print("\nTesting merge_dataframes on news DFs and data_all...")
    data_all = dfs[0]
    
    # Just take a subset to merge quickly
    valid_stocks = ['000001.SZ', '600715.SH', '603019.SH', '002377.SZ']
    data_subset = data_all[data_all['ts_code'].isin(valid_stocks)].copy()
    
    try:
        merged = merge_dataframes([data_subset, news_market, news_stock])
        print("\nMerged columns:", merged.columns.tolist())
        print("Sample merged rows:")
        print(merged[['trade_date', 'ts_code', 'close', 'news_market_impact', 'news_stock_impact', 'news_sector_impact']].dropna(subset=['news_market_impact']).head(10))
    except Exception as e:
        print("Merge error:", e)

if __name__ == '__main__':
    test_pipeline()

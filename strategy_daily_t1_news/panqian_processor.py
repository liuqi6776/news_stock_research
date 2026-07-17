"""
Specialized News Processor for news_major1 (Panqian Jiyao)
- Logic: article_date is the date for which we want to predict.
- Features for that prediction should come from article_date - 1 (the last trading day).
"""
import os
import json
import pandas as pd
from typing import Tuple

def process_panqian_news(
    news_dir: str, 
    start_date: str = None, 
    end_date: str = None, 
    industry_map_path: str = r'c:\Users\liuqi\quant_system_v2\stock_industry_map_cached.parquet'
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if not os.path.exists(news_dir):
        return pd.DataFrame(), pd.DataFrame()
        
    market_records = []
    stock_records = []
    sector_records = []
    
    for filename in os.listdir(news_dir):
        if not filename.endswith('.json'): continue
        filepath = os.path.join(news_dir, filename)
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except: continue
            
        date_str = data.get("article_date", "")
        if not date_str: continue
            
        # This trade_date is the date we are predicting FOR.
        trade_date = pd.to_datetime(date_str)
        date_formatted = trade_date.strftime('%Y%m%d')
        
        if start_date and date_formatted < start_date: continue
        if end_date and date_formatted > end_date: continue
            
        market_impact = data.get("market_impact", 0)
        market_records.append({'trade_date': trade_date, 'news_market_impact': float(market_impact)})
        
        for s in data.get("stocks", []):
            code = s.get("stock_code")
            if not code: continue
            ts_code = f"{code}.SH" if code.startswith('6') else f"{code}.SZ" if (code.startswith('0') or code.startswith('3')) else f"{code}.BJ" if (code.startswith('4') or code.startswith('8')) else code
            stock_records.append({'trade_date': trade_date, 'ts_code': ts_code, 'news_stock_impact': float(s.get("impact", 0))})
            
        for sec in data.get("sectors", []):
            sector_name = sec.get("sector_name")
            if sector_name:
                sector_records.append({'trade_date': trade_date, 'sector_name': sector_name, 'news_sector_impact': float(sec.get("impact", 0))})
                
    news_mkt = pd.DataFrame(market_records)
    news_stk = pd.DataFrame(stock_records)
    news_sec = pd.DataFrame(sector_records)
    
    # Map Sector to TS_Code
    industry_map = pd.read_parquet(industry_map_path) if os.path.exists(industry_map_path) else pd.DataFrame()
    concept_map_path = os.path.join(os.path.dirname(industry_map_path), 'tushare_concept_map_cached.parquet')
    concept_map = pd.read_parquet(concept_map_path) if os.path.exists(concept_map_path) else pd.DataFrame()
    
    mapped_dfs = []
    if not news_sec.empty:
        if not industry_map.empty:
            mapped_ind = pd.merge(news_sec, industry_map, left_on='sector_name', right_on='industry', how='inner')
            mapped_dfs.append(mapped_ind[['trade_date', 'ts_code', 'news_sector_impact']])
        if not concept_map.empty:
            import re
            for _, row in news_sec.iterrows():
                matched = concept_map[concept_map['concept_name'].str.contains(re.escape(row['sector_name']), na=False, case=False)].copy()
                if not matched.empty:
                    matched['trade_date'] = row['trade_date']
                    matched['news_sector_impact'] = row['news_sector_impact']
                    mapped_dfs.append(matched[['trade_date', 'ts_code', 'news_sector_impact']])
    
    if mapped_dfs:
        mapped = pd.concat(mapped_dfs).groupby(['trade_date', 'ts_code'], as_index=False).mean()
    else:
        mapped = pd.DataFrame(columns=['trade_date', 'ts_code', 'news_sector_impact'])
        
    if not news_stk.empty or not mapped.empty:
        final_stk = pd.merge(news_stk, mapped, on=['trade_date', 'ts_code'], how='outer').fillna(0)
    else:
        final_stk = pd.DataFrame(columns=['trade_date', 'ts_code', 'news_stock_impact', 'news_sector_impact'])
        
    return news_mkt, final_stk

"""
News data processor module

Parses JSON files containing news impact evaluations and converts them into DataFrames
suitable for merging into the data pipeline.
"""

import os
import json
import pandas as pd
from typing import Tuple

def load_and_process_news(
    news_dir: str, 
    start_date: str = None, 
    end_date: str = None, 
    industry_map_path: str = 'stock_industry_map_cached.parquet',
    valid_dates: pd.Series = None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Parse news JSONs and return (news_market_df, news_stock_sector_df)
    
    Args:
        news_dir: Directory containing JSON files
        start_date: Start date string (YYYYMMDD)
        end_date: End date string (YYYYMMDD)
        industry_map_path: Path to industry mapping parquet file
        
    Returns:
        Tuple of (Market impact DataFrame, Stock & Sector impact DataFrame)
    """
    if not os.path.exists(news_dir):
        return pd.DataFrame(), pd.DataFrame()
        
    market_records = []
    stock_records = []
    sector_records = []
    
    for filename in os.listdir(news_dir):
        if not filename.endswith('.json'):
            continue
            
        filepath = os.path.join(news_dir, filename)
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception:
            continue
            
        date_str = data.get("article_date", "")
        if not date_str:
            continue
            
        # Convert date "2025-10-28" to pd.Timestamp
        trade_date = pd.to_datetime(date_str)
        date_formatted = trade_date.strftime('%Y%m%d')
        
        if start_date and date_formatted < start_date[:8]:
            continue
        if end_date and date_formatted > end_date[:8]:
            continue
            
        # Market impact
        market_impact = data.get("market_impact", 0)
        market_records.append({
            'trade_date': trade_date, 
            'news_market_impact': float(market_impact)
        })
        
        # Stock impact
        for s in data.get("stocks", []):
            code = s.get("stock_code")
            if not code:
                continue
            
            # Map code to ts_code format
            if code.startswith('6'):
                ts_code = f"{code}.SH"
            elif code.startswith('0') or code.startswith('3'):
                ts_code = f"{code}.SZ"
            elif code.startswith('4') or code.startswith('8'):
                ts_code = f"{code}.BJ"
            else:
                ts_code = code
                
            impact = s.get("impact", 0)
            stock_records.append({
                'trade_date': trade_date, 
                'ts_code': ts_code, 
                'news_stock_impact': float(impact)
            })
            
        # Sector impact
        for sec in data.get("sectors", []):
            sector_name = sec.get("sector_name")
            impact = sec.get("impact", 0)
            if sector_name:
                sector_records.append({
                    'trade_date': trade_date,
                    'sector_name': sector_name,
                    'news_sector_impact': float(impact)
                })
                
    # 1. Market DataFrame
    news_market_df = pd.DataFrame(market_records)
    if not news_market_df.empty:
        if valid_dates is not None and not valid_dates.empty:
            def get_next_trade_date(d):
                future = valid_dates[valid_dates > d]
                return future.iloc[0] if not future.empty else d
            news_market_df['trade_date'] = news_market_df['trade_date'].apply(get_next_trade_date)
        news_market_df = news_market_df.groupby('trade_date', as_index=False).mean()
        
    # Load Industry Map
    industry_map_df = pd.DataFrame()
    if os.path.exists(industry_map_path):
        try:
            industry_map_df = pd.read_parquet(industry_map_path)
            # ensure 'ts_code' and 'industry' columns are present
            if 'ts_code' not in industry_map_df.columns or 'industry' not in industry_map_df.columns:
                industry_map_df = pd.DataFrame()
        except Exception:
            industry_map_df = pd.DataFrame()
            
    # Load Concept Map
    concept_map_path = os.path.join(os.path.dirname(industry_map_path), 'tushare_concept_map_cached.parquet')
    concept_map_df = pd.DataFrame()
    if os.path.exists(concept_map_path):
        try:
            concept_map_df = pd.read_parquet(concept_map_path)
        except Exception:
            pass
            
    # 2. Stock DataFrame
    stock_df = pd.DataFrame(stock_records)
    if not stock_df.empty:
        if valid_dates is not None and not valid_dates.empty:
            def get_next_trade_date(d):
                future = valid_dates[valid_dates > d]
                return future.iloc[0] if not future.empty else d
            stock_df['trade_date'] = stock_df['trade_date'].apply(get_next_trade_date)
        stock_df = stock_df.groupby(['trade_date', 'ts_code'], as_index=False).mean()
    else:
        stock_df = pd.DataFrame(columns=['trade_date', 'ts_code', 'news_stock_impact'])
        
    # 3. Sector DataFrame mapped to stocks
    sector_df = pd.DataFrame(sector_records)
    mapped_dfs = []
    
    if not sector_df.empty:
        if valid_dates is not None and not valid_dates.empty:
            def get_next_trade_date(d):
                future = valid_dates[valid_dates > d]
                return future.iloc[0] if not future.empty else d
            sector_df['trade_date'] = sector_df['trade_date'].apply(get_next_trade_date)
        sector_df = sector_df.groupby(['trade_date', 'sector_name'], as_index=False).mean()
        
        # Match Industry
        if not industry_map_df.empty:
            mapped_ind = pd.merge(sector_df, industry_map_df, left_on='sector_name', right_on='industry', how='inner')
            if not mapped_ind.empty:
                mapped_dfs.append(mapped_ind[['trade_date', 'ts_code', 'news_sector_impact']])
                
        # Match Concepts
        if not concept_map_df.empty:
            import re
            mapped_con_list = []
            for _, row in sector_df.iterrows():
                s_name = row['sector_name']
                if not isinstance(s_name, str) or not s_name: continue
                try:
                    mask = concept_map_df['concept_name'].str.contains(re.escape(s_name), na=False)
                    matched_concepts = concept_map_df[mask]
                    if not matched_concepts.empty:
                        m_merge = matched_concepts.copy()
                        m_merge['trade_date'] = row['trade_date']
                        m_merge['news_sector_impact'] = row['news_sector_impact']
                        mapped_con_list.append(m_merge)
                except Exception:
                    pass
            if mapped_con_list:
                m_con_df = pd.concat(mapped_con_list, ignore_index=True)
                mapped_dfs.append(m_con_df[['trade_date', 'ts_code', 'news_sector_impact']])
                
        if mapped_dfs:
            mapped = pd.concat(mapped_dfs, ignore_index=True)
            mapped = mapped.groupby(['trade_date', 'ts_code'], as_index=False).mean()
        else:
            mapped = pd.DataFrame(columns=['trade_date', 'ts_code', 'news_sector_impact'])
    else:
        mapped = pd.DataFrame(columns=['trade_date', 'ts_code', 'news_sector_impact'])
        
    # Combine Stock and Sector DataFrames
    if not stock_df.empty or not mapped.empty:
        if stock_df.empty:
            news_stock_sector_df = mapped
        elif mapped.empty:
            news_stock_sector_df = stock_df
        else:
            news_stock_sector_df = pd.merge(stock_df, mapped, on=['trade_date', 'ts_code'], how='outer')
            
        # Fill missing sub-impacts where one existed but other didn't
        if 'news_stock_impact' not in news_stock_sector_df.columns:
            news_stock_sector_df['news_stock_impact'] = 0.0
        if 'news_sector_impact' not in news_stock_sector_df.columns:
            news_stock_sector_df['news_sector_impact'] = 0.0
            
        news_stock_sector_df['news_stock_impact'] = news_stock_sector_df['news_stock_impact'].fillna(0.0)
        news_stock_sector_df['news_sector_impact'] = news_stock_sector_df['news_sector_impact'].fillna(0.0)
    else:
        news_stock_sector_df = pd.DataFrame(columns=['trade_date', 'ts_code', 'news_stock_impact', 'news_sector_impact'])
        
    return news_market_df, news_stock_sector_df

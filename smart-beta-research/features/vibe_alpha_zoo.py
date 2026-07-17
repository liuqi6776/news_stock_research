import sys
import os
import glob
import importlib
import pandas as pd
import numpy as np
import random

def calculate_vibe_alphas(df: pd.DataFrame, num_factors: int = 40) -> pd.DataFrame:
    """
    Calculate alpha factors using Vibe-Trading's Alpha Zoo.
    It pivots the dataframe to a wide panel, computes the alphas, and merges them back.
    """
    print("\n" + "="*60)
    print(f"开始计算 Vibe-Trading Alpha Zoo 因子 (选取 {num_factors} 个)")
    print("="*60)
    
    agent_dir = os.path.join(os.path.dirname(__file__), '..', 'vibe_trading_repo', 'agent')
    if agent_dir not in sys.path:
        sys.path.insert(0, agent_dir)

    print("Converting to wide panel format...")
    cols = ['open', 'high', 'low', 'close', 'vol', 'amount', 'vwap']
    
    if 'amount' not in df.columns and 'vol' in df.columns:
        df['amount'] = df['close'] * df['vol'] * 100 
    
    if 'vwap' not in df.columns:
        df['vwap'] = (df['high'] + df['low'] + df['close']) / 3.0

    panel = {}
    for col in cols:
        if col in df.columns:
            panel[col] = df.pivot(index='trade_date', columns='ts_code', values=col)
            
    if 'vol' in panel:
        panel['volume'] = panel['vol']

    alpha_101_files = glob.glob(os.path.join(agent_dir, 'src', 'factors', 'zoo', 'alpha101', 'alpha_*.py'))
    gtja_191_files = glob.glob(os.path.join(agent_dir, 'src', 'factors', 'zoo', 'gtja191', 'alpha_*.py'))
    
    # Deterministic selection for reproducibility
    random.seed(42)
    alpha_101_files.sort()
    gtja_191_files.sort()
    
    num_each = num_factors // 2
    selected_files = random.sample(alpha_101_files, min(num_each, len(alpha_101_files))) + \
                     random.sample(gtja_191_files, min(num_each, len(gtja_191_files)))

    alpha_results = []
    
    for file_path in selected_files:
        # e.g. c:\...\agent\src\factors\zoo\alpha101\alpha_002.py -> src.factors.zoo.alpha101.alpha_002
        rel_path = os.path.relpath(file_path, agent_dir)
        module_name = rel_path.replace('.py', '').replace(os.sep, '.')
        
        try:
            mod = importlib.import_module(module_name)
            if hasattr(mod, 'compute'):
                res = mod.compute(panel)
                alpha_id = getattr(mod, 'ALPHA_ID', module_name.split('.')[-1])
                
                # Convert back to long format
                res_long = res.unstack().reset_index()
                res_long.columns = ['ts_code', 'trade_date', alpha_id]
                alpha_results.append(res_long)
                print(f"  计算完成: {alpha_id}")
        except Exception as e:
            print(f"  计算失败 {module_name}: {e}")

    if not alpha_results:
        print("未计算任何 Vibe Alphas.")
        return df

    print(f"合并 {len(alpha_results)} 个 Vibe 因子到主数据集...")
    from functools import reduce
    
    merged_alphas = reduce(lambda left, right: pd.merge(left, right, on=['ts_code', 'trade_date'], how='outer'), alpha_results)
    df = pd.merge(df, merged_alphas, on=['ts_code', 'trade_date'], how='left')
    
    print("Vibe-Trading Alpha 计算完成!")
    return df

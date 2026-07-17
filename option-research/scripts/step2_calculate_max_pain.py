"""
step2_calculate_max_pain.py
从每日下载的期权数据中计算 50ETF & 300ETF 期权的日度最大痛点价格 (Max Pain Price)。

改进说明：
1. 消除前视偏差：期权持仓量 (OI) 采用 T-1 日收盘的结算持仓量 (oi_lag1) 来计算第 T 日的 Max Pain。
2. 引入 Placebo 检验：同时计算离标的现价最近的行权价 (Placebo Nearest Strike)，以排除 base-rate 幻觉。
3. 交易日到期天数：days_to_expiry 采用实际交易日数量计算，而非日历日天数。
"""
import os
import glob
import pandas as pd
import numpy as np

# 目录配置
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(BASE_DIR, 'data')
DAILY_DIR = os.path.join(DATA_DIR, 'daily')

BASIC_FILE = os.path.join(DATA_DIR, 'opt_basic.parquet')
UNDERLYING_FILE = os.path.join(DATA_DIR, 'underlying_daily.parquet')
OUTPUT_FILE = os.path.join(DATA_DIR, 'max_pain_history.csv')

def calculate_max_pain_for_chain(df_chain, oi_col='oi_lag1'):
    """
    向量化计算某一期权链的最大痛点价格
    """
    calls = df_chain[df_chain['call_put'] == 'C']
    puts = df_chain[df_chain['call_put'] == 'P']
    
    if len(calls) == 0 or len(puts) == 0:
        return np.nan, 0, 0
        
    # 提取行权价与持仓量 (OI)
    call_strikes = calls['exercise_price'].values
    call_oi = calls[oi_col].fillna(0.0).values
    
    put_strikes = puts['exercise_price'].values
    put_oi = puts[oi_col].fillna(0.0).values
    
    # 设定标的价格搜寻格点 S
    all_strikes = np.concatenate([call_strikes, put_strikes])
    min_s = max(0.5, all_strikes.min() - 0.2)
    max_s = all_strikes.max() + 0.2
    
    # 构建 0.005 步长的精细格点
    s_grid = np.arange(min_s, max_s, 0.005)
    
    # 转换为 2D 矩阵进行向量化计算
    S = s_grid[:, np.newaxis]
    
    # 计算 Calls 价值: max(S - K, 0) * Call_OI
    call_payoffs = np.maximum(S - call_strikes[np.newaxis, :], 0.0)
    call_pain = np.dot(call_payoffs, call_oi)
    
    # 计算 Puts 价值: max(K - S, 0) * Put_OI
    put_payoffs = np.maximum(put_strikes[np.newaxis, :] - S, 0.0)
    put_pain = np.dot(put_payoffs, put_oi)
    
    total_pain = call_pain + put_pain
    
    # 寻找 Pain 最小的格点价格
    best_idx = np.argmin(total_pain)
    max_pain_price = s_grid[best_idx]
    
    total_call_oi = np.sum(call_oi)
    total_put_oi = np.sum(put_oi)
    
    return max_pain_price, total_call_oi, total_put_oi

def main():
    print(">>> Starting Robust Max Pain Calculation Process...", flush=True)
    
    if not os.path.exists(BASIC_FILE) or not os.path.exists(UNDERLYING_FILE):
        raise FileNotFoundError("Basic contract info or underlying daily prices not found. Please run step1 first.")
        
    df_basic = pd.read_parquet(BASIC_FILE)
    df_und = pd.read_parquet(UNDERLYING_FILE)
    
    # 建立 underlying mapping
    df_basic['underlying_code'] = np.where(df_basic['symbol'].str.startswith('510050'), '510050.SH', '510300.SH')
    df_und['trade_date'] = df_und['trade_date'].astype(str)
    
    # 标的价格对照
    und_price_map = df_und.set_index(['trade_date', 'ts_code'])['close'].to_dict()
    # 排序后的交易日序列，用于计算交易日 DTE
    all_trade_dates = sorted(df_und['trade_date'].unique())
    
    # 搜寻并读取所有每日数据文件
    daily_files = sorted(glob.glob(os.path.join(DAILY_DIR, "opt_daily_*.parquet")))
    print(f"Found {len(daily_files)} daily options files to load.", flush=True)
    
    all_day_dfs = []
    for f in daily_files:
        filename = os.path.basename(f)
        date = filename.split('_')[-1].split('.')[0]
        try:
            df_day = pd.read_parquet(f)
            if not df_day.empty:
                df_day['trade_date'] = date
                all_day_dfs.append(df_day[['trade_date', 'ts_code', 'oi']])
        except Exception:
            continue
            
    df_all_opts = pd.concat(all_day_dfs, ignore_index=True)
    print(f"Loaded {len(df_all_opts)} options day-contract records. Processing lags...", flush=True)
    
    # 对每个期权合约，按日期排序，并将持仓量 (OI) 滞后一天 (shift 1) 消除前视偏差
    df_all_opts = df_all_opts.sort_values(['ts_code', 'trade_date']).reset_index(drop=True)
    df_all_opts['oi_lag1'] = df_all_opts.groupby('ts_code')['oi'].shift(1).fillna(0.0)
    
    # 缓存 contract 属性映射
    contract_info = df_basic.set_index('ts_code')[['call_put', 'exercise_price', 'last_edate', 'underlying_code']].to_dict(orient='index')
    
    df_all_opts['call_put'] = df_all_opts['ts_code'].map(lambda x: contract_info.get(x, {}).get('call_put', None))
    df_all_opts['exercise_price'] = df_all_opts['ts_code'].map(lambda x: contract_info.get(x, {}).get('exercise_price', np.nan))
    df_all_opts['last_edate'] = df_all_opts['ts_code'].map(lambda x: contract_info.get(x, {}).get('last_edate', None))
    df_all_opts['underlying_code'] = df_all_opts['ts_code'].map(lambda x: contract_info.get(x, {}).get('underlying_code', None))
    
    df_all_opts = df_all_opts.dropna(subset=['call_put', 'exercise_price', 'last_edate', 'underlying_code'])
    print(f"Mapped and cleaned options metadata. Total active rows: {len(df_all_opts)}", flush=True)
    
    # 分日期和标的计算
    records = []
    
    for (date, und_code), group in df_all_opts.groupby(['trade_date', 'underlying_code']):
        # 获取标的收盘价
        und_close = und_price_map.get((date, und_code), np.nan)
        if pd.isna(und_close):
            continue
            
        # 筛选尚未到期的到期日 (last_edate >= date)
        active_dates = group[group['last_edate'] >= date]['last_edate'].unique()
        if len(active_dates) == 0:
            continue
            
        # 锁定最临近的主力合约链
        expiry_date = min(active_dates)
        df_chain = group[group['last_edate'] == expiry_date].copy()
        
        # 计算最大痛点价格 (基于滞后持仓量 oi_lag1)
        max_pain, oi_call, oi_put = calculate_max_pain_for_chain(df_chain, oi_col='oi_lag1')
        
        if not pd.isna(max_pain):
            # Placebo 计算：离当前 ETF 收盘价最近的行权价
            all_strikes = df_chain['exercise_price'].unique()
            if len(all_strikes) > 0:
                placebo_price = all_strikes[np.argmin(np.abs(all_strikes - und_close))]
            else:
                placebo_price = np.nan
                
            # 计算交易日 DTE
            try:
                idx_t = all_trade_dates.index(date)
                # 寻找第一个 >= expiry_date 的交易日索引
                idx_E = None
                for idx, dt in enumerate(all_trade_dates):
                    if dt >= expiry_date:
                        idx_E = idx
                        break
                trading_dte = max(0, idx_E - idx_t) if idx_E is not None else np.nan
            except Exception:
                trading_dte = np.nan
                
            records.append({
                'trade_date': date,
                'underlying_code': und_code,
                'underlying_close': und_close,
                'expiry_date': expiry_date,
                'days_to_expiry': trading_dte, # 升级为交易日
                'max_pain_price': max_pain,
                'placebo_price': placebo_price, # Placebo 价格
                'total_oi_call_lag1': oi_call,
                'total_oi_put_lag1': oi_put
            })
            
    if records:
        df_out = pd.DataFrame(records)
        df_out = df_out.sort_values(['underlying_code', 'trade_date']).reset_index(drop=True)
        df_out.to_csv(OUTPUT_FILE, index=False)
        print(f"[SUCCESS] Calculated robust Max Pain history! Saved {len(df_out)} records to {OUTPUT_FILE}")
    else:
        print("[WARNING] No records calculated.")

if __name__ == '__main__':
    main()

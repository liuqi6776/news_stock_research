"""
Super-Quant 实盘信号生成器 (V7.0)
说明：支持 Super-Monthly (20日) 与 Super-Weekly (5日) 策略。
"""
import os
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import RobustScaler
import joblib
import sys

# 配置路径
DATA_DIR      = r'D:\iquant_data\data_v2\data_day1'
BASIC_DIR     = r'D:\iquant_data\data_v2\other_day1'
CHIP_DIR      = r'D:\iquant_data\data_v2\cyq1'
OUT_DIR       = r'C:\Users\liuqi\quant_system_v2'

MODELS = {
    'monthly': os.path.join(OUT_DIR, 'super_monthly_model.joblib'),
    'weekly':  os.path.join(OUT_DIR, 'super_weekly_model.joblib')
}

# 必须包含 Monster Gene 因子
FEATURE_COLS = ['mom_20', 'mom_60', 'bias_20', 'ep', 'bp', 'log_mv', 'chip_score', 'chip_bottom_heavy',
                'mom_20_rank', 'mom_60_rank', 'ep_rank', 'bp_rank']
# 周频特征略有不同，但为了通用，我们使用全量特征池
WEEKLY_COLS = ['mom_5', 'mom_20', 'bias_5', 'ep', 'bp', 'log_mv', 'chip_score', 'chip_bottom_heavy',
                'mom_5_rank', 'mom_20_rank', 'ep_rank', 'bp_rank']

def build_features_latest(df, mode='monthly'):
    df = df.copy().sort_values(['ts_code', 'trade_date'])
    g = df.groupby('ts_code')['close']
    for w in [5, 20, 60]:
        df[f'mom_{w}'] = g.transform(lambda x: x / x.shift(w) - 1)
        df[f'bias_{w}'] = (df['close'] - g.transform(lambda x: x.rolling(w).mean())) / (df['close'].rolling(w).mean() + 1e-8)
    
    df['ep'] = 1.0 / (df['pe'] + 1e-8)
    df['bp'] = 1.0 / (df['pb'] + 1e-8)
    df['log_mv'] = np.log(df['circ_mv'] + 1)
    df['chip_score'] = df['winner_rate'] * (df['close'] > df['cost_50pct']).astype(int)
    df['chip_bottom_heavy'] = (df['cost_85pct'] - df['cost_50pct']) / (df['cost_50pct'] - df['cost_15pct'] + 1e-8)
    
    ranks = ['mom_5', 'mom_20', 'mom_60', 'ep', 'bp', 'chip_score', 'chip_bottom_heavy']
    for col in ranks:
        if col in df.columns:
            df[f'{col}_rank'] = df.groupby('trade_date')[col].rank(pct=True)
    return df

def generate_signals(mode='weekly'):
    print(f"正在获取 [{mode.upper()}] 最新信号...")
    all_dates = sorted([f.replace('.parquet', '') for f in os.listdir(DATA_DIR) if f.endswith('.parquet')])
    latest_dates = all_dates[-100:]
    
    files = []
    for ds in latest_dates:
        try:
            p = pd.read_parquet(os.path.join(DATA_DIR, f"{ds}.parquet"), columns=['ts_code','trade_date','open','close','pre_close'])
            b = pd.read_parquet(os.path.join(BASIC_DIR, f"{ds}.parquet"), columns=['ts_code','pe','pb','circ_mv'])
            chip_path = os.path.join(CHIP_DIR, f"{ds}.parquet")
            if os.path.exists(chip_path):
                c = pd.read_parquet(chip_path, columns=['ts_code','winner_rate','cost_15pct','cost_50pct','cost_85pct'])
            else:
                continue
            m = pd.merge(p, pd.merge(b, c, on='ts_code', how='left'), on='ts_code')
            files.append(m)
        except: continue
        
    full_df = pd.concat(files, ignore_index=True)
    full_df = build_features_latest(full_df, mode=mode)
    last_day = full_df[full_df['trade_date'] == full_df['trade_date'].max()].copy()
    
    model_path = MODELS.get(mode)
    if not os.path.exists(model_path):
        print(f"错误：未找到 {mode} 模型文件。请先运行对应的回测脚本。")
        return

    model, scaler = joblib.load(model_path)
    cols = WEEKLY_COLS if mode == 'weekly' else FEATURE_COLS
    
    X = scaler.transform(last_day[cols].fillna(0))
    last_day['prob'] = model.predict_proba(X)[:, 1]
    picks = last_day.sort_values('prob', ascending=False).head(3)
    
    print("\n" + "="*45)
    print(f"  策略模式: {mode.upper()} | 信号日期: {last_day['trade_date'].iloc[0]}")
    print(f"  建议操作: 明日开盘 (T+1)")
    print("="*45)
    for i, (idx, row) in enumerate(picks.iterrows()):
        print(f"[{i+1}] 代码: {row['ts_code']} | 预测胜率: {row['prob']*100:.2f}% | 筹码分: {row['chip_score']:.2f}")
    print("="*45)
    freq_desc = "5 个交易日（1周）" if mode == 'weekly' else "20 个交易日（1个月）"
    print(f"实操建议：\n1. 明日集合竞价买入以上标的。\n2. 持仓周期: {freq_desc}。\n3. 硬性止损: -15%。")

if __name__ == "__main__":
    mode = 'weekly'
    if len(sys.argv) > 1:
        mode = sys.argv[1].lower()
    generate_signals(mode)

"""
方案 A++ Super-Weekly: 高频化与 T+1 严格对齐版
目标：
1. 修正新闻对齐（0延迟，盘前即开盘使用）。
2. 技术指标统一后移 1 日（用昨日收盘 + 今日新闻，做今日开盘决策）。
3. 严格 T+1 回测：逐日循环，买入卖出完全分离，NAV 每日更新。
"""
import os
import sys
import warnings
import pandas as pd
import numpy as np
import tushare as ts
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm
import xgboost as xgb
from sklearn.preprocessing import RobustScaler
import joblib

warnings.filterwarnings('ignore')

TUSHARE_TOKEN = '421ff94dd31be789aa7f95e61ad6fad5bcefa250a0b2c4d298224aa'
DATA_DIR      = r'D:\iquant_data\data_v2\data_day1'
BASIC_DIR     = r'D:\iquant_data\data_v2\other_day1'
CHIP_DIR      = r'D:\iquant_data\data_v2\cyq1'
OUT_DIR       = r'C:\Users\liuqi\quant_system_v2'

INITIAL_CAP   = 100_000.0
TOP_N         = 3        
COMMISSION    = 0.0003
STAMP_DUTY    = 0.0005
SLIPPAGE      = 0.001
REBAL_FREQ    = 5        

def get_limit_price(code, pre_close, direction='up'):
    ratio = 0.2 if code.startswith(('30', '68')) else 0.1
    if direction == 'up':
        return round(pre_close * (1 + ratio), 2)
    return round(pre_close * (1 - ratio), 2)

def load_super_data(start, end):
    files = []
    date_strs = sorted([f.replace('.parquet', '') for f in os.listdir(DATA_DIR) if f.endswith('.parquet')])
    for ds in tqdm(date_strs, desc="加载全维数据"):
        if not (start <= ds <= end): continue
        try:
            p = pd.read_parquet(os.path.join(DATA_DIR, f"{ds}.parquet"), columns=['ts_code','trade_date','open','close','high','low','pre_close','vol'])
            b = pd.read_parquet(os.path.join(BASIC_DIR, f"{ds}.parquet"), columns=['ts_code','pe','pb','circ_mv'])
            chip_path = os.path.join(CHIP_DIR, f"{ds}.parquet")
            c = pd.read_parquet(chip_path, columns=['ts_code','winner_rate','cost_15pct','cost_50pct','cost_85pct']) if os.path.exists(chip_path) else pd.DataFrame(columns=['ts_code','winner_rate','cost_15pct','cost_50pct','cost_85pct'])
            m = pd.merge(p, b, on='ts_code')
            m = pd.merge(m, c, on='ts_code', how='left')
            files.append(m)
        except: continue
    df = pd.concat(files, ignore_index=True)
    df['trade_date'] = pd.to_datetime(df['trade_date'].astype(str))
    
    # 获取新闻（修正后，article_date 即 trade_date）
    from infra_data.storage import DataStorage
    storage = DataStorage()
    # 注意：此时 load_news_data 已不再自动前移日期
    news_market_df, news_stock_sector_df = storage.load_news_data(start, end, None)
    
    if not news_market_df.empty:
        df = pd.merge(df, news_market_df, on='trade_date', how='left')
    else:
        df['news_market_impact'] = 0.0
    if not news_stock_sector_df.empty:
        df = pd.merge(df, news_stock_sector_df, on=['trade_date', 'ts_code'], how='left')
    else:
        df['news_stock_impact'], df['news_sector_impact'] = 0.0, 0.0
    
    df[['news_market_impact', 'news_stock_impact', 'news_sector_impact']] = df[['news_market_impact', 'news_stock_impact', 'news_sector_impact']].fillna(0.0)
    return df

def build_super_features(df):
    """
    逻辑：
    1. 计算基于今日收盘的指标。
    2. 将包含收盘信息的指标 shift(1)，代表“昨日收盘状态”。
    3. 合并后的 Row[T] 包含：T-1 的技术信息 + T 的开盘新闻。
    """
    df = df.copy().sort_values(['ts_code', 'trade_date'])
    g = df.groupby('ts_code')
    
    # 基础指标 (T 时刻)
    for w in [5, 20]:
        df[f'mom_{w}'] = g['close'].transform(lambda x: x / x.shift(w) - 1)
        df[f'bias_{w}'] = (df['close'] - g['close'].transform(lambda x: x.rolling(w).mean())) / (g['close'].transform(lambda x: x.rolling(w).mean()) + 1e-8)
    
    df['ep'] = 1.0 / (df['pe'] + 1e-8)
    df['bp'] = 1.0 / (df['pb'] + 1e-8)
    df['log_mv'] = np.log(df['circ_mv'] + 1e-5)
    df['chip_score'] = df['winner_rate'] * (df['close'] > df['cost_50pct']).astype(int)
    df['chip_bottom_heavy'] = (df['cost_85pct'] - df['cost_50pct']) / (df['cost_50pct'] - df['cost_15pct'] + 1e-8)
    
    # 前移 1 日：将需要“收盘后才知道”的信息移到下一个交易日。
    tech_cols = ['mom_5', 'mom_20', 'bias_5', 'bias_20', 'ep', 'bp', 'log_mv', 'chip_score', 'chip_bottom_heavy']
    for col in tech_cols:
        if col in df.columns:
            df[col] = g[col].shift(1)
            # 添加排名特征
            df[f'{col}_rank'] = df.groupby('trade_date')[col].rank(pct=True)
            
    # 新闻特征（保持 T 时刻，代表盘前提早获知，今日开盘可决策）
    # news_cols = ['news_market_impact', 'news_stock_impact', 'news_sector_impact']
    
    return df.dropna(subset=['mom_5']) # 删掉第一天没指标的

def add_labels(df, horizon=5):
    """
    Row[T] 包含特征后，Label 应为 T 的开盘到 T+horizon 的开盘。
    """
    df = df.sort_values(['ts_code', 'trade_date'])
    g = df.groupby('ts_code')
    
    # entry 为当天开盘（已经有了昨日收盘技术指标和今天上午新闻）
    entry = df['open']
    # exit 为 5 个交易日后的开盘
    exit_ = g['open'].shift(-horizon)
    
    df['ret'] = (exit_ - entry) / (entry + 1e-8)
    df['label'] = (df['ret'] > 0.02).astype(int)
    return df

FEATURE_COLS = ['mom_5', 'mom_20', 'bias_5', 'ep', 'bp', 'log_mv', 'chip_score', 'chip_bottom_heavy',
                'mom_5_rank', 'mom_20_rank', 'ep_rank', 'bp_rank', 
                'news_market_impact', 'news_stock_impact', 'news_sector_impact']

def train_super_model(train_df):
    sub = train_df.dropna(subset=FEATURE_COLS + ['label']).copy()
    if len(sub) < 1000: return None, None
    X = sub[FEATURE_COLS].replace([np.inf, -np.inf], 0).fillna(0)
    y = sub['label']
    
    # 类别均衡
    pos = sub[y == 1]
    neg = sub[y == 0].sample(min(len(pos)*2, len(sub)-len(pos)), random_state=42) if len(pos)>0 else sub.head(100)
    bal = pd.concat([pos, neg])
    
    scaler = RobustScaler()
    X_bal = scaler.fit_transform(bal[FEATURE_COLS])
    model = xgb.XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.03, subsample=0.8, eval_metric='logloss', n_jobs=-1, tree_method='hist')
    model.fit(X_bal, bal['label'])
    return model, scaler

def run_super_backtest(df):
    print("\n" + "!"*50 + "\n  STRICT T+1 DAILY BACKTEST\n" + "!"*50)
    # 过滤测试集：从 2024 年开始（保留 2023 训练）
    test_df = df[df['trade_date'] >= '2024-01-01'].copy()
    all_dates = sorted(test_df['trade_date'].unique())
    
    # 预先构建字典
    prices_dict = df.set_index(['trade_date', 'ts_code'])[['open', 'close', 'pre_close']].to_dict('index')
    
    capital = INITIAL_CAP
    holdings = [] # List of {ts_code, shares, buy_date, buy_px}
    equity = []
    
    last_train_month = -1
    cur_model, cur_scaler = None, None
    
    for i, date in enumerate(tqdm(all_dates, desc="Backtesting Daily")):
        # 1. 每日 NAV 记录 (用前一日收盘价估值，如果是持仓第一天则用今日收盘)
        mv = 0
        for pos in holdings:
            p_data = prices_dict.get((date, pos['ts_code']), None)
            cv = p_data['close'] if p_data else pos['buy_px']
            mv += pos['shares'] * cv
        equity.append({'date': date, 'nav': capital + mv})
        
        # 2. 卖出逻辑 (严格 T+1: 只有 buy_date < date 才能卖)
        if i % REBAL_FREQ == 0: # 假设周频调仓，逻辑在调仓日触发
            for pos in list(holdings):
                if pos['buy_date'] < date: # 严格 T+1
                    px_sell = prices_dict.get((date, pos['ts_code']))
                    if px_sell:
                        # 检查跌停
                        down_limit = get_limit_price(pos['ts_code'], px_sell['pre_close'], 'down')
                        if px_sell['open'] <= down_limit: continue
                        
                        exit_px = px_sell['open'] * (1 - SLIPPAGE)
                        revenue = pos['shares'] * exit_px
                        capital += (revenue - max(5, revenue*COMMISSION) - revenue*STAMP_DUTY)
                        holdings.remove(pos)
        
        # 3. 训练/更新模型
        if date.month != last_train_month:
            train_data = df[(df['trade_date'] < date) & (df['trade_date'] >= date - pd.Timedelta(days=365*2))]
            cur_model, cur_scaler = train_super_model(train_data)
            last_train_month = date.month
            if cur_model:
                joblib.dump((cur_model, cur_scaler), os.path.join(OUT_DIR, 'super_weekly_model_v2.joblib'))

        # 4. 买入逻辑
        if i % REBAL_FREQ == 0 and len(holdings) < TOP_N:
            day_data = test_df[test_df['trade_date'] == date].dropna(subset=FEATURE_COLS)
            if cur_model and not day_data.empty:
                X = cur_scaler.transform(day_data[FEATURE_COLS].fillna(0))
                day_data['prob'] = cur_model.predict_proba(X)[:, 1]
                
                # 过滤涨停、过滤 688、过滤市值 > 50B
                day_data = day_data[~day_data['ts_code'].str.startswith('688')]
                day_data = day_data[day_data['circ_mv'] <= 5000000] # 单位万
                
                picks = day_data.sort_values('prob', ascending=False).head(TOP_N)
                
                remaining_slots = TOP_N - len(holdings)
                if remaining_slots > 0:
                    cash_per = capital / remaining_slots
                    for _, row in picks.iterrows():
                        if any(h['ts_code'] == row['ts_code'] for h in holdings): continue
                        
                        px_buy = prices_dict.get((date, row['ts_code']))
                        if px_buy:
                            up_limit = get_limit_price(row['ts_code'], px_buy['pre_close'], 'up')
                            if px_buy['open'] >= up_limit: continue
                            
                            buy_px = px_buy['open'] * (1 + SLIPPAGE)
                            shares = int(cash_per / buy_px / 100) * 100
                            if shares >= 100:
                                cost = shares * buy_px + max(5, shares * buy_px * COMMISSION)
                                if capital >= cost:
                                    capital -= cost
                                    holdings.append({'ts_code': row['ts_code'], 'shares': shares, 'buy_px': buy_px, 'buy_date': date})
                                    if len(holdings) >= TOP_N: break
                                    
    return pd.DataFrame(equity)

if __name__ == "__main__":
    df = load_super_data('20220101', '20260327')
    df = build_super_features(df)
    df = add_labels(df, horizon=5)
    
    eq_df = run_super_backtest(df)
    
    final_nav = eq_df.iloc[-1]['nav']
    print(f"\nFinal NAV: {final_nav:.2f} (Total Return: {(final_nav/INITIAL_CAP-1)*100:.2f}%)")
    
    results_path = os.path.join(OUT_DIR, 'super_weekly_news_equity.csv')
    eq_df.to_csv(results_path, index=False)
    print(f"Results saved to {results_path}")
    
    # Simple Plot
    plt.figure(figsize=(10, 6))
    plt.plot(eq_df['date'], eq_df['nav'], label='Super-Weekly (Strict T+1)')
    plt.title('Corrected Super-Weekly Backtest with News Features')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(OUT_DIR, 'super_weekly_v2_curve.png'))

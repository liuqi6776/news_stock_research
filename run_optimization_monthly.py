"""
方案 A 优化：月频模型增强（基本面 + 行业中性化）

优化点：
1. 引入 PE, PB, 流通市值 (circ_mv) 等基本面数据
2. 引入行业分类，计算行业中性化因子（EP_neutral, BP_neutral）
3. 引入行业动量因子（Sector Momentum）
4. 仅针对月频 (20d) 标签和策略进行训练和回测
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
import matplotlib.dates as mdates
from tqdm import tqdm
import xgboost as xgb
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import roc_auc_score

warnings.filterwarnings('ignore')

# ============================================================
# 配置 (同步用户最新 Token)
# ============================================================
TUSHARE_TOKEN = '421ff94dd31be789aa7f95e61ad6fad5bcefa250a0b2c4d298224aa'
DATA_DIR      = r'D:\iquant_data\data_v2\data_day1'
BASIC_DIR     = r'D:\iquant_data\data_v2\other_day1'
SSE_FILE      = r'C:\Users\liuqi\quant_system_v2\sse_index_2023.csv'
OUT_DIR       = r'C:\Users\liuqi\quant_system_v2'

INITIAL_CAP   = 100_000.0
TOP_N         = 5
COMMISSION    = 0.0003
STAMP_DUTY    = 0.0005
SLIPPAGE      = 0.001
STOP_LOSS_M   = -0.15    # 月策略止损 -15%

DATA_START    = '20210101'
TEST_START    = '20230101'
TEST_END      = '20260101'

# ============================================================
# 数据加载增强
# ============================================================
def load_data(start: str, end: str) -> pd.DataFrame:
    """合并加载价格和基础指标 (PE/PB/MV)"""
    files = []
    sd, ed = pd.to_datetime(start), pd.to_datetime(end)
    
    # 获取日期列表
    date_strs = sorted([f.replace('.parquet', '') for f in os.listdir(DATA_DIR) if f.endswith('.parquet')])
    
    for ds in tqdm(date_strs, desc="加载行情与指标"):
        dt = pd.to_datetime(ds)
        if sd <= dt <= ed:
            p_file = os.path.join(DATA_DIR, f"{ds}.parquet")
            b_file = os.path.join(BASIC_DIR, f"{ds}.parquet")
            if not os.path.exists(p_file) or not os.path.exists(b_file):
                continue
            
            try:
                # 价格
                p_df = pd.read_parquet(p_file, columns=['ts_code', 'trade_date', 'open', 'high', 'low', 'close', 'vol', 'pct_chg'])
                # 指标
                b_df = pd.read_parquet(b_file, columns=['ts_code', 'pe', 'pb', 'circ_mv'])
                
                merged = pd.merge(p_df, b_df, on='ts_code', how='inner')
                files.append(merged)
            except:
                continue
                
    if not files: return pd.DataFrame()
    df = pd.concat(files, ignore_index=True)
    df['trade_date'] = pd.to_datetime(df['trade_date'].astype(str))
    # 限制主板
    df = df[df['ts_code'].str.match(r'^(00|60)')]
    return df.sort_values(['ts_code', 'trade_date']).reset_index(drop=True)

def get_industry_mapping():
    """获取行业映射"""
    map_file = os.path.join(OUT_DIR, 'stock_industry_map_cached.parquet')
    if os.path.exists(map_file):
        return pd.read_parquet(map_file)
    
    print("正在从 Tushare 获取行业分类...")
    pro = ts.pro_api(TUSHARE_TOKEN)
    try:
        df = pro.stock_basic(exchange='', list_status='L', fields='ts_code,industry')
        df.to_parquet(map_file)
        return df
    except Exception as e:
        print(f"Tushare 获取失败: {e}，将不使用行业中性化")
        return pd.DataFrame()

# ============================================================
# 优化版特征工程
# ============================================================
def build_optimized_features(df: pd.DataFrame, industry_df: pd.DataFrame) -> pd.DataFrame:
    print("构建优化版特征 (基本面 + 行业 + 技术)...")
    df = df.copy().sort_values(['ts_code', 'trade_date'])
    
    # --- 1. 基础技术指标 (从 Plan B 继承) ---
    g = df.groupby('ts_code')['close']
    for w in [5, 10, 20, 60]:
        df[f'mom_{w}'] = g.transform(lambda x: x / x.shift(w) - 1)
        ma = g.transform(lambda x: x.rolling(w).mean())
        df[f'bias_{w}'] = (df['close'] - ma) / (ma + 1e-8)
        
    # 成交量比
    vol_g = df.groupby('ts_code')['vol']
    df['vol_ratio_5_20'] = (
        vol_g.transform(lambda x: x.rolling(5).mean()) /
        vol_g.transform(lambda x: x.rolling(20).mean()).clip(1e-6)
    )
    
    # 波动率
    df['log_ret'] = df.groupby('ts_code')['close'].transform(lambda x: np.log(x / x.shift(1)))
    df['vol_20']  = df.groupby('ts_code')['log_ret'].transform(lambda x: x.rolling(20).std() * np.sqrt(252))
    
    # 布林带宽度
    ma20  = g.transform(lambda x: x.rolling(20).mean())
    std20 = g.transform(lambda x: x.rolling(20).std())
    df['boll_width'] = (2 * std20) / ma20.clip(1e-6)
    
    # 通道位置
    hi60 = df.groupby('ts_code')['high'].transform(lambda x: x.rolling(60).max())
    lo60 = df.groupby('ts_code')['low'].transform(lambda x: x.rolling(60).min())
    df['channel_pos'] = (df['close'] - lo60) / (hi60 - lo60 + 1e-6)

    # --- 2. 基本面因子 ---
    df['ep'] = 1.0 / (df['pe'] + 1e-8)
    df['bp'] = 1.0 / (df['pb'] + 1e-8)
    df['log_mv'] = np.log(df['circ_mv'] + 1)
    
    # --- 3. 行业中性化 ---
    if not industry_df.empty:
        df = pd.merge(df, industry_df, on='ts_code', how='left')
        df['industry'] = df['industry'].fillna('unknown')
        idx_cols = ['trade_date', 'industry']
        df['ep_neutral'] = df['ep'] - df.groupby(idx_cols)['ep'].transform('mean')
        df['bp_neutral'] = df['bp'] - df.groupby(idx_cols)['bp'].transform('mean')
        df['sector_ret_20'] = df.groupby(idx_cols)['mom_20'].transform('mean')
    else:
        df['ep_neutral'] = df.iloc[:, 0] * 0
        df['bp_neutral'] = df.iloc[:, 0] * 0
        df['sector_ret_20'] = 0
        
    # --- 4. 截面排名 ---
    for col in ['mom_20', 'mom_60', 'vol_ratio_5_20', 'bias_20', 'channel_pos', 'ep', 'circ_mv']:
        df[f'{col}_rank'] = df.groupby('trade_date')[col].rank(pct=True)
    
    return df

def add_labels(df: pd.DataFrame, horizon: int = 20) -> pd.DataFrame:
    df = df.sort_values(['ts_code', 'trade_date'])
    # 标签：未来 20 日收益率
    entry = df.groupby('ts_code')['open'].shift(-1)
    exit_ = df.groupby('ts_code')['open'].shift(-horizon - 1)
    df['ret_label'] = (exit_ - entry) / (entry + 1e-8)
    df['label'] = (df['ret_label'] > 0.01).astype(int) # 目标：跑赢 1%
    return df

FEATURE_COLS = [
    'mom_5', 'mom_20', 'mom_60', 'bias_5', 'bias_20', 'bias_60',
    'vol_ratio_5_20', 'vol_20', 'boll_width', 'channel_pos',
    'ep', 'bp', 'log_mv', 'ep_neutral', 'bp_neutral', 'sector_ret_20',
    'mom_20_rank', 'mom_60_rank', 'bias_20_rank', 'ep_rank', 'circ_mv_rank'
]

# ============================================================
# 训练与测试
# ============================================================
def train_model(train_df: pd.DataFrame):
    sub = train_df.dropna(subset=FEATURE_COLS + ['label']).copy()
    if sub.empty: return None, None
    X = sub[FEATURE_COLS].replace([np.inf, -np.inf], 0).fillna(0)
    y = sub['label']
    
    # 平衡
    pos, neg = sub[y == 1], sub[y == 0]
    n = min(len(pos), len(neg))
    if n < 100: return None, None
    
    bal = pd.concat([pos.sample(n, random_state=42), neg.sample(n, random_state=42)])
    scaler = RobustScaler()
    X_bal = scaler.fit_transform(bal[FEATURE_COLS].fillna(0))
    
    model = xgb.XGBClassifier(
        n_estimators=200, max_depth=5, learning_rate=0.04,
        subsample=0.8, colsample_bytree=0.8, eval_metric='logloss'
    )
    model.fit(X_bal, bal['label'])
    
    auc = roc_auc_score(y, model.predict_proba(scaler.transform(X))[:, 1])
    print(f"      Train AUC: {auc:.3f}")
    return model, scaler

def run_monthly_optimization(df: pd.DataFrame):
    print("\n" + "="*50)
    print("  月频优化策略回测 (Option A)")
    print("="*50)
    
    test_dates = sorted(df[df['trade_date'] >= TEST_START]['trade_date'].unique())
    rebal_dates = list(pd.Series(test_dates).groupby(pd.Series(test_dates).dt.to_period('M')).first())
    
    # 建立查找字典加速回测
    price_lookup = {}
    for _, row in df.iterrows():
        price_lookup[(row['trade_date'], row['ts_code'])] = {'open': row['open'], 'close': row['close']}
    
    capital = INITIAL_CAP
    positions = {}
    equity = []
    trade_log = []
    
    cur_model, cur_scaler = None, None
    last_q = None
    
    for i, date in enumerate(tqdm(rebal_dates)):
        # 1. 滚动重训 (半年一次或每季度)
        q = (date.year, (date.month-1)//3)
        if q != last_q:
            train_data = df[(df['trade_date'] < date) & (df['trade_date'] >= date - pd.Timedelta(days=365*2))]
            print(f"\n  [重训] {date.date()}")
            cur_model, cur_scaler = train_model(train_data)
            last_q = q
            
        # 2. 卖出上月持仓
        for ts_code, pos in list(positions.items()):
            key = (date, ts_code)
            if key in price_lookup:
                exit_px = price_lookup[key]['open'] * (1 - SLIPPAGE)
                revenue = pos['shares'] * exit_px
                fee = max(5, revenue * COMMISSION) + revenue * STAMP_DUTY
                net_revenue = revenue - fee
                capital += net_revenue
                pnl = net_revenue - pos['shares'] * pos['cost']
                trade_log.append({'date': date, 'ts_code': ts_code, 'action': 'SELL', 'price': exit_px, 'pnl': pnl})
        positions = {}
        
        # 3. 选股买入
        day_data = df[df['trade_date'] == date].dropna(subset=FEATURE_COLS)
        if not day_data.empty and cur_model:
            X = cur_scaler.transform(day_data[FEATURE_COLS].fillna(0))
            day_data['prob'] = cur_model.predict_proba(X)[:, 1]
            
            # 选 Top-N (阈值提升至 0.60 以保证质量)
            picks = day_data[day_data['prob'] > 0.60].sort_values('prob', ascending=False).head(TOP_N)
            if not picks.empty:
                cash_per_stock = capital / TOP_N
                for _, row in picks.iterrows():
                    key = (date, row['ts_code'])
                    if key in price_lookup:
                        buy_px = price_lookup[key]['open'] * (1 + SLIPPAGE)
                        shares = int(cash_per_stock / buy_px / 100) * 100
                        if shares >= 100:
                            cost = shares * buy_px
                            fee = max(5, cost * COMMISSION)
                            capital -= (cost + fee)
                            positions[row['ts_code']] = {'shares': shares, 'cost': buy_px}
                            trade_log.append({'date': date, 'ts_code': row['ts_code'], 'action': 'BUY', 'price': buy_px, 'pnl': 0})
                            
        # 4. 净值记录
        mv = sum(pos['shares'] * price_lookup.get((date, code), {'close': pos['cost']})['close'] 
                 for code, pos in positions.items())
        equity.append({'date': date, 'nav': capital + mv})
        
    eq_df = pd.DataFrame(equity)
    tr_df = pd.DataFrame(trade_log)
    eq_df['nav_norm'] = eq_df['nav'] / INITIAL_CAP
    final_ret = (eq_df.iloc[-1]['nav_norm'] - 1) * 100
    print(f"\n最终收益率: {final_ret:+.2f}%")
    return eq_df, tr_df

def plot_final(opt_eq, baseline_eq_path):
    plt.figure(figsize=(12, 6))
    plt.plot(opt_eq['date'], opt_eq['nav_norm'], label='Optimized Monthly (Option A)', color='gold', linewidth=2.5)
    
    # 尝试加载之前的 Baseline (Plan B)
    if os.path.exists(baseline_eq_path):
        try:
            base = pd.read_csv(baseline_eq_path)
            plt.plot(pd.to_datetime(base['date']), base['nav']/INITIAL_CAP, label='Plan B Baseline', color='gray', linestyle='--')
        except: pass
        
    plt.title("Monthly Optimization Comparison: Plan B vs Option A")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(os.path.join(OUT_DIR, 'monthly_optimization_comparison.png'), dpi=150)
    print(f"图表已保存: monthly_optimization_comparison.png")

if __name__ == "__main__":
    # 1. 加载数据
    df = load_data(DATA_START, TEST_END)
    # 2. 行业映射
    industry_df = get_industry_mapping()
    # 3. 特征工程
    df = build_optimized_features(df, industry_df)
    # 4. 标签
    df = add_labels(df, horizon=20)
    # 5. 回测
    opt_eq, opt_tr = run_monthly_optimization(df)
    # 6. 绘图与保存
    opt_eq.to_csv(os.path.join(OUT_DIR, 'opt_monthly_equity.csv'), index=False)
    opt_tr.to_csv(os.path.join(OUT_DIR, 'opt_monthly_trades.csv'), index=False)
    plot_final(opt_eq, os.path.join(OUT_DIR, 'planB_trades_monthly.csv'))

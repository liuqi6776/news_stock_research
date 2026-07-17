"""
Super-Weekly V16: Market Regime Adaptive Strategy
=================================================
V15 -> V16 核心改进:
1. 市场状态三分类 (牛/震荡/熊) 替代 V15 粗糙的 4 信号布尔判断
2. 动态仓位控制: 牛市=100%, 震荡=50%, 熊市=20%或空仓
3. 自适应换仓周期: 高波动期3天, 低波动期10天
4. 截面排名标签替代固定3%阈值 (牛熊市样本更均衡)
5. 信号强度过滤: 熊市只选 prob > 0.7 的最强信号
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
VIX_DIR       = r'D:\iquant_data\data_v2\vix1'
MARGIN_DIR    = r'D:\iquant_data\data_v2\margin1'
OUT_DIR       = r'C:\Users\liuqi\quant_system_v2'

INITIAL_CAP   = 100_000.0
TOP_N         = 5         # 候选池大小 (V15=3, V16扩大选更多)
MAX_HOLDINGS  = 3         # 最大持仓数 (牛市满仓3只, 震荡2只, 熊市1只)
SLIPPAGE      = 0.001
COMMISSION    = 0.0003
STAMP_DUTY    = 0.0005

# 风险管理参数
STOP_LOSS      = -0.08   # -8% 止损 (V15=-10%, V16更紧)
TAKE_PROFIT    = 0.25    # +25% 止盈 (V15=30%, V16更早锁利)
MAX_HOLD_DAYS  = 8       # 最大持仓8天 (V15=10)

# ==================== 市场状态识别 ====================

def WMA(s, n):
    weights = np.arange(1, n + 1)
    return s.rolling(n).apply(lambda x: np.dot(x, weights) / weights.sum(), raw=True)

def calc_HMA(s, n):
    half_n = int(n / 2)
    sqrt_n = int(np.sqrt(n))
    hma_raw = 2 * WMA(s, half_n) - WMA(s, n)
    return WMA(hma_raw, sqrt_n)

def calc_KAMA(s, n=10, fast=2, slow=30):
    change = abs(s - s.shift(n))
    volatility = abs(s - s.shift(1)).rolling(n).sum()
    er = change / (volatility + 1e-8)
    sc = (er * (2/(fast+1) - 2/(slow+1)) + 2/(slow+1)) ** 2
    kama = np.zeros(len(s))
    for i in range(len(s)):
        if i < n:
            kama[i] = s.iloc[i]
        else:
            kama[i] = kama[i-1] + sc.iloc[i] * (s.iloc[i] - kama[i-1])
    return pd.Series(kama, index=s.index)

def calc_DMI(df, n=14):
    up = df['high'] - df['high'].shift(1)
    down = df['low'].shift(1) - df['low']
    p_dm = np.where((up > down) & (up > 0), up, 0)
    n_dm = np.where((down > up) & (down > 0), down, 0)
    tr = pd.concat([df['high'] - df['low'],
                    abs(df['high'] - df['close'].shift(1)),
                    abs(df['low'] - df['close'].shift(1))], axis=1).max(axis=1)
    smooth_tr = tr.rolling(n).sum()
    smooth_pdm = pd.Series(p_dm).rolling(n).sum()
    smooth_ndm = pd.Series(n_dm).rolling(n).sum()
    p_di = 100 * smooth_pdm / (smooth_tr + 1e-8)
    n_di = 100 * smooth_ndm / (smooth_tr + 1e-8)
    dx = 100 * abs(p_di - n_di) / (p_di + n_di + 1e-8)
    adx = dx.rolling(n).mean()
    return p_di, n_di, adx

def classify_market_regime(idx):
    """
    市场状态三分类 (基于中证1000指数)
    
    返回 DataFrame 添加列:
    - regime: 1=bull(牛市), 0=neutral(震荡), -1=bear(熊市)
    - regime_score: 综合评分 [-100, 100], >30=牛, <-30=熊, 其余震荡
    - rebal_freq: 自适应换仓频率 (牛市5天, 震荡7天, 熊市3天)
    - max_positions: 最大持仓数 (牛市3, 震荡2, 熊市1)
    - position_pct: 目标仓位比例 (牛市1.0, 震荡0.5, 熊市0.2)
    - min_prob: 最低买入概率阈值 (牛市0.4, 震荡0.5, 熊市0.7)
    """
    idx = idx.sort_values('trade_date').copy()
    
    # === Component 1: 趋势 (MA alignment, weight=30%) ===
    idx['ma20'] = idx['close'].rolling(20).mean()
    idx['ma60'] = idx['close'].rolling(60).mean()
    idx['ma120'] = idx['close'].rolling(120).mean()
    
    # MA多头排列得分: 20>60>120 = +30, 20<60<120 = -30
    idx['ma_align'] = (
        (idx['close'] > idx['ma20']).astype(int) * 10 +
        (idx['ma20'] > idx['ma60']).astype(int) * 10 +
        (idx['ma60'] > idx['ma120']).astype(int) * 10
    )
    
    # === Component 2: 动量 (价格vs 20/60日均线, weight=25%) ===
    idx['price_vs_ma20'] = (idx['close'] / idx['ma20'] - 1) * 100  # % deviation
    idx['price_vs_ma60'] = (idx['close'] / idx['ma60'] - 1) * 100
    idx['momentum_score'] = np.clip(
        (idx['price_vs_ma20'] * 0.6 + idx['price_vs_ma60'] * 0.4) * 2, -25, 25
    )
    
    # === Component 3: 市场宽度 (涨跌家数比, 用成交量代理, weight=15%) ===
    idx['vol_ma5'] = idx['vol'].rolling(5).mean()
    idx['vol_ma20'] = idx['vol'].rolling(20).mean()
    vol_ratio = idx['vol_ma5'] / (idx['vol_ma20'] + 1e-8)
    # 放量上涨=好, 缩量下跌=差
    idx['breadth_score'] = np.where(
        idx['close'] > idx['close'].shift(1),
        np.clip((vol_ratio - 1) * 15, -15, 15),
        np.clip((1 - vol_ratio) * 15, -15, 15)
    )
    
    # === Component 4: 波动率 (VIX替代: 20日收益率标准差, weight=15%) ===
    idx['returns'] = idx['close'].pct_change()
    idx['volatility_20'] = idx['returns'].rolling(20).std() * 100
    vol_median = idx['volatility_20'].rolling(60).median()
    # 低波动=好, 高波动=差
    idx['vol_score'] = np.clip(
        (vol_median - idx['volatility_20']) / (vol_median + 1e-8) * 15, -15, 15
    )
    
    # === Component 5: HMA/KAMA 趋势确认 (weight=15%) ===
    idx['hma'] = calc_HMA(idx['close'], 10)
    idx['kama'] = calc_KAMA(idx['close'], 10)
    idx['hma_score'] = np.where(idx['close'] > idx['hma'], 8, -8)
    idx['kama_score'] = np.where(idx['close'] > idx['kama'], 7, -7)
    
    # === 综合评分 ===
    idx['regime_score'] = (
        idx['ma_align'] +           # [-30, 30]
        idx['momentum_score'] +     # [-25, 25]
        idx['breadth_score'] +      # [-15, 15]
        idx['vol_score'] +          # [-15, 15]
        idx['hma_score'] +          # [-8, 8]
        idx['kama_score']           # [-7, 7]
    )
    
    # === 分类 (V16.1: 收紧阈值, 减少误分类) ===
    def classify(score):
        if score > 15:      # 放宽牛市门槛 (30->15)
            return 1   # 牛市
        elif score < -15:   # 放宽熊市门槛 (30->15)
            return -1  # 熊市
        else:
            return 0   # 震荡
    
    idx['regime'] = idx['regime_score'].apply(classify)
    
    # === 自适应参数 (V16.1: 震荡期也保持较高仓位) ===
    idx['rebal_freq'] = idx['regime'].map({1: 5, 0: 5, -1: 3})        # 震荡也5天
    idx['max_positions'] = idx['regime'].map({1: 3, 0: 3, -1: 1})      # 震荡也3只
    idx['position_pct'] = idx['regime'].map({1: 1.0, 0: 0.7, -1: 0.2}) # 震荡70%仓位
    idx['min_prob'] = idx['regime'].map({1: 0.4, 0: 0.45, -1: 0.65})   # 震荡适度提高
    
    return idx

def get_limit_price(code, pre_close, direction='up'):
    ratio = 0.2 if code.startswith(('30', '68')) else 0.1
    if direction == 'up':
        return round(pre_close * (1 + ratio), 2)
    return round(pre_close * (1 - ratio), 2)

def load_vix_data(start, end):
    files = []
    date_strs = sorted([f.replace('.parquet', '') for f in os.listdir(VIX_DIR) if f.endswith('.parquet')])
    for ds in date_strs:
        if not (start <= ds <= end): continue
        try:
            df = pd.read_parquet(os.path.join(VIX_DIR, f"{ds}.parquet"))
            df['trade_date'] = ds
            files.append(df)
        except: continue
    if files:
        vix_df = pd.concat(files, ignore_index=True)
        vix_df['trade_date'] = pd.to_datetime(vix_df['trade_date'])
        vix_df = vix_df.rename(columns={'close': 'vix_close', 'pct_chg': 'vix_pct_chg'})
        return vix_df[['trade_date', 'vix_close', 'vix_pct_chg']]
    return pd.DataFrame(columns=['trade_date', 'vix_close', 'vix_pct_chg'])

def load_margin_data(start, end):
    files = []
    date_strs = sorted([f.replace('.parquet', '') for f in os.listdir(MARGIN_DIR) if f.endswith('.parquet')])
    for ds in date_strs:
        if not (start <= ds <= end): continue
        try:
            df = pd.read_parquet(os.path.join(MARGIN_DIR, f"{ds}.parquet"))
            df['trade_date'] = ds
            files.append(df)
        except: continue
    if files:
        margin_df = pd.concat(files, ignore_index=True)
        margin_df['trade_date'] = pd.to_datetime(margin_df['trade_date'])
        margin_df = margin_df.rename(columns={'rzye': 'margin_rzye', 'rzmre': 'margin_rzmre', 'rqye': 'margin_rqye', 'rzrqye': 'margin_rzrqye'})
        return margin_df[['trade_date', 'margin_rzye', 'margin_rzmre', 'margin_rqye', 'margin_rzrqye']]
    return pd.DataFrame(columns=['trade_date', 'margin_rzye', 'margin_rzmre', 'margin_rqye', 'margin_rzrqye'])

def load_data(start, end):
    print("V16: 正在加载数据并计算市场状态...")
    pro = ts.pro_api(TUSHARE_TOKEN)
    idx = pro.index_daily(ts_code='000852.SH', start_date=start, end_date=end)
    idx['trade_date'] = pd.to_datetime(idx['trade_date'])
    idx = idx.sort_values('trade_date').reset_index(drop=True)
    
    # === V16 核心: 市场状态分类 ===
    idx = classify_market_regime(idx)
    regime_map = idx.set_index('trade_date')[['regime', 'regime_score', 'rebal_freq', 'max_positions', 'position_pct', 'min_prob']].to_dict('index')
    
    # 打印市场状态分布
    regime_counts = idx.groupby('regime').size()
    print(f"  市场状态分布: 牛市={regime_counts.get(1,0)}天, 震荡={regime_counts.get(0,0)}天, 熊市={regime_counts.get(-1,0)}天")
    
    files = []
    date_strs = sorted([f.replace('.parquet', '') for f in os.listdir(DATA_DIR) if f.endswith('.parquet')])
    for ds in tqdm(date_strs, desc="加载 Parquet"):
        if not (start <= ds <= end): continue
        try:
            p = pd.read_parquet(os.path.join(DATA_DIR, f"{ds}.parquet"), columns=['ts_code','trade_date','open','close','high','low','pre_close'])
            b = pd.read_parquet(os.path.join(BASIC_DIR, f"{ds}.parquet"), columns=['ts_code','pe','pb','circ_mv'])
            c_path = os.path.join(CHIP_DIR, f"{ds}.parquet")
            if os.path.exists(c_path):
                c = pd.read_parquet(c_path, columns=['ts_code','winner_rate','cost_15pct','cost_50pct','cost_85pct'])
            else: continue
            files.append(pd.merge(pd.merge(p, b, on='ts_code'), c, on='ts_code'))
        except: continue
    df = pd.concat(files, ignore_index=True)
    df['trade_date'] = pd.to_datetime(df['trade_date'].astype(str))
    
    # 合并市场状态参数到每只股票每日数据
    for col in ['regime', 'regime_score', 'rebal_freq', 'max_positions', 'position_pct', 'min_prob']:
        df[col] = df['trade_date'].map(lambda d: regime_map.get(d, {}).get(col, 0))
    
    # 加载中国波指数据
    vix_df = load_vix_data(start, end)
    if not vix_df.empty:
        df = pd.merge(df, vix_df, on='trade_date', how='left')
    
    # 加载融资融券数据
    margin_df = load_margin_data(start, end)
    if not margin_df.empty:
        df = pd.merge(df, margin_df, on='trade_date', how='left')
    
    df = df.drop_duplicates(subset=['ts_code', 'trade_date'])
    
    return df

def build_features(df):
    df = df.copy().sort_values(['ts_code', 'trade_date'])
    g = df.groupby('ts_code')['close']
    for w in [5, 20]:
        df[f'mom_{w}'] = g.transform(lambda x: x / x.shift(w) - 1)
        df[f'bias_{w}'] = (df['close'] - g.transform(lambda x: x.rolling(w).mean())) / (g.transform(lambda x: x.rolling(w).mean()) + 1e-8)
    
    # 额外动量窗口
    df['mom_10'] = g.transform(lambda x: x / x.shift(10) - 1)
    df['mom_60'] = g.transform(lambda x: x / x.shift(60) - 1)
    
    # 波动率
    df['vol_20'] = g.pct_change().transform(lambda x: x.rolling(20).std())
    
    # 换手率相关 (如果有的话)
    df['ep'] = 1.0 / (df['pe'] + 1e-8)
    df['bp'] = 1.0 / (df['pb'] + 1e-8)
    df['log_mv'] = np.log(df['circ_mv'] + 1)
    
    # 筹码
    df['chip_score'] = df['winner_rate'] * (df['close'] > df['cost_50pct']).astype(int)
    df['chip_bottom_heavy'] = (df['cost_85pct'] - df['cost_50pct']) / (df['cost_50pct'] - df['cost_15pct'] + 1e-8)
    
    # VIX
    if 'vix_close' in df.columns:
        df['vix_rank'] = df.groupby('trade_date')['vix_close'].rank(pct=True)
        df['vix_pct_chg_rank'] = df.groupby('trade_date')['vix_pct_chg'].rank(pct=True)
    
    # 融资融券
    if 'margin_rzye' in df.columns:
        df['margin_rzye_rank'] = df.groupby('trade_date')['margin_rzye'].rank(pct=True)
        df['margin_rzrqye_rank'] = df.groupby('trade_date')['margin_rzrqye'].rank(pct=True)
    
    # 截面排名
    for col in ['mom_5', 'mom_10', 'mom_20', 'mom_60', 'ep', 'bp', 'chip_score', 'chip_bottom_heavy', 'vol_20']:
        df[f'{col}_rank'] = df.groupby('trade_date')[col].rank(pct=True)
    
    # === V16 核心改进: 截面排名标签 ===
    # 不再用固定 3% 阈值, 改用同日所有股票收益率排名前 20%
    print("V16: 生成截面排名标签...")
    df['future_return'] = df.groupby('ts_code')['close'].pct_change(5).shift(-5)
    df['return_rank'] = df.groupby('trade_date')['future_return'].rank(pct=True)
    
    # 前 25% = 1 (涨), 后 25% = 0 (跌/平), 中间 50% 不参与训练
    df['label'] = np.where(df['return_rank'] >= 0.75, 1, np.where(df['return_rank'] <= 0.25, 0, np.nan))
    
    # 打印标签分布
    label_counts = df['label'].value_counts()
    total_labeled = label_counts.sum()
    print(f"  标签分布: 涨(1)={label_counts.get(1,0)}, 跌(0)={label_counts.get(0,0)}, 未标记={len(df)-total_labeled}")
    
    return df

FEATURE_COLS = [
    'mom_5', 'mom_10', 'mom_20', 'mom_60', 'bias_5', 'bias_20',
    'ep', 'bp', 'log_mv', 'chip_score', 'chip_bottom_heavy', 'vol_20',
    'mom_5_rank', 'mom_10_rank', 'mom_20_rank', 'mom_60_rank',
    'ep_rank', 'bp_rank', 'chip_score_rank', 'chip_bottom_heavy_rank', 'vol_20_rank',
]
# 条件添加 (只有数据存在时)
OPTIONAL_FEATURE_COLS = ['vix_rank', 'vix_pct_chg_rank', 'margin_rzye_rank', 'margin_rzrqye_rank']

def train_model(train_df):
    # 筛选存在的特征列
    valid_cols = [col for col in FEATURE_COLS + OPTIONAL_FEATURE_COLS if col in train_df.columns]
    
    # 只用有标签的行
    sub = train_df.dropna(subset=['label']).copy()
    sub = sub.dropna(subset=valid_cols)
    
    if len(sub) < 100:
        print(f"  警告: 训练样本不足 ({len(sub)}), 跳过本轮训练")
        return None, None, valid_cols
    
    X = sub[valid_cols].replace([np.inf, -np.inf], 0).fillna(0)
    y = sub['label']
    
    # 标签均衡化: 统计两个类别的数量, 使用 scale_pos_weight
    n_pos = (y == 1).sum()
    n_neg = (y == 0).sum()
    scale_pos = n_neg / (n_pos + 1e-8) if n_pos > 0 else 1.0
    
    scaler = RobustScaler()
    X_s = scaler.fit_transform(X)
    
    model = xgb.XGBClassifier(
        n_estimators=150,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
    )
    model.fit(X_s, y, verbose=False)
    
    return model, scaler, valid_cols

def run_backtest(df):
    print("\n" + "!"*60)
    print("  Super-Weekly V16: Market Regime Adaptive Strategy")
    print("!"*60)
    
    test_dates = sorted(df[df['trade_date'] >= '2023-01-01']['trade_date'].unique())
    prices = df.set_index(['trade_date', 'ts_code'])[['open', 'close', 'high', 'low', 'pre_close']].to_dict('index')
    
    # 市场状态数据
    regime_data = df.groupby('trade_date').first()[['regime', 'regime_score', 'rebal_freq', 'max_positions', 'position_pct', 'min_prob']].to_dict('index')
    
    capital = INITIAL_CAP
    holdings = []
    equity = []
    regime_log = []  # 记录每日市场状态
    
    cur_model, cur_scaler, valid_cols = None, None, []
    last_year = None
    
    # 统计
    trade_count_by_regime = {1: 0, 0: 0, -1: 0}
    
    for i, date in enumerate(tqdm(test_dates[:-1])):
        d_signal = date
        d_trade  = test_dates[i+1]
        
        rd = regime_data.get(d_signal, {})
        regime = int(rd.get('regime', 0))
        rebal_freq = int(rd.get('rebal_freq', 5))
        max_positions = int(rd.get('max_positions', 3))
        position_pct = float(rd.get('position_pct', 1.0))
        min_prob = float(rd.get('min_prob', 0.4))
        
        # 1. 每年更新模型
        year = date.year
        if year != last_year:
            train_data = df[(df['trade_date'] < date) & (df['trade_date'] >= date - pd.Timedelta(days=365*3))]
            new_model, new_scaler, new_cols = train_model(train_data)
            if new_model is not None:
                cur_model, cur_scaler, valid_cols = new_model, new_scaler, new_cols
                print(f"  {year}年模型训练完成, 特征数={len(valid_cols)}")
            last_year = year
        
        # 2. 每日风控检查
        stocks_to_sell = []
        for pos in holdings:
            px_current = prices.get((d_signal, pos['ts_code']))
            if px_current:
                ret = px_current['close'] / pos['buy_px'] - 1
                pos['days_held'] += 1
                if ret < STOP_LOSS:
                    stocks_to_sell.append((pos['ts_code'], 'StopLoss'))
                elif ret > TAKE_PROFIT:
                    stocks_to_sell.append((pos['ts_code'], 'TakeProfit'))
                elif pos['days_held'] >= MAX_HOLD_DAYS:
                    stocks_to_sell.append((pos['ts_code'], 'TimeExit'))
        
        # 3. 自适应换仓检查
        is_rebal_day = (i % rebal_freq == 0)
        
        # 执行卖出 (T+1 开盘)
        for pos in list(holdings):
            should_exit = False
            if pos['ts_code'] in [s[0] for s in stocks_to_sell]:
                should_exit = True
            elif is_rebal_day:
                should_exit = True
            
            if should_exit:
                px_sell = prices.get((d_trade, pos['ts_code']))
                if px_sell:
                    down_limit = get_limit_price(pos['ts_code'], px_sell['pre_close'], 'down')
                    if px_sell['open'] > down_limit:
                        exit_px = px_sell['open'] * (1 - SLIPPAGE)
                        revenue = pos['shares'] * exit_px
                        capital += (revenue - max(5, revenue*COMMISSION) - revenue*STAMP_DUTY)
                        holdings.remove(pos)
        
        # 4. 买入逻辑
        if is_rebal_day and len(holdings) < max_positions and cur_model:
            
            # 熊市额外限制: 如果 regime_score < -40, 完全不买
            regime_score = rd.get('regime_score', 0)
            if regime == -1 and regime_score < -40:
                mv = sum(p['shares'] * prices.get((d_trade, p['ts_code']), {'close': p['buy_px']})['close'] for p in holdings)
                equity.append({'date': d_trade, 'nav': capital + mv})
                regime_log.append({'date': d_trade, 'regime': regime, 'positions': len(holdings)})
                continue
            
            day_data = df[df['trade_date'] == d_signal].dropna(subset=valid_cols)
            if not day_data.empty:
                X = cur_scaler.transform(day_data[valid_cols].fillna(0))
                day_data['prob'] = cur_model.predict_proba(X)[:, 1]
                
                # 排除当前持仓
                current_codes = [p['ts_code'] for p in holdings]
                candidates = day_data[~day_data['ts_code'].isin(current_codes)]
                
                # V16: 应用最低概率阈值
                candidates = candidates[candidates['prob'] >= min_prob]
                
                if not candidates.empty:
                    # 排序取 top
                    n_buy = max_positions - len(holdings)
                    picks = candidates.sort_values('prob', ascending=False).head(n_buy)
                    
                    # V16: 动态仓位 (position_pct 控制总投入比例)
                    cash_for_trading = capital * position_pct
                    cash_per = cash_for_trading / len(picks) if len(picks) > 0 else 0
                    
                    for _, row in picks.iterrows():
                        px_buy = prices.get((d_trade, row['ts_code']))
                        if px_buy:
                            up_limit = get_limit_price(row['ts_code'], px_buy['pre_close'], 'up')
                            if px_buy['open'] < up_limit:
                                buy_px = px_buy['open'] * (1 + SLIPPAGE)
                                shares = int(min(cash_per, capital) / buy_px / 100) * 100
                                if shares >= 100:
                                    capital -= (shares * buy_px + max(5, shares*buy_px*COMMISSION))
                                    holdings.append({
                                        'ts_code': row['ts_code'],
                                        'shares': shares,
                                        'buy_px': buy_px,
                                        'days_held': 0,
                                        'prob': row['prob']
                                    })
                                    trade_count_by_regime[regime] += 1
        
        mv = sum(p['shares'] * prices.get((d_trade, p['ts_code']), {'close': p['buy_px']})['close'] for p in holdings)
        equity.append({'date': d_trade, 'nav': capital + mv})
        regime_log.append({'date': d_trade, 'regime': regime, 'positions': len(holdings)})
    
    print(f"\n  交易次数统计: 牛市={trade_count_by_regime[1]}, 震荡={trade_count_by_regime[0]}, 熊市={trade_count_by_regime[-1]}")
    
    return pd.DataFrame(equity), pd.DataFrame(regime_log)

if __name__ == "__main__":
    df = load_data('20200101', '20260101')
    df = build_features(df)
    eq_df, regime_df = run_backtest(df)
    
    # === 计算指标 ===
    total_ret = (eq_df['nav'].iloc[-1] / INITIAL_CAP - 1) * 100
    days = (eq_df['date'].iloc[-1] - eq_df['date'].iloc[0]).days
    annual_ret = ((eq_df['nav'].iloc[-1] / INITIAL_CAP) ** (365/days) - 1) * 100
    
    # 最大回撤
    peak = eq_df['nav'].cummax()
    drawdown = (eq_df['nav'] - peak) / peak
    max_dd = drawdown.min() * 100
    
    # 夏普
    daily_ret = eq_df['nav'].pct_change().dropna()
    sharpe = (daily_ret.mean() / daily_ret.std() * np.sqrt(252)) if daily_ret.std() > 0 else 0
    
    # 卡玛 (Calmar)
    calmar = annual_ret / abs(max_dd) if max_dd != 0 else 0
    
    # 胜率
    wins = (daily_ret > 0).sum()
    win_rate = wins / len(daily_ret) * 100 if len(daily_ret) > 0 else 0
    
    print(f"\n{'='*60}")
    print(f"  V16 Market Regime Adaptive - Performance")
    print(f"{'='*60}")
    print(f"  总收益:     {total_ret:+.2f}%")
    print(f"  年化收益:   {annual_ret:+.2f}%")
    print(f"  最大回撤:   {max_dd:.2f}%")
    print(f"  夏普比率:   {sharpe:.2f}")
    print(f"  卡玛比率:   {calmar:.2f}")
    print(f"  日胜率:     {win_rate:.1f}%")
    print(f"  回测天数:   {days}")
    print(f"{'='*60}")
    
    # 市场状态分布
    print(f"\n  市场状态分布:")
    regime_names = {1: 'Bull', 0: 'Neutral', -1: 'Bear'}
    for r, name in regime_names.items():
        count = (regime_df['regime'] == r).sum()
        pct = count / len(regime_df) * 100
        print(f"    {name:8s}: {count:4d} days ({pct:.1f}%)")
    
    # 保存净值曲线
    eq_df.to_csv(os.path.join(OUT_DIR, 'super_weekly_v16_equity.csv'), index=False)
    
    # 绘图
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. 净值曲线
    ax1 = axes[0, 0]
    ax1.plot(eq_df['date'], eq_df['nav'], linewidth=1.5, color='#2196F3', label='V16 Adaptive')
    ax1.set_title('V16 Market Regime Adaptive - Equity Curve', fontsize=12)
    ax1.set_ylabel('NAV')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. 回撤
    ax2 = axes[0, 1]
    ax2.fill_between(eq_df['date'], drawdown * 100, 0, alpha=0.4, color='red')
    ax2.set_title('Drawdown (%)', fontsize=12)
    ax2.set_ylabel('Drawdown %')
    ax2.grid(True, alpha=0.3)
    
    # 3. 市场状态背景
    ax3 = axes[1, 0]
    regime_colors = {1: '#4CAF50', 0: '#FFC107', -1: '#F44336'}
    for i in range(len(regime_df)-1):
        r = regime_df.iloc[i]['regime']
        ax3.axvspan(regime_df.iloc[i]['date'], regime_df.iloc[i+1]['date'],
                    alpha=0.3, color=regime_colors.get(r, 'gray'))
    ax3.plot(eq_df['date'], eq_df['nav'], linewidth=1.5, color='black')
    ax3.set_title('NAV with Market Regime (Green=Bull, Yellow=Neutral, Red=Bear)', fontsize=10)
    ax3.set_ylabel('NAV')
    ax3.grid(True, alpha=0.3)
    
    # 4. 年度收益
    ax4 = axes[1, 1]
    eq_df['year'] = eq_df['date'].dt.year
    yearly = eq_df.groupby('year').apply(lambda x: (x['nav'].iloc[-1] / x['nav'].iloc[0] - 1) * 100)
    colors = ['#4CAF50' if v > 0 else '#F44336' for v in yearly]
    ax4.bar(yearly.index, yearly.values, color=colors, alpha=0.8)
    ax4.set_title('Yearly Returns (%)', fontsize=12)
    ax4.set_ylabel('Return %')
    ax4.axhline(y=0, color='black', linewidth=0.5)
    ax4.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'weekly_v16_curve.png'), dpi=150)
    print(f"\n  Chart saved: weekly_v16_curve.png")
    print(f"  Equity saved: super_weekly_v16_equity.csv")

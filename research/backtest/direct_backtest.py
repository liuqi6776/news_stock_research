"""
直接运行回测，不使用类封装，减少复杂度
"""
import pandas as pd
import numpy as np
import os
import pickle
import json
import gc
from datetime import datetime
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif

# ============ 配置 ============
COST_RATE = 0.003
SLIPPAGE = 0.002
LIMIT_THRESHOLD = 0.5

def is_main_board(ts_code):
    return ts_code.startswith(('60', '00', '002', '003'))

def get_limit_pct(ts_code):
    if ts_code.startswith(('688', '689', '30', '301')):
        return 20.0
    if ts_code.startswith(('8', '4')):
        return 30.0
    return 10.0

# ============ 路径 ============
data_dir = r'D:\iquant_data\data_v2'
price_dir = os.path.join(data_dir, 'data_day1')
rank_dir = os.path.join(data_dir, 'ths_rank1')
chip_dir = os.path.join(data_dir, 'cyq1')
other_dir = os.path.join(data_dir, 'other_day1')
output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'results')
model_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')
news_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'news_major1')

os.makedirs(output_dir, exist_ok=True)
os.makedirs(model_dir, exist_ok=True)

print(f"数据目录: {price_dir}")
print(f"输出目录: {output_dir}")
print(f"模型目录: {model_dir}")

# ============ 加载新闻 ============
def load_news(news_dir):
    market_records, stock_records = [], []
    if not os.path.exists(news_dir):
        return pd.DataFrame(market_records), pd.DataFrame(stock_records)
    for filename in os.listdir(news_dir):
        if not filename.endswith('.json'):
            continue
        try:
            with open(os.path.join(news_dir, filename), 'r', encoding='utf-8') as f:
                data = json.load(f)
        except:
            continue
        date_str = data.get("article_date", "")
        if not date_str:
            continue
        trade_date = pd.to_datetime(date_str).strftime('%Y%m%d')
        market_impact = data.get("market_impact", 0)
        market_records.append({'trade_date': trade_date, 'news_market_impact': float(market_impact)})
        for s in data.get("stocks", []):
            code = s.get("stock_code", "")
            if not code:
                continue
            ts_code = f"{code}.SH" if code.startswith('6') else f"{code}.SZ"
            stock_records.append({'trade_date': trade_date, 'ts_code': ts_code, 'news_stock_impact': float(s.get("impact", 0))})
    return pd.DataFrame(market_records), pd.DataFrame(stock_records)

news_mkt, news_stk = load_news(news_dir)
if not news_mkt.empty:
    print(f"新闻: {len(news_mkt)} 市场, {len(news_stk)} 个股")

# ============ 特征计算 ============
def calc_features(hist_df):
    if len(hist_df) < 20:
        return None
    df = hist_df.sort_values('trade_date').reset_index(drop=True).copy()

    # 缠论
    df['top_fractal'] = ((df['high'] > df['high'].shift(1)) & (df['high'] > df['high'].shift(2)) &
                         (df['high'] > df['high'].shift(-1)) & (df['high'] > df['high'].shift(-2))).astype(int)
    df['bottom_fractal'] = ((df['low'] < df['low'].shift(1)) & (df['low'] < df['low'].shift(2)) &
                            (df['low'] < df['low'].shift(-1)) & (df['low'] < df['low'].shift(-2))).astype(int)

    df['bi_direction'] = 0
    df.loc[df['bottom_fractal'].diff().fillna(0) > 0, 'bi_direction'] = 1
    df.loc[df['top_fractal'].diff().fillna(0) > 0, 'bi_direction'] = -1

    df['zhongshu_high'] = df['high'].rolling(20).max().shift(1)
    df['zhongshu_low'] = df['low'].rolling(20).min().shift(1)
    df['zhongshu'] = (df['zhongshu_high'] > df['zhongshu_low']).astype(int)
    df['zhongshu_strength'] = (df['zhongshu_high'] - df['zhongshu_low']) / (df['zhongshu_low'] + 1e-8)

    ema_fast = df['close'].ewm(span=12).mean()
    ema_slow = df['close'].ewm(span=26).mean()
    macd = ema_fast - ema_slow
    df['top_divergence'] = ((df['high'] > df['high'].shift(1)) & (macd < macd.shift(1))).astype(int)
    df['bottom_divergence'] = ((df['low'] < df['low'].shift(1)) & (macd > macd.shift(1))).astype(int)

    for w in [5, 10, 20]:
        df[f'mom_{w}d'] = df['close'].pct_change(w)
        df[f'vol_{w}d'] = df['close'].pct_change().rolling(w).std() * np.sqrt(252)
        df[f'vol_ratio_{w}d'] = df['vol'] / df['vol'].rolling(w).mean()

    df['amount'] = df['close'] * df['vol']
    for w in [5, 10, 20]:
        df[f'amount_ratio_{w}d'] = df['amount'] / df['amount'].rolling(w).mean()

    for w in [5, 10, 20, 60]:
        df[f'ma_dist_{w}d'] = (df['close'] - df['close'].rolling(w).mean()) / df['close'].rolling(w).mean()

    df['ma_bull'] = ((df['close'].rolling(5).mean() > df['close'].rolling(10).mean()) &
                     (df['close'].rolling(10).mean() > df['close'].rolling(20).mean())).astype(int)
    df['ma_bear'] = ((df['close'].rolling(5).mean() < df['close'].rolling(10).mean()) &
                     (df['close'].rolling(10).mean() < df['close'].rolling(20).mean())).astype(int)

    for w in [6, 12, 24]:
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(w).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(w).mean()
        df[f'rsi_{w}d'] = 100 - (100 / (1 + gain / loss))

    df['macd'] = df['close'].ewm(span=12).mean() - df['close'].ewm(span=26).mean()
    df['macd_hist'] = df['macd'] - df['macd'].ewm(span=9).mean()

    bb_m = df['close'].rolling(20).mean()
    bb_s = df['close'].rolling(20).std()
    df['bb_width'] = 4 * bb_s / bb_m
    df['bb_position'] = (df['close'] - bb_m + 2*bb_s) / (4*bb_s + 1e-8)

    low_list = df['low'].rolling(9, min_periods=9).min()
    high_list = df['high'].rolling(9, min_periods=9).max()
    rsv = (df['close'] - low_list) / (high_list - low_list) * 100
    df['kdj_k'] = rsv.ewm(com=2).mean()
    df['kdj_d'] = df['kdj_k'].ewm(com=2).mean()
    df['kdj_j'] = 3*df['kdj_k'] - 2*df['kdj_d']

    return df.iloc[-1]

# ============ 数据加载 ============
all_dates = sorted([f.replace('.parquet', '') for f in os.listdir(price_dir) if f.endswith('.parquet')])
print(f"总交易日: {len(all_dates)}")
print(f"范围: {all_dates[0]} - {all_dates[-1]}")

price_cache = {}
cache_dates = []

def get_day(date):
    if date in price_cache:
        return price_cache[date]

    p = os.path.join(price_dir, f"{date}.parquet")
    if not os.path.exists(p):
        return None
    try:
        df = pd.read_parquet(p)
    except:
        return None

    df = df[df['ts_code'].apply(is_main_board)]
    if len(df) == 0:
        return None

    df['price_change'] = (df['close'] - df['pre_close']) / df['pre_close']
    df['body_size'] = abs(df['close'] - df['open']) / df['pre_close']
    df['amplitude'] = (df['high'] - df['low']) / df['pre_close']
    df['trade_date'] = date

    # rank
    pr = os.path.join(rank_dir, f"{date}.parquet")
    if os.path.exists(pr):
        try:
            rdf = pd.read_parquet(pr)
            rdf = rdf.sort_values('hot', ascending=False).drop_duplicates('ts_code', keep='first')
            rdf['hot_rank_pct'] = rdf['hot'].rank(pct=True)
            df = df.merge(rdf[['ts_code', 'hot_rank_pct']], on='ts_code', how='left')
        except:
            df['hot_rank_pct'] = 0.5
    else:
        df['hot_rank_pct'] = 0.5

    # chip
    pc = os.path.join(chip_dir, f"{date}.parquet")
    if os.path.exists(pc):
        try:
            cdf = pd.read_parquet(pc)
            if 'cost_85pct' in cdf.columns:
                cdf['chip_concentration'] = (cdf['cost_85pct'] - cdf['cost_15pct']) / (cdf['cost_50pct'] + 1e-8)
            else:
                cdf['chip_concentration'] = 0.1
            if 'winner_rate' not in cdf.columns:
                cdf['winner_rate'] = 50.0
            df = df.merge(cdf[['ts_code', 'chip_concentration', 'winner_rate']], on='ts_code', how='left')
        except:
            df['chip_concentration'] = 0.1
            df['winner_rate'] = 50.0
    else:
        df['chip_concentration'] = 0.1
        df['winner_rate'] = 50.0

    # other
    po = os.path.join(other_dir, f"{date}.parquet")
    if os.path.exists(po):
        try:
            odf = pd.read_parquet(po, columns=['ts_code', 'circ_mv', 'turnover_rate', 'volume_ratio'])
            df = df.merge(odf, on='ts_code', how='left')
        except:
            df['circ_mv'] = 0
            df['turnover_rate'] = 0
            df['volume_ratio'] = 1
    else:
        df['circ_mv'] = 0
        df['turnover_rate'] = 0
        df['volume_ratio'] = 1

    # news
    if not news_mkt.empty:
        nm = news_mkt[news_mkt['trade_date'] == date]
        df['news_market_impact'] = nm['news_market_impact'].max() if not nm.empty else 0.0
    else:
        df['news_market_impact'] = 0.0

    if not news_stk.empty:
        ns = news_stk[news_stk['trade_date'] == date]
        if not ns.empty and 'ts_code' in ns.columns:
            ns_agg = ns.groupby('ts_code')['news_stock_impact'].max().reset_index()
            df = df.merge(ns_agg[['ts_code', 'news_stock_impact']], on='ts_code', how='left')
            df['news_stock_impact'] = df['news_stock_impact'].fillna(0.0)
        else:
            df['news_stock_impact'] = 0.0
    else:
        df['news_stock_impact'] = 0.0

    price_cache[date] = df
    cache_dates.append(date)
    if len(cache_dates) > 80:
        old = cache_dates.pop(0)
        if old in price_cache:
            del price_cache[old]

    return df

def get_stock_hist(ts_code, end_date, n_days=70):
    try:
        end_idx = all_dates.index(end_date)
    except ValueError:
        return None

    start_idx = max(0, end_idx - n_days + 1)
    dates = all_dates[start_idx:end_idx+1]

    data = []
    for d in dates:
        df = get_day(d)
        if df is not None:
            row = df[df['ts_code'] == ts_code]
            if not row.empty:
                data.append(row.iloc[0])

    if len(data) < 20:
        return None

    return pd.DataFrame(data)

# ============ 训练数据准备 ============
def prepare_data(dates):
    print(f"处理 {len(dates)} 天...")
    all_samples = []

    for i in range(len(dates) - 2):
        d_curr = dates[i]
        d_t1 = dates[i+1]
        d_t2 = dates[i+2]

        df_t = get_day(d_curr)
        if df_t is None:
            continue

        df_t1 = get_day(d_t1)
        df_t2 = get_day(d_t2)
        if df_t1 is None or df_t2 is None:
            continue

        # 只处理活跃股票
        df_t_sorted = df_t.sort_values('vol', ascending=False)
        n_stocks = max(100, int(len(df_t_sorted) * 0.6))
        active_stocks = df_t_sorted.iloc[:n_stocks]['ts_code'].unique()

        for ts_code in active_stocks:
            hist = get_stock_hist(ts_code, d_curr, n_days=70)
            if hist is None:
                continue

            feat = calc_features(hist)
            if feat is None:
                continue

            t1_row = df_t1[df_t1['ts_code'] == ts_code]
            t2_row = df_t2[df_t2['ts_code'] == ts_code]
            if t1_row.empty or t2_row.empty:
                continue

            t1_open = t1_row.iloc[0]['open']
            t2_close = t2_row.iloc[0]['close']

            if pd.isna(t1_open) or pd.isna(t2_close):
                continue

            label_ret = t2_close / t1_open - 1
            label = 1 if label_ret > 0.02 else 0

            sample = {
                'ts_code': ts_code, 'trade_date': d_curr, 'label': label, 'label_ret': label_ret,
                'price_change': feat['price_change'], 'body_size': feat['body_size'],
                'amplitude': feat['amplitude'], 'hot_rank_pct': feat['hot_rank_pct'],
                'chip_concentration': feat['chip_concentration'], 'winner_rate': feat['winner_rate'],
                'circ_mv': feat['circ_mv'], 'turnover_rate': feat['turnover_rate'], 'volume_ratio': feat['volume_ratio'],
                'top_fractal': feat['top_fractal'], 'bottom_fractal': feat['bottom_fractal'],
                'bi_direction': feat['bi_direction'], 'zhongshu': feat['zhongshu'],
                'zhongshu_strength': feat['zhongshu_strength'],
                'top_divergence': feat['top_divergence'], 'bottom_divergence': feat['bottom_divergence'],
                'mom_5d': feat['mom_5d'], 'mom_10d': feat['mom_10d'], 'mom_20d': feat['mom_20d'],
                'vol_5d': feat['vol_5d'], 'vol_10d': feat['vol_10d'], 'vol_20d': feat['vol_20d'],
                'vol_ratio_5d': feat['vol_ratio_5d'], 'vol_ratio_10d': feat['vol_ratio_10d'], 'vol_ratio_20d': feat['vol_ratio_20d'],
                'amount_ratio_5d': feat['amount_ratio_5d'], 'amount_ratio_10d': feat['amount_ratio_10d'],
                'ma_dist_5d': feat['ma_dist_5d'], 'ma_dist_10d': feat['ma_dist_10d'],
                'ma_dist_20d': feat['ma_dist_20d'], 'ma_dist_60d': feat['ma_dist_60d'],
                'ma_bull': feat['ma_bull'], 'ma_bear': feat['ma_bear'],
                'rsi_6d': feat['rsi_6d'], 'rsi_12d': feat['rsi_12d'], 'rsi_24d': feat['rsi_24d'],
                'macd': feat['macd'], 'macd_hist': feat['macd_hist'],
                'bb_width': feat['bb_width'], 'bb_position': feat['bb_position'],
                'kdj_k': feat['kdj_k'], 'kdj_d': feat['kdj_d'], 'kdj_j': feat['kdj_j'],
                'news_market_impact': feat['news_market_impact'], 'news_stock_impact': feat['news_stock_impact'],
            }
            all_samples.append(sample)

        if (i + 1) % 100 == 0:
            print(f"  已处理 {i+1}/{len(dates)-2} 天, 样本: {len(all_samples)}")
            gc.collect()

    if not all_samples:
        return pd.DataFrame(), []

    full = pd.DataFrame(all_samples)
    feature_cols = [c for c in full.columns if c not in ['ts_code', 'trade_date', 'label', 'label_ret']]
    full[feature_cols] = full[feature_cols].fillna(0)
    print(f"最终样本: {len(full)}, 正样本: {full['label'].mean():.2%}")
    return full, feature_cols

# ============ 模型训练 ============
def select_features(X, y, feature_cols, k=20):
    if X.shape[1] <= k:
        return feature_cols, X
    selector = SelectKBest(f_classif, k=k)
    X_selected = selector.fit_transform(X, y)
    mask = selector.get_support()
    selected = [feature_cols[i] for i in range(len(feature_cols)) if mask[i]]
    return selected, X_selected

def train_model(train_df, feature_cols):
    X = train_df[feature_cols].fillna(0)
    y = train_df['label']
    selected, X_sel = select_features(X, y, feature_cols, k=min(20, len(feature_cols)))
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_sel)
    pos_weight = len(y) / y.sum() - 1 if y.sum() > 0 else 1
    model = xgb.XGBClassifier(
        n_estimators=100, max_depth=4, learning_rate=0.05,
        subsample=0.7, colsample_bytree=0.7, random_state=42,
        eval_metric='auc', n_jobs=4, tree_method='hist',
        scale_pos_weight=pos_weight, reg_alpha=0.1, reg_lambda=1.0, min_child_weight=5
    )
    model.fit(X_scaled, y)
    return model, scaler, selected

# ============ 预测 ============
def predict_day(model, scaler, features, date):
    df = get_day(date)
    if df is None:
        return pd.DataFrame()

    df_sorted = df.sort_values('vol', ascending=False)
    n_stocks = max(100, int(len(df_sorted) * 0.6))
    active_stocks = df_sorted.iloc[:n_stocks]['ts_code'].unique()

    preds = []
    for ts_code in active_stocks:
        hist = get_stock_hist(ts_code, date, n_days=70)
        if hist is None:
            continue
        feat = calc_features(hist)
        if feat is None:
            continue
        preds.append({
            'ts_code': ts_code, 'trade_date': date,
            **{c: feat[c] for c in features if c in feat}
        })

    if not preds:
        return pd.DataFrame()

    pred_df = pd.DataFrame(preds)
    for c in features:
        if c not in pred_df.columns:
            pred_df[c] = 0.0
    pred_df[features] = pred_df[features].fillna(0)
    X = pred_df[features]
    X_scaled = scaler.transform(X)
    pred_df['prob'] = model.predict_proba(X_scaled)[:, 1]
    return pred_df[['ts_code', 'trade_date', 'prob'] + features]

# ============ 回测 ============
def backtest(predictions_df, test_dates, min_prob=0.55):
    trades = []
    skipped_up = skipped_down = 0
    capital = 100000.0
    daily_nav = []

    pred_lookup = {}
    for _, row in predictions_df.iterrows():
        pred_lookup[(row['ts_code'], str(row['trade_date']))] = row['prob']

    for i in range(len(test_dates) - 2):
        d_curr, d_t1, d_t2 = test_dates[i], test_dates[i+1], test_dates[i+2]

        day_df = get_day(d_curr)
        if day_df is None:
            daily_nav.append({'date': d_t2, 'nav': capital})
            continue

        best_prob, best_code = 0, None
        for _, row in day_df.iterrows():
            tc = row['ts_code']
            prob = pred_lookup.get((tc, d_curr), 0)
            if prob > best_prob:
                best_prob = prob
                best_code = tc

        if best_prob < min_prob or best_code is None:
            daily_nav.append({'date': d_t2, 'nav': capital})
            continue

        p1 = os.path.join(price_dir, f"{d_t1}.parquet")
        p2 = os.path.join(price_dir, f"{d_t2}.parquet")
        if not os.path.exists(p1) or not os.path.exists(p2):
            daily_nav.append({'date': d_t2, 'nav': capital})
            continue

        try:
            df1 = pd.read_parquet(p1)
            df2 = pd.read_parquet(p2)
        except:
            daily_nav.append({'date': d_t2, 'nav': capital})
            continue

        t1 = df1[df1['ts_code'] == best_code]
        t2 = df2[df2['ts_code'] == best_code]
        if t1.empty or t2.empty:
            daily_nav.append({'date': d_t2, 'nav': capital})
            continue

        t1_open = float(t1.iloc[0]['open'])
        t1_pre = float(t1.iloc[0]['pre_close'])
        t2_close = float(t2.iloc[0]['close'])
        t2_low = float(t2.iloc[0]['low'])
        t2_open = float(t2.iloc[0]['open'])

        limit_pct = get_limit_pct(best_code)

        if (t1_open - t1_pre) / t1_pre * 100 >= (limit_pct - LIMIT_THRESHOLD):
            skipped_up += 1
            daily_nav.append({'date': d_t2, 'nav': capital})
            continue

        if (t2_low - t1_open) / t1_open * 100 <= -(limit_pct - LIMIT_THRESHOLD):
            sell_price = min(t2_open, t1_open * (1 - limit_pct/100))
            skipped_down += 1
        else:
            sell_price = t2_close

        buy_price = t1_open * (1 + SLIPPAGE)
        sell_price = sell_price * (1 - SLIPPAGE)
        ret = sell_price / buy_price - 1 - COST_RATE
        capital *= (1 + ret)

        trades.append({
            'date_t': d_curr, 'date_t1': d_t1, 'date_t2': d_t2,
            'ts_code': best_code, 'prob': best_prob,
            'buy_price': buy_price, 'sell_price': sell_price, 'return': ret
        })
        daily_nav.append({'date': d_t2, 'nav': capital})

    if len(trades) == 0:
        return {'return': 0, 'sharpe': 0, 'max_dd': 0, 'n_trades': 0,
                'win_rate': 0, 'avg_return': 0, 'skipped_limit_up': skipped_up,
                'skipped_limit_down': skipped_down, 'daily_nav': daily_nav, 'trades': pd.DataFrame()}

    tdf = pd.DataFrame(trades)
    total_ret = capital / 100000.0 - 1
    nav = pd.DataFrame(daily_nav)
    nav['ret'] = nav['nav'].pct_change()
    vol = nav['ret'].std() * np.sqrt(252)
    sharpe = (total_ret / (len(test_dates)/252)) / vol if vol > 0 else 0
    nav['cummax'] = nav['nav'].cummax()
    nav['dd'] = (nav['nav'] - nav['cummax']) / nav['cummax']
    max_dd = nav['dd'].min()

    return {
        'return': total_ret, 'sharpe': sharpe, 'max_dd': max_dd,
        'n_trades': len(trades), 'win_rate': (tdf['return'] > 0).mean(),
        'avg_return': tdf['return'].mean(),
        'skipped_limit_up': skipped_up, 'skipped_limit_down': skipped_down,
        'daily_nav': daily_nav, 'trades': tdf
    }

# ============ 主程序 ============
def main():
    start_date = '20210101'
    end_date = '20260331'
    train_years = 1
    test_years = 1
    min_prob = 0.55

    print("="*80)
    print("V6 Stream 回测")
    print("="*80)
    print(f"回测期: {start_date} - {end_date}")
    print(f"训练: {train_years}年, 测试: {test_years}年, 阈值: {min_prob}")
    print("="*80)

    dates = [d for d in all_dates if start_date <= d <= end_date]
    if len(dates) < train_years * 252 + test_years * 252:
        print("数据不足!")
        return []

    years = []
    cy = dates[0][:4]
    yd = []
    for d in dates:
        if d[:4] == cy:
            yd.append(d)
        else:
            years.append((cy, yd))
            cy = d[:4]
            yd = [d]
    if yd:
        years.append((cy, yd))

    print(f"总年数: {len(years)}")

    results = []
    all_equity = []
    all_trades = []

    for i in range(train_years, len(years), test_years):
        train_dates = []
        for _, ydates in years[i-train_years:i]:
            train_dates.extend(ydates)
        test_dates = []
        for _, ydates in years[i:i+test_years]:
            test_dates.extend(ydates)

        if len(test_dates) < 20:
            continue

        period = years[i][0]
        print(f"\n{'='*60}")
        print(f"回测年份: {period}")
        print(f"训练: {train_dates[0]} - {train_dates[-1]} ({len(train_dates)}天)")
        print(f"测试: {test_dates[0]} - {test_dates[-1]} ({len(test_dates)}天)")
        print(f"{'='*60}")

        print("准备训练数据...")
        train_df, feature_cols = prepare_data(train_dates)
        if len(train_df) == 0:
            print("训练数据不足，跳过")
            continue

        print(f"训练样本: {len(train_df)}, 特征: {len(feature_cols)}")
        print("训练模型...")
        model, scaler, selected = train_model(train_df, feature_cols)

        importance = model.feature_importances_
        top = [selected[i] for i in np.argsort(importance)[-10:]]
        print(f"Top特征: {', '.join(reversed(top))}")

        # 保存模型
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        mp = os.path.join(model_dir, f"model_v6_{period}_{ts}.pkl")
        sp = os.path.join(model_dir, f"scaler_v6_{period}_{ts}.pkl")
        fp = os.path.join(model_dir, f"features_v6_{period}_{ts}.json")
        with open(mp, 'wb') as f: pickle.dump(model, f)
        with open(sp, 'wb') as f: pickle.dump(scaler, f)
        with open(fp, 'w') as f: json.dump(selected, f)
        print(f"模型已保存: {mp}")

        print("生成预测...")
        all_preds = []
        for j, d in enumerate(test_dates):
            pred = predict_day(model, scaler, selected, d)
            if not pred.empty:
                all_preds.append(pred)
            if (j + 1) % 50 == 0:
                print(f"  预测进度: {j+1}/{len(test_dates)}")
                gc.collect()

        if all_preds:
            pred_df = pd.concat(all_preds, ignore_index=True)
            pred_file = os.path.join(model_dir, f"predictions_v6_{period}.csv")
            pred_df.to_csv(pred_file, index=False)
            print(f"预测已保存: {pred_file}")

            print("回测...")
            bt = backtest(pred_df, test_dates, min_prob)
        else:
            bt = {'return': 0, 'sharpe': 0, 'max_dd': 0, 'n_trades': 0,
                  'win_rate': 0, 'avg_return': 0, 'skipped_limit_up': 0,
                  'skipped_limit_down': 0, 'daily_nav': [], 'trades': pd.DataFrame()}

        print(f"收益: {bt['return']*100:.2f}%, 夏普: {bt['sharpe']:.2f}, 回撤: {bt['max_dd']*100:.2f}%")
        print(f"交易: {bt['n_trades']}, 胜率: {bt['win_rate']*100:.2f}%")

        results.append({
            'period': period, 'train_start': train_dates[0], 'train_end': train_dates[-1],
            'test_start': test_dates[0], 'test_end': test_dates[-1],
            'train_samples': len(train_df), 'test_return': bt['return'],
            'test_sharpe': bt['sharpe'], 'test_max_dd': bt['max_dd'],
            'n_trades': bt['n_trades'], 'win_rate': bt['win_rate'],
            'avg_return': bt['avg_return'], 'top_features': top
        })

        all_equity.extend(bt['daily_nav'])
        if bt['n_trades'] > 0:
            all_trades.append(bt['trades'])

        price_cache.clear()
        cache_dates = []
        gc.collect()

    # 保存结果
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')

    if results:
        rdf = pd.DataFrame([{
            'period': r['period'], 'train_start': r['train_start'], 'train_end': r['train_end'],
            'test_start': r['test_start'], 'test_end': r['test_end'],
            'train_samples': r['train_samples'], 'test_return': r['test_return'],
            'test_sharpe': r['test_sharpe'], 'test_max_dd': r['test_max_dd'],
            'n_trades': r['n_trades'], 'win_rate': r['win_rate'],
            'avg_return': r['avg_return'], 'top_features': ','.join(r['top_features'])
        } for r in results])
        rf = os.path.join(output_dir, f'yearly_results_v6_{ts}.csv')
        rdf.to_csv(rf, index=False)
        print(f"\n结果: {rf}")

    if all_equity:
        edf = pd.DataFrame(all_equity).drop_duplicates('date').sort_values('date')
        ef = os.path.join(output_dir, f'equity_curve_v6_{ts}.csv')
        edf.to_csv(ef, index=False)
        print(f"权益: {ef}")

        init_nav = edf['nav'].iloc[0]
        final_nav = edf['nav'].iloc[-1]
        total_ret = final_nav / init_nav - 1
        edf['ret'] = edf['nav'].pct_change()
        vol = edf['ret'].std() * np.sqrt(252)
        years = len(edf) / 252
        ann_ret = (1 + total_ret) ** (1/years) - 1 if years > 0 else 0
        sharpe = ann_ret / vol if vol > 0 else 0
        edf['cummax'] = edf['nav'].cummax()
        edf['dd'] = (edf['nav'] - edf['cummax']) / edf['cummax']
        max_dd = edf['dd'].min()

        print(f"\n{'='*60}")
        print("总体结果")
        print(f"{'='*60}")
        print(f"总收益: {total_ret*100:.2f}%")
        print(f"年化: {ann_ret*100:.2f}%")
        print(f"夏普: {sharpe:.2f}")
        print(f"最大回撤: {max_dd*100:.2f}%")
        print(f"{'='*60}")

    if all_trades:
        tdf = pd.concat(all_trades, ignore_index=True)
        tf = os.path.join(output_dir, f'all_trades_v6_{ts}.csv')
        tdf.to_csv(tf, index=False)
        print(f"交易: {tf}")
        print(f"总交易: {len(tdf)}, 胜率: {(tdf['return']>0).mean()*100:.2f}%, 平均: {tdf['return'].mean()*100:.2f}%")

    return results

if __name__ == '__main__':
    main()

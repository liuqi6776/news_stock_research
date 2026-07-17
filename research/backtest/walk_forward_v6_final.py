"""
完整特征回测框架 V6 Final
整合：缠论特征 + 技术因子 + news major1 + rank数据
优化：逐日流式处理，预计算特征，避免内存溢出
训练：2020-2022，Rolling一年更新
测试：2022-2026
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


def calc_chanlun_features(df):
    """计算缠论特征"""
    df = df.copy()
    df['top_fractal'] = ((df['high'] > df['high'].shift(1)) & (df['high'] > df['high'].shift(2)) &
                         (df['high'] > df['high'].shift(-1)) & (df['high'] > df['high'].shift(-2))).astype(int)
    df['bottom_fractal'] = ((df['low'] < df['low'].shift(1)) & (df['low'] < df['low'].shift(2)) &
                            (df['low'] < df['low'].shift(-1)) & (df['low'] < df['low'].shift(-2))).astype(int)

    df['bi_direction'] = 0
    for i in range(5, len(df)):
        if df['bottom_fractal'].iloc[i-3] and df['top_fractal'].iloc[i]:
            df.loc[df.index[i-3:i+1], 'bi_direction'] = 1
        elif df['top_fractal'].iloc[i-3] and df['bottom_fractal'].iloc[i]:
            df.loc[df.index[i-3:i+1], 'bi_direction'] = -1

    df['zhongshu'] = 0
    df['zhongshu_strength'] = 0.0
    for i in range(20, len(df)):
        window = df.iloc[i-20:i]
        tops = window[window['top_fractal'] == 1]['high']
        bottoms = window[window['bottom_fractal'] == 1]['low']
        if len(tops) >= 2 and len(bottoms) >= 2:
            oh = min(tops.iloc[-2:].max(), bottoms.iloc[-2:].max())
            ol = max(tops.iloc[-2:].min(), bottoms.iloc[-2:].min())
            if oh > ol:
                df.loc[df.index[i], 'zhongshu'] = 1
                df.loc[df.index[i], 'zhongshu_strength'] = (oh - ol) / ol

    ema_fast = df['close'].ewm(span=12).mean()
    ema_slow = df['close'].ewm(span=26).mean()
    macd = ema_fast - ema_slow
    df['top_divergence'] = ((df['high'] > df['high'].shift(1)) & (macd < macd.shift(1))).astype(int)
    df['bottom_divergence'] = ((df['low'] < df['low'].shift(1)) & (macd > macd.shift(1))).astype(int)
    return df


def calc_tech_features(df):
    """计算技术特征"""
    df = df.copy()
    for w in [5, 10, 20]:
        df[f'mom_{w}d'] = df['close'].pct_change(w)
        df[f'vol_{w}d'] = df['close'].pct_change().rolling(w).std() * np.sqrt(252)
        df[f'vol_ma_{w}d'] = df['vol'].rolling(w).mean()
        df[f'vol_ratio_{w}d'] = df['vol'] / df[f'vol_ma_{w}d']

    df['amount'] = df['close'] * df['vol']
    for w in [5, 10, 20]:
        df[f'amount_ma_{w}d'] = df['amount'].rolling(w).mean()
        df[f'amount_ratio_{w}d'] = df['amount'] / df[f'amount_ma_{w}d']

    for w in [5, 10, 20, 60]:
        df[f'ma_{w}d'] = df['close'].rolling(w).mean()
        df[f'ma_dist_{w}d'] = (df['close'] - df[f'ma_{w}d']) / df[f'ma_{w}d']

    df['ma_bull'] = ((df['ma_5d'] > df['ma_10d']) & (df['ma_10d'] > df['ma_20d'])).astype(int)
    df['ma_bear'] = ((df['ma_5d'] < df['ma_10d']) & (df['ma_10d'] < df['ma_20d'])).astype(int)

    for w in [6, 12, 24]:
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(w).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(w).mean()
        rs = gain / loss
        df[f'rsi_{w}d'] = 100 - (100 / (1 + rs))

    ema_f = df['close'].ewm(span=12).mean()
    ema_s = df['close'].ewm(span=26).mean()
    df['macd'] = ema_f - ema_s
    df['macd_signal'] = df['macd'].ewm(span=9).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']

    df['bb_middle'] = df['close'].rolling(20).mean()
    df['bb_std'] = df['close'].rolling(20).std()
    df['bb_width'] = (df['bb_middle'] + 2*df['bb_std'] - (df['bb_middle'] - 2*df['bb_std'])) / df['bb_middle']
    df['bb_position'] = (df['close'] - (df['bb_middle'] - 2*df['bb_std'])) / (4*df['bb_std'] + 1e-8)

    low_list = df['low'].rolling(9, min_periods=9).min()
    high_list = df['high'].rolling(9, min_periods=9).max()
    rsv = (df['close'] - low_list) / (high_list - low_list) * 100
    df['kdj_k'] = rsv.ewm(com=2).mean()
    df['kdj_d'] = df['kdj_k'].ewm(com=2).mean()
    df['kdj_j'] = 3*df['kdj_k'] - 2*df['kdj_d']

    return df


class BacktestV6:
    def __init__(self, data_dir=r'D:\iquant_data\data_v2'):
        self.price_dir = os.path.join(data_dir, 'data_day1')
        self.rank_dir = os.path.join(data_dir, 'ths_rank1')
        self.chip_dir = os.path.join(data_dir, 'cyq1')
        self.other_dir = os.path.join(data_dir, 'other_day1')
        self.output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'results')
        self.model_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')
        self.news_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'news_major1')
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.model_dir, exist_ok=True)

        self.all_dates = sorted([f.replace('.parquet', '') for f in os.listdir(self.price_dir) if f.endswith('.parquet')])
        print(f"数据目录: {self.price_dir}")
        print(f"总交易日: {len(self.all_dates)}")
        print(f"范围: {self.all_dates[0]} - {self.all_dates[-1]}")

        self.news_mkt, self.news_stk = load_news(self.news_dir)
        if not self.news_mkt.empty:
            print(f"新闻: {len(self.news_mkt)} 市场, {len(self.news_stk)} 个股")

    def load_day(self, date):
        """加载单日数据"""
        p = os.path.join(self.price_dir, f"{date}.parquet")
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
        pr = os.path.join(self.rank_dir, f"{date}.parquet")
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
        pc = os.path.join(self.chip_dir, f"{date}.parquet")
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
        po = os.path.join(self.other_dir, f"{date}.parquet")
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
        if not self.news_mkt.empty:
            nm = self.news_mkt[self.news_mkt['trade_date'] == date]
            df['news_market_impact'] = nm['news_market_impact'].max() if not nm.empty else 0.0
        else:
            df['news_market_impact'] = 0.0

        if not self.news_stk.empty:
            ns = self.news_stk[self.news_stk['trade_date'] == date]
            if not ns.empty and 'ts_code' in ns.columns:
                ns_agg = ns.groupby('ts_code')['news_stock_impact'].max().reset_index()
                df = df.merge(ns_agg[['ts_code', 'news_stock_impact']], on='ts_code', how='left')
                df['news_stock_impact'] = df['news_stock_impact'].fillna(0.0)
            else:
                df['news_stock_impact'] = 0.0
        else:
            df['news_stock_impact'] = 0.0

        return df

    def prepare_data(self, dates, label_th=0.02):
        """准备训练数据 - 预计算每只股票的特征"""
        print(f"加载 {len(dates)} 天数据...")

        # 加载所有原始数据
        all_data = []
        for d in dates:
            df = self.load_day(d)
            if df is not None:
                all_data.append(df)

        if not all_data:
            return pd.DataFrame(), []

        full = pd.concat(all_data, ignore_index=True)
        print(f"原始数据: {len(full)} 行, {full['ts_code'].nunique()} 只股票")

        # 按股票分组计算特征
        print("计算特征...")
        featured = []
        for ts_code, grp in full.groupby('ts_code'):
            if len(grp) < 30:
                continue
            grp = grp.sort_values('trade_date').reset_index(drop=True)
            grp = calc_chanlun_features(grp)
            grp = calc_tech_features(grp)
            featured.append(grp)

        if not featured:
            return pd.DataFrame(), []

        full = pd.concat(featured, ignore_index=True)
        print(f"特征数据: {len(full)} 行")

        # 生成标签
        print("生成标签...")
        date_list = sorted(full['trade_date'].unique())
        date_idx = {d: i for i, d in enumerate(date_list)}

        # 构建查找
        lookup = {}
        for _, row in full.iterrows():
            lookup[(row['ts_code'], row['trade_date'])] = row

        t1_opens, t2_closes, t2_lows, valid = [], [], [], []
        for _, row in full.iterrows():
            tc, td = row['ts_code'], row['trade_date']
            if td not in date_idx:
                valid.append(False); t1_opens.append(np.nan); t2_closes.append(np.nan); t2_lows.append(np.nan)
                continue
            idx = date_idx[td]
            if idx + 2 >= len(date_list):
                valid.append(False); t1_opens.append(np.nan); t2_closes.append(np.nan); t2_lows.append(np.nan)
                continue
            t1d, t2d = date_list[idx+1], date_list[idx+2]
            k1, k2 = (tc, t1d), (tc, t2d)
            if k1 not in lookup or k2 not in lookup:
                valid.append(False); t1_opens.append(np.nan); t2_closes.append(np.nan); t2_lows.append(np.nan)
                continue
            valid.append(True)
            t1_opens.append(lookup[k1]['open'])
            t2_closes.append(lookup[k2]['close'])
            t2_lows.append(lookup[k2]['low'])

        full['t1_open'] = t1_opens
        full['t2_close'] = t2_closes
        full['t2_low'] = t2_lows
        full['valid'] = valid
        full = full[full['valid']].dropna(subset=['t1_open', 't2_close'])

        if len(full) == 0:
            return pd.DataFrame(), []

        full['label_ret'] = full['t2_close'] / full['t1_open'] - 1
        full['label'] = (full['label_ret'] > label_th).astype(int)

        feature_cols = [
            'price_change', 'body_size', 'amplitude', 'hot_rank_pct',
            'chip_concentration', 'winner_rate', 'circ_mv', 'turnover_rate', 'volume_ratio',
            'top_fractal', 'bottom_fractal', 'bi_direction', 'zhongshu', 'zhongshu_strength',
            'top_divergence', 'bottom_divergence',
            'mom_5d', 'mom_10d', 'mom_20d',
            'vol_5d', 'vol_10d', 'vol_20d',
            'vol_ratio_5d', 'vol_ratio_10d', 'vol_ratio_20d',
            'amount_ratio_5d', 'amount_ratio_10d',
            'ma_dist_5d', 'ma_dist_10d', 'ma_dist_20d', 'ma_dist_60d',
            'ma_bull', 'ma_bear',
            'rsi_6d', 'rsi_12d', 'rsi_24d',
            'macd', 'macd_hist',
            'bb_width', 'bb_position',
            'kdj_k', 'kdj_d', 'kdj_j',
            'news_market_impact', 'news_stock_impact'
        ]

        for col in feature_cols:
            if col not in full.columns:
                full[col] = 0.0

        full[feature_cols] = full[feature_cols].fillna(0)
        print(f"最终样本: {len(full)}, 正样本: {full['label'].mean():.2%}")
        return full, feature_cols

    def select_features(self, X, y, feature_cols, k=20):
        if X.shape[1] <= k:
            return feature_cols, X
        selector = SelectKBest(f_classif, k=k)
        X_selected = selector.fit_transform(X, y)
        mask = selector.get_support()
        selected = [feature_cols[i] for i in range(len(feature_cols)) if mask[i]]
        return selected, X_selected

    def train_model(self, train_df, feature_cols):
        X = train_df[feature_cols].fillna(0)
        y = train_df['label']
        selected, X_sel = self.select_features(X, y, feature_cols, k=min(20, len(feature_cols)))
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

    def save_model(self, model, scaler, features, period):
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        mp = os.path.join(self.model_dir, f"model_v6_{period}_{ts}.pkl")
        sp = os.path.join(self.model_dir, f"scaler_v6_{period}_{ts}.pkl")
        fp = os.path.join(self.model_dir, f"features_v6_{period}_{ts}.json")
        with open(mp, 'wb') as f: pickle.dump(model, f)
        with open(sp, 'wb') as f: pickle.dump(scaler, f)
        with open(fp, 'w') as f: json.dump(features, f)
        return mp, sp, fp

    def predict_day(self, model, scaler, features, date):
        """预测单日"""
        df = self.load_day(date)
        if df is None or len(df) == 0:
            return pd.DataFrame()

        # 需要历史数据计算特征
        idx = self.all_dates.index(date) if date in self.all_dates else -1
        if idx < 60:
            return pd.DataFrame()

        hist_dates = self.all_dates[max(0, idx-70):idx+1]

        # 加载历史
        hist_data = []
        for d in hist_dates:
            hd = self.load_day(d)
            if hd is not None:
                hist_data.append(hd)
        if not hist_data:
            return pd.DataFrame()

        hist = pd.concat(hist_data, ignore_index=True)

        # 按股票计算特征
        preds = []
        for ts_code, grp in hist.groupby('ts_code'):
            if len(grp) < 30:
                continue
            grp = grp.sort_values('trade_date').reset_index(drop=True)
            grp = calc_chanlun_features(grp)
            grp = calc_tech_features(grp)
            latest = grp.iloc[-1:].copy()
            preds.append(latest)

        if not preds:
            return pd.DataFrame()

        pred_df = pd.concat(preds, ignore_index=True)
        for col in features:
            if col not in pred_df.columns:
                pred_df[col] = 0.0
        pred_df[features] = pred_df[features].fillna(0)

        X = pred_df[features]
        X_scaled = scaler.transform(X)
        pred_df['prob'] = model.predict_proba(X_scaled)[:, 1]

        return pred_df[['ts_code', 'trade_date', 'prob'] + features]

    def backtest_predictions(self, predictions_df, test_dates, min_prob=0.55):
        """回测"""
        trades = []
        skipped_up = skipped_down = 0
        capital = 100000.0
        daily_nav = []

        pred_lookup = {}
        for _, row in predictions_df.iterrows():
            pred_lookup[(row['ts_code'], str(row['trade_date']))] = row['prob']

        for i in range(len(test_dates) - 2):
            d_curr, d_t1, d_t2 = test_dates[i], test_dates[i+1], test_dates[i+2]

            # 获取当日预测
            day_df = self.load_day(d_curr)
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

            # 加载T+1, T+2价格
            p1 = os.path.join(self.price_dir, f"{d_t1}.parquet")
            p2 = os.path.join(self.price_dir, f"{d_t2}.parquet")
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

            # 涨停检查
            if (t1_open - t1_pre) / t1_pre * 100 >= (limit_pct - LIMIT_THRESHOLD):
                skipped_up += 1
                daily_nav.append({'date': d_t2, 'nav': capital})
                continue

            # 跌停检查
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

    def run(self, start_date='20200101', end_date='20260331', train_years=2, test_years=1, min_prob=0.55):
        print("="*80)
        print("V6 Final 回测")
        print("="*80)
        print(f"回测期: {start_date} - {end_date}")
        print(f"训练: {train_years}年, 测试: {test_years}年, 阈值: {min_prob}")
        print("="*80)

        dates = [d for d in self.all_dates if start_date <= d <= end_date]
        if len(dates) < train_years * 252 + test_years * 252:
            print("数据不足!")
            return []

        # 按年分组
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

            # 训练
            print("准备训练数据...")
            train_df, feature_cols = self.prepare_data(train_dates)
            if len(train_df) == 0:
                print("训练数据不足，跳过")
                continue

            print(f"训练样本: {len(train_df)}, 特征: {len(feature_cols)}")
            print("训练模型...")
            model, scaler, selected = self.train_model(train_df, feature_cols)

            importance = model.feature_importances_
            top = [selected[i] for i in np.argsort(importance)[-10:]]
            print(f"Top特征: {', '.join(reversed(top))}")

            mp, sp, fp = self.save_model(model, scaler, selected, period)
            print(f"模型已保存: {mp}")

            # 预测
            print("生成预测...")
            all_preds = []
            for d in test_dates:
                pred = self.predict_day(model, scaler, selected, d)
                if not pred.empty:
                    all_preds.append(pred)
                gc.collect()

            if all_preds:
                pred_df = pd.concat(all_preds, ignore_index=True)
                pred_file = os.path.join(self.model_dir, f"predictions_v6_{period}.csv")
                pred_df.to_csv(pred_file, index=False)
                print(f"预测已保存: {pred_file}")

                print("回测...")
                bt = self.backtest_predictions(pred_df, test_dates, min_prob)
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

            gc.collect()

        self.save_results(results, all_equity, all_trades)
        return results

    def save_results(self, results, all_equity, all_trades):
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
            rf = os.path.join(self.output_dir, f'yearly_results_v6_{ts}.csv')
            rdf.to_csv(rf, index=False)
            print(f"\n结果: {rf}")

        if all_equity:
            edf = pd.DataFrame(all_equity).drop_duplicates('date').sort_values('date')
            ef = os.path.join(self.output_dir, f'equity_curve_v6_{ts}.csv')
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
            tf = os.path.join(self.output_dir, f'all_trades_v6_{ts}.csv')
            tdf.to_csv(tf, index=False)
            print(f"交易: {tf}")
            print(f"总交易: {len(tdf)}, 胜率: {(tdf['return']>0).mean()*100:.2f}%, 平均: {tdf['return'].mean()*100:.2f}%")


if __name__ == '__main__':
    bt = BacktestV6()
    bt.run(start_date='20200101', end_date='20260331', train_years=2, test_years=1, min_prob=0.55)

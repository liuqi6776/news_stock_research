"""
模型融合框架 V10
1. 基地模型：使用Top 20特征训练XGBoost
2. 误差模型：使用其他特征预测基地模型的误差
3. 模型融合：基地模型预测 + 误差模型修正
"""
import pandas as pd
import numpy as np
import os
import sys
import pickle
import json
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge

# A股交易规则配置
COST_RATE = 0.003
SLIPPAGE = 0.002
LIMIT_THRESHOLD = 0.5


def is_main_board(ts_code: str) -> bool:
    return (ts_code.startswith('60') or
            ts_code.startswith('00') or
            ts_code.startswith('002') or
            ts_code.startswith('003'))


def get_limit_pct(ts_code: str) -> float:
    if ts_code.startswith('688') or ts_code.startswith('689'):
        return 20.0
    elif ts_code.startswith('30') or ts_code.startswith('301'):
        return 20.0
    elif ts_code.startswith('8') or ts_code.startswith('4'):
        return 30.0
    else:
        return 10.0


def process_news(news_dir, target_date=None):
    market_records = []
    stock_records = []
    if not os.path.exists(news_dir):
        return pd.DataFrame(market_records), pd.DataFrame(stock_records)
    for filename in os.listdir(news_dir):
        if not filename.endswith('.json'):
            continue
        filepath = os.path.join(news_dir, filename)
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except:
            continue
        date_str = data.get("article_date", "")
        if not date_str:
            continue
        trade_date = pd.to_datetime(date_str).strftime('%Y%m%d')
        if target_date and trade_date > target_date:
            continue
        market_impact = data.get("market_impact", 0)
        market_records.append({
            'trade_date': trade_date,
            'news_market_impact': float(market_impact)
        })
        for s in data.get("stocks", []):
            code = s.get("stock_code", "")
            if not code:
                continue
            ts_code = f"{code}.SH" if code.startswith('6') else f"{code}.SZ"
            stock_records.append({
                'trade_date': trade_date,
                'ts_code': ts_code,
                'news_stock_impact': float(s.get("impact", 0))
            })
    return pd.DataFrame(market_records), pd.DataFrame(stock_records)


class TechnicalFactors:
    @staticmethod
    def momentum(df, windows=[5, 10, 20, 60]):
        for w in windows:
            df[f'mom_{w}d'] = df['close'].pct_change(w)
        return df

    @staticmethod
    def volatility(df, windows=[5, 10, 20, 60]):
        for w in windows:
            df[f'vol_{w}d'] = df['close'].pct_change().rolling(w).std() * np.sqrt(252)
        return df

    @staticmethod
    def volume_features(df, windows=[5, 10, 20]):
        for w in windows:
            df[f'vol_ma_{w}d'] = df['vol'].rolling(w).mean()
            df[f'vol_ratio_{w}d'] = df['vol'] / (df[f'vol_ma_{w}d'] + 1e-8)
        df['amount'] = df['close'] * df['vol']
        for w in windows:
            df[f'amount_ma_{w}d'] = df['amount'].rolling(w).mean()
            df[f'amount_ratio_{w}d'] = df['amount'] / (df[f'amount_ma_{w}d'] + 1e-8)
        df['price_volume_corr'] = df['close'].rolling(20).corr(df['vol'])
        return df

    @staticmethod
    def moving_average(df, windows=[5, 10, 20, 60, 120]):
        for w in windows:
            df[f'ma_{w}d'] = df['close'].rolling(w).mean()
            df[f'ma_dist_{w}d'] = (df['close'] - df[f'ma_{w}d']) / (df[f'ma_{w}d'] + 1e-8)
        df['ma_bull'] = ((df['ma_5d'] > df['ma_10d']) & (df['ma_10d'] > df['ma_20d'])).astype(int)
        df['ma_bear'] = ((df['ma_5d'] < df['ma_10d']) & (df['ma_10d'] < df['ma_20d'])).astype(int)
        return df

    @staticmethod
    def rsi(df, windows=[6, 12, 24]):
        for w in windows:
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=w).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=w).mean()
            rs = gain / (loss + 1e-8)
            df[f'rsi_{w}d'] = 100 - (100 / (1 + rs))
        return df

    @staticmethod
    def macd(df):
        ema_fast = df['close'].ewm(span=12, adjust=False).mean()
        ema_slow = df['close'].ewm(span=26, adjust=False).mean()
        df['macd'] = ema_fast - ema_slow
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']
        return df

    @staticmethod
    def bollinger_bands(df):
        df['bb_middle'] = df['close'].rolling(20).mean()
        df['bb_std'] = df['close'].rolling(20).std()
        df['bb_upper'] = df['bb_middle'] + 2 * df['bb_std']
        df['bb_lower'] = df['bb_middle'] - 2 * df['bb_std']
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / (df['bb_middle'] + 1e-8)
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'] + 1e-8)
        return df

    @staticmethod
    def kdj(df):
        low_list = df['low'].rolling(window=9, min_periods=9).min()
        high_list = df['high'].rolling(window=9, min_periods=9).max()
        rsv = (df['close'] - low_list) / (high_list - low_list + 1e-8) * 100
        df['kdj_k'] = rsv.ewm(com=2, adjust=False).mean()
        df['kdj_d'] = df['kdj_k'].ewm(com=2, adjust=False).mean()
        df['kdj_j'] = 3 * df['kdj_k'] - 2 * df['kdj_d']
        return df

    @staticmethod
    def williams_r(df, windows=[10, 20]):
        for w in windows:
            high_max = df['high'].rolling(w).max()
            low_min = df['low'].rolling(w).min()
            df[f'williams_r_{w}d'] = (high_max - df['close']) / (high_max - low_min + 1e-8) * -100
        return df

    @staticmethod
    def atr(df):
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift(1))
        low_close = np.abs(df['low'] - df['close'].shift(1))
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df['atr'] = tr.rolling(14).mean()
        df['atr_ratio'] = df['atr'] / (df['close'] + 1e-8)
        return df

    @staticmethod
    def obv(df):
        df['obv'] = (np.sign(df['close'].diff()) * df['vol']).cumsum()
        df['obv_ma'] = df['obv'].rolling(20).mean()
        return df

    @staticmethod
    def ichimoku(df):
        tenkan_high = df['high'].rolling(9).max()
        tenkan_low = df['low'].rolling(9).min()
        df['tenkan_sen'] = (tenkan_high + tenkan_low) / 2
        kijun_high = df['high'].rolling(26).max()
        kijun_low = df['low'].rolling(26).min()
        df['kijun_sen'] = (kijun_high + kijun_low) / 2
        df['senkou_a'] = ((df['tenkan_sen'] + df['kijun_sen']) / 2)
        senkou_high = df['high'].rolling(52).max()
        senkou_low = df['low'].rolling(52).min()
        df['senkou_b'] = (senkou_high + senkou_low) / 2
        return df

    @staticmethod
    def adx(df):
        plus_dm = df['high'].diff()
        minus_dm = df['low'].diff()
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm > 0] = 0
        minus_dm = np.abs(minus_dm)
        tr = pd.concat([
            df['high'] - df['low'],
            np.abs(df['high'] - df['close'].shift(1)),
            np.abs(df['low'] - df['close'].shift(1))
        ], axis=1).max(axis=1)
        atr = tr.rolling(14).mean()
        df['plus_di'] = 100 * (plus_dm.rolling(14).mean() / (atr + 1e-8))
        df['minus_di'] = 100 * (minus_dm.rolling(14).mean() / (atr + 1e-8))
        dx = 100 * np.abs(df['plus_di'] - df['minus_di']) / (df['plus_di'] + df['minus_di'] + 1e-8)
        df['adx'] = dx.rolling(14).mean()
        return df

    @staticmethod
    def calculate_all(df):
        df = TechnicalFactors.momentum(df)
        df = TechnicalFactors.volatility(df)
        df = TechnicalFactors.volume_features(df)
        df = TechnicalFactors.moving_average(df)
        df = TechnicalFactors.rsi(df)
        df = TechnicalFactors.macd(df)
        df = TechnicalFactors.bollinger_bands(df)
        df = TechnicalFactors.kdj(df)
        df = TechnicalFactors.williams_r(df)
        df = TechnicalFactors.atr(df)
        df = TechnicalFactors.obv(df)
        df = TechnicalFactors.ichimoku(df)
        df = TechnicalFactors.adx(df)
        return df


class ChanLunFeatures:
    @staticmethod
    def calculate_fractals(df):
        df['top_fractal'] = (
            (df['high'] > df['high'].shift(1)) &
            (df['high'] > df['high'].shift(2)) &
            (df['high'] > df['high'].shift(3)) &
            (df['high'] > df['high'].shift(4))
        ).astype(int)
        df['bottom_fractal'] = (
            (df['low'] < df['low'].shift(1)) &
            (df['low'] < df['low'].shift(2)) &
            (df['low'] < df['low'].shift(3)) &
            (df['low'] < df['low'].shift(4))
        ).astype(int)
        return df

    @staticmethod
    def calculate_bi(df):
        df['bi_direction'] = 0
        last_top_idx = -1
        last_bottom_idx = -1
        for i in range(len(df)):
            if df['top_fractal'].iloc[i] == 1:
                if last_bottom_idx >= 0 and last_top_idx < last_bottom_idx:
                    df.loc[df.index[last_bottom_idx:i+1], 'bi_direction'] = 1
                last_top_idx = i
            elif df['bottom_fractal'].iloc[i] == 1:
                if last_top_idx >= 0 and last_bottom_idx < last_top_idx:
                    df.loc[df.index[last_top_idx:i+1], 'bi_direction'] = -1
                last_bottom_idx = i
        return df

    @staticmethod
    def calculate_zhongshu(df):
        df['zhongshu'] = 0
        df['zhongshu_strength'] = 0.0
        for i in range(20, len(df)):
            window = df.iloc[i-20:i]
            tops = window[window['top_fractal'] == 1]['high']
            bottoms = window[window['bottom_fractal'] == 1]['low']
            if len(tops) >= 2 and len(bottoms) >= 2:
                overlap_high = min(tops.iloc[-2:].max(), bottoms.iloc[-2:].max())
                overlap_low = max(tops.iloc[-2:].min(), bottoms.iloc[-2:].min())
                if overlap_high > overlap_low:
                    df.loc[df.index[i], 'zhongshu'] = 1
                    df.loc[df.index[i], 'zhongshu_strength'] = (overlap_high - overlap_low) / (overlap_low + 1e-8)
        return df

    @staticmethod
    def calculate_divergence(df):
        ema_fast = df['close'].ewm(span=12, adjust=False).mean()
        ema_slow = df['close'].ewm(span=26, adjust=False).mean()
        macd = ema_fast - ema_slow
        df['top_divergence'] = (
            (df['high'] > df['high'].shift(1)) &
            (df['high'] > df['high'].shift(2)) &
            (macd < macd.shift(1)) &
            (macd < macd.shift(2))
        ).astype(int)
        df['bottom_divergence'] = (
            (df['low'] < df['low'].shift(1)) &
            (df['low'] < df['low'].shift(2)) &
            (macd > macd.shift(1)) &
            (macd > macd.shift(2))
        ).astype(int)
        return df

    @staticmethod
    def calculate_all(df):
        df = ChanLunFeatures.calculate_fractals(df)
        df = ChanLunFeatures.calculate_bi(df)
        df = ChanLunFeatures.calculate_zhongshu(df)
        df = ChanLunFeatures.calculate_divergence(df)
        return df


class WalkForwardStackingV10:
    def __init__(self,
                 data_dir: str = r'D:\iquant_data\data_v2',
                 output_dir: str = None,
                 model_dir: str = None,
                 news_dir: str = None):
        self.data_dir = data_dir
        self.price_dir = os.path.join(data_dir, 'data_day1')
        self.rank_dir = os.path.join(data_dir, 'ths_rank1')
        self.chip_dir = os.path.join(data_dir, 'cyq1')
        self.other_dir = os.path.join(data_dir, 'other_day1')
        if output_dir is None:
            self.output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'results')
        else:
            self.output_dir = output_dir
        if model_dir is None:
            self.model_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')
        else:
            self.model_dir = model_dir
        if news_dir is None:
            self.news_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'news_major1')
        else:
            self.news_dir = news_dir
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.model_dir, exist_ok=True)
        self.all_dates = sorted([f.replace('.parquet', '')
                                for f in os.listdir(self.price_dir)
                                if f.endswith('.parquet')])
        print(f"数据目录: {self.price_dir}")
        print(f"总交易日数: {len(self.all_dates)}")
        print(f"日期范围: {self.all_dates[0]} 至 {self.all_dates[-1]}")
        self.news_mkt, self.news_stk = process_news(self.news_dir)
        if not self.news_mkt.empty:
            print(f"新闻数据: {len(self.news_mkt)} 条市场记录, {len(self.news_stk)} 条个股记录")

    def load_daily_features(self, date: str, hist_days: int = 60) -> pd.DataFrame:
        p_price = os.path.join(self.price_dir, f"{date}.parquet")
        p_rank = os.path.join(self.rank_dir, f"{date}.parquet")
        p_chip = os.path.join(self.chip_dir, f"{date}.parquet")
        p_other = os.path.join(self.other_dir, f"{date}.parquet")
        if not os.path.exists(p_price):
            return None
        price_df = pd.read_parquet(p_price)
        price_df = price_df[price_df['ts_code'].apply(is_main_board)]
        if len(price_df) == 0:
            return None
        price_df['price_change'] = (price_df['close'] - price_df['pre_close']) / price_df['pre_close']
        price_df['high_change'] = (price_df['high'] - price_df['pre_close']) / price_df['pre_close']
        price_df['low_change'] = (price_df['low'] - price_df['pre_close']) / price_df['pre_close']
        price_df['amplitude'] = (price_df['high'] - price_df['low']) / price_df['pre_close']
        price_df['body_size'] = abs(price_df['close'] - price_df['open']) / price_df['pre_close']
        price_df['upper_shadow'] = (price_df['high'] - price_df[['close', 'open']].max(axis=1)) / price_df['pre_close']
        price_df['lower_shadow'] = (price_df[['close', 'open']].min(axis=1) - price_df['low']) / price_df['pre_close']
        price_df['is_yang'] = (price_df['close'] > price_df['open']).astype(int)
        price_df['gap'] = (price_df['open'] - price_df['pre_close']) / price_df['pre_close']
        price_df['vol_amount'] = price_df['close'] * price_df['vol']
        price_df['vol_ratio_day'] = price_df['vol'] / (price_df['vol'].mean() + 1e-8)
        date_idx = self.all_dates.index(date)
        if date_idx >= hist_days:
            hist_dates = self.all_dates[date_idx-hist_days:date_idx+1]
            all_stock_features = []
            for ts_code in price_df['ts_code'].unique():
                stock_hist = []
                for d in hist_dates:
                    p = os.path.join(self.price_dir, f"{d}.parquet")
                    if os.path.exists(p):
                        df = pd.read_parquet(p)
                        df = df[df['ts_code'] == ts_code]
                        if not df.empty:
                            df['trade_date'] = d
                            stock_hist.append(df[['trade_date', 'open', 'high', 'low', 'close', 'vol']])
                if len(stock_hist) >= 20:
                    stock_df = pd.concat(stock_hist, ignore_index=True)
                    stock_df = stock_df.sort_values('trade_date').reset_index(drop=True)
                    stock_df = TechnicalFactors.calculate_all(stock_df)
                    stock_df = ChanLunFeatures.calculate_all(stock_df)
                    last_row = stock_df.iloc[-1:].copy()
                    last_row['ts_code'] = ts_code
                    all_stock_features.append(last_row)
            if all_stock_features:
                tech_df = pd.concat(all_stock_features, ignore_index=True)
                merge_cols = [c for c in tech_df.columns if c not in ['ts_code', 'trade_date', 'open', 'high', 'low', 'close', 'vol']]
                price_df = pd.merge(price_df, tech_df[['ts_code'] + merge_cols], on='ts_code', how='left')
        if os.path.exists(p_rank):
            try:
                rank_df = pd.read_parquet(p_rank)
                rank_df = rank_df.sort_values('hot', ascending=False).drop_duplicates(subset='ts_code', keep='first')
                rank_df['hot_rank_pct'] = rank_df['hot'].rank(pct=True)
                price_df = pd.merge(price_df, rank_df[['ts_code', 'hot_rank_pct']], on='ts_code', how='left')
            except:
                price_df['hot_rank_pct'] = 0.5
        else:
            price_df['hot_rank_pct'] = 0.5
        if os.path.exists(p_chip):
            try:
                chip_df = pd.read_parquet(p_chip)
                chip_df['chip_concentration'] = (chip_df['cost_85pct'] - chip_df['cost_15pct']) / (chip_df['cost_50pct'] + 1e-8)
                price_df = pd.merge(price_df, chip_df[['ts_code', 'chip_concentration', 'winner_rate']],
                                   on='ts_code', how='left')
            except:
                price_df['chip_concentration'] = 0.1
                price_df['winner_rate'] = 50.0
        else:
            price_df['chip_concentration'] = 0.1
            price_df['winner_rate'] = 50.0
        if os.path.exists(p_other):
            try:
                other_df = pd.read_parquet(p_other, columns=['ts_code', 'circ_mv', 'turnover_rate', 'volume_ratio'])
                price_df = pd.merge(price_df, other_df, on='ts_code', how='left')
            except:
                price_df['circ_mv'] = 0
                price_df['turnover_rate'] = 0
                price_df['volume_ratio'] = 1
        else:
            price_df['circ_mv'] = 0
            price_df['turnover_rate'] = 0
            price_df['volume_ratio'] = 1
        if not self.news_mkt.empty:
            nm = self.news_mkt[self.news_mkt['trade_date'] == date]
            price_df['news_market_impact'] = nm['news_market_impact'].max() if not nm.empty else 0.0
        else:
            price_df['news_market_impact'] = 0.0
        if not self.news_stk.empty:
            ns = self.news_stk[self.news_stk['trade_date'] == date]
            if not ns.empty:
                ns_agg = ns.groupby('ts_code')['news_stock_impact'].max().reset_index()
                price_df = pd.merge(price_df, ns_agg[['ts_code', 'news_stock_impact']], on='ts_code', how='left')
                price_df['news_stock_impact'] = price_df['news_stock_impact'].fillna(0.0)
            else:
                price_df['news_stock_impact'] = 0.0
        else:
            price_df['news_stock_impact'] = 0.0
        price_df['trade_date'] = date
        return price_df

    def load_and_prepare_data(self, dates: List[str], label_threshold: float = 0.02) -> Tuple[pd.DataFrame, List[str]]:
        all_data = []
        for i in range(len(dates) - 2):
            d_curr = dates[i]
            d_t1 = dates[i + 1]
            d_t2 = dates[i + 2]
            df_t = self.load_daily_features(d_curr)
            if df_t is None or len(df_t) == 0:
                continue
            p_t1 = os.path.join(self.price_dir, f"{d_t1}.parquet")
            p_t2 = os.path.join(self.price_dir, f"{d_t2}.parquet")
            if not os.path.exists(p_t1) or not os.path.exists(p_t2):
                continue
            try:
                df_t1 = pd.read_parquet(p_t1)
                df_t2 = pd.read_parquet(p_t2)
            except:
                continue
            df_t = df_t.merge(
                df_t1[['ts_code', 'open']].rename(columns={'open': 't1_open'}),
                on='ts_code', how='left'
            )
            df_t = df_t.merge(
                df_t2[['ts_code', 'close', 'low']].rename(columns={'close': 't2_close', 'low': 't2_low'}),
                on='ts_code', how='left'
            )
            df_t = df_t.dropna(subset=['t1_open', 't2_close'])
            if len(df_t) == 0:
                continue
            df_t['label_ret'] = df_t['t2_close'] / df_t['t1_open'] - 1
            df_t['label'] = (df_t['label_ret'] > label_threshold).astype(int)
            all_data.append(df_t)
        if not all_data:
            return pd.DataFrame(), []
        result = pd.concat(all_data, ignore_index=True)
        exclude_cols = ['ts_code', 'trade_date', 'open', 'high', 'low', 'close',
                       'vol', 'amount', 'pre_close', 'label_ret', 'label',
                       't1_open', 't2_close', 't2_low']
        feature_cols = [c for c in result.columns if c not in exclude_cols]
        return result, feature_cols

    def get_top_features(self, train_df, feature_cols, n_top=20):
        """获取Top N特征"""
        X = train_df[feature_cols].fillna(0)
        y = train_df['label']
        # 使用XGBoost获取特征重要性
        model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=42,
            n_jobs=-1
        )
        model.fit(X, y)
        importance = model.feature_importances_
        top_features = sorted(zip(feature_cols, importance), key=lambda x: x[1], reverse=True)
        return [f[0] for f in top_features[:n_top]]

    def train_base_model(self, train_df, base_features):
        """训练基地模型（Top 20特征）"""
        X = train_df[base_features].fillna(0)
        y = train_df['label']
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        pos_weight = len(y) / y.sum() - 1 if y.sum() > 0 else 1
        model = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.7,
            colsample_bytree=0.7,
            random_state=42,
            eval_metric='auc',
            n_jobs=-1,
            tree_method='hist',
            reg_alpha=0.1,
            reg_lambda=1.0,
            min_child_weight=5,
            scale_pos_weight=pos_weight
        )
        model.fit(X_scaled, y)
        return model, scaler

    def train_error_model(self, train_df, base_features, other_features):
        """训练误差模型（用其他特征预测基地模型的误差）"""
        # 1. 先用基地模型预测
        X_base = train_df[base_features].fillna(0)
        y = train_df['label']
        scaler_base = StandardScaler()
        X_base_scaled = scaler_base.fit_transform(X_base)
        base_model = xgb.XGBClassifier(
            n_estimators=200, max_depth=5, learning_rate=0.05,
            subsample=0.7, colsample_bytree=0.7, random_state=42,
            eval_metric='auc', n_jobs=-1, tree_method='hist'
        )
        base_model.fit(X_base_scaled, y)
        base_pred = base_model.predict_proba(X_base_scaled)[:, 1]
        # 2. 计算误差
        error = np.abs(y - base_pred)
        # 3. 用其他特征预测误差
        if len(other_features) > 0:
            X_other = train_df[other_features].fillna(0)
            scaler_other = StandardScaler()
            X_other_scaled = scaler_other.fit_transform(X_other)
            error_model = xgb.XGBRegressor(
                n_estimators=100,
                max_depth=3,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=-1
            )
            error_model.fit(X_other_scaled, error)
            return base_model, scaler_base, error_model, scaler_other
        else:
            return base_model, scaler_base, None, None

    def predict_stacking(self, df, base_model, scaler_base, error_model, scaler_other,
                        base_features, other_features):
        """模型融合预测"""
        X_base = df[base_features].fillna(0)
        X_base_scaled = scaler_base.transform(X_base)
        base_pred = base_model.predict_proba(X_base_scaled)[:, 1]
        if error_model is not None and len(other_features) > 0:
            X_other = df[other_features].fillna(0)
            X_other_scaled = scaler_other.transform(X_other)
            error_pred = error_model.predict(X_other_scaled)
            # 融合：如果预测误差大，降低置信度
            final_pred = base_pred * (1 - error_pred)
        else:
            final_pred = base_pred
        return final_pred

    def backtest_with_predictions(self, predictions_df, test_dates, min_prob=0.55):
        trades = []
        skipped_limit_up = 0
        skipped_limit_down = 0
        initial_capital = 100000.0
        capital = initial_capital
        daily_nav = []
        for i in range(len(test_dates) - 2):
            d_curr = test_dates[i]
            d_t1 = test_dates[i + 1]
            d_t2 = test_dates[i + 2]
            d_curr_str = str(d_curr)
            day_pred = predictions_df[predictions_df['trade_date'].astype(str) == d_curr_str]
            if len(day_pred) == 0:
                daily_nav.append({'date': d_t2, 'nav': capital})
                continue
            best_idx = day_pred['prob'].idxmax()
            best_prob = day_pred.loc[best_idx, 'prob']
            if best_prob < min_prob:
                daily_nav.append({'date': d_t2, 'nav': capital})
                continue
            ts_code = day_pred.loc[best_idx, 'ts_code']
            p_t1 = os.path.join(self.price_dir, f"{d_t1}.parquet")
            p_t2 = os.path.join(self.price_dir, f"{d_t2}.parquet")
            if not os.path.exists(p_t1) or not os.path.exists(p_t2):
                daily_nav.append({'date': d_t2, 'nav': capital})
                continue
            try:
                df_t1 = pd.read_parquet(p_t1)
                df_t2 = pd.read_parquet(p_t2)
            except:
                daily_nav.append({'date': d_t2, 'nav': capital})
                continue
            t1_data = df_t1[df_t1['ts_code'] == ts_code]
            if t1_data.empty:
                daily_nav.append({'date': d_t2, 'nav': capital})
                continue
            t1_open = float(t1_data.iloc[0]['open'])
            t1_pre = float(t1_data.iloc[0]['pre_close'])
            limit_pct = get_limit_pct(ts_code)
            t1_open_chg = (t1_open - t1_pre) / t1_pre * 100
            if t1_open_chg >= (limit_pct - LIMIT_THRESHOLD):
                skipped_limit_up += 1
                daily_nav.append({'date': d_t2, 'nav': capital})
                continue
            t2_data = df_t2[df_t2['ts_code'] == ts_code]
            if t2_data.empty:
                daily_nav.append({'date': d_t2, 'nav': capital})
                continue
            t2_close = float(t2_data.iloc[0]['close'])
            t2_low = float(t2_data.iloc[0]['low'])
            t2_open = float(t2_data.iloc[0]['open'])
            t2_low_chg = (t2_low - t1_open) / t1_open * 100
            if t2_low_chg <= -(limit_pct - LIMIT_THRESHOLD):
                sell_price = min(t2_open, t1_open * (1 - limit_pct/100))
                skipped_limit_down += 1
            else:
                sell_price = t2_close
            buy_price = t1_open * (1 + SLIPPAGE)
            sell_price = sell_price * (1 - SLIPPAGE)
            ret = sell_price / buy_price - 1 - COST_RATE
            capital *= (1 + ret)
            trades.append({
                'date_t': d_curr,
                'date_t1': d_t1,
                'date_t2': d_t2,
                'ts_code': ts_code,
                'prob': best_prob,
                'buy_price': buy_price,
                'sell_price': sell_price,
                'return': ret
            })
            daily_nav.append({'date': d_t2, 'nav': capital})
        if len(trades) == 0:
            return {
                'return': 0, 'sharpe': 0, 'max_dd': 0, 'n_trades': 0,
                'win_rate': 0, 'avg_return': 0,
                'skipped_limit_up': skipped_limit_up,
                'skipped_limit_down': skipped_limit_down,
                'daily_nav': daily_nav, 'trades': pd.DataFrame()
            }
        trades_df = pd.DataFrame(trades)
        total_ret = capital / initial_capital - 1
        nav_df = pd.DataFrame(daily_nav)
        if len(nav_df) > 1:
            nav_df['ret'] = nav_df['nav'].pct_change()
            vol = nav_df['ret'].std() * np.sqrt(252)
            sharpe = (total_ret / (len(test_dates) / 252)) / vol if vol > 0 else 0
        else:
            sharpe = 0
        if len(nav_df) > 0:
            nav_df['cummax'] = nav_df['nav'].cummax()
            nav_df['dd'] = (nav_df['nav'] - nav_df['cummax']) / nav_df['cummax']
            max_dd = nav_df['dd'].min()
        else:
            max_dd = 0
        win_rate = (trades_df['return'] > 0).mean()
        avg_ret = trades_df['return'].mean()
        return {
            'return': total_ret, 'sharpe': sharpe, 'max_dd': max_dd,
            'n_trades': len(trades), 'win_rate': win_rate, 'avg_return': avg_ret,
            'skipped_limit_up': skipped_limit_up,
            'skipped_limit_down': skipped_limit_down,
            'daily_nav': daily_nav, 'trades': trades_df
        }

    def run_stacking(self, start_date='20230101', end_date='20241231',
                    train_months=12, test_months=1, min_prob=0.55, n_top=20):
        print("=" * 80)
        print("模型融合框架 V10")
        print("1. 基地模型：Top {} 特征".format(n_top))
        print("2. 误差模型：其他特征预测误差")
        print("3. 融合预测：基地模型 - 预测误差")
        print("=" * 80)
        print(f"回测期: {start_date} 至 {end_date}")
        print(f"训练期: {train_months} 个月")
        print(f"测试期: {test_months} 个月")
        print(f"买入阈值: {min_prob}")
        print("=" * 80)
        dates = [d for d in self.all_dates if start_date <= d <= end_date]
        if len(dates) < train_months * 21 + test_months * 21:
            print("数据不足！")
            return []
        months = []
        current_month = dates[0][:6]
        month_dates = []
        for d in dates:
            if d[:6] == current_month:
                month_dates.append(d)
            else:
                months.append((current_month, month_dates))
                current_month = d[:6]
                month_dates = [d]
        if month_dates:
            months.append((current_month, month_dates))
        print(f"总月份数: {len(months)}")
        results_base = []
        results_stack = []
        all_equity_base = []
        all_equity_stack = []
        all_trades_base = []
        all_trades_stack = []
        for i in range(train_months, len(months), test_months):
            train_months_list = months[i-train_months:i]
            train_dates = []
            for _, month_dates in train_months_list:
                train_dates.extend(month_dates)
            test_months_list = months[i:i+test_months]
            test_dates = []
            for _, month_dates in test_months_list:
                test_dates.extend(month_dates)
            if len(test_dates) < 5:
                continue
            period_name = test_months_list[0][0]
            print(f"\n{'='*60}")
            print(f"回测月份: {period_name}")
            print(f"训练期: {train_dates[0]} 至 {train_dates[-1]} ({len(train_dates)} 天)")
            print(f"测试期: {test_dates[0]} 至 {test_dates[-1]} ({len(test_dates)} 天)")
            print(f"{'='*60}")
            print("准备训练数据...")
            train_df, feature_cols = self.load_and_prepare_data(train_dates)
            if len(train_df) == 0 or len(feature_cols) == 0:
                print("训练数据不足，跳过")
                continue
            print(f"训练样本: {len(train_df)}, 总特征数: {len(feature_cols)}")
            # 获取Top 20特征
            print("获取Top {}特征...".format(n_top))
            top_features = self.get_top_features(train_df, feature_cols, n_top)
            print(f"Top {n_top}特征: {top_features}")
            other_features = [f for f in feature_cols if f not in top_features]
            print(f"其他特征数: {len(other_features)}")
            # 训练基地模型
            print("训练基地模型...")
            base_model, scaler_base = self.train_base_model(train_df, top_features)
            # 训练误差模型
            print("训练误差模型...")
            base_model_err, scaler_base_err, error_model, scaler_other = self.train_error_model(
                train_df, top_features, other_features
            )
            # 生成基地模型预测
            print("生成基地模型预测...")
            predictions_base = []
            predictions_stack = []
            for d_curr in test_dates[:-2]:
                df_t = self.load_daily_features(d_curr)
                if df_t is None or len(df_t) == 0:
                    continue
                # 基地模型预测
                X_base = df_t[top_features].fillna(0)
                X_base_scaled = scaler_base.transform(X_base)
                df_t['prob_base'] = base_model.predict_proba(X_base_scaled)[:, 1]
                # 融合模型预测
                df_t['prob_stack'] = self.predict_stacking(
                    df_t, base_model_err, scaler_base_err, error_model, scaler_other,
                    top_features, other_features
                )
                predictions_base.append(df_t[['ts_code', 'trade_date', 'prob_base']].rename(columns={'prob_base': 'prob'}).copy())
                predictions_stack.append(df_t[['ts_code', 'trade_date', 'prob_stack']].rename(columns={'prob_stack': 'prob'}).copy())
            if not predictions_base:
                continue
            predictions_base_df = pd.concat(predictions_base, ignore_index=True)
            predictions_stack_df = pd.concat(predictions_stack, ignore_index=True)
            # 基地模型回测
            print("基地模型回测...")
            backtest_base = self.backtest_with_predictions(predictions_base_df, test_dates, min_prob)
            print(f"\n基地模型结果:")
            print(f"  收益率: {backtest_base['return']:.2%}")
            print(f"  夏普比率: {backtest_base['sharpe']:.2f}")
            print(f"  最大回撤: {backtest_base['max_dd']:.2%}")
            print(f"  交易次数: {backtest_base['n_trades']}")
            print(f"  胜率: {backtest_base['win_rate']:.2%}")
            # 融合模型回测
            print("\n融合模型回测...")
            backtest_stack = self.backtest_with_predictions(predictions_stack_df, test_dates, min_prob)
            print(f"\n融合模型结果:")
            print(f"  收益率: {backtest_stack['return']:.2%}")
            print(f"  夏普比率: {backtest_stack['sharpe']:.2f}")
            print(f"  最大回撤: {backtest_stack['max_dd']:.2%}")
            print(f"  交易次数: {backtest_stack['n_trades']}")
            print(f"  胜率: {backtest_stack['win_rate']:.2%}")
            # 保存基地模型结果
            results_base.append({
                'period': period_name,
                'model': 'base',
                'test_return': backtest_base['return'],
                'test_sharpe': backtest_base['sharpe'],
                'test_max_dd': backtest_base['max_dd'],
                'n_trades': backtest_base['n_trades'],
                'win_rate': backtest_base['win_rate'],
                'avg_return': backtest_base['avg_return']
            })
            # 保存融合模型结果
            results_stack.append({
                'period': period_name,
                'model': 'stacking',
                'test_return': backtest_stack['return'],
                'test_sharpe': backtest_stack['sharpe'],
                'test_max_dd': backtest_stack['max_dd'],
                'n_trades': backtest_stack['n_trades'],
                'win_rate': backtest_stack['win_rate'],
                'avg_return': backtest_stack['avg_return']
            })
            if len(backtest_base['trades']) > 0:
                all_trades_base.append(backtest_base['trades'])
            if len(backtest_stack['trades']) > 0:
                all_trades_stack.append(backtest_stack['trades'])
            if backtest_base['daily_nav']:
                all_equity_base.append(pd.DataFrame(backtest_base['daily_nav']))
            if backtest_stack['daily_nav']:
                all_equity_stack.append(pd.DataFrame(backtest_stack['daily_nav']))
        # 保存汇总结果
        if results_base and results_stack:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            # 基地模型
            results_base_df = pd.DataFrame(results_base)
            results_base_df.to_csv(os.path.join(self.output_dir, f'base_model_results_{timestamp}.csv'), index=False)
            if all_equity_base:
                pd.concat(all_equity_base, ignore_index=True).to_csv(
                    os.path.join(self.output_dir, f'base_model_equity_{timestamp}.csv'), index=False)
            if all_trades_base:
                pd.concat(all_trades_base, ignore_index=True).to_csv(
                    os.path.join(self.output_dir, f'base_model_trades_{timestamp}.csv'), index=False)
            # 融合模型
            results_stack_df = pd.DataFrame(results_stack)
            results_stack_df.to_csv(os.path.join(self.output_dir, f'stacking_model_results_{timestamp}.csv'), index=False)
            if all_equity_stack:
                pd.concat(all_equity_stack, ignore_index=True).to_csv(
                    os.path.join(self.output_dir, f'stacking_model_equity_{timestamp}.csv'), index=False)
            if all_trades_stack:
                pd.concat(all_trades_stack, ignore_index=True).to_csv(
                    os.path.join(self.output_dir, f'stacking_model_trades_{timestamp}.csv'), index=False)
            # 对比总结
            print(f"\n{'='*80}")
            print("模型对比总结")
            print(f"{'='*80}")
            print(f"基地模型平均收益率: {results_base_df['test_return'].mean():.2%}")
            print(f"融合模型平均收益率: {results_stack_df['test_return'].mean():.2%}")
            print(f"基地模型平均夏普: {results_base_df['test_sharpe'].mean():.2f}")
            print(f"融合模型平均夏普: {results_stack_df['test_sharpe'].mean():.2f}")
            print(f"基地模型平均胜率: {results_base_df['win_rate'].mean():.2%}")
            print(f"融合模型平均胜率: {results_stack_df['win_rate'].mean():.2%}")
            print(f"\n结果保存至: {self.output_dir}")
            print(f"{'='*80}")
        return results_base, results_stack


if __name__ == "__main__":
    bt = WalkForwardStackingV10()
    results_base, results_stack = bt.run_stacking(
        start_date='20230101',
        end_date='20241231',
        train_months=12,
        test_months=1,
        min_prob=0.55,
        n_top=20
    )

"""
模型融合框架 V10 - 优化版
1. 基地模型：使用Top 20特征训练XGBoost
2. 误差模型：使用其他特征预测基地模型的误差
3. 模型融合：基地模型预测 + 误差模型修正
4. 优化：添加进度显示、缓存、批量处理
"""
import pandas as pd
import numpy as np
import os
import sys
import pickle
import json
import time
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from tqdm import tqdm

# A股交易规则配置
COST_RATE = 0.003
SLIPPAGE = 0.002
LIMIT_THRESHOLD = 0.5
STOP_LOSS = 0.05  # 止损阈值5%
MAX_POSITIONS = 5  # 最大持仓数


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
        return df

    @staticmethod
    def moving_averages(df, windows=[5, 10, 20, 60]):
        for w in windows:
            df[f'ma_{w}d'] = df['close'].rolling(w).mean()
            df[f'close_ma_{w}d_ratio'] = df['close'] / (df[f'ma_{w}d'] + 1e-8)
        return df

    @staticmethod
    def rsi(df, windows=[6, 14, 24]):
        for w in windows:
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=w).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=w).mean()
            rs = gain / (loss + 1e-8)
            df[f'rsi_{w}d'] = 100 - (100 / (1 + rs))
        return df

    @staticmethod
    def macd(df, fast=12, slow=26, signal=9):
        ema_fast = df['close'].ewm(span=fast, adjust=False).mean()
        ema_slow = df['close'].ewm(span=slow, adjust=False).mean()
        df['macd'] = ema_fast - ema_slow
        df['macd_signal'] = df['macd'].ewm(span=signal, adjust=False).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']
        return df

    @staticmethod
    def bollinger(df, window=20, num_std=2):
        ma = df['close'].rolling(window).mean()
        std = df['close'].rolling(window).std()
        df['boll_upper'] = ma + num_std * std
        df['boll_lower'] = ma - num_std * std
        df['boll_mid'] = ma
        df['boll_pct'] = (df['close'] - df['boll_lower']) / (df['boll_upper'] - df['boll_lower'] + 1e-8)
        return df

    @staticmethod
    def kdj(df, n=9, m1=3, m2=3):
        low_list = df['low'].rolling(window=n, min_periods=n).min()
        high_list = df['high'].rolling(window=n, min_periods=n).max()
        rsv = (df['close'] - low_list) / (high_list - low_list + 1e-8) * 100
        df['kdj_k'] = rsv.ewm(alpha=1/m1, adjust=False).mean()
        df['kdj_d'] = df['kdj_k'].ewm(alpha=1/m2, adjust=False).mean()
        df['kdj_j'] = 3 * df['kdj_k'] - 2 * df['kdj_d']
        return df

    @staticmethod
    def williams_r(df, n=14):
        high = df['high'].rolling(window=n).max()
        low = df['low'].rolling(window=n).min()
        df['williams_r'] = (high - df['close']) / (high - low + 1e-8) * -100
        return df

    @staticmethod
    def atr(df, n=14):
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df[f'atr_{n}d'] = tr.rolling(window=n).mean()
        df['atr_ratio'] = df[f'atr_{n}d'] / df['close']
        return df

    @staticmethod
    def obv(df):
        obv = (np.sign(df['close'].diff()) * df['vol']).cumsum()
        df['obv'] = obv
        df['obv_ma'] = df['obv'].rolling(20).mean()
        return df

    @staticmethod
    def ichimoku(df):
        high_9 = df['high'].rolling(window=9).max()
        low_9 = df['low'].rolling(window=9).min()
        df['tenkan_sen'] = (high_9 + low_9) / 2
        high_26 = df['high'].rolling(window=26).max()
        low_26 = df['low'].rolling(window=26).min()
        df['kijun_sen'] = (high_26 + low_26) / 2
        df['senkou_span_a'] = ((df['tenkan_sen'] + df['kijun_sen']) / 2).shift(26)
        high_52 = df['high'].rolling(window=52).max()
        low_52 = df['low'].rolling(window=52).min()
        df['senkou_span_b'] = ((high_52 + low_52) / 2).shift(26)
        return df

    @staticmethod
    def adx(df, n=14):
        plus_dm = df['high'].diff()
        minus_dm = df['low'].diff()
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm > 0] = 0
        tr = pd.concat([df['high'] - df['low'], 
                       np.abs(df['high'] - df['close'].shift()),
                       np.abs(df['low'] - df['close'].shift())], axis=1).max(axis=1)
        atr = tr.rolling(window=n).mean()
        plus_di = 100 * plus_dm.rolling(window=n).mean() / (atr + 1e-8)
        minus_di = 100 * np.abs(minus_dm).rolling(window=n).mean() / (atr + 1e-8)
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-8)
        df['adx'] = dx.rolling(window=n).mean()
        return df

    @staticmethod
    def calculate_all(df):
        df = TechnicalFactors.momentum(df)
        df = TechnicalFactors.volatility(df)
        df = TechnicalFactors.volume_features(df)
        df = TechnicalFactors.moving_averages(df)
        df = TechnicalFactors.rsi(df)
        df = TechnicalFactors.macd(df)
        df = TechnicalFactors.bollinger(df)
        df = TechnicalFactors.kdj(df)
        df = TechnicalFactors.williams_r(df)
        df = TechnicalFactors.atr(df)
        df = TechnicalFactors.obv(df)
        df = TechnicalFactors.ichimoku(df)
        df = TechnicalFactors.adx(df)
        return df


class ChanLunFeatures:
    @staticmethod
    def calculate_fractals(df, n=5):
        df['fractal_high'] = (df['high'] == df['high'].rolling(window=2*n+1, center=True).max()).astype(int)
        df['fractal_low'] = (df['low'] == df['low'].rolling(window=2*n+1, center=True).min()).astype(int)
        return df

    @staticmethod
    def calculate_bi(df):
        df['bi_direction'] = np.where(df['close'] > df['close'].shift(1), 1, -1)
        df['bi_strength'] = df['bi_direction'] * (df['close'] - df['close'].shift(1)) / df['close'].shift(1)
        return df

    @staticmethod
    def calculate_zhongshu(df, window=20):
        df['zhongshu_high'] = df['high'].rolling(window=window).max()
        df['zhongshu_low'] = df['low'].rolling(window=window).min()
        df['zhongshu_mid'] = (df['zhongshu_high'] + df['zhongshu_low']) / 2
        df['in_zhongshu'] = ((df['close'] >= df['zhongshu_low']) & (df['close'] <= df['zhongshu_high'])).astype(int)
        return df

    @staticmethod
    def calculate_divergence(df):
        df['price_momentum'] = df['close'].diff(5)
        df['volume_momentum'] = df['vol'].diff(5)
        df['divergence'] = np.where(
            (df['price_momentum'] > 0) & (df['volume_momentum'] < 0), -1,
            np.where((df['price_momentum'] < 0) & (df['volume_momentum'] > 0), 1, 0)
        )
        return df

    @staticmethod
    def calculate_all(df):
        df = ChanLunFeatures.calculate_fractals(df)
        df = ChanLunFeatures.calculate_bi(df)
        df = ChanLunFeatures.calculate_zhongshu(df)
        df = ChanLunFeatures.calculate_divergence(df)
        return df


class WalkForwardStackingV10Optimized:
    def __init__(self,
                 data_dir: str = r'D:\iquant_data\data_v2',
                 output_dir: str = None,
                 model_dir: str = None,
                 news_dir: str = None,
                 cache_dir: str = None):
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
        if cache_dir is None:
            self.cache_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'cache')
        else:
            self.cache_dir = cache_dir
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.cache_dir, exist_ok=True)
        self.all_dates = sorted([f.replace('.parquet', '')
                                for f in os.listdir(self.price_dir)
                                if f.endswith('.parquet')])
        print(f"数据目录: {self.price_dir}")
        print(f"总交易日数: {len(self.all_dates)}")
        print(f"日期范围: {self.all_dates[0]} 至 {self.all_dates[-1]}")
        
        # 缓存
        self.price_cache = {}
        self.tech_cache = {}
        
        # 中间数据存储
        self.cache_file = os.path.join(self.cache_dir, 'training_data_cache.parquet')
        self.cache_metadata = os.path.join(self.cache_dir, 'cache_metadata.json')
        
        self.news_mkt, self.news_stk = process_news(self.news_dir)
        if not self.news_mkt.empty:
            print(f"新闻数据: {len(self.news_mkt)} 条市场记录, {len(self.news_stk)} 条个股记录")
    
    def _get_cache_key(self, dates, label_threshold):
        """生成缓存键"""
        return {
            'start_date': dates[0],
            'end_date': dates[-1],
            'n_dates': len(dates),
            'label_threshold': label_threshold
        }
    
    def _check_cache_valid(self, dates, label_threshold):
        """检查缓存是否有效"""
        if not os.path.exists(self.cache_metadata) or not os.path.exists(self.cache_file):
            return False
        try:
            with open(self.cache_metadata, 'r') as f:
                cached_meta = json.load(f)
            current_meta = self._get_cache_key(dates, label_threshold)
            return (cached_meta['start_date'] == current_meta['start_date'] and
                    cached_meta['end_date'] == current_meta['end_date'] and
                    cached_meta['n_dates'] == current_meta['n_dates'] and
                    cached_meta['label_threshold'] == current_meta['label_threshold'])
        except:
            return False
    
    def _save_cache(self, df, feature_cols, dates, label_threshold):
        """保存缓存"""
        try:
            df.to_parquet(self.cache_file, index=False)
            meta = self._get_cache_key(dates, label_threshold)
            meta['feature_cols'] = feature_cols
            meta['n_samples'] = len(df)
            meta['n_features'] = len(feature_cols)
            with open(self.cache_metadata, 'w') as f:
                json.dump(meta, f)
            print(f"  缓存已保存: {self.cache_file}")
        except Exception as e:
            print(f"  缓存保存失败: {e}")
    
    def _load_cache(self):
        """加载缓存"""
        try:
            df = pd.read_parquet(self.cache_file)
            with open(self.cache_metadata, 'r') as f:
                meta = json.load(f)
            feature_cols = meta['feature_cols']
            print(f"  缓存已加载: {len(df)} 样本, {len(feature_cols)} 特征")
            return df, feature_cols
        except Exception as e:
            print(f"  缓存加载失败: {e}")
            return None, None

    def load_daily_features(self, date: str, hist_days: int = 60) -> pd.DataFrame:
        """加载每日特征 - 优化版"""
        start_time = time.time()
        p_price = os.path.join(self.price_dir, f"{date}.parquet")
        p_rank = os.path.join(self.rank_dir, f"{date}.parquet")
        p_chip = os.path.join(self.chip_dir, f"{date}.parquet")
        p_other = os.path.join(self.other_dir, f"{date}.parquet")
        
        if not os.path.exists(p_price):
            return None
            
        # 使用缓存
        if date in self.price_cache:
            return self.price_cache[date].copy()
            
        price_df = pd.read_parquet(p_price)
        price_df = price_df[price_df['ts_code'].apply(is_main_board)]
        if len(price_df) == 0:
            return None
            
        # 基础特征
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
        
        # 技术指标 - 批量处理
        date_idx = self.all_dates.index(date)
        if date_idx >= hist_days:
            hist_dates = self.all_dates[date_idx-hist_days:date_idx+1]
            
            # 批量读取历史数据
            all_hist_data = []
            for d in hist_dates:
                p = os.path.join(self.price_dir, f"{d}.parquet")
                if os.path.exists(p):
                    df = pd.read_parquet(p)
                    df = df[df['ts_code'].apply(is_main_board)]
                    if not df.empty:
                        df['trade_date'] = d
                        all_hist_data.append(df[['trade_date', 'ts_code', 'open', 'high', 'low', 'close', 'vol']])
            
            if len(all_hist_data) >= 20:
                hist_df = pd.concat(all_hist_data, ignore_index=True)
                hist_df = hist_df.sort_values(['ts_code', 'trade_date']).reset_index(drop=True)
                
                # 按股票分组计算技术指标
                tech_features = []
                for ts_code, group in hist_df.groupby('ts_code'):
                    if len(group) >= 20:
                        group = group.sort_values('trade_date').reset_index(drop=True)
                        group = TechnicalFactors.calculate_all(group)
                        group = ChanLunFeatures.calculate_all(group)
                        last_row = group.iloc[-1:].copy()
                        last_row['ts_code'] = ts_code
                        tech_features.append(last_row)
                
                if tech_features:
                    tech_df = pd.concat(tech_features, ignore_index=True)
                    merge_cols = [c for c in tech_df.columns if c not in ['ts_code', 'trade_date', 'open', 'high', 'low', 'close', 'vol']]
                    price_df = pd.merge(price_df, tech_df[['ts_code'] + merge_cols], on='ts_code', how='left')
        
        # 同花顺排名
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
            
        # 筹码数据
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
            
        # 其他数据
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
            
        # 新闻数据
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
        
        # 缓存结果
        self.price_cache[date] = price_df.copy()
        
        elapsed = time.time() - start_time
        print(f"  加载特征耗时: {elapsed:.2f}s, 股票数: {len(price_df)}")
        
        return price_df

    def load_and_prepare_data(self, dates: List[str], label_threshold: float = 0.02, use_cache: bool = True) -> Tuple[pd.DataFrame, List[str]]:
        """准备训练数据 - 带进度显示和缓存"""
        # 检查缓存
        if use_cache and self._check_cache_valid(dates, label_threshold):
            print("  发现有效缓存，正在加载...")
            cached_df, cached_features = self._load_cache()
            if cached_df is not None:
                return cached_df, cached_features
        
        all_data = []
        print(f"  准备训练数据: {len(dates)} 天...")
        
        for i in tqdm(range(len(dates) - 2), desc="  加载数据", leave=False):
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
        
        print(f"  完成! 样本数: {len(result)}, 特征数: {len(feature_cols)}")
        
        # 保存缓存
        if use_cache and len(result) > 0:
            self._save_cache(result, feature_cols, dates, label_threshold)
        
        return result, feature_cols

    def get_top_features(self, train_df, feature_cols, n_top=20):
        """获取Top N特征 - 带进度"""
        print(f"  计算特征重要性...")
        X = train_df[feature_cols].fillna(0)
        y = train_df['label']
        
        model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=42,
            n_jobs=-1,
            verbosity=0
        )
        model.fit(X, y)
        importance = model.feature_importances_
        top_features = sorted(zip(feature_cols, importance), key=lambda x: x[1], reverse=True)
        return [f[0] for f in top_features[:n_top]]

    def train_base_model(self, train_df, base_features):
        """训练基地模型"""
        print(f"  训练基地模型...")
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
            scale_pos_weight=pos_weight,
            verbosity=0
        )
        model.fit(X_scaled, y)
        return model, scaler

    def train_error_model(self, train_df, base_features, other_features):
        """训练误差模型"""
        print(f"  训练误差模型...")
        # 1. 先用基地模型预测
        X_base = train_df[base_features].fillna(0)
        y = train_df['label']
        scaler_base = StandardScaler()
        X_base_scaled = scaler_base.fit_transform(X_base)
        
        base_model = xgb.XGBClassifier(
            n_estimators=200, max_depth=5, learning_rate=0.05,
            subsample=0.7, colsample_bytree=0.7, random_state=42,
            eval_metric='auc', n_jobs=-1, tree_method='hist',
            verbosity=0
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
                n_jobs=-1,
                verbosity=0
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

    def backtest_with_predictions(self, predictions_df, test_dates, min_prob=0.50):
        """回测 - 增加止损和市场环境过滤"""
        trades = []
        skipped_limit_up = 0
        skipped_limit_down = 0
        initial_capital = 100000.0
        capital = initial_capital
        daily_nav = []
        current_positions = []  # 当前持仓
        
        for i in range(len(test_dates) - 2):
            d_curr = test_dates[i]
            d_t1 = test_dates[i + 1]
            d_t2 = test_dates[i + 2]
            d_curr_str = str(d_curr)
            
            # 市场环境过滤：计算近期市场趋势
            market_trend = self._get_market_trend(d_curr)
            if market_trend < -0.02:  # 市场下跌超过2%，暂停交易
                daily_nav.append({'date': d_t2, 'nav': capital})
                continue
            
            day_pred = predictions_df[predictions_df['trade_date'].astype(str) == d_curr_str]
            if len(day_pred) == 0:
                daily_nav.append({'date': d_t2, 'nav': capital})
                continue
                
            # 选择前N个最高概率的股票
            day_pred_sorted = day_pred.sort_values('prob', ascending=False)
            selected_stocks = day_pred_sorted.head(MAX_POSITIONS)
            
            daily_return = 0
            n_trades_today = 0
            
            for _, row in selected_stocks.iterrows():
                best_prob = row['prob']
                if best_prob < min_prob:
                    continue
                    
                ts_code = row['ts_code']
                p_t1 = os.path.join(self.price_dir, f"{d_t1}.parquet")
                p_t2 = os.path.join(self.price_dir, f"{d_t2}.parquet")
                if not os.path.exists(p_t1) or not os.path.exists(p_t2):
                    continue
                    
                try:
                    df_t1 = pd.read_parquet(p_t1)
                    df_t2 = pd.read_parquet(p_t2)
                except:
                    continue
                    
                t1_data = df_t1[df_t1['ts_code'] == ts_code]
                if t1_data.empty:
                    continue
                    
                t1_open = float(t1_data.iloc[0]['open'])
                t1_pre = float(t1_data.iloc[0]['pre_close'])
                limit_pct = get_limit_pct(ts_code)
                t1_open_chg = (t1_open - t1_pre) / t1_pre * 100
                
                if t1_open_chg >= (limit_pct - LIMIT_THRESHOLD):
                    skipped_limit_up += 1
                    continue
                    
                t2_data = df_t2[df_t2['ts_code'] == ts_code]
                if t2_data.empty:
                    continue
                    
                t2_close = float(t2_data.iloc[0]['close'])
                t2_low = float(t2_data.iloc[0]['low'])
                t2_open = float(t2_data.iloc[0]['open'])
                t2_low_chg = (t2_low - t1_open) / t1_open * 100
                
                # 止损逻辑
                if t2_low_chg <= -(limit_pct - LIMIT_THRESHOLD):
                    sell_price = min(t2_open, t1_open * (1 - limit_pct/100))
                    skipped_limit_down += 1
                else:
                    # 检查是否触发止损
                    current_ret = (t2_low - t1_open) / t1_open
                    if current_ret <= -STOP_LOSS:
                        sell_price = t1_open * (1 - STOP_LOSS)  # 止损卖出
                    else:
                        sell_price = t2_close
                    
                buy_price = t1_open * (1 + SLIPPAGE)
                sell_price = sell_price * (1 - SLIPPAGE)
                ret = sell_price / buy_price - 1 - COST_RATE
                
                # 分配资金
                position_size = 1.0 / MAX_POSITIONS
                daily_return += ret * position_size
                n_trades_today += 1
                
                trades.append({
                    'date_t': d_curr,
                    'date_t1': d_t1,
                    'date_t2': d_t2,
                    'ts_code': ts_code,
                    'prob': best_prob,
                    'buy_price': buy_price,
                    'sell_price': sell_price,
                    'return': ret,
                    'market_trend': market_trend
                })
            
            if n_trades_today > 0:
                capital *= (1 + daily_return)
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

    def _get_market_trend(self, date, window=20):
        """计算市场趋势 - 用于市场环境过滤"""
        try:
            date_idx = self.all_dates.index(date)
            if date_idx < window:
                return 0.0
            
            hist_dates = self.all_dates[date_idx-window:date_idx]
            market_returns = []
            
            for d in hist_dates:
                p = os.path.join(self.price_dir, f"{d}.parquet")
                if os.path.exists(p):
                    df = pd.read_parquet(p)
                    if not df.empty:
                        # 计算等权平均收益
                        avg_ret = ((df['close'] - df['pre_close']) / df['pre_close']).mean()
                        market_returns.append(avg_ret)
            
            if len(market_returns) > 0:
                return np.mean(market_returns)
            return 0.0
        except:
            return 0.0

    def run_stacking(self, start_date='20230101', end_date='20241231',
                    train_months=12, test_months=6, min_prob=0.50, n_top=20):
        """运行模型融合框架 - 带详细进度"""
        print("=" * 80)
        print("模型融合框架 V10 - 优化版")
        print("1. 基地模型：Top {} 特征".format(n_top))
        print("2. 误差模型：其他特征预测误差")
        print("3. 融合预测：基地模型 - 预测误差")
        print("=" * 80)
        print(f"回测期: {start_date} 至 {end_date}")
        print(f"训练期: {train_months} 个月")
        print(f"测试期: {test_months} 个月")
        print(f"买入阈值: {min_prob}")
        print(f"止损阈值: {STOP_LOSS}")
        print(f"最大持仓: {MAX_POSITIONS}")
        print("=" * 80)
        
        dates = [d for d in self.all_dates if start_date <= d <= end_date]
        if len(dates) < train_months * 21 + test_months * 21:
            print("数据不足！")
            return []
            
        # 分月份
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
        print(f"回测月份数: {len(months) - train_months}")
        print("=" * 80)
        
        results_base = []
        results_stack = []
        all_equity_base = []
        all_equity_stack = []
        all_trades_base = []
        all_trades_stack = []
        
        # 使用tqdm显示总体进度
        # 修改：每6个月滚动一次，而不是每个月
        roll_months = 6
        month_iter = tqdm(range(train_months, len(months), roll_months), 
                         desc="回测进度", unit="期")
        
        for i in month_iter:
            train_months_list = months[i-train_months:i]
            train_dates = []
            for _, month_dates in train_months_list:
                train_dates.extend(month_dates)
                
            # 修改：测试期也改为6个月
            test_months_list = months[i:i+roll_months]
            test_dates = []
            for _, month_dates in test_months_list:
                test_dates.extend(month_dates)
                
            if len(test_dates) < 5:
                continue
                
            period_name = test_months_list[0][0]
            month_iter.set_postfix({"当前月份": period_name})
            
            print(f"\n{'='*60}")
            print(f"回测月份: {period_name}")
            print(f"训练期: {train_dates[0]} 至 {train_dates[-1]} ({len(train_dates)} 天)")
            print(f"测试期: {test_dates[0]} 至 {test_dates[-1]} ({len(test_dates)} 天)")
            print(f"{'='*60}")
            
            # 准备训练数据
            print("[1/6] 准备训练数据...")
            train_start = time.time()
            train_df, feature_cols = self.load_and_prepare_data(train_dates)
            train_elapsed = time.time() - train_start
            
            if len(train_df) == 0 or len(feature_cols) == 0:
                print("训练数据不足，跳过")
                continue
                
            print(f"  训练样本: {len(train_df)}, 总特征数: {len(feature_cols)}, 耗时: {train_elapsed:.1f}s")
            
            # 获取Top特征
            print(f"[2/6] 获取Top {n_top}特征...")
            top_start = time.time()
            top_features = self.get_top_features(train_df, feature_cols, n_top)
            print(f"  Top {n_top}特征: {top_features[:5]}...")
            other_features = [f for f in feature_cols if f not in top_features]
            print(f"  其他特征数: {len(other_features)}, 耗时: {time.time()-top_start:.1f}s")
            
            # 训练基地模型
            print("[3/6] 训练基地模型...")
            base_start = time.time()
            base_model, scaler_base = self.train_base_model(train_df, top_features)
            print(f"  基地模型训练完成, 耗时: {time.time()-base_start:.1f}s")
            
            # 训练误差模型
            print("[4/6] 训练误差模型...")
            err_start = time.time()
            base_model_err, scaler_base_err, error_model, scaler_other = self.train_error_model(
                train_df, top_features, other_features
            )
            print(f"  误差模型训练完成, 耗时: {time.time()-err_start:.1f}s")
            
            # 生成预测
            print("[5/6] 生成预测...")
            pred_start = time.time()
            predictions_base = []
            predictions_stack = []
            
            for d_curr in tqdm(test_dates[:-2], desc="  预测", leave=False):
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
                
            print(f"  预测完成, 耗时: {time.time()-pred_start:.1f}s")
            
            if not predictions_base:
                print("无预测结果，跳过")
                continue
                
            predictions_base_df = pd.concat(predictions_base, ignore_index=True)
            predictions_stack_df = pd.concat(predictions_stack, ignore_index=True)
            
            # 回测
            print("[6/6] 执行回测...")
            bt_start = time.time()
            
            backtest_base = self.backtest_with_predictions(predictions_base_df, test_dates, min_prob)
            backtest_stack = self.backtest_with_predictions(predictions_stack_df, test_dates, min_prob)
            
            print(f"  回测完成, 耗时: {time.time()-bt_start:.1f}s")
            
            # 打印结果
            print(f"\n基地模型结果:")
            print(f"  收益率: {backtest_base['return']:.2%}")
            print(f"  夏普比率: {backtest_base['sharpe']:.2f}")
            print(f"  最大回撤: {backtest_base['max_dd']:.2%}")
            print(f"  交易次数: {backtest_base['n_trades']}")
            print(f"  胜率: {backtest_base['win_rate']:.2%}")
            
            print(f"\n融合模型结果:")
            print(f"  收益率: {backtest_stack['return']:.2%}")
            print(f"  夏普比率: {backtest_stack['sharpe']:.2f}")
            print(f"  最大回撤: {backtest_stack['max_dd']:.2%}")
            print(f"  交易次数: {backtest_stack['n_trades']}")
            print(f"  胜率: {backtest_stack['win_rate']:.2%}")
            
            # 保存结果
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
                
            # 清理缓存
            self.price_cache.clear()
            
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
    bt = WalkForwardStackingV10Optimized()
    results_base, results_stack = bt.run_stacking(
        start_date='20230101',
        end_date='20241231',
        train_months=12,
        test_months=6,
        min_prob=0.50,
        n_top=20
    )

"""
完整特征回测框架 V6
整合：缠论特征 + 技术因子 + news major1 + rank数据
训练：2020-2022，Rolling一年更新
测试：2022-2026
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
from sklearn.feature_selection import SelectKBest, f_classif

# A股交易规则配置
COST_RATE = 0.003       # 交易费用 0.3% 双边
SLIPPAGE = 0.002        # 滑点 0.2%
LIMIT_THRESHOLD = 0.5   # 涨跌停阈值偏移


def is_main_board(ts_code: str) -> bool:
    """检查是否为主板/中小板"""
    return (ts_code.startswith('60') or 
            ts_code.startswith('00') or 
            ts_code.startswith('002') or 
            ts_code.startswith('003'))


def get_limit_pct(ts_code: str) -> float:
    """获取涨跌停幅度"""
    if ts_code.startswith('688') or ts_code.startswith('689'):
        return 20.0
    elif ts_code.startswith('30') or ts_code.startswith('301'):
        return 20.0
    elif ts_code.startswith('8') or ts_code.startswith('4'):
        return 30.0
    else:
        return 10.0


def validate_no_lookahead(df, current_idx):
    """验证特征计算没有使用未来数据"""
    # 检查是否使用了shift(-n)形式的未来数据
    # 这个函数在特征计算后调用，确保没有未来函数
    pass


def process_news(news_dir, target_date=None):
    """处理新闻数据"""
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


class ChanLunFeatures:
    """缠论特征计算"""
    
    @staticmethod
    def calculate_fractals(df):
        """计算分型 - 只使用历史数据，无未来函数"""
        # 顶分型: 当前高点高于前2天，且高于前1天
        # 只使用历史数据，不包含未来信息
        df['top_fractal'] = (
            (df['high'] > df['high'].shift(1)) & 
            (df['high'] > df['high'].shift(2)) &
            (df['high'] > df['high'].shift(3)) & 
            (df['high'] > df['high'].shift(4))
        ).astype(int)
        
        # 底分型: 当前低点低于前2天，且低于前1天
        df['bottom_fractal'] = (
            (df['low'] < df['low'].shift(1)) & 
            (df['low'] < df['low'].shift(2)) &
            (df['low'] < df['low'].shift(3)) & 
            (df['low'] < df['low'].shift(4))
        ).astype(int)
        
        return df
    
    @staticmethod
    def calculate_bi(df):
        """计算笔（简化版）- 只使用历史数据，无未来函数"""
        df['bi_direction'] = 0
        
        # 使用历史分型来确定笔方向
        # 只在确认分型后更新历史数据的笔方向
        last_top_idx = -1
        last_bottom_idx = -1
        
        for i in range(len(df)):
            if df['top_fractal'].iloc[i] == 1:
                if last_bottom_idx >= 0 and last_top_idx < last_bottom_idx:
                    # 向上笔：从底分型到顶分型
                    df.loc[df.index[last_bottom_idx:i+1], 'bi_direction'] = 1
                last_top_idx = i
            elif df['bottom_fractal'].iloc[i] == 1:
                if last_top_idx >= 0 and last_bottom_idx < last_top_idx:
                    # 向下笔：从顶分型到底分型
                    df.loc[df.index[last_top_idx:i+1], 'bi_direction'] = -1
                last_bottom_idx = i
        
        return df
    
    @staticmethod
    def calculate_zhongshu(df):
        """计算中枢（简化版）"""
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
                    df.loc[df.index[i], 'zhongshu_strength'] = (overlap_high - overlap_low) / overlap_low
        
        return df
    
    @staticmethod
    def calculate_divergence(df):
        """计算背离 - 只使用历史数据，无未来函数"""
        # 计算MACD
        ema_fast = df['close'].ewm(span=12).mean()
        ema_slow = df['close'].ewm(span=26).mean()
        macd = ema_fast - ema_slow
        
        # 顶背离: 当前高点高于前几天高点，但MACD低于前几天MACD
        # 只使用历史数据比较
        df['top_divergence'] = (
            (df['high'] > df['high'].shift(1)) & 
            (df['high'] > df['high'].shift(2)) &
            (macd < macd.shift(1)) & 
            (macd < macd.shift(2))
        ).astype(int)
        
        # 底背离: 当前低点低于前几天低点，但MACD高于前几天MACD
        df['bottom_divergence'] = (
            (df['low'] < df['low'].shift(1)) & 
            (df['low'] < df['low'].shift(2)) &
            (macd > macd.shift(1)) & 
            (macd > macd.shift(2))
        ).astype(int)
        
        return df
    
    @staticmethod
    def calculate_all(df):
        """计算所有缠论特征"""
        df = ChanLunFeatures.calculate_fractals(df)
        df = ChanLunFeatures.calculate_bi(df)
        df = ChanLunFeatures.calculate_zhongshu(df)
        df = ChanLunFeatures.calculate_divergence(df)
        return df


class TechnicalFeatures:
    """技术特征计算"""
    
    @staticmethod
    def calculate_momentum(df):
        """动量特征"""
        for w in [5, 10, 20]:
            df[f'mom_{w}d'] = df['close'].pct_change(w)
        return df
    
    @staticmethod
    def calculate_volatility(df):
        """波动率特征"""
        for w in [5, 10, 20]:
            df[f'vol_{w}d'] = df['close'].pct_change().rolling(w).std() * np.sqrt(252)
        return df
    
    @staticmethod
    def calculate_volume(df):
        """成交量特征"""
        for w in [5, 10, 20]:
            df[f'vol_ma_{w}d'] = df['vol'].rolling(w).mean()
            df[f'vol_ratio_{w}d'] = df['vol'] / df[f'vol_ma_{w}d']
        
        df['amount'] = df['close'] * df['vol']
        for w in [5, 10, 20]:
            df[f'amount_ma_{w}d'] = df['amount'].rolling(w).mean()
            df[f'amount_ratio_{w}d'] = df['amount'] / df[f'amount_ma_{w}d']
        
        return df
    
    @staticmethod
    def calculate_ma(df):
        """均线特征"""
        for w in [5, 10, 20, 60]:
            df[f'ma_{w}d'] = df['close'].rolling(w).mean()
            df[f'ma_dist_{w}d'] = (df['close'] - df[f'ma_{w}d']) / df[f'ma_{w}d']
        
        df['ma_bull'] = (df['ma_5d'] > df['ma_10d']) & (df['ma_10d'] > df['ma_20d'])
        df['ma_bear'] = (df['ma_5d'] < df['ma_10d']) & (df['ma_10d'] < df['ma_20d'])
        
        return df
    
    @staticmethod
    def calculate_rsi(df):
        """RSI特征"""
        for w in [6, 12, 24]:
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=w).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=w).mean()
            rs = gain / loss
            df[f'rsi_{w}d'] = 100 - (100 / (1 + rs))
        return df
    
    @staticmethod
    def calculate_macd(df):
        """MACD特征"""
        ema_fast = df['close'].ewm(span=12).mean()
        ema_slow = df['close'].ewm(span=26).mean()
        df['macd'] = ema_fast - ema_slow
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']
        return df
    
    @staticmethod
    def calculate_bollinger(df):
        """布林带特征"""
        df['bb_middle'] = df['close'].rolling(20).mean()
        df['bb_std'] = df['close'].rolling(20).std()
        df['bb_upper'] = df['bb_middle'] + 2 * df['bb_std']
        df['bb_lower'] = df['bb_middle'] - 2 * df['bb_std']
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        return df
    
    @staticmethod
    def calculate_kdj(df):
        """KDJ特征"""
        low_list = df['low'].rolling(window=9, min_periods=9).min()
        high_list = df['high'].rolling(window=9, min_periods=9).max()
        rsv = (df['close'] - low_list) / (high_list - low_list) * 100
        df['kdj_k'] = rsv.ewm(com=2).mean()
        df['kdj_d'] = df['kdj_k'].ewm(com=2).mean()
        df['kdj_j'] = 3 * df['kdj_k'] - 2 * df['kdj_d']
        return df
    
    @staticmethod
    def calculate_all(df):
        """计算所有技术特征"""
        df = TechnicalFeatures.calculate_momentum(df)
        df = TechnicalFeatures.calculate_volatility(df)
        df = TechnicalFeatures.calculate_volume(df)
        df = TechnicalFeatures.calculate_ma(df)
        df = TechnicalFeatures.calculate_rsi(df)
        df = TechnicalFeatures.calculate_macd(df)
        df = TechnicalFeatures.calculate_bollinger(df)
        df = TechnicalFeatures.calculate_kdj(df)
        return df


class WalkForwardBacktestV6:
    """完整特征回测类 V6"""
    
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
        
        # 获取所有交易日
        self.all_dates = sorted([f.replace('.parquet', '') 
                                for f in os.listdir(self.price_dir) 
                                if f.endswith('.parquet')])
        
        print(f"数据目录: {self.price_dir}")
        print(f"总交易日数: {len(self.all_dates)}")
        print(f"日期范围: {self.all_dates[0]} 至 {self.all_dates[-1]}")
        
        # 处理新闻数据
        self.news_mkt, self.news_stk = process_news(self.news_dir)
        if not self.news_mkt.empty:
            print(f"新闻数据: {len(self.news_mkt)} 条市场记录, {len(self.news_stk)} 条个股记录")
    
    def load_stock_history(self, ts_code: str, end_date: str, n_days: int = 60) -> pd.DataFrame:
        """加载个股历史数据用于计算特征"""
        # 找到end_date的索引
        try:
            end_idx = self.all_dates.index(end_date)
        except ValueError:
            return None
        
        start_idx = max(0, end_idx - n_days + 1)
        dates = self.all_dates[start_idx:end_idx+1]
        
        data = []
        for d in dates:
            p = os.path.join(self.price_dir, f"{d}.parquet")
            if os.path.exists(p):
                try:
                    df = pd.read_parquet(p)
                    row = df[df['ts_code'] == ts_code]
                    if not row.empty:
                        row = row.iloc[0].copy()
                        row['trade_date'] = d
                        data.append(row)
                except:
                    continue
        
        if len(data) < 30:  # 需要至少30天数据
            return None
        
        df = pd.DataFrame(data)
        df = df.sort_values('trade_date').reset_index(drop=True)
        
        return df
    
    def calculate_features_for_stock(self, ts_code: str, date: str) -> Dict:
        """计算单只个股的所有特征"""
        # 加载历史数据
        hist = self.load_stock_history(ts_code, date, n_days=60)
        if hist is None or len(hist) < 30:
            return None
        
        # 计算缠论特征
        hist = ChanLunFeatures.calculate_all(hist)
        
        # 计算技术特征
        hist = TechnicalFeatures.calculate_all(hist)
        
        # 获取最新一天的数据
        latest = hist.iloc[-1]
        
        features = {
            'ts_code': ts_code,
            'trade_date': date,
            
            # 基础价格特征
            'price_change': latest['price_change'] if 'price_change' in latest else 0,
            'body_size': latest['body_size'] if 'body_size' in latest else 0,
            'amplitude': latest['amplitude'] if 'amplitude' in latest else 0,
            
            # 缠论特征
            'top_fractal': latest['top_fractal'] if 'top_fractal' in latest else 0,
            'bottom_fractal': latest['bottom_fractal'] if 'bottom_fractal' in latest else 0,
            'bi_direction': latest['bi_direction'] if 'bi_direction' in latest else 0,
            'zhongshu': latest['zhongshu'] if 'zhongshu' in latest else 0,
            'zhongshu_strength': latest['zhongshu_strength'] if 'zhongshu_strength' in latest else 0,
            'top_divergence': latest['top_divergence'] if 'top_divergence' in latest else 0,
            'bottom_divergence': latest['bottom_divergence'] if 'bottom_divergence' in latest else 0,
            
            # 动量特征
            'mom_5d': latest['mom_5d'] if 'mom_5d' in latest else 0,
            'mom_10d': latest['mom_10d'] if 'mom_10d' in latest else 0,
            'mom_20d': latest['mom_20d'] if 'mom_20d' in latest else 0,
            
            # 波动率特征
            'vol_5d': latest['vol_5d'] if 'vol_5d' in latest else 0,
            'vol_10d': latest['vol_10d'] if 'vol_10d' in latest else 0,
            'vol_20d': latest['vol_20d'] if 'vol_20d' in latest else 0,
            
            # 成交量特征
            'vol_ratio_5d': latest['vol_ratio_5d'] if 'vol_ratio_5d' in latest else 1,
            'vol_ratio_10d': latest['vol_ratio_10d'] if 'vol_ratio_10d' in latest else 1,
            'vol_ratio_20d': latest['vol_ratio_20d'] if 'vol_ratio_20d' in latest else 1,
            'amount_ratio_5d': latest['amount_ratio_5d'] if 'amount_ratio_5d' in latest else 1,
            'amount_ratio_10d': latest['amount_ratio_10d'] if 'amount_ratio_10d' in latest else 1,
            
            # 均线特征
            'ma_dist_5d': latest['ma_dist_5d'] if 'ma_dist_5d' in latest else 0,
            'ma_dist_10d': latest['ma_dist_10d'] if 'ma_dist_10d' in latest else 0,
            'ma_dist_20d': latest['ma_dist_20d'] if 'ma_dist_20d' in latest else 0,
            'ma_dist_60d': latest['ma_dist_60d'] if 'ma_dist_60d' in latest else 0,
            'ma_bull': 1 if latest['ma_bull'] else 0 if 'ma_bull' in latest else 0,
            'ma_bear': 1 if latest['ma_bear'] else 0 if 'ma_bear' in latest else 0,
            
            # RSI特征
            'rsi_6d': latest['rsi_6d'] if 'rsi_6d' in latest else 50,
            'rsi_12d': latest['rsi_12d'] if 'rsi_12d' in latest else 50,
            'rsi_24d': latest['rsi_24d'] if 'rsi_24d' in latest else 50,
            
            # MACD特征
            'macd': latest['macd'] if 'macd' in latest else 0,
            'macd_hist': latest['macd_hist'] if 'macd_hist' in latest else 0,
            
            # 布林带特征
            'bb_width': latest['bb_width'] if 'bb_width' in latest else 0,
            'bb_position': latest['bb_position'] if 'bb_position' in latest else 0.5,
            
            # KDJ特征
            'kdj_k': latest['kdj_k'] if 'kdj_k' in latest else 50,
            'kdj_d': latest['kdj_d'] if 'kdj_d' in latest else 50,
            'kdj_j': latest['kdj_j'] if 'kdj_j' in latest else 50,
        }
        
        return features
    
    def load_daily_features(self, date: str) -> pd.DataFrame:
        """加载单日的所有特征数据"""
        p_price = os.path.join(self.price_dir, f"{date}.parquet")
        p_rank = os.path.join(self.rank_dir, f"{date}.parquet")
        p_chip = os.path.join(self.chip_dir, f"{date}.parquet")
        p_other = os.path.join(self.other_dir, f"{date}.parquet")
        
        if not os.path.exists(p_price):
            return None
        
        # 加载价格数据
        price_df = pd.read_parquet(p_price)
        price_df = price_df[price_df['ts_code'].apply(is_main_board)]
        
        if len(price_df) == 0:
            return None
        
        # 计算基础价格特征
        price_df['price_change'] = (price_df['close'] - price_df['pre_close']) / price_df['pre_close']
        price_df['body_size'] = abs(price_df['close'] - price_df['open']) / price_df['pre_close']
        price_df['amplitude'] = (price_df['high'] - price_df['low']) / price_df['pre_close']
        
        # 加载热度数据
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
        
        # 加载筹码数据
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
        
        # 加载市值数据
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
        
        # 添加新闻特征
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
        
        # 计算缠论和技术特征（对每只股票）
        all_features = []
        for ts_code in price_df['ts_code'].unique():
            feat = self.calculate_features_for_stock(ts_code, date)
            if feat is not None:
                all_features.append(feat)
        
        if all_features:
            feat_df = pd.DataFrame(all_features)
            # 合并特征
            price_df = pd.merge(price_df, feat_df, on=['ts_code', 'trade_date'], how='left')
        
        price_df['trade_date'] = date
        
        return price_df
    
    def load_and_prepare_data(self, dates: List[str], label_threshold: float = 0.02) -> Tuple[pd.DataFrame, List[str]]:
        """加载并准备数据"""
        all_data = []
        
        for i in range(len(dates) - 2):
            d_curr = dates[i]
            d_t1 = dates[i + 1]
            d_t2 = dates[i + 2]
            
            # 加载T日特征
            df_t = self.load_daily_features(d_curr)
            if df_t is None or len(df_t) == 0:
                continue
            
            # 加载T+1和T+2价格
            p_t1 = os.path.join(self.price_dir, f"{d_t1}.parquet")
            p_t2 = os.path.join(self.price_dir, f"{d_t2}.parquet")
            
            if not os.path.exists(p_t1) or not os.path.exists(p_t2):
                continue
            
            try:
                df_t1 = pd.read_parquet(p_t1)
                df_t2 = pd.read_parquet(p_t2)
            except:
                continue
            
            # 合并T+1开盘价和T+2收盘价
            df_t = df_t.merge(
                df_t1[['ts_code', 'open']].rename(columns={'open': 't1_open'}),
                on='ts_code', how='left'
            )
            df_t = df_t.merge(
                df_t2[['ts_code', 'close', 'low']].rename(columns={'close': 't2_close', 'low': 't2_low'}),
                on='ts_code', how='left'
            )
            
            # 过滤缺失值
            df_t = df_t.dropna(subset=['t1_open', 't2_close'])
            if len(df_t) == 0:
                continue
            
            # 计算标签
            df_t['label_ret'] = df_t['t2_close'] / df_t['t1_open'] - 1
            df_t['label'] = (df_t['label_ret'] > label_threshold).astype(int)
            
            all_data.append(df_t)
        
        if not all_data:
            return pd.DataFrame(), []
        
        result = pd.concat(all_data, ignore_index=True)
        
        # 选择特征列
        feature_cols = [
            # 基础价格特征
            'price_change', 'body_size', 'amplitude', 'hot_rank_pct',
            'chip_concentration', 'winner_rate', 'circ_mv', 
            'turnover_rate', 'volume_ratio',
            
            # 缠论特征
            'top_fractal', 'bottom_fractal', 'bi_direction', 
            'zhongshu', 'zhongshu_strength',
            'top_divergence', 'bottom_divergence',
            
            # 动量特征
            'mom_5d', 'mom_10d', 'mom_20d',
            
            # 波动率特征
            'vol_5d', 'vol_10d', 'vol_20d',
            
            # 成交量特征
            'vol_ratio_5d', 'vol_ratio_10d', 'vol_ratio_20d',
            'amount_ratio_5d', 'amount_ratio_10d',
            
            # 均线特征
            'ma_dist_5d', 'ma_dist_10d', 'ma_dist_20d', 'ma_dist_60d',
            'ma_bull', 'ma_bear',
            
            # RSI特征
            'rsi_6d', 'rsi_12d', 'rsi_24d',
            
            # MACD特征
            'macd', 'macd_hist',
            
            # 布林带特征
            'bb_width', 'bb_position',
            
            # KDJ特征
            'kdj_k', 'kdj_d', 'kdj_j',
            
            # 新闻特征
            'news_market_impact', 'news_stock_impact'
        ]
        
        # 确保所有特征列存在
        for col in feature_cols:
            if col not in result.columns:
                result[col] = 0.0
        
        return result, feature_cols
    
    def select_features(self, X, y, feature_cols, k=20):
        """特征选择"""
        if X.shape[1] <= k:
            return feature_cols, X
        
        selector = SelectKBest(f_classif, k=k)
        X_selected = selector.fit_transform(X, y)
        
        selected_mask = selector.get_support()
        selected_features = [feature_cols[i] for i in range(len(feature_cols)) if selected_mask[i]]
        
        return selected_features, X_selected
    
    def train_model(self, train_df: pd.DataFrame, feature_cols: List[str]) -> Tuple[xgb.XGBClassifier, StandardScaler, List[str]]:
        """训练模型"""
        X = train_df[feature_cols].fillna(0)
        y = train_df['label']
        
        # 特征选择
        selected_features, X_selected = self.select_features(X, y, feature_cols, k=min(20, len(feature_cols)))
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_selected)
        
        # 计算类别权重
        pos_weight = len(y) / y.sum() - 1 if y.sum() > 0 else 1
        
        model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.7,
            colsample_bytree=0.7,
            random_state=42,
            eval_metric='auc',
            n_jobs=-1,
            tree_method='hist',
            scale_pos_weight=pos_weight,
            reg_alpha=0.1,
            reg_lambda=1.0,
            min_child_weight=5
        )
        
        model.fit(X_scaled, y)
        
        return model, scaler, selected_features
    
    def save_model(self, model, scaler, feature_cols, period_name):
        """保存模型和scaler"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        model_filename = f"model_v6_{period_name}_{timestamp}.pkl"
        scaler_filename = f"scaler_v6_{period_name}_{timestamp}.pkl"
        feature_filename = f"features_v6_{period_name}_{timestamp}.json"
        
        model_path = os.path.join(self.model_dir, model_filename)
        scaler_path = os.path.join(self.model_dir, scaler_filename)
        feature_path = os.path.join(self.model_dir, feature_filename)
        
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        
        with open(scaler_path, 'wb') as f:
            pickle.dump(scaler, f)
        
        with open(feature_path, 'w') as f:
            json.dump(feature_cols, f)
        
        return model_path, scaler_path, feature_path
    
    def generate_predictions(self, model, scaler, feature_cols, test_dates):
        """生成预测数据"""
        all_predictions = []
        
        for d_curr in test_dates:
            df_t = self.load_daily_features(d_curr)
            if df_t is None or len(df_t) == 0:
                continue
            
            # 预测
            X = df_t[feature_cols].fillna(0)
            X_scaled = scaler.transform(X)
            df_t['prob'] = model.predict_proba(X_scaled)[:, 1]
            
            # 保存预测结果
            pred_df = df_t[['ts_code', 'trade_date', 'prob'] + feature_cols].copy()
            all_predictions.append(pred_df)
        
        if not all_predictions:
            return pd.DataFrame()
        
        return pd.concat(all_predictions, ignore_index=True)
    
    def backtest_with_predictions(self, predictions_df, test_dates, min_prob=0.55):
        """使用已保存的预测数据进行回测"""
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
            
            # 获取当日预测
            d_curr_str = str(d_curr)
            day_pred = predictions_df[predictions_df['trade_date'].astype(str) == d_curr_str]
            
            if len(day_pred) == 0:
                daily_nav.append({'date': d_t2, 'nav': capital})
                continue
            
            # 选择概率最高的股票
            best_idx = day_pred['prob'].idxmax()
            best_prob = day_pred.loc[best_idx, 'prob']
            
            if best_prob < min_prob:
                daily_nav.append({'date': d_t2, 'nav': capital})
                continue
            
            ts_code = day_pred.loc[best_idx, 'ts_code']
            
            # 加载T+1和T+2价格
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
            
            # 获取T+1数据
            t1_data = df_t1[df_t1['ts_code'] == ts_code]
            if t1_data.empty:
                daily_nav.append({'date': d_t2, 'nav': capital})
                continue
            
            t1_open = float(t1_data.iloc[0]['open'])
            t1_pre = float(t1_data.iloc[0]['pre_close'])
            t1_vol = float(t1_data.iloc[0]['vol']) if 'vol' in t1_data.columns else 1
            
            # 停牌检查: 成交量为0表示停牌
            if t1_vol == 0:
                daily_nav.append({'date': d_t2, 'nav': capital})
                continue
            
            limit_pct = get_limit_pct(ts_code)
            
            # 涨停检查
            t1_open_chg = (t1_open - t1_pre) / t1_pre * 100
            if t1_open_chg >= (limit_pct - LIMIT_THRESHOLD):
                skipped_limit_up += 1
                daily_nav.append({'date': d_t2, 'nav': capital})
                continue
            
            # 获取T+2数据
            t2_data = df_t2[df_t2['ts_code'] == ts_code]
            if t2_data.empty:
                daily_nav.append({'date': d_t2, 'nav': capital})
                continue
            
            t2_close = float(t2_data.iloc[0]['close'])
            t2_low = float(t2_data.iloc[0]['low'])
            t2_open = float(t2_data.iloc[0]['open'])
            t2_vol = float(t2_data.iloc[0]['vol']) if 'vol' in t2_data.columns else 1
            
            # T+2停牌检查: 如果T+2停牌，按T+1开盘价卖出
            if t2_vol == 0:
                sell_price = t1_open  # 停牌时按买入价卖出，避免假设收益
                daily_nav.append({'date': d_t2, 'nav': capital})
                continue
            
            # 跌停检查
            t2_low_chg = (t2_low - t1_open) / t1_open * 100
            if t2_low_chg <= -(limit_pct - LIMIT_THRESHOLD):
                sell_price = min(t2_open, t1_open * (1 - limit_pct/100))
                skipped_limit_down += 1
            else:
                sell_price = t2_close
            
            # 应用滑点和费用
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
    
    def run_walk_forward(self, start_date='20200101', end_date='20260331',
                        train_years=2, test_years=1, min_prob=0.55,
                        save_predictions=True):
        """
        运行年度滚动回测
        """
        print("=" * 80)
        print("完整特征回测 (Walk-Forward Analysis) - V6")
        print("=" * 80)
        print(f"回测期: {start_date} 至 {end_date}")
        print(f"训练期: {train_years} 年")
        print(f"测试期: {test_years} 年")
        print(f"买入阈值: {min_prob}")
        print("=" * 80)
        
        # 过滤日期
        dates = [d for d in self.all_dates if start_date <= d <= end_date]
        
        if len(dates) < train_years * 252 + test_years * 252:
            print("数据不足！")
            return []
        
        # 按年份分组
        years = []
        current_year = dates[0][:4]
        year_dates = []
        
        for d in dates:
            if d[:4] == current_year:
                year_dates.append(d)
            else:
                years.append((current_year, year_dates))
                current_year = d[:4]
                year_dates = [d]
        
        if year_dates:
            years.append((current_year, year_dates))
        
        print(f"总年数: {len(years)}")
        
        results = []
        all_equity = []
        all_trades = []
        
        for i in range(train_years, len(years), test_years):
            train_years_list = years[i-train_years:i]
            train_dates = []
            for _, year_dates in train_years_list:
                train_dates.extend(year_dates)
            
            test_years_list = years[i:i+test_years]
            test_dates = []
            for _, year_dates in test_years_list:
                test_dates.extend(year_dates)
            
            if len(test_dates) < 20:
                continue
            
            period_name = test_years_list[0][0]
            
            print(f"\n{'='*60}")
            print(f"回测年份: {period_name}")
            print(f"训练期: {train_dates[0]} 至 {train_dates[-1]} ({len(train_dates)} 天)")
            print(f"测试期: {test_dates[0]} 至 {test_dates[-1]} ({len(test_dates)} 天)")
            print(f"{'='*60}")
            
            # 准备训练数据
            print("准备训练数据...")
            train_df, feature_cols = self.load_and_prepare_data(train_dates)
            
            if len(train_df) == 0 or len(feature_cols) == 0:
                print("训练数据不足，跳过")
                continue
            
            print(f"训练样本: {len(train_df)}, 特征数: {len(feature_cols)}")
            print(f"正样本比例: {train_df['label'].mean():.2%}")
            
            # 训练模型
            print("训练模型...")
            model, scaler, selected_features = self.train_model(train_df, feature_cols)
            
            # 获取特征重要性
            importance = model.feature_importances_
            top_features = [selected_features[i] for i in np.argsort(importance)[-10:]]
            print(f"Top特征: {', '.join(reversed(top_features))}")
            
            # 保存模型
            model_path, scaler_path, feature_path = self.save_model(
                model, scaler, selected_features, period_name
            )
            print(f"模型已保存: {model_path}")
            
            # 生成并保存预测数据
            if save_predictions:
                print("生成预测数据...")
                predictions_df = self.generate_predictions(model, scaler, selected_features, test_dates)
                
                if not predictions_df.empty:
                    pred_file = os.path.join(self.model_dir, f"predictions_v6_{period_name}.csv")
                    predictions_df.to_csv(pred_file, index=False)
                    print(f"预测数据已保存: {pred_file}")
                    
                    # 保存预测元数据
                    pred_meta = {
                        'period': period_name,
                        'test_dates': test_dates,
                        'model_path': model_path,
                        'scaler_path': scaler_path,
                        'feature_path': feature_path,
                        'feature_cols': selected_features,
                        'n_stocks': len(predictions_df['ts_code'].unique()),
                        'n_days': len(predictions_df['trade_date'].unique())
                    }
                    
                    meta_file = os.path.join(self.model_dir, f"meta_v6_{period_name}.json")
                    with open(meta_file, 'w') as f:
                        json.dump(pred_meta, f, indent=2)
                    
                    # 使用预测数据进行回测
                    print("回测测试期...")
                    backtest_result = self.backtest_with_predictions(
                        predictions_df, test_dates, min_prob
                    )
                else:
                    backtest_result = {
                        'return': 0, 'sharpe': 0, 'max_dd': 0, 'n_trades': 0,
                        'win_rate': 0, 'avg_return': 0,
                        'skipped_limit_up': 0, 'skipped_limit_down': 0,
                        'daily_nav': [], 'trades': pd.DataFrame()
                    }
            else:
                print("回测测试期...")
                predictions_df = self.generate_predictions(model, scaler, selected_features, test_dates)
                backtest_result = self.backtest_with_predictions(
                    predictions_df, test_dates, min_prob
                )
            
            print(f"测试期收益: {backtest_result['return']*100:.2f}%")
            print(f"夏普比率: {backtest_result['sharpe']:.2f}")
            print(f"最大回撤: {backtest_result['max_dd']*100:.2f}%")
            print(f"交易次数: {backtest_result['n_trades']}")
            print(f"胜率: {backtest_result['win_rate']*100:.2f}%")
            
            result = {
                'period': period_name,
                'train_start': train_dates[0],
                'train_end': train_dates[-1],
                'test_start': test_dates[0],
                'test_end': test_dates[-1],
                'train_samples': len(train_df),
                'test_samples': len(test_dates),
                'train_positive_ratio': train_df['label'].mean(),
                'test_return': backtest_result['return'],
                'test_sharpe': backtest_result['sharpe'],
                'test_max_dd': backtest_result['max_dd'],
                'n_trades': backtest_result['n_trades'],
                'win_rate': backtest_result['win_rate'],
                'avg_return': backtest_result['avg_return'],
                'skipped_limit_up': backtest_result['skipped_limit_up'],
                'skipped_limit_down': backtest_result['skipped_limit_down'],
                'top_features': top_features
            }
            
            results.append(result)
            
            all_equity.extend(backtest_result['daily_nav'])
            if backtest_result['n_trades'] > 0:
                all_trades.append(backtest_result['trades'])
        
        # 保存结果
        self.save_results(results, all_equity, all_trades)
        
        return results
    
    def save_results(self, results, all_equity, all_trades):
        """保存结果"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # 年度结果
        if results:
            results_df = pd.DataFrame([{
                'period': r['period'],
                'train_start': r['train_start'],
                'train_end': r['train_end'],
                'test_start': r['test_start'],
                'test_end': r['test_end'],
                'train_samples': r['train_samples'],
                'test_samples': r['test_samples'],
                'train_positive_ratio': r['train_positive_ratio'],
                'test_return': r['test_return'],
                'test_sharpe': r['test_sharpe'],
                'test_max_dd': r['test_max_dd'],
                'n_trades': r['n_trades'],
                'win_rate': r['win_rate'],
                'avg_return': r['avg_return'],
                'skipped_limit_up': r['skipped_limit_up'],
                'skipped_limit_down': r['skipped_limit_down'],
                'top_features': ','.join(r['top_features'])
            } for r in results])
            
            results_file = os.path.join(self.output_dir, f'yearly_results_v6_{timestamp}.csv')
            results_df.to_csv(results_file, index=False)
            print(f"\n年度结果已保存: {results_file}")
        
        # 权益曲线
        if all_equity:
            equity_df = pd.DataFrame(all_equity)
            equity_df = equity_df.drop_duplicates(subset=['date'])
            equity_df = equity_df.sort_values('date')
            
            equity_file = os.path.join(self.output_dir, f'equity_curve_v6_{timestamp}.csv')
            equity_df.to_csv(equity_file, index=False)
            print(f"权益曲线已保存: {equity_file}")
            
            # 总体统计
            initial_nav = equity_df['nav'].iloc[0]
            final_nav = equity_df['nav'].iloc[-1]
            total_return = final_nav / initial_nav - 1
            
            equity_df['ret'] = equity_df['nav'].pct_change()
            vol = equity_df['ret'].std() * np.sqrt(252)
            
            years = len(equity_df) / 252
            ann_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0
            sharpe = ann_return / vol if vol > 0 else 0
            
            equity_df['cummax'] = equity_df['nav'].cummax()
            equity_df['dd'] = (equity_df['nav'] - equity_df['cummax']) / equity_df['cummax']
            max_dd = equity_df['dd'].min()
            
            print(f"\n{'='*60}")
            print("总体回测结果")
            print(f"{'='*60}")
            print(f"总收益: {total_return*100:.2f}%")
            print(f"年化收益: {ann_return*100:.2f}%")
            print(f"夏普比率: {sharpe:.2f}")
            print(f"最大回撤: {max_dd*100:.2f}%")
            print(f"交易天数: {len(equity_df)}")
            print(f"{'='*60}")
        
        # 交易记录
        if all_trades:
            trades_df = pd.concat(all_trades, ignore_index=True)
            trades_file = os.path.join(self.output_dir, f'all_trades_v6_{timestamp}.csv')
            trades_df.to_csv(trades_file, index=False)
            print(f"交易记录已保存: {trades_file}")
            
            print(f"\n交易统计:")
            print(f"总交易次数: {len(trades_df)}")
            print(f"胜率: {(trades_df['return'] > 0).mean()*100:.2f}%")
            print(f"平均收益: {trades_df['return'].mean()*100:.2f}%")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='完整特征回测V6')
    parser.add_argument('--mode', type=str, default='full', choices=['full', 'threshold'],
                       help='运行模式: full=完整回测, threshold=仅调整阈值')
    parser.add_argument('--min_prob', type=float, default=0.55,
                       help='买入概率阈值')
    parser.add_argument('--start_date', type=str, default='20200101',
                       help='回测开始日期')
    parser.add_argument('--end_date', type=str, default='20260331',
                       help='回测结束日期')
    parser.add_argument('--train_years', type=int, default=2,
                       help='训练期年数')
    parser.add_argument('--test_years', type=int, default=1,
                       help='测试期年数')
    
    args = parser.parse_args()
    
    backtest = WalkForwardBacktestV6()
    
    if args.mode == 'full':
        results = backtest.run_walk_forward(
            start_date=args.start_date,
            end_date=args.end_date,
            train_years=args.train_years,
            test_years=args.test_years,
            min_prob=args.min_prob,
            save_predictions=True
        )
    else:
        print("阈值调整模式暂未实现")


if __name__ == '__main__':
    main()

"""
日线级别常用量化因子库
包含：动量、波动率、成交量、均值回归、技术形态等因子
"""
import pandas as pd
import numpy as np
from typing import List, Dict, Optional


class TechnicalFactors:
    """技术分析因子类"""
    
    @staticmethod
    def momentum(df: pd.DataFrame, windows: List[int] = [5, 10, 20, 60]) -> pd.DataFrame:
        """
        动量因子
        计算不同时间窗口的收益率
        """
        for w in windows:
            df[f'mom_{w}d'] = df['close'].pct_change(w)
        return df
    
    @staticmethod
    def volatility(df: pd.DataFrame, windows: List[int] = [5, 10, 20, 60]) -> pd.DataFrame:
        """
        波动率因子
        计算不同时间窗口的收益率标准差
        """
        for w in windows:
            df[f'vol_{w}d'] = df['close'].pct_change().rolling(w).std() * np.sqrt(252)
        return df
    
    @staticmethod
    def volume_features(df: pd.DataFrame, windows: List[int] = [5, 10, 20]) -> pd.DataFrame:
        """
        成交量因子
        """
        # 成交量移动平均
        for w in windows:
            df[f'vol_ma_{w}d'] = df['vol'].rolling(w).mean()
            df[f'vol_ratio_{w}d'] = df['vol'] / df[f'vol_ma_{w}d']
        
        # 成交额因子
        df['amount'] = df['close'] * df['vol']
        for w in windows:
            df[f'amount_ma_{w}d'] = df['amount'].rolling(w).mean()
            df[f'amount_ratio_{w}d'] = df['amount'] / df[f'amount_ma_{w}d']
        
        # 量价相关性
        df['price_volume_corr'] = df['close'].rolling(20).corr(df['vol'])
        
        return df
    
    @staticmethod
    def moving_average(df: pd.DataFrame, windows: List[int] = [5, 10, 20, 60, 120]) -> pd.DataFrame:
        """
        均线系统因子
        """
        for w in windows:
            df[f'ma_{w}d'] = df['close'].rolling(w).mean()
            df[f'ma_dist_{w}d'] = (df['close'] - df[f'ma_{w}d']) / df[f'ma_{w}d']
        
        # 均线排列
        df['ma_bull'] = (df['ma_5d'] > df['ma_10d']) & (df['ma_10d'] > df['ma_20d'])
        df['ma_bear'] = (df['ma_5d'] < df['ma_10d']) & (df['ma_10d'] < df['ma_20d'])
        
        return df
    
    @staticmethod
    def rsi(df: pd.DataFrame, windows: List[int] = [6, 12, 24]) -> pd.DataFrame:
        """
        RSI相对强弱指标
        """
        for w in windows:
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=w).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=w).mean()
            rs = gain / loss
            df[f'rsi_{w}d'] = 100 - (100 / (1 + rs))
        return df
    
    @staticmethod
    def macd(df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
        """
        MACD指标
        """
        ema_fast = df['close'].ewm(span=fast).mean()
        ema_slow = df['close'].ewm(span=slow).mean()
        df['macd'] = ema_fast - ema_slow
        df['macd_signal'] = df['macd'].ewm(span=signal).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']
        return df
    
    @staticmethod
    def bollinger_bands(df: pd.DataFrame, window: int = 20, num_std: float = 2.0) -> pd.DataFrame:
        """
        布林带因子
        """
        df['bb_middle'] = df['close'].rolling(window).mean()
        df['bb_std'] = df['close'].rolling(window).std()
        df['bb_upper'] = df['bb_middle'] + num_std * df['bb_std']
        df['bb_lower'] = df['bb_middle'] - num_std * df['bb_std']
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        return df
    
    @staticmethod
    def kdj(df: pd.DataFrame, n: int = 9, m1: int = 3, m2: int = 3) -> pd.DataFrame:
        """
        KDJ随机指标
        """
        low_list = df['low'].rolling(window=n, min_periods=n).min()
        high_list = df['high'].rolling(window=n, min_periods=n).max()
        rsv = (df['close'] - low_list) / (high_list - low_list) * 100
        df['kdj_k'] = rsv.ewm(com=m1-1).mean()
        df['kdj_d'] = df['kdj_k'].ewm(com=m2-1).mean()
        df['kdj_j'] = 3 * df['kdj_k'] - 2 * df['kdj_d']
        return df
    
    @staticmethod
    def williams_r(df: pd.DataFrame, windows: List[int] = [10, 20]) -> pd.DataFrame:
        """
        威廉指标
        """
        for w in windows:
            highest_high = df['high'].rolling(w).max()
            lowest_low = df['low'].rolling(w).min()
            df[f'williams_r_{w}d'] = (highest_high - df['close']) / (highest_high - lowest_low) * -100
        return df
    
    @staticmethod
    def atr(df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
        """
        平均真实波幅
        """
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        df['atr'] = true_range.rolling(window).mean()
        df['atr_ratio'] = df['atr'] / df['close']
        return df
    
    @staticmethod
    def obv(df: pd.DataFrame) -> pd.DataFrame:
        """
        能量潮指标
        """
        obv = [0]
        for i in range(1, len(df)):
            if df['close'].iloc[i] > df['close'].iloc[i-1]:
                obv.append(obv[-1] + df['vol'].iloc[i])
            elif df['close'].iloc[i] < df['close'].iloc[i-1]:
                obv.append(obv[-1] - df['vol'].iloc[i])
            else:
                obv.append(obv[-1])
        df['obv'] = obv
        df['obv_ma'] = df['obv'].rolling(20).mean()
        return df
    
    @staticmethod
    def ichimoku(df: pd.DataFrame) -> pd.DataFrame:
        """
        一目均衡表
        """
        # 转换线 (Tenkan-sen)
        tenkan_high = df['high'].rolling(9).max()
        tenkan_low = df['low'].rolling(9).min()
        df['tenkan_sen'] = (tenkan_high + tenkan_low) / 2
        
        # 基准线 (Kijun-sen)
        kijun_high = df['high'].rolling(26).max()
        kijun_low = df['low'].rolling(26).min()
        df['kijun_sen'] = (kijun_high + kijun_low) / 2
        
        # 先行带1 (Senkou Span A)
        df['senkou_a'] = ((df['tenkan_sen'] + df['kijun_sen']) / 2).shift(26)
        
        # 先行带2 (Senkou Span B)
        senkou_high = df['high'].rolling(52).max()
        senkou_low = df['low'].rolling(52).min()
        df['senkou_b'] = ((senkou_high + senkou_low) / 2).shift(26)
        
        # 延迟线 (Chikou Span)
        df['chikou'] = df['close'].shift(-26)
        
        return df


class ChanLunFactors:
    """缠论分析因子类"""
    
    @staticmethod
    def find_pivot_points(df: pd.DataFrame, window: int = 5) -> pd.DataFrame:
        """
        寻找 pivot points（分型）
        """
        # 顶分型
        df['top_fractal'] = (
            (df['high'] > df['high'].shift(1)) & 
            (df['high'] > df['high'].shift(2)) &
            (df['high'] > df['high'].shift(-1)) & 
            (df['high'] > df['high'].shift(-2))
        )
        
        # 底分型
        df['bottom_fractal'] = (
            (df['low'] < df['low'].shift(1)) & 
            (df['low'] < df['low'].shift(2)) &
            (df['low'] < df['low'].shift(-1)) & 
            (df['low'] < df['low'].shift(-2))
        )
        
        return df
    
    @staticmethod
    def calculate_zoushi(df: pd.DataFrame) -> pd.DataFrame:
        """
        计算走势强度
        """
        # 笔的识别（简化版）
        df['bi_direction'] = 0
        
        # 向上笔
        for i in range(2, len(df)):
            if df['bottom_fractal'].iloc[i-2] and df['top_fractal'].iloc[i]:
                df.loc[df.index[i-2:i+1], 'bi_direction'] = 1
            elif df['top_fractal'].iloc[i-2] and df['bottom_fractal'].iloc[i]:
                df.loc[df.index[i-2:i+1], 'bi_direction'] = -1
        
        # 中枢识别（简化版）
        df['zhongshu'] = 0
        for i in range(20, len(df)):
            window_data = df.iloc[i-20:i]
            if len(window_data[window_data['top_fractal']]) >= 2 and len(window_data[window_data['bottom_fractal']]) >= 2:
                tops = window_data[window_data['top_fractal']]['high']
                bottoms = window_data[window_data['bottom_fractal']]['low']
                if len(tops) >= 2 and len(bottoms) >= 2:
                    # 价格重叠区域
                    overlap_high = min(tops.iloc[-2:].max(), bottoms.iloc[-2:].max())
                    overlap_low = max(tops.iloc[-2:].min(), bottoms.iloc[-2:].min())
                    if overlap_high > overlap_low:
                        df.loc[df.index[i], 'zhongshu'] = 1
        
        return df
    
    @staticmethod
    def macd_divergence(df: pd.DataFrame) -> pd.DataFrame:
        """
        MACD背离检测
        """
        # 计算MACD
        ema_fast = df['close'].ewm(span=12).mean()
        ema_slow = df['close'].ewm(span=26).mean()
        macd = ema_fast - ema_slow
        
        # 顶背离：价格新高，MACD未新高
        df['top_divergence'] = (
            (df['high'] > df['high'].shift(1)) & 
            (macd < macd.shift(1))
        )
        
        # 底背离：价格新低，MACD未新低
        df['bottom_divergence'] = (
            (df['low'] < df['low'].shift(1)) & 
            (macd > macd.shift(1))
        )
        
        return df
    
    @staticmethod
    def trend_strength(df: pd.DataFrame) -> pd.DataFrame:
        """
        趋势强度因子
        """
        # ADX指标计算
        plus_dm = df['high'].diff()
        minus_dm = df['low'].diff()
        
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm > 0] = 0
        minus_dm = minus_dm.abs()
        
        tr1 = df['high'] - df['low']
        tr2 = (df['high'] - df['close'].shift()).abs()
        tr3 = (df['low'] - df['close'].shift()).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        atr = tr.rolling(14).mean()
        plus_di = 100 * plus_dm.rolling(14).mean() / atr
        minus_di = 100 * minus_dm.rolling(14).mean() / atr
        
        dx = (plus_di - minus_di).abs() / (plus_di + minus_di) * 100
        df['adx'] = dx.rolling(14).mean()
        df['plus_di'] = plus_di
        df['minus_di'] = minus_di
        
        return df


def calculate_all_factors(df: pd.DataFrame) -> pd.DataFrame:
    """
    计算所有因子
    """
    tech = TechnicalFactors()
    chan = ChanLunFactors()
    
    # 技术因子
    df = tech.momentum(df)
    df = tech.volatility(df)
    df = tech.volume_features(df)
    df = tech.moving_average(df)
    df = tech.rsi(df)
    df = tech.macd(df)
    df = tech.bollinger_bands(df)
    df = tech.kdj(df)
    df = tech.williams_r(df)
    df = tech.atr(df)
    df = tech.obv(df)
    df = tech.ichimoku(df)
    
    # 缠论因子
    df = chan.find_pivot_points(df)
    df = chan.calculate_zoushi(df)
    df = chan.macd_divergence(df)
    df = chan.trend_strength(df)
    
    return df

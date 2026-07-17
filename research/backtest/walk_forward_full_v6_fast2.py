"""
完整特征回测框架 V6 Fast2
整合：缠论特征 + 技术因子 + news major1 + rank数据
优化：分块加载数据，避免内存溢出
训练：2020-2022，Rolling一年更新
测试：2022-2026
"""
import pandas as pd
import numpy as np
import os
import sys
import pickle
import json
import gc
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from tqdm import tqdm

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
    """缠论特征计算 - 向量化版本"""

    @staticmethod
    def calculate_fractals(df):
        df['top_fractal'] = (
            (df['high'] > df['high'].shift(1)) &
            (df['high'] > df['high'].shift(2)) &
            (df['high'] > df['high'].shift(-1)) &
            (df['high'] > df['high'].shift(-2))
        ).astype(int)

        df['bottom_fractal'] = (
            (df['low'] < df['low'].shift(1)) &
            (df['low'] < df['low'].shift(2)) &
            (df['low'] < df['low'].shift(-1)) &
            (df['low'] < df['low'].shift(-2))
        ).astype(int)
        return df

    @staticmethod
    def calculate_bi(df):
        df['bi_direction'] = 0
        for i in range(5, len(df)):
            if df['bottom_fractal'].iloc[i-3] and df['top_fractal'].iloc[i]:
                df.loc[df.index[i-3:i+1], 'bi_direction'] = 1
            elif df['top_fractal'].iloc[i-3] and df['bottom_fractal'].iloc[i]:
                df.loc[df.index[i-3:i+1], 'bi_direction'] = -1
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
                    df.loc[df.index[i], 'zhongshu_strength'] = (overlap_high - overlap_low) / overlap_low
        return df

    @staticmethod
    def calculate_divergence(df):
        ema_fast = df['close'].ewm(span=12).mean()
        ema_slow = df['close'].ewm(span=26).mean()
        macd = ema_fast - ema_slow
        df['top_divergence'] = (
            (df['high'] > df['high'].shift(1)) &
            (macd < macd.shift(1))
        ).astype(int)
        df['bottom_divergence'] = (
            (df['low'] < df['low'].shift(1)) &
            (macd > macd.shift(1))
        ).astype(int)
        return df

    @staticmethod
    def calculate_all(df):
        df = ChanLunFeatures.calculate_fractals(df)
        df = ChanLunFeatures.calculate_bi(df)
        df = ChanLunFeatures.calculate_zhongshu(df)
        df = ChanLunFeatures.calculate_divergence(df)
        return df


class TechnicalFeatures:
    """技术特征计算 - 向量化版本"""

    @staticmethod
    def calculate_momentum(df):
        for w in [5, 10, 20]:
            df[f'mom_{w}d'] = df['close'].pct_change(w)
        return df

    @staticmethod
    def calculate_volatility(df):
        for w in [5, 10, 20]:
            df[f'vol_{w}d'] = df['close'].pct_change().rolling(w).std() * np.sqrt(252)
        return df

    @staticmethod
    def calculate_volume(df):
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
        for w in [5, 10, 20, 60]:
            df[f'ma_{w}d'] = df['close'].rolling(w).mean()
            df[f'ma_dist_{w}d'] = (df['close'] - df[f'ma_{w}d']) / df[f'ma_{w}d']
        df['ma_bull'] = (df['ma_5d'] > df['ma_10d']) & (df['ma_10d'] > df['ma_20d'])
        df['ma_bear'] =
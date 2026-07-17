"""
截面选股因子模块 - Cross-Sectional Stock Selection Factors
包含：基本面因子、行为金融因子、另类数据因子
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import warnings
warnings.filterwarnings('ignore')


# ==================== 数据加载 ====================

DATA_PATHS = {
    'price': r'D:\iquant_data\data_v2\data_day1',
    'fundamental': r'D:\iquant_data\data_v2\fundamental1',
    'income': r'D:\iquant_data\data_v2\income1',
    'moneyflow': r'D:\iquant_data\data_v2\moneyflow1',
    'industry': r'D:\iquant_data\data_v2\industry1',
    'cyq': r'D:\iquant_data\data_v2\cyq1',
    'ths_rank': r'D:\iquant_data\data_v2\ths_rank1',
    'ths_news': r'D:\iquant_data\data_v2\ths_news1',
    'board': r'D:\iquant_data\data_v2\board1',
}


def load_price_range(start_date: str, end_date: str) -> pd.DataFrame:
    """加载价格数据范围"""
    price_dir = DATA_PATHS['price']
    all_files = sorted([f.replace('.parquet', '') for f in os.listdir(price_dir) if f.endswith('.parquet')])
    
    try:
        s_idx = all_files.index(start_date)
    except ValueError:
        s_idx = 0
    try:
        e_idx = all_files.index(end_date)
    except ValueError:
        e_idx = len(all_files) - 1
    
    dates = all_files[s_idx:e_idx+1]
    dfs = []
    for d in dates:
        p = os.path.join(price_dir, f"{d}.parquet")
        df = pd.read_parquet(p)
        df['trade_date'] = d
        dfs.append(df)
    
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()


def load_fundamental_data() -> pd.DataFrame:
    """加载基本面指标数据（季度/年度）"""
    p = os.path.join(DATA_PATHS['fundamental'], 'fina_indicator_cache.parquet')
    if os.path.exists(p):
        df = pd.read_parquet(p)
        df['ann_date'] = df['ann_date'].astype(str)
        df['end_date'] = df['end_date'].astype(str)
        return df
    return pd.DataFrame()


def load_industry_map() -> pd.DataFrame:
    """加载行业映射"""
    p = os.path.join(DATA_PATHS['industry'], 'industry.parquet')
    if os.path.exists(p):
        return pd.read_parquet(p)
    return pd.DataFrame()


def load_moneyflow_range(start_date: str, end_date: str) -> pd.DataFrame:
    """加载资金流数据"""
    mf_dir = DATA_PATHS['moneyflow']
    all_files = sorted([f.replace('.parquet', '') for f in os.listdir(mf_dir) if f.endswith('.parquet')])
    
    try:
        s_idx = all_files.index(start_date)
    except ValueError:
        s_idx = 0
    try:
        e_idx = all_files.index(end_date)
    except ValueError:
        e_idx = len(all_files) - 1
    
    dates = all_files[s_idx:e_idx+1]
    dfs = []
    for d in dates:
        p = os.path.join(mf_dir, f"{d}.parquet")
        if os.path.exists(p):
            df = pd.read_parquet(p)
            df['trade_date'] = d
            dfs.append(df)
    
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()


def load_cyq_range(start_date: str, end_date: str) -> pd.DataFrame:
    """加载筹码分布数据"""
    cyq_dir = DATA_PATHS['cyq']
    all_files = sorted([f.replace('.parquet', '') for f in os.listdir(cyq_dir) if f.endswith('.parquet')])
    
    try:
        s_idx = all_files.index(start_date)
    except ValueError:
        s_idx = 0
    try:
        e_idx = all_files.index(end_date)
    except ValueError:
        e_idx = len(all_files) - 1
    
    dates = all_files[s_idx:e_idx+1]
    dfs = []
    for d in dates:
        p = os.path.join(cyq_dir, f"{d}.parquet")
        if os.path.exists(p):
            df = pd.read_parquet(p)
            df['trade_date'] = d
            dfs.append(df)
    
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()


# ==================== 因子预处理工具 ====================

def winsorize_series(s: pd.Series, lower: float = 0.01, upper: float = 0.99) -> pd.Series:
    """去极值（Winsorize）"""
    q_low = s.quantile(lower)
    q_high = s.quantile(upper)
    return s.clip(lower=q_low, upper=q_high)


def standardize_series(s: pd.Series) -> pd.Series:
    """标准化（z-score）"""
    mean = s.mean()
    std = s.std()
    if std == 0 or pd.isna(std):
        return pd.Series(0, index=s.index)
    return (s - mean) / std


def neutralize_industry(df: pd.DataFrame, factor_col: str, industry_col: str = 'industry') -> pd.Series:
    """行业中性化：在每个行业内标准化"""
    result = pd.Series(np.nan, index=df.index)
    for ind, group in df.groupby(industry_col):
        if len(group) > 1:
            result.loc[group.index] = standardize_series(group[factor_col])
    return result


def neutralize_market_cap(df: pd.DataFrame, factor_col: str, cap_col: str = 'log_mv') -> pd.Series:
    """市值中性化：对市值取残差"""
    from scipy import stats
    valid = df[[factor_col, cap_col]].dropna()
    if len(valid) < 10:
        return pd.Series(np.nan, index=df.index)
    
    slope, intercept, _, _, _ = stats.linregress(valid[cap_col], valid[factor_col])
    residual = df[factor_col] - (intercept + slope * df[cap_col])
    return residual


# ==================== 一、基本面因子 ====================

class FundamentalFactors:
    """基本面因子：从静态估值转向动态增长与质量"""
    
    @staticmethod
    def calculate_growth_acceleration(funda_df: pd.DataFrame) -> pd.DataFrame:
        """
        成长性因子：增速的加速度
        本期增速 - 上期增速
        """
        funda_df = funda_df.sort_values(['ts_code', 'end_date'])
        
        # 营收增速加速度
        funda_df['or_yoy_accel'] = funda_df.groupby('ts_code')['or_yoy'].diff()
        # 净利润增速加速度
        funda_df['netprofit_yoy_accel'] = funda_df.groupby('ts_code')['netprofit_yoy'].diff()
        
        return funda_df
    
    @staticmethod
    def calculate_profit_quality(funda_df: pd.DataFrame) -> pd.DataFrame:
        """
        盈利质量因子
        """
        # ROE稳定性（变异系数的倒数）
        funda_df = funda_df.sort_values(['ts_code', 'end_date'])
        funda_df['roe_cv'] = funda_df.groupby('ts_code')['roe'].transform(
            lambda x: x.rolling(4, min_periods=2).std() / (x.rolling(4, min_periods=2).mean().abs() + 1e-8)
        )
        funda_df['roe_stability'] = 1 / (funda_df['roe_cv'] + 1e-8)
        
        # 毛利率变化趋势
        funda_df['gpm_trend'] = funda_df.groupby('ts_code')['grossprofit_margin'].transform(
            lambda x: x.rolling(4, min_periods=2).apply(lambda y: np.polyfit(range(len(y)), y, 1)[0] if len(y) >= 2 else 0, raw=True)
        )
        
        # 净利润率变化趋势
        funda_df['npm_trend'] = funda_df.groupby('ts_code')['netprofit_margin'].transform(
            lambda x: x.rolling(4, min_periods=2).apply(lambda y: np.polyfit(range(len(y)), y, 1)[0] if len(y) >= 2 else 0, raw=True)
        )
        
        return funda_df
    
    @staticmethod
    def calculate_financial_health(funda_df: pd.DataFrame) -> pd.DataFrame:
        """
        财务健康因子
        """
        # 资产负债率变化（越低越好，所以取负）
        funda_df['debt_change'] = funda_df.groupby('ts_code')['debt_to_assets'].diff()
        funda_df['debt_improve'] = -funda_df['debt_change']
        
        # 速动比率改善
        funda_df['quick_change'] = funda_df.groupby('ts_code')['quick_ratio'].diff()
        
        return funda_df
    
    @classmethod
    def get_all_fundamental_factors(cls, funda_df: pd.DataFrame) -> pd.DataFrame:
        """计算所有基本面因子"""
        funda_df = cls.calculate_growth_acceleration(funda_df.copy())
        funda_df = cls.calculate_profit_quality(funda_df)
        funda_df = cls.calculate_financial_health(funda_df)
        
        # 选择最新公告的数据
        funda_df = funda_df.sort_values(['ts_code', 'ann_date'], ascending=[True, False])
        latest = funda_df.groupby('ts_code').first().reset_index()
        
        factor_cols = [
            'ts_code',
            'roe', 'roe_dt', 'roe_stability',
            'or_yoy', 'or_yoy_accel',
            'netprofit_yoy', 'netprofit_yoy_accel',
            'netprofit_margin', 'grossprofit_margin',
            'gpm_trend', 'npm_trend',
            'debt_to_assets', 'debt_improve',
            'quick_ratio', 'quick_change',
            'current_ratio'
        ]
        
        available_cols = [c for c in factor_cols if c in latest.columns]
        return latest[available_cols]


# ==================== 二、行为金融与市场面因子 ====================

class BehavioralFactors:
    """行为金融与市场面因子：A股特色的聪明钱与错误定价"""
    
    @staticmethod
    def calculate_reversal(price_df: pd.DataFrame) -> pd.DataFrame:
        """
        反转效应因子（精细化）
        """
        price_df = price_df.sort_values(['ts_code', 'trade_date'])
        
        # 短期收益（1个月 ≈ 20交易日）
        price_df['ret_1m'] = price_df.groupby('ts_code')['close'].pct_change(20)
        # 3个月收益
        price_df['ret_3m'] = price_df.groupby('ts_code')['close'].pct_change(60)
        # 1个月收益加速度（反转强度）
        price_df['ret_1m_accel'] = price_df.groupby('ts_code')['ret_1m'].diff(5)
        
        return price_df
    
    @staticmethod
    def calculate_overnight_intraday(price_df: pd.DataFrame) -> pd.DataFrame:
        """
        隔夜与日内收益率分离
        """
        # 隔夜收益率 = (今开 - 昨收) / 昨收
        price_df['overnight_ret'] = (price_df['open'] - price_df['pre_close']) / price_df['pre_close']
        # 日内收益率 = (今收 - 今开) / 今开
        price_df['intraday_ret'] = (price_df['close'] - price_df['open']) / price_df['open']
        
        # 5日平均
        price_df = price_df.sort_values(['ts_code', 'trade_date'])
        price_df['overnight_ret_5d'] = price_df.groupby('ts_code')['overnight_ret'].rolling(5, min_periods=1).mean().values
        price_df['intraday_ret_5d'] = price_df.groupby('ts_code')['intraday_ret'].rolling(5, min_periods=1).mean().values
        
        # 低隔夜 + 高日内反转（买入信号）
        price_df['overnight_intraday_spread'] = price_df['intraday_ret_5d'] - price_df['overnight_ret_5d']
        
        return price_df
    
    @staticmethod
    def calculate_moneyflow_factors(mf_df: pd.DataFrame) -> pd.DataFrame:
        """
        资金流因子（聪明钱）
        """
        mf_df = mf_df.sort_values(['ts_code', 'trade_date'])
        
        # 大单净流入占比（超大单+大单）
        mf_df['lg_elg_buy'] = mf_df['buy_lg_amount'] + mf_df['buy_elg_amount']
        mf_df['lg_elg_sell'] = mf_df['sell_lg_amount'] + mf_df['sell_elg_amount']
        mf_df['big_net_inflow_ratio'] = (mf_df['lg_elg_buy'] - mf_df['lg_elg_sell']) / (mf_df['lg_elg_buy'] + mf_df['lg_elg_sell'] + 1e-8)
        
        # 散户资金流向（小单）
        mf_df['sm_net_inflow'] = mf_df['buy_sm_amount'] - mf_df['sell_sm_amount']
        mf_df['sm_net_ratio'] = mf_df['sm_net_inflow'] / (mf_df['buy_sm_amount'] + mf_df['sell_sm_amount'] + 1e-8)
        
        # 主力资金与散户背离（大单买、小单卖 = 聪明钱流入）
        mf_df['smart_dumb_spread'] = mf_df['big_net_inflow_ratio'] - mf_df['sm_net_ratio']
        
        # 5日累计
        mf_df['big_net_5d'] = mf_df.groupby('ts_code')['big_net_inflow_ratio'].rolling(5, min_periods=1).mean().values
        mf_df['smart_dumb_5d'] = mf_df.groupby('ts_code')['smart_dumb_spread'].rolling(5, min_periods=1).mean().values
        
        return mf_df
    
    @staticmethod
    def calculate_chips_factors(cyq_df: pd.DataFrame) -> pd.DataFrame:
        """
        筹码集中度因子
        """
        if cyq_df.empty:
            return cyq_df
        
        cyq_df = cyq_df.sort_values(['ts_code', 'trade_date'])
        
        # 筹码集中度：90%成本区间 / 50%成本区间（越小越集中）
        cyq_df['chip_90_10'] = cyq_df['cost_95pct'] - cyq_df['cost_5pct']
        cyq_df['chip_70_30'] = cyq_df['cost_85pct'] - cyq_df['cost_15pct']
        cyq_df['chip_concentration'] = cyq_df['chip_90_10'] / (cyq_df['cost_50pct'] + 1e-8)
        
        # 获利盘比例变化（筹码集中+低位=吸筹信号）
        cyq_df['winner_change'] = cyq_df.groupby('ts_code')['winner_rate'].diff()
        
        return cyq_df
    
    @staticmethod
    def calculate_idiosyncratic_volatility(price_df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
        """
        特质波动率（Idiosyncratic Volatility）
        个股收益对市场回归后的残差波动率
        使用滚动窗口高效计算
        """
        price_df = price_df.sort_values(['ts_code', 'trade_date'])
        price_df['daily_ret'] = price_df.groupby('ts_code')['close'].pct_change()
        
        # 计算市场日收益（所有股票等权平均）
        market_ret = price_df.groupby('trade_date')['daily_ret'].mean().rename('market_ret')
        price_df = price_df.merge(market_ret, on='trade_date', how='left')
        
        def calc_group_ivol(group):
            group = group.sort_values('trade_date')
            # 滚动回归: beta = cov(y, x) / var(x)
            rolling_cov = group['daily_ret'].rolling(window, min_periods=window//2).cov(group['market_ret'])
            rolling_var = group['market_ret'].rolling(window, min_periods=window//2).var()
            beta = rolling_cov / (rolling_var + 1e-12)
            
            # alpha = mean(y) - beta * mean(x)
            alpha = group['daily_ret'].rolling(window, min_periods=window//2).mean() - beta * group['market_ret'].rolling(window, min_periods=window//2).mean()
            
            # 残差
            residual = group['daily_ret'] - alpha - beta * group['market_ret']
            
            # 残差滚动标准差（特质波动率）
            ivol = residual.rolling(window, min_periods=window//2).std()
            return ivol
        
        price_df['ivol'] = price_df.groupby('ts_code', group_keys=False).apply(calc_group_ivol)
        
        return price_df
    
    @classmethod
    def get_all_behavioral_factors(cls, price_df: pd.DataFrame, mf_df: pd.DataFrame = None, cyq_df: pd.DataFrame = None) -> pd.DataFrame:
        """计算所有行为金融因子"""
        price_df = cls.calculate_reversal(price_df.copy())
        price_df = cls.calculate_overnight_intraday(price_df)
        price_df = cls.calculate_idiosyncratic_volatility(price_df)
        
        # 合并资金流
        if mf_df is not None and not mf_df.empty:
            mf_df = cls.calculate_moneyflow_factors(mf_df.copy())
            mf_latest = mf_df.groupby('ts_code').last().reset_index()
            price_df = price_df.merge(
                mf_latest[['ts_code', 'big_net_inflow_ratio', 'smart_dumb_spread', 'big_net_5d', 'smart_dumb_5d']],
                on='ts_code', how='left'
            )
        
        # 合并筹码
        if cyq_df is not None and not cyq_df.empty:
            cyq_df = cls.calculate_chips_factors(cyq_df.copy())
            cyq_latest = cyq_df.groupby('ts_code').last().reset_index()
            price_df = price_df.merge(
                cyq_latest[['ts_code', 'chip_concentration', 'winner_rate', 'winner_change']],
                on='ts_code', how='left'
            )
        
        return price_df


# ==================== 三、另类数据与事件驱动因子 ====================

class AlternativeFactors:
    """另类数据因子：舆情、热度、产业链"""
    
    @staticmethod
    def calculate_news_sentiment(news_df: pd.DataFrame) -> pd.DataFrame:
        """
        新闻舆情因子
        """
        if news_df.empty:
            return news_df
        
        # 同花顺新闻：new_gs(利好), new_bs(利空), new_gi(个股)
        # 计算净情绪 = 利好 - 利空
        news_df['sentiment_net'] = news_df['new_gs'] - news_df['new_bs']
        news_df['sentiment_total'] = news_df['new_gs'] + news_df['new_bs']
        news_df['sentiment_ratio'] = news_df['sentiment_net'] / (news_df['sentiment_total'] + 1e-8)
        
        return news_df
    
    @staticmethod
    def calculate_hot_rank(rank_df: pd.DataFrame) -> pd.DataFrame:
        """
        热度排名因子（反转用：热度高可能意味着短期过热）
        """
        if rank_df.empty:
            return rank_df
        
        # 热度取负（热度越高，因子值越低 = 反转信号）
        rank_df['hot_inv'] = -rank_df['hot']
        
        return rank_df
    
    @classmethod
    def get_all_alternative_factors(cls, news_df: pd.DataFrame = None, rank_df: pd.DataFrame = None) -> pd.DataFrame:
        """计算所有另类数据因子"""
        result = pd.DataFrame()
        
        if news_df is not None and not news_df.empty:
            news_df = cls.calculate_news_sentiment(news_df.copy())
            result = news_df.groupby('ts_code').agg({
                'sentiment_net': 'sum',
                'sentiment_total': 'sum',
                'sentiment_ratio': 'mean',
                'new_gs': 'sum',
                'new_bs': 'sum'
            }).reset_index()
        
        if rank_df is not None and not rank_df.empty:
            rank_df = cls.calculate_hot_rank(rank_df.copy())
            if result.empty:
                result = rank_df[['ts_code', 'hot', 'hot_inv']]
            else:
                result = result.merge(rank_df[['ts_code', 'hot', 'hot_inv']], on='ts_code', how='outer')
        
        return result


# ==================== 因子合成与组合 ====================

class FactorCombiner:
    """因子组合方法论：非线性合成、动态择时、分域建模"""
    
    @staticmethod
    def preprocess_factors(df: pd.DataFrame, factor_cols: List[str], 
                          industry_col: str = 'industry',
                          cap_col: str = None) -> pd.DataFrame:
        """
        严格的去极值-中性化-标准化
        """
        df = df.copy()
        
        for col in factor_cols:
            if col not in df.columns:
                continue
            
            # 1. 去极值
            df[f'{col}_w'] = winsorize_series(df[col])
            
            # 2. 行业中性化
            if industry_col in df.columns:
                df[f'{col}_ni'] = neutralize_industry(df, f'{col}_w', industry_col)
            else:
                df[f'{col}_ni'] = df[f'{col}_w']
            
            # 3. 市值中性化
            if cap_col and cap_col in df.columns:
                df[f'{col}_nm'] = neutralize_market_cap(df, f'{col}_ni', cap_col)
            else:
                df[f'{col}_nm'] = df[f'{col}_ni']
            
            # 4. 标准化
            df[f'{col}_std'] = standardize_series(df[f'{col}_nm'])
        
        return df
    
    @staticmethod
    def linear_combination(df: pd.DataFrame, factor_cols: List[str], weights: Optional[Dict[str, float]] = None) -> pd.Series:
        """
        线性加权合成（基准方法）
        """
        std_cols = [f'{c}_std' for c in factor_cols if f'{c}_std' in df.columns]
        
        if not std_cols:
            return pd.Series(np.nan, index=df.index)
        
        if weights is None:
            weights = {c: 1.0/len(std_cols) for c in std_cols}
        
        score = pd.Series(0.0, index=df.index)
        for col in std_cols:
            w = weights.get(col, 1.0/len(std_cols))
            score += df[col].fillna(0) * w
        
        return score
    
    @staticmethod
    def tree_model_combination(df: pd.DataFrame, factor_cols: List[str], 
                               target_col: str = 'future_ret',
                               model_type: str = 'xgb') -> pd.Series:
        """
        非线性合成：XGBoost / LightGBM
        """
        from sklearn.ensemble import GradientBoostingRegressor
        
        std_cols = [f'{c}_std' for c in factor_cols if f'{c}_std' in df.columns]
        
        train_df = df[std_cols + [target_col]].dropna()
        if len(train_df) < 100:
            return FactorCombiner.linear_combination(df, factor_cols)
        
        X = train_df[std_cols]
        y = train_df[target_col]
        
        if model_type == 'xgb':
            try:
                import xgboost as xgb
                model = xgb.XGBRegressor(n_estimators=100, max_depth=4, learning_rate=0.05, random_state=42)
            except ImportError:
                model = GradientBoostingRegressor(n_estimators=100, max_depth=4, random_state=42)
        elif model_type == 'lgb':
            try:
                import lightgbm as lgb
                model = lgb.LGBMRegressor(n_estimators=100, max_depth=4, learning_rate=0.05, random_state=42)
            except ImportError:
                model = GradientBoostingRegressor(n_estimators=100, max_depth=4, random_state=42)
        else:
            model = GradientBoostingRegressor(n_estimators=100, max_depth=4, random_state=42)
        
        model.fit(X, y)
        
        # 预测
        pred_df = df[std_cols].fillna(0)
        score = pd.Series(model.predict(pred_df), index=df.index)
        
        return score


import os

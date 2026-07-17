"""
Study 007: 截面选股策略（Cross-Sectional Stock Selection）
配置参数
"""

# ==================== 数据配置 ====================
DATA_CONFIG = {
    'price_dir': r'D:\iquant_data\data_v2\data_day1',
    'fundamental_cache': r'D:\iquant_data\data_v2\fundamental1\fina_indicator_cache.parquet',
    'industry_map': r'D:\iquant_data\data_v2\industry1\industry.parquet',
    'moneyflow_dir': r'D:\iquant_data\data_v2\moneyflow1',
    'cyq_dir': r'D:\iquant_data\data_v2\cyq1',
    'ths_rank_dir': r'D:\iquant_data\data_v2\ths_rank1',
    'ths_news_dir': r'D:\iquant_data\data_v2\ths_news1',
}

# ==================== 时间配置 ====================
TIME_CONFIG = {
    'train_start': '20200101',  # 训练开始（因子计算需要历史数据）
    'train_end': '20221231',    # 训练结束
    'test_start': '20230101',   # 测试开始
    'test_end': '20241231',     # 测试结束
}

# ==================== 因子配置 ====================
FACTOR_CONFIG = {
    # 基本面因子
    'fundamental_factors': [
        'roe', 'roe_dt', 'roe_stability',
        'or_yoy', 'or_yoy_accel',
        'netprofit_yoy', 'netprofit_yoy_accel',
        'netprofit_margin', 'grossprofit_margin',
        'gpm_trend', 'npm_trend',
        'debt_to_assets', 'debt_improve',
        'quick_ratio', 'quick_change',
    ],
    
    # 行为金融因子
    'behavioral_factors': [
        'ret_1m', 'ret_3m', 'ret_1m_accel',
        'overnight_ret_5d', 'intraday_ret_5d', 'overnight_intraday_spread',
        'big_net_inflow_ratio', 'smart_dumb_spread', 'big_net_5d', 'smart_dumb_5d',
        'chip_concentration', 'winner_rate', 'winner_change',
        'ivol',
    ],
    
    # 另类数据因子
    'alternative_factors': [
        'sentiment_net', 'sentiment_ratio', 'new_gs', 'new_bs',
        'hot', 'hot_inv',
    ],
    
    # 技术因子（从现有框架复用）
    'technical_factors': [
        'mom_5d', 'mom_10d', 'mom_20d', 'mom_60d',
        'vol_5d', 'vol_10d', 'vol_20d',
    ],
}

# ==================== 回测配置 ====================
BACKTEST_CONFIG = {
    'rebalance_freq': 'monthly',  # monthly / weekly
    'top_n': 50,                  # 选股数量
    'weight_method': 'equal',     # equal / score_weighted / cap_weighted
    'cost_rate': 0.003,           # 单边交易成本
    'long_short': False,
    'industry_neutral': True,
    'max_industry_pct': 0.30,
}

# ==================== 模型配置 ====================
MODEL_CONFIG = {
    'model_type': 'xgb',  # xgb / lgb / linear
    'use_tree_model': True,
    'n_estimators': 100,
    'max_depth': 4,
    'learning_rate': 0.05,
}

# ==================== 分域配置 ====================
DOMAIN_CONFIG = {
    'enable_domain_model': True,
    'domain_by': 'market_cap',  # market_cap / industry
    'cap_bins': [0, 10, 50, 100, 500, float('inf')],  # 市值分档（亿）
    'cap_labels': ['微型', '小盘', '中盘', '大盘', '超大盘'],
}

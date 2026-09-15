# -*- coding: utf-8 -*-
"""
Multi-Asset 4h Dataset Builder & Feature Engineering (2020-2026 Full Horizon)
构建多币种4小时数据集与时空特征工程模块 (BTC, ETH, SOL, BNB + Macro + On-Chain TVL/Stablecoins + News Sentiment)
支持严格三段式切分：
  - 训练集 (Train): 2020-08-11 至 2023-12-31 (3.4年，样本内拟合)
  - 验证集 (Val / Tuning): 2024-01-01 至 2025-12-31 (2.0年，标的与频次调优)
  - 终极封存盲测集 (Blind Test): 2026-01-01 至 2026-09-13 (8.5个月，最终实证检验)
"""
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

TOKENS = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'BNBUSDT']


def compute_token_features(
    df: pd.DataFrame, 
    df_btc: pd.DataFrame, 
    df_eth: pd.DataFrame, 
    df_sol: pd.DataFrame, 
    df_macro: pd.DataFrame,
    df_onchain: pd.DataFrame,
    df_basis: pd.DataFrame = None,
    df_funding: pd.DataFrame = None,
    df_okx: pd.DataFrame = None,
    token_name: str = 'ETHUSDT'
) -> pd.DataFrame:
    """计算单个代币的时空动量、微观结构、宏观美股、以太坊链上资金流、新闻情绪与衍生品特征 (共33维)"""
    c = df['close']
    h = df['high']
    l = df['low']
    v = df['volume']
    tbv = df['taker_buy_volume'] if 'taker_buy_volume' in df.columns else v * 0.5

    feats = pd.DataFrame(index=df.index)

    # 1. 多尺度动量对数收益率
    feats['ret_1'] = np.log(c / c.shift(1))
    feats['ret_3'] = np.log(c / c.shift(3))
    feats['ret_6'] = np.log(c / c.shift(6))
    feats['ret_18'] = np.log(c / c.shift(18))
    feats['ret_42'] = np.log(c / c.shift(42))

    # 2. 均线偏离度 (Trend Bias)
    ema20 = c.ewm(span=20, adjust=False).mean()
    ema60 = c.ewm(span=60, adjust=False).mean()
    feats['bias_ema20'] = (c - ema20) / (ema20 + 1e-8)
    feats['bias_ema60'] = (c - ema60) / (ema60 + 1e-8)

    # 3. 波动率与布林带
    tr1 = h - l
    tr2 = (h - c.shift(1)).abs()
    tr3 = (l - c.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr14 = tr.rolling(14).mean()
    feats['atr_ratio'] = atr14 / (c + 1e-8)
    feats['hl_range'] = (h - l) / (c + 1e-8)

    sma20 = c.rolling(20).mean()
    rstd20 = c.rolling(20).std()
    bb_upper = sma20 + 2 * rstd20
    bb_lower = sma20 - 2 * rstd20
    feats['bb_pct_b'] = (c - bb_lower) / (bb_upper - bb_lower + 1e-8)
    feats['bb_bandwidth'] = (bb_upper - bb_lower) / (sma20 + 1e-8)

    # 4. 微观结构与主动订单流失衡 (Order Flow Imbalance)
    feats['taker_buy_ratio'] = tbv / (v + 1e-8)
    vol_ma18 = v.rolling(18).mean()
    feats['volume_ratio'] = v / (vol_ma18 + 1e-8)

    # 5. 跨币种相对强弱特征 (Cross-Asset Relational Features)
    btc_c = df_btc['close']
    eth_c = df_eth['close']
    sol_c = df_sol['close']
    
    btc_ret1 = np.log(btc_c / btc_c.shift(1))
    eth_ret1 = np.log(eth_c / eth_c.shift(1))
    
    feats['rel_ret_btc'] = feats['ret_1'] - btc_ret1
    feats['rel_ret_eth'] = feats['ret_1'] - eth_ret1

    eth_btc_ratio = eth_c / btc_c
    sol_eth_ratio = sol_c / eth_c
    feats['eth_btc_zscore'] = (eth_btc_ratio - eth_btc_ratio.rolling(60).mean()) / (eth_btc_ratio.rolling(60).std() + 1e-8)
    feats['sol_eth_zscore'] = (sol_eth_ratio - sol_eth_ratio.rolling(60).mean()) / (sol_eth_ratio.rolling(60).std() + 1e-8)

    # 6. 美股宏观参数对齐 (严格因果前向填充，滞后1天，无未来函数)
    macro_daily_lag = df_macro.shift(1)
    macro_aligned = macro_daily_lag.reindex(df.index.normalize(), method='ffill')
    feats['ndx_ret_1d'] = macro_aligned['ndx_ret_1d'].values
    feats['spx_ret_1d'] = macro_aligned['spx_ret_1d'].values
    feats['ndx_ma20_bias'] = macro_aligned['ndx_ma20_bias'].values

    # 7. 交易时段与周期编码 (US Session & Hour Encoding)
    hours = df.index.hour
    feats['is_us_session'] = ((hours >= 12) & (hours <= 20)).astype(float)
    feats['hour_sin'] = np.sin(2 * np.pi * hours / 24.0)
    feats['hour_cos'] = np.cos(2 * np.pi * hours / 24.0)

    # 8. 以太坊链上资金流与加密新闻情绪特征 (严格滞后 1 天以保证因果性)
    onchain_lag = df_onchain.shift(1)
    onchain_aligned = onchain_lag.reindex(df.index.normalize(), method='ffill')
    
    feats['tvl_flow_1d'] = onchain_aligned['tvl_flow_1d'].values
    feats['tvl_flow_7d'] = onchain_aligned['tvl_flow_7d'].values
    feats['stb_flow_1d'] = onchain_aligned['stb_flow_1d'].values
    feats['stb_flow_7d'] = onchain_aligned['stb_flow_7d'].values
    feats['fng_score'] = onchain_aligned['fng_score'].values / 100.0
    feats['fng_bias'] = onchain_aligned['fng_bias'].values

    # 9. 衍生品微观结构特征 (现货-永续基差、资金费率与OKX跨交易所价差)
    if df_basis is not None and token_name in df_basis.columns:
        basis_series = df_basis[token_name].reindex(df.index).ffill().fillna(0.0)
        basis_mean = basis_series.rolling(72).mean()
        basis_std = basis_series.rolling(72).std() + 1e-8
        feats['basis_zscore_72'] = ((basis_series - basis_mean) / basis_std).fillna(0.0)
        feats['basis_mom_6'] = (basis_series - basis_series.shift(6)).fillna(0.0)
    else:
        feats['basis_zscore_72'] = 0.0
        feats['basis_mom_6'] = 0.0

    if df_funding is not None and token_name in df_funding.columns:
        funding_lag = df_funding[token_name].shift(1)
        feats['funding_rate_lag'] = funding_lag.reindex(df.index, method='ffill').fillna(0.0)
    else:
        feats['funding_rate_lag'] = 0.0

    if df_okx is not None and token_name in df_okx.columns:
        okx_c = df_okx[token_name].reindex(df.index).ffill()
        spread = (okx_c - df['close']) / (df['close'] + 1e-8)
        spread_mean = spread.rolling(18).mean()
        spread_std = spread.rolling(18).std() + 1e-8
        feats['okx_binance_spread_z18'] = ((spread - spread_mean) / spread_std).fillna(0.0)
    else:
        feats['okx_binance_spread_z18'] = 0.0

    # 10. 缠论分形拓扑几何特征 (8维)
    try:
        from crypto_quant.chan_features import compute_chan_features
    except ModuleNotFoundError:
        try:
            from chan_features import compute_chan_features
        except Exception:
            compute_chan_features = None

    if compute_chan_features is not None:
        chan_feats = compute_chan_features(df)
        for col in chan_feats.columns:
            feats[col] = chan_feats[col].values

    # 11. 预测目标 (Micro 4h/8h + Multi-Horizon Waves 3d/6d/12d)
    feats['target_ret_4h'] = np.log(c.shift(-1) / c)
    feats['target_ret_8h'] = np.log(c.shift(-2) / c)
    feats['target_wave_3d'] = np.log(c.shift(-18) / c)   # 3-day swing return
    feats['target_wave_6d'] = np.log(c.shift(-36) / c)   # 6-day wave return
    feats['target_wave_12d'] = np.log(c.shift(-72) / c)  # 12-day macro wave return

    return feats


class CryptoMultiAssetDataset(Dataset):
    """PyTorch Dataset for Spatio-Temporal Crypto Transformer"""
    def __init__(self, X_tensors, y_4h, y_8h, timestamps, close_prices, open_prices):
        self.X = torch.tensor(X_tensors, dtype=torch.float32)  # (N, K, L, D)
        self.y_4h = torch.tensor(y_4h, dtype=torch.float32)    # (N, K)
        self.y_8h = torch.tensor(y_8h, dtype=torch.float32)    # (N, K)
        self.timestamps = timestamps
        self.close_prices = close_prices  # dict of token -> close array
        self.open_prices = open_prices    # dict of token -> open array

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return {
            'X': self.X[idx],
            'y_4h': self.y_4h[idx],
            'y_8h': self.y_8h[idx],
            'idx': idx
        }


def prepare_crypto_datasets(lookback_len: int = 12, train_end='2023-12-31', val_end='2025-12-31'):
    """
    严格三段式切分：
    - Train Set: 2020-08-11 -> 2023-12-31
    - Val Set (2024-2025 探索集): 2024-01-01 -> 2025-12-31
    - Blind Test Set (2026 终极封存测试集): 2026-01-01 -> 2026-09-13
    """
    print("Loading 2020-2026 4h crypto data from cache...")
    raw_dfs = {}
    for t in TOKENS:
        p = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', f'{t}_4h_2020_2026.parquet')
        if not os.path.exists(p):
            p = f'data/{t}_4h_2020_2026.parquet' 
        if not os.path.exists(p):
            p = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', f'{t}_4h_2021_2026.parquet')
        raw_dfs[t] = pd.read_parquet(p)

    macro_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'us_stock_macro.parquet')
    if not os.path.exists(macro_path):
        raise FileNotFoundError(f"Macro stock cache {macro_path} not found.")
    df_macro = pd.read_parquet(macro_path)

    onchain_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'eth_onchain_sentiment_daily.parquet')
    if not os.path.exists(onchain_path):
        raise FileNotFoundError(f"On-chain sentiment cache {onchain_path} not found.")
    df_onchain = pd.read_parquet(onchain_path)

    # 加载衍生品特征 (基差、资金费率与OKX价差)
    basis_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'binance_basis_4h.parquet')
    df_basis = pd.read_parquet(basis_path) if os.path.exists(basis_path) else None
    if df_basis is not None:
        df_basis.index = df_basis.index.tz_localize(None)

    funding_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'binance_funding_8h.parquet')
    df_funding = pd.read_parquet(funding_path) if os.path.exists(funding_path) else None
    if df_funding is not None:
        df_funding.index = df_funding.index.tz_localize(None)

    okx_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'okx_swap_candles_4h.parquet')
    df_okx = pd.read_parquet(okx_path) if os.path.exists(okx_path) else None
    if df_okx is not None:
        df_okx.index = df_okx.index.tz_localize(None)

    # 计算各币种特征
    print("Engineering 33 spatio-temporal, on-chain, sentiment & derivatives features for all assets...")
    feat_dfs = {}
    for t in TOKENS:
        feat_dfs[t] = compute_token_features(
            raw_dfs[t], 
            raw_dfs['BTCUSDT'], 
            raw_dfs['ETHUSDT'], 
            raw_dfs['SOLUSDT'], 
            df_macro,
            df_onchain,
            df_basis,
            df_funding,
            df_okx,
            token_name=t
        )

    # 确定有效索引：联合检查全部4大资产特征与目标有效性 (解决审查问题 11: 杜绝单一标的掩码导致的 NaN 泄露)
    valid_mask = pd.Series(True, index=feat_dfs['BTCUSDT'].index)
    for t in TOKENS:
        valid_mask &= ~feat_dfs[t]['ret_42'].isna()
        valid_mask &= ~feat_dfs[t]['target_ret_8h'].isna()
        valid_mask &= ~feat_dfs[t]['target_ret_4h'].isna()
        valid_mask &= ~feat_dfs[t]['ndx_ret_1d'].isna()
        valid_mask &= ~feat_dfs[t]['tvl_flow_7d'].isna()
    common_idx = feat_dfs['BTCUSDT'][valid_mask].index

    feature_cols = [c for c in feat_dfs['ETHUSDT'].columns if not c.startswith('target_')]
    num_features = len(feature_cols)
    print(f"Features count: {num_features}")

    T = len(common_idx)
    K = len(TOKENS)
    feature_matrix = np.zeros((T, K, num_features), dtype=np.float32)
    target_4h_matrix = np.zeros((T, K), dtype=np.float32)
    target_8h_matrix = np.zeros((T, K), dtype=np.float32)

    for k, t in enumerate(TOKENS):
        fdf = feat_dfs[t].loc[common_idx]
        feature_matrix[:, k, :] = fdf[feature_cols].fillna(0.0).values
        target_4h_matrix[:, k] = fdf['target_ret_4h'].values
        target_8h_matrix[:, k] = fdf['target_ret_8h'].values

    # 标准化：严格只用 Train Set (2020-08 到 2023-12) 计算均值方差
    train_core_indices = np.where(common_idx <= train_end)[0]
    print(f"Normalizing features strictly using Train period ({common_idx[train_core_indices[0]].date()} to {common_idx[train_core_indices[-1]].date()})...")
    scaler_mean = np.nanmean(feature_matrix[train_core_indices], axis=(0, 1), keepdims=True)
    scaler_std = np.nanstd(feature_matrix[train_core_indices], axis=(0, 1), keepdims=True)
    scaler_std[scaler_std < 1e-6] = 1.0

    norm_features = (feature_matrix - scaler_mean) / scaler_std
    norm_features = np.clip(norm_features, -5.0, 5.0)

    # 构造滚动窗口序列
    N_samples = T - lookback_len + 1
    X_all = np.zeros((N_samples, K, lookback_len, num_features), dtype=np.float32)
    y_4h_all = np.zeros((N_samples, K), dtype=np.float32)
    y_8h_all = np.zeros((N_samples, K), dtype=np.float32)
    timestamps = common_idx[lookback_len - 1:]

    for i in range(N_samples):
        X_all[i] = norm_features[i : i + lookback_len].transpose(1, 0, 2)
        y_4h_all[i] = target_4h_matrix[i + lookback_len - 1]
        y_8h_all[i] = target_8h_matrix[i + lookback_len - 1]

    # 保存各币种的价格行情
    close_prices = {t: raw_dfs[t].loc[timestamps, 'close'].values for t in TOKENS}
    open_prices = {t: raw_dfs[t].loc[timestamps, 'open'].values for t in TOKENS}

    # 严格三段式样本切分
    train_mask = timestamps <= train_end
    val_mask = (timestamps > train_end) & (timestamps <= val_end)
    blind_test_mask = timestamps > val_end

    print(f"Total sequences (L={lookback_len}): {len(timestamps)}")
    print(f"1. Train Set (2020-2023):      {train_mask.sum()} bars ({timestamps[train_mask][0]} -> {timestamps[train_mask][-1]})")
    print(f"2. Val/Tuning Set (2024-2025): {val_mask.sum()} bars ({timestamps[val_mask][0]} -> {timestamps[val_mask][-1]})")
    print(f"3. Blind Test Set (2026 YTD):  {blind_test_mask.sum()} bars ({timestamps[blind_test_mask][0]} -> {timestamps[blind_test_mask][-1]})")

    train_ds = CryptoMultiAssetDataset(
        X_all[train_mask], y_4h_all[train_mask], y_8h_all[train_mask],
        timestamps[train_mask],
        {t: close_prices[t][train_mask] for t in TOKENS},
        {t: open_prices[t][train_mask] for t in TOKENS}
    )
    val_ds = CryptoMultiAssetDataset(
        X_all[val_mask], y_4h_all[val_mask], y_8h_all[val_mask],
        timestamps[val_mask],
        {t: close_prices[t][val_mask] for t in TOKENS},
        {t: open_prices[t][val_mask] for t in TOKENS}
    )
    blind_test_ds = CryptoMultiAssetDataset(
        X_all[blind_test_mask], y_4h_all[blind_test_mask], y_8h_all[blind_test_mask],
        timestamps[blind_test_mask],
        {t: close_prices[t][blind_test_mask] for t in TOKENS},
        {t: open_prices[t][blind_test_mask] for t in TOKENS}
    )

    metadata = {
        'tokens': TOKENS,
        'feature_cols': feature_cols,
        'num_features': num_features,
        'lookback_len': lookback_len,
        'scaler_mean': scaler_mean,
        'scaler_std': scaler_std,
        'timestamps': timestamps,
        'val_timestamps': timestamps[val_mask],
        'blind_test_timestamps': timestamps[blind_test_mask]
    }

    return train_ds, val_ds, blind_test_ds, metadata


class ChanWaveDataset(Dataset):
    """PyTorch Dataset for Spatio-Temporal Chan-Lun Wave Transformer (Phase 20)"""
    def __init__(self, X_tensors, y_3d, y_6d, y_12d, y_exp, timestamps, close_prices, open_prices):
        self.X = torch.tensor(X_tensors, dtype=torch.float32)        # (N, K, L=18, D=41)
        self.y_3d = torch.tensor(y_3d, dtype=torch.float32)          # (N, K)
        self.y_6d = torch.tensor(y_6d, dtype=torch.float32)          # (N, K)
        self.y_12d = torch.tensor(y_12d, dtype=torch.float32)        # (N, K)
        self.y_exp = torch.tensor(y_exp, dtype=torch.float32)        # (N, K)
        self.timestamps = timestamps
        self.close_prices = close_prices
        self.open_prices = open_prices

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return {
            'X': self.X[idx],
            'y_3d': self.y_3d[idx],
            'y_6d': self.y_6d[idx],
            'y_12d': self.y_12d[idx],
            'y_exp': self.y_exp[idx],
            'idx': idx
        }


def prepare_chan_wave_datasets(lookback_len: int = 18, train_end='2023-12-31', val_end='2025-12-31'):
    """
    Spatio-Temporal Chan-Lun Wave Dataset Pipeline (Phase 20):
    - Features: 41 dimensions (33 base features + 8 Chan-Lun topological features)
    - Targets: 3-day (18-bar), 6-day (36-bar), 12-day (72-bar) wave returns + true expansion label
    - Strict 3-way split: 2020-2023 Train, 2024-2025 Val, 2026 Locked Stress Set
    """
    print("Loading 2020-2026 4h crypto data for Chan-Lun Wave Transformer...")
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(root_dir, 'data')

    raw_dfs = {}
    for t in TOKENS:
        p = os.path.join(data_dir, f'{t}_4h_2020_2026.parquet')
        if not os.path.exists(p):
            p = os.path.join(data_dir, f'{t}_4h_2021_2026.parquet')
        raw_dfs[t] = pd.read_parquet(p)

    df_macro = pd.read_parquet(os.path.join(data_dir, 'us_stock_macro.parquet'))
    df_onchain = pd.read_parquet(os.path.join(data_dir, 'eth_onchain_sentiment_daily.parquet'))

    basis_path = os.path.join(data_dir, 'binance_basis_4h.parquet')
    df_basis = pd.read_parquet(basis_path) if os.path.exists(basis_path) else None
    if df_basis is not None and df_basis.index.tz is not None:
        df_basis.index = df_basis.index.tz_localize(None)

    funding_path = os.path.join(data_dir, 'binance_funding_8h.parquet')
    df_funding = pd.read_parquet(funding_path) if os.path.exists(funding_path) else None
    if df_funding is not None and df_funding.index.tz is not None:
        df_funding.index = df_funding.index.tz_localize(None)

    okx_path = os.path.join(data_dir, 'okx_swap_candles_4h.parquet')
    df_okx = pd.read_parquet(okx_path) if os.path.exists(okx_path) else None
    if df_okx is not None and df_okx.index.tz is not None:
        df_okx.index = df_okx.index.tz_localize(None)

    print("Engineering 41 spatio-temporal & Chan-Lun topological features...")
    feat_dfs = {}
    for t in TOKENS:
        feat_dfs[t] = compute_token_features(
            raw_dfs[t], raw_dfs['BTCUSDT'], raw_dfs['ETHUSDT'], raw_dfs['SOLUSDT'],
            df_macro, df_onchain, df_basis, df_funding, df_okx, token_name=t
        )

    # Valid mask across all 4 tokens
    valid_mask = pd.Series(True, index=feat_dfs['BTCUSDT'].index)
    for t in TOKENS:
        valid_mask &= ~feat_dfs[t]['ret_42'].isna()
        valid_mask &= ~feat_dfs[t]['target_wave_12d'].isna()
        valid_mask &= ~feat_dfs[t]['ndx_ret_1d'].isna()
        valid_mask &= ~feat_dfs[t]['tvl_flow_7d'].isna()
        valid_mask &= ~feat_dfs[t]['chan_hub_dist'].isna()

    common_idx = feat_dfs['BTCUSDT'][valid_mask].index
    feature_cols = [c for c in feat_dfs['ETHUSDT'].columns if not c.startswith('target_')]
    num_features = len(feature_cols)
    print(f"Chan-Lun Wave features count: {num_features} (Columns: {feature_cols[-8:]})")

    T = len(common_idx)
    K = len(TOKENS)
    feature_matrix = np.zeros((T, K, num_features), dtype=np.float32)
    target_3d_matrix = np.zeros((T, K), dtype=np.float32)
    target_6d_matrix = np.zeros((T, K), dtype=np.float32)
    target_12d_matrix = np.zeros((T, K), dtype=np.float32)
    target_exp_matrix = np.zeros((T, K), dtype=np.float32)

    for k, t in enumerate(TOKENS):
        fdf = feat_dfs[t].loc[common_idx]
        feature_matrix[:, k, :] = fdf[feature_cols].fillna(0.0).values
        target_3d_matrix[:, k] = fdf['target_wave_3d'].values
        target_6d_matrix[:, k] = fdf['target_wave_6d'].values
        target_12d_matrix[:, k] = fdf['target_wave_12d'].values
        target_exp_matrix[:, k] = (fdf['target_wave_12d'] > 0.03).astype(float).values

    # Train scaler strictly on Train period (<= train_end)
    train_core_indices = np.where(common_idx <= train_end)[0]
    scaler_mean = np.nanmean(feature_matrix[train_core_indices], axis=(0, 1), keepdims=True)
    scaler_std = np.nanstd(feature_matrix[train_core_indices], axis=(0, 1), keepdims=True)
    scaler_std[scaler_std < 1e-6] = 1.0

    norm_features = np.clip((feature_matrix - scaler_mean) / scaler_std, -5.0, 5.0)

    # Sequence rolling
    N_samples = T - lookback_len + 1
    X_all = np.zeros((N_samples, K, lookback_len, num_features), dtype=np.float32)
    y_3d_all = np.zeros((N_samples, K), dtype=np.float32)
    y_6d_all = np.zeros((N_samples, K), dtype=np.float32)
    y_12d_all = np.zeros((N_samples, K), dtype=np.float32)
    y_exp_all = np.zeros((N_samples, K), dtype=np.float32)
    timestamps = common_idx[lookback_len - 1:]

    for i in range(N_samples):
        X_all[i] = norm_features[i : i + lookback_len].transpose(1, 0, 2)
        y_3d_all[i] = target_3d_matrix[i + lookback_len - 1]
        y_6d_all[i] = target_6d_matrix[i + lookback_len - 1]
        y_12d_all[i] = target_12d_matrix[i + lookback_len - 1]
        y_exp_all[i] = target_exp_matrix[i + lookback_len - 1]

    close_prices = {t: raw_dfs[t].loc[timestamps, 'close'].values for t in TOKENS}
    open_prices = {t: raw_dfs[t].loc[timestamps, 'open'].values for t in TOKENS}

    train_mask = timestamps <= train_end
    val_mask = (timestamps > train_end) & (timestamps <= val_end)
    blind_test_mask = timestamps > val_end

    print(f"Total Chan-Wave sequences (L={lookback_len}): {len(timestamps)}")
    print(f"1. Train Set (2020-2023):      {train_mask.sum()} bars")
    print(f"2. Val Set (2024-2025):        {val_mask.sum()} bars")
    print(f"3. Blind Test Set (2026 YTD):  {blind_test_mask.sum()} bars")

    train_ds = ChanWaveDataset(
        X_all[train_mask], y_3d_all[train_mask], y_6d_all[train_mask], y_12d_all[train_mask], y_exp_all[train_mask],
        timestamps[train_mask],
        {t: close_prices[t][train_mask] for t in TOKENS},
        {t: open_prices[t][train_mask] for t in TOKENS}
    )
    val_ds = ChanWaveDataset(
        X_all[val_mask], y_3d_all[val_mask], y_6d_all[val_mask], y_12d_all[val_mask], y_exp_all[val_mask],
        timestamps[val_mask],
        {t: close_prices[t][val_mask] for t in TOKENS},
        {t: open_prices[t][val_mask] for t in TOKENS}
    )
    blind_test_ds = ChanWaveDataset(
        X_all[blind_test_mask], y_3d_all[blind_test_mask], y_6d_all[blind_test_mask], y_12d_all[blind_test_mask], y_exp_all[blind_test_mask],
        timestamps[blind_test_mask],
        {t: close_prices[t][blind_test_mask] for t in TOKENS},
        {t: open_prices[t][blind_test_mask] for t in TOKENS}
    )

    metadata = {
        'tokens': TOKENS,
        'feature_cols': feature_cols,
        'num_features': num_features,
        'lookback_len': lookback_len,
        'scaler_mean': scaler_mean,
        'scaler_std': scaler_std,
        'timestamps': timestamps,
        'val_timestamps': timestamps[val_mask],
        'blind_test_timestamps': timestamps[blind_test_mask]
    }

    return train_ds, val_ds, blind_test_ds, metadata


if __name__ == '__main__':
    train_ds, val_ds, blind_test_ds, meta = prepare_chan_wave_datasets(lookback_len=18)
    print("Chan-Lun Wave Dataset Builder test passed successfully!")

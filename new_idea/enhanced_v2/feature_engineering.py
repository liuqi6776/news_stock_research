"""
Enhanced Feature Engineering v2
Three pillars:
  1. Chan Theory (缠论): 笔、线段、中枢、背驰
  2. Peter Lynch Fundamentals: PEG、营收增长、利润率
  3. Quant Factors: 动量、反转、波动率、流动性

All features computed using T-1 and earlier data only (no look-ahead bias)
"""
import os
import sys
import pandas as pd
import numpy as np
import tushare as ts
import time
from datetime import datetime
from tqdm import tqdm

DATA_DIR = r'D:\iquant_data\data_v2'
PRICE_DIR = os.path.join(DATA_DIR, 'data_day1')
OTHER_DIR = os.path.join(DATA_DIR, 'other_day1')
RANK_DIR = os.path.join(DATA_DIR, 'ths_rank1')
CHIP_DIR = os.path.join(DATA_DIR, 'cyq1')
FUND_DIR = os.path.join(DATA_DIR, 'fundamental1')

TUSHARE_TOKEN = "421ff94dd31be789aa7f95e61ad6fad5bcefa250a0b2c4d298224aa"
pro = ts.pro_api(TUSHARE_TOKEN)


# ============================================================
# 1. CHAN THEORY FEATURES (缠论特征)
# ============================================================

def compute_chan_features(price_hist, lookback=60):
    """
    Compute Chan Theory features from OHLC price history (optimized version).
    Simplified inclusion processing using vectorized operations.
    """
    results = []

    for ts_code, group in price_hist.groupby('ts_code'):
        g = group.sort_values('trade_date').tail(lookback).reset_index(drop=True)
        if len(g) < 10:
            continue

        h = g['high'].values
        l = g['low'].values
        c = g['close'].values
        n = len(g)

        # --- Simplified 包含关系处理 ---
        merged_h = [h[0]]
        merged_l = [l[0]]
        merged_dir = [1 if c[0] > g['open'].values[0] else -1]

        for i in range(1, n):
            cur_h, cur_l = h[i], l[i]
            prev_h, prev_l = merged_h[-1], merged_l[-1]

            if cur_h <= prev_h and cur_l >= prev_l:
                if merged_dir[-1] == 1:
                    merged_l[-1] = cur_l
                else:
                    merged_h[-1] = cur_h
            elif cur_h >= prev_h and cur_l <= prev_l:
                if merged_dir[-1] == 1:
                    merged_h[-1] = cur_h
                else:
                    merged_l[-1] = cur_l
            else:
                merged_h.append(cur_h)
                merged_l.append(cur_l)
                merged_dir.append(1 if cur_h > prev_h else -1)

        # --- 顶分型/底分型 ---
        tops, bottoms = [], []
        for j in range(1, len(merged_h) - 1):
            if merged_h[j] > merged_h[j - 1] and merged_h[j] > merged_h[j + 1]:
                tops.append(j)
            if merged_l[j] < merged_l[j - 1] and merged_l[j] < merged_l[j + 1]:
                bottoms.append(j)

        bi_count = len(tops) + len(bottoms)

        # --- 中枢 ---
        zhongshu_count = 0
        zhongshu_width = 0.0
        zg, zd = 0, 0
        if len(tops) >= 2 and len(bottoms) >= 2:
            recent_tops = tops[-3:] if len(tops) >= 3 else tops
            recent_bottoms = bottoms[-3:] if len(bottoms) >= 3 else bottoms
            top_highs = [merged_h[t] for t in recent_tops]
            bottom_lows = [merged_l[b] for b in recent_bottoms]
            zg = min(top_highs)
            zd = max(bottom_lows)
            if zg > zd:
                zhongshu_count = 1
                zhongshu_width = (zg - zd) / (zd + 1e-8)

        # --- MACD背驰 (vectorized) ---
        macd_divergence = 0
        if n >= 20:
            ema12 = _ema(c, 12)
            ema26 = _ema(c, 26)
            dif = ema12 - ema26
            dea = _ema(dif, 9)
            macd_hist = 2 * (dif - dea)

            if len(macd_hist) >= 10:
                recent_macd = np.mean(macd_hist[-5:])
                prev_macd = np.mean(macd_hist[-10:-5])
                recent_price = np.mean(c[-5:])
                prev_price = np.mean(c[-10:-5])

                if recent_price > prev_price and recent_macd < prev_macd:
                    macd_divergence = -1
                elif recent_price < prev_price and recent_macd > prev_macd:
                    macd_divergence = 1

        # --- 笔方向 ---
        current_bi_dir = 0
        if tops and bottoms:
            current_bi_dir = -1 if tops[-1] > bottoms[-1] else 1

        # --- 离开中枢距离 ---
        leave_zhongshu = 0.0
        if zhongshu_count > 0:
            leave_zhongshu = (c[-1] - (zg + zd) / 2) / ((zg + zd) / 2 + 1e-8)

        results.append({
            'ts_code': ts_code,
            'chan_bi_count': bi_count,
            'chan_zhongshu_count': zhongshu_count,
            'chan_zhongshu_width': zhongshu_width,
            'chan_macd_divergence': macd_divergence,
            'chan_bi_direction': current_bi_dir,
            'chan_leave_zhongshu': leave_zhongshu,
        })

    return pd.DataFrame(results)


def _ema(data, period):
    result = np.zeros_like(data, dtype=float)
    result[0] = data[0]
    alpha = 2.0 / (period + 1)
    for i in range(1, len(data)):
        result[i] = alpha * data[i] + (1 - alpha) * result[i - 1]
    return result


# ============================================================
# 2. PETER LYNCH FUNDAMENTAL FEATURES (彼得林奇基本面)
# ============================================================

def download_fundamental_data(ts_codes=None, save_dir=FUND_DIR):
    """Download and cache fundamental data from Tushare (batch mode)"""
    os.makedirs(save_dir, exist_ok=True)
    cache_file = os.path.join(save_dir, 'fina_indicator_cache.parquet')

    if os.path.exists(cache_file):
        cache = pd.read_parquet(cache_file)
        cache_date = os.path.getmtime(cache_file)
        days_old = (time.time() - cache_date) / 86400
        if days_old < 7:
            print(f"  基本面缓存有效 ({days_old:.1f}天前)，跳过下载")
            return cache

    print("  下载财务指标数据 (逐只下载热门+主板股票)...")
    all_fina = []
    try:
        rank_files = sorted([f for f in os.listdir(RANK_DIR) if f.endswith('.parquet')])
        if rank_files:
            latest_rank = pd.read_parquet(os.path.join(RANK_DIR, rank_files[-1]))
            hot_codes = latest_rank['ts_code'].tolist()
        else:
            stock_list = pro.stock_basic(exchange='', list_status='L', fields='ts_code')
            hot_codes = stock_list['ts_code'].tolist()

        main_codes = [c for c in hot_codes if c.startswith('60') or c.startswith('00')][:800]

        for i, code in enumerate(tqdm(main_codes, desc="  财务指标")):
            try:
                df = pro.fina_indicator(
                    ts_code=code,
                    start_date='20230101',
                    end_date='20261231',
                    fields='ts_code,ann_date,end_date,roe,roe_dt,or_yoy,netprofit_yoy,'
                           'netprofit_margin,grossprofit_margin,eps,dt_eps,'
                           'current_ratio,quick_ratio,debt_to_assets'
                )
                if not df.empty:
                    all_fina.append(df)
            except:
                time.sleep(0.5)
            if (i + 1) % 200 == 0:
                time.sleep(5)
    except Exception as e:
        print(f"  下载失败: {str(e)[:60]}")

    if all_fina:
        result = pd.concat(all_fina, ignore_index=True)
        result = result.drop_duplicates(subset=['ts_code', 'end_date'], keep='last')
        result.to_parquet(cache_file)
        print(f"  保存成功: {cache_file} ({len(result)}条)")
        return result
    return pd.DataFrame()


def compute_lynch_features(fina_df, daily_basic_df, target_date):
    """
    Compute Peter Lynch style fundamental features.
    Key concepts:
    - PEG < 1 = undervalued (Lynch's favorite metric)
    - Revenue growth + profit margin = quality growth
    - ROE consistency = management quality
    """
    if fina_df.empty:
        return pd.DataFrame()

    fina_df = fina_df.copy()
    fina_df['ann_date'] = fina_df['ann_date'].astype(str)

    fina_available = fina_df[fina_df['ann_date'] <= target_date].copy()
    if fina_available.empty:
        return pd.DataFrame()

    latest = fina_available.sort_values('ann_date').groupby('ts_code').last().reset_index()

    latest['or_yoy'] = pd.to_numeric(latest['or_yoy'], errors='coerce')
    latest['netprofit_yoy'] = pd.to_numeric(latest['netprofit_yoy'], errors='coerce')
    latest['netprofit_margin'] = pd.to_numeric(latest['netprofit_margin'], errors='coerce')
    latest['grossprofit_margin'] = pd.to_numeric(latest['grossprofit_margin'], errors='coerce')
    latest['roe'] = pd.to_numeric(latest['roe'], errors='coerce')
    latest['debt_to_assets'] = pd.to_numeric(latest['debt_to_assets'], errors='coerce')

    if not daily_basic_df.empty:
        db = daily_basic_df[['ts_code', 'pe']].copy()
        db['pe'] = pd.to_numeric(db['pe'], errors='coerce')
        latest = pd.merge(latest, db, on='ts_code', how='left')

        latest['lynch_peg'] = np.where(
            (latest['or_yoy'] > 0) & (latest['pe'] > 0),
            latest['pe'] / latest['or_yoy'],
            np.nan
        )
        latest['lynch_peg_rank'] = latest['lynch_peg'].rank(pct=True)
    else:
        latest['lynch_peg'] = np.nan
        latest['lynch_peg_rank'] = np.nan

    latest['lynch_quality_score'] = (
        np.where(latest['or_yoy'] > 20, 1, 0) +
        np.where(latest['netprofit_margin'] > 15, 1, 0) +
        np.where(latest['roe'] > 15, 1, 0) +
        np.where(latest['debt_to_assets'] < 50, 1, 0) +
        np.where(latest['grossprofit_margin'] > 30, 1, 0)
    )

    latest['lynch_growth_value'] = np.where(
        latest['or_yoy'] > 0,
        latest['netprofit_margin'] * latest['or_yoy'] / 100.0,
        0
    )

    latest['lynch_earnings_momentum'] = np.where(
        latest['netprofit_yoy'] > 0, 1,
        np.where(latest['netprofit_yoy'] < 0, -1, 0)
    )

    roe_hist = fina_available.sort_values('ann_date').groupby('ts_code').tail(4)
    if not roe_hist.empty:
        roe_std = roe_hist.groupby('ts_code')['roe'].std()
        latest = pd.merge(latest, roe_std.rename('lynch_roe_stability'), on='ts_code', how='left')
        latest['lynch_roe_stability'] = latest['lynch_roe_stability'].fillna(0)
    else:
        latest['lynch_roe_stability'] = 0

    return latest[[
        'ts_code', 'lynch_peg', 'lynch_peg_rank', 'lynch_quality_score',
        'lynch_growth_value', 'lynch_roe_stability', 'lynch_earnings_momentum',
        'or_yoy', 'netprofit_yoy', 'netprofit_margin', 'grossprofit_margin', 'roe', 'debt_to_assets'
    ]]


# ============================================================
# 3. QUANT FACTOR FEATURES (量化因子)
# ============================================================

def compute_quant_factors(price_hist, other_hist, chip_hist, lookback=20):
    """
    Compute traditional quant factors:
    - Momentum: multi-horizon returns
    - Reversal: short-term overreaction
    - Volatility: realized vol, ATR
    - Liquidity: turnover, volume ratio
    - Technical: RSI, Bollinger, MA cross
    """
    results = []

    for ts_code, group in price_hist.groupby('ts_code'):
        g = group.sort_values('trade_date').tail(lookback + 10).reset_index(drop=True)
        if len(g) < lookback:
            continue

        c = g['close'].values
        h = g['high'].values
        l = g['low'].values
        v = g['vol'].values
        amt = g['amount'].values
        n = len(g)

        # --- 3.1 Momentum ---
        mom_1d = c[-1] / c[-2] - 1 if n >= 2 else 0
        mom_3d = c[-1] / c[-4] - 1 if n >= 4 else 0
        mom_5d = c[-1] / c[-6] - 1 if n >= 6 else 0
        mom_10d = c[-1] / c[-11] - 1 if n >= 11 else 0
        mom_20d = c[-1] / c[-21] - 1 if n >= 21 else 0

        # --- 3.2 Reversal (short-term overreaction) ---
        reversal_1d = -mom_1d  # contrarian next-day
        reversal_3d = -mom_3d

        # --- 3.3 Volatility ---
        rets = np.diff(c[-lookback:]) / c[-lookback:-1]
        realized_vol = np.std(rets) * np.sqrt(252) if len(rets) > 1 else 0

        atr = 0
        if n >= 14:
            tr = np.maximum(h[-14:] - l[-14:],
                            np.maximum(np.abs(h[-14:] - np.append(c[-15:-14], c[-14:-1])),
                                       np.abs(l[-14:] - np.append(c[-15:-14], c[-14:-1]))))
            atr = np.mean(tr)

        # --- 3.4 RSI ---
        rsi_14 = _compute_rsi(c, 14)

        # --- 3.5 Bollinger Bands ---
        ma20 = np.mean(c[-20:]) if n >= 20 else c[-1]
        std20 = np.std(c[-20:]) if n >= 20 else 0
        bb_position = (c[-1] - ma20) / (2 * std20 + 1e-8)

        # --- 3.6 MA Cross ---
        ma5 = np.mean(c[-5:]) if n >= 5 else c[-1]
        ma10 = np.mean(c[-10:]) if n >= 10 else c[-1]
        ma20_val = np.mean(c[-20:]) if n >= 20 else c[-1]
        ma_cross_5_10 = ma5 / ma10 - 1
        ma_cross_10_20 = ma10 / ma20_val - 1

        # --- 3.7 Volume factors ---
        vol_ma5 = np.mean(v[-5:]) if n >= 5 else v[-1]
        vol_ma20 = np.mean(v[-20:]) if n >= 20 else v[-1]
        vol_ratio_5_20 = vol_ma5 / (vol_ma20 + 1e-8)

        # --- 3.8 Price-Volume Divergence ---
        pv_corr = 0
        if n >= 10:
            price_change = np.diff(c[-10:]) / c[-10:-1]
            vol_change = np.diff(v[-10:].astype(float)) / (v[-10:-1].astype(float) + 1e-8)
            if len(price_change) > 2 and np.std(price_change) > 0 and np.std(vol_change) > 0:
                pv_corr = np.corrcoef(price_change, vol_change)[0, 1]
                if np.isnan(pv_corr):
                    pv_corr = 0

        results.append({
            'ts_code': ts_code,
            'qf_mom_1d': mom_1d,
            'qf_mom_3d': mom_3d,
            'qf_mom_5d': mom_5d,
            'qf_mom_10d': mom_10d,
            'qf_mom_20d': mom_20d,
            'qf_reversal_1d': reversal_1d,
            'qf_reversal_3d': reversal_3d,
            'qf_realized_vol': realized_vol,
            'qf_atr_pct': atr / (c[-1] + 1e-8),
            'qf_rsi_14': rsi_14,
            'qf_bb_position': bb_position,
            'qf_ma_cross_5_10': ma_cross_5_10,
            'qf_ma_cross_10_20': ma_cross_10_20,
            'qf_vol_ratio_5_20': vol_ratio_5_20,
            'qf_pv_corr': pv_corr,
        })

    return pd.DataFrame(results)


def _compute_rsi(close, period=14):
    if len(close) < period + 1:
        return 50.0
    deltas = np.diff(close[-(period + 1):])
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    avg_gain = np.mean(gains)
    avg_loss = np.mean(losses)
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))


# ============================================================
# 4. INTEGRATION: Build all features for a given date
# ============================================================

_global_cache = {
    'all_dates': None,
    'date_idx': None,
    'price_hist_cache': {},
    'fina_df': None,
}


def _get_all_dates():
    if _global_cache['all_dates'] is None:
        _global_cache['all_dates'] = sorted(
            [f.replace('.parquet', '') for f in os.listdir(PRICE_DIR) if f.endswith('.parquet')])
        _global_cache['date_idx'] = {d: i for i, d in enumerate(_global_cache['all_dates'])}
    return _global_cache['all_dates'], _global_cache['date_idx']


def _get_price_hist(d_curr, lookback=60):
    all_dates, date_idx = _get_all_dates()
    curr_idx = date_idx.get(d_curr)
    if curr_idx is None:
        return None

    cache_key = d_curr
    if cache_key in _global_cache['price_hist_cache']:
        return _global_cache['price_hist_cache'][cache_key]

    lookback_dates = all_dates[max(0, curr_idx - lookback):curr_idx + 1]
    price_frames = []
    for ld in lookback_dates:
        lp = os.path.join(PRICE_DIR, f"{ld}.parquet")
        if os.path.exists(lp):
            lpdf = pd.read_parquet(lp, columns=['ts_code', 'open', 'high', 'low', 'close', 'vol', 'amount'])
            lpdf['trade_date'] = ld
            price_frames.append(lpdf)

    if price_frames:
        result = pd.concat(price_frames, ignore_index=True)
    else:
        result = None

    _global_cache['price_hist_cache'][cache_key] = result
    if len(_global_cache['price_hist_cache']) > 10:
        oldest = list(_global_cache['price_hist_cache'].keys())[0]
        del _global_cache['price_hist_cache'][oldest]

    return result


def _get_fina_df():
    if _global_cache['fina_df'] is None:
        fina_cache = os.path.join(FUND_DIR, 'fina_indicator_cache.parquet')
        if os.path.exists(fina_cache):
            _global_cache['fina_df'] = pd.read_parquet(fina_cache)
        else:
            _global_cache['fina_df'] = pd.DataFrame()
    return _global_cache['fina_df']


def build_enhanced_features(d_curr, news_mkt=None, news_stk=None):
    """Build complete feature set for date d_curr"""

    all_dates, date_idx = _get_all_dates()
    curr_idx = date_idx.get(d_curr)
    if curr_idx is None:
        return None

    # --- Load base data ---
    p_price = os.path.join(PRICE_DIR, f"{d_curr}.parquet")
    p_other = os.path.join(OTHER_DIR, f"{d_curr}.parquet")
    p_rank = os.path.join(RANK_DIR, f"{d_curr}.parquet")
    p_chip = os.path.join(CHIP_DIR, f"{d_curr}.parquet")

    if not os.path.exists(p_price) or not os.path.exists(p_other):
        return None

    price_df = pd.read_parquet(p_price)
    other_df = pd.read_parquet(p_other)

    # --- Base features (original 5) ---
    if os.path.exists(p_rank):
        rank_df = pd.read_parquet(p_rank)
        rank_df = rank_df.sort_values('hot', ascending=False).drop_duplicates(subset='ts_code', keep='first')
        rank_df['hot_rank_pct'] = rank_df['hot'].rank(pct=True)
    else:
        rank_df = pd.DataFrame({'ts_code': price_df['ts_code'], 'hot_rank_pct': 0.5})

    if os.path.exists(p_chip):
        chip_df = pd.read_parquet(p_chip)
        chip_df['chip_concentration'] = (chip_df['cost_85pct'] - chip_df['cost_15pct']) / (
                chip_df['cost_50pct'] + 1e-8)
    else:
        chip_df = pd.DataFrame({'ts_code': price_df['ts_code'], 'chip_concentration': 0.1, 'winner_rate': 50.0})

    # Merge base
    df = pd.merge(rank_df[['ts_code', 'hot_rank_pct']], price_df, on='ts_code', how='inner')
    df = pd.merge(df, chip_df[['ts_code', 'chip_concentration', 'winner_rate']], on='ts_code', how='left')
    df = pd.merge(df, other_df[['ts_code', 'circ_mv', 'pe', 'pb', 'turnover_rate']], on='ts_code', how='left')

    # Main board filter
    df = df[df['ts_code'].str.startswith('60') | df['ts_code'].str.startswith('00')]
    df = df[df['circ_mv'] <= 3000000]

    # Fill NaN
    df['chip_concentration'] = df['chip_concentration'].fillna(df['chip_concentration'].median())
    df['winner_rate'] = df['winner_rate'].fillna(50.0)

    # --- News features ---
    if news_mkt is not None and not news_mkt.empty:
        df['news_market_impact'] = news_mkt['news_market_impact'].max()
    else:
        df['news_market_impact'] = 0.0
    if news_stk is not None and not news_stk.empty:
        ns_agg = news_stk.groupby('ts_code')['news_stock_impact'].max().reset_index()
        df = pd.merge(df, ns_agg, on='ts_code', how='left')
        df['news_stock_impact'] = df['news_stock_impact'].fillna(0.0)
    else:
        df['news_stock_impact'] = 0.0

    # --- Chan Theory features ---
    price_hist = _get_price_hist(d_curr, lookback=60)
    if price_hist is not None:
        chan_feats = compute_chan_features(price_hist, lookback=60)
        df = pd.merge(df, chan_feats, on='ts_code', how='left')
    else:
        for col in ['chan_bi_count', 'chan_zhongshu_count', 'chan_zhongshu_width',
                     'chan_macd_divergence', 'chan_bi_direction', 'chan_leave_zhongshu']:
            df[col] = 0

    # --- Peter Lynch features ---
    fina_df = _get_fina_df()
    if not fina_df.empty:
        lynch_feats = compute_lynch_features(fina_df, other_df, d_curr)
        if not lynch_feats.empty:
            df = pd.merge(df, lynch_feats, on='ts_code', how='left')
        else:
            for col in ['lynch_peg', 'lynch_peg_rank', 'lynch_quality_score',
                        'lynch_growth_value', 'lynch_roe_stability', 'lynch_earnings_momentum',
                        'or_yoy', 'netprofit_yoy', 'netprofit_margin', 'grossprofit_margin', 'roe_y', 'debt_to_assets']:
                df[col] = 0
    else:
        for col in ['lynch_peg', 'lynch_peg_rank', 'lynch_quality_score',
                    'lynch_growth_value', 'lynch_roe_stability', 'lynch_earnings_momentum',
                    'or_yoy', 'netprofit_yoy', 'netprofit_margin', 'grossprofit_margin', 'roe_y', 'debt_to_assets']:
            df[col] = 0

    # --- Quant Factor features ---
    if price_hist is not None:
        quant_feats = compute_quant_factors(price_hist, other_df, chip_df, lookback=20)
        df = pd.merge(df, quant_feats, on='ts_code', how='left')
    else:
        for col in ['qf_mom_1d', 'qf_mom_3d', 'qf_mom_5d', 'qf_mom_10d', 'qf_mom_20d',
                     'qf_reversal_1d', 'qf_reversal_3d', 'qf_realized_vol', 'qf_atr_pct',
                     'qf_rsi_14', 'qf_bb_position', 'qf_ma_cross_5_10', 'qf_ma_cross_10_20',
                     'qf_vol_ratio_5_20', 'qf_pv_corr']:
            df[col] = 0

    # Fill all NaN with 0
    df = df.fillna(0)

    return df


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == 'download_fund':
        print("Downloading fundamental data...")
        download_fundamental_data()
        print("Done!")
    else:
        print("Usage: python feature_engineering.py download_fund")

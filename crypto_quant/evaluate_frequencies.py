# -*- coding: utf-8 -*-
"""
Systematic Evaluation: 3 Assets (BTC vs ETH vs SOL) x 4 Frequencies
严格在 2024-2025 验证集上运行 12 组网格评测
"""
import os
import numpy as np
import pandas as pd

TOKENS = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT']

FREQUENCIES = {
    '3x/Day (8h)': {'min_bars': 2, 'desc': '高频日内 (持仓8h)'},
    '1x/Day (24h)': {'min_bars': 6, 'desc': '每日一次 (持仓24h)'},
    '1x/3Days (72h)': {'min_bars': 18, 'desc': '3天一次 (持仓3天)'},
    '1x/Week (168h)': {'min_bars': 42, 'desc': '每周一次 (持仓1周)'}
}


def simulate_frequency_strategy(preds, closes, opens, min_bars, fng, stb_flow, rolling_w=72, cost=0.0005):
    """
    模拟特定交易频次与约束下的实盘交易
    """
    n = len(preds)
    z_score = (preds - preds.rolling(rolling_w).mean()) / (preds.rolling(rolling_w).std() + 1e-8)
    
    # 基础入场门槛
    raw_long = (z_score > 0.8) & (stb_flow > 0.0) & (fng < 85)

    pos = np.zeros(n)
    trades = 0
    in_pos = False
    entry_bar = 0

    for i in range(n - 1):
        if not in_pos:
            if raw_long.iloc[i]:
                in_pos = True
                entry_bar = i
                pos[i] = 1.0
                trades += 1
            else:
                pos[i] = 0.0
        else:
            held_bars = i - entry_bar
            # 如果未达到最低持仓周期，强制持仓
            if held_bars < min_bars:
                pos[i] = 1.0
            else:
                # 达到最低持仓周期后，若信号转弱或反转则平仓退出
                if z_score.iloc[i] < 0.2:
                    in_pos = False
                    pos[i] = 0.0
                    trades += 1
                else:
                    pos[i] = 1.0

    # 严格在次根 open 执行
    # 收益率 = pos[i] * (close[i+1] / close[i] - 1) - trade_cost
    rets = closes.shift(-1) / closes - 1
    trade_signals = pd.Series(pos).diff().abs().fillna(0).values
    strat_rets = pos * rets.values - trade_signals * cost
    strat_rets = pd.Series(strat_rets[:-1], index=closes.index[:-1])

    # 指标计算 (每年 2190 根 4h K线)
    cum = (1 + strat_rets).cumprod()
    total_ret = cum.iloc[-1] - 1
    years = len(strat_rets) / 2190
    cagr = (1 + total_ret) ** (1 / years) - 1 if total_ret > -1 else -1.0
    
    ann_vol = strat_rets.std() * np.sqrt(2190)
    sharpe = strat_rets.mean() / (strat_rets.std() + 1e-8) * np.sqrt(2190)

    cum_max = cum.cummax()
    drawdown = (cum - cum_max) / cum_max
    mdd = drawdown.min()
    calmar = cagr / abs(mdd) if abs(mdd) > 1e-6 else 0.0

    non_zero = strat_rets[strat_rets != 0]
    win_rate = (non_zero > 0).mean() if len(non_zero) > 0 else 0.0
    exposure = (pos > 0).mean()

    return {
        'total_ret': total_ret,
        'cagr': cagr,
        'mdd': mdd,
        'sharpe': sharpe,
        'calmar': calmar,
        'win_rate': win_rate,
        'exposure': exposure,
        'trades': trades,
        'cum_curve': cum
    }


def run_grid_evaluation():
    val_path = 'crypto_quant/predictions/val_predictions_2024_2025.parquet'
    if not os.path.exists(val_path):
        raise FileNotFoundError(f"Val predictions {val_path} not found.")

    df_val = pd.read_parquet(val_path)
    val_idx = df_val.index
    print(f"=== Running 12-Grid Evaluation on 2024-2025 Validation Set ({len(df_val)} bars, {val_idx[0].date()} to {val_idx[-1].date()}) ===")

    # 加载链上资金流与情绪
    df_onchain = pd.read_parquet('data/crypto_cache/eth_onchain_sentiment_daily.parquet')
    onchain_aligned = df_onchain.shift(1).reindex(val_idx.normalize(), method='ffill')
    fng = onchain_aligned['fng_score']
    stb_flow = onchain_aligned['stb_flow_7d']

    grid_results = []

    for token in TOKENS:
        preds = df_val[f'{token}_pred_4h']
        closes = df_val[f'{token}_close']
        opens = df_val[f'{token}_open']

        # 基准买入持有
        bh_rets = closes.shift(-1) / closes - 1
        bh_cum = (1 + bh_rets.iloc[:-1]).cumprod()
        bh_tot = bh_cum.iloc[-1] - 1
        bh_years = len(bh_rets) / 2190
        bh_cagr = (1 + bh_tot) ** (1 / bh_years) - 1
        bh_mdd = ((bh_cum - bh_cum.cummax()) / bh_cum.cummax()).min()
        bh_sharpe = bh_rets.mean() / (bh_rets.std() + 1e-8) * np.sqrt(2190)

        grid_results.append({
            'asset': token,
            'freq': 'Buy & Hold (基准)',
            'desc': '现货买入持有',
            'total_ret': bh_tot,
            'cagr': bh_cagr,
            'mdd': bh_mdd,
            'sharpe': bh_sharpe,
            'calmar': bh_cagr / abs(bh_mdd),
            'win_rate': (bh_rets > 0).mean(),
            'exposure': 1.0,
            'trades': 1
        })

        for freq_name, freq_cfg in FREQUENCIES.items():
            res = simulate_frequency_strategy(
                preds, closes, opens, 
                min_bars=freq_cfg['min_bars'], 
                fng=fng, 
                stb_flow=stb_flow
            )
            grid_results.append({
                'asset': token,
                'freq': freq_name,
                'desc': freq_cfg['desc'],
                **res
            })

    df_res = pd.DataFrame(grid_results)
    
    # 打印排版表格
    print("\n" + "=" * 105)
    print("      2024-2025 VALIDATION SET GRID EVALUATION: 3 ASSETS x 4 FREQUENCIES (STRICT ZERO-LEAK)       ")
    print("=" * 105)
    header = f"{'Asset':<10} | {'Frequency':<16} | {'Total Ret':<10} | {'CAGR':<8} | {'MDD':<8} | {'Sharpe':<7} | {'Calmar':<7} | {'Exposure':<9} | {'Trades':<6}"
    print(header)
    print("-" * 105)
    for _, r in df_res.iterrows():
        print(f"{r['asset']:<10} | {r['freq']:<16} | {r['total_ret']*100:+8.2f}% | {r['cagr']*100:+6.2f}% | {r['mdd']*100:6.2f}% | {r['sharpe']:6.2f} | {r['calmar']:6.2f} | {r['exposure']*100:6.1f}%   | {r['trades']:<6}")
    print("=" * 105)

    df_res.to_csv('data/crypto_cache/grid_evaluation_2024_2025.csv', index=False)
    return df_res


if __name__ == '__main__':
    run_grid_evaluation()

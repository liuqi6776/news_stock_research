# -*- coding: utf-8 -*-
"""
Out-of-Sample Backtesting Engine for CryptoSTTransformer (Augmented with On-Chain & Sentiment)
时空关系 Transformer 样本外实盘级别回测引擎 (融合链上资金流与新闻情绪)
"""
import os
import numpy as np
import pandas as pd

TOKENS = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'BNBUSDT']


def calculate_metrics(returns_series, bars_per_year=2190):
    """计算专业机构量化指标 (4小时周期，每年 2,190 根K线)"""
    cum = (1 + returns_series).cumprod()
    total_ret = cum.iloc[-1] - 1
    n_bars = len(returns_series)
    years = n_bars / bars_per_year
    ann_ret = (1 + total_ret) ** (1 / years) - 1 if total_ret > -1 else -1.0
    
    ann_vol = returns_series.std() * np.sqrt(bars_per_year)
    sharpe = (returns_series.mean() / (returns_series.std() + 1e-8)) * np.sqrt(bars_per_year)

    downside = returns_series[returns_series < 0]
    down_vol = downside.std() * np.sqrt(bars_per_year) if len(downside) > 0 else 1e-8
    sortino = (returns_series.mean() * bars_per_year) / (down_vol + 1e-8)

    cum_max = cum.cummax()
    drawdown = (cum - cum_max) / cum_max
    mdd = drawdown.min()

    calmar = ann_ret / abs(mdd) if abs(mdd) > 1e-6 else 0.0

    non_zero = returns_series[returns_series != 0]
    win_rate = (non_zero > 0).mean() if len(non_zero) > 0 else 0.0
    gross_profits = non_zero[non_zero > 0].sum()
    gross_losses = abs(non_zero[non_zero < 0].sum())
    profit_factor = gross_profits / (gross_losses + 1e-8)

    return {
        'total_ret': total_ret,
        'ann_ret': ann_ret,
        'ann_vol': ann_vol,
        'sharpe': sharpe,
        'sortino': sortino,
        'mdd': mdd,
        'calmar': calmar,
        'win_rate': win_rate,
        'profit_factor': profit_factor,
        'cum_curve': cum
    }


def run_transformer_backtest():
    pred_path = 'crypto_quant/predictions/test_predictions.parquet'
    if not os.path.exists(pred_path):
        raise FileNotFoundError(f"Predictions file {pred_path} not found.")

    df_pred = pd.read_parquet(pred_path)
    test_idx = df_pred.index

    # 1. 加载底层 4h 真实行情
    raw_dfs = {t: pd.read_parquet(f'data/crypto_cache/{t}_4h_2021_2026.parquet') for t in TOKENS}

    # 计算各资产逐根下根K线收益率 (严格在次根 open 挂单进场，无未来函数)
    rets_df = pd.DataFrame(index=test_idx)
    for t in TOKENS:
        c = raw_dfs[t].loc[test_idx, 'close']
        rets_df[t] = c.shift(-1) / c - 1

    # 加载链上资金流与情绪日线数据 (严格因果前向填充，滞后 1 天)
    onchain_path = 'data/crypto_cache/eth_onchain_sentiment_daily.parquet'
    df_onchain = pd.read_parquet(onchain_path)
    onchain_aligned = df_onchain.shift(1).reindex(test_idx.normalize(), method='ffill')
    fng = onchain_aligned['fng_score'].values
    stb_flow_7d = onchain_aligned['stb_flow_7d'].values

    # 费率摩擦：币安 Taker 手续费 0.04% + 滑点 0.01% = 单边 0.05%
    taker_cost = 0.0005
    rolling_w = 72  # 12天滚动窗口

    results = {}

    # -------------------------------------------------------------
    # 策略 1: ETH 双层分级强化系统 (Transformer + 链上资金流 + 情绪风控)
    # -------------------------------------------------------------
    eth_pred = df_pred['eth_pred_4h']
    z_score = (eth_pred - eth_pred.rolling(rolling_w).mean()) / (eth_pred.rolling(rolling_w).std() + 1e-8)

    # 复合条件：模型高置信度(z>1.0) + 链上稳定币净流入(stb>0) + 避开非理性癫狂(fng<85)
    cond_hierarchical = (z_score > 1.0) & (stb_flow_7d > 0.0) & (fng < 85)
    eth_sig_hier = cond_hierarchical.astype(float)
    eth_tr_hier = eth_sig_hier.diff().abs().fillna(0)
    eth_rets_hier = (eth_sig_hier * rets_df['ETHUSDT'] - eth_tr_hier * taker_cost).iloc[:-1]
    results['ETH Hierarchical (AI+OnChain+News)'] = {**calculate_metrics(eth_rets_hier), 'trades': int(eth_tr_hier.sum())}

    # -------------------------------------------------------------
    # 策略 2: ETH 单纯基准 Transformer 策略 (z > 1.0)
    # -------------------------------------------------------------
    eth_sig_base = (z_score > 1.0).astype(float)
    eth_tr_base = eth_sig_base.diff().abs().fillna(0)
    eth_rets_base = (eth_sig_base * rets_df['ETHUSDT'] - eth_tr_base * taker_cost).iloc[:-1]
    results['ETH Baseline Transformer (z>1.0)'] = {**calculate_metrics(eth_rets_base), 'trades': int(eth_tr_base.sum())}

    # -------------------------------------------------------------
    # 策略 3: 多币种跨资产轮动策略 (Multi-Asset Top-1 Rotation)
    # -------------------------------------------------------------
    preds_df = pd.DataFrame(index=test_idx)
    for t in TOKENS:
        preds_df[t] = df_pred[f'{t}_pred_4h']

    top1_token = preds_df.idxmax(axis=1)
    rot_rets = []
    current_holding = None
    rot_trade_count = 0

    for i in range(len(test_idx) - 1):
        target_t = top1_token.iloc[i]
        expected_ret = preds_df.loc[test_idx[i], target_t]

        if expected_ret > 0.0:
            bar_ret = rets_df.loc[test_idx[i], target_t]
            fee_deduction = taker_cost if target_t != current_holding else 0.0
            if target_t != current_holding:
                rot_trade_count += 1
            current_holding = target_t
            rot_rets.append(bar_ret - fee_deduction)
        else:
            fee_deduction = taker_cost if current_holding is not None else 0.0
            if current_holding is not None:
                rot_trade_count += 1
            current_holding = None
            rot_rets.append(0.0 - fee_deduction)

    rot_strat_rets = pd.Series(rot_rets, index=test_idx[:-1])
    results['Multi-Asset Rotation (Top-1)'] = {**calculate_metrics(rot_strat_rets), 'trades': rot_trade_count}

    # -------------------------------------------------------------
    # 基准策略 (Benchmarks)
    # -------------------------------------------------------------
    eth_bh_rets = rets_df['ETHUSDT'].iloc[:-1]
    btc_bh_rets = rets_df['BTCUSDT'].iloc[:-1]

    results['ETH Buy & Hold'] = {**calculate_metrics(eth_bh_rets), 'trades': 1}
    results['BTC Buy & Hold'] = {**calculate_metrics(btc_bh_rets), 'trades': 1}

    # 导出净值曲线到 CSV
    equity_df = pd.DataFrame(index=eth_rets_hier.index)
    equity_df['ETH_Hierarchical'] = results['ETH Hierarchical (AI+OnChain+News)']['cum_curve']
    equity_df['ETH_Baseline'] = results['ETH Baseline Transformer (z>1.0)']['cum_curve']
    equity_df['Multi_Asset_Rotation'] = results['Multi-Asset Rotation (Top-1)']['cum_curve']
    equity_df['ETH_Buy_Hold'] = results['ETH Buy & Hold']['cum_curve']
    equity_df['BTC_Buy_Hold'] = results['BTC Buy & Hold']['cum_curve']

    # 打印对比表格
    print("\n" + "=" * 94)
    print("        OUT-OF-SAMPLE 2.7-YEAR PERFORMANCE COMPARISON: BASELINE vs AUGMENTED (2024-2026)      ")
    print("=" * 94)
    header = f"{'Strategy / Benchmark':<35} | {'Total Ret':<10} | {'CAGR':<8} | {'MDD':<8} | {'Sharpe':<7} | {'Calmar':<7} | {'Win%':<6} | {'Trades':<6}"
    print(header)
    print("-" * 94)
    for name, m in results.items():
        print(f"{name:<35} | {m['total_ret']*100:+8.2f}% | {m['ann_ret']*100:+6.2f}% | {m['mdd']*100:6.2f}% | {m['sharpe']:6.2f} | {m['calmar']:6.2f} | {m['win_rate']*100:5.1f}% | {m['trades']:<6}")
    print("=" * 94)

    # 年度收益分解表 (Annual Breakdown)
    print("\n" + "=" * 72)
    print("                     ANNUAL RETURN BREAKDOWN (%)                     ")
    print("=" * 72)
    print(f"{'Strategy / Year':<35} | {'2024':<10} | {'2025':<10} | {'2026 (YTD)':<10}")
    print("-" * 72)

    for s_name, col in [
        ('ETH Hierarchical (AI+OnChain+News)', 'ETH_Hierarchical'),
        ('ETH Baseline Transformer', 'ETH_Baseline'),
        ('Multi-Asset Rotation', 'Multi_Asset_Rotation'),
        ('ETH Buy & Hold', 'ETH_Buy_Hold'),
        ('BTC Buy & Hold', 'BTC_Buy_Hold')
    ]:
        s = equity_df[col]
        s_2024 = s.loc[s.index.year == 2024]
        s_2025 = s.loc[s.index.year == 2025]
        s_2026 = s.loc[s.index.year == 2026]
        
        ret_2024 = (s_2024.iloc[-1] / s_2024.iloc[0] - 1) if len(s_2024) > 0 else 0.0
        ret_2025 = (s_2025.iloc[-1] / s_2025.iloc[0] - 1) if len(s_2025) > 0 else 0.0
        ret_2026 = (s_2026.iloc[-1] / s_2026.iloc[0] - 1) if len(s_2026) > 0 else 0.0
        print(f"{s_name:<35} | {ret_2024*100:+8.1f}%  | {ret_2025*100:+8.1f}%  | {ret_2026*100:+8.1f}%")
    print("=" * 72)

    equity_path = 'data/crypto_cache/transformer_backtest_equity.csv'
    equity_df.to_csv(equity_path)
    print(f"\nEquity curves successfully saved to {equity_path}")

    return results, equity_df


if __name__ == '__main__':
    run_transformer_backtest()

# -*- coding: utf-8 -*-
"""
Analysis of Leverage Feasibility, Per-Trade Return Distribution, and Leveraged Equity Curves
for Ethereum (ETH) and Solana (SOL) Spatio-Temporal Transformer Strategies.
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy import stats

# Set plot style and font
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

def find_project_root():
    current = os.path.abspath(os.path.dirname(__file__))
    candidates = [
        os.path.abspath(os.path.join(current, '..')),
        current,
        os.getcwd(),
        os.path.abspath(os.path.join(os.getcwd(), '..'))
    ]
    for c in candidates:
        if os.path.exists(os.path.join(c, 'data')) and os.path.exists(os.path.join(c, 'predictions')):
            return c
    return os.getcwd()

root_dir = find_project_root()
sys.path.insert(0, root_dir)

docs_dir = os.path.join(root_dir, 'docs')
os.makedirs(docs_dir, exist_ok=True)

# 1. Load Data
df_pred = pd.read_parquet(os.path.join(root_dir, 'predictions', 'test_predictions.parquet'))
val_df = df_pred.loc['2024-01-01':'2025-12-31']
val_idx = val_df.index

data_dir = os.path.join(root_dir, 'data')
raw_dfs = {t: pd.read_parquet(os.path.join(data_dir, f'{t}_4h_2020_2026.parquet')).loc[val_idx] for t in ['ETHUSDT', 'SOLUSDT']}

df_onchain = pd.read_parquet(os.path.join(data_dir, 'eth_onchain_sentiment_daily.parquet'))
onchain_aligned = df_onchain.shift(1).reindex(val_idx.normalize(), method='ffill')
onchain_aligned.index = val_idx

fng_arr = onchain_aligned['fng_score'].values
stb_flow_arr = onchain_aligned['stb_flow_7d'].values


def extract_discrete_trades_and_leverage(token, leverage_list=[1.0, 1.5, 2.0, 3.0]):
    preds = val_df[f'{token}_pred_4h']
    opens = raw_dfs[token]['open']
    p_series = pd.Series(preds.values, index=val_idx)

    # Strictly causal shift(1) prior z-score
    prior_mean = p_series.shift(1).rolling(72).mean()
    prior_std = p_series.shift(1).rolling(72).std() + 1e-8
    z_vals = ((p_series - prior_mean) / prior_std).values

    # High-confidence regime filter
    raw_sig = (z_vals > 1.0) & (stb_flow_arr > 0.0) & (fng_arr < 85)

    n = len(preds)
    pos = np.zeros(n)
    in_pos = False
    entry_bar = 0
    trades = []

    for i in range(n - 2):
        if not in_pos:
            if raw_sig[i]:
                in_pos = True
                entry_bar = i
                pos[i] = 1.0
            else:
                pos[i] = 0.0
        else:
            if not raw_sig[i]:
                in_pos = False
                pos[i] = 0.0
                # Trade is entered at open[entry_bar + 1] and exited at open[i + 1]
                entry_idx = entry_bar + 1
                exit_idx = i + 1
                entry_price = opens.iloc[entry_idx]
                exit_price = opens.iloc[exit_idx]
                gross_ret = exit_price / entry_price - 1.0
                duration_bars = exit_idx - entry_idx
                duration_hours = duration_bars * 4

                # Taker fee 0.05% entry + 0.05% exit = 0.10%
                # Funding fee approx 0.01% per 8h in perpetuals = 0.005% per 4h bar
                taker_fee = 2 * 0.0005
                funding_fee = duration_bars * 0.00005
                net_ret = gross_ret - taker_fee - funding_fee

                trades.append({
                    'entry_time': opens.index[entry_idx],
                    'exit_time': opens.index[exit_idx],
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'duration_hours': duration_hours,
                    'duration_bars': duration_bars,
                    'gross_ret': gross_ret,
                    'taker_fee': taker_fee,
                    'funding_fee': funding_fee,
                    'net_ret': net_ret
                })
            else:
                pos[i] = 1.0

    trades_df = pd.DataFrame(trades)

    # Bar-by-bar leverage simulation
    o_series = pd.Series(opens.values, index=val_idx)
    rets_oto = (o_series.shift(-2) / o_series.shift(-1) - 1).values
    trade_signals = pd.Series(pos).diff().abs().fillna(0).values

    # Benchmark Buy & Hold
    bh_rets = (o_series.shift(-2) / o_series.shift(-1) - 1).iloc[:-2]
    bh_cum = (1 + bh_rets).cumprod()

    lev_results = {
        'Buy & Hold': {
            'cum': bh_cum,
            'total_ret': bh_cum.iloc[-1] - 1.0,
            'cagr': (bh_cum.iloc[-1]) ** (1 / (len(bh_rets) / 2190)) - 1.0,
            'mdd': ((bh_cum - bh_cum.cummax()) / bh_cum.cummax()).min(),
            'daily_sharpe': bh_cum.resample('1D').last().ffill().pct_change().dropna().mean() / (bh_cum.resample('1D').last().ffill().pct_change().dropna().std() + 1e-8) * np.sqrt(365),
            'calmar': ((bh_cum.iloc[-1]) ** (1 / (len(bh_rets) / 2190)) - 1.0) / abs(((bh_cum - bh_cum.cummax()) / bh_cum.cummax()).min())
        }
    }

    dates_series = opens.index[:-2]

    for lev in leverage_list:
        # Cost scales with leverage
        cost_bar = trade_signals * 0.0005 * lev
        # Funding rate on borrowed margin: (lev - 1) * 0.00005 per 4h bar, plus taker costs
        funding_bar = pos * 0.00005 * max(0.0, lev - 1.0)
        strat_rets = (pos * rets_oto * lev - cost_bar - funding_bar)[:-2]

        cum = pd.Series((1 + strat_rets).cumprod(), index=dates_series)
        total_ret = cum.iloc[-1] - 1.0
        cagr = (1 + total_ret) ** (1 / (len(strat_rets) / 2190)) - 1.0 if total_ret > -1 else -1.0
        cum_max = cum.cummax()
        dd = (cum - cum_max) / cum_max
        mdd = dd.min()

        daily_equity = cum.resample('1D').last().ffill()
        daily_rets = daily_equity.pct_change().dropna()
        daily_sharpe = daily_rets.mean() / (daily_rets.std() + 1e-8) * np.sqrt(365)
        calmar = cagr / abs(mdd) if abs(mdd) > 1e-6 else 0.0

        lev_results[f'{lev:.1f}x Leverage' if lev > 1.0 else '1.0x (Original)'] = {
            'cum': cum,
            'drawdown': dd,
            'total_ret': total_ret,
            'cagr': cagr,
            'mdd': mdd,
            'daily_sharpe': daily_sharpe,
            'calmar': calmar
        }

    return trades_df, lev_results, dates_series


print("Computing ETH and SOL trade metrics and leverage simulation...")
eth_trades, eth_lev, eth_dates = extract_discrete_trades_and_leverage('ETHUSDT', [1.0, 1.5, 2.0, 3.0])
sol_trades, sol_lev, sol_dates = extract_discrete_trades_and_leverage('SOLUSDT', [1.0, 1.5, 2.0, 3.0])


# ==============================================================================
# PLOT 1: Trade Return Distributions (ETH & SOL)
# ==============================================================================
fig, axes = plt.subplots(2, 2, figsize=(16, 12), gridspec_kw={'width_ratios': [3, 1], 'height_ratios': [1, 1]})
plt.subplots_adjust(hspace=0.28, wspace=0.18)

colors = {'ETH': '#1f77b4', 'SOL': '#9467bd', 'green': '#2ca02c', 'red': '#d62728', 'dark': '#333333'}

for row_idx, (token_name, trades_df, color) in enumerate([('ETHUSDT (Ethereum)', eth_trades, colors['ETH']), ('SOLUSDT (Solana)', sol_trades, colors['SOL'])]):
    net_rets = trades_df['net_ret'] * 100.0  # in percent
    win_trades = net_rets[net_rets > 0]
    loss_trades = net_rets[net_rets <= 0]
    win_rate = len(win_trades) / len(net_rets) * 100.0
    avg_win = win_trades.mean()
    avg_loss = abs(loss_trades.mean())
    payoff = avg_win / (avg_loss + 1e-8)
    profit_factor = win_trades.sum() / (abs(loss_trades.sum()) + 1e-8)
    mean_ret = net_rets.mean()
    median_ret = net_rets.median()
    skewness = stats.skew(net_rets)
    kurt = stats.kurtosis(net_rets)
    max_win = net_rets.max()
    max_loss = net_rets.min()

    # Left: Histogram + KDE
    ax_hist = axes[row_idx, 0]
    n_bins = 40
    counts, bins, patches = ax_hist.hist(net_rets, bins=n_bins, density=True, alpha=0.55, color=color, edgecolor='white', linewidth=0.8)

    # Color negative bins red and positive green
    for b_idx, patch in enumerate(patches):
        if bins[b_idx] < 0:
            patch.set_facecolor('#e74c3c')
            patch.set_alpha(0.5)
        else:
            patch.set_facecolor('#2ecc71')
            patch.set_alpha(0.6)

    # KDE curve
    kde = stats.gaussian_kde(net_rets)
    x_grid = np.linspace(net_rets.min() - 1, net_rets.max() + 1, 300)
    ax_hist.plot(x_grid, kde(x_grid), color='#2c3e50', linewidth=2.0, label='KDE Density Fit')

    # Reference lines
    ax_hist.axvline(0, color='black', linestyle='-', linewidth=1.2, alpha=0.7)
    ax_hist.axvline(mean_ret, color='#2980b9', linestyle='--', linewidth=1.8, label=f'Mean: {mean_ret:+.2f}%')
    ax_hist.axvline(median_ret, color='#8e44ad', linestyle=':', linewidth=2.0, label=f'Median: {median_ret:+.2f}%')

    ax_hist.set_title(f'{token_name} - Per-Trade Net Return Distribution (N = {len(trades_df)} Trades)', fontsize=13, fontweight='bold', pad=10)
    ax_hist.set_xlabel('Per-Trade Net Return (Deducted 0.10% Taker Fee & Funding) [%]', fontsize=10)
    ax_hist.set_ylabel('Probability Density', fontsize=10)
    ax_hist.grid(True, linestyle='--', alpha=0.35)
    ax_hist.legend(loc='upper right', fontsize=9.5, framealpha=0.9)

    # Stats text box
    stats_text = (
        f"Trades: {len(trades_df)}\n"
        f"Win Rate: {win_rate:.1f}%\n"
        f"Profit Factor: {profit_factor:.2f}\n"
        f"Payoff (Win/Loss): {payoff:.2f}:1\n"
        f"Avg Win: {avg_win:+.2f}%\n"
        f"Avg Loss: -{avg_loss:.2f}%\n"
        f"Max Win: {max_win:+.2f}%\n"
        f"Max Loss: {max_loss:+.2f}%\n"
        f"Skewness: {skewness:+.2f} (Fat Right Tail)\n"
        f"Avg Duration: {trades_df['duration_hours'].mean():.1f}h"
    )
    ax_hist.text(0.03, 0.95, stats_text, transform=ax_hist.transAxes, verticalalignment='top',
                 fontsize=9.5, fontfamily='monospace',
                 bbox=dict(boxstyle='round,pad=0.6', facecolor='#f8f9fa', edgecolor='#bdc3c7', alpha=0.95))

    # Right: Boxplot / Distribution Shape
    ax_box = axes[row_idx, 1]
    bp = ax_box.boxplot(net_rets, patch_artist=True, vert=True, widths=0.45,
                        boxprops=dict(facecolor=color, color='#2c3e50', alpha=0.6),
                        medianprops=dict(color='#e74c3c', linewidth=2.0),
                        whiskerprops=dict(color='#2c3e50', linewidth=1.2),
                        capprops=dict(color='#2c3e50', linewidth=1.2),
                        flierprops=dict(marker='o', markerfacecolor='#e67e22', markersize=4, alpha=0.6, markeredgecolor='none'))
    ax_box.axhline(0, color='black', linestyle='--', linewidth=1.0, alpha=0.7)
    ax_box.set_title('Outlier Dispersion', fontsize=11, fontweight='bold', pad=10)
    ax_box.set_ylabel('Net Return [%]', fontsize=10)
    ax_box.set_xticklabels([token_name.split()[0]])
    ax_box.grid(True, linestyle='--', alpha=0.35)

plt.suptitle('Cross-Asset Transformer Trading System: Per-Trade Profitability & Tail Risk Profile (2024-2025)',
             fontsize=15, fontweight='bold', y=0.995)

dist_plot_path = os.path.join(docs_dir, 'eth_sol_trade_distribution.png')
fig.savefig(dist_plot_path, dpi=300, bbox_inches='tight')
plt.close(fig)
print(f"Saved Trade Distribution Plot to {dist_plot_path}")


# ==============================================================================
# PLOT 2: Comparative Equity Curves Under Different Leverage Levels
# ==============================================================================
fig, (ax_eth, ax_sol) = plt.subplots(2, 1, figsize=(16, 12), sharex=True)
plt.subplots_adjust(hspace=0.18)

lev_styles = {
    '1.0x (Original)': {'color': '#2ecc71', 'lw': 2.0, 'ls': '-'},
    '1.5x Leverage':   {'color': '#3498db', 'lw': 2.2, 'ls': '-'},
    '2.0x Leverage':   {'color': '#f39c12', 'lw': 2.4, 'ls': '-'},
    '3.0x Leverage':   {'color': '#e74c3c', 'lw': 2.2, 'ls': '--'},
    'Buy & Hold':      {'color': '#7f8c8d', 'lw': 1.6, 'ls': ':'}
}

for ax, token_name, lev_res in [(ax_eth, 'ETHUSDT (Ethereum) - Leverage Sensitivity & Alpha', eth_lev),
                                (ax_sol, 'SOLUSDT (Solana) - Leverage Sensitivity & Alpha', sol_lev)]:
    for label, data in lev_res.items():
        style = lev_styles[label]
        cum_series = data['cum']
        tot = data['total_ret'] * 100.0
        mdd = data['mdd'] * 100.0
        sh = data['daily_sharpe']
        cal = data['calmar']
        legend_label = f"{label:16s} | Total: {tot:+7.1f}% | MDD: {mdd:5.1f}% | Daily Sh: {sh:4.2f} | Calmar: {cal:4.2f}"
        ax.plot(cum_series.index, cum_series.values, label=legend_label,
                color=style['color'], linewidth=style['lw'], linestyle=style['ls'])

    ax.set_title(token_name, fontsize=13, fontweight='bold', pad=8)
    ax.set_ylabel('Portfolio Equity (Starting = $1.00)', fontsize=11)
    ax.grid(True, linestyle='--', alpha=0.4)
    ax.legend(loc='upper left', fontsize=9.2, framealpha=0.92, prop={'family': 'monospace'})
    ax.set_yscale('log')
    ax.yaxis.set_major_formatter(ticker.ScalarFormatter())

ax_sol.set_xlabel('Date (2024 - 2025 Walk-Forward Validation Period)', fontsize=11)

plt.suptitle('Crypto Transformer Strategy: Leverage Stress Test & Risk-Return Scaling Analysis\n(Includes 0.10% Taker Slippage & Continuous Borrow Funding Rates)',
             fontsize=14, fontweight='bold', y=0.985)

lev_plot_path = os.path.join(docs_dir, 'eth_sol_leverage_comparison.png')
fig.savefig(lev_plot_path, dpi=300, bbox_inches='tight')
plt.close(fig)
print(f"Saved Leverage Comparison Plot to {lev_plot_path}")

# Print tabular breakdown
print("\n" + "=" * 80)
print("LEVERAGE BACKTEST COMPARISON SUMMARY (2024-2025)")
print("=" * 80)
for t_name, lev_res in [('ETHUSDT', eth_lev), ('SOLUSDT', sol_lev)]:
    print(f"\n--- {t_name} ---")
    for k, v in lev_res.items():
        print(f"  {k:18s} | Total Ret: {v['total_ret']*100:+7.2f}% | CAGR: {v['cagr']*100:+6.2f}% | MDD: {v['mdd']*100:6.2f}% | Daily Sharpe: {v['daily_sharpe']:5.2f} | Calmar: {v['calmar']:5.2f}")

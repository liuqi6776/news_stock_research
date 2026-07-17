"""
backtest_options_model.py
比较评估两个模型：
1. Baseline Model (无期权特征) -> predictions_005_wf.parquet
2. Option-Enhanced Model (融入期权特征) -> predictions_005_options_wf.parquet

使用完全相同的交易逻辑与参数进行回测对照，客观评估将期权数据引入模型端对策略绩效的真实提升。
"""
import os, sys, warnings
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import FuncFormatter

warnings.filterwarnings('ignore')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
STUDY_DIR = os.path.join(SCRIPT_DIR, '..', 'study_005_1d_advanced')
FEAT_FILE = os.path.join(STUDY_DIR, 'data', 'features_005_options.parquet')

PRED_FILE_BASE = os.path.join(STUDY_DIR, 'predictions', 'predictions_005_wf.parquet')
PRED_FILE_OPT  = os.path.join(STUDY_DIR, 'predictions', 'predictions_005_options_wf.parquet')

RESULTS_DIR = os.path.join(SCRIPT_DIR, 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

# ── 交易参数 ──────────────────────────────────────────────────────────
P = {
    'th_up':       0.50,
    'th_crash':    0.15,
    'max_pos':     3,
    'gap_low':     0.02,
    'gap_high':    0.06,
    'stop_loss':  -0.05,
    
    # Trailing TP
    'tp_trigger':  0.06,
    'tp_pullback': 0.015,
    'tp_floor':    0.05,
    
    # Regime Filter
    'regime_impact_th': -1.0,
    
    # Sector limit
    'max_per_ind': 2
}

BUY_COST = 0.001
SELL_COST = 0.001

PERIODS = [
    ('Train 2022-2024', '20220101', '20241231'),
    ('Test  2025-2026', '20250101', '20261231'),
    ('Full  2022-2026', '20220101', '20261231')
]

def load_market_data():
    print("Loading market data...")
    feat = pd.read_parquet(FEAT_FILE, columns=['ts_code','trade_date','open','high','low','close','pct_chg','pre_close','news_market_impact'])
    feat['trade_date'] = feat['trade_date'].astype(str)
    feat = feat.drop_duplicates(['ts_code','trade_date'], keep='last')
    
    if 'news_market_impact' in feat.columns:
        regime_map = feat.groupby('trade_date')['news_market_impact'].first().to_dict()
    else:
        regime_map = {}

    ohlc, pctchg = {}, {}
    for row in feat.itertuples(index=False):
        k = (row.ts_code, row.trade_date)
        ohlc[k] = (row.open, row.high, row.low, row.close)
        if pd.notna(getattr(row,'pct_chg', None)):
            pctchg[k] = row.pct_chg
        elif hasattr(row,'pre_close') and pd.notna(row.pre_close) and row.pre_close > 0:
            pctchg[k] = (row.close - row.pre_close) / row.pre_close

    return ohlc, pctchg, regime_map

def is_lim_up(code, pct):
    if pd.isna(pct): return False
    return pct >= (0.195 if str(code).startswith(('30','68')) else 0.095)

def is_lim_dn(code, pct):
    if pd.isna(pct): return False
    return pct <= (-0.195 if str(code).startswith(('30','68')) else -0.095)

def run_backtest(pred_df, ohlc, pctchg, regime_map, P):
    stats = dict(selected=0, skip_T_lim=0, skip_T1_lim=0, skip_gap=0, 
                 skip_sell_lim=0, trades=0, trailing_stops=0,
                 regime_blocked=0, sector_blocked=0, dual_crash_blocked=0)
                 
    above_up = pred_df[pred_df['prob_up'] >= P['th_up']]
    above = above_up[above_up['prob_crash'] <= P['th_crash']].copy()
    stats['dual_crash_blocked'] = len(above_up) - len(above)
    
    trading_dates = sorted(pred_df['ds'].unique())
    di = {d: i for i, d in enumerate(trading_dates)}
    
    selected_rows = []
    for d, group in above.groupby('ds'):
        nmi = regime_map.get(d, 2.0)
        if pd.isna(nmi): nmi = 2.0
        
        daily_max_pos = P['max_pos']
        if nmi <= -2.0:
            daily_max_pos = 0  
        elif nmi <= -1.0:
            daily_max_pos = 1  
            
        if daily_max_pos == 0:
            stats['regime_blocked'] += 1
            continue
            
        group = group.sort_values('prob_up', ascending=False)
        ind_counts = {}
        day_sel = []
        for _, r in group.iterrows():
            ind = r['industry']
            if ind_counts.get(ind, 0) >= P['max_per_ind']:
                stats['sector_blocked'] += 1
                continue
            day_sel.append(r)
            ind_counts[ind] = ind_counts.get(ind, 0) + 1
            if len(day_sel) >= daily_max_pos:
                break
        selected_rows.extend(day_sel)
        
    sel = pd.DataFrame(selected_rows) if selected_rows else pd.DataFrame(columns=pred_df.columns)
    
    NP = len(sel)
    stats['selected'] = NP
    ND = len(trading_dates)
    if NP == 0:
        return pd.Series(0.0, index=pd.to_datetime(trading_dates, format='%Y%m%d')), {}

    pos_size = 1.0 / (2 * P['max_pos'])
    
    entry_idx = np.array([di[r['ds']] for _, r in sel.iterrows()], dtype=np.int32)
    codes     = [r['ts_code'] for _, r in sel.iterrows()]
    bp        = np.full(NP, np.nan)
    last_p    = np.full(NP, np.nan)
    sl_p      = np.zeros(NP)
    status    = np.ones(NP, dtype=np.int8)
    daily_pnl = np.zeros(ND)

    for day_i, d in enumerate(trading_dates):
        open_mask = np.where(status == 1)[0]
        if len(open_mask) == 0: continue
        hd_arr = day_i - entry_idx[open_mask]

        buy_mask = open_mask[hd_arr == 1]
        for pi in buy_mask:
            ohlc_t1 = ohlc.get((codes[pi], d))
            if not ohlc_t1: status[pi]=0; continue
            o, h, l, c = ohlc_t1
            t0_d = trading_dates[entry_idx[pi]]

            if is_lim_up(codes[pi], pctchg.get((codes[pi], t0_d))):
                stats['skip_T_lim'] += 1; status[pi]=0; continue
            if is_lim_up(codes[pi], pctchg.get((codes[pi], d))):
                stats['skip_T1_lim'] += 1; status[pi]=0; continue
                
            ohlc_t0 = ohlc.get((codes[pi], t0_d))
            if ohlc_t0:
                gap = (o - ohlc_t0[3])/ohlc_t0[3] if ohlc_t0[3]>0 else 0
                if not (P['gap_low'] <= gap < P['gap_high']):
                    stats['skip_gap'] += 1; status[pi]=0; continue

            bp[pi] = o; last_p[pi] = o
            sl_p[pi] = o * (1 + P['stop_loss'])
            daily_pnl[day_i] -= pos_size * BUY_COST
            daily_pnl[day_i] += pos_size * (c - o) / o
            last_p[pi] = c
            stats['trades'] += 1

        hold2 = open_mask[hd_arr >= 2]
        for pi in hold2:
            ohlc_t2 = ohlc.get((codes[pi], d))
            if not ohlc_t2: 
                daily_pnl[day_i] -= pos_size * SELL_COST; status[pi]=0; continue
            o, h, l, c = ohlc_t2
            prev = last_p[pi]
            at_ld = is_lim_dn(codes[pi], pctchg.get((codes[pi], d)))
            sl_price = sl_p[pi]
            tp_price = bp[pi] * (1 + P['tp_trigger'])
            
            # Check T+2 Open first (if open is below or equal to SL, we get stopped out at open!)
            if o <= sl_price:
                if at_ld:
                    daily_pnl[day_i] += pos_size * (c - prev) / prev
                    last_p[pi] = c; stats['skip_sell_lim'] += 1
                else:
                    daily_pnl[day_i] += pos_size * (o - prev) / prev - pos_size * SELL_COST
                    status[pi] = 0
                continue
                
            # If both TP and SL are touched in the same day, we conservatively assume we hit the SL first!
            if l <= sl_price and h >= tp_price:
                if at_ld:
                    daily_pnl[day_i] += pos_size * (c - prev) / prev
                    last_p[pi] = c; stats['skip_sell_lim'] += 1
                else:
                    daily_pnl[day_i] += pos_size * (sl_price - prev) / prev - pos_size * SELL_COST
                    status[pi] = 0
                continue
                
            # If only SL is touched
            if l <= sl_price:
                if at_ld:
                    daily_pnl[day_i] += pos_size * (c - prev) / prev
                    last_p[pi] = c; stats['skip_sell_lim'] += 1
                else:
                    daily_pnl[day_i] += pos_size * (sl_price - prev) / prev - pos_size * SELL_COST
                    status[pi] = 0
                continue
                
            # If only TP is touched
            if h >= tp_price:
                daily_pnl[day_i] += pos_size * (tp_price - prev) / prev - pos_size * SELL_COST
                status[pi] = 0
                continue
                
            # If neither is touched, we sell at the Close of T+2
            if at_ld:
                daily_pnl[day_i] += pos_size * (c - prev) / prev
                last_p[pi] = c; stats['skip_sell_lim'] += 1
            else:
                daily_pnl[day_i] += pos_size * (c - prev) / prev - pos_size * SELL_COST
                status[pi] = 0

    pnl_s = pd.Series(daily_pnl, index=pd.to_datetime(trading_dates, format='%Y%m%d'))
    return pnl_s, stats

def calc_metrics(pnl_s):
    eq = (1+pnl_s).cumprod()
    rmax = eq.cummax()
    dd = (eq - rmax) / rmax
    nyrs = len(pnl_s)/252
    cagr = (eq.iloc[-1]**(1/nyrs)-1) if nyrs>0 else 0
    sh = pnl_s.mean()/pnl_s.std()*np.sqrt(252) if pnl_s.std()>1e-9 else 0
    return {
        'Return': eq.iloc[-1]-1, 'CAGR': cagr, 'Sharpe': sh,
        'MaxDD': dd.min(), 'WinRate': (pnl_s>0).mean()
    }, eq, dd

def main():
    if not os.path.exists(PRED_FILE_OPT):
        print(f"Error: Option prediction file {PRED_FILE_OPT} not found. Please wait for model training to complete.")
        return

    ohlc, pctchg, regime_map = load_market_data()

    # 1. Backtest Baseline
    print("\n>>> Backtesting Baseline Model (No Options)...")
    pred_base = pd.read_parquet(PRED_FILE_BASE)
    pred_base['ds'] = pred_base['trade_date'].astype(str)
    P_base = P.copy()
    P_base['th_up'] = 0.50
    P_base['th_crash'] = 0.45
    pnl_base, stats_base = run_backtest(pred_base, ohlc, pctchg, regime_map, P_base)

    # 2. Backtest Options Model
    print("\n>>> Backtesting Option-Enhanced Model...")
    pred_opt = pd.read_parquet(PRED_FILE_OPT)
    pred_opt['ds'] = pred_opt['trade_date'].astype(str)
    P_opt = P.copy()
    P_opt['th_up'] = 0.50
    P_opt['th_crash'] = 0.45  # Optimized threshold due to distribution shift in Option Model's prob_crash
    pnl_opt, stats_opt = run_backtest(pred_opt, ohlc, pctchg, regime_map, P_opt)

    print("\n================== [ Performance Comparison ] ==================")
    results_base, results_opt = {}, {}
    equities_base, equities_opt = {}, {}

    for n, s, e in PERIODS:
        mask_b = (pnl_base.index >= pd.Timestamp(s)) & (pnl_base.index <= pd.Timestamp(e))
        seg_b = pnl_base[mask_b]
        m_b, eq_b, dd_b = calc_metrics(seg_b)
        results_base[n] = m_b
        equities_base[n] = (eq_b, dd_b)

        mask_o = (pnl_opt.index >= pd.Timestamp(s)) & (pnl_opt.index <= pd.Timestamp(e))
        seg_o = pnl_opt[mask_o]
        m_o, eq_o, dd_o = calc_metrics(seg_o)
        results_opt[n] = m_o
        equities_opt[n] = (eq_o, dd_o)

        print(f"\n[{n}]")
        print(f"  Baseline Model : CAGR {m_b['CAGR']:>6.1%}, Sharpe {m_b['Sharpe']:>4.2f}, MaxDD {m_b['MaxDD']:>6.1%}, Return {m_b['Return']:>7.1%}")
        print(f"  Option Model   : CAGR {m_o['CAGR']:>6.1%}, Sharpe {m_o['Sharpe']:>4.2f}, MaxDD {m_o['MaxDD']:>6.1%}, Return {m_o['Return']:>7.1%}")

    # Plot Comparison Chart
    print("\nPlotting comparison chart...")
    fig = plt.figure(figsize=(14, 12))
    gs = gridspec.GridSpec(2, 1, height_ratios=[1.8, 1.0], hspace=0.25)

    colors_base = {'Train 2022-2024':'#888888', 'Test  2025-2026':'#777777', 'Full  2022-2026':'#aaaaaa'}
    colors_opt  = {'Train 2022-2024':'#1f77b4', 'Test  2025-2026':'#2ca02c', 'Full  2022-2026':'#d62728'}

    ax1 = fig.add_subplot(gs[0])
    # Full curves
    eq_b_full, _ = equities_base['Full  2022-2026']
    eq_o_full, _ = equities_opt['Full  2022-2026']
    
    ax1.plot(eq_b_full.index, eq_b_full.values, color='#888888', ls='--', lw=1.8, label='Baseline Model (No Options)')
    ax1.plot(eq_o_full.index, eq_o_full.values, color='#d62728', lw=2.5, label='Option-Enhanced Model (Global Feature)')
    
    ax1.axhline(1, color='#8b949e', ls='--', lw=0.8, alpha=0.6)
    ax1.set_title('Stock Model Stacking Comparison: Baseline vs Option-Enhanced Model', fontsize=14, pad=12, fontweight='bold')
    ax1.set_ylabel('Cumulative Equity', fontsize=12)
    ax1.legend(loc='upper left', fontsize=11)
    ax1.grid(True, alpha=0.2)

    ax2 = fig.add_subplot(gs[1])
    _, dd_b_full = equities_base['Full  2022-2026']
    _, dd_o_full = equities_opt['Full  2022-2026']
    
    ax2.fill_between(dd_b_full.index, dd_b_full.values, 0, color='#888888', alpha=0.15)
    ax2.plot(dd_b_full.index, dd_b_full.values, color='#888888', ls='--', lw=1.0)
    ax2.fill_between(dd_o_full.index, dd_o_full.values, 0, color='#d62728', alpha=0.25)
    ax2.plot(dd_o_full.index, dd_o_full.values, color='#d62728', lw=1.2, label='Option Model Drawdown')
    
    ax2.set_title('Drawdown Comparison', fontsize=11)
    ax2.set_ylabel('Drawdown (%)', fontsize=12)
    ax2.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y:.0%}'))
    ax2.grid(True, alpha=0.2)
    ax2.legend(loc='lower left')

    # Visual Theme
    for ax in [ax1, ax2]:
        ax.set_facecolor('#0d1117')
        ax.tick_params(colors='#8b949e')
        for s in ['bottom','top','left','right']:
            ax.spines[s].set_color('#30363d')
    fig.patch.set_facecolor('#161b22')

    chart_out = os.path.join(RESULTS_DIR, 'model_options_comparison.png')
    plt.savefig(chart_out, dpi=150, bbox_inches='tight', facecolor='#161b22')
    plt.close()
    print(f"Comparison chart saved to: {chart_out}")

    # Save metrics to CSV
    metrics_data = []
    for n in ['Train 2022-2024', 'Test  2025-2026', 'Full  2022-2026']:
        m_b = results_base[n]
        m_o = results_opt[n]
        metrics_data.append({
            'Period': n,
            'Base_Return (%)': m_b['Return'] * 100,
            'Base_CAGR (%)': m_b['CAGR'] * 100,
            'Base_Sharpe': m_b['Sharpe'],
            'Base_MaxDD (%)': m_b['MaxDD'] * 100,
            'Opt_Return (%)': m_o['Return'] * 100,
            'Opt_CAGR (%)': m_o['CAGR'] * 100,
            'Opt_Sharpe': m_o['Sharpe'],
            'Opt_MaxDD (%)': m_o['MaxDD'] * 100,
        })
    df_met = pd.DataFrame(metrics_data)
    csv_out = os.path.join(RESULTS_DIR, 'model_comparison_metrics.csv')
    df_met.to_csv(csv_out, index=False)
    print(f"Comparison metrics CSV saved to: {csv_out}")

if __name__ == '__main__':
    main()

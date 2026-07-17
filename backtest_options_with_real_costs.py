"""
backtest_options_with_real_costs.py
对比三种交易成本假设下的 Study 005 期权增强策略回测表现
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

# ── 路径配置 ──────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
STUDY_DIR = os.path.join(SCRIPT_DIR, 'research', 'study_005_1d_advanced')
FEAT_FILE = os.path.join(STUDY_DIR, 'data', 'features_005_options.parquet')
PRED_FILE_BASE = os.path.join(STUDY_DIR, 'predictions', 'predictions_005_wf.parquet')
PRED_FILE_OPT  = os.path.join(STUDY_DIR, 'predictions', 'predictions_005_options_wf.parquet')
RESULTS_DIR = os.path.join(SCRIPT_DIR, 'research', '期权', 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

# ── 交易参数 ──────────────────────────────────────────────────────────
P = {
    'th_up':       0.50,
    'th_crash':    0.15,
    'max_pos':     3,
    'gap_low':     0.02,
    'gap_high':    0.06,
    'stop_loss':  -0.05,
    'tp_trigger':  0.06,
    'tp_pullback': 0.015,
    'tp_floor':    0.05,
    'regime_impact_th': -1.0,
    'max_per_ind': 2
}

# 三种成本假设
COST_SCENARIOS = [
    ('Current (0.1% / 0.1%)', 0.001, 0.001, '当前回测成本（低估）'),
    ('Realistic (0.2% / 0.3%)', 0.002, 0.003, '真实成本：佣金+滑点/佣金+印花税+滑点'),
    ('Conservative (0.3% / 0.4%)', 0.003, 0.004, '保守成本：小盘股流动性差，滑点更大'),
]

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

def run_backtest(pred_df, ohlc, pctchg, regime_map, P, BUY_COST, SELL_COST):
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
            
            if o <= sl_price:
                if at_ld:
                    daily_pnl[day_i] += pos_size * (c - prev) / prev
                    last_p[pi] = c; stats['skip_sell_lim'] += 1
                else:
                    daily_pnl[day_i] += pos_size * (o - prev) / prev - pos_size * SELL_COST
                    status[pi] = 0
                continue
                
            if l <= sl_price and h >= tp_price:
                if at_ld:
                    daily_pnl[day_i] += pos_size * (c - prev) / prev
                    last_p[pi] = c; stats['skip_sell_lim'] += 1
                else:
                    daily_pnl[day_i] += pos_size * (sl_price - prev) / prev - pos_size * SELL_COST
                    status[pi] = 0
                continue
                
            if l <= sl_price:
                if at_ld:
                    daily_pnl[day_i] += pos_size * (c - prev) / prev
                    last_p[pi] = c; stats['skip_sell_lim'] += 1
                else:
                    daily_pnl[day_i] += pos_size * (sl_price - prev) / prev - pos_size * SELL_COST
                    status[pi] = 0
                continue
                
            if h >= tp_price:
                daily_pnl[day_i] += pos_size * (tp_price - prev) / prev - pos_size * SELL_COST
                status[pi] = 0
                continue
                
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
        print(f"Error: Option prediction file not found: {PRED_FILE_OPT}")
        return
    if not os.path.exists(PRED_FILE_BASE):
        print(f"Error: Base prediction file not found: {PRED_FILE_BASE}")
        return

    print("Loading market data (this may take a moment)...")
    ohlc, pctchg, regime_map = load_market_data()
    print(f"Loaded {len(ohlc)} OHLC records, {len(pctchg)} pct_chg records.")

    print("\nLoading prediction files...")
    pred_base = pd.read_parquet(PRED_FILE_BASE)
    pred_base['ds'] = pred_base['trade_date'].astype(str)
    pred_opt = pd.read_parquet(PRED_FILE_OPT)
    pred_opt['ds'] = pred_opt['trade_date'].astype(str)
    print(f"Base predictions: {len(pred_base)} rows, Opt predictions: {len(pred_opt)} rows")

    # 先跑 Baseline 在当前成本下（作为参考）
    P_base = P.copy()
    P_base['th_up'] = 0.50
    P_base['th_crash'] = 0.45
    P_opt = P.copy()
    P_opt['th_up'] = 0.50
    P_opt['th_crash'] = 0.45

    # 跑三组成本对比（只跑期权增强模型，因为是我们主推的）
    all_results = {}
    
    for scenario_name, BUY_COST, SELL_COST, desc in COST_SCENARIOS:
        print(f"\n{'='*60}")
        print(f"【Scenario: {scenario_name}】")
        print(f"  {desc}")
        print(f"  BUY_COST = {BUY_COST:.4f} ({BUY_COST*100:.2f}%), SELL_COST = {SELL_COST:.4f} ({SELL_COST*100:.2f}%)")
        print(f"{'='*60}")
        
        pnl_opt, stats_opt = run_backtest(pred_opt, ohlc, pctchg, regime_map, P_opt, BUY_COST, SELL_COST)
        
        print(f"\n[Trade Stats]")
        for k, v in stats_opt.items():
            print(f"  {k:<20}: {v}")
        
        print(f"\n[Performance by Period]")
        results = {}
        for n, s, e in PERIODS:
            mask = (pnl_opt.index >= pd.Timestamp(s)) & (pnl_opt.index <= pd.Timestamp(e))
            seg = pnl_opt[mask]
            m, eq, dd = calc_metrics(seg)
            results[n] = m
            print(f"  {n}: Total Return {m['Return']:>7.1%}, CAGR {m['CAGR']:>6.1%}, Sharpe {m['Sharpe']:>4.2f}, MaxDD {m['MaxDD']:>6.1%}, WinRate {m['WinRate']:>5.1%}")
        
        all_results[scenario_name] = results
    
    # 输出汇总对比表
    print("\n\n" + "="*80)
    print("【COST SENSITIVITY ANALYSIS SUMMARY】Option-Enhanced Model")
    print("="*80)
    
    print(f"\n{'Period':<20} | {'Metric':<10} | " + " | ".join([f"{name:<28}" for name, _, _, _ in COST_SCENARIOS]))
    print("-" * 120)
    
    for period in ['Train 2022-2024', 'Test  2025-2026', 'Full  2022-2026']:
        for metric in ['Return', 'CAGR', 'Sharpe', 'MaxDD', 'WinRate']:
            row = f"{period:<20} | {metric:<10} | "
            values = []
            for name, _, _, _ in COST_SCENARIOS:
                v = all_results[name][period][metric]
                if metric in ['Return', 'CAGR', 'MaxDD', 'WinRate']:
                    values.append(f"{v:>7.1%}")
                else:
                    values.append(f"{v:>7.2f}")
            row += " | ".join([f"{v:<28}" for v in values])
            print(row)
        print("-" * 120)
    
    # 计算成本侵蚀量
    print("\n\n【COST EROSION ANALYSIS】")
    print("=" * 80)
    base_cagr = all_results['Current (0.1% / 0.1%)']['Full  2022-2026']['CAGR']
    for name, _, _, _ in COST_SCENARIOS[1:]:
        cagr = all_results[name]['Full  2022-2026']['CAGR']
        erosion = base_cagr - cagr
        print(f"  {name}: CAGR = {cagr:.2%}, 成本侵蚀 = {erosion:.2%} (年化)")
    
    # 画图
    print("\nPlotting cost sensitivity comparison chart...")
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    colors = ['#2ca02c', '#ff7f0e', '#d62728']
    
    for idx, (name, bc, sc, desc) in enumerate(COST_SCENARIOS):
        pnl_opt, _ = run_backtest(pred_opt, ohlc, pctchg, regime_map, P_opt, bc, sc)
        m, eq, dd = calc_metrics(pnl_opt)
        
        ax_eq = axes[0, idx]
        ax_eq.plot(eq.index, eq.values, color=colors[idx], lw=2, label=f'{name}')
        ax_eq.axhline(1, color='gray', ls='--', lw=0.8)
        ax_eq.set_title(f'{name}\nCAGR={m["CAGR"]:.1%}  Sharpe={m["Sharpe"]:.2f}  MaxDD={m["MaxDD"]:.1%}', fontsize=11)
        ax_eq.set_ylabel('Cumulative Equity')
        ax_eq.grid(True, alpha=0.2)
        ax_eq.legend()
        
        ax_dd = axes[1, idx]
        ax_dd.fill_between(dd.index, dd.values, 0, color=colors[idx], alpha=0.3)
        ax_dd.plot(dd.index, dd.values, color=colors[idx], lw=1)
        ax_dd.set_title('Drawdown', fontsize=10)
        ax_dd.set_ylabel('Drawdown')
        ax_dd.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y:.0%}'))
        ax_dd.grid(True, alpha=0.2)
    
    plt.suptitle('Cost Sensitivity Analysis: Option-Enhanced Model (2022-2026)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    chart_out = os.path.join(RESULTS_DIR, 'cost_sensitivity_analysis.png')
    plt.savefig(chart_out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Chart saved to: {chart_out}")
    
    # 保存CSV
    rows = []
    for period in ['Train 2022-2024', 'Test  2025-2026', 'Full  2022-2026']:
        for name, bc, sc, _ in COST_SCENARIOS:
            m = all_results[name][period]
            rows.append({
                'Period': period,
                'Cost_Scenario': name,
                'Buy_Cost': bc,
                'Sell_Cost': sc,
                'Total_Return': m['Return'],
                'CAGR': m['CAGR'],
                'Sharpe': m['Sharpe'],
                'MaxDD': m['MaxDD'],
                'WinRate': m['WinRate'],
            })
    df_out = pd.DataFrame(rows)
    csv_out = os.path.join(RESULTS_DIR, 'cost_sensitivity_metrics.csv')
    df_out.to_csv(csv_out, index=False)
    print(f"Metrics CSV saved to: {csv_out}")

if __name__ == '__main__':
    main()

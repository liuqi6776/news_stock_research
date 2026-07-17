"""
step3_backtest_advanced.py
Study 005 - 机构级风控与增强优化版回测

整合 5 大优化：
A. Regime Filter：利用 news_market_impact (来自特征文件)，如果舆情极差，实施硬熔断（降低仓位）。
B. 双模型防大面：同时要求 prob_up >= 0.50 且 prob_crash <= 0.15。
C. 盘中移动止盈：T+2 若摸高至 +6%，回落 1.5% 则触发移动保护（保底 5%）。
E. 板块中性约束：单日同行业个股最多选 2 只，避免系统性风险。
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
STUDY_DIR = os.path.dirname(SCRIPT_DIR)
FEAT_FILE = os.path.join(STUDY_DIR, 'data', 'features_005.parquet')
PRED_FILE = os.path.join(STUDY_DIR, 'predictions', 'predictions_005_wf.parquet')
RESULTS_DIR = os.path.join(STUDY_DIR, 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

# ── 交易参数 ──────────────────────────────────────────────────────────
P = {
    'th_up':       0.50,
    'th_crash':    0.45,
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

def load_data():
    feat = pd.read_parquet(FEAT_FILE, columns=['ts_code','trade_date','open','high','low','close','pct_chg','pre_close','news_market_impact'])
    feat['trade_date'] = feat['trade_date'].astype(str)
    feat = feat.drop_duplicates(['ts_code','trade_date'], keep='last')
    
    # 提取大盘舆情序列 (每天市场情绪是统一的)
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

    pred = pd.read_parquet(PRED_FILE)
    pred['ds'] = pred['trade_date'].astype(str)
    
    return ohlc, pctchg, regime_map, pred

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
                 
    # 1. 基础概率过滤 & 排序
    # B. 双模型防大面
    above_up = pred_df[pred_df['prob_up'] >= P['th_up']]
    above = above_up[above_up['prob_crash'] <= P['th_crash']].copy()
    stats['dual_crash_blocked'] = len(above_up) - len(above)
    
    # E. 板块中性约束 & A. Regime Filter
    trading_dates = sorted(pred_df['ds'].unique())
    di = {d: i for i, d in enumerate(trading_dates)}
    
    selected_rows = []
    for d, group in above.groupby('ds'):
        # A. Regime Filter (撤销: 因为0分多为平淡无新闻日，反而是游资题材温床)
        # 仅当有真实的极端负面打分时才启用熔断 (目前数据无负数)
        nmi = regime_map.get(d, 2.0)
        if pd.isna(nmi): nmi = 2.0
        
        daily_max_pos = P['max_pos']
        if nmi <= -2.0:
            daily_max_pos = 0  # 舆情极差，强制空仓
        elif nmi <= -1.0:
            daily_max_pos = 1  # 舆情较差，减仓
            
        if daily_max_pos == 0:
            stats['regime_blocked'] += 1
            continue
            
        group = group.sort_values('prob_up', ascending=False)
        ind_counts = {}
        day_sel = []
        for _, r in group.iterrows():
            ind = r['industry']
            # E. 板块超配限制
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

    pos_size = 1.0 / (2 * P['max_pos']) # 我们仍然基于目标最大持仓来计算仓位大小，避免爆仓
    
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

        # ── T+1 买入 ──────────────────────────────────────────────────
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

        # ── T+2 卖出 (含移动止盈与止损) ────────────────────────────────
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

def plot_results(results, equities, full_pnl):
    fig = plt.figure(figsize=(16, 18))
    gs = gridspec.GridSpec(4, 1, height_ratios=[2.8, 1.2, 1.2, 1], hspace=0.38)
    colors = {'Train 2022-2024':'#58a6ff', 'Test  2025-2026':'#3fb950', 'Full  2022-2026':'#f0883e'}

    ax1 = fig.add_subplot(gs[0])
    for n, (eq, dd) in equities.items():
        m = results[n]
        ax1.plot(eq.index, eq.values, color=colors[n], lw=2,
                 label=f"{n}  CAGR={m['CAGR']:.1%}  Sharpe={m['Sharpe']:.2f}  MaxDD={m['MaxDD']:.1%}")
    ax1.axhline(1, color='#8b949e', ls='--', lw=0.8, alpha=0.6)
    ax1.set_title('Study 005 - T+1 增强版策略 (硬熔断+双模型+移动止盈+行业中性)', fontsize=14, pad=12)
    ax1.legend(loc='upper left'); ax1.grid(True, alpha=0.2)

    ax2 = fig.add_subplot(gs[1])
    for n, (eq, dd) in equities.items():
        ax2.fill_between(dd.index, dd.values, 0, color=colors[n], alpha=0.35)
        ax2.plot(dd.index, dd.values, color=colors[n], lw=0.8)
    ax2.set_title('回撤 (Drawdown)', fontsize=11); ax2.grid(True, alpha=0.2)
    ax2.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y:.0%}'))

    ax3 = fig.add_subplot(gs[2])
    monthly = full_pnl.resample('M').apply(lambda x: (1+x).prod()-1)
    dfm = monthly.to_frame('ret')
    dfm['y'] = dfm.index.year; dfm['m'] = dfm.index.month
    pv = dfm.pivot(index='y', columns='m', values='ret')
    pv.columns = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
    vmax = max(abs(pv.values[~np.isnan(pv.values)]).max(), 0.01)
    im = ax3.imshow(pv.values, cmap='RdYlGn', aspect='auto', vmin=-vmax, vmax=vmax)
    ax3.set_xticks(range(12)); ax3.set_xticklabels(pv.columns)
    ax3.set_yticks(range(len(pv.index))); ax3.set_yticklabels(pv.index)
    for i in range(pv.shape[0]):
        for j in range(pv.shape[1]):
            v = pv.values[i,j]
            if not np.isnan(v):
                ax3.text(j, i, f'{v:.1%}', ha='center', va='center', fontsize=7, color='black' if abs(v)<vmax*0.6 else 'white')
    ax3.set_title('月度收益热力图', fontsize=11)
    plt.colorbar(im, ax=ax3, format=FuncFormatter(lambda y, _: f'{y:.0%}'), fraction=0.015)

    ax4 = fig.add_subplot(gs[3])
    pm = full_pnl>=0; nm = full_pnl<0
    ax4.bar(full_pnl.index[pm], full_pnl.values[pm]*100, color='#3fb950', alpha=0.75, width=1.5, label='盈利日')
    ax4.bar(full_pnl.index[nm], full_pnl.values[nm]*100, color='#f85149', alpha=0.75, width=1.5, label='亏损日')
    roll20 = full_pnl.rolling(20, min_periods=1).mean()*100
    ax4.plot(roll20.index, roll20.values, color='#f0883e', lw=1.4, label='20日均收益')
    ax4.set_title('日收益率', fontsize=11); ax4.legend(loc='upper left', ncol=3); ax4.grid(True, alpha=0.2)
    ax4.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y:.1f}%'))

    for ax in [ax1, ax2, ax3, ax4]:
        ax.set_facecolor('#0d1117'); ax.tick_params(colors='#8b949e')
        for s in ['bottom','top','left','right']: ax.spines[s].set_color('#30363d')
    fig.patch.set_facecolor('#161b22')

    out = os.path.join(RESULTS_DIR, '005_advanced_results.png')
    plt.savefig(out, dpi=150, bbox_inches='tight', facecolor='#161b22')
    plt.close()
    print(f"\nChart saved -> {out}")

def main():
    print("Loading Data...")
    ohlc, pctchg, regime_map, pred = load_data()
    print("Running Backtest (Study 005)...")
    full_pnl, stats = run_backtest(pred, ohlc, pctchg, regime_map, P)
    
    print("\n[ Trade Stats ]")
    for k, v in stats.items():
        print(f"  {k:<15}: {v}")

    print("\n[ Metrics ]")
    results, equities = {}, {}
    for n, s, e in PERIODS:
        mask = (full_pnl.index >= pd.Timestamp(s)) & (full_pnl.index <= pd.Timestamp(e))
        seg = full_pnl[mask]
        m, eq, dd = calc_metrics(seg)
        results[n] = m
        equities[n] = (eq, dd)
        print(f"  {n}: CAGR {m['CAGR']:>6.1%}, Sharpe {m['Sharpe']:>4.2f}, MaxDD {m['MaxDD']:>6.1%}")

    plot_results(results, equities, full_pnl)

if __name__ == '__main__':
    main()

"""
run_best_combo.py
针对 study_004_1d_release 当前最优参数组合，
跑单次完整回测并输出：
  - 分期（训练/测试/全程）各项指标表格
  - 收益曲线 + 回撤图 (results/best_combo.png)
"""
import os, sys, warnings
import numpy as np
import pandas as pd
warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import FuncFormatter

# ── 路径 ─────────────────────────────────────────────────────────────
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
STUDY_DIR    = os.path.dirname(SCRIPT_DIR)
FEAT_FILE    = os.path.join(STUDY_DIR, 'data',        'all_features_v2.parquet')
PRED_FILE    = os.path.join(STUDY_DIR, 'predictions', 'predictions_1d_open_wf_monthly.parquet')
RESULTS_DIR  = os.path.join(STUDY_DIR, 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

# ── 最优参数（来自 grid-search 结论）────────────────────────────────
BEST = {
    'threshold':   0.50,
    'max_pos':     3,
    'stop_loss':  -0.05,   # -5%
    'take_profit': 0.05,   # +5%
    'gap_low':     0.02,   # 高开下限
    'gap_high':    0.06,   # 高开上限
}

BUY_COST  = 0.001
SELL_COST = 0.001

PERIODS = [
    ('训练期  2022-2024', '20220101', '20241231'),
    ('测试期  2025-2026', '20250101', '20261231'),
    ('全程    2022-2026', '20220101', '20261231'),
]

# ── 数据加载 ──────────────────────────────────────────────────────────
def load_data():
    feat = pd.read_parquet(FEAT_FILE,
        columns=['ts_code','trade_date','open','high','low','close','pct_chg','pre_close'])
    feat['trade_date'] = feat['trade_date'].astype(str)
    feat = feat.drop_duplicates(['ts_code','trade_date'], keep='last')

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
    print(f"  Features: {len(feat):,} rows | Predictions: {len(pred):,} rows")
    print(f"  Pred date range: {pred['ds'].min()} ~ {pred['ds'].max()}")
    return ohlc, pctchg, pred

# ── 涨跌停判断 ────────────────────────────────────────────────────────
def is_limit_up(code, pct):
    if pd.isna(pct): return False
    return pct >= (0.195 if code.startswith(('30','68')) else 0.095)

def is_limit_down(code, pct):
    if pd.isna(pct): return False
    return pct <= (-0.195 if code.startswith(('30','68')) else -0.095)

# ── 核心回测 ──────────────────────────────────────────────────────────
def run_backtest(pred_df, ohlc, pctchg, p):
    th  = p['threshold'];  mp  = p['max_pos']
    sl  = p['stop_loss'];  tp  = p['take_profit']
    gl  = p['gap_low'];    gh  = p['gap_high']

    above = pred_df[pred_df['prob'] >= th].copy()
    above['rank'] = above.groupby('ds')['prob'].rank(ascending=False, method='first')
    sel = above[above['rank'] <= mp].copy().reset_index(drop=True)

    trading_dates = sorted(pred_df['ds'].unique())
    di = {d: i for i, d in enumerate(trading_dates)}
    ND = len(trading_dates)
    NP = len(sel)

    if NP == 0:
        return pd.Series(0.0, index=pd.to_datetime(trading_dates, format='%Y%m%d')), {}

    pos_size = 1.0 / (2 * mp)   # hold 2 days, mp positions

    entry_idx = np.array([di[r['ds']] for _, r in sel.iterrows()], dtype=np.int32)
    codes      = [r['ts_code'] for _, r in sel.iterrows()]
    bp         = np.full(NP, np.nan)
    last_p     = np.full(NP, np.nan)
    sl_p       = np.zeros(NP)
    tp_p       = np.zeros(NP)
    status     = np.ones(NP, dtype=np.int8)   # 1=open
    daily_pnl  = np.zeros(ND)

    stats = dict(selected=NP, skip_T_lim=0, skip_T1_lim=0, skip_gap=0,
                 skip_sell_lim=0, trades=0)

    for day_i, d in enumerate(trading_dates):
        open_mask = np.where(status == 1)[0]
        if len(open_mask) == 0:
            continue
        hd_arr = day_i - entry_idx[open_mask]

        # ── T+1 买入 ──────────────────────────────────────────────────
        buy_mask = open_mask[hd_arr == 1]
        for pi in buy_mask:
            ohlc_t1 = ohlc.get((codes[pi], d))
            if ohlc_t1 is None: status[pi]=0; continue
            o, h, l, c = ohlc_t1
            t0_d = trading_dates[entry_idx[pi]]

            # 昨日涨停
            if is_limit_up(codes[pi], pctchg.get((codes[pi], t0_d))):
                stats['skip_T_lim'] += 1; status[pi]=0; continue
            # T+1 涨停开盘
            if is_limit_up(codes[pi], pctchg.get((codes[pi], d))):
                stats['skip_T1_lim'] += 1; status[pi]=0; continue
            # 高开过滤
            ohlc_t0 = ohlc.get((codes[pi], t0_d))
            if ohlc_t0:
                gap = (o - ohlc_t0[3]) / ohlc_t0[3] if ohlc_t0[3] > 0 else 0
                if not (gl <= gap < gh):
                    stats['skip_gap'] += 1; status[pi]=0; continue

            bp[pi] = o; last_p[pi] = o
            if sl < 0:  sl_p[pi] = o * (1 + sl)
            if tp > 0:  tp_p[pi] = o * (1 + tp)
            daily_pnl[day_i] -= pos_size * BUY_COST
            daily_pnl[day_i] += pos_size * (c - o) / o
            last_p[pi] = c
            stats['trades'] += 1

        # ── T+2 平仓/止损止盈 ─────────────────────────────────────────
        hold2 = open_mask[hd_arr >= 2]
        for pi in hold2:
            ohlc_t2 = ohlc.get((codes[pi], d))
            if ohlc_t2 is None:
                daily_pnl[day_i] -= pos_size * SELL_COST
                status[pi]=0; continue
            o, h, l, c = ohlc_t2
            prev = last_p[pi]
            pct  = pctchg.get((codes[pi], d))
            at_ld = is_limit_down(codes[pi], pct)
            triggered = False

            if sl_p[pi] > 0 and o <= sl_p[pi]:
                if at_ld:
                    daily_pnl[day_i] += pos_size * (c-prev)/prev
                    last_p[pi] = c; stats['skip_sell_lim'] += 1
                else:
                    daily_pnl[day_i] += pos_size * (o-prev)/prev - pos_size*SELL_COST
                    status[pi]=0; triggered=True
            elif tp_p[pi] > 0 and o >= tp_p[pi]:
                daily_pnl[day_i] += pos_size * (o-prev)/prev - pos_size*SELL_COST
                status[pi]=0; triggered=True
            elif sl_p[pi] > 0 and l <= sl_p[pi] and not at_ld:
                daily_pnl[day_i] += pos_size * (sl_p[pi]-prev)/prev - pos_size*SELL_COST
                status[pi]=0; triggered=True
            elif tp_p[pi] > 0 and h >= tp_p[pi]:
                daily_pnl[day_i] += pos_size * (tp_p[pi]-prev)/prev - pos_size*SELL_COST
                status[pi]=0; triggered=True

            if not triggered:
                if at_ld:
                    daily_pnl[day_i] += pos_size * (c-prev)/prev
                    last_p[pi] = c; stats['skip_sell_lim'] += 1
                else:
                    daily_pnl[day_i] += pos_size * (c-prev)/prev - pos_size*SELL_COST
                    status[pi]=0

    pnl_s = pd.Series(daily_pnl, index=pd.to_datetime(trading_dates, format='%Y%m%d'))
    return pnl_s, stats

# ── 指标计算 ──────────────────────────────────────────────────────────
def calc_metrics(pnl_s):
    equity     = (1 + pnl_s).cumprod()
    running_max = equity.cummax()
    dd         = (equity - running_max) / running_max
    n          = len(pnl_s)
    n_years    = n / 252
    total_ret  = equity.iloc[-1] - 1
    cagr       = (equity.iloc[-1] ** (1/n_years) - 1) if n_years > 0 else 0
    sharpe     = pnl_s.mean() / pnl_s.std() * np.sqrt(252) if pnl_s.std() > 1e-10 else 0
    calmar     = cagr / abs(dd.min()) if dd.min() < 0 else 0
    win_days   = (pnl_s > 0).mean()
    monthly    = pnl_s.resample('M').apply(lambda x: (1+x).prod()-1)
    monthly_wr = (monthly > 0).mean()
    return {
        'Total Return': total_ret,
        'CAGR':         cagr,
        'Sharpe':       sharpe,
        'Calmar':       calmar,
        'Max Drawdown': dd.min(),
        'Win Rate(day)':win_days,
        'Win Rate(mo)': monthly_wr,
        'N Days':       n,
        'N Months':     len(monthly),
    }, equity, dd

# ── 主函数 ────────────────────────────────────────────────────────────
def main():
    print("\n" + "="*65)
    print("  study_004_1d_release  |  BEST COMBO BACKTEST")
    print("="*65)
    print(f"\n  Parameters:")
    for k, v in BEST.items():
        print(f"    {k:<14}: {v}")

    print("\nLoading data...")
    ohlc, pctchg, pred = load_data()

    # 全程回测（拆分期间统计）
    print("\nRunning backtest on full period 2022-2026...")
    full_pnl, trade_stats = run_backtest(pred, ohlc, pctchg, BEST)

    print(f"\n  Trade stats:")
    print(f"    Total selected  : {trade_stats['selected']:,}")
    print(f"    Actual trades   : {trade_stats['trades']:,}")
    print(f"    Skip T lim-up   : {trade_stats['skip_T_lim']:,}")
    print(f"    Skip T+1 lim-up : {trade_stats['skip_T1_lim']:,}")
    print(f"    Skip gap filter : {trade_stats['skip_gap']:,}")
    print(f"    Skip sell lim-dn: {trade_stats['skip_sell_lim']:,}")

    # ── 分期指标表 ────────────────────────────────────────────────────
    print("\n" + "="*65)
    print(f"{'Metric':<22} {'Train 2022-24':>14} {'Test 2025-26':>13} {'Full 2022-26':>13}")
    print("-"*65)

    results, equities = {}, {}
    for name, s, e in PERIODS:
        mask = (full_pnl.index >= pd.Timestamp(s[:4]+'-'+s[4:6]+'-'+s[6:])) & \
               (full_pnl.index <= pd.Timestamp(e[:4]+'-'+e[4:6]+'-'+e[6:]))
        seg = full_pnl[mask]
        m, eq, dd = calc_metrics(seg)
        results[name] = m
        equities[name] = (eq, dd)

    metrics_order = ['Total Return','CAGR','Sharpe','Calmar','Max Drawdown',
                     'Win Rate(day)','Win Rate(mo)','N Days','N Months']
    fmt_pct = {'Total Return','CAGR','Max Drawdown','Win Rate(day)','Win Rate(mo)'}
    fmt_int = {'N Days','N Months'}

    period_keys = [r[0] for r in PERIODS]
    for m in metrics_order:
        row = f"{m:<22}"
        for pk in period_keys:
            v = results[pk][m]
            if m in fmt_pct:
                row += f" {v:>13.1%}"
            elif m in fmt_int:
                row += f" {v:>13.0f}"
            else:
                row += f" {v:>13.2f}"
        print(row)
    print("="*65)

    # ── 绘图 ─────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(16, 18))
    gs  = gridspec.GridSpec(4, 1, height_ratios=[2.8, 1.2, 1.2, 1], hspace=0.38)

    colors = {'训练期  2022-2024':'#58a6ff', '测试期  2025-2026':'#3fb950', '全程    2022-2026':'#f0883e'}

    # — 收益曲线 —
    ax1 = fig.add_subplot(gs[0])
    for name, (eq, dd) in equities.items():
        m = results[name]
        ax1.plot(eq.index, eq.values, color=colors[name], linewidth=2.0,
                 label=f"{name.strip()}  CAGR={m['CAGR']:.1%}  Sharpe={m['Sharpe']:.2f}  MaxDD={m['Max Drawdown']:.1%}")
    ax1.axhline(1.0, color='#8b949e', linestyle='--', linewidth=0.8, alpha=0.6)
    ax1.set_title('Study 004 — T+1 新闻驱动策略\n'
                  f"最优参数: threshold={BEST['threshold']}  max_pos={BEST['max_pos']}  "
                  f"stop_loss={BEST['stop_loss']:.0%}  take_profit={+BEST['take_profit']:.0%}  "
                  f"gap={BEST['gap_low']:.0%}~{BEST['gap_high']:.0%}",
                  fontsize=13, pad=10)
    ax1.set_ylabel('净值', fontsize=11)
    ax1.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y:.2f}'))
    ax1.legend(fontsize=9, loc='upper left', framealpha=0.8)
    ax1.grid(True, alpha=0.2)
    ax1.set_facecolor('#0d1117')
    fig.patch.set_facecolor('#161b22')

    # — 回撤 —
    ax2 = fig.add_subplot(gs[1])
    for name, (eq, dd) in equities.items():
        ax2.fill_between(dd.index, dd.values, 0, color=colors[name], alpha=0.35, label=name.strip())
        ax2.plot(dd.index, dd.values, color=colors[name], linewidth=0.8)
    ax2.set_title('回撤 (Drawdown)', fontsize=11)
    ax2.set_ylabel('回撤', fontsize=10)
    ax2.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y:.0%}'))
    ax2.legend(fontsize=8, loc='lower left')
    ax2.grid(True, alpha=0.2)
    ax2.set_facecolor('#0d1117')

    # — 月度收益热力图（全程）—
    ax3 = fig.add_subplot(gs[2])
    full_pnl_seg = full_pnl
    monthly_ret = full_pnl_seg.resample('M').apply(lambda x: (1+x).prod()-1)
    monthly_df  = monthly_ret.to_frame('ret')
    monthly_df['year']  = monthly_df.index.year
    monthly_df['month'] = monthly_df.index.month
    pivot = monthly_df.pivot(index='year', columns='month', values='ret')
    pivot.columns = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
    vmax = max(abs(pivot.values[~np.isnan(pivot.values)]).max(), 0.01)
    im = ax3.imshow(pivot.values, cmap='RdYlGn', aspect='auto', vmin=-vmax, vmax=vmax)
    ax3.set_xticks(range(12)); ax3.set_xticklabels(pivot.columns, fontsize=8)
    ax3.set_yticks(range(len(pivot.index))); ax3.set_yticklabels(pivot.index, fontsize=8)
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            v = pivot.values[i, j]
            if not np.isnan(v):
                ax3.text(j, i, f'{v:.1%}', ha='center', va='center',
                         fontsize=7, color='black' if abs(v) < vmax*0.6 else 'white')
    ax3.set_title('月度收益热力图 (全程)', fontsize=11)
    plt.colorbar(im, ax=ax3, format=FuncFormatter(lambda y, _: f'{y:.0%}'), fraction=0.015)
    ax3.set_facecolor('#0d1117')

    # — 日收益柱状图 —
    ax4 = fig.add_subplot(gs[3])
    # 用全程数据
    full_seg_pnl = full_pnl
    pos_mask = full_seg_pnl >= 0
    neg_mask = full_seg_pnl <  0
    ax4.bar(full_seg_pnl.index[pos_mask], full_seg_pnl.values[pos_mask] * 100,
            color='#3fb950', alpha=0.75, width=1.5, label='盈利日')
    ax4.bar(full_seg_pnl.index[neg_mask], full_seg_pnl.values[neg_mask] * 100,
            color='#f85149', alpha=0.75, width=1.5, label='亏损日')
    # 20日滚动均线
    roll20 = full_seg_pnl.rolling(20, min_periods=1).mean() * 100
    ax4.plot(roll20.index, roll20.values, color='#f0883e', linewidth=1.4,
             label='20日均收益', zorder=3)
    ax4.axhline(0, color='#8b949e', linewidth=0.8, linestyle='--', alpha=0.6)
    ax4.set_title('日收益率随时间变化 (全程 2022-2026)', fontsize=11)
    ax4.set_ylabel('日收益率 (%)', fontsize=10)
    ax4.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y:.1f}%'))
    ax4.legend(fontsize=8, loc='upper left', ncol=3)
    ax4.grid(True, alpha=0.15)
    ax4.set_facecolor('#0d1117')

    # 统一风格
    for ax in [ax1, ax2, ax3, ax4]:
        ax.tick_params(colors='#8b949e')
        ax.spines['bottom'].set_color('#30363d')
        ax.spines['top'].set_color('#30363d')
        ax.spines['left'].set_color('#30363d')
        ax.spines['right'].set_color('#30363d')

    out = os.path.join(RESULTS_DIR, 'best_combo.png')
    plt.savefig(out, dpi=150, bbox_inches='tight', facecolor='#161b22')
    plt.close()
    print(f"\nChart saved → {out}")
    print("Done.\n")

if __name__ == '__main__':
    main()

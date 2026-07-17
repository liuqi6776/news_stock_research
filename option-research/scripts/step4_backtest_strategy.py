"""
step4_backtest_strategy.py
基于最大痛点引力信号回测标的 ETF (50ETF & 300ETF) 交易策略。
回测对比：
1. 理论多空 (Long-Short) 策略
2. 实用多头 (Long-Only) 策略
3. 实用空头 (Short-Only) 策略
4. 基准买入持有 (Benchmark Buy & Hold) 策略

交易执行细节：
- 避免看前偏差 (Look-ahead bias)：在第 t 日收盘后生成交易信号，在第 t+1 日开盘时以开盘价执行交易，持仓至下一信号变更。
- 交易成本：双边计 0.05%（包含佣金和冲击成本，ETF免印花税）。
- 参数扫描：不同的到期前天数 (max_dte) 和进入偏差阈值 (threshold)。
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 目录配置
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(BASE_DIR, 'data')
PLOTS_DIR = os.path.join(BASE_DIR, 'plots')
os.makedirs(PLOTS_DIR, exist_ok=True)

FEATURE_FILE = os.path.join(DATA_DIR, 'max_pain_features.csv')
UNDERLYING_FILE = os.path.join(DATA_DIR, 'underlying_daily.parquet')
PERFORMANCE_CSV = os.path.join(DATA_DIR, 'backtest_performance.csv')

def calculate_performance_metrics(nav_series, benchmark_nav=None, risk_free_rate=0.02):
    """
    计算绩效指标：年化收益率、年化夏普比率、最大回撤、卡玛比率
    """
    daily_returns = nav_series.pct_change().dropna()
    if len(daily_returns) == 0:
        return {
            'cagr': 0.0, 'sharpe': 0.0, 'max_dd': 0.0, 'calmar': 0.0
        }
        
    n_days = len(nav_series)
    years = n_days / 242.0  # 每年按242个交易日计算
    
    # 年化收益率 (CAGR)
    cagr = (nav_series.iloc[-1] / nav_series.iloc[0]) ** (1.0 / years) - 1.0
    
    # 年化波动率
    vol = daily_returns.std() * np.sqrt(242.0)
    
    # 夏普比率 (Sharpe Ratio)
    mean_ret = daily_returns.mean() * 242.0
    sharpe = (mean_ret - risk_free_rate) / vol if vol != 0 else 0.0
    
    # 最大回撤 (Max Drawdown)
    cum_max = nav_series.cummax()
    drawdowns = (nav_series - cum_max) / cum_max
    max_dd = drawdowns.min()
    
    # 卡玛比率 (Calmar Ratio)
    calmar = cagr / abs(max_dd) if max_dd != 0 else 0.0
    
    return {
        'cagr': cagr,
        'vol': vol,
        'sharpe': sharpe,
        'max_dd': max_dd,
        'calmar': calmar
    }

def run_single_backtest(df_asset, max_dte, threshold, execution='open', cost_rate=0.0005):
    """
    对单个标的资产运行策略回测。
    execution: 'open' (次日开盘价执行) 或 'close' (当日收盘价执行)
    """
    df_sub = df_asset.sort_values('trade_date').reset_index(drop=True).copy()
    
    # 生成基础仓位信号 (S_t)
    # 信号生成规则：到期日前 max_dte 天内，如果价格偏离度超过 threshold，则触发交易
    df_sub['signal'] = 0
    
    # 偏离信号
    long_cond = (df_sub['days_to_expiry'] <= max_dte) & (df_sub['deviation'] < -threshold) & (df_sub['days_to_expiry'] > 0)
    short_cond = (df_sub['days_to_expiry'] <= max_dte) & (df_sub['deviation'] > threshold) & (df_sub['days_to_expiry'] > 0)
    
    df_sub.loc[long_cond, 'signal'] = 1
    df_sub.loc[short_cond, 'signal'] = -1
    
    # 到期日当天收盘必须强平，仓位归零
    df_sub.loc[df_sub['days_to_expiry'] == 0, 'signal'] = 0
    
    # 生成持仓仓位 (Position)
    if execution == 'open':
        # 次日开盘执行：第 t 日收盘产生的信号，在第 t+1 日开盘成交并持有一整天
        # 故第 t+1 日的收益由第 t 日的信号 signal_t 决定
        df_sub['position'] = df_sub['signal'].shift(1).fillna(0)
        
        # 计算第 t+1 日的日收益率（开盘到开盘）
        # R^{open}_{t+1} = (Open_{t+2} - Open_{t+1}) / Open_{t+1}
        # 为了保证数据对齐：
        df_sub['open_price'] = df_sub['open']
        df_sub['next_open'] = df_sub['open'].shift(-1)
        # 用 open 价格计算每日收益
        # 第 t 交易日的 open_return 是从 t 的开盘到 t+1 的开盘
        df_sub['daily_asset_return'] = (df_sub['next_open'] - df_sub['open_price']) / df_sub['open_price']
        
        # 对于最后一行，我们没有 next_open，直接用 close - open
        last_idx = df_sub.index[-1]
        df_sub.loc[last_idx, 'daily_asset_return'] = (df_sub.loc[last_idx, 'close'] - df_sub.loc[last_idx, 'open']) / df_sub.loc[last_idx, 'open']
        
    else:
        # 当日收盘执行：第 t 日收盘产生的信号直接交易，享受第 t+1 日的收盘价变动
        df_sub['position'] = df_sub['signal'].fillna(0)
        df_sub['daily_asset_return'] = df_sub['close'].pct_change().fillna(0.0)
        
    # 计算仓位变化以计收手续费
    df_sub['prev_position'] = df_sub['position'].shift(1).fillna(0)
    df_sub['trades'] = (df_sub['position'] - df_sub['prev_position']).abs()
    
    # 策略每日收益（扣除手续费前）
    # 策略的 position = 1 代表持有 ETF，-1 代表做空 ETF，0 代表空仓
    # 多空 (Long-Short)
    df_sub['ret_ls'] = df_sub['position'] * df_sub['daily_asset_return'] - df_sub['trades'] * cost_rate
    
    # 多头 (Long-Only)
    pos_long = np.where(df_sub['position'] > 0, 1.0, 0.0)
    trades_long = np.abs(pos_long - np.roll(pos_long, 1))
    trades_long[0] = pos_long[0] # 首日仓位建仓
    df_sub['ret_lo'] = pos_long * df_sub['daily_asset_return'] - trades_long * cost_rate
    
    # 空头 (Short-Only)
    pos_short = np.where(df_sub['position'] < 0, -1.0, 0.0)
    trades_short = np.abs(pos_short - np.roll(pos_short, 1))
    trades_short[0] = np.abs(pos_short[0])
    df_sub['ret_so'] = pos_short * df_sub['daily_asset_return'] - trades_short * cost_rate
    
    # 标的 Buy & Hold 收益率
    df_sub['ret_bm'] = df_sub['daily_asset_return']
    
    # 计算净值曲线 (NAV)
    df_sub['nav_ls'] = (1.0 + df_sub['ret_ls']).cumprod()
    df_sub['nav_lo'] = (1.0 + df_sub['ret_lo']).cumprod()
    df_sub['nav_so'] = (1.0 + df_sub['ret_so']).cumprod()
    df_sub['nav_bm'] = (1.0 + df_sub['ret_bm']).cumprod()
    
    return df_sub

def main():
    print(">>> Starting Strategy Backtesting Sweep...", flush=True)
    if not os.path.exists(FEATURE_FILE) or not os.path.exists(UNDERLYING_FILE):
        raise FileNotFoundError("Features or underlying daily files not found. Run step3 first.")
        
    df_feats = pd.read_csv(FEATURE_FILE)
    df_und = pd.read_parquet(UNDERLYING_FILE)
    
    # 转换为 string 以防合并类型冲突
    df_feats['trade_date'] = df_feats['trade_date'].astype(str)
    df_und['trade_date'] = df_und['trade_date'].astype(str)
    
    df_und = df_und.rename(columns={'ts_code': 'underlying_code'})
    
    df_merged = pd.merge(
        df_feats,
        df_und[['trade_date', 'underlying_code', 'open', 'high', 'low', 'close', 'vol', 'amount']],
        on=['trade_date', 'underlying_code'],
        how='inner'
    )
    
    # 参数扫描设置
    max_dtes = [3, 5, 10]
    thresholds = [0.0, 0.005, 0.01, 0.015]
    
    all_results = []
    
    for und_code in ['510050.SH', '510300.SH']:
        print(f"\n==================== Backtest Sweep for {und_code} ====================")
        df_asset = df_merged[df_merged['underlying_code'] == und_code].copy()
        
        for max_dte in max_dtes:
            for threshold in thresholds:
                # 运行回测（次日开盘执行）
                df_backtest = run_single_backtest(df_asset, max_dte, threshold, execution='open')
                
                # 计算指标
                m_ls = calculate_performance_metrics(df_backtest['nav_ls'])
                m_lo = calculate_performance_metrics(df_backtest['nav_lo'])
                m_so = calculate_performance_metrics(df_backtest['nav_so'])
                m_bm = calculate_performance_metrics(df_backtest['nav_bm'])
                
                # 统计交易次数
                total_trades = df_backtest['trades'].sum()
                
                print(f"Params: max_dte={max_dte:<2} thresh={threshold:.3f} | Trades: {total_trades:.1f}")
                print(f"  Long-Short | CAGR: {m_ls['cagr']*100:6.2f}% | Sharpe: {m_ls['sharpe']:.2f} | MaxDD: {m_ls['max_dd']*100:6.2f}%")
                print(f"  Long-Only  | CAGR: {m_lo['cagr']*100:6.2f}% | Sharpe: {m_lo['sharpe']:.2f} | MaxDD: {m_lo['max_dd']*100:6.2f}%")
                print(f"  Short-Only | CAGR: {m_so['cagr']*100:6.2f}% | Sharpe: {m_so['sharpe']:.2f} | MaxDD: {m_so['max_dd']*100:6.2f}%")
                print(f"  Benchmark  | CAGR: {m_bm['cagr']*100:6.2f}% | Sharpe: {m_bm['sharpe']:.2f} | MaxDD: {m_bm['max_dd']*100:6.2f}%")
                
                all_results.append({
                    'underlying_code': und_code,
                    'max_dte': max_dte,
                    'threshold': threshold,
                    'trades': total_trades,
                    'ls_cagr': m_ls['cagr'], 'ls_sharpe': m_ls['sharpe'], 'ls_max_dd': m_ls['max_dd'],
                    'lo_cagr': m_lo['cagr'], 'lo_sharpe': m_lo['sharpe'], 'lo_max_dd': m_lo['max_dd'],
                    'so_cagr': m_so['cagr'], 'so_sharpe': m_so['sharpe'], 'so_max_dd': m_so['max_dd'],
                    'bm_cagr': m_bm['cagr'], 'bm_sharpe': m_bm['sharpe'], 'bm_max_dd': m_bm['max_dd'],
                })
                
    df_perf = pd.DataFrame(all_results)
    df_perf.to_csv(PERFORMANCE_CSV, index=False)
    print(f"\nSaved performance sweep results to {PERFORMANCE_CSV}")
    
    # ==================== 4. 绘制最佳净值曲线 ====================
    # 寻找多空策略 Sharpe 最高的参数组合并绘图
    print("\n>>> Plotting Optimal Portfolio NAV Curves...")
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    
    for i, und_code in enumerate(['510050.SH', '510300.SH']):
        ax = axes[i]
        df_sub_perf = df_perf[df_perf['underlying_code'] == und_code]
        
        # 寻找多空策略 Sharpe 最高的一行
        best_row = df_sub_perf.loc[df_sub_perf['ls_sharpe'].idxmax()]
        best_max_dte = int(best_row['max_dte'])
        best_threshold = best_row['threshold']
        
        # 重新生成该最优回测序列
        df_asset = df_merged[df_merged['underlying_code'] == und_code].copy()
        df_opt = run_single_backtest(df_asset, best_max_dte, best_threshold, execution='open')
        
        # 转换 trade_date 为 datetime 以便画图
        df_opt['date_dt'] = pd.to_datetime(df_opt['trade_date'])
        
        ax.plot(df_opt['date_dt'], df_opt['nav_ls'], label=f"Optimal Long-Short (dte={best_max_dte}, thresh={best_threshold})", color='red', linewidth=2)
        ax.plot(df_opt['date_dt'], df_opt['nav_lo'], label=f"Optimal Long-Only", color='blue', linewidth=1.5)
        ax.plot(df_opt['date_dt'], df_opt['nav_so'], label=f"Optimal Short-Only", color='purple', linestyle='--', linewidth=1.2)
        ax.plot(df_opt['date_dt'], df_opt['nav_bm'], label=f"Benchmark Buy & Hold ({und_code})", color='gray', linestyle=':', linewidth=1.5)
        
        ax.set_title(f"{und_code} Optimal Strategy Performance (Based on Max Pain)", fontsize=13, fontweight='bold')
        ax.set_ylabel("Normalized NAV", fontsize=11)
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.legend(fontsize=10)
        
        # 打印详细指标
        print(f"\nOptimal params for {und_code}: max_dte={best_max_dte}, threshold={best_threshold}")
        print(f"  Long-Short Sharpe: {best_row['ls_sharpe']:.2f} | CAGR: {best_row['ls_cagr']*100:.2f}% | MaxDD: {best_row['ls_max_dd']*100:.2f}%")
        print(f"  Long-Only  Sharpe: {best_row['lo_sharpe']:.2f} | CAGR: {best_row['lo_cagr']*100:.2f}% | MaxDD: {best_row['lo_max_dd']*100:.2f}%")
        print(f"  Benchmark  Sharpe: {best_row['bm_sharpe']:.2f} | CAGR: {best_row['bm_cagr']*100:.2f}% | MaxDD: {best_row['bm_max_dd']*100:.2f}%")
        
    plt.xlabel("Date", fontsize=11)
    plt.suptitle("Strategy Performance Comparison (Next-Open Execution, Transaction Cost Included)", fontsize=15, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    plot_nav_path = os.path.join(PLOTS_DIR, 'max_pain_backtest.png')
    plt.savefig(plot_nav_path, dpi=300)
    plt.close()
    print(f"Saved optimal NAV plot to {plot_nav_path}")
    
if __name__ == '__main__':
    main()

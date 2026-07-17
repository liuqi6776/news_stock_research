"""
# step4_portfolio_backtest.py
# 高回真度A股组合回测系统（双策略对比版：纯多因子排序 vs 期权风控大闸）。
# 支持：
# 1. 真实交易摩擦：买入 0.2% (佣金+滑点)，卖出 0.3% (印花税+佣金+滑点)。
# 2. 交易限制过滤：涨停无法买入，跌停或停牌无法卖出（被迫持仓）。
# 3. 行业中性与分散化：单行业持仓比例上限（例如 20%）。
# 4. 20日定期调仓。
# 5. 每日 NAV 跟踪与可视化。
# 6. 期权风控大闸比较：对比开启与关闭期权大盘风控（QVIX Z-Score > 2.0 时清仓）的实际净值表现。
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(PROJECT_DIR, 'data')
PRED_DIR = os.path.join(PROJECT_DIR, 'predictions')
RESULTS_DIR = os.path.join(PROJECT_DIR, 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

PRED_FILE = os.path.join(PRED_DIR, 'predictions_longterm.parquet')
FEATURES_FILE = os.path.join(DATA_DIR, 'features_longterm.parquet')

# 策略参数
PORTFOLIO_SIZE = 100                 # 持仓股票数 (扩大为100以分散风险，充当Smart-Beta底仓)
HOLDING_DAYS = 20                  # 调仓周期 (20个交易日)
MAX_WEIGHT_PER_INDUSTRY = 0.20     # 单个行业最大权重 (20%)
INITIAL_CAPITAL = 1000000.0        # 初始资金 (100万)

# 交易成本
BUY_COST_RATE = 0.002              # 买入佣金+滑点 = 0.2%
SELL_COST_RATE = 0.003             # 卖出印花税+佣金+滑点 = 0.3%

# 期权大盘风控大闸配置
QVIX_PANIC_THRESHOLD = 2.0         # QVIX Z-Score 恐慌阈值
PCR_50_PANIC_THRESHOLD = 0.95      # 50ETF PCR 恐慌阈值 (已停用，仅作记录)

def is_limit_up(row, code=None):
    if code is None:
        code = row['ts_code']
    limit = 0.198 if (code.startswith('30') or code.startswith('68')) else 0.098
    return row['next_open'] >= row['close'] * (1 + limit)

def is_limit_down(row, code=None):
    if code is None:
        code = row['ts_code']
    limit = 0.198 if (code.startswith('30') or code.startswith('68')) else 0.098
    return row['next_open'] <= row['close'] * (1 - limit)

def simulate_strategy(df, df_by_date, trade_dates, rebalance_dates, use_options_filter=True):
    print(f"\n>>> Running Simulation (Options Filter: {use_options_filter}) ...", flush=True)
    
    current_holdings = {}  # {ts_code: {'val': val, 'buy_price': buy_price, 'is_first_day': bool}}
    cash = INITIAL_CAPITAL
    daily_navs = []
    
    for idx, dt in enumerate(trade_dates):
        if dt not in df_by_date:
            prev_nav = daily_navs[-1]['nav'] if daily_navs else INITIAL_CAPITAL
            daily_navs.append({'trade_date': dt, 'nav': prev_nav, 'cash': cash, 'holdings_val': 0})
            continue
            
        day_data = df_by_date[dt]
        day_prices = day_data.set_index('ts_code').to_dict(orient='index')
        
        # 1. 更新持仓股票市值
        holdings_val = 0
        if current_holdings:
            updated_holdings = {}
            for code, prev_item in current_holdings.items():
                prev_val = prev_item['val']
                if code in day_prices:
                    pct_chg = day_prices[code]['pct_chg']
                    if pd.isna(pct_chg):
                        pct_chg = 0.0
                    
                    if prev_item['is_first_day']:
                        # 买入首日：使用 收盘价 / 买入开盘价 乘以初始买入市值
                        close_price = day_prices[code]['close']
                        buy_price = prev_item['buy_price']
                        if buy_price > 0:
                            val = prev_val * (close_price / buy_price)
                        else:
                            val = prev_val * (1 + pct_chg)
                    else:
                        # 后续持仓日：使用 pct_chg (昨收 -> 今收) 更新
                        val = prev_val * (1 + pct_chg)
                        
                    updated_holdings[code] = {
                        'val': val,
                        'buy_price': None,
                        'is_first_day': False
                    }
                    holdings_val += val
                else:
                    # 停牌：市值保持不变，is_first_day 状态也保留
                    updated_holdings[code] = {
                        'val': prev_val,
                        'buy_price': prev_item['buy_price'],
                        'is_first_day': prev_item['is_first_day']
                    }
                    holdings_val += prev_val
            current_holdings = updated_holdings
            
        current_nav = cash + holdings_val
        daily_navs.append({
            'trade_date': dt,
            'nav': current_nav,
            'cash': cash,
            'holdings_val': holdings_val
        })
        
        # 2. 每日风控监测
        is_panic = False
        if use_options_filter:
            qvix_z = day_data['opt_qvix_zscore'].iloc[0] if len(day_data) > 0 else 0.0
            is_downtrend = day_data['downtrend'].iloc[0] if 'downtrend' in day_data.columns and len(day_data) > 0 else False
            if qvix_z > QVIX_PANIC_THRESHOLD or is_downtrend:
                is_panic = True
                        
        # 3. 定期调仓
        if dt in rebalance_dates:
            if idx + 1 >= len(trade_dates):
                continue
                
            next_dt = trade_dates[idx+1]
            if next_dt not in df_by_date:
                continue
                
            next_day_prices = df_by_date[next_dt].set_index('ts_code').to_dict(orient='index')
            
            # A. 统计被迫持仓
            sell_candidates = []
            forced_holdings = {}
            for code, prev_item in current_holdings.items():
                val = prev_item['val']
                untradeable = False
                if code not in next_day_prices:
                    untradeable = True
                else:
                    row = next_day_prices[code]
                    if row['vol'] == 0 or is_limit_down(row, code):
                        untradeable = True
                if untradeable:
                    forced_holdings[code] = {
                        'val': val,
                        'buy_price': None,
                        'is_first_day': False
                    }
                else:
                    sell_candidates.append(code)
            
            # B. 恐慌状态：不买新仓，只保留被迫持仓
            if is_panic:
                for code in sell_candidates:
                    val = current_holdings[code]['val']
                    cost = val * SELL_COST_RATE
                    cash += (val - cost)
                current_holdings = forced_holdings
                continue
                
            # C. 正常调仓：卖出可卖仓位
            for code in sell_candidates:
                val = current_holdings[code]['val']
                cost = val * SELL_COST_RATE
                cash += (val - cost)
                
            # 选股与买入
            candidates = day_data[day_data['pred_score'].notna()].copy()
            candidates = candidates.merge(
                df_by_date[next_dt][['ts_code', 'next_open', 'vol']], 
                on='ts_code', 
                suffixes=('', '_next')
            )
            candidates = candidates[
                (candidates['vol_next'] > 0) & 
                (~candidates.apply(lambda r: is_limit_up(r, r['ts_code']), axis=1))
            ]
            candidates = candidates.sort_values(by='pred_score', ascending=False)
            
            selected_codes = []
            industry_counts = {}
            max_stocks_per_industry = int(PORTFOLIO_SIZE * MAX_WEIGHT_PER_INDUSTRY)
            
            for code in forced_holdings.keys():
                ind = day_prices.get(code, {}).get('industry', 'Unknown')
                industry_counts[ind] = industry_counts.get(ind, 0) + 1
                
            for _, row in candidates.iterrows():
                if len(selected_codes) + len(forced_holdings) >= PORTFOLIO_SIZE:
                    break
                code = row['ts_code']
                if code in forced_holdings:
                    continue
                ind = row['industry']
                count = industry_counts.get(ind, 0)
                if count < max_stocks_per_industry:
                    selected_codes.append((code, row['next_open']))  # 保存代码与执行买价
                    industry_counts[ind] = count + 1
                    
            total_target_stocks = len(selected_codes) + len(forced_holdings)
            if total_target_stocks > 0:
                total_val = cash + sum(item['val'] for item in forced_holdings.values())
                new_holdings = {code: item for code, item in forced_holdings.items()}
                
                if len(selected_codes) > 0:
                    buy_value_per_stock = cash / len(selected_codes)
                    for code, buy_price in selected_codes:
                        cost = buy_value_per_stock * BUY_COST_RATE
                        new_holdings[code] = {
                            'val': buy_value_per_stock - cost,
                            'buy_price': buy_price,
                            'is_first_day': True
                        }
                    cash = 0.0
                current_holdings = new_holdings
            else:
                current_holdings = {code: item for code, item in forced_holdings.items()}
                
    nav_df = pd.DataFrame(daily_navs)
    nav_df['trade_date'] = pd.to_datetime(nav_df['trade_date'])
    nav_df = nav_df.set_index('trade_date')
    return nav_df

def compute_metrics(nav_series, name):
    years = len(nav_series) / 252.0
    final_nav = nav_series.iloc[-1]
    total_ret = final_nav / INITIAL_CAPITAL - 1
    cagr = (final_nav / INITIAL_CAPITAL) ** (1.0 / years) - 1
    
    daily_rets = nav_series.pct_change().dropna()
    ann_vol = daily_rets.std() * np.sqrt(252)
    sharpe = (daily_rets.mean() * 252) / ann_vol if ann_vol > 0 else 0
    
    cum_max = nav_series.cummax()
    drawdowns = (nav_series - cum_max) / cum_max
    max_dd = drawdowns.min()
    
    return {
        'Portfolio': name,
        'Total Return': f"{total_ret:.2%}",
        'CAGR': f"{cagr:.2%}",
        'Volatility': f"{ann_vol:.2%}",
        'Sharpe': f"{sharpe:.2f}",
        'Max Drawdown': f"{max_dd:.2%}",
        'final_nav_raw': final_nav,
        'cagr_raw': cagr,
        'vol_raw': ann_vol,
        'sharpe_raw': sharpe,
        'max_dd_raw': max_dd
    }

def run_backtest(pred_file=PRED_FILE, results_dir=RESULTS_DIR, save_plot=True):
    print(f"Loading predictions from {pred_file}...", flush=True)
    pred_df = pd.read_parquet(pred_file)
    pred_df = pred_df.drop_duplicates(subset=['trade_date', 'ts_code'])
    
    print("Loading volume and option columns from features...", flush=True)
    feat_df = pd.read_parquet(FEATURES_FILE, columns=['trade_date', 'ts_code', 'vol', 'opt_qvix_zscore', 'opt_pcr_vol_50'])
    feat_df = feat_df.drop_duplicates(subset=['trade_date', 'ts_code'])
    
    df = pred_df.merge(feat_df, on=['trade_date', 'ts_code'], how='left')
    df['vol'] = df['vol'].fillna(0)
    df['opt_qvix_zscore'] = df['opt_qvix_zscore'].fillna(0)
    df['opt_pcr_vol_50'] = df['opt_pcr_vol_50'].fillna(0)
    
    df = df.sort_values(['trade_date', 'ts_code']).reset_index(drop=True)
    
    trade_dates = sorted(df['trade_date'].unique())
    rebalance_dates = set(trade_dates[::HOLDING_DAYS])
    
    # 预加载中证1000指数，计算趋势过滤 (MA20)
    index_file = os.path.join(DATA_DIR, 'index_regime.csv')
    if os.path.exists(index_file):
        print("Loading CSI 1000 index and calculating MA20 trend filter...", flush=True)
        idx_df_raw = pd.read_csv(index_file)
        idx_df_raw['trade_date'] = idx_df_raw['trade_date'].astype(str)
        idx_df_raw = idx_df_raw.sort_values('trade_date').reset_index(drop=True)
        idx_df_raw['ma20'] = idx_df_raw['close'].rolling(20).mean()
        idx_df_raw['downtrend'] = idx_df_raw['close'] < idx_df_raw['ma20']
        
        trend_map = idx_df_raw.set_index('trade_date')['downtrend'].to_dict()
        df['downtrend'] = df['trade_date'].map(trend_map).fillna(False)
        
        idx_df = idx_df_raw[idx_df_raw['trade_date'].isin(trade_dates)].copy()
        idx_df['pct_chg'] = idx_df['close'].pct_change().fillna(0.0)
        idx_df_prices = idx_df.set_index('trade_date')['pct_chg'].to_dict()
        csi1000_returns = pd.Series([idx_df_prices.get(dt, 0.0) for dt in trade_dates], index=pd.to_datetime(trade_dates))
        csi1000_nav = (1 + csi1000_returns).cumprod() * INITIAL_CAPITAL
    else:
        print("⚠️ CSI 1000 index file not found. Wind-control trend filter deactivated.", flush=True)
        df['downtrend'] = False
        csi1000_nav = None
        
    df_by_date = {dt: g for dt, g in df.groupby('trade_date')}
    
    # 1. 运行纯多因子策略 (No Options Filter)
    nav_pure = simulate_strategy(df, df_by_date, trade_dates, rebalance_dates, use_options_filter=False)
    
    # 2. 运行期权+均线风控大闸策略 (With Options Filter & Trend Filter)
    nav_hedged = simulate_strategy(df, df_by_date, trade_dates, rebalance_dates, use_options_filter=True)
    
    # 3. 获取等权基准净值
    bench_returns = df.groupby('trade_date')['pct_chg'].mean().fillna(0.0)
    bench_returns.index = pd.to_datetime(bench_returns.index)
    bench_nav = (1 + bench_returns).cumprod() * INITIAL_CAPITAL
    if csi1000_nav is None:
        csi1000_nav = bench_nav
        
    # 对齐索引
    nav_pure['benchmark'] = bench_nav.reindex(nav_pure.index).ffill()
    nav_hedged['benchmark'] = bench_nav.reindex(nav_hedged.index).ffill()
    nav_pure['csi1000'] = csi1000_nav.reindex(nav_pure.index).ffill()
    nav_hedged['csi1000'] = csi1000_nav.reindex(nav_hedged.index).ffill()
    
    # 4. 计算并展示绩效指标
    metrics_pure = compute_metrics(nav_pure['nav'], 'Smart-Beta Base (No Filter)')
    metrics_hedged = compute_metrics(nav_hedged['nav'], 'Smart-Beta (Trend + VIX Filter)')
    metrics_bench = compute_metrics(nav_pure['benchmark'], 'Benchmark (Market Equal-Weight)')
    metrics_csi1000 = compute_metrics(nav_pure['csi1000'], 'Benchmark (CSI 1000 Index)')
    
    summary_df = pd.DataFrame([metrics_pure, metrics_hedged, metrics_bench, metrics_csi1000])
    
    print("\n==========================================================================")
    print("                Small-Cap Smart-Beta Base Holdings Comparison             ")
    print("==========================================================================")
    print(summary_df[['Portfolio', 'Total Return', 'CAGR', 'Volatility', 'Sharpe', 'Max Drawdown']].to_string(index=False))
    print("==========================================================================")
    
    # 保存数据结果
    os.makedirs(results_dir, exist_ok=True)
    results_nav = pd.DataFrame({
        'Strategy_Pure': nav_pure['nav'],
        'Strategy_Hedged': nav_hedged['nav'],
        'Benchmark_EqualWeight': nav_pure['benchmark'],
        'Benchmark_CSI1000': nav_pure['csi1000']
    })
    results_nav.to_csv(os.path.join(results_dir, 'portfolio_comparison_nav.csv'))
    summary_df.to_csv(os.path.join(results_dir, 'portfolio_comparison_metrics.csv'), index=False)
    
    if save_plot:
        # 5. 绘制多曲线对比图
        plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
        fig, ax = plt.subplots(figsize=(12, 7))
        
        ax.plot(results_nav.index, results_nav['Strategy_Pure'] / INITIAL_CAPITAL, 
                label='Smart-Beta Base (No Filter)', color='#1565c0', linewidth=2.0)
        ax.plot(results_nav.index, results_nav['Strategy_Hedged'] / INITIAL_CAPITAL, 
                label='Smart-Beta (Trend + VIX Filter)', color='#2e7d32', linewidth=2.5)
        ax.plot(results_nav.index, results_nav['Benchmark_EqualWeight'] / INITIAL_CAPITAL, 
                label='Benchmark (Market Equal-Weight)', color='#78909c', linewidth=1.5, linestyle='--')
        ax.plot(results_nav.index, results_nav['Benchmark_CSI1000'] / INITIAL_CAPITAL, 
                label='Benchmark (CSI 1000 Index)', color='#e65100', linewidth=1.8, linestyle='-.')
        
        ax.set_title('A-Share Multi-Factor Portfolio Backtest (Out-of-Sample)', fontsize=14, fontweight='bold', pad=15)
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Normalized Net Asset Value (NAV)', fontsize=12)
        ax.legend(loc='upper left', frameon=True, facecolor='white', framealpha=0.9, edgecolor='lightgray')
        
        plt.tight_layout()
        out_png = os.path.join(results_dir, 'portfolio_backtest_nav.png')
        plt.savefig(out_png, dpi=300)
        print(f"Saved comparison plot to {out_png}")
        plt.close()

if __name__ == '__main__':
    run_backtest()

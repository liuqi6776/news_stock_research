import pandas as pd
import numpy as np
import akshare as ak
import os
import matplotlib.pyplot as plt

def main():
    print("==================================================")
    print("      Option Volatility (QVIX) Backtest Filter    ")
    print("==================================================")

    # 1. Load predictions
    pred_path = "research/study_005_1d_advanced/predictions/predictions_005_wf.parquet"
    if not os.path.exists(pred_path):
        print(f"Error: Prediction file not found at {pred_path}")
        return
        
    print("Loading stock predictions...")
    df_pred = pd.read_parquet(pred_path, columns=['trade_date', 'ts_code', 'industry', 'prob_up', 'prob_crash', 'actual_return'])
    df_pred['trade_date'] = df_pred['trade_date'].astype(str)
    
    # 2. Fetch QVIX historical data
    print("Fetching 50ETF QVIX...")
    try:
        df_qvix = ak.index_option_50etf_qvix()
        df_qvix['date'] = pd.to_datetime(df_qvix['date'])
        df_qvix['trade_date'] = df_qvix['date'].dt.strftime('%Y%m%d')
        
        # Calculate rolling Z-score
        df_qvix['qvix_ma'] = df_qvix['close'].rolling(20).mean()
        df_qvix['qvix_std'] = df_qvix['close'].rolling(20).std()
        df_qvix['qvix_zscore'] = (df_qvix['close'] - df_qvix['qvix_ma']) / df_qvix['qvix_std']
        
        # Shift QVIX by 1 day to prevent look-ahead bias
        df_qvix['prev_qvix'] = df_qvix['close'].shift(1)
        df_qvix['prev_zscore'] = df_qvix['qvix_zscore'].shift(1)
        
        df_qvix_clean = df_qvix[['trade_date', 'prev_qvix', 'prev_zscore', 'close']].rename(columns={'close': 'today_qvix'})
    except Exception as e:
        print("Error fetching QVIX:", e)
        return

    # Merge stock predictions with shifted QVIX
    print("Merging stock predictions with QVIX data...")
    df_merged = pd.merge(df_pred, df_qvix_clean, on='trade_date', how='inner')
    
    # Sort by trade_date
    df_merged = df_merged.sort_values('trade_date').reset_index(drop=True)
    
    # Backtest configurations
    max_positions = 3
    transaction_cost = 0.003 # 0.3% commission + slippage
    
    dates_list = []
    daily_rets_base = []
    daily_rets_filt = []
    daily_rets_comp = []
    
    trades_baseline = 0
    trades_filtered = 0
    trades_complacency = 0
    
    print("Running optimized backtest loop...")
    grouped = df_merged.groupby('trade_date')
    
    for date_str, day_df in grouped:
        # Get options data
        prev_qvix = day_df['prev_qvix'].iloc[0]
        prev_zscore = day_df['prev_zscore'].iloc[0]
        
        # 1. Conservative candidates
        candidates = day_df[(day_df['prob_up'] >= 0.50) & (day_df['prob_crash'] <= 0.15)].copy()
        
        if candidates.empty:
            candidates = day_df.nlargest(5, 'prob_up').copy()
            
        candidates = candidates.sort_values('prob_up', ascending=False)
        
        selected_stocks = []
        ind_counts = {}
        for _, row in candidates.iterrows():
            ind = row.get('industry', 'Unknown')
            if ind_counts.get(ind, 0) >= 2:
                continue
            selected_stocks.append(row)
            ind_counts[ind] = ind_counts.get(ind, 0) + 1
            if len(selected_stocks) >= max_positions:
                break
                
        # Calculate daily baseline return (skip NaNs in actual_return)
        if selected_stocks:
            rets = [r['actual_return'] - transaction_cost for r in selected_stocks if pd.notna(r['actual_return'])]
            if rets:
                daily_ret = np.mean(rets)
                num_trades = len(rets)
            else:
                daily_ret = 0.0
                num_trades = 0
        else:
            daily_ret = 0.0
            num_trades = 0
            
        # Record date
        dates_list.append(pd.to_datetime(date_str, format='%Y%m%d'))
        
        # 1. Baseline
        daily_rets_base.append(daily_ret)
        trades_baseline += num_trades
        
        # 2. Filtered: Tail Risk Filter (QVIX >= 28)
        if pd.notna(prev_qvix) and prev_qvix >= 28.0:
            daily_ret_filt = 0.0
            num_trades_filt = 0
        else:
            daily_ret_filt = daily_ret
            num_trades_filt = num_trades
        daily_rets_filt.append(daily_ret_filt)
        trades_filtered += num_trades_filt
        
        # 3. Spike + Complacency Filter (QVIX >= 28 or Z-score <= -1.5)
        if (pd.notna(prev_qvix) and prev_qvix >= 28.0) or (pd.notna(prev_zscore) and prev_zscore <= -1.5):
            daily_ret_comp = 0.0
            num_trades_comp = 0
        else:
            daily_ret_comp = daily_ret
            num_trades_comp = num_trades
        daily_rets_comp.append(daily_ret_comp)
        trades_complacency += num_trades_comp

    # Compute cumulative equity curves
    equity_baseline = np.cumprod(1.0 + np.array(daily_rets_base))
    equity_filtered = np.cumprod(1.0 + np.array(daily_rets_filt))
    equity_complacency = np.cumprod(1.0 + np.array(daily_rets_comp))
    
    # 3. Analyze Results
    print("\n================== Performance Metrics ==================")
    
    # Calculate returns
    total_ret_b = (equity_baseline[-1] - 1.0) * 100
    total_ret_f = (equity_filtered[-1] - 1.0) * 100
    total_ret_c = (equity_complacency[-1] - 1.0) * 100
    
    # Calculate Max Drawdown
    def calc_max_dd(eq_curve):
        eq_series = pd.Series(eq_curve)
        cum_max = eq_series.cummax()
        drawdown = (eq_series - cum_max) / cum_max
        return drawdown.min() * 100
        
    dd_b = calc_max_dd(equity_baseline)
    dd_f = calc_max_dd(equity_filtered)
    dd_c = calc_max_dd(equity_complacency)
    
    # Daily returns for Sharpe Ratio
    rets_b = pd.Series(daily_rets_base)
    rets_f = pd.Series(daily_rets_filt)
    rets_c = pd.Series(daily_rets_comp)
    
    def calc_sharpe(rets):
        if rets.std() == 0:
            return 0
        return (rets.mean() / rets.std()) * np.sqrt(242)
        
    sharpe_b = calc_sharpe(rets_b)
    sharpe_f = calc_sharpe(rets_f)
    sharpe_c = calc_sharpe(rets_c)
    
    print(f"Baseline Strategy:")
    print(f"  Total Return: {total_ret_b:.2f}% | Max Drawdown: {dd_b:.2f}% | Sharpe Ratio: {sharpe_b:.4f} | Trades: {trades_baseline}")
    print(f"Option Tail Risk Filter (QVIX >= 28):")
    print(f"  Total Return: {total_ret_f:.2f}% | Max Drawdown: {dd_f:.2f}% | Sharpe Ratio: {sharpe_f:.4f} | Trades: {trades_filtered}")
    print(f"Spike + Complacency Filter (QVIX >= 28 or Z <= -1.5):")
    print(f"  Total Return: {total_ret_c:.2f}% | Max Drawdown: {dd_c:.2f}% | Sharpe Ratio: {sharpe_c:.4f} | Trades: {trades_complacency}")
    
    # Create results folder
    os.makedirs("research/期权/results", exist_ok=True)
    
    # Plot equity curve
    plt.figure(figsize=(12, 6))
    plt.plot(dates_list, equity_baseline, label='Baseline (Conservative)', color='#888888', alpha=0.7)
    plt.plot(dates_list, equity_filtered, label='Option Tail Risk Filter (QVIX >= 28)', color='#1f77b4', linewidth=2)
    plt.plot(dates_list, equity_complacency, label='Spike + Complacency Filter', color='#d62728', linewidth=2)
    
    plt.title('A-Share Option Volatility Filter Backtest (T+1)', fontsize=14, fontweight='bold')
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Cumulative Equity', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(fontsize=11)
    
    # Save chart
    plt.tight_layout()
    chart_path = "research/期权/results/equity_curve_comparison.png"
    plt.savefig(chart_path, dpi=300)
    print(f"Saved comparison chart to {chart_path}")
    
    # Save CSV metrics
    df_metrics = pd.DataFrame(index=['Baseline', 'Option Tail Risk Filter', 'Spike + Complacency Filter'])
    df_metrics['Total Return (%)'] = [total_ret_b, total_ret_f, total_ret_c]
    df_metrics['Max Drawdown (%)'] = [dd_b, dd_f, dd_c]
    df_metrics['Sharpe Ratio'] = [sharpe_b, sharpe_f, sharpe_c]
    df_metrics.to_csv("research/期权/results/backtest_metrics.csv")
    print("Saved metrics to research/期权/results/backtest_metrics.csv")

if __name__ == "__main__":
    main()

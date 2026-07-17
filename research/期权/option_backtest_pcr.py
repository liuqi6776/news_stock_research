import pandas as pd
import numpy as np
import akshare as ak
import os
import matplotlib.pyplot as plt

def main():
    print("==================================================")
    print("   Put-Call Ratio (PCR) & Volatility Backtest    ")
    print("==================================================")

    # 1. Load predictions
    pred_path = "research/study_005_1d_advanced/predictions/predictions_005_wf.parquet"
    if not os.path.exists(pred_path):
        print(f"Error: Prediction file not found at {pred_path}")
        return
        
    print("Loading stock predictions...")
    df_pred = pd.read_parquet(pred_path, columns=['trade_date', 'ts_code', 'industry', 'prob_up', 'prob_crash', 'actual_return'])
    df_pred['trade_date'] = df_pred['trade_date'].astype(str)
    
    # 2. Load historical PCR data
    pcr_path = "research/期权/data/historical_pcr.csv"
    if not os.path.exists(pcr_path):
        print(f"Error: PCR file not found at {pcr_path}. Please wait for the download to complete.")
        return
        
    print("Loading Put-Call Ratio data...")
    df_pcr = pd.read_csv(pcr_path)
    df_pcr['date'] = pd.to_datetime(df_pcr['date'])
    df_pcr['trade_date'] = df_pcr['date'].dt.strftime('%Y%m%d')
    
    # Shift PCR to prevent look-ahead bias (yesterday's PCR for today's trading decisions)
    df_pcr = df_pcr.sort_values('trade_date').reset_index(drop=True)
    df_pcr['prev_pcr_50'] = df_pcr['pcr_50'].shift(1)
    df_pcr['prev_pcr_300'] = df_pcr['pcr_300'].shift(1)
    
    df_pcr_clean = df_pcr[['trade_date', 'prev_pcr_50', 'prev_pcr_300', 'pcr_50']].rename(columns={'pcr_50': 'today_pcr_50'})

    # 3. Fetch QVIX historical data
    print("Fetching 50ETF QVIX...")
    try:
        df_qvix = ak.index_option_50etf_qvix()
        df_qvix['date'] = pd.to_datetime(df_qvix['date'])
        df_qvix['trade_date'] = df_qvix['date'].dt.strftime('%Y%m%d')
        
        # Shift QVIX by 1 day
        df_qvix['prev_qvix'] = df_qvix['close'].shift(1)
        df_qvix_clean = df_qvix[['trade_date', 'prev_qvix']]
    except Exception as e:
        print("Error fetching QVIX:", e)
        return

    # Merge stock predictions, PCR, and QVIX
    print("Merging stock predictions with Options data...")
    df_merged = pd.merge(df_pred, df_pcr_clean, on='trade_date', how='inner')
    df_merged = pd.merge(df_merged, df_qvix_clean, on='trade_date', how='inner')
    
    # Sort by trade_date
    df_merged = df_merged.sort_values('trade_date').reset_index(drop=True)
    
    # Backtest configurations
    max_positions = 3
    transaction_cost = 0.003 # 0.3% commission + slippage
    
    dates_list = []
    daily_rets_base = []
    daily_rets_filt_pcr = []
    daily_rets_filt_comb = []
    
    trades_baseline = 0
    trades_pcr = 0
    trades_combined = 0
    
    print("Running optimized backtest loop...")
    grouped = df_merged.groupby('trade_date')
    
    for date_str, day_df in grouped:
        # Get yesterday's options indicators
        prev_pcr = day_df['prev_pcr_50'].iloc[0]
        prev_qvix = day_df['prev_qvix'].iloc[0]
        
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
                
        # Calculate daily return (ignoring NaNs)
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
        
        # 2. PCR Complacency Filter (PCR <= 0.65)
        # Rule: If yesterday's Put-Call Ratio <= 0.65 (extreme greed/complacency), we stay in cash
        if pd.notna(prev_pcr) and prev_pcr <= 0.65:
            daily_ret_pcr = 0.0
            num_trades_pcr = 0
        else:
            daily_ret_pcr = daily_ret
            num_trades_pcr = num_trades
        daily_rets_filt_pcr.append(daily_ret_pcr)
        trades_pcr += num_trades_pcr
        
        # 3. Combined Option Volatility & PCR Filter
        # Rule: If yesterday's PCR <= 0.65 (complacency) OR yesterday's QVIX >= 30 (high risk/turbulent regime), stay in cash
        is_risky = False
        if pd.notna(prev_pcr) and prev_pcr <= 0.65:
            is_risky = True
        if pd.notna(prev_qvix) and prev_qvix >= 30.0:
            is_risky = True
            
        if is_risky:
            daily_ret_comb = 0.0
            num_trades_comb = 0
        else:
            daily_ret_comb = daily_ret
            num_trades_comb = num_trades
        daily_rets_filt_comb.append(daily_ret_comb)
        trades_combined += num_trades_comb

    # Compute cumulative equity curves
    equity_baseline = np.cumprod(1.0 + np.array(daily_rets_base))
    equity_pcr = np.cumprod(1.0 + np.array(daily_rets_filt_pcr))
    equity_combined = np.cumprod(1.0 + np.array(daily_rets_filt_comb))
    
    # 3. Analyze Results
    print("\n================== Performance Metrics ==================")
    
    total_ret_b = (equity_baseline[-1] - 1.0) * 100
    total_ret_p = (equity_pcr[-1] - 1.0) * 100
    total_ret_c = (equity_combined[-1] - 1.0) * 100
    
    def calc_max_dd(eq_curve):
        eq_series = pd.Series(eq_curve)
        cum_max = eq_series.cummax()
        drawdown = (eq_series - cum_max) / cum_max
        return drawdown.min() * 100
        
    dd_b = calc_max_dd(equity_baseline)
    dd_p = calc_max_dd(equity_pcr)
    dd_c = calc_max_dd(equity_combined)
    
    rets_b = pd.Series(daily_rets_base)
    rets_p = pd.Series(daily_rets_filt_pcr)
    rets_c = pd.Series(daily_rets_filt_comb)
    
    def calc_sharpe(rets):
        if rets.std() == 0:
            return 0
        return (rets.mean() / rets.std()) * np.sqrt(242)
        
    sharpe_b = calc_sharpe(rets_b)
    sharpe_p = calc_sharpe(rets_p)
    sharpe_c = calc_sharpe(rets_c)
    
    print(f"Baseline Strategy (Conservative):")
    print(f"  Total Return: {total_ret_b:.2f}% | Max Drawdown: {dd_b:.2f}% | Sharpe Ratio: {sharpe_b:.4f} | Trades: {trades_baseline}")
    print(f"Option PCR Complacency Filter (PCR <= 0.65):")
    print(f"  Total Return: {total_ret_p:.2f}% | Max Drawdown: {dd_p:.2f}% | Sharpe Ratio: {sharpe_p:.4f} | Trades: {trades_pcr}")
    print(f"Combined Options Filter (PCR <= 0.65 or QVIX >= 30):")
    print(f"  Total Return: {total_ret_c:.2f}% | Max Drawdown: {dd_c:.2f}% | Sharpe Ratio: {sharpe_c:.4f} | Trades: {trades_combined}")
    
    # Create results folder
    os.makedirs("research/期权/results", exist_ok=True)
    
    # Plot equity curve
    plt.figure(figsize=(12, 6))
    plt.plot(dates_list, equity_baseline, label='Baseline (Conservative)', color='#888888', alpha=0.7)
    plt.plot(dates_list, equity_pcr, label='Option PCR Complacency Filter (PCR <= 0.65)', color='#2ca02c', linewidth=2)
    plt.plot(dates_list, equity_combined, label='Combined Options Filter (PCR + Volatility)', color='#9467bd', linewidth=2)
    
    plt.title('A-Share Options Timing Filter Backtest (T+1)', fontsize=14, fontweight='bold')
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Cumulative Equity', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(fontsize=11)
    
    # Save chart
    plt.tight_layout()
    chart_path = "research/期权/results/equity_curve_pcr_comparison.png"
    plt.savefig(chart_path, dpi=300)
    print(f"Saved comparison chart to {chart_path}")
    
    # Save CSV metrics
    df_metrics = pd.DataFrame(index=['Baseline', 'Option PCR Complacency Filter', 'Combined Options Filter'])
    df_metrics['Total Return (%)'] = [total_ret_b, total_ret_p, total_ret_c]
    df_metrics['Max Drawdown (%)'] = [dd_b, dd_p, dd_c]
    df_metrics['Sharpe Ratio'] = [sharpe_b, sharpe_p, sharpe_c]
    df_metrics.to_csv("research/期权/results/pcr_backtest_metrics.csv")
    print("Saved metrics to research/期权/results/pcr_backtest_metrics.csv")

if __name__ == "__main__":
    main()
